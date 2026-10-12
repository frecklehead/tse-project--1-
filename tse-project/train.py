"""Train the speaker-conditioned causal Conv-TasNet. Safe for Colab: auto-resumes from out-dir/last.pt.

  python train.py --data-root data --train-speech data/train_speech.jsonl --train-emb data/train_emb.pt \
      --val-speech data/val_speech.jsonl --val-emb data/val_emb.pt --train-noise data/noise_train.jsonl \
      --val-noise data/noise_train.jsonl --out-dir runs/film --cond film --amp
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from tse.data import STAGES, TSEDataset
from tse.losses import si_sdr, sisdr_loss
from tse.models import MODEL_KEYS, build_model, count_params


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--train-speech", required=True)
    p.add_argument("--train-emb", required=True)
    p.add_argument("--train-noise", default=None, help="omit to use synthetic noise (smoke tests)")
    p.add_argument("--val-speech", required=True)
    p.add_argument("--val-emb", required=True)
    p.add_argument("--val-noise", default=None)
    p.add_argument("--out-dir", required=True)
    # model
    p.add_argument("--cond", default="film", choices=["none", "concat", "film", "gating"])
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--L", type=int, default=16)
    p.add_argument("--B", type=int, default=128)
    p.add_argument("--H", type=int, default=256)
    p.add_argument("--P", type=int, default=3)
    p.add_argument("--X", type=int, default=8)
    p.add_argument("--R", type=int, default=3)
    # training
    p.add_argument("--segment", type=float, default=3.0, help="training segment length (s)")
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--steps-per-epoch", type=int, default=500, help="batches per epoch")
    p.add_argument("--stage-epochs", type=int, default=8, help="epochs per curriculum stage")
    p.add_argument("--val-size", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--amp", action="store_true", help="mixed precision (GPU only)")
    p.add_argument("--max-hours", type=float, default=None, help="stop cleanly after this many hours")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def save_atomic(state, path):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def validate(model, dl, device):
    model.eval()
    imps, abss = [], []
    for b in dl:
        mix, tgt, emb = b["mixture"].to(device), b["target"].to(device), b["emb"].to(device)
        est = model(mix, emb)
        s_est, s_mix = si_sdr(est, tgt), si_sdr(mix, tgt)
        imps.append((s_est - s_mix).cpu())
        abss.append(s_est.cpu())
    return torch.cat(imps).mean().item(), torch.cat(abss).mean().item()


def main():
    a = get_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = a.amp and device.type == "cuda"
    print(f"device={device} amp={use_amp}")

    cfg = {k: getattr(a, k) for k in MODEL_KEYS}
    model = build_model(cfg).to(device)
    print(f"model params: {count_params(model):,}  cfg={cfg}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_ds = TSEDataset(a.data_root, a.train_speech, a.train_emb, a.train_noise,
                          segment_sec=a.segment, length=a.steps_per_epoch * a.batch)
    val_ds = TSEDataset(a.data_root, a.val_speech, a.val_emb, a.val_noise,
                        segment_sec=a.segment, length=a.val_size, deterministic=True, seed=12345)
    train_dl = DataLoader(train_ds, batch_size=a.batch, num_workers=a.workers, drop_last=True,
                          pin_memory=device.type == "cuda")
    val_dl = DataLoader(val_ds, batch_size=a.batch, num_workers=a.workers)

    last_path, best_path = os.path.join(a.out_dir, "last.pt"), os.path.join(a.out_dir, "best.pt")
    start_epoch, best = 0, -1e9
    if os.path.exists(last_path) and not a.no_resume:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        if ck["model_cfg"] != cfg:
            raise SystemExit("Checkpoint model config differs from current arguments. "
                             "Use a new --out-dir or pass --no-resume.")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"RESUMED from {last_path}: continuing at epoch {start_epoch}, best val SI-SDRi={best:.2f}")

    log_path = os.path.join(a.out_dir, "log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "stage", "train_loss", "val_sisdri", "val_sisdr", "lr", "minutes"])

    t0 = time.time()
    for epoch in range(start_epoch, a.epochs):
        stage = min(len(STAGES) - 1, epoch // a.stage_epochs)
        train_ds.set_stage(stage)
        model.train()
        opt.zero_grad(set_to_none=True)
        tot, n, te = 0.0, 0, time.time()
        for step, b in enumerate(train_dl):
            mix, tgt, emb = b["mixture"].to(device), b["target"].to(device), b["emb"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                est = model(mix, emb)
            loss = sisdr_loss(est, tgt)
            scaler.scale(loss / a.accum).backward()
            if (step + 1) % a.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            tot += loss.item()
            n += 1
        v_imp, v_abs = validate(model, val_dl, device)
        sched.step(v_imp)
        lr = opt.param_groups[0]["lr"]
        mins = (time.time() - te) / 60
        print(f"epoch {epoch:3d} stage {stage} train_loss {tot / max(n, 1):7.3f} "
              f"val SI-SDRi {v_imp:6.2f} dB (SI-SDR {v_abs:6.2f}) lr {lr:.1e} [{mins:.1f} min]")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, stage, f"{tot / max(n, 1):.4f}", f"{v_imp:.3f}", f"{v_abs:.3f}", lr, f"{mins:.2f}"])

        if v_imp > best:
            best = v_imp
            save_atomic({"model": model.state_dict(), "model_cfg": cfg, "epoch": epoch, "val_sisdri": v_imp}, best_path)
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "model_cfg": cfg}, last_path)

        if a.max_hours and (time.time() - t0) / 3600 >= a.max_hours:
            print(f"--max-hours reached; stopped cleanly after epoch {epoch}. Re-run the same command to resume.")
            break
    with open(os.path.join(a.out_dir, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2)


if __name__ == "__main__":
    main()
