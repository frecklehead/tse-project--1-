"""Evaluate a checkpoint on deterministic mixtures: SI-SDR, SI-SDRi, STOI, (PESQ), speed.

  python evaluate.py --ckpt runs/film/best.pt --data-root data --speech data/test_speech.jsonl \
      --emb data/test_emb.pt --noise data/noise_test.jsonl --n 300 --out results/film_unseen_unseen.csv

Condition control: --snr-range -5 20  --tir-range -5 10
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
from pystoi import stoi
from torch.utils.data import DataLoader

from tse.data import TSEDataset
from tse.losses import si_sdr
from tse.models import build_model, count_params

try:
    from pesq import pesq as pesq_fn
except Exception:  # pesq is optional (needs a C compiler to install)
    pesq_fn = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--speech", required=True)
    p.add_argument("--emb", required=True)
    p.add_argument("--noise", default=None)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--segment", type=float, default=4.0)
    p.add_argument("--snr-range", type=float, nargs=2, default=[-5, 20])
    p.add_argument("--tir-range", type=float, nargs=2, default=[-5, 10])
    p.add_argument("--seed", type=int, default=999)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default="results/eval.csv")
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)  # CPU-safe for Colab checkpoints
    model = build_model(ck["model_cfg"])
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"params: {count_params(model):,}  cfg: {ck['model_cfg']}")

    ds = TSEDataset(a.data_root, a.speech, a.emb, a.noise, segment_sec=a.segment, length=a.n,
                    deterministic=True, seed=a.seed, p_no_interferer=0.0)
    ds.set_ranges(a.snr_range, a.tir_range)
    dl = DataLoader(ds, batch_size=1, num_workers=0)

    rows, t_proc, t_audio = [], 0.0, 0.0
    with torch.no_grad():
        for i, b in enumerate(dl):
            mix, tgt, emb = b["mixture"], b["target"], b["emb"]
            t0 = time.time()
            est = model(mix, emb)
            t_proc += time.time() - t0
            t_audio += mix.size(-1) / 16000
            s_est, s_mix = si_sdr(est, tgt).item(), si_sdr(mix, tgt).item()
            r, e, m = tgt[0].numpy(), est[0].numpy(), mix[0].numpy()
            row = {"i": i, "snr": b["snr"].item(), "tir": b["tir"].item(), "sisdr_mix": s_mix, "sisdr_est": s_est,
                   "sisdri": s_est - s_mix, "stoi_mix": stoi(r, m, 16000), "stoi_est": stoi(r, e, 16000)}
            if pesq_fn is not None:
                try:
                    row["pesq_mix"] = pesq_fn(16000, r, m, "wb")
                    row["pesq_est"] = pesq_fn(16000, r, e, "wb")
                except Exception:
                    pass
            rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "i", k))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def mean(k):
        v = [r[k] for r in rows if k in r]
        return float(np.mean(v)) if v else float("nan")

    print(f"\n=== {len(rows)} mixtures | SNR {a.snr_range} dB | TIR {a.tir_range} dB ===")
    print(f"SI-SDR  mixture {mean('sisdr_mix'):6.2f} -> estimate {mean('sisdr_est'):6.2f}   (SI-SDRi {mean('sisdri'):6.2f} dB)")
    print(f"STOI    mixture {mean('stoi_mix'):6.3f} -> estimate {mean('stoi_est'):6.3f}")
    if pesq_fn is not None:
        print(f"PESQ    mixture {mean('pesq_mix'):6.3f} -> estimate {mean('pesq_est'):6.3f}")
    else:
        print("PESQ    skipped (install the optional 'pesq' package)")
    print(f"Speed   RTF = {t_proc / t_audio:.3f} on CPU with {a.threads} threads (full-clip, offline causal model)")
    print(f"per-utterance results -> {a.out}")


if __name__ == "__main__":
    main()
