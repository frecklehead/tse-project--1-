"""Build JSONL manifests for speech (LibriSpeech) and noise (e.g. DEMAND).

All paths in manifests are relative to --root so they work on laptop, Colab and Drive.

Examples
  python prepare_data.py speech --root data --subdir LibriSpeech/train-clean-100 --out data/train_speech.jsonl --max-utts 40
  python prepare_data.py noise  --root data --subdir DEMAND --holdout PCAFETER,STRAFFIC,TMETRO,DWASHING \
        --out-train data/noise_train.jsonl --out-test data/noise_test.jsonl
"""
import argparse
import json
import os
import random
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def rel(path, root):
    return os.path.relpath(path, root).replace(os.sep, "/")


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {path}")


def cmd_speech(a):
    rng = random.Random(a.seed)
    excluded = set()
    if a.exclude:
        with open(a.exclude, encoding="utf-8") as f:
            excluded = {json.loads(l)["path"] for l in f if l.strip()}
    by = {}
    for dp, _, fns in os.walk(os.path.join(a.root, a.subdir)):
        for fn in fns:
            if not fn.lower().endswith((".flac", ".wav")):
                continue
            full = os.path.join(dp, fn)
            r = rel(full, a.root)
            if r in excluded:
                continue
            spk = os.path.basename(os.path.dirname(os.path.dirname(full)))  # .../<spk>/<chapter>/<file>
            by.setdefault(spk, []).append((r, full))
    rows = []
    for spk, files in sorted(by.items()):
        rng.shuffle(files)
        kept = 0
        for r, full in files:
            info = sf.info(full)
            if info.samplerate != 16000 or info.frames < a.min_sec * 16000:
                continue
            rows.append({"path": r, "speaker": spk, "frames": int(info.frames)})
            kept += 1
            if kept >= a.max_utts:
                break
    print(f"{len({r['speaker'] for r in rows})} speakers")
    write_jsonl(rows, a.out)


def cmd_noise(a):
    holdout = set(a.holdout.split(",")) if a.holdout else set()
    train, test = [], []
    for dp, _, fns in os.walk(os.path.join(a.root, a.subdir)):
        for fn in sorted(fns):
            if not fn.lower().endswith(".wav"):
                continue
            full = os.path.join(dp, fn)
            cat = os.path.basename(dp)
            info = sf.info(full)
            if info.samplerate != 16000:  # resample once and store a 16 kHz copy
                x, sr = sf.read(full, dtype="float32")
                if x.ndim > 1:
                    x = x.mean(axis=1)
                g = gcd(16000, sr)
                x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
                outdir = os.path.join(a.root, "noise16k", cat)
                os.makedirs(outdir, exist_ok=True)
                full = os.path.join(outdir, fn)
                sf.write(full, x, 16000)
                frames = len(x)
            else:
                frames = int(info.frames)
            row = {"path": rel(full, a.root), "category": cat, "frames": frames}
            (test if cat in holdout else train).append(row)
    write_jsonl(train, a.out_train)
    if holdout:
        write_jsonl(test, a.out_test)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("speech")
    s.add_argument("--root", required=True)
    s.add_argument("--subdir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--max-utts", type=int, default=40, help="max utterances per speaker")
    s.add_argument("--min-sec", type=float, default=4.0)
    s.add_argument("--exclude", default=None, help="manifest whose files are skipped")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_speech)
    n = sub.add_parser("noise")
    n.add_argument("--root", required=True)
    n.add_argument("--subdir", required=True)
    n.add_argument("--holdout", default="", help="comma-separated noise categories kept for testing")
    n.add_argument("--out-train", required=True)
    n.add_argument("--out-test", default="noise_test.jsonl")
    n.set_defaults(fn=cmd_noise)
    args = p.parse_args()
    args.fn(args)
