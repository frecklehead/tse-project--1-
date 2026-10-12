"""Extract a target speaker from a noisy recording.

  python infer.py --ckpt runs/film/best.pt --mixture noisy.wav --enroll my_voice.wav --out clean.wav
"""
import argparse
from math import gcd

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from tse.models import build_model
from tse.speaker import embed_wave, load_ecapa

SR = 16000


def load_wav(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--mixture", required=True)
    p.add_argument("--enroll", required=True, help="3-5 s clean recording of the target speaker")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = build_model(ck["model_cfg"])
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    enroll = load_wav(a.enroll)[: 5 * SR]
    emb = embed_wave(load_ecapa(device), torch.from_numpy(enroll).unsqueeze(0), device).to(device)
    mix = load_wav(a.mixture)
    with torch.no_grad():
        est = model(torch.from_numpy(mix).unsqueeze(0).to(device), emb)[0].cpu().numpy()
    est = est / (np.max(np.abs(est)) + 1e-8) * 0.9
    sf.write(a.out, est, SR)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
