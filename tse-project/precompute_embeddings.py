"""Compute one ECAPA-TDNN embedding per utterance and save {relative path: tensor(192)}.

  python precompute_embeddings.py --root data --manifests data/train_speech.jsonl --out data/train_emb.pt
"""
import argparse
import math
import os

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from tse.data import load_manifest
from tse.speaker import embed_wave, load_ecapa

SR = 16000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--manifests", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--enroll-sec", type=float, default=4.0, help="enrollment clip length (3-5 s)")
    p.add_argument("--batch", type=int, default=32)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = load_ecapa(device)
    seg = int(a.enroll_sec * SR)
    items = []
    for m in a.manifests:
        items += load_manifest(m)
    out = {}
    for s in tqdm(range(0, len(items), a.batch)):
        chunk = items[s:s + a.batch]
        wavs = []
        for it in chunk:
            x, _ = sf.read(os.path.join(a.root, it["path"]), start=0, stop=seg, dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            if len(x) < seg:  # tile short clips to a fixed length so they can be batched
                x = np.tile(x, int(math.ceil(seg / max(len(x), 1))))[:seg]
            wavs.append(torch.from_numpy(x))
        emb = embed_wave(clf, torch.stack(wavs), device)
        for it, e in zip(chunk, emb):
            out[it["path"]] = e
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(out, a.out)
    print(f"saved {len(out)} embeddings -> {a.out}")


if __name__ == "__main__":
    main()
