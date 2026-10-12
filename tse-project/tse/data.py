"""Dataset: builds target-speaker-extraction mixtures on the fly."""
import json
import math
import os
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .mixing import mix_signals

SR = 16000

# Curriculum stages: (SNR range dB, TIR range dB). Easy -> hard.
STAGES = [
    ((10.0, 20.0), (5.0, 10.0)),
    ((0.0, 20.0), (0.0, 10.0)),
    ((-5.0, 20.0), (-5.0, 10.0)),
]


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_crop(path, frames, seg, rng, tile):
    """Read exactly `seg` samples: random crop if long enough, else tile/pad."""
    if frames > seg:
        start = int(rng.integers(0, frames - seg + 1))
        x, _ = sf.read(path, start=start, stop=start + seg, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        return x
    x, _ = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if len(x) == 0:
        return np.zeros(seg, dtype=np.float32)
    if tile:
        return np.tile(x, int(math.ceil(seg / len(x))))[:seg]
    return np.pad(x, (0, seg - len(x)))


def synthetic_noise(n, rng):
    """Coloured Gaussian noise (fallback when no noise files are given)."""
    beta = rng.uniform(0.0, 2.0)  # 0 = white, 2 = brown
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1.0
    spec = spec / (f ** (beta / 2))
    x = np.fft.irfft(spec, n)
    return (x / (np.std(x) + 1e-8)).astype(np.float32)


class TSEDataset(Dataset):
    """Each item: mixture, clean target, enrollment embedding (192-d).

    Manifest rows: {"path": relative path, "speaker": id, "frames": int}.
    Embedding file: dict {relative path: tensor(192)} from precompute_embeddings.py.
    The enrollment embedding comes from a DIFFERENT utterance of the target speaker.
    """

    def __init__(self, data_root, speech_manifest, emb_path, noise_manifest=None,
                 segment_sec=3.0, length=1000, deterministic=False, seed=0,
                 p_no_interferer=0.2):
        self.root = data_root
        self.seg = int(segment_sec * SR)
        self.length = length
        self.deterministic = deterministic
        self.seed = seed
        self.p_no_int = p_no_interferer
        self.emb = torch.load(emb_path, map_location="cpu")
        by = defaultdict(list)
        for it in load_manifest(speech_manifest):
            if it["path"] in self.emb:
                by[it["speaker"]].append(it)
        self.by_spk = {s: v for s, v in by.items() if len(v) >= 2}
        self.speakers = sorted(self.by_spk)
        if len(self.speakers) < 2:
            raise ValueError("Need at least 2 speakers with >= 2 utterances each "
                             "(check manifest and embeddings file).")
        self.noises = load_manifest(noise_manifest) if noise_manifest else []
        self.set_stage(len(STAGES) - 1)

    def set_stage(self, stage):
        self.snr_range, self.tir_range = STAGES[stage]

    def set_ranges(self, snr_range, tir_range):
        self.snr_range, self.tir_range = tuple(snr_range), tuple(tir_range)

    def __len__(self):
        return self.length

    def _path(self, it):
        return os.path.join(self.root, it["path"])

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + idx) if self.deterministic else np.random.default_rng()
        spk = self.speakers[int(rng.integers(len(self.speakers)))]
        utts = self.by_spk[spk]
        i, j = rng.choice(len(utts), 2, replace=False)
        tgt_u, enr_u = utts[int(i)], utts[int(j)]
        target = read_crop(self._path(tgt_u), tgt_u["frames"], self.seg, rng, tile=False)

        interferer = None
        if rng.random() >= self.p_no_int:
            ispk = spk
            while ispk == spk:
                ispk = self.speakers[int(rng.integers(len(self.speakers)))]
            iu = self.by_spk[ispk][int(rng.integers(len(self.by_spk[ispk])))]
            interferer = read_crop(self._path(iu), iu["frames"], self.seg, rng, tile=True)

        if self.noises:
            nu = self.noises[int(rng.integers(len(self.noises)))]
            noise = read_crop(self._path(nu), nu["frames"], self.seg, rng, tile=True)
        else:
            noise = synthetic_noise(self.seg, rng)

        snr = float(rng.uniform(*self.snr_range))
        tir = float(rng.uniform(*self.tir_range))
        gain = float(10 ** (rng.uniform(-6, 3) / 20))
        mixture, tgt = mix_signals(target, interferer, noise, snr, tir, gain=gain)
        return {
            "mixture": torch.from_numpy(mixture),
            "target": torch.from_numpy(tgt),
            "emb": self.emb[enr_u["path"]].float(),
            "snr": torch.tensor(snr),
            "tir": torch.tensor(tir),
        }
