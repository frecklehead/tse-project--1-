"""Synthetic mixture creation: X = S_t + g_i * S_i + g_n * N."""
import numpy as np

EPS = 1e-8


def power(x: np.ndarray) -> float:
    return float(np.mean(np.square(x))) + EPS


def mix_signals(target, interferer, noise, snr_db, tir_db, gain=1.0, peak=0.95):
    """Mix target with an optional interferer and noise.

    snr_db: target power over noise power, in dB.
    tir_db: target power over interferer power, in dB (ignored if interferer is None).
    Returns (mixture, scaled_target), both float32 and peak-limited by the same factor.
    """
    pt = power(target)
    mixture = target.astype(np.float64).copy()
    if interferer is not None:
        gi = np.sqrt(pt / (power(interferer) * 10 ** (tir_db / 10)))
        mixture += gi * interferer
    if noise is not None:
        gn = np.sqrt(pt / (power(noise) * 10 ** (snr_db / 10)))
        mixture += gn * noise
    tgt = target.astype(np.float64)
    mixture = mixture * gain
    tgt = tgt * gain
    m = np.max(np.abs(mixture)) + EPS
    if m > peak:
        scale = peak / m
        mixture = mixture * scale
        tgt = tgt * scale
    return mixture.astype(np.float32), tgt.astype(np.float32)
