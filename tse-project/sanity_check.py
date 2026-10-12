"""Quick correctness checks. No downloads needed. Run this FIRST:  python sanity_check.py"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from tse.losses import si_sdr, sisdr_loss
from tse.mixing import mix_signals, power
from tse.models import TSEConvTasNet, count_params


def check_mixing():
    rng = np.random.default_rng(0)
    t, i, n = (rng.standard_normal(16000) for _ in range(3))
    mix, tgt = mix_signals(t, i, n, snr_db=5.0, tir_db=3.0, gain=1.0, peak=1e9)
    # recover noise + interferer parts and verify the requested ratios
    gi = np.sqrt(power(t) / (power(i) * 10 ** 0.3))
    gn = np.sqrt(power(t) / (power(n) * 10 ** 0.5))
    snr = 10 * np.log10(power(t) / power(gn * n))
    tir = 10 * np.log10(power(t) / power(gi * i))
    assert abs(snr - 5.0) < 1e-3 and abs(tir - 3.0) < 1e-3
    assert np.allclose(mix, t + gi * i + gn * n, atol=1e-5)
    print("[ok] mixing: SNR/TIR match requested values")


def check_sisdr():
    x = torch.randn(2, 16000)
    assert si_sdr(x, x).min() > 60
    assert si_sdr(0.5 * x, x).min() > 60  # scale invariant
    assert si_sdr(torch.randn(2, 16000), x).max() < 5
    print("[ok] SI-SDR: identical/scaled signals ~ high, unrelated ~ low")


def check_model(cond):
    torch.manual_seed(0)
    m = TSEConvTasNet(N=64, L=16, B=32, H=64, P=3, X=4, R=2, cond=cond).eval()
    e = F.normalize(torch.randn(2, 192), dim=-1)
    x = torch.randn(2, 16000)
    with torch.no_grad():
        y = m(x, e)
    assert y.shape == x.shape, y.shape
    # odd length also works
    with torch.no_grad():
        assert m(torch.randn(1, 12345), e[:1]).shape == (1, 12345)
    # causality: changing the future must not change the past (up to one encoder window)
    s = 8000
    x2 = x.clone()
    x2[:, s:] += torch.randn(2, 16000 - s)
    with torch.no_grad():
        y2 = m(x2, e)
    safe = s - m.L - 1
    diff = (y[:, :safe] - y2[:, :safe]).abs().max().item()
    assert diff < 1e-4, f"model is not causal (diff={diff})"
    # one training step
    m.train()
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    tgt = torch.randn(2, 16000)
    loss = sisdr_loss(m(x, e), tgt)
    loss.backward()
    opt.step()
    print(f"[ok] model cond={cond:8s} shape, causality (diff={diff:.1e}), train step; params={count_params(m):,}")


def check_speed():
    m = TSEConvTasNet(cond="film").eval()  # default (small) config
    e = F.normalize(torch.randn(1, 192), dim=-1)
    x = torch.randn(1, 16000 * 4)
    torch.set_num_threads(4)
    with torch.no_grad():
        m(x, e)
        t0 = time.time()
        m(x, e)
        dt = time.time() - t0
    print(f"[ok] default model: {count_params(m):,} params, RTF on this CPU = {dt / 4:.3f} (4 s clip, 4 threads)")


if __name__ == "__main__":
    check_mixing()
    check_sisdr()
    for c in ["none", "concat", "film", "gating"]:
        check_model(c)
    check_speed()
    print("\nAll sanity checks passed.")
