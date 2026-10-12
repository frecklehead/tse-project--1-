"""Causal Conv-TasNet with speaker conditioning (none / concat / FiLM / gating)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CumLN(nn.Module):
    """Cumulative layer norm: statistics over channels and all frames up to t (causal)."""

    def __init__(self, channels, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):  # (B, C, T)
        dtype = x.dtype
        x = x.float()
        B, C, T = x.shape
        count = C * torch.arange(1, T + 1, device=x.device, dtype=torch.float32)
        mean = x.sum(1).cumsum(1) / count
        var = (x.pow(2).sum(1).cumsum(1) / count - mean.pow(2)).clamp(min=0)
        mean = mean.unsqueeze(1)
        std = (var + self.eps).sqrt().unsqueeze(1)
        y = (x - mean) / std * self.gamma + self.beta
        return y.to(dtype)


class CausalTCNBlock(nn.Module):
    def __init__(self, B, H, P, dilation):
        super().__init__()
        self.conv_in = nn.Conv1d(B, H, 1)
        self.act1 = nn.PReLU()
        self.norm1 = CumLN(H)
        self.pad = (P - 1) * dilation  # left padding only -> causal
        self.dw = nn.Conv1d(H, H, P, dilation=dilation, groups=H)
        self.act2 = nn.PReLU()
        self.norm2 = CumLN(H)
        self.conv_out = nn.Conv1d(H, B, 1)

    def forward(self, x):
        y = self.norm1(self.act1(self.conv_in(x)))
        y = F.pad(y, (self.pad, 0))
        y = self.norm2(self.act2(self.dw(y)))
        return x + self.conv_out(y)


class Conditioner(nn.Module):
    """Fuses a speaker embedding (B, E) into features (B, C, T)."""

    def __init__(self, mode, C, E):
        super().__init__()
        self.mode = mode
        if mode == "concat":
            self.proj = nn.Conv1d(C + E, C, 1)
        elif mode == "film":
            self.fc = nn.Linear(E, 2 * C)
        elif mode == "gating":
            self.fc = nn.Linear(E, C)
        elif mode != "none":
            raise ValueError(f"unknown conditioning mode: {mode}")

    def forward(self, x, e):
        if self.mode == "none":
            return x
        if self.mode == "concat":
            et = e.unsqueeze(-1).expand(-1, -1, x.size(-1))
            return self.proj(torch.cat([x, et], dim=1))
        if self.mode == "film":
            g, b = self.fc(e).chunk(2, dim=-1)
            return x * (1 + g.unsqueeze(-1)) + b.unsqueeze(-1)
        return x * torch.sigmoid(self.fc(e)).unsqueeze(-1)  # gating


class TSEConvTasNet(nn.Module):
    """forward(mixture (B,T), emb (B,E)) -> estimated target (B,T).

    Output sample t depends on input up to about t + L (one encoder window of
    lookahead, i.e. 1 ms at L=16, 16 kHz); everything else is strictly causal.
    """

    def __init__(self, N=256, L=16, B=128, H=256, P=3, X=8, R=3, emb_dim=192, cond="film"):
        super().__init__()
        self.L, self.stride = L, L // 2
        self.encoder = nn.Conv1d(1, N, L, stride=self.stride, bias=False)
        self.norm = CumLN(N)
        self.bottleneck = nn.Conv1d(N, B, 1)
        self.conds = nn.ModuleList()
        self.repeats = nn.ModuleList()
        for _ in range(R):
            self.conds.append(Conditioner(cond, B, emb_dim))
            self.repeats.append(nn.Sequential(*[CausalTCNBlock(B, H, P, 2 ** x) for x in range(X)]))
        self.mask = nn.Sequential(nn.PReLU(), nn.Conv1d(B, N, 1), nn.Sigmoid())
        self.decoder = nn.ConvTranspose1d(N, 1, L, stride=self.stride, bias=False)

    def _pad_amount(self, T):
        if T < self.L:
            return self.L - T
        return (self.stride - (T - self.L) % self.stride) % self.stride

    def forward(self, mix, emb):
        T = mix.size(-1)
        x = F.pad(mix, (0, self._pad_amount(T))).unsqueeze(1)
        w = F.relu(self.encoder(x))
        y = self.bottleneck(self.norm(w))
        for cond, rep in zip(self.conds, self.repeats):
            y = rep(cond(y, emb))
        out = self.decoder(w * self.mask(y)).squeeze(1)
        return out[..., :T]


MODEL_KEYS = ["N", "L", "B", "H", "P", "X", "R", "cond"]


def build_model(cfg: dict) -> TSEConvTasNet:
    return TSEConvTasNet(**{k: cfg[k] for k in MODEL_KEYS})


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
