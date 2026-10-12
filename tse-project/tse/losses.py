"""Loss and metric: scale-invariant SDR."""
import torch


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """SI-SDR in dB for tensors of shape (B, T). Returns shape (B,)."""
    est = est.float()
    ref = ref.float()
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    proj = alpha * ref
    noise = est - proj
    return 10 * torch.log10((proj.pow(2).sum(-1) + eps) / (noise.pow(2).sum(-1) + eps))


def sisdr_loss(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Negative mean SI-SDR (minimise this)."""
    return -si_sdr(est, ref).mean()
