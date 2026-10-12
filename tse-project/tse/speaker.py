"""Pretrained ECAPA-TDNN speaker encoder (SpeechBrain), frozen."""
import torch
import torch.nn.functional as F


def load_ecapa(device, savedir="pretrained/ecapa"):
    try:
        from speechbrain.inference.speaker import EncoderClassifier  # speechbrain >= 1.0
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier  # older versions
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=savedir,
        run_opts={"device": str(device)},
    )


@torch.no_grad()
def embed_wave(classifier, wav: torch.Tensor, device) -> torch.Tensor:
    """wav: (B, T) float tensor at 16 kHz -> (B, 192) L2-normalised embeddings."""
    e = classifier.encode_batch(wav.to(device))  # (B, 1, 192)
    return F.normalize(e.squeeze(1), dim=-1).cpu()
