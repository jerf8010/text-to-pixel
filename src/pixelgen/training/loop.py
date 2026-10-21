"""Small helpers shared by the training scripts."""
from __future__ import annotations

import torch


def get_device() -> str:
    """Return the best available device: cuda, mps or cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten nested dicts into dotted keys, e.g. {"a": {"b": 1}} -> {"a.b": 1}."""
    out = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten_dict(value, prefix=f"{name}."))
        else:
            out[name] = value
    return out


@torch.no_grad()
def evaluate(model, loader, device, kind: str, max_batches: int = 20) -> float:
    """Return the mean loss over up to ``max_batches`` batches of ``loader``.

    The model is put in eval mode for the pass and restored to train mode after.

    Args:
        model: a DecoderOnlyTransformer or an EncoderDecoderTransformer.
        loader: DataLoader yielding batches for the given ``kind``.
        device: device to run on.
        kind: ``"decoder_only"`` or ``"encoder_decoder"``.
        max_batches: cap on the number of batches, to keep evaluation fast.
    """
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        losses.append(compute_loss(model, batch, kind, device).item())
    model.train()
    return sum(losses) / len(losses)

def count_params(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_loss(model, batch: dict, kind: str, device) -> torch.Tensor:
    """Compute the training loss of one batch for either model type.

    Args:
        model: a DecoderOnlyTransformer or an EncoderDecoderTransformer.
        batch: dict of tensors produced by the matching dataset.
        kind: ``"decoder_only"`` or ``"encoder_decoder"``.
        device: device to move the batch to.
    """
    batch = {k: v.to(device) for k, v in batch.items()}
    if kind == "decoder_only":
        _, loss = model(batch["x"], batch["y"])
    elif kind == "encoder_decoder":
        _, loss = model(
            batch["src"], batch["tgt_in"], batch["src_pad_mask"], batch["tgt_out"]
        )
    else:
        raise ValueError(f"unknown model type: {kind}")
    return loss