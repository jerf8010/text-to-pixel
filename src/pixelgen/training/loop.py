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
def evaluate(model, loader, device: str, max_batches: int = 20) -> float:
    """Mean loss over up to ``max_batches`` batches, with dropout off and no gradients."""
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x, y = batch["x"].to(device), batch["y"].to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)