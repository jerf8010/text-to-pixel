"""Sampling images from the decoder-only model."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from pixelgen.data.shapes import CANVAS


@torch.no_grad()
def generate_images(
    model: torch.nn.Module,
    prefix_ids: list[list[int]],
    offset: int,
    n_colors: int,
    device: str,
    temperature: float = 1.0,
) -> np.ndarray:
    """Sample a batch of CANVAS x CANVAS images, one pixel at a time, in parallel.

    Args:
        model: decoder-only model trained on ``[caption, <sep>, pixels]`` sequences.
        prefix_ids: one prefix per image (padded caption ids followed by ``<sep>``).
            All prefixes must have the same length.
        offset: id of palette index 0 in the model vocabulary.
        n_colors: number of palette indices; only these ids may be sampled.
        device: device the model lives on.
        temperature: softmax temperature (lower is more conservative).

    Returns:
        uint8 array of shape (B, CANVAS, CANVAS) with palette indices.
    """
    was_training = model.training
    model.eval()

    seq = torch.tensor(prefix_ids, dtype=torch.long, device=device)  # (B, L)
    prefix_len = seq.shape[1]
    lo, hi = offset, offset + n_colors  # ids that correspond to palette colors

    for _ in range(CANVAS * CANVAS):
        logits, _ = model(seq)  # (B, T, vocab)
        logits = logits[:, -1, lo:hi] / temperature  # last position, colors only
        probs = F.softmax(logits, dim=-1)  # (B, n_colors)
        nxt = torch.multinomial(probs, num_samples=1) + lo  # (B, 1), global ids
        seq = torch.cat([seq, nxt], dim=1)

    if was_training:
        model.train()
    pixels = seq[:, prefix_len:] - offset  # (B, CANVAS * CANVAS)
    return pixels.reshape(-1, CANVAS, CANVAS).cpu().numpy().astype(np.uint8)


def generate_image(model, prefix_ids, offset, n_colors, device, temperature=1.0) -> np.ndarray:
    """Sample a single image; returns a (CANVAS, CANVAS) uint8 array."""
    return generate_images(model, [prefix_ids], offset, n_colors, device, temperature)[0]