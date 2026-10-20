"""Validity rate: how often the model draws an image that matches its caption."""
from __future__ import annotations

import numpy as np
import torch

from pixelgen.data.shapes import ShapeSpec
from pixelgen.data.tokenizer import TextTokenizer
from pixelgen.evaluation.metrics import scene_from_row, valid_renders
from pixelgen.evaluation.sampling import generate_images


def evaluate_validity(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    spec: ShapeSpec,
    tok: TextTokenizer,
    *,
    max_len: int,
    sep_id: int,
    offset: int,
    n_colors: int,
    device: str,
    n_samples: int = 200,
    batch_size: int = 100,
    temperature: float = 1.0,
    seed: int = 0,
) -> dict[str, float]:
    """Generate one image per sampled caption and report how many are correct.

    An image is correct when it exactly matches one of the renders of the caption's scene
    (any jitter). The same ``seed`` selects the same captions, so different models can be
    compared on identical examples.

    Returns:
        ``valid_rate`` overall, ``valid_rate_1obj`` and ``valid_rate_2obj`` split by number
        of objects (NaN if a group has no examples), and ``n_samples``.
    """
    torch.manual_seed(seed)  # reproducible sampling
    rng = np.random.default_rng(seed)
    n = len(arrays["images"])
    indices = rng.choice(n, size=min(n_samples, n), replace=False)

    hits: dict[int, list[bool]] = {1: [], 2: []}
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        prefixes = [
            tok.encode(str(arrays["captions"][i]), max_len) + [sep_id] for i in batch
        ]
        images = generate_images(model, prefixes, offset, n_colors, device, temperature)
        for i, image in zip(batch, images):
            scene = scene_from_row(arrays, int(i))
            hits[len(scene.objs)].append(image.tobytes() in valid_renders(scene, spec))

    all_hits = hits[1] + hits[2]
    result = {"valid_rate": float(np.mean(all_hits)), "n_samples": float(len(all_hits))}
    for n_obj, group in hits.items():
        result[f"valid_rate_{n_obj}obj"] = float(np.mean(group)) if group else float("nan")
    return result