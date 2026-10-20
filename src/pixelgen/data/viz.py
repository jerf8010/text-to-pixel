"""Visualización de muestras. Usa Figure directamente para no tocar el backend de matplotlib."""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure


def save_grid(
    images: np.ndarray,
    captions: list[str],
    palette_rgb: np.ndarray,
    path: str | Path,
    cols: int = 8,
    title: str | None = None,
) -> None:
    """Guarda una cuadrícula de imágenes de índices de paleta con su caption."""
    n = len(images)
    rows = (n + cols - 1) // cols
    fig = Figure(figsize=(cols * 1.6, rows * 1.9))
    axes = np.atleast_1d(fig.subplots(rows, cols)).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, img, cap in zip(axes, images, captions):
        ax.imshow(palette_rgb[img], interpolation="nearest")
        ax.set_title(textwrap.fill(cap, 24), fontsize=6)
    if title:
        fig.suptitle(title, fontsize=10)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
