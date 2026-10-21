"""Metrics for generated images: does an image match the scene its caption describes?"""
from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np

from pixelgen.data.shapes import JITTERS, Obj, Scene, ShapeSpec, render


def valid_renders(scene: Scene, spec: ShapeSpec) -> set[bytes]:
    """Return every correct image for a scene, as raw bytes.

    A caption does not say how far each object is shifted, so the scene drawn with any
    jitter combination is a valid answer. Combinations in which the objects overlap
    cannot be rendered and are skipped.
    """
    valid = set()
    for jitter in product(JITTERS, repeat=len(scene.objs)):
        image = render(replace(scene, jitter=jitter), spec)
        if image is not None:
            valid.add(image.tobytes())
    return valid


def scene_from_row(arrays: dict[str, np.ndarray], i: int) -> Scene:
    """Rebuild the scene of example ``i`` from the attribute arrays stored in a split."""
    n_obj = int(arrays["n_obj"][i])
    objs = tuple(
        Obj(int(arrays["color"][i, j]), int(arrays["shape"][i, j]), int(arrays["size"][i, j]))
        for j in range(n_obj)
    )
    jitter = tuple(
        (int(arrays["jitter"][i, j, 0]), int(arrays["jitter"][i, j, 1])) for j in range(n_obj)
    )
    return Scene(objs, int(arrays["pos"][i]), int(arrays["rel"][i]), jitter)