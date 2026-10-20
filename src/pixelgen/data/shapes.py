"""Primitivas del dataset sintético: dibujar objetos, renderizar escenas y captions.

Un caption es una función pura de los atributos de la escena, de modo que el texto
y la imagen no pueden desalinearse.

Convenciones:
  * Lienzo de 16x16 con índices de paleta: 0 = fondo, 1..K = colores.
  * Una escena tiene 1 objeto (con una posición) o 2 objetos (con una relación).
  * Cada objeto lleva un desplazamiento ("jitter") de -1, 0 o +1 píxel por eje.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CANVAS = 16

COLOR_RGB: dict[str, tuple[int, int, int]] = {
    "red": (220, 40, 40),
    "blue": (40, 80, 220),
    "green": (40, 170, 70),
    "yellow": (240, 210, 40),
    "purple": (140, 60, 180),
    "orange": (240, 140, 30),
    "white": (240, 240, 240),
    "cyan": (40, 200, 200),
}
BACKGROUND_RGB = (0, 0, 0)
KNOWN_SHAPES = ("circle", "square", "triangle", "cross")

POS_NAMES = ["left", "right", "top", "bottom", "center"]
POS_PHRASE = {
    "left": "on the left",
    "right": "on the right",
    "top": "at the top",
    "bottom": "at the bottom",
    "center": "in the center",
}
# Centro (fila, columna) de un objeto único según su posición
POS_CENTER = {
    "left": (8, 4),
    "right": (8, 11),
    "top": (4, 8),
    "bottom": (11, 8),
    "center": (8, 8),
}

REL_NAMES = ["left of", "right of", "above", "below"]
# Centros (fila, columna) de (A, B) para la escena "A <relación> B"
REL_CENTERS = {
    "left of": ((8, 4), (8, 11)),
    "right of": ((8, 11), (8, 4)),
    "above": ((4, 8), (11, 8)),
    "below": ((11, 8), (4, 8)),
}

JITTERS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))


@dataclass(frozen=True)
class ShapeSpec:
    """Vocabulario del dataset. Los atributos se guardan como índices en estas listas."""

    colors: tuple[str, ...]
    shapes: tuple[str, ...]
    sizes: tuple[tuple[str, int], ...]  # (nombre, radio)

    def palette_rgb(self) -> np.ndarray:
        rows = [BACKGROUND_RGB] + [COLOR_RGB[c] for c in self.colors]
        return np.array(rows, dtype=np.uint8)


@dataclass(frozen=True)
class Obj:
    color: int
    shape: int
    size: int


@dataclass(frozen=True)
class Scene:
    objs: tuple[Obj, ...]                 # 1 o 2 objetos
    pos: int                              # índice en POS_NAMES (1 objeto) o -1
    rel: int                              # índice en REL_NAMES (2 objetos) o -1
    jitter: tuple[tuple[int, int], ...]   # (dy, dx) por objeto


def shape_mask(shape: str, cy: int, cx: int, r: int, n: int = CANVAS) -> np.ndarray:
    """Máscara booleana (n, n) de una forma de radio r centrada en (cy, cx)."""
    y, x = np.ogrid[:n, :n]
    dy, dx = y - cy, x - cx
    if shape == "circle":
        return dy * dy + dx * dx <= r * r + 1
    if shape == "square":
        return (np.abs(dy) <= r - 1) & (np.abs(dx) <= r - 1)
    if shape == "triangle":  # vértice arriba, base abajo
        return (dy >= -r) & (dy <= r) & (np.abs(dx) <= (dy + r) // 2)
    if shape == "cross":
        return ((dy == 0) & (np.abs(dx) <= r)) | ((dx == 0) & (np.abs(dy) <= r))
    raise ValueError(f"Forma desconocida: {shape!r}")


def object_centers(scene: Scene) -> list[tuple[int, int]]:
    """Centros (fila, columna) de cada objeto, con su desplazamiento aplicado."""
    if len(scene.objs) == 1:
        base = [POS_CENTER[POS_NAMES[scene.pos]]]
    else:
        base = list(REL_CENTERS[REL_NAMES[scene.rel]])
    return [(cy + dy, cx + dx) for (cy, cx), (dy, dx) in zip(base, scene.jitter)]


def render(scene: Scene, spec: ShapeSpec) -> np.ndarray | None:
    """Imagen (16, 16) uint8 con índices de paleta, o None si los objetos se solapan."""
    img = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    occupied = np.zeros((CANVAS, CANVAS), dtype=bool)
    for obj, (cy, cx) in zip(scene.objs, object_centers(scene)):
        mask = shape_mask(spec.shapes[obj.shape], cy, cx, spec.sizes[obj.size][1])
        if (mask & occupied).any():
            return None
        occupied |= mask
        img[mask] = obj.color + 1
    return img


def object_phrase(obj: Obj, spec: ShapeSpec) -> str:
    size_name = spec.sizes[obj.size][0]
    return f"a {size_name} {spec.colors[obj.color]} {spec.shapes[obj.shape]}"


def caption(scene: Scene, spec: ShapeSpec) -> str:
    """Texto de la escena. No incluye el jitter: varias imágenes comparten caption."""
    if len(scene.objs) == 1:
        where = POS_PHRASE[POS_NAMES[scene.pos]]
        return f"{object_phrase(scene.objs[0], spec)} {where}"
    a, b = scene.objs
    return f"{object_phrase(a, spec)} {REL_NAMES[scene.rel]} {object_phrase(b, spec)}"
