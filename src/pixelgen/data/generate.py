"""Muestreo de escenas, splits composicionales y escritura a disco."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .shapes import (
    CANVAS,
    COLOR_RGB,
    JITTERS,
    KNOWN_SHAPES,
    POS_NAMES,
    REL_NAMES,
    Obj,
    Scene,
    ShapeSpec,
    caption,
    render,
)

SEEN_SPLITS = ("train", "val", "test_iid")


def build_spec(cfg: dict) -> ShapeSpec:
    if cfg.get("canvas", CANVAS) != CANVAS:
        raise ValueError(f"El generador está fijado a {CANVAS}x{CANVAS}")
    unknown_colors = set(cfg["colors"]) - set(COLOR_RGB)
    unknown_shapes = set(cfg["shapes"]) - set(KNOWN_SHAPES)
    if unknown_colors or unknown_shapes:
        raise ValueError(f"Colores/formas desconocidos: {unknown_colors | unknown_shapes}")
    sizes = tuple((name, int(r)) for name, r in cfg["sizes"].items())
    return ShapeSpec(tuple(cfg["colors"]), tuple(cfg["shapes"]), sizes)


def heldout_ids(cfg: dict, spec: ShapeSpec) -> frozenset[tuple[int, int]]:
    return frozenset(
        (spec.colors.index(c), spec.shapes.index(s)) for c, s in cfg["heldout_pairs"]
    )


def is_heldout(scene: Scene, heldout: frozenset[tuple[int, int]]) -> bool:
    return any((o.color, o.shape) in heldout for o in scene.objs)


def enumerate_single_scenes(spec: ShapeSpec) -> list[Scene]:
    """Todas las escenas de un objeto (incluido el jitter), en orden determinista."""
    scenes = []
    for color in range(len(spec.colors)):
        for shape in range(len(spec.shapes)):
            for size in range(len(spec.sizes)):
                for pos in range(len(POS_NAMES)):
                    for jitter in JITTERS:
                        scenes.append(Scene((Obj(color, shape, size),), pos, -1, (jitter,)))
    return scenes


def sample_double_scene(rng: np.random.Generator, spec: ShapeSpec) -> Scene | None:
    """Escena de dos objetos al azar; None si son idénticos o se solapan."""
    nc, ns, nz = len(spec.colors), len(spec.shapes), len(spec.sizes)
    objs = tuple(
        Obj(int(rng.integers(nc)), int(rng.integers(ns)), int(rng.integers(nz)))
        for _ in range(2)
    )
    if objs[0] == objs[1]:
        return None
    rel = int(rng.integers(len(REL_NAMES)))
    jitter = tuple(JITTERS[int(rng.integers(len(JITTERS)))] for _ in range(2))
    scene = Scene(objs, -1, rel, jitter)
    return scene if render(scene, spec) is not None else None


def generate_splits(cfg: dict) -> tuple[ShapeSpec, dict[str, list[Scene]]]:
    """Devuelve el vocabulario y las escenas de train, val, test_iid y test_comp."""
    spec = build_spec(cfg)
    heldout = heldout_ids(cfg, spec)
    rng = np.random.default_rng(cfg["seed"])
    quota = {
        "train": cfg["n_train"],
        "val": cfg["n_val"],
        "test_iid": cfg["n_test_iid"],
        "test_comp": cfg["n_test_comp"],
    }

    # 1) Escenas de un objeto: se enumeran todas y se reparten entre splits.
    singles = enumerate_single_scenes(spec)
    singles = [singles[i] for i in rng.permutation(len(singles))]
    seen = [s for s in singles if not is_heldout(s, heldout)]
    comp = [s for s in singles if is_heldout(s, heldout)]
    a, b = int(0.8 * len(seen)), int(0.9 * len(seen))
    pools = {"train": seen[:a], "val": seen[a:b], "test_iid": seen[b:], "test_comp": comp}
    splits = {
        k: list(pools[k][: int(round(cfg["single_fraction"] * quota[k]))]) for k in quota
    }
    used = {s for scenes in splits.values() for s in scenes}

    # 2) Escenas de dos objetos: se muestrean hasta completar la cuota de cada split.
    need = {k: quota[k] - len(splits[k]) for k in quota}
    if any(v < 0 for v in need.values()):
        raise ValueError("single_fraction deja cuotas negativas; revisa la configuración")
    max_tries = 100 * sum(quota.values()) + 100_000
    tries = 0
    while any(v > 0 for v in need.values()):
        tries += 1
        if tries > max_tries:
            raise RuntimeError("No se pudieron completar las cuotas; reduce los tamaños")
        scene = sample_double_scene(rng, spec)
        if scene is None or scene in used:
            continue
        if is_heldout(scene, heldout):
            target = "test_comp"
        else:
            open_splits = [k for k in SEEN_SPLITS if need[k] > 0]
            if not open_splits:
                continue
            weights = np.array([need[k] for k in open_splits], dtype=float)
            target = open_splits[int(rng.choice(len(open_splits), p=weights / weights.sum()))]
        if need[target] == 0:
            continue
        splits[target].append(scene)
        used.add(scene)
        need[target] -= 1

    # 3) Barajar con la semilla: los prefijos de train son subconjuntos anidados.
    for k in splits:
        order = rng.permutation(len(splits[k]))
        splits[k] = [splits[k][i] for i in order]

    check_splits(splits, heldout, spec)
    return spec, splits


def check_splits(
    splits: dict[str, list[Scene]], heldout: frozenset[tuple[int, int]], spec: ShapeSpec
) -> None:
    for k in SEEN_SPLITS:
        assert not any(is_heldout(s, heldout) for s in splits[k]), f"{k} contiene pares retenidos"
    assert all(is_heldout(s, heldout) for s in splits["test_comp"]), "test_comp tiene escenas vistas"
    names = list(splits)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            assert not (set(splits[x]) & set(splits[y])), f"{x} y {y} comparten escenas"
    # Cada color y forma retenidos debe verse en train con otros compañeros.
    train_objs = [o for s in splits["train"] for o in s.objs]
    train_colors = {o.color for o in train_objs}
    train_shapes = {o.shape for o in train_objs}
    for color, shape in heldout:
        assert color in train_colors and shape in train_shapes, "par retenido sin cobertura"


def scenes_to_arrays(scenes: list[Scene], spec: ShapeSpec) -> dict[str, np.ndarray]:
    n = len(scenes)
    arrays = {
        "images": np.zeros((n, CANVAS, CANVAS), dtype=np.uint8),
        "n_obj": np.zeros(n, dtype=np.uint8),
        "color": -np.ones((n, 2), dtype=np.int8),
        "shape": -np.ones((n, 2), dtype=np.int8),
        "size": -np.ones((n, 2), dtype=np.int8),
        "pos": np.array([s.pos for s in scenes], dtype=np.int8),
        "rel": np.array([s.rel for s in scenes], dtype=np.int8),
        "jitter": np.zeros((n, 2, 2), dtype=np.int8),
    }
    captions = []
    for i, scene in enumerate(scenes):
        img = render(scene, spec)
        assert img is not None
        arrays["images"][i] = img
        arrays["n_obj"][i] = len(scene.objs)
        for j, (obj, jit) in enumerate(zip(scene.objs, scene.jitter)):
            arrays["color"][i, j] = obj.color
            arrays["shape"][i, j] = obj.shape
            arrays["size"][i, j] = obj.size
            arrays["jitter"][i, j] = jit
        captions.append(caption(scene, spec))
    arrays["captions"] = np.array(captions, dtype=str)
    return arrays


def hash_arrays(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in sorted(arrays):
        arr = arrays[key]
        h.update(key.encode())
        if arr.dtype.kind == "U":
            h.update("\n".join(arr.tolist()).encode())
        else:
            h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def write_dataset(
    cfg: dict, spec: ShapeSpec, splits: dict[str, list[Scene]], out_dir: Path
) -> dict:
    """Escribe un .npz por split y un manifest.json. Devuelve el manifiesto."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts, hashes, words = {}, {}, set()
    for name, scenes in splits.items():
        arrays = scenes_to_arrays(scenes, spec)
        np.savez_compressed(out_dir / f"{name}.npz", **arrays)
        counts[name] = len(scenes)
        hashes[name] = hash_arrays(arrays)
        for text in arrays["captions"]:
            words.update(text.split())
    manifest = {
        "name": cfg["name"],
        "seed": cfg["seed"],
        "config": cfg,
        "counts": counts,
        "sha256": hashes,
        "text_vocab": sorted(words),
        "palette_rgb": spec.palette_rgb().tolist(),
        "numpy_version": np.__version__,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
