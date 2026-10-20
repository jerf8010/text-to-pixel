from pathlib import Path

import numpy as np
import pytest
import yaml

from pixelgen.data.generate import (
    build_spec,
    enumerate_single_scenes,
    generate_splits,
    hash_arrays,
    heldout_ids,
    is_heldout,
    sample_double_scene,
    scenes_to_arrays,
)
from pixelgen.data.shapes import POS_NAMES, REL_NAMES, caption, render, shape_mask

CFG_PATH = Path(__file__).resolve().parents[1] / "configs" / "data" / "shapes_16.yaml"


@pytest.fixture(scope="module")
def cfg():
    c = yaml.safe_load(CFG_PATH.read_text())
    c.update(n_train=400, n_val=60, n_test_iid=60, n_test_comp=120)
    return c


@pytest.fixture(scope="module")
def spec(cfg):
    return build_spec(cfg)


def reference_area(shape, r):
    """Área de la forma en un lienzo grande, donde nada se recorta."""
    return int(shape_mask(shape, 32, 32, r, n=64).sum())


def test_single_objects_are_never_clipped(spec):
    for scene in enumerate_single_scenes(spec):
        img = render(scene, spec)
        obj = scene.objs[0]
        expected = reference_area(spec.shapes[obj.shape], spec.sizes[obj.size][1])
        assert int((img > 0).sum()) == expected


def test_double_scenes_do_not_overlap(spec):
    rng = np.random.default_rng(123)
    checked = 0
    while checked < 300:
        scene = sample_double_scene(rng, spec)
        if scene is None:
            continue
        img = render(scene, spec)
        expected = sum(
            reference_area(spec.shapes[o.shape], spec.sizes[o.size][1]) for o in scene.objs
        )
        assert int((img > 0).sum()) == expected  # si se solaparan, habría menos píxeles
        checked += 1


def test_captions_are_injective_in_attributes(spec):
    attrs_to_caps, caps_to_attrs = {}, {}
    for scene in enumerate_single_scenes(spec):
        attrs = (scene.objs, scene.pos)
        cap = caption(scene, spec)
        attrs_to_caps.setdefault(attrs, set()).add(cap)
        caps_to_attrs.setdefault(cap, set()).add(attrs)
    assert all(len(v) == 1 for v in attrs_to_caps.values())
    assert all(len(v) == 1 for v in caps_to_attrs.values())
    expected = len(spec.colors) * len(spec.shapes) * len(spec.sizes) * len(POS_NAMES)
    assert len(caps_to_attrs) == expected


def test_caption_mentions_every_attribute(spec):
    rng = np.random.default_rng(7)
    for _ in range(100):
        scene = sample_double_scene(rng, spec)
        if scene is None:
            continue
        text = caption(scene, spec)
        for o in scene.objs:
            assert spec.colors[o.color] in text and spec.shapes[o.shape] in text
        assert REL_NAMES[scene.rel] in text


def test_generation_is_deterministic(cfg):
    spec_a, a = generate_splits(cfg)
    spec_b, b = generate_splits(cfg)
    for name in a:
        assert hash_arrays(scenes_to_arrays(a[name], spec_a)) == hash_arrays(
            scenes_to_arrays(b[name], spec_b)
        )
    other = dict(cfg, seed=cfg["seed"] + 1)
    spec_c, c = generate_splits(other)
    assert hash_arrays(scenes_to_arrays(a["train"], spec_a)) != hash_arrays(
        scenes_to_arrays(c["train"], spec_c)
    )


def test_heldout_pairs_are_isolated(cfg):
    spec, splits = generate_splits(cfg)
    heldout = heldout_ids(cfg, spec)
    for name in ("train", "val", "test_iid"):
        assert not any(is_heldout(s, heldout) for s in splits[name])
    assert all(is_heldout(s, heldout) for s in splits["test_comp"])
    assert {n: len(v) for n, v in splits.items()} == {
        "train": cfg["n_train"],
        "val": cfg["n_val"],
        "test_iid": cfg["n_test_iid"],
        "test_comp": cfg["n_test_comp"],
    }


def test_image_values_are_valid_palette_indices(cfg):
    spec, splits = generate_splits(cfg)
    arrays = scenes_to_arrays(splits["train"], spec)
    assert arrays["images"].shape == (cfg["n_train"], 16, 16)
    assert arrays["images"].max() <= len(spec.colors)
