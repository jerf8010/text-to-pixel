from pathlib import Path

import numpy as np
import pytest
import yaml

from pixelgen.data.generate import build_spec, enumerate_single_scenes, sample_double_scene
from pixelgen.data.shapes import render
from pixelgen.evaluation.metrics import valid_renders, scene_from_row

CFG_PATH = Path(__file__).resolve().parents[1] / "configs" / "data" / "shapes_16.yaml"


@pytest.fixture(scope="module")
def spec():
    return build_spec(yaml.safe_load(CFG_PATH.read_text()))


def test_single_object_has_at_most_nine_valid_images(spec):
    for scene in enumerate_single_scenes(spec)[:50]:
        valid = valid_renders(scene, spec)
        assert render(scene, spec).tobytes() in valid
        assert 1 <= len(valid) <= 9


def test_two_objects_have_at_most_81_valid_images(spec):
    rng = np.random.default_rng(0)
    checked = 0
    while checked < 50:
        scene = sample_double_scene(rng, spec)
        if scene is None:
            continue
        valid = valid_renders(scene, spec)
        assert render(scene, spec).tobytes() in valid
        assert 1 <= len(valid) <= 81
        checked += 1


def test_a_different_scene_is_not_valid(spec):
    scenes = enumerate_single_scenes(spec)
    first, other = scenes[0], scenes[-1]
    assert first.objs != other.objs
    assert render(other, spec).tobytes() not in valid_renders(first, spec)


def test_scene_rebuilt_from_stored_attributes_matches_the_stored_image(spec, env):
    arrays = dict(np.load(env["dir"] / "test_comp.npz"))
    for i in range(len(arrays["images"])):
        scene = scene_from_row(arrays, i)
        assert render(scene, spec).tobytes() == arrays["images"][i].tobytes()
        assert arrays["images"][i].tobytes() in valid_renders(scene, spec)