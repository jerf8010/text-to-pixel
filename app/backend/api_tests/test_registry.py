import numpy as np
import pytest

from pixelgen_api.errors import ConfigError, UnknownModelError
from pixelgen_api.services.registry import (
    ModelRegistry,
    build_vocabulary_groups,
    sample_examples,
)
from api_tests.fakes import make_entry


def test_registry_keeps_configuration_order():
    state: dict = {"seed": 0}
    registry = ModelRegistry([make_entry("b", "B", "x", state), make_entry("a", "A", "x", state)])
    assert [e.id for e in registry.all()] == ["b", "a"]


def test_registry_rejects_duplicates_and_empty_lists():
    state: dict = {"seed": 0}
    with pytest.raises(ConfigError):
        ModelRegistry([make_entry("a", "A", "x", state), make_entry("a", "A2", "x", state)])
    with pytest.raises(ConfigError):
        ModelRegistry([])


def test_unknown_model_lists_the_available_ones():
    state: dict = {"seed": 0}
    registry = ModelRegistry([make_entry("a", "A", "x", state)])
    with pytest.raises(UnknownModelError) as err:
        registry.get("zzz")
    assert err.value.details == {"available": ["a"]}


def test_vocabulary_groups():
    words = ["a", "small", "large", "red", "circle", "left", "of"]
    config = {"sizes": {"small": 2, "large": 3}, "colors": ["red"], "shapes": ["circle"]}
    assert build_vocabulary_groups(words, config) == {
        "sizes": ["small", "large"],
        "colors": ["red"],
        "shapes": ["circle"],
        "other": ["a", "left", "of"],
    }


def test_sample_examples_are_distinct_and_deterministic(tmp_path):
    captions = np.array(["a b", "a b", "c d", "e f", "g h", "i j"])
    np.savez(tmp_path / "split.npz", captions=captions)
    first = sample_examples(tmp_path, "split", n=3, seed=0)
    assert first == sample_examples(tmp_path, "split", n=3, seed=0)
    assert len(first) == len(set(first)) == 3
    assert set(first) <= set(captions.tolist())


def test_sample_examples_never_asks_for_more_than_exist(tmp_path):
    np.savez(tmp_path / "split.npz", captions=np.array(["a b", "c d"]))
    assert len(sample_examples(tmp_path, "split", n=10)) == 2
