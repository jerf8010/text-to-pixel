"""Catalogue of the models the app serves, and the loader that builds it."""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from pixelgen_api.errors import ConfigError, UnknownModelError

logger = logging.getLogger("pixelgen_api")

#: Maps a list of captions to a uint8 array of shape (B, 16, 16).
GenerateFn = Callable[[list[str]], np.ndarray]

EXAMPLES_PER_GROUP = 4


@dataclass(frozen=True)
class ModelEntry:
    """One served model.

    Attributes:
        id: stable identifier used by the API.
        label: name shown in the interface.
        kind: ``decoder_only`` or ``encoder_decoder``.
        n_params: number of trainable parameters.
        description: one-line explanation for the interface.
        make_generator: builds a ``GenerateFn`` for a sampling temperature.
    """

    id: str
    label: str
    kind: str
    n_params: int
    description: str
    make_generator: Callable[[float], GenerateFn]


@dataclass(frozen=True)
class ExampleSet:
    """A titled group of example prompts taken from the dataset."""

    title: str
    description: str
    prompts: list[str]


@dataclass(frozen=True)
class DatasetInfo:
    """Facts about the dataset that the interface needs.

    Attributes:
        words: text vocabulary.
        groups: vocabulary split into ``sizes``, ``colors``, ``shapes`` and ``other``.
        max_words: maximum prompt length in words.
        palette: RGB color of every palette index.
        examples: example prompts, seen and held-out combinations.
    """

    words: list[str]
    groups: dict[str, list[str]]
    max_words: int
    palette: list[list[int]]
    examples: list[ExampleSet] = field(default_factory=list)


class ModelRegistry:
    """Lookup of served models by id, preserving the configured order."""

    def __init__(self, entries: list[ModelEntry]) -> None:
        if not entries:
            raise ConfigError("at least one model is required")
        ids = [e.id for e in entries]
        if len(set(ids)) != len(ids):
            raise ConfigError(f"duplicate model ids: {ids}")
        self._entries = {e.id: e for e in entries}

    def all(self) -> list[ModelEntry]:
        """Return all entries in configuration order."""
        return list(self._entries.values())

    def get(self, model_id: str) -> ModelEntry:
        """Return one entry or raise ``UnknownModelError``."""
        try:
            return self._entries[model_id]
        except KeyError:
            raise UnknownModelError(
                f"Unknown model '{model_id}'.", details={"available": list(self._entries)}
            ) from None


def build_vocabulary_groups(words: list[str], data_config: dict) -> dict[str, list[str]]:
    """Split ``words`` into sizes, colors, shapes and everything else."""
    sizes = [w for w in words if w in set(data_config.get("sizes", {}))]
    colors = [w for w in words if w in set(data_config.get("colors", []))]
    shapes = [w for w in words if w in set(data_config.get("shapes", []))]
    used = set(sizes) | set(colors) | set(shapes)
    return {
        "sizes": sizes,
        "colors": colors,
        "shapes": shapes,
        "other": [w for w in words if w not in used],
    }


def sample_examples(data_dir: Path, split: str, n: int, seed: int = 0) -> list[str]:
    """Pick ``n`` distinct captions from a dataset split, deterministically."""
    captions = np.load(data_dir / f"{split}.npz")["captions"]
    unique = sorted({str(c) for c in captions})
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(unique), size=min(n, len(unique)), replace=False)
    return [unique[i] for i in sorted(chosen)]


def load_catalog(config_path: Path, device: str) -> tuple[ModelRegistry, DatasetInfo]:
    """Load every model listed in ``config_path`` from its MLflow run.

    Heavy imports (torch, mlflow, pixelgen) happen here, so the rest of the API
    can be imported and tested without them.

    Raises:
        ConfigError: if the file is malformed or the models disagree on the data.
    """
    from pixelgen.evaluation.loading import load_run

    config = yaml.safe_load(config_path.read_text())
    items = config.get("models") or []
    if not items:
        raise ConfigError(f"{config_path} lists no models")

    entries: list[ModelEntry] = []
    reference = None
    ref_loaded = None
    for item in items:
        loaded = load_run(item["run_id"], device)
        entries.append(
            ModelEntry(
                id=item["id"],
                label=item["label"],
                kind=loaded.kind,
                n_params=sum(p.numel() for p in loaded.model.parameters()),
                description=item.get("description", ""),
                make_generator=loaded.generate_fn,
            )
        )
        signature = (loaded.manifest["text_vocab"], loaded.manifest["palette_rgb"])
        if reference is None:
            reference, ref_loaded = signature, loaded
        elif signature != reference:
            raise ConfigError(f"model '{item['id']}' was trained on a different vocabulary or palette")
        logger.info("loaded model %s (%s, %d params)", item["id"], loaded.kind, entries[-1].n_params)

    manifest = ref_loaded.manifest
    words = list(manifest["text_vocab"])
    examples = [
        ExampleSet(
            "Combinations seen in training",
            "Color-shape pairs the models saw while training.",
            sample_examples(ref_loaded.data_dir, "test_iid", EXAMPLES_PER_GROUP),
        ),
        ExampleSet(
            "Held-out combinations",
            "Color-shape pairs never seen in training: a test of generalisation.",
            sample_examples(ref_loaded.data_dir, "test_comp", EXAMPLES_PER_GROUP),
        ),
    ]
    dataset = DatasetInfo(
        words=words,
        groups=build_vocabulary_groups(words, manifest.get("config", {})),
        max_words=ref_loaded.sizes["max_len"],
        palette=[[int(c) for c in rgb] for rgb in manifest["palette_rgb"]],
        examples=examples,
    )
    return ModelRegistry(entries), dataset
