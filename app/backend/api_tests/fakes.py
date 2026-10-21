"""Test doubles: a service with fake models, so no torch or checkpoints are needed."""
from __future__ import annotations

import numpy as np
from pixelgen_api.services.generation import GenerationService
from pixelgen_api.services.registry import DatasetInfo, ExampleSet, ModelEntry, ModelRegistry

WORDS = ["a", "small", "large", "red", "blue", "circle", "square", "left", "right", "of"]
PALETTE = [[0, 0, 0], [220, 40, 40], [40, 80, 220]] + [[i * 20, i * 20, i * 20] for i in range(3, 9)]


def make_entry(model_id: str, label: str, kind: str, state: dict) -> ModelEntry:
    """A model whose output depends only on the seed last set through ``state``."""

    def make_generator(temperature: float):
        def generate(captions: list[str]) -> np.ndarray:
            rng = np.random.default_rng(state["seed"])
            return rng.integers(0, 9, size=(len(captions), 16, 16), dtype=np.uint8)

        return generate

    return ModelEntry(model_id, label, kind, 1000, f"{label} (fake)", make_generator)


def build_service() -> tuple[GenerationService, dict]:
    """Return a service with two fake models and the state dict that records the seed."""
    state: dict = {"seed": None, "seeds_seen": []}

    def seed_fn(seed: int) -> None:
        state["seed"] = seed
        state["seeds_seen"].append(seed)

    registry = ModelRegistry(
        [
            make_entry("decoder-only", "Decoder-only", "decoder_only", state),
            make_entry("encoder-decoder", "Encoder-decoder", "encoder_decoder", state),
        ]
    )
    dataset = DatasetInfo(
        words=WORDS,
        groups={
            "sizes": ["small", "large"],
            "colors": ["red", "blue"],
            "shapes": ["circle", "square"],
            "other": ["a", "left", "right", "of"],
        },
        max_words=10,
        palette=PALETTE,
        examples=[ExampleSet("Seen", "Seen in training", ["a small red circle left of a large blue square"])],
    )
    return GenerationService(registry, dataset, seed_fn=seed_fn), state
