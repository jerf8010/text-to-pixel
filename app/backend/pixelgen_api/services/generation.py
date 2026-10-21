"""Image generation use case: validate, seed, run the chosen models, time them."""
from __future__ import annotations

import logging
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from pixelgen_api.limits import CANVAS, MAX_SEED
from pixelgen_api.services.prompt import PromptValidator
from pixelgen_api.services.registry import DatasetInfo, ModelRegistry

logger = logging.getLogger("pixelgen_api")


@dataclass(frozen=True)
class ModelOutput:
    """Images produced by one model."""

    id: str
    label: str
    images: list[list[list[int]]]  # variant -> row -> column -> palette index
    elapsed_ms: float


@dataclass(frozen=True)
class GenerationOutcome:
    """Result of one generation request."""

    prompt: str
    seed: int
    results: list[ModelOutput]


def _torch_seed(seed: int) -> None:
    """Seed PyTorch's global generator (imported lazily to keep the service light)."""
    import torch

    torch.manual_seed(seed)


class GenerationService:
    """Generates images for a prompt with one or more models.

    All models receive the same seed, so a result can be reproduced and two models
    can be compared under the same sampling conditions. Sampling uses PyTorch's
    global random state, so requests are serialised with a lock.

    Args:
        registry: the served models.
        dataset: vocabulary and palette information.
        seed_fn: function that seeds the sampler; injectable for tests.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        dataset: DatasetInfo,
        *,
        seed_fn: Callable[[int], None] = _torch_seed,
    ) -> None:
        self.registry = registry
        self.dataset = dataset
        self._validator = PromptValidator(dataset.words, dataset.max_words)
        self._seed_fn = seed_fn
        self._lock = threading.Lock()

    def generate(
        self,
        prompt: str,
        model_ids: list[str],
        *,
        n_variants: int = 1,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> GenerationOutcome:
        """Generate ``n_variants`` images per model.

        Raises:
            InvalidPromptError: if the prompt is not valid for the models.
            UnknownModelError: if a model id is not served.
        """
        canonical = self._validator.validate(prompt)
        entries = [self.registry.get(i) for i in dict.fromkeys(model_ids)]
        used_seed = seed if seed is not None else secrets.randbelow(MAX_SEED + 1)

        results: list[ModelOutput] = []
        with self._lock:
            for entry in entries:
                generate = entry.make_generator(temperature)
                self._seed_fn(used_seed)
                started = time.perf_counter()
                images = generate([canonical] * n_variants)
                elapsed_ms = (time.perf_counter() - started) * 1000
                if images.shape != (n_variants, CANVAS, CANVAS):
                    raise RuntimeError(f"model '{entry.id}' returned images of shape {images.shape}")
                results.append(ModelOutput(entry.id, entry.label, images.tolist(), elapsed_ms))
                logger.info("%s generated %d image(s) in %.0f ms", entry.id, n_variants, elapsed_ms)
        return GenerationOutcome(canonical, used_seed, results)
