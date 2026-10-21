"""Wrap each architecture into a ``generate_fn(captions) -> images`` for evaluation."""
from __future__ import annotations

import torch

from pixelgen.data.tokenizer import TextTokenizer
from pixelgen.evaluation.sampling import generate_images, generate_images_encdec
from pixelgen.evaluation.validity import GenerateFn


def decoder_only_generator(
    model: torch.nn.Module,
    tok: TextTokenizer,
    *,
    max_len: int,
    n_colors: int,
    device: str,
    temperature: float = 1.0,
) -> GenerateFn:
    """Return a ``generate_fn`` that samples images with a decoder-only model."""
    sep_id = tok.vocab_size
    offset = sep_id + 1

    def generate(captions):
        prefixes = [tok.encode(c, max_len) + [sep_id] for c in captions]
        return generate_images(model, prefixes, offset, n_colors, device, temperature)

    return generate


def encoder_decoder_generator(
    model: torch.nn.Module,
    tok: TextTokenizer,
    *,
    max_len: int,
    n_colors: int,
    device: str,
    temperature: float = 1.0,
) -> GenerateFn:
    """Return a ``generate_fn`` that samples images with an encoder-decoder model."""

    def generate(captions):
        src = torch.tensor([tok.encode(c, max_len) for c in captions], dtype=torch.long)
        src_pad_mask = src == tok.pad_id
        return generate_images_encdec(
            model, src, src_pad_mask, n_colors, device, temperature
        )

    return generate