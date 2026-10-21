"""Sampling images from the decoder-only and encoder-decoder models."""
from __future__ import annotations

import numpy as np
import torch

from pixelgen.evaluation.sampling import (
    generate_image,
    generate_images,
    generate_images_encdec,
)
from pixelgen.models.decoder_only import DecoderOnlyTransformer
from pixelgen.models.encoder_decoder import EncoderDecoderTransformer

N_COLORS, OFFSET, MAX_LEN = 9, 30, 10
VOCAB = OFFSET + N_COLORS  # text ids + <sep> + colors


def make_model():
    torch.manual_seed(0)
    return DecoderOnlyTransformer(
        vocab_size=VOCAB, n_embd=32, n_head=4, n_layer=2,
        block_size=MAX_LEN + 256, dropout=0.0,
    )


def test_batch_generation_shape_and_range():
    prefix = list(range(MAX_LEN)) + [OFFSET - 1]
    images = generate_images(make_model(), [prefix] * 3, OFFSET, N_COLORS, "cpu")
    assert images.shape == (3, 16, 16)
    assert images.dtype == np.uint8
    assert images.min() >= 0 and images.max() < N_COLORS


def test_single_generation_shape_and_range():
    prefix = list(range(MAX_LEN)) + [OFFSET - 1]
    image = generate_image(make_model(), prefix, OFFSET, N_COLORS, "cpu")
    assert image.shape == (16, 16)
    assert image.dtype == np.uint8
    assert image.max() < N_COLORS


def test_generate_images_encdec_shape_range_and_seed():
    torch.manual_seed(0)
    model = EncoderDecoderTransformer(
        src_vocab_size=29, tgt_vocab_size=10, n_embd=32, n_head=4,
        n_layer=1, max_src_len=10, max_tgt_len=256, dropout=0.0,
    )
    src = torch.randint(0, 29, (3, 10))
    pad = torch.zeros(3, 10, dtype=torch.bool)

    torch.manual_seed(1)
    a = generate_images_encdec(model, src, pad, n_colors=9, device="cpu")
    torch.manual_seed(1)
    b = generate_images_encdec(model, src, pad, n_colors=9, device="cpu")

    assert a.shape == (3, 16, 16)
    assert a.dtype == np.uint8
    assert a.max() < 9  # never samples the BOS id
    assert np.array_equal(a, b)  # same seed, same images