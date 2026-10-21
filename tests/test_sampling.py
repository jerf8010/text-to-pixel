"""Sampling images from the decoder-only model."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from pixelgen.data.shapes import CANVAS
import numpy as np
import torch

from pixelgen.evaluation.sampling import generate_image, generate_images
from pixelgen.models.decoder_only import DecoderOnlyTransformer

N_COLORS, OFFSET, MAX_LEN = 9, 30, 10
VOCAB = OFFSET + N_COLORS  # text ids + <sep> + colors


def make_model():
    torch.manual_seed(0)
    return DecoderOnlyTransformer(
        vocab_size=VOCAB, n_embd=32, n_head=4, n_layer=2,
        block_size=MAX_LEN + 256, dropout=0.0,
    )

@torch.no_grad()
def generate_image(
    model: torch.nn.Module,
    prefix_ids: list[int],
    offset: int,
    n_colors: int,
    device: str,
    temperature: float = 1.0,
) -> np.ndarray:
    """Sample a CANVAS x CANVAS image of palette indices, one pixel at a time.

    Args:
        model: decoder-only model trained on ``[caption, <sep>, pixels]`` sequences.
        prefix_ids: token ids of the padded caption followed by ``<sep>``.
        offset: id of palette index 0 in the model vocabulary.
        n_colors: number of palette indices; only these ids may be sampled.
        device: device the model lives on.
        temperature: softmax temperature (lower is more conservative).

    Returns:
        uint8 array of shape (CANVAS, CANVAS) with palette indices.
    """
    was_training = model.training
    model.eval()

    seq = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    lo, hi = offset, offset + n_colors  # ids that correspond to palette colors

    for _ in range(CANVAS * CANVAS):
        logits, _ = model(seq)  # (1, T, vocab)
        logits = logits[:, -1, lo:hi] / temperature  # last position, colors only
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1) + lo  # back to global ids
        seq = torch.cat([seq, nxt], dim=1)

    if was_training:
        model.train()
    pixels = seq[0, len(prefix_ids):] - offset
    return pixels.reshape(CANVAS, CANVAS).cpu().numpy().astype(np.uint8)


def test_batch_generation_shape_and_range():
    prefix = list(range(MAX_LEN)) + [OFFSET - 1]
    images = generate_images(make_model(), [prefix] * 3, OFFSET, N_COLORS, "cpu")
    assert images.shape == (3, 16, 16)
    assert images.dtype == np.uint8
    assert images.min() >= 0 and images.max() < N_COLORS

def test_generate_images_encdec_shape_range_and_seed():
    from pixelgen.evaluation.sampling import generate_images_encdec
    from pixelgen.models.encoder_decoder import EncoderDecoderTransformer

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
    assert a.max() < 9          # never samples the BOS id
    assert np.array_equal(a, b) # same seed, same images