from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from pixelgen.data.generate import build_spec
from pixelgen.evaluation.generators import (
    decoder_only_generator,
    encoder_decoder_generator,
)
from pixelgen.evaluation.validity import evaluate_validity
from pixelgen.models.decoder_only import DecoderOnlyTransformer
from pixelgen.models.encoder_decoder import EncoderDecoderTransformer

CFG_PATH = Path(__file__).resolve().parents[1] / "configs" / "data" / "shapes_16.yaml"


@pytest.fixture(scope="module")
def common(env):
    spec = build_spec(yaml.safe_load(CFG_PATH.read_text()))
    arrays = dict(np.load(env["dir"] / "test_comp.npz"))
    return arrays, spec


@pytest.fixture(scope="module", params=["decoder_only", "encoder_decoder"])
def generate_fn(request, env):
    tok, max_len, n_colors = env["tok"], env["max_len"], env["n_colors"]
    torch.manual_seed(0)
    if request.param == "decoder_only":
        model = DecoderOnlyTransformer(
            vocab_size=tok.vocab_size + 1 + n_colors, n_embd=32, n_head=4,
            n_layer=2, block_size=max_len + 256, dropout=0.0,
        )
        make = decoder_only_generator
    else:
        model = EncoderDecoderTransformer(
            src_vocab_size=tok.vocab_size, tgt_vocab_size=n_colors + 1, n_embd=32,
            n_head=4, n_layer=1, max_src_len=max_len, max_tgt_len=256, dropout=0.0,
        )
        make = encoder_decoder_generator
    return make(model, tok, max_len=max_len, n_colors=n_colors, device="cpu")


def test_validity_report_is_well_formed(common, generate_fn):
    arrays, spec = common
    result = evaluate_validity(arrays, spec, generate_fn, n_samples=12, batch_size=6)
    assert result["n_samples"] == 12
    assert 0.0 <= result["valid_rate"] <= 1.0


def test_validity_is_reproducible_with_a_seed(common, generate_fn):
    arrays, spec = common
    a = evaluate_validity(arrays, spec, generate_fn, n_samples=12, batch_size=6, seed=3)
    b = evaluate_validity(arrays, spec, generate_fn, n_samples=12, batch_size=6, seed=3)
    assert a["valid_rate"] == b["valid_rate"]