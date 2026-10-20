from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from pixelgen.data.generate import build_spec
from pixelgen.evaluation.validity import evaluate_validity
from pixelgen.models.decoder_only import DecoderOnlyTransformer

CFG_PATH = Path(__file__).resolve().parents[1] / "configs" / "data" / "shapes_16.yaml"


@pytest.fixture(scope="module")
def setup(env):
    spec = build_spec(yaml.safe_load(CFG_PATH.read_text()))
    tok, max_len, n_colors = env["tok"], env["max_len"], env["n_colors"]
    sep_id = tok.vocab_size
    offset = sep_id + 1
    torch.manual_seed(0)
    model = DecoderOnlyTransformer(
        vocab_size=offset + n_colors, n_embd=32, n_head=4, n_layer=2,
        block_size=max_len + 256, dropout=0.0,
    )
    arrays = dict(np.load(env["dir"] / "test_comp.npz"))
    kwargs = dict(max_len=max_len, sep_id=sep_id, offset=offset, n_colors=n_colors, device="cpu")
    return model, arrays, spec, tok, kwargs


def test_validity_report_is_well_formed(setup):
    model, arrays, spec, tok, kwargs = setup
    result = evaluate_validity(model, arrays, spec, tok, n_samples=12, batch_size=6, **kwargs)
    assert result["n_samples"] == 12
    assert 0.0 <= result["valid_rate"] <= 1.0


def test_validity_is_reproducible_with_a_seed(setup):
    model, arrays, spec, tok, kwargs = setup
    a = evaluate_validity(model, arrays, spec, tok, n_samples=12, batch_size=6, seed=3, **kwargs)
    b = evaluate_validity(model, arrays, spec, tok, n_samples=12, batch_size=6, seed=3, **kwargs)
    assert a["valid_rate"] == b["valid_rate"]