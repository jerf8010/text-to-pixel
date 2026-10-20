import pytest
import torch

from pixelgen.data.dataset import ShapesDataset
from pixelgen.data.dataset_gpt import DecoderOnlyDataset
from pixelgen.data.shapes import CANVAS

N_PIXELS = CANVAS * CANVAS


@pytest.fixture(scope="module")
def base(env):
    return ShapesDataset(env["dir"] / "train.npz", env["tok"], env["max_len"], env["n_colors"])


@pytest.fixture(scope="module")
def dds(base, env):
    return DecoderOnlyDataset(base, env["tok"].vocab_size, env["max_len"])


def sample_indices(ds):
    return [0, 1, 7, len(ds) // 2, len(ds) - 1]


def test_length_shapes_and_dtypes(dds, base, env):
    assert len(dds) == len(base)
    item = dds[0]
    expected = env["max_len"] + N_PIXELS  # text + <sep> + pixels, minus the shifted token
    assert item["x"].shape == (expected,)
    assert item["y"].shape == (expected,)
    assert item["x"].dtype == torch.long
    assert item["y"].dtype == torch.long


def test_id_ranges_do_not_overlap(dds, env):
    text_vocab_size = env["tok"].vocab_size
    assert dds.sep_id == text_vocab_size
    assert dds.offset == text_vocab_size + 1
    assert dds.vocab_size == text_vocab_size + 1 + env["n_colors"]


def test_inputs_are_valid_ids(dds):
    for i in sample_indices(dds):
        x = dds[i]["x"]
        assert int(x.min()) >= 0  # masking y must never leak -100 into x
        assert int(x.max()) < dds.vocab_size


def test_only_pixels_are_graded(dds, env):
    for i in sample_indices(dds):
        y = dds[i]["y"]
        n = env["max_len"]
        assert (y[:n] == DecoderOnlyDataset.IGNORE_INDEX).all()
        graded = y[n:]
        assert (graded != DecoderOnlyDataset.IGNORE_INDEX).all()
        assert int(graded.min()) >= dds.offset
        assert int(graded.max()) < dds.vocab_size


def test_targets_match_the_image(dds, base, env):
    for i in sample_indices(dds):
        graded = dds[i]["y"][env["max_len"]:]
        assert torch.equal(graded - dds.offset, base[i]["tgt_out"])


def test_input_layout(dds, base, env):
    n = env["max_len"]
    for i in sample_indices(dds):
        x = dds[i]["x"]
        item = base[i]
        assert torch.equal(x[:n], item["src"])  # caption, padded
        assert x[n] == dds.sep_id  # separator right after the caption
        assert torch.equal(x[n + 1:], item["tgt_out"][:-1] + dds.offset)  # pixels shifted