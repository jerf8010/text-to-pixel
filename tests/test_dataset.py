from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from pixelgen.data.dataset import ShapesDataset
from pixelgen.data.shapes import CANVAS

CFG_PATH = Path(__file__).resolve().parents[1] / "configs" / "data" / "shapes_16.yaml"
SMALL_SPLITS = dict(n_train=400, n_val=60, n_test_iid=60, n_test_comp=120)
N_PIXELS = CANVAS * CANVAS

@pytest.fixture(scope="module")
def ds(env):
    return ShapesDataset(env["dir"] / "train.npz", env["tok"], env["max_len"], env["n_colors"])


@pytest.fixture(scope="module")
def raw(env):
    """The arrays exactly as stored on disk, for comparison."""
    return dict(np.load(env["dir"] / "train.npz"))


def sample_indices(ds):
    return [0, 1, 7, len(ds) // 2, len(ds) - 1]


def test_length_matches_split(ds, env):
    assert len(ds) == env["manifest"]["counts"]["train"]


def test_item_shapes_and_dtypes(ds, env):
    item = ds[0]
    assert item["src"].shape == (env["max_len"],)
    assert item["src_pad_mask"].shape == (env["max_len"],)
    assert item["tgt_in"].shape == (N_PIXELS,)
    assert item["tgt_out"].shape == (N_PIXELS,)
    assert item["src"].dtype == torch.long
    assert item["tgt_in"].dtype == torch.long
    assert item["tgt_out"].dtype == torch.long
    assert item["src_pad_mask"].dtype == torch.bool


def test_teacher_forcing_shift(ds):
    for i in sample_indices(ds):
        item = ds[i]
        assert item["tgt_in"][0] == ds.bos_id
        assert torch.equal(item["tgt_in"][1:], item["tgt_out"][:-1])


def test_target_reconstructs_the_stored_image(ds, raw):
    for i in sample_indices(ds):
        image = ds[i]["tgt_out"].reshape(CANVAS, CANVAS).numpy()
        assert np.array_equal(image, raw["images"][i])


def test_pad_mask_marks_exactly_the_padding(ds, env):
    for i in sample_indices(ds):
        item = ds[i]
        n_words = len(ds.captions[i].split())
        assert torch.equal(item["src_pad_mask"], item["src"] == ds.pad_id)
        assert int(item["src_pad_mask"].sum()) == env["max_len"] - n_words


def test_bos_is_outside_the_palette(ds, env, raw):
    assert ds.bos_id == env["n_colors"]
    assert ds.image_vocab_size == env["n_colors"] + 1
    assert int(raw["images"].max()) < env["n_colors"]  # pixels never collide with <bos>
    item = ds[0]
    assert (item["tgt_in"][1:] != ds.bos_id).all()  # <bos> only appears at position 0


def test_dataloader_batches(ds, env):
    batch = next(iter(DataLoader(ds, batch_size=8)))
    assert batch["src"].shape == (8, env["max_len"])
    assert batch["src_pad_mask"].shape == (8, env["max_len"])
    assert batch["tgt_in"].shape == (8, N_PIXELS)
    assert batch["tgt_out"].shape == (8, N_PIXELS)