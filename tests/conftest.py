from pathlib import Path

import numpy as np
import pytest
import yaml

from pixelgen.data.generate import generate_splits, write_dataset
from pixelgen.data.tokenizer import TextTokenizer

CFG_PATH = Path(__file__).resolve().parents[1] / "configs" / "data" / "shapes_16.yaml"
SMALL_SPLITS = dict(n_train=400, n_val=60, n_test_iid=60, n_test_comp=120)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Generate a tiny dataset in a temporary folder so tests never touch data/processed."""
    cfg = yaml.safe_load(CFG_PATH.read_text())
    cfg.update(SMALL_SPLITS)
    spec, splits = generate_splits(cfg)
    out_dir = tmp_path_factory.mktemp("shapes")
    manifest = write_dataset(cfg, spec, splits, out_dir)

    max_len = 0
    for split in manifest["counts"]:
        captions = np.load(out_dir / f"{split}.npz")["captions"]
        max_len = max(max_len, max(len(c.split()) for c in captions))

    return {
        "dir": out_dir,
        "manifest": manifest,
        "tok": TextTokenizer(manifest["text_vocab"]),
        "max_len": max_len,
        "n_colors": len(manifest["palette_rgb"]),
    }