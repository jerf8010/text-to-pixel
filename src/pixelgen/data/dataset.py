"""PyTorch dataset that serves tokenized captions and flattened pixel sequences."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pixelgen.data.tokenizer import TextTokenizer


class ShapesDataset(Dataset):
    """Caption -> image dataset for the encoder-decoder model.

    Each item is a dict of tensors:
        src:          caption token ids, right-padded to ``max_len``.      (max_len,)
        src_pad_mask: True where ``src`` is padding.                       (max_len,)
        tgt_in:       decoder input, ``[<bos>, p0, ..., p254]``.           (H*W,)
        tgt_out:      decoder target, ``[p0, ..., p255]``.                 (H*W,)

    Pixels are palette indices read row by row. ``tgt_in`` is ``tgt_out`` shifted one
    position to the right, which is what teacher forcing needs.

    Args:
        npz_path: path to a split file written by ``write_dataset`` (e.g. ``train.npz``).
        tokenizer: text tokenizer used to encode the captions.
        max_len: fixed caption length in tokens (the same for every split).
        n_colors: number of palette indices (background + colors). The ``<bos>`` id is
            ``n_colors``, the first id not used by the palette.
    """

    def __init__(
        self,
        npz_path: str | Path,
        tokenizer: TextTokenizer,
        max_len: int,
        n_colors: int,
    ) -> None:
        # Load every array once; indexing an NpzFile would re-read it from disk each time.
        arrays = dict(np.load(npz_path))

        images = arrays["images"]  # (N, H, W)
        self.images = images.reshape(len(images), -1)  # (N, H*W), row-major
        self.captions = arrays["captions"]

        # Encode all captions once instead of on every __getitem__ call.
        self.src = np.array(
            [tokenizer.encode(c, max_len) for c in self.captions], dtype=np.int64
        )

        self.bos_id = n_colors
        self.pad_id = tokenizer.pad_id

    @property
    def image_vocab_size(self) -> int:
        """Number of ids the decoder can receive as input: palette indices plus <bos>."""
        return self.bos_id + 1

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        pixels = torch.from_numpy(self.images[i]).long()  # (H*W,)

        # Teacher forcing: the decoder sees <bos> plus every pixel except the last one.
        bos = torch.tensor([self.bos_id], dtype=torch.long)
        tgt_in = torch.cat([bos, pixels[:-1]])

        src = torch.from_numpy(self.src[i])  # (max_len,)
        src_pad_mask = src == self.pad_id

        return {
            "src": src,
            "src_pad_mask": src_pad_mask,
            "tgt_in": tgt_in,
            "tgt_out": pixels,
        }