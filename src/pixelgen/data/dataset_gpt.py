"""Dataset for the decoder-only baseline: caption and image in a single token sequence."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from pixelgen.data.dataset import ShapesDataset


class DecoderOnlyDataset(Dataset):
    """Wrap a ``ShapesDataset`` so each caption and its image form one sequence.

    The sequence is ``[text (max_len), <sep>, pixels (256)]`` and uses a single vocabulary
    made of three non-overlapping id ranges:

        0 .. text_vocab_size - 1                 text tokens (including <pad>, <unk>)
        text_vocab_size                          <sep>
        text_vocab_size + 1 .. vocab_size - 1    palette colors (palette index + offset)

    Each item is a dict of tensors with the sequence shifted by one position:
        x: model input,  ``seq[:-1]``.                                  (max_len + 256,)
        y: next-token targets, ``seq[1:]``, with the text positions set to -100 so the
           loss ignores them and only the pixels are graded.            (max_len + 256,)

    Args:
        base: dataset that provides the tokenized caption and the pixels.
        text_vocab_size: number of text token ids, including <pad> and <unk>.
        max_len: fixed caption length in tokens (the same used to build ``base``).
    """

    IGNORE_INDEX = -100  # value ignored by torch.nn.functional.cross_entropy

    def __init__(self, base: ShapesDataset, text_vocab_size: int, max_len: int) -> None:
        n_colors = base.bos_id  # in ShapesDataset, <bos> is the first id after the palette
        self.base = base
        self.max_len = max_len
        self.sep_id = text_vocab_size
        self.offset = text_vocab_size + 1  # palette index 0 maps to this id
        self.vocab_size = text_vocab_size + 1 + n_colors

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        item = self.base[i]

        # Caption (padded to max_len), separator, then pixels shifted into their own id range.
        sep = torch.tensor([self.sep_id], dtype=torch.long)
        seq = torch.cat([item["src"], sep, item["tgt_out"] + self.offset])

        # Next-token prediction: the target is the input shifted one position to the left.
        x = seq[:-1].clone()
        y = seq[1:].clone()  # clone: slicing returns a view, and the next line writes to y

        # Do not grade the model on predicting the caption (it is given as context).
        y[: self.max_len] = self.IGNORE_INDEX
        return {"x": x, "y": y}