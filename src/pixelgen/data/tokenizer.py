"""Word-level text tokenizer for the dataset captions."""
from __future__ import annotations

from collections.abc import Iterable

PAD, UNK = "<pad>", "<unk>"


class TextTokenizer:
    """Map each word to an integer id.

    Ids 0 and 1 are reserved for <pad> and <unk>. The remaining words are sorted
    alphabetically so the mapping is deterministic.
    """

    def __init__(self, words: Iterable[str]) -> None:
        self.tokens = [PAD, UNK] + sorted(set(words))
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.id_to_token = {i: token for i, token in enumerate(self.tokens)}

    @property
    def vocab_size(self) -> int:
        """Number of distinct tokens, including <pad> and <unk>."""
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        """Id of the <pad> token."""
        return self.token_to_id[PAD]

    @property
    def unk_id(self) -> int:
        """Id of the <unk> token."""
        return self.token_to_id[UNK]

    def encode(self, caption: str, max_len: int) -> list[int]:
        """Convert a caption into token ids, right-padded with <pad> to max_len.

        Words outside the vocabulary are encoded as <unk>.

        Raises:
            ValueError: if the caption has more than max_len words.
        """
        words = caption.split()
        if len(words) > max_len:
            raise ValueError(f"Caption of {len(words)} words is above max_len={max_len}")
        ids = [self.token_to_id.get(word, self.unk_id) for word in words]
        ids += [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """Convert token ids back into text, skipping <pad> tokens."""
        return " ".join(self.id_to_token[i] for i in ids if i != self.pad_id)