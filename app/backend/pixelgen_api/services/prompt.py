"""Prompt normalisation and validation against the model vocabulary."""
from __future__ import annotations

import difflib
import re
from collections.abc import Sequence

from pixelgen_api.errors import InvalidPromptError

_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)


def tokenize_prompt(text: str) -> list[str]:
    """Lower-case ``text``, drop punctuation and split it into words."""
    return _PUNCTUATION.sub(" ", text.lower()).split()


class PromptValidator:
    """Check that a prompt only uses words the model knows and fits its input length.

    The tokenizer maps unknown words to ``<unk>`` without complaint, which would
    silently produce a meaningless image; rejecting them here gives the user an
    actionable error instead.

    Args:
        words: vocabulary of the text tokenizer.
        max_words: maximum number of words the model accepts.
    """

    def __init__(self, words: Sequence[str], max_words: int) -> None:
        self._words = list(words)
        self._known = set(self._words)
        self._max_words = max_words

    def validate(self, text: str) -> str:
        """Return the canonical prompt (lower-case, single spaces).

        Raises:
            InvalidPromptError: if the prompt is empty, too long or has unknown words.
        """
        tokens = tokenize_prompt(text)
        if not tokens:
            raise InvalidPromptError("empty_prompt", "Write a description to generate an image.")
        if len(tokens) > self._max_words:
            raise InvalidPromptError(
                "prompt_too_long",
                f"The prompt has {len(tokens)} words; the maximum is {self._max_words}.",
                details={"max_words": self._max_words, "n_words": len(tokens)},
            )
        unknown = list(dict.fromkeys(t for t in tokens if t not in self._known))
        if unknown:
            suggestions = {
                word: difflib.get_close_matches(word, self._words, n=3, cutoff=0.6)
                for word in unknown
            }
            raise InvalidPromptError(
                "unknown_words",
                "These words are not in the model vocabulary: " + ", ".join(unknown) + ".",
                details={"unknown": unknown, "suggestions": suggestions},
            )
        return " ".join(tokens)
