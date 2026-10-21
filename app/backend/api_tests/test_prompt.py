import pytest

from pixelgen_api.errors import InvalidPromptError
from pixelgen_api.services.prompt import PromptValidator, tokenize_prompt

WORDS = ["a", "small", "red", "circle", "square", "left", "of"]


def validator(max_words: int = 6) -> PromptValidator:
    return PromptValidator(WORDS, max_words)


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize_prompt("  A Small, RED circle! ") == ["a", "small", "red", "circle"]


def test_validate_returns_canonical_prompt():
    assert validator().validate("A  small   red circle.") == "a small red circle"


def test_empty_prompt_is_rejected():
    with pytest.raises(InvalidPromptError) as err:
        validator().validate("  ... ")
    assert err.value.code == "empty_prompt"


def test_too_long_prompt_is_rejected():
    with pytest.raises(InvalidPromptError) as err:
        validator(max_words=3).validate("a small red circle")
    assert err.value.code == "prompt_too_long"
    assert err.value.details == {"max_words": 3, "n_words": 4}


def test_unknown_words_are_reported_with_suggestions():
    with pytest.raises(InvalidPromptError) as err:
        validator().validate("a small purple circel")
    assert err.value.code == "unknown_words"
    assert err.value.details["unknown"] == ["purple", "circel"]
    assert "circle" in err.value.details["suggestions"]["circel"]


def test_unknown_words_are_listed_once():
    with pytest.raises(InvalidPromptError) as err:
        validator().validate("zzz zzz red")
    assert err.value.details["unknown"] == ["zzz"]
