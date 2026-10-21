import pytest
from pixelgen_api.errors import InvalidPromptError, UnknownModelError

from api_tests.fakes import build_service

PROMPT = "a small red circle left of a large blue square"
BOTH = ["decoder-only", "encoder-decoder"]


def test_generates_requested_variants_for_every_model():
    service, _ = build_service()
    outcome = service.generate(PROMPT, BOTH, n_variants=3, seed=1)
    assert [r.id for r in outcome.results] == BOTH
    for result in outcome.results:
        assert len(result.images) == 3
        assert len(result.images[0]) == 16 and len(result.images[0][0]) == 16
        assert result.elapsed_ms >= 0


def test_same_seed_reproduces_the_result():
    service, _ = build_service()
    a = service.generate(PROMPT, BOTH, seed=7)
    b = service.generate(PROMPT, BOTH, seed=7)
    assert [r.images for r in a.results] == [r.images for r in b.results]


def test_every_model_is_seeded_with_the_same_seed():
    service, state = build_service()
    service.generate(PROMPT, BOTH, seed=5)
    assert state["seeds_seen"] == [5, 5]


def test_random_seed_is_returned_so_the_result_can_be_reproduced():
    service, _ = build_service()
    first = service.generate(PROMPT, ["decoder-only"])
    again = service.generate(PROMPT, ["decoder-only"], seed=first.seed)
    assert first.results[0].images == again.results[0].images


def test_prompt_is_normalised():
    service, _ = build_service()
    assert service.generate("  A Small red CIRCLE. ", BOTH, seed=0).prompt == "a small red circle"


def test_duplicate_model_ids_run_once():
    service, state = build_service()
    outcome = service.generate(PROMPT, ["decoder-only", "decoder-only"], seed=0)
    assert len(outcome.results) == 1
    assert state["seeds_seen"] == [0]


def test_unknown_model_is_rejected():
    service, _ = build_service()
    with pytest.raises(UnknownModelError):
        service.generate(PROMPT, ["nope"])


def test_invalid_prompt_is_rejected_before_any_model_runs():
    service, state = build_service()
    with pytest.raises(InvalidPromptError):
        service.generate("a green hexagon", BOTH)
    assert state["seeds_seen"] == []
