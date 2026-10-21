import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from pixelgen_api.main import create_app  # noqa: E402
from pixelgen_api.settings import Settings  # noqa: E402
from api_tests.fakes import build_service  # noqa: E402

PROMPT = "a small red circle left of a large blue square"


@pytest.fixture()
def client():
    service, _ = build_service()
    settings = Settings(static_dir=None)
    return TestClient(create_app(settings, service))


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_models(client):
    body = client.get("/api/models").json()
    assert [m["id"] for m in body] == ["decoder-only", "encoder-decoder"]
    assert body[0]["kind"] == "decoder_only"


def test_vocabulary(client):
    body = client.get("/api/vocabulary").json()
    assert "circle" in body["words"]
    assert body["max_words"] == 10
    assert len(body["palette"]) == 9
    assert body["examples"][0]["prompts"]


def test_generate(client):
    response = client.post(
        "/api/generate",
        json={"prompt": PROMPT, "models": ["decoder-only", "encoder-decoder"], "n_variants": 2, "seed": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["seed"] == 3 and body["prompt"] == PROMPT
    assert [r["id"] for r in body["results"]] == ["decoder-only", "encoder-decoder"]
    pixels = body["results"][0]["images"][0]["pixels"]
    assert len(pixels) == 16 and len(pixels[0]) == 16


def test_unknown_words_return_a_structured_error(client):
    response = client.post("/api/generate", json={"prompt": "a green hexagon", "models": ["decoder-only"]})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "unknown_words"
    assert error["details"]["unknown"] == ["green", "hexagon"]


def test_unknown_model(client):
    response = client.post("/api/generate", json={"prompt": PROMPT, "models": ["nope"]})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unknown_model"


def test_invalid_body_uses_the_same_error_envelope(client):
    response = client.post("/api/generate", json={"prompt": PROMPT, "models": ["decoder-only"], "n_variants": 99})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["details"]["problems"][0]["field"] == "n_variants"
