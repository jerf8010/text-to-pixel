"""HTTP routes. Thin layer: parse the request, call the service, shape the response."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from pixelgen_api.schemas import (
    ExampleGroup,
    GeneratedImage,
    GenerateRequest,
    GenerateResponse,
    ModelInfo,
    ModelResult,
    VocabularyResponse,
)
from pixelgen_api.services.generation import GenerationService

router = APIRouter(prefix="/api")


def get_service(request: Request) -> GenerationService:
    """Dependency that returns the service created at startup."""
    return request.app.state.service


ServiceDep = Annotated[GenerationService, Depends(get_service)]


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@router.get("/models", response_model=list[ModelInfo])
def list_models(service: ServiceDep) -> list[ModelInfo]:
    """List the models that can be used for generation."""
    return [
        ModelInfo(
            id=e.id, label=e.label, kind=e.kind, n_params=e.n_params, description=e.description
        )
        for e in service.registry.all()
    ]


@router.get("/vocabulary", response_model=VocabularyResponse)
def get_vocabulary(service: ServiceDep) -> VocabularyResponse:
    """Vocabulary, palette and example prompts, so the UI can guide the user."""
    d = service.dataset
    return VocabularyResponse(
        words=d.words,
        groups=d.groups,
        max_words=d.max_words,
        palette=d.palette,
        examples=[
            ExampleGroup(title=e.title, description=e.description, prompts=e.prompts)
            for e in d.examples
        ],
    )


@router.post("/generate", response_model=GenerateResponse)
def generate(body: GenerateRequest, service: ServiceDep) -> GenerateResponse:
    """Generate images for a prompt with the selected models.

    Declared with ``def`` (not ``async def``) so FastAPI runs the blocking model
    inference in a worker thread instead of freezing the event loop.
    """
    outcome = service.generate(
        body.prompt,
        body.models,
        n_variants=body.n_variants,
        temperature=body.temperature,
        seed=body.seed,
    )
    return GenerateResponse(
        prompt=outcome.prompt,
        seed=outcome.seed,
        results=[
            ModelResult(
                id=r.id,
                label=r.label,
                images=[GeneratedImage(pixels=img) for img in r.images],
                elapsed_ms=r.elapsed_ms,
            )
            for r in outcome.results
        ],
    )
