"""Request and response models of the HTTP API (the contract with the frontend)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from pixelgen_api.limits import MAX_SEED, MAX_VARIANTS


class GenerateRequest(BaseModel):
    """Body of ``POST /api/generate``."""

    prompt: str = Field(min_length=1, max_length=200)
    models: list[str] = Field(min_length=1, max_length=4, description="Ids of the models to run.")
    n_variants: int = Field(default=1, ge=1, le=MAX_VARIANTS)
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)


class GeneratedImage(BaseModel):
    """One 16x16 image as palette indices (``pixels[row][col]``)."""

    pixels: list[list[int]]


class ModelResult(BaseModel):
    """Images produced by one model for one request."""

    id: str
    label: str
    images: list[GeneratedImage]
    elapsed_ms: float


class GenerateResponse(BaseModel):
    """Response of ``POST /api/generate``."""

    prompt: str = Field(description="Normalised prompt that was actually used.")
    seed: int = Field(description="Seed used; send it back to reproduce the result.")
    results: list[ModelResult]


class ModelInfo(BaseModel):
    """Public description of a served model."""

    id: str
    label: str
    kind: str
    n_params: int
    description: str


class ExampleGroup(BaseModel):
    """A titled list of example prompts."""

    title: str
    description: str
    prompts: list[str]


class VocabularyResponse(BaseModel):
    """Everything the frontend needs to help the user write a valid prompt."""

    words: list[str]
    groups: dict[str, list[str]]
    max_words: int
    palette: list[list[int]] = Field(description="RGB color of every palette index.")
    examples: list[ExampleGroup]


class ErrorBody(BaseModel):
    """Uniform error payload."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Envelope of every error response."""

    error: ErrorBody
