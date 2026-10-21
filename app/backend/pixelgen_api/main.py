"""Application factory.

Run from the project root::

    python -m uvicorn --app-dir app/backend pixelgen_api.main:app --port 8000
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from pixelgen_api.errors import AppError
from pixelgen_api.routes import router
from pixelgen_api.services.generation import GenerationService
from pixelgen_api.settings import Settings

logger = logging.getLogger("pixelgen_api")


def _error(status: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    """Build the uniform error envelope ``{"error": {code, message, details}}``."""
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "details": details or {}}},
    )


def _build_service(settings: Settings) -> GenerationService:
    """Load the models listed in the settings (slow: imports torch and reads checkpoints)."""
    from pixelgen.training.loop import get_device
    from pixelgen_api.services.registry import load_catalog

    device = get_device()
    logger.info("loading models on %s from %s", device, settings.models_config)
    registry, dataset = load_catalog(settings.models_config, device)
    return GenerationService(registry, dataset)


def create_app(settings: Settings | None = None, service: GenerationService | None = None) -> FastAPI:
    """Create the FastAPI application.

    Args:
        settings: configuration; read from the environment when omitted.
        service: ready-made service, used by tests to avoid loading real models.
            When omitted, the models are loaded at startup.
    """
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if getattr(app.state, "service", None) is None:
            app.state.service = _build_service(settings)
        yield

    app = FastAPI(title="pixelgen API", version="1.0.0", lifespan=lifespan)
    app.state.service = service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return _error(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        problems = [
            {"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]}
            for e in exc.errors()
        ]
        return _error(422, "invalid_request", "The request is not valid.", {"problems": problems})

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error")
        return _error(500, "internal_error", "Something went wrong while generating the image.")

    app.include_router(router)

    # Serve the built frontend last, so it never shadows the API routes.
    if settings.static_dir is not None and settings.static_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="frontend")
    return app


app = create_app()
