"""Runtime configuration, read once from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Relative paths are resolved against the working directory, which must be the
# project root: the MLflow database and the dataset paths stored in each run are
# relative to it as well.
DEFAULT_MODELS_CONFIG = Path("app/backend/models.yaml")
# The no-build web UI. To serve the Angular build instead, set
# PIXELGEN_STATIC_DIR=app/frontend/dist/pixelgen-app/browser
DEFAULT_STATIC_DIR = Path("app/web")
DEFAULT_CORS_ORIGINS = ("http://localhost:4200",)


@dataclass(frozen=True)
class Settings:
    """Application settings.

    Attributes:
        models_config: YAML file that lists the models to serve.
        static_dir: folder with the web UI (plain files or a built frontend); served at
            ``/`` if it exists.
        cors_origins: origins allowed to call the API from a browser (dev server).
    """

    models_config: Path = DEFAULT_MODELS_CONFIG
    static_dir: Path | None = DEFAULT_STATIC_DIR
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from ``PIXELGEN_*`` environment variables."""
        origins = os.environ.get("PIXELGEN_CORS_ORIGINS")
        static_dir = os.environ.get("PIXELGEN_STATIC_DIR")
        return cls(
            models_config=Path(os.environ.get("PIXELGEN_MODELS_CONFIG", DEFAULT_MODELS_CONFIG)),
            static_dir=Path(static_dir) if static_dir else DEFAULT_STATIC_DIR,
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip())
            if origins
            else DEFAULT_CORS_ORIGINS,
        )
