"""Domain errors. Each ``AppError`` maps to one HTTP status and one stable error code."""
from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for errors that are reported to the client.

    Attributes:
        status_code: HTTP status of the response.
        code: stable machine-readable identifier (the frontend switches on it).
        message: human-readable explanation.
        details: optional extra data (for example the unknown words of a prompt).
    """

    status_code = 400
    code = "error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidPromptError(AppError):
    """The prompt cannot be turned into a valid model input."""

    status_code = 422

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details=details)
        self.code = code


class UnknownModelError(AppError):
    """The request names a model that is not served."""

    status_code = 422
    code = "unknown_model"


class ConfigError(RuntimeError):
    """The server configuration is inconsistent (raised at startup, not over HTTP)."""
