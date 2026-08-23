"""Erreurs publiques et sûres à sérialiser."""

from __future__ import annotations


class RAGError(Exception):
    """Erreur attendue avec un code stable pour la CLI."""

    def __init__(self, code: str, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code


class ConfigurationError(RAGError):
    def __init__(self, message: str) -> None:
        super().__init__("configuration_error", message, exit_code=2)


class APIError(RAGError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message, exit_code=3)

