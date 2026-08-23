"""Adaptateur minimal du SDK Mistral avec politique de nouvelle tentative."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .errors import APIError


@dataclass(slots=True)
class EmbeddingResponse:
    vectors: list[list[float]]
    tokens: int = 0


@dataclass(slots=True)
class ChatResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


def _attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _status_code(exc: BaseException) -> int | None:
    for owner in (exc, getattr(exc, "response", None)):
        if owner is None:
            continue
        status = getattr(owner, "status_code", None) or getattr(owner, "status", None)
        if isinstance(status, int):
            return status
    return None


def _retryable(exc: BaseException) -> bool:
    status = _status_code(exc)
    if status is not None:
        return status in {408, 429} or 500 <= status <= 599
    return isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower()


class MistralGateway:
    """Façade injectable, afin que tous les tests restent hors ligne."""

    def __init__(
        self,
        api_key: str,
        *,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if client is None:
            try:
                # Depuis mistralai 2.9.4, l'import documenté passe par
                # `mistralai.client`. Le fallback conserve la compatibilité
                # avec les distributions 2.x qui l'exposaient à la racine.
                try:
                    from mistralai.client import Mistral
                except ImportError:
                    from mistralai import Mistral
            except ImportError as exc:  # pragma: no cover - dépend de l'installation
                raise APIError(
                    "sdk_unavailable", "Le SDK mistralai n'est pas installé."
                ) from exc
            client = Mistral(api_key=api_key)
        self.client = client
        self._sleep = sleep

    def _call(self, operation: Callable[[], Any]) -> Any:
        for attempt in range(3):
            try:
                return operation()
            except Exception as exc:  # Le SDK expose plusieurs familles d'erreurs.
                status = _status_code(exc)
                if _retryable(exc) and attempt < 2:
                    self._sleep(0.5 * (2**attempt))
                    continue
                if status in {401, 403}:
                    raise APIError("authentication_error", "Authentification Mistral refusée.") from exc
                if status == 402:
                    raise APIError("payment_required", "Le compte Mistral ne permet pas cet appel.") from exc
                if status == 429:
                    raise APIError("rate_limit_error", "Limite de requêtes Mistral atteinte.") from exc
                if status == 408:
                    raise APIError("timeout_error", "L'appel à Mistral a expiré.") from exc
                if status is not None and 500 <= status <= 599:
                    raise APIError("upstream_error", "Le service Mistral est temporairement indisponible.") from exc
                if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
                    raise APIError("timeout_error", "L'appel à Mistral a expiré.") from exc
                raise APIError("mistral_api_error", "L'appel à l'API Mistral a échoué.") from exc
        raise AssertionError("unreachable")

    def embed(self, texts: Sequence[str], model: str) -> EmbeddingResponse:
        response = self._call(
            lambda: self.client.embeddings.create(model=model, inputs=list(texts))
        )
        data = _attr(response, "data", [])
        vectors = [list(_attr(item, "embedding", [])) for item in data]
        if len(vectors) != len(texts) or any(not vector for vector in vectors):
            raise APIError("invalid_api_response", "Mistral a renvoyé des embeddings invalides.")
        usage = _attr(response, "usage")
        tokens = int(_attr(usage, "total_tokens", _attr(usage, "prompt_tokens", 0)) or 0)
        return EmbeddingResponse(vectors=vectors, tokens=tokens)

    def chat(self, messages: list[dict[str, str]], model: str) -> ChatResponse:
        response = self._call(
            lambda: self.client.chat.complete(model=model, messages=messages)
        )
        choices = _attr(response, "choices", [])
        if not choices:
            raise APIError("invalid_api_response", "Mistral n'a renvoyé aucune réponse.")
        content = _attr(_attr(choices[0], "message"), "content", "")
        if isinstance(content, list):
            content = "".join(str(_attr(part, "text", "")) for part in content)
        text = str(content or "").strip()
        if not text:
            raise APIError("invalid_api_response", "Mistral a renvoyé une réponse vide.")
        usage = _attr(response, "usage")
        return ChatResponse(
            text=text,
            input_tokens=int(_attr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(_attr(usage, "completion_tokens", 0) or 0),
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
