"""Passerelle OpenAI de secours et routage du chat entre fournisseurs."""

from __future__ import annotations

from typing import Any, Sequence

import httpx

from .errors import APIError
from .mistral import ChatResponse, EmbeddingResponse, MistralGateway


OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

# Échecs Mistral pour lesquels basculer sur OpenAI a un sens : la clé est
# valide, mais le modèle, le quota ou le service ne permettent pas de répondre.
FALLBACK_CODES = frozenset(
    {
        "model_not_allowed",
        "payment_required",
        "rate_limit_error",
        "timeout_error",
        "upstream_error",
        "mistral_api_error",
    }
)


class OpenAIChatGateway:
    """Client minimal de l'API Chat Completions d'OpenAI, sans SDK dédié."""

    def __init__(self, api_key: str, model: str, *, client: httpx.Client | None = None) -> None:
        """Prépare un client HTTP authentifié ; `client` est injectable en test."""

        self.model = model
        self.client = client or httpx.Client(timeout=120)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def chat(self, messages: list[dict[str, str]], model: str | None = None) -> ChatResponse:
        """Génère une réponse avec le modèle OpenAI configuré."""

        # Le nom de modèle Mistral reçu en paramètre n'a pas de sens ici.
        payload = {"model": self.model, "messages": messages}
        try:
            response = self.client.post(OPENAI_CHAT_URL, json=payload, headers=self._headers)
        except httpx.TimeoutException as exc:
            raise APIError("timeout_error", "L'appel à OpenAI a expiré.") from exc
        except httpx.HTTPError as exc:
            raise APIError("openai_api_error", "L'appel à l'API OpenAI a échoué.") from exc

        status = response.status_code
        if status in {401, 403}:
            raise APIError("authentication_error", "Authentification OpenAI refusée (OPENAI_API_KEY).")
        if status == 429:
            raise APIError("rate_limit_error", "Limite de requêtes ou crédit OpenAI atteint.")
        if status >= 500:
            raise APIError("upstream_error", "Le service OpenAI est temporairement indisponible.")
        if status >= 400:
            raise APIError("openai_api_error", f"Requête OpenAI refusée (HTTP {status}).")

        data: dict[str, Any] = response.json()
        choices = data.get("choices") or []
        text = str(((choices[0] if choices else {}).get("message") or {}).get("content") or "").strip()
        if not text:
            raise APIError("invalid_api_response", "OpenAI a renvoyé une réponse vide.")
        usage = data.get("usage") or {}
        return ChatResponse(
            text=text,
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            model=f"openai/{self.model}",
        )

    def close(self) -> None:
        """Ferme le client HTTP."""

        self.client.close()


class RoutedGateway:
    """Embeddings toujours Mistral ; chat Mistral avec repli OpenAI éventuel."""

    def __init__(
        self,
        primary: MistralGateway | Any,
        fallback: OpenAIChatGateway | Any | None = None,
        *,
        provider: str = "mistral",
    ) -> None:
        """Associe la passerelle principale à une éventuelle passerelle de secours."""

        self.primary = primary
        self.fallback = fallback
        self.provider = provider

    def embed(self, texts: Sequence[str], model: str) -> EmbeddingResponse:
        """Les vecteurs de l'index sont Mistral : on ne change jamais de fournisseur ici."""

        return self.primary.embed(texts, model)

    def chat(self, messages: list[dict[str, str]], model: str) -> ChatResponse:
        """Interroge le fournisseur choisi, puis bascule si l'échec est récupérable."""

        if self.provider == "openai" and self.fallback is not None:
            return self.fallback.chat(messages, model)
        try:
            return self.primary.chat(messages, model)
        except APIError as exc:
            if self.fallback is None or exc.code not in FALLBACK_CODES:
                raise
            return self.fallback.chat(messages, model)

    def close(self) -> None:
        """Ferme toutes les passerelles ouvertes."""

        self.primary.close()
        if self.fallback is not None:
            self.fallback.close()


def build_gateway(settings: Any) -> RoutedGateway:
    """Construit la passerelle à partir de la configuration validée."""

    fallback = None
    if settings.openai_api_key:
        fallback = OpenAIChatGateway(settings.openai_api_key, settings.openai_chat_model)
    return RoutedGateway(MistralGateway(settings.api_key), fallback, provider=settings.chat_provider)
