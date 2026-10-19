"""Tests hors ligne du routage du chat entre Mistral et le repli OpenAI."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import MagicMock, patch

import httpx

from obsidian_rag.errors import APIError, ConfigurationError
from obsidian_rag.mistral import ChatResponse
from obsidian_rag.providers import OpenAIChatGateway, RoutedGateway
from obsidian_rag.settings import Settings


def _openai(handler) -> OpenAIChatGateway:
    """Passerelle OpenAI branchée sur un transport HTTP simulé."""

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAIChatGateway("sk-test", "gpt-5-mini", client=client)


class OpenAIChatGatewayTests(unittest.TestCase):
    """Vérifie la requête envoyée et la traduction des erreurs OpenAI."""

    def test_sends_configured_model_and_parses_answer(self) -> None:
        """Le modèle OpenAI configuré remplace le nom de modèle Mistral."""

        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            seen["auth"] = request.headers["Authorization"]
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Réponse [S1]"}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 3},
                },
            )

        response = _openai(handler).chat([{"role": "user", "content": "Q"}], "mistral-small-latest")

        self.assertEqual(seen["body"]["model"], "gpt-5-mini")
        self.assertEqual(seen["auth"], "Bearer sk-test")
        self.assertEqual(response.text, "Réponse [S1]")
        self.assertEqual((response.input_tokens, response.output_tokens), (12, 3))
        self.assertEqual(response.model, "openai/gpt-5-mini")

    def test_maps_http_errors_to_safe_api_errors(self) -> None:
        """Un 401 devient une erreur publique sans détail technique."""

        gateway = _openai(lambda _request: httpx.Response(401, json={"error": "secret"}))
        with self.assertRaises(APIError) as ctx:
            gateway.chat([{"role": "user", "content": "Q"}])
        self.assertEqual(ctx.exception.code, "authentication_error")
        self.assertNotIn("secret", ctx.exception.message)


class RoutedGatewayTests(unittest.TestCase):
    """Vérifie quand le chat bascule ou non sur le fournisseur de secours."""

    def setUp(self) -> None:
        """Prépare une passerelle principale et un repli factices."""

        self.primary = MagicMock()
        self.fallback = MagicMock()
        self.fallback.chat.return_value = ChatResponse(text="secours", model="openai/gpt-5-mini")

    def test_falls_back_on_recoverable_mistral_error(self) -> None:
        """Un modèle hors abonnement déclenche le repli OpenAI."""

        self.primary.chat.side_effect = APIError("model_not_allowed", "hors abonnement")
        gateway = RoutedGateway(self.primary, self.fallback)

        self.assertEqual(gateway.chat([], "mistral-small-latest").text, "secours")

    def test_does_not_hide_non_recoverable_errors(self) -> None:
        """Une clé Mistral invalide reste visible : les embeddings échoueraient aussi."""

        self.primary.chat.side_effect = APIError("authentication_error", "clé refusée")
        gateway = RoutedGateway(self.primary, self.fallback)

        with self.assertRaises(APIError):
            gateway.chat([], "mistral-small-latest")
        self.fallback.chat.assert_not_called()

    def test_without_fallback_the_mistral_error_propagates(self) -> None:
        """Sans clé OpenAI, l'erreur Mistral d'origine est conservée."""

        self.primary.chat.side_effect = APIError("rate_limit_error", "quota")
        with self.assertRaises(APIError):
            RoutedGateway(self.primary, None).chat([], "m")

    def test_openai_provider_skips_mistral_chat_but_not_embeddings(self) -> None:
        """CHAT_PROVIDER=openai ne change pas le fournisseur des embeddings."""

        gateway = RoutedGateway(self.primary, self.fallback, provider="openai")
        gateway.chat([], "m")
        gateway.embed(["texte"], "mistral-embed")

        self.primary.chat.assert_not_called()
        self.primary.embed.assert_called_once_with(["texte"], "mistral-embed")


class ProviderSettingsTests(unittest.TestCase):
    """Vérifie la lecture des variables liées au fournisseur de chat."""

    def _env(self, **extra: str) -> dict[str, str]:
        """Environnement minimal valide, complété par `extra`."""

        return {"MISTRAL_API_KEY": "k", "OBSIDIAN_VAULT_PATH": "/tmp", **extra}

    def test_defaults_to_light_mistral_model(self) -> None:
        """Sans configuration, le chat utilise mistral-small et pas de repli."""

        with patch.dict(os.environ, self._env(), clear=True):
            settings = Settings.from_env(env_file=os.devnull)
        self.assertEqual(settings.chat_model, "mistral-small-latest")
        self.assertEqual(settings.chat_provider, "mistral")
        self.assertEqual(settings.openai_api_key, "")

    def test_openai_provider_requires_a_key(self) -> None:
        """Forcer OpenAI sans clé est une erreur de configuration explicite."""

        with patch.dict(os.environ, self._env(CHAT_PROVIDER="openai"), clear=True):
            with self.assertRaises(ConfigurationError):
                Settings.from_env(env_file=os.devnull)


if __name__ == "__main__":
    unittest.main()
