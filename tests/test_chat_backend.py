from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from chat_backend import ChatBackendError, answer_question
from obsidian_rag.errors import RAGError


class ChatBackendTests(unittest.TestCase):
    @patch("chat_backend.Settings.from_env", return_value=object())
    @patch("chat_backend.ObsidianRAG")
    def test_returns_answer_and_limits_history(self, rag_class: MagicMock, _settings: MagicMock) -> None:
        rag = rag_class.return_value.__enter__.return_value
        rag.ask.return_value = SimpleNamespace(answer="Réponse [S1]")
        history = [(f"q{i}", f"a{i}") for i in range(8)]

        result = answer_question("Question", history)

        self.assertEqual(result, "Réponse [S1]")
        rag.ask.assert_called_once_with("Question", history=history[-6:])

    @patch("chat_backend.Settings.from_env")
    def test_exposes_only_safe_rag_message(self, settings: MagicMock) -> None:
        settings.side_effect = RAGError("test_error", "Message sûr")

        with self.assertRaisesRegex(ChatBackendError, "Message sûr"):
            answer_question("Question", [])

    @patch("chat_backend.Settings.from_env", side_effect=RuntimeError("secret"))
    def test_sanitizes_unexpected_errors(self, _settings: MagicMock) -> None:
        with self.assertRaisesRegex(ChatBackendError, "Une erreur interne est survenue"):
            answer_question("Question", [])


if __name__ == "__main__":
    unittest.main()
