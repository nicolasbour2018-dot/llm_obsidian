from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import chat_backend
from chat_backend import ChatBackendError, ChatResult, answer_question, ask
from obsidian_rag.errors import RAGError


def _source(**overrides: object) -> SimpleNamespace:
    """Fabrique une source minimale conforme au contrat du moteur."""

    values = {
        "id": "S1",
        "path": "3 Ressource/Note.md",
        "title": "Note",
        "heading": "Intro",
        "score": 0.8765,
        "excerpt": "Extrait",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class ChatBackendTests(unittest.TestCase):
    """Vérifie la frontière de sécurité entre Streamlit et le moteur RAG."""

    @patch("chat_backend.Settings.from_env", return_value=object())
    @patch("chat_backend.ObsidianRAG")
    def test_returns_answer_and_limits_history(self, rag_class: MagicMock, _settings: MagicMock) -> None:
        """Le backend renvoie le texte et transmet au plus six échanges."""

        rag = rag_class.return_value.__enter__.return_value
        rag.ask.return_value = SimpleNamespace(answer="Réponse [S1]", sources=[], diagnostics={})
        history = [(f"q{i}", f"a{i}") for i in range(8)]

        result = answer_question("Question", history)

        self.assertEqual(result, "Réponse [S1]")
        rag.ask.assert_called_once_with(
            "Question", history=history[-6:], memory="", path_prefixes=None, top_k=None
        )

    @patch("chat_backend.Settings.from_env", return_value=object())
    @patch("chat_backend.ObsidianRAG")
    def test_ask_returns_sources_and_warning(self, rag_class: MagicMock, _settings: MagicMock) -> None:
        """`ask` sérialise les sources et remonte l'avertissement de citation."""

        rag = rag_class.return_value.__enter__.return_value
        rag.ask.return_value = SimpleNamespace(
            answer="Réponse",
            sources=[_source()],
            diagnostics={"citation_warning": "Citation manquante", "usage": {}},
        )

        result = ask(
            "Question",
            [],
            path_prefixes=("1 Projet",),
            top_k=4,
            memory="Je préfère le français.",
        )

        self.assertIsInstance(result, ChatResult)
        self.assertEqual(result.answer, "Réponse")
        self.assertEqual(result.warning, "Citation manquante")
        self.assertEqual(result.sources[0]["path"], "3 Ressource/Note.md")
        self.assertEqual(result.sources[0]["title"], "Note")
        self.assertAlmostEqual(result.sources[0]["score"], 0.8765)
        self.assertIn("usage", result.diagnostics)
        rag.ask.assert_called_once_with(
            "Question",
            history=[],
            memory="Je préfère le français.",
            path_prefixes=["1 Projet"],
            top_k=4,
        )

    @patch("chat_backend.Settings.from_env", return_value=object())
    @patch("chat_backend.ObsidianRAG")
    def test_ask_without_warning(self, rag_class: MagicMock, _settings: MagicMock) -> None:
        """Sans avertissement du moteur, `warning` reste vide."""

        rag = rag_class.return_value.__enter__.return_value
        rag.ask.return_value = SimpleNamespace(answer="Ok", sources=[], diagnostics={})

        self.assertIsNone(ask("Question", []).warning)

    @patch("chat_backend.Settings.from_env", return_value=object())
    @patch("chat_backend.ObsidianRAG")
    def test_helpers_delegate_to_engine(self, rag_class: MagicMock, _settings: MagicMock) -> None:
        """Les fonctions annexes délèguent au moteur et renvoient des types simples."""

        rag = rag_class.return_value.__enter__.return_value
        rag.search.return_value = [_source()]
        rag.related.return_value = [_source(id="S2", title="Voisine")]
        rag.stats.return_value = {"notes": 2, "chunks": 5, "folders": {"1 Projet": 2}}
        rag.list_notes.return_value = [("1 Projet/A.md", "A")]
        rag.refresh_index.return_value = SimpleNamespace(added=1, updated=2, deleted=3, embedded_chunks=4)
        rag.distill_memory.return_value = ["Fait durable"]

        self.assertEqual(chat_backend.search("q", top_k=3)[0]["title"], "Note")
        rag.search.assert_called_once_with("q", path_prefixes=None, top_k=3)
        self.assertEqual(chat_backend.related("1 Projet/A.md", k=2)[0]["title"], "Voisine")
        rag.related.assert_called_once_with("1 Projet/A.md", k=2)
        self.assertEqual(chat_backend.stats()["chunks"], 5)
        self.assertEqual(chat_backend.list_notes(), [("1 Projet/A.md", "A")])
        self.assertEqual(
            chat_backend.reindex(force=True),
            {"added": 1, "updated": 2, "deleted": 3, "embedded_chunks": 4},
        )
        rag.refresh_index.assert_called_once_with(force=True)
        self.assertEqual(chat_backend.distill_memory("q", "a"), ["Fait durable"])

    @patch("chat_backend.Settings.from_env")
    def test_exposes_only_safe_rag_message(self, settings: MagicMock) -> None:
        """Une erreur métier conserve uniquement son message public."""

        settings.side_effect = RAGError("test_error", "Message sûr")

        with self.assertRaisesRegex(ChatBackendError, "Message sûr"):
            answer_question("Question", [])

    @patch("chat_backend.Settings.from_env", return_value=object())
    @patch("chat_backend.ObsidianRAG")
    def test_sanitizes_errors_raised_during_call(self, rag_class: MagicMock, _settings: MagicMock) -> None:
        """Une erreur levée pendant l'appel au moteur est elle aussi assainie."""

        rag = rag_class.return_value.__enter__.return_value
        rag.search.side_effect = RuntimeError("clé secrète")

        with self.assertRaisesRegex(ChatBackendError, "Une erreur interne est survenue") as ctx:
            chat_backend.search("q")
        self.assertNotIn("secrète", str(ctx.exception))

    @patch("chat_backend.Settings.from_env", side_effect=RuntimeError("secret"))
    def test_sanitizes_unexpected_errors(self, _settings: MagicMock) -> None:
        """Une exception inconnue ne divulgue pas son détail technique."""

        with self.assertRaisesRegex(ChatBackendError, "Une erreur interne est survenue"):
            answer_question("Question", [])


if __name__ == "__main__":
    unittest.main()
