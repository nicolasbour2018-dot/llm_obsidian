"""Tests de brain_vault (dossiers temporaires uniquement)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

from brain_vault import BrainVault, sanitize_filename


class BrainVaultTests(unittest.TestCase):
    """Mémoire LLM et archivage des conversations."""

    def setUp(self) -> None:
        """Crée un faux coffre temporaire contenant « 0 Second Cerveau »."""

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "0 Second Cerveau"
        self.root.mkdir()
        self.vault = BrainVault(self.root)

    def tearDown(self) -> None:
        """Supprime le coffre temporaire."""

        self._tmp.cleanup()

    def test_read_memory_absent(self) -> None:
        self.assertEqual(self.vault.read_memory(), "")

    def test_append_memory_creates_file_and_section(self) -> None:
        added = self.vault.append_memory(["Aime le café", "", "  "], link="0 Second Cerveau/Conversations/x")
        self.assertEqual(added, 1)
        text = self.vault.memory_path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("---\ntype: llm-memory"))
        self.assertIn("## Souvenirs", text)
        self.assertIn(f"- {date.today().isoformat()} — Aime le café ([[0 Second Cerveau/Conversations/x]])", text)

    def test_append_memory_adds_missing_section_and_dedupes(self) -> None:
        self.vault.memory_path.write_text("---\ntype: llm-memory\n---\n# Mémoire LLM\n\nIntro.\n", encoding="utf-8")
        self.assertEqual(self.vault.append_memory(["Fait A", "fait a", "Fait B"]), 2)
        self.assertEqual(self.vault.append_memory(["FAIT A", "Fait C"], link="lien"), 1)
        body = self.vault.read_memory()
        self.assertIn("Intro.", body)
        lines = [line for line in body.splitlines() if line.startswith("- ")]
        self.assertEqual(len(lines), 3)
        self.assertEqual(body.count("## Souvenirs"), 1)

    def test_append_memory_inserts_inside_section(self) -> None:
        self.vault.memory_path.write_text(
            "---\ntype: llm-memory\n---\n# M\n\n## Souvenirs\n- 2026-01-01 — Ancien\n\n## Autre\ntexte\n",
            encoding="utf-8",
        )
        self.assertEqual(self.vault.append_memory(["ancien", "Nouveau"]), 1)
        body = self.vault.read_memory()
        self.assertLess(body.index("Nouveau"), body.index("## Autre"))
        self.assertGreater(body.index("Nouveau"), body.index("Ancien"))

    def test_write_memory_preserves_frontmatter(self) -> None:
        front = "---\ntype: llm-memory\ntags: [a]\n---\n"
        self.vault.memory_path.write_text(front + "# Ancien\n", encoding="utf-8")
        self.vault.write_memory("# Nouveau\n\nContenu")
        text = self.vault.memory_path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith(front))
        self.assertEqual(self.vault.read_memory(), "# Nouveau\n\nContenu")

    def test_write_memory_creates_frontmatter(self) -> None:
        self.vault.write_memory("Corps")
        self.assertTrue(self.vault.memory_path.read_text(encoding="utf-8").startswith("---\ntype: llm-memory"))
        self.assertEqual(self.vault.read_memory(), "Corps")

    def test_save_conversation_rewrites_same_file(self) -> None:
        started = datetime(2026, 10, 18, 20, 42)
        messages = [
            {"role": "user", "content": "Bonjour ?"},
            {
                "role": "assistant",
                "content": "Salut.",
                "sources": [{"path": "Dossier/Note.md", "title": "Note"}, {"path": "Dossier/Note.md", "title": "Note"}],
            },
        ]
        link = self.vault.save_conversation("abc-123", "Mon/titre: test?", messages, started_at=started)
        self.assertEqual(link, "0 Second Cerveau/Conversations/2026-10-18 2042 - Mon titre test")
        messages.append({"role": "user", "content": "Encore ?"})
        link2 = self.vault.save_conversation(
            "abc-123", "Titre changé", messages, started_at=datetime(2026, 10, 19, 9, 0)
        )
        self.assertEqual(link, link2)
        files = list((self.root / "Conversations").glob("*.md"))
        self.assertEqual(len(files), 1)
        text = files[0].read_text(encoding="utf-8")
        self.assertIn('conversation-id: "abc-123"', text)
        self.assertIn("type: conversation-ia", text)
        self.assertIn("tags: [conversation-ia]", text)
        self.assertEqual(text.count("## 🙋 Question"), 2)
        self.assertIn("## 🤖 Réponse", text)
        self.assertIn("Sources : [[Dossier/Note|Note]]\n", text)
        # Un autre identifiant crée un autre fichier.
        self.vault.save_conversation("autre", "Mon titre test", messages, started_at=started)
        self.assertEqual(len(list((self.root / "Conversations").glob("*.md"))), 2)
        # Aucun fichier temporaire ne doit traîner.
        self.assertEqual(list((self.root / "Conversations").glob(".tmp-*")), [])

    def test_sanitize_filename(self) -> None:
        self.assertEqual(sanitize_filename('  a/b\\c:d*e?"f<g>h|i#j^k[l]m  '), "a b c d e f g h i j k l m")
        self.assertEqual(len(sanitize_filename("x" * 200)), 60)
        self.assertEqual(sanitize_filename("///"), "Sans titre")


if __name__ == "__main__":
    unittest.main()
