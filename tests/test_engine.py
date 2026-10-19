"""Tests du moteur RAG (exclusions, recherche hybride, mémoire) avec une passerelle factice."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Sequence

from obsidian_rag.mistral import ChatResponse, EmbeddingResponse
from obsidian_rag.rag import ObsidianRAG
from obsidian_rag.settings import Settings
from obsidian_rag.vault import iter_notes

# Chaque mot-clé correspond à une dimension du vecteur factice.
KEYWORDS = ("jardin", "tomate", "python", "code", "voyage", "japon")


class FakeGateway:
    """Passerelle déterministe : embeddings par mots-clés, chat à texte fixe."""

    def __init__(self, reply: str = "Réponse [S1].") -> None:
        """Initialise la réponse de chat renvoyée à chaque appel."""

        self.reply = reply
        self.chat_calls: list[list[dict[str, str]]] = []

    def embed(self, texts: Sequence[str], model: str) -> EmbeddingResponse:
        """Compte les mots-clés de chaque texte ; une dimension de biais évite le vecteur nul."""

        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([float(lowered.count(word)) for word in KEYWORDS] + [0.1])
        return EmbeddingResponse(vectors=vectors, tokens=len(texts))

    def chat(self, messages: list[dict[str, str]], model: str) -> ChatResponse:
        """Mémorise les messages reçus et renvoie le texte fixe."""

        self.chat_calls.append(messages)
        return ChatResponse(text=self.reply)

    def close(self) -> None:
        """Rien à libérer."""


class EngineTest(unittest.TestCase):
    """Scénarios de bout en bout sur un coffre temporaire."""

    def setUp(self) -> None:
        """Crée un petit coffre et un index temporaires."""

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.vault = root / "vault"
        files = {
            "Jardin/Tomates.md": "# Tomates\n\nLe jardin produit des tomates. Tomate et jardin.",
            "Jardin/Potager.md": "# Potager\n\nOrganisation du jardin et des tomates.",
            "Dev/Python.md": "# Python\n\nDu code python, encore du code.",
            "Voyage/Japon.md": "# Japon\n\nVoyage au Japon. Mot rare : zygomorphe.",
            "Accueil.md": "# Accueil\n\nPage d'accueil du jardin.",
            "0 Second Cerveau/Conversations/2026-01-01.md": "# Conv\n\nConversation jardin tomate.",
            "Recently Deleted/Vieux.md": "# Vieux\n\nAncienne note python.",
        }
        for rel, content in files.items():
            path = self.vault / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        self.gateway = FakeGateway()
        self.settings = Settings(
            api_key="test",
            vault_path=self.vault,
            index_path=root / "index.sqlite3",
            top_k=4,
        )
        self.rag = ObsidianRAG(self.settings, gateway=self.gateway)

    def tearDown(self) -> None:
        """Ferme l'index et supprime les fichiers temporaires."""

        self.rag.close()
        self.tmp.cleanup()

    def test_exclusion_des_dossiers(self) -> None:
        """Les dossiers exclus ne sont ni parcourus ni indexés."""

        paths = {note.path for note in iter_notes(self.vault, self.settings.exclude_dirs)}
        self.assertNotIn("Recently Deleted/Vieux.md", paths)
        self.assertNotIn("0 Second Cerveau/Conversations/2026-01-01.md", paths)
        self.assertIn("Jardin/Tomates.md", paths)
        self.rag.refresh_index()
        indexed = {path for path, _ in self.rag.list_notes()}
        self.assertEqual(indexed, paths)

    def test_notes_exclues_supprimees_de_l_index(self) -> None:
        """Une note déjà indexée puis exclue disparaît au refresh suivant."""

        self.rag.index.refresh(self.vault)
        self.assertIn("Recently Deleted/Vieux.md", {p for p, _ in self.rag.list_notes()})
        update = self.rag.refresh_index()
        self.assertEqual(update.deleted, 2)
        self.assertNotIn("Recently Deleted/Vieux.md", {p for p, _ in self.rag.list_notes()})

    def test_recherche_hybride_mot_rare(self) -> None:
        """Un mot absent des dimensions d'embedding est retrouvé grâce à FTS5."""

        self.assertTrue(self.rag.index.fts_enabled)
        sources = self.rag.search("zygomorphe", top_k=1)
        self.assertEqual(sources[0].path, "Voyage/Japon.md")
        self.assertEqual(sources[0].id, "S1")

    def test_requete_fts_avec_syntaxe_speciale(self) -> None:
        """Les caractères spéciaux ne provoquent pas d'erreur FTS."""

        sources = self.rag.search('jardin "AND (OR NEAR* tomate')
        self.assertTrue(sources)

    def test_filtre_path_prefixes(self) -> None:
        """Le filtre par dossier restreint les résultats."""

        sources = self.rag.search("jardin tomate", path_prefixes=["Dev"])
        self.assertTrue(sources)
        self.assertTrue(all(source.path.startswith("Dev/") for source in sources))

    def test_citation_warning_non_bloquant(self) -> None:
        """Une citation inventée produit un avertissement et non une exception."""

        self.gateway.reply = "Les tomates poussent [S99]."
        result = self.rag.ask("Que dit le jardin sur les tomates ?", memory="Aime les tomates.")
        self.assertIn("citation_warning", result.diagnostics)
        self.assertEqual(result.diagnostics["sources_count"], len(result.sources))
        user_message = self.gateway.chat_calls[-1][1]["content"]
        self.assertIn("MÉMOIRE PERSISTANTE", user_message)

        self.gateway.reply = "Aucune citation ici."
        result = self.rag.ask("tomates ?", history=[("q", "r")])
        self.assertIn("citation_warning", result.diagnostics)

        self.gateway.reply = "Les tomates [S1]."
        result = self.rag.ask("tomates ?")
        self.assertNotIn("citation_warning", result.diagnostics)

    def test_similar_notes(self) -> None:
        """La note la plus proche de Tomates est Potager ; note inconnue → []."""

        self.rag.refresh_index()
        related = self.rag.related("Jardin/Tomates.md", k=2)
        self.assertEqual(related[0].path, "Jardin/Potager.md")
        self.assertEqual(related[0].id, "R1")
        self.assertNotIn("Jardin/Tomates.md", {source.path for source in related})
        self.assertEqual(self.rag.related("Inconnue.md"), [])

    def test_stats(self) -> None:
        """Les statistiques comptent notes, chunks et dossiers de premier niveau."""

        self.rag.refresh_index()
        stats = self.rag.stats()
        self.assertEqual(stats["notes"], 5)
        self.assertGreaterEqual(stats["chunks"], 5)
        self.assertEqual(
            stats["folders"], {"(racine)": 1, "Dev": 1, "Jardin": 2, "Voyage": 1}
        )

    def test_fts_rempli_pour_index_ancien(self) -> None:
        """Un index sans FTS est complété à la réouverture, sans ré-embedding."""

        self.rag.refresh_index()
        self.rag.index.connection.execute("DELETE FROM chunks_fts")
        self.rag.index.connection.commit()
        self.rag.close()
        self.rag = ObsidianRAG(self.settings, gateway=self.gateway)
        update = self.rag.refresh_index()
        self.assertEqual(update.embedded_chunks, 0)
        self.assertEqual(self.rag.search("zygomorphe", top_k=1)[0].path, "Voyage/Japon.md")

    def test_distill_memory(self) -> None:
        """« AUCUN » donne une liste vide ; les puces sont extraites (max 3)."""

        self.gateway.reply = "AUCUN"
        self.assertEqual(self.rag.distill_memory("q", "r"), [])
        self.gateway.reply = "- Préfère le café\n- Vit à Lyon\nbavardage\n- Fait du vélo\n- En trop"
        self.assertEqual(
            self.rag.distill_memory("q", "r"),
            ["Préfère le café", "Vit à Lyon", "Fait du vélo"],
        )

    def test_settings_exclude_env(self) -> None:
        """RAG_EXCLUDE_DIRS : absente → défaut, vide → tuple vide, liste → trimée."""

        import os
        from unittest import mock

        env = {"MISTRAL_API_KEY": "k", "OBSIDIAN_VAULT_PATH": str(self.vault)}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                Settings.from_env(env_file=os.devnull).exclude_dirs,
                ("0 Second Cerveau/Conversations", "Recently Deleted"),
            )
        with mock.patch.dict(os.environ, {**env, "RAG_EXCLUDE_DIRS": ""}, clear=True):
            self.assertEqual(Settings.from_env(env_file=os.devnull).exclude_dirs, ())
        with mock.patch.dict(os.environ, {**env, "RAG_EXCLUDE_DIRS": " A , B/C "}, clear=True):
            self.assertEqual(Settings.from_env(env_file=os.devnull).exclude_dirs, ("A", "B/C"))


if __name__ == "__main__":
    unittest.main()
