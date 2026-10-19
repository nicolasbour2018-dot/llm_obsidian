"""Index SQLite transactionnel, recherche cosinus en mémoire et plein texte FTS5."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .errors import ConfigurationError, RAGError
from .mistral import EmbeddingResponse
from .models import IndexUpdate
from .vault import CHUNK_FORMAT_VERSION, Chunk, Note, chunk_note, iter_notes


@dataclass(frozen=True, slots=True)
class SearchHit:
    """Extrait retrouvé dans l'index, accompagné de son score cosinus."""

    path: str
    title: str
    heading: str
    text: str
    score: float


Embedder = Callable[[Sequence[str], str], EmbeddingResponse]

# Constante classique de la fusion Reciprocal Rank Fusion.
RRF_K = 60
# Nombre maximal de résultats plein texte pris en compte dans la fusion.
FTS_LIMIT = 200


def _fts_query(text: str) -> str:
    """Construit une requête FTS5 sûre : mots ≥ 3 caractères, entre guillemets, joints par OR."""

    words: list[str] = []
    for word in re.findall(r"\w+", text):
        if len(word) >= 3 and word.casefold() not in {w.casefold() for w in words}:
            words.append(word)
    # Les guillemets internes sont doublés pour neutraliser la syntaxe FTS.
    return " OR ".join('"' + word.replace('"', '""') + '"' for word in words[:32])


def _matches_prefixes(path: str, prefixes: Sequence[str] | None) -> bool:
    """Vrai si aucun filtre n'est demandé ou si le chemin est sous l'un des préfixes."""

    if not prefixes:
        return True
    for prefix in prefixes:
        prefix = prefix.strip().strip("/")
        if not prefix or path == prefix or path.startswith(prefix + "/"):
            return True
    return False


class SQLiteIndex:
    """Stocke les notes et embeddings, puis effectue la recherche vectorielle."""

    def __init__(self, path: Path, embed_model: str, embedder: Embedder) -> None:
        """Ouvre l'index et injecte la fonction d'embedding à utiliser."""

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(path)
        except (OSError, sqlite3.Error) as exc:
            raise ConfigurationError(f"Impossible d'ouvrir l'index SQLite : {path}") from exc
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.embed_model = embed_model
        self.embedder = embedder
        self.last_embedding_tokens = 0
        self.fts_enabled = False
        self._create_schema()
        self._create_fts()

    def _create_schema(self) -> None:
        """Crée le schéma idempotent et active la suppression en cascade."""

        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                path TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                title TEXT NOT NULL,
                aliases_json TEXT NOT NULL,
                tags_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_path TEXT NOT NULL REFERENCES documents(path) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                heading TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                text TEXT NOT NULL,
                vector BLOB NOT NULL,
                dimensions INTEGER NOT NULL,
                UNIQUE(document_path, position)
            );
            CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks(document_path);
            """
        )
        self.connection.commit()

    def _create_fts(self) -> None:
        """Crée la table FTS5 si possible et la resynchronise depuis `chunks` si besoin."""

        try:
            self.connection.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                   USING fts5(text, title, tokenize = 'unicode61 remove_diacritics 2')"""
            )
            self.connection.commit()
        except sqlite3.OperationalError:
            # SQLite compilé sans FTS5 : on se contente de la recherche sémantique.
            self.fts_enabled = False
            return
        self.fts_enabled = True
        try:
            fts_count = self.connection.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
            chunk_count = self.connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
            # Index ancien (ou désynchronisé) : on remplit FTS sans aucun ré-embedding.
            if fts_count != chunk_count:
                with self.connection:
                    self.connection.execute("DELETE FROM chunks_fts")
                    self.connection.execute(
                        """INSERT INTO chunks_fts(rowid, text, title)
                           SELECT c.id, c.text, d.title
                           FROM chunks c JOIN documents d ON d.path = c.document_path"""
                    )
        except sqlite3.OperationalError:
            self.fts_enabled = False

    def _metadata(self) -> dict[str, str]:
        """Retourne les versions techniques associées au contenu de l'index."""

        return {row["key"]: row["value"] for row in self.connection.execute("SELECT key, value FROM metadata")}

    def refresh(
        self, vault_path: Path, *, force: bool = False, exclude: Sequence[str] = ()
    ) -> IndexUpdate:
        """Synchronise incrémentalement l'index avec le coffre, hors dossiers exclus."""

        self.last_embedding_tokens = 0
        try:
            notes = {note.path: note for note in iter_notes(vault_path, exclude)}
        except OSError as exc:
            raise RAGError("vault_read_error", "Impossible de lire le coffre Obsidian.") from exc
        existing = {
            row["path"]: row["fingerprint"]
            for row in self.connection.execute("SELECT path, fingerprint FROM documents")
        }
        metadata = self._metadata()
        # Un changement de modèle ou de format invalide tous les vecteurs existants.
        incompatible = (
            bool(existing)
            and (
                metadata.get("embedding_model") != self.embed_model
                or metadata.get("chunk_format") != CHUNK_FORMAT_VERSION
            )
        )
        rebuild = force or incompatible
        if rebuild:
            added_paths = set(notes)
            updated_paths: set[str] = set()
            deleted_paths = set(existing) - set(notes)
        else:
            added_paths = set(notes) - set(existing)
            updated_paths = {
                path for path in set(notes) & set(existing) if notes[path].fingerprint != existing[path]
            }
            deleted_paths = set(existing) - set(notes)

        changed_paths = sorted(added_paths | updated_paths)
        prepared: dict[str, tuple[Note, list[Chunk], list[np.ndarray]]] = {}

        # Tous les appels réseau précèdent la transaction : l'ancien index reste valide
        # si un lot échoue.
        pending: list[tuple[str, int, str]] = []
        chunks_by_path: dict[str, list[Chunk]] = {}
        for path in changed_paths:
            chunks = chunk_note(notes[path])
            chunks_by_path[path] = chunks
            pending.extend((path, i, chunk.text) for i, chunk in enumerate(chunks))

        # Des lots bornés évitent de dépasser la taille maximale d'une requête API.
        raw_vectors: dict[tuple[str, int], np.ndarray] = {}
        for offset in range(0, len(pending), 32):
            batch = pending[offset : offset + 32]
            response = self.embedder([item[2] for item in batch], self.embed_model)
            self.last_embedding_tokens += response.tokens
            for item, vector in zip(batch, response.vectors, strict=True):
                array = np.asarray(vector, dtype=np.float32)
                if array.ndim != 1 or not array.size or not np.isfinite(array).all():
                    raise RAGError("invalid_embedding", "Un vecteur d'embedding est invalide.")
                raw_vectors[(item[0], item[1])] = array

        for path in changed_paths:
            chunks = chunks_by_path[path]
            prepared[path] = (
                notes[path],
                chunks,
                [raw_vectors[(path, i)] for i in range(len(chunks))],
            )

        try:
            # Le gestionnaire SQLite valide toutes les mutations ensemble ou les annule.
            with self.connection:
                if rebuild:
                    if self.fts_enabled:
                        self.connection.execute("DELETE FROM chunks_fts")
                    self.connection.execute("DELETE FROM chunks")
                    self.connection.execute("DELETE FROM documents")
                else:
                    for path in sorted(deleted_paths | updated_paths):
                        if self.fts_enabled:
                            self.connection.execute(
                                "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE document_path = ?)",
                                (path,),
                            )
                        self.connection.execute("DELETE FROM documents WHERE path = ?", (path,))
                for path in changed_paths:
                    note, chunks, vectors = prepared[path]
                    self.connection.execute(
                        "INSERT INTO documents(path, fingerprint, title, aliases_json, tags_json) VALUES (?, ?, ?, ?, ?)",
                        (note.path, note.fingerprint, note.title, json.dumps(note.aliases), json.dumps(note.tags)),
                    )
                    self.connection.executemany(
                        """INSERT INTO chunks(
                            document_path, position, heading, fingerprint, text, vector, dimensions
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        [
                            (
                                note.path,
                                chunk.position,
                                chunk.heading,
                                chunk.fingerprint,
                                chunk.text,
                                vector.tobytes(),
                                int(vector.size),
                            )
                            for chunk, vector in zip(chunks, vectors, strict=True)
                        ],
                    )
                    if self.fts_enabled:
                        # Même transaction : l'index plein texte reste aligné sur `chunks`.
                        self.connection.execute(
                            """INSERT INTO chunks_fts(rowid, text, title)
                               SELECT id, text, ? FROM chunks WHERE document_path = ?""",
                            (note.title, note.path),
                        )
                self.connection.execute(
                    "INSERT OR REPLACE INTO metadata(key, value) VALUES ('embedding_model', ?)",
                    (self.embed_model,),
                )
                self.connection.execute(
                    "INSERT OR REPLACE INTO metadata(key, value) VALUES ('chunk_format', ?)",
                    (CHUNK_FORMAT_VERSION,),
                )
        except sqlite3.Error as exc:
            raise RAGError("index_write_error", "Impossible de mettre à jour l'index SQLite.") from exc

        return IndexUpdate(
            added=len(added_paths),
            updated=len(updated_paths),
            deleted=len(deleted_paths),
            embedded_chunks=len(pending),
        )

    @staticmethod
    def _query_array(query_vector: Iterable[float]) -> tuple[np.ndarray, float]:
        """Valide le vecteur de requête et retourne aussi sa norme."""

        query = np.asarray(list(query_vector), dtype=np.float32)
        query_norm = float(np.linalg.norm(query)) if query.size else 0.0
        if query.ndim != 1 or not query.size or not np.isfinite(query).all() or query_norm == 0:
            raise RAGError("invalid_embedding", "Le vecteur de la question est invalide.")
        return query, query_norm

    def _fts_ranking(self, query_text: str) -> list[int]:
        """Retourne les ids de chunks classés par pertinence FTS5 (vide en cas d'échec)."""

        if not self.fts_enabled:
            return []
        expression = _fts_query(query_text)
        if not expression:
            return []
        try:
            rows = self.connection.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
                (expression, FTS_LIMIT),
            ).fetchall()
        except sqlite3.OperationalError:
            # Toute erreur FTS dégrade proprement vers la recherche sémantique seule.
            return []
        return [int(row[0]) for row in rows]

    def search(
        self,
        query_vector: Iterable[float],
        top_k: int,
        *,
        per_note: int = 2,
        query_text: str | None = None,
        path_prefixes: Sequence[str] | None = None,
    ) -> list[SearchHit]:
        """Classe les chunks (cosinus, fusionné par RRF avec FTS5 si `query_text`) en diversifiant les notes."""

        query, query_norm = self._query_array(query_vector)

        candidates: dict[int, SearchHit] = {}
        rows = self.connection.execute(
            """SELECT c.id, c.document_path, d.title, c.heading, c.text, c.vector, c.dimensions
               FROM chunks c JOIN documents d ON d.path = c.document_path"""
        )
        for row in rows:
            if not _matches_prefixes(row["document_path"], path_prefixes):
                continue
            vector = np.frombuffer(row["vector"], dtype=np.float32, count=row["dimensions"])
            # Ignore les anciens vecteurs incompatibles plutôt que de fausser le calcul.
            if vector.size != query.size:
                continue
            norm = float(np.linalg.norm(vector))
            score = float(np.dot(query, vector) / (query_norm * norm)) if norm else 0.0
            candidates[int(row["id"])] = SearchHit(
                row["document_path"], row["title"], row["heading"], row["text"], score
            )

        semantic = sorted(candidates, key=lambda cid: candidates[cid].score, reverse=True)
        ordered = semantic
        if query_text:
            lexical = [cid for cid in self._fts_ranking(query_text) if cid in candidates]
            if lexical:
                # Fusion RRF : le score affiché reste le cosinus, seul l'ordre change.
                fused: dict[int, float] = {}
                for rank, cid in enumerate(semantic, 1):
                    fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
                for rank, cid in enumerate(lexical, 1):
                    fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
                ordered = sorted(fused, key=lambda cid: (fused[cid], candidates[cid].score), reverse=True)

        # Le quota par note empêche un long document de monopoliser les résultats.
        counts: dict[str, int] = {}
        selected: list[SearchHit] = []
        for cid in ordered:
            hit = candidates[cid]
            if counts.get(hit.path, 0) >= per_note:
                continue
            selected.append(hit)
            counts[hit.path] = counts.get(hit.path, 0) + 1
            if len(selected) >= top_k:
                break
        return selected

    def stats(self) -> dict[str, Any]:
        """Compte notes et chunks, et répartit les notes par dossier de premier niveau."""

        notes = self.connection.execute("SELECT count(*) FROM documents").fetchone()[0]
        chunks = self.connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
        folders: dict[str, int] = {}
        for row in self.connection.execute("SELECT path FROM documents"):
            path = row["path"]
            folder = path.split("/", 1)[0] if "/" in path else "(racine)"
            folders[folder] = folders.get(folder, 0) + 1
        return {"notes": int(notes), "chunks": int(chunks), "folders": dict(sorted(folders.items()))}

    def list_notes(self) -> list[tuple[str, str]]:
        """Liste les notes indexées sous forme (chemin, titre), triées par chemin."""

        return [
            (row["path"], row["title"])
            for row in self.connection.execute("SELECT path, title FROM documents ORDER BY path")
        ]

    def similar_notes(self, path: str, k: int = 5) -> list[SearchHit]:
        """Trouve les notes les plus proches d'une note via son vecteur moyen (sans appel API)."""

        rows = self.connection.execute(
            "SELECT vector, dimensions FROM chunks WHERE document_path = ?", (path,)
        ).fetchall()
        if not rows or k <= 0:
            return []
        vectors = [np.frombuffer(row["vector"], dtype=np.float32, count=row["dimensions"]) for row in rows]
        size = vectors[0].size
        vectors = [vector for vector in vectors if vector.size == size]
        mean = np.mean(np.stack(vectors), axis=0)
        mean_norm = float(np.linalg.norm(mean))
        if not mean_norm or not np.isfinite(mean).all():
            return []

        # Meilleur chunk de chaque autre note.
        best: dict[str, SearchHit] = {}
        others = self.connection.execute(
            """SELECT c.document_path, d.title, c.heading, c.text, c.vector, c.dimensions
               FROM chunks c JOIN documents d ON d.path = c.document_path
               WHERE c.document_path != ?""",
            (path,),
        )
        for row in others:
            vector = np.frombuffer(row["vector"], dtype=np.float32, count=row["dimensions"])
            if vector.size != size:
                continue
            norm = float(np.linalg.norm(vector))
            score = float(np.dot(mean, vector) / (mean_norm * norm)) if norm else 0.0
            current = best.get(row["document_path"])
            if current is None or score > current.score:
                best[row["document_path"]] = SearchHit(
                    row["document_path"], row["title"], row["heading"], row["text"], score
                )
        return sorted(best.values(), key=lambda hit: hit.score, reverse=True)[:k]

    def close(self) -> None:
        """Ferme explicitement la connexion SQLite."""

        self.connection.close()
