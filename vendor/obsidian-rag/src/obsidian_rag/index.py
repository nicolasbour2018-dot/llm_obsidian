"""Index SQLite transactionnel et recherche cosinus en mémoire."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .errors import ConfigurationError, RAGError
from .mistral import EmbeddingResponse
from .models import IndexUpdate
from .vault import CHUNK_FORMAT_VERSION, Chunk, Note, chunk_note, iter_notes


@dataclass(frozen=True, slots=True)
class SearchHit:
    path: str
    title: str
    heading: str
    text: str
    score: float


Embedder = Callable[[Sequence[str], str], EmbeddingResponse]


class SQLiteIndex:
    def __init__(self, path: Path, embed_model: str, embedder: Embedder) -> None:
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
        self._create_schema()

    def _create_schema(self) -> None:
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

    def _metadata(self) -> dict[str, str]:
        return {row["key"]: row["value"] for row in self.connection.execute("SELECT key, value FROM metadata")}

    def refresh(self, vault_path: Path, *, force: bool = False) -> IndexUpdate:
        self.last_embedding_tokens = 0
        try:
            notes = {note.path: note for note in iter_notes(vault_path)}
        except OSError as exc:
            raise RAGError("vault_read_error", "Impossible de lire le coffre Obsidian.") from exc
        existing = {
            row["path"]: row["fingerprint"]
            for row in self.connection.execute("SELECT path, fingerprint FROM documents")
        }
        metadata = self._metadata()
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
            with self.connection:
                if rebuild:
                    self.connection.execute("DELETE FROM chunks")
                    self.connection.execute("DELETE FROM documents")
                else:
                    for path in sorted(deleted_paths | updated_paths):
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

    def search(self, query_vector: Iterable[float], top_k: int, *, per_note: int = 2) -> list[SearchHit]:
        query = np.asarray(list(query_vector), dtype=np.float32)
        query_norm = float(np.linalg.norm(query))
        if query.ndim != 1 or not query.size or not np.isfinite(query).all() or query_norm == 0:
            raise RAGError("invalid_embedding", "Le vecteur de la question est invalide.")

        candidates: list[SearchHit] = []
        rows = self.connection.execute(
            """SELECT c.document_path, d.title, c.heading, c.text, c.vector, c.dimensions
               FROM chunks c JOIN documents d ON d.path = c.document_path"""
        )
        for row in rows:
            vector = np.frombuffer(row["vector"], dtype=np.float32, count=row["dimensions"])
            if vector.size != query.size:
                continue
            norm = float(np.linalg.norm(vector))
            score = float(np.dot(query, vector) / (query_norm * norm)) if norm else 0.0
            candidates.append(
                SearchHit(row["document_path"], row["title"], row["heading"], row["text"], score)
            )

        counts: dict[str, int] = {}
        selected: list[SearchHit] = []
        for hit in sorted(candidates, key=lambda item: item.score, reverse=True):
            if counts.get(hit.path, 0) >= per_note:
                continue
            selected.append(hit)
            counts[hit.path] = counts.get(hit.path, 0) + 1
            if len(selected) >= top_k:
                break
        return selected

    def close(self) -> None:
        self.connection.close()
