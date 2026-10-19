"""Objets de transfert stables pour l'API Python et le JSON CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class IndexUpdate:
    """Résume les mutations et embeddings produits par une synchronisation."""

    added: int = 0
    updated: int = 0
    deleted: int = 0
    embedded_chunks: int = 0

    def to_dict(self) -> dict[str, int]:
        """Sérialise le résumé pour l'API JSON."""

        return asdict(self)


@dataclass(slots=True)
class Source:
    """Décrit un extrait autorisé à étayer la réponse générée."""

    id: str
    path: str
    title: str
    heading: str
    score: float
    excerpt: str

    def to_dict(self) -> dict[str, Any]:
        """Sérialise la source avec un score stable et lisible."""

        data = asdict(self)
        data["score"] = round(self.score, 4)
        return data


@dataclass(slots=True)
class Usage:
    """Agrège les tokens consommés par la recherche et la génération."""

    chat_input_tokens: int = 0
    chat_output_tokens: int = 0
    embedding_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        """Sérialise les compteurs pour les diagnostics."""

        return asdict(self)


@dataclass(slots=True)
class AnswerResult:
    """Résultat public regroupant réponse, preuves et diagnostics."""

    answer: str
    sources: list[Source]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Construit la représentation stable exposée par la CLI."""

        return {
            "answer": self.answer,
            "sources": [source.to_dict() for source in self.sources],
            "diagnostics": self.diagnostics,
        }


# Chaque tuple représente une paire (question, réponse) déjà validée.
History = list[tuple[str, str]]
