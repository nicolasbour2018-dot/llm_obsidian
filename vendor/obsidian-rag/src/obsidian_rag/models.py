"""Objets de transfert stables pour l'API Python et le JSON CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class IndexUpdate:
    added: int = 0
    updated: int = 0
    deleted: int = 0
    embedded_chunks: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class Source:
    id: str
    path: str
    title: str
    heading: str
    score: float
    excerpt: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["score"] = round(self.score, 4)
        return data


@dataclass(slots=True)
class Usage:
    chat_input_tokens: int = 0
    chat_output_tokens: int = 0
    embedding_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class AnswerResult:
    answer: str
    sources: list[Source]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [source.to_dict() for source in self.sources],
            "diagnostics": self.diagnostics,
        }


History = list[tuple[str, str]]

