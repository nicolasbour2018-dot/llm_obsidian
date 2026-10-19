"""Adaptateur sûr entre l'interface Streamlit et le paquet Obsidian RAG."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from obsidian_rag import ObsidianRAG, Settings
from obsidian_rag.errors import RAGError


# Sérialise les appels dans un même processus : SQLite et le client Mistral sont
# créés par requête, mais plusieurs sessions Streamlit peuvent viser le même index.
_INFERENCE_LOCK = Lock()

# Six échanges bornent la taille du contexte et le coût de l'embedding.
_MAX_HISTORY = 6


class ChatBackendError(Exception):
    """Erreur dont le texte peut être affiché sans révéler de secret."""


@dataclass
class ChatResult:
    """Réponse prête à afficher : texte, sources sérialisées et avertissement."""

    answer: str
    sources: list[dict] = field(default_factory=list)
    warning: str | None = None
    diagnostics: dict = field(default_factory=dict)


@contextmanager
def _rag() -> Iterator[ObsidianRAG]:
    """Ouvre le moteur sous verrou et convertit toute erreur en message public."""

    try:
        with _INFERENCE_LOCK:
            settings = Settings.from_env()
            with ObsidianRAG(settings) as rag:
                yield rag
    except ChatBackendError:
        raise
    except RAGError as exc:
        # Les RAGError portent volontairement un message public et assaini.
        raise ChatBackendError(exc.message) from None
    except Exception:
        # Masque les détails techniques susceptibles de contenir un secret.
        raise ChatBackendError("Une erreur interne est survenue.") from None


def _source_dict(source: Any) -> dict:
    """Convertit une `Source` du moteur en dictionnaire simple et sérialisable."""

    score = getattr(source, "score", 0.0)
    return {
        "id": getattr(source, "id", ""),
        "path": getattr(source, "path", ""),
        "title": getattr(source, "title", ""),
        "heading": getattr(source, "heading", ""),
        "score": float(score) if score is not None else 0.0,
        "excerpt": getattr(source, "excerpt", ""),
    }


def ask(
    question: str,
    history: Sequence[tuple[str, str]],
    *,
    path_prefixes: Sequence[str] | None = None,
    top_k: int | None = None,
    memory: str = "",
) -> ChatResult:
    """Exécute une question RAG et renvoie une réponse enrichie de ses sources."""

    with _rag() as rag:
        result = rag.ask(
            question,
            history=list(history)[-_MAX_HISTORY:],
            memory=memory,
            path_prefixes=list(path_prefixes) if path_prefixes else None,
            top_k=top_k,
        )
        diagnostics = dict(getattr(result, "diagnostics", None) or {})
        return ChatResult(
            answer=result.answer,
            sources=[_source_dict(source) for source in (getattr(result, "sources", None) or [])],
            warning=diagnostics.get("citation_warning") or None,
            diagnostics=diagnostics,
        )


def answer_question(question: str, history: Sequence[tuple[str, str]]) -> str:
    """Compatibilité : renvoie uniquement le texte de la réponse."""

    return ask(question, history).answer


def search(
    query: str,
    *,
    path_prefixes: Sequence[str] | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """Recherche sémantique pure, sans génération par le LLM."""

    with _rag() as rag:
        sources = rag.search(
            query,
            path_prefixes=list(path_prefixes) if path_prefixes else None,
            top_k=top_k,
        )
        return [_source_dict(source) for source in sources]


def related(path: str, k: int = 5) -> list[dict]:
    """Renvoie les notes les plus proches d'une note donnée."""

    with _rag() as rag:
        return [_source_dict(source) for source in rag.related(path, k=k)]


def stats() -> dict:
    """Renvoie le nombre de notes, de passages et la répartition par dossier."""

    with _rag() as rag:
        return dict(rag.stats())


def list_notes() -> list[tuple[str, str]]:
    """Liste les notes indexées sous forme de couples (chemin, titre)."""

    with _rag() as rag:
        return [(str(path), str(title)) for path, title in rag.list_notes()]


def reindex(force: bool = False) -> dict:
    """Met à jour l'index et résume les changements appliqués."""

    with _rag() as rag:
        update = rag.refresh_index(force=force)
        return {
            "added": getattr(update, "added", 0),
            "updated": getattr(update, "updated", 0),
            "deleted": getattr(update, "deleted", 0),
            "embedded_chunks": getattr(update, "embedded_chunks", 0),
        }


def distill_memory(question: str, answer: str) -> list[str]:
    """Extrait d'un échange les faits durables à mémoriser."""

    with _rag() as rag:
        return list(rag.distill_memory(question, answer))
