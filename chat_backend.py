"""Adaptateur sûr entre l'interface Streamlit et le paquet Obsidian RAG."""

from __future__ import annotations

from collections.abc import Sequence
from threading import Lock

from obsidian_rag import ObsidianRAG, Settings
from obsidian_rag.errors import RAGError


_INFERENCE_LOCK = Lock()


class ChatBackendError(Exception):
    """Erreur dont le texte peut être affiché sans révéler de secret."""


def answer_question(question: str, history: Sequence[tuple[str, str]]) -> str:
    """Retourne uniquement le texte de réponse du moteur RAG."""

    try:
        with _INFERENCE_LOCK:
            settings = Settings.from_env()
            with ObsidianRAG(settings) as rag:
                result = rag.ask(question, history=list(history)[-6:])
        return result.answer
    except RAGError as exc:
        raise ChatBackendError(exc.message) from None
    except Exception:
        raise ChatBackendError("Une erreur interne est survenue.") from None
