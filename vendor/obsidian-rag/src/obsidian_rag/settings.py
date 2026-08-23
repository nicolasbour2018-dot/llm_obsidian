"""Chargement et validation de la configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str
    vault_path: Path
    chat_model: str = "mistral-large-latest"
    embed_model: str = "mistral-embed"
    top_k: int = 6
    index_path: Path = Path(".cache/obsidian-rag/index.sqlite3")

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "Settings":
        """Charge `.env` sans remplacer les variables déjà exportées."""
        load_dotenv(dotenv_path=env_file, override=False)
        api_key = os.getenv("MISTRAL_API_KEY", "").strip()
        vault_raw = os.getenv("OBSIDIAN_VAULT_PATH", "").strip()
        if not api_key:
            raise ConfigurationError("La variable MISTRAL_API_KEY est obligatoire.")
        if not vault_raw:
            raise ConfigurationError("La variable OBSIDIAN_VAULT_PATH est obligatoire.")

        vault_path = Path(vault_raw).expanduser().resolve()
        if not vault_path.exists():
            raise ConfigurationError(f"Le coffre Obsidian n'existe pas : {vault_path}")
        if not vault_path.is_dir():
            raise ConfigurationError(f"Le chemin du coffre n'est pas un dossier : {vault_path}")

        top_k_raw = os.getenv("RAG_TOP_K", "6").strip()
        try:
            top_k = int(top_k_raw)
        except ValueError as exc:
            raise ConfigurationError("RAG_TOP_K doit être un entier strictement positif.") from exc
        if top_k <= 0:
            raise ConfigurationError("RAG_TOP_K doit être un entier strictement positif.")

        index_path = Path(
            os.getenv("RAG_INDEX_PATH", ".cache/obsidian-rag/index.sqlite3")
        ).expanduser()
        if not index_path.is_absolute():
            index_path = (Path.cwd() / index_path).resolve()

        chat_model = os.getenv("MISTRAL_CHAT_MODEL", "mistral-large-latest").strip()
        embed_model = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed").strip()
        if not chat_model or not embed_model:
            raise ConfigurationError("Les noms de modèles Mistral ne peuvent pas être vides.")

        return cls(
            api_key=api_key,
            vault_path=vault_path,
            chat_model=chat_model,
            embed_model=embed_model,
            top_k=top_k,
            index_path=index_path,
        )

