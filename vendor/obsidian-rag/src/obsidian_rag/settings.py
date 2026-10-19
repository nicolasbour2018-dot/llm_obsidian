"""Chargement et validation de la configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class Settings:
    """Configuration immuable et validée nécessaire à une session RAG."""

    api_key: str
    vault_path: Path
    chat_model: str = "mistral-small-latest"
    embed_model: str = "mistral-embed"
    top_k: int = 6
    index_path: Path = Path(".cache/obsidian-rag/index.sqlite3")
    # Dossiers (relatifs au coffre) jamais indexés : conversations archivées, corbeille…
    exclude_dirs: tuple[str, ...] = ("0 Second Cerveau/Conversations", "Recently Deleted")
    # « mistral » (repli OpenAI si une clé est fournie) ou « openai » (chat direct).
    chat_provider: str = "mistral"
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-5-mini"

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "Settings":
        """Charge `.env`, donne priorité au shell et valide chaque valeur."""

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

        # Les chemins relatifs sont ancrés dans le répertoire de lancement afin
        # que la CLI et l'application partagent le même emplacement prévisible.
        index_path = Path(
            os.getenv("RAG_INDEX_PATH", ".cache/obsidian-rag/index.sqlite3")
        ).expanduser()
        if not index_path.is_absolute():
            index_path = (Path.cwd() / index_path).resolve()

        chat_model = os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest").strip()
        embed_model = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed").strip()
        if not chat_model or not embed_model:
            raise ConfigurationError("Les noms de modèles Mistral ne peuvent pas être vides.")

        # Variable absente → valeurs par défaut ; chaîne vide → aucune exclusion.
        exclude_raw = os.getenv("RAG_EXCLUDE_DIRS")
        if exclude_raw is None:
            exclude_dirs = cls.__dataclass_fields__["exclude_dirs"].default
        else:
            exclude_dirs = tuple(
                part.strip().strip("/") for part in exclude_raw.split(",") if part.strip().strip("/")
            )

        chat_provider = os.getenv("CHAT_PROVIDER", "mistral").strip().lower() or "mistral"
        if chat_provider not in {"mistral", "openai"}:
            raise ConfigurationError("CHAT_PROVIDER doit valoir « mistral » ou « openai ».")
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if chat_provider == "openai" and not openai_api_key:
            raise ConfigurationError("CHAT_PROVIDER=openai exige la variable OPENAI_API_KEY.")
        openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini").strip() or "gpt-5-mini"

        return cls(
            api_key=api_key,
            vault_path=vault_path,
            chat_model=chat_model,
            embed_model=embed_model,
            top_k=top_k,
            index_path=index_path,
            exclude_dirs=exclude_dirs,
            chat_provider=chat_provider,
            openai_api_key=openai_api_key,
            openai_chat_model=openai_chat_model,
        )
