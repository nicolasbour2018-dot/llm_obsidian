"""Accès en écriture au dossier « 0 Second Cerveau » du coffre Obsidian.

Ce module gère la note de mémoire durable de l'assistant et l'archivage des
conversations en notes Markdown. Il n'utilise que la bibliothèque standard.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any


DEFAULT_BRAIN_DIR = "/vault/0 Second Cerveau"
MEMORY_FILENAME = "Mémoire LLM.md"
CONVERSATIONS_DIRNAME = "Conversations"
MEMORY_SECTION = "## Souvenirs"

# Frontmatter par défaut si la note de mémoire n'existe pas encore.
_DEFAULT_MEMORY_FRONTMATTER = "---\ntype: llm-memory\ntags: [second-cerveau, memoire-ia]\n---\n"
_DEFAULT_MEMORY_BODY = (
    "# Mémoire LLM\n\n"
    "Faits et préférences durables retenus par l'assistant. Cette note est modifiable\n"
    "librement : elle est relue à chaque question.\n\n"
    f"{MEMORY_SECTION}\n"
)

# Caractères interdits dans les noms de fichiers ou cassant les wikilinks.
_FORBIDDEN_CHARS = re.compile(r'[/\\:*?"<>|#^\[\]]')
_WHITESPACE = re.compile(r"\s+")
# Ligne de souvenir : "- AAAA-MM-JJ — fait ([[lien]])" (date et lien facultatifs).
_MEMORY_LINE = re.compile(
    r"^\s*[-*]\s+(?:\d{4}-\d{2}-\d{2}\s+—\s+)?(?P<fact>.*?)(?:\s+\(\[\[[^\]]*\]\]\))?\s*$"
)
_CONV_ID_LINE = re.compile(r"^conversation-id:\s*(?P<id>.+?)\s*$")


class BrainVaultError(Exception):
    """Erreur dont le texte peut être affiché sans révéler de secret."""


def sanitize_filename(name: str, max_length: int = 60) -> str:
    """Assainit un titre pour en faire un nom de fichier Obsidian sûr."""

    # Retire les caractères interdits puis normalise les espaces.
    cleaned = _FORBIDDEN_CHARS.sub(" ", name or "")
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    # Un nom commençant par un point deviendrait un fichier caché.
    cleaned = cleaned.lstrip(". ")
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip()
    return cleaned or "Sans titre"


def split_frontmatter(text: str) -> tuple[str, str]:
    """Sépare le frontmatter YAML (délimiteurs inclus) du corps de la note."""

    if text.startswith("---"):
        lines = text.splitlines(keepends=True)
        if lines and lines[0].strip() == "---":
            for index in range(1, len(lines)):
                if lines[index].strip() == "---":
                    return "".join(lines[: index + 1]), "".join(lines[index + 1 :])
    return "", text


def _yaml_quote(value: str) -> str:
    """Encadre une valeur YAML de guillemets doubles en échappant le nécessaire."""

    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    return f'"{escaped}"'


def _normalize_fact(text: str) -> str:
    """Normalise un fait pour la détection de doublons (casse et espaces)."""

    return _WHITESPACE.sub(" ", text).strip().casefold()


class BrainVault:
    """Lecture/écriture de la mémoire LLM et des conversations dans le coffre."""

    def __init__(self, root: Path) -> None:
        """Mémorise le dossier racine « 0 Second Cerveau »."""

        self.root = Path(root)
        self.memory_path = self.root / MEMORY_FILENAME
        self.conversations_dir = self.root / CONVERSATIONS_DIRNAME

    @classmethod
    def from_env(cls) -> "BrainVault":
        """Construit l'instance depuis la variable d'environnement BRAIN_DIR."""

        return cls(Path(os.environ.get("BRAIN_DIR") or DEFAULT_BRAIN_DIR))

    # ------------------------------------------------------------------ outils

    def _read_text(self, path: Path) -> str:
        """Lit un fichier texte ; renvoie "" s'il n'existe pas."""

        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""
        except OSError:
            raise BrainVaultError(f"Impossible de lire la note « {path.name} ».") from None

    def _atomic_write(self, path: Path, content: str) -> None:
        """Écrit un fichier de façon atomique (fichier temporaire + os.replace)."""

        tmp_name: str | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".md", dir=str(path.parent))
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
            os.replace(tmp_name, path)
            tmp_name = None
        except OSError:
            raise BrainVaultError(
                f"Impossible d'écrire la note « {path.name} » dans le coffre "
                "(vérifiez que le dossier est monté en écriture)."
            ) from None
        finally:
            # Nettoie le fichier temporaire en cas d'échec.
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    def _vault_relative(self, path: Path) -> str:
        """Renvoie le chemin de note (sans .md) relatif au coffre, pour un wikilink."""

        # Le coffre est le parent du dossier « 0 Second Cerveau ».
        rel = Path(self.root.name) / path.relative_to(self.root)
        return rel.with_suffix("").as_posix()

    # ----------------------------------------------------------------- mémoire

    def read_memory(self) -> str:
        """Renvoie le contenu de la mémoire sans frontmatter ("" si absente)."""

        _, body = split_frontmatter(self._read_text(self.memory_path))
        return body.strip("\n")

    def write_memory(self, text: str) -> None:
        """Remplace le corps de la mémoire en préservant (ou créant) le frontmatter."""

        frontmatter, _ = split_frontmatter(self._read_text(self.memory_path))
        if not frontmatter:
            frontmatter = _DEFAULT_MEMORY_FRONTMATTER
        if not frontmatter.endswith("\n"):
            frontmatter += "\n"
        body = text.strip("\n") + "\n"
        self._atomic_write(self.memory_path, frontmatter + body)

    def append_memory(self, facts: Sequence[str], link: str | None = None) -> int:
        """Ajoute des faits datés sous « ## Souvenirs » ; renvoie le nombre ajouté."""

        current = self._read_text(self.memory_path)
        frontmatter, body = split_frontmatter(current)
        if not current:
            frontmatter, body = _DEFAULT_MEMORY_FRONTMATTER, _DEFAULT_MEMORY_BODY

        lines = body.splitlines()
        # Localise la section des souvenirs ; la crée à la fin si absente.
        try:
            section_index = next(i for i, line in enumerate(lines) if line.strip() == MEMORY_SECTION)
        except StopIteration:
            while lines and not lines[-1].strip():
                lines.pop()
            if lines:
                lines.append("")
            lines.append(MEMORY_SECTION)
            section_index = len(lines) - 1

        # Fin de la section : prochain titre de niveau ≤ 2 ou fin de fichier.
        end_index = len(lines)
        for i in range(section_index + 1, len(lines)):
            if re.match(r"^#{1,2}\s", lines[i]):
                end_index = i
                break

        # Faits déjà connus (dans la section) pour le dédoublonnage.
        known: set[str] = set()
        for line in lines[section_index + 1 : end_index]:
            match = _MEMORY_LINE.match(line)
            if match and match.group("fact"):
                known.add(_normalize_fact(match.group("fact")))

        today = date.today().isoformat()
        link_suffix = f" ([[{link}]])" if link else ""
        new_lines: list[str] = []
        for fact in facts:
            cleaned = _WHITESPACE.sub(" ", str(fact or "")).strip()
            key = _normalize_fact(cleaned)
            if not cleaned or key in known:
                continue
            known.add(key)
            new_lines.append(f"- {today} — {cleaned}{link_suffix}")

        if not new_lines:
            return 0

        # Insère après la dernière ligne non vide de la section.
        insert_at = end_index
        while insert_at > section_index + 1 and not lines[insert_at - 1].strip():
            insert_at -= 1
        lines[insert_at:insert_at] = new_lines
        if insert_at + len(new_lines) < len(lines) and lines[insert_at + len(new_lines)].strip():
            lines.insert(insert_at + len(new_lines), "")

        if frontmatter and not frontmatter.endswith("\n"):
            frontmatter += "\n"
        self._atomic_write(self.memory_path, frontmatter + "\n".join(lines).rstrip("\n") + "\n")
        return len(new_lines)

    # ----------------------------------------------------------- conversations

    def _find_conversation(self, conv_id: str) -> Path | None:
        """Retrouve le fichier d'une conversation via son champ conversation-id."""

        if not self.conversations_dir.is_dir():
            return None
        try:
            candidates = sorted(self.conversations_dir.glob("*.md"))
        except OSError:
            return None
        for path in candidates:
            try:
                with path.open(encoding="utf-8") as handle:
                    head = [handle.readline() for _ in range(12)]
            except OSError:
                continue
            if not head or head[0].strip() != "---":
                continue
            for line in head[1:]:
                if line.strip() == "---":
                    break
                match = _CONV_ID_LINE.match(line.strip())
                if match and match.group("id").strip('"\'') == conv_id:
                    return path
        return None

    def save_conversation(
        self,
        conv_id: str,
        title: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        started_at: datetime | None = None,
    ) -> str:
        """Écrit (ou réécrit) la note d'une conversation ; renvoie son wikilink."""

        started = started_at or datetime.now()
        path = self._find_conversation(conv_id)
        if path is None:
            base = f"{started:%Y-%m-%d %H%M} - {sanitize_filename(title)}"
            path = self.conversations_dir / f"{base}.md"
            counter = 2
            # Évite d'écraser une autre conversation portant le même nom.
            while path.exists():
                path = self.conversations_dir / f"{base} ({counter}).md"
                counter += 1

        frontmatter = (
            "---\n"
            "type: conversation-ia\n"
            f"conversation-id: {_yaml_quote(conv_id)}\n"
            f"date: {started.isoformat(timespec='minutes')}\n"
            "tags: [conversation-ia]\n"
            "---\n"
        )
        parts: list[str] = [f"# {(title or 'Conversation').strip()}\n"]
        for message in messages:
            role = message.get("role")
            content = str(message.get("content") or "").strip()
            if role == "user":
                parts.append(f"## 🙋 Question\n\n{content}\n")
            elif role == "assistant":
                section = f"## 🤖 Réponse\n\n{content}\n"
                links = self._format_sources(message.get("sources") or [])
                if links:
                    section += f"\nSources : {links}\n"
                parts.append(section)

        self._atomic_write(path, frontmatter + "\n".join(parts))
        return self._vault_relative(path)

    @staticmethod
    def _format_sources(sources: Sequence[Mapping[str, Any]]) -> str:
        """Formate les sources en wikilinks [[chemin sans .md|Titre]] dédoublonnés."""

        seen: set[str] = set()
        links: list[str] = []
        for source in sources:
            raw_path = str(source.get("path") or "").strip()
            if not raw_path:
                continue
            target = raw_path[:-3] if raw_path.lower().endswith(".md") else raw_path
            if target in seen:
                continue
            seen.add(target)
            label = str(source.get("title") or "").replace("|", " ").replace("]", " ").strip()
            links.append(f"[[{target}|{label}]]" if label else f"[[{target}]]")
        return ", ".join(links)
