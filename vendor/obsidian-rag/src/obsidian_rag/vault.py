"""Lecture en lecture seule et découpage déterministe des notes Markdown."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


MAX_CHARS = 3_500
OVERLAP_CHARS = 350
# La version change dès que le découpage évolue pour déclencher un réindexage.
CHUNK_FORMAT_VERSION = "headings-paragraphs-v1:max3500:overlap350"
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass(frozen=True, slots=True)
class Note:
    """Représentation normalisée d'une note Markdown et de ses métadonnées."""

    path: str
    absolute_path: Path
    fingerprint: str
    title: str
    aliases: tuple[str, ...]
    tags: tuple[str, ...]
    body: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """Segment indexable d'une note, ordonné et identifié par son contenu."""

    position: int
    heading: str
    text: str
    fingerprint: str


def _string_list(value: Any) -> tuple[str, ...]:
    """Normalise les formes YAML possibles en tuple de chaînes non vides."""

    if value is None:
        return ()
    if isinstance(value, str):
        # Obsidian accepte aussi bien une chaîne qu'une liste YAML.
        values = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        values = [str(part).strip() for part in value]
    else:
        values = [str(value).strip()]
    return tuple(value for value in values if value)


def parse_note(path: Path, vault_path: Path) -> Note:
    """Lit une note et extrait sans risque son frontmatter et son titre."""

    raw = path.read_text(encoding="utf-8", errors="replace")
    body = raw
    metadata: dict[str, Any] = {}
    if raw.startswith("---\n") or raw.startswith("---\r\n"):
        lines = raw.splitlines(keepends=True)
        end = next((i for i, line in enumerate(lines[1:], 1) if line.strip() == "---"), None)
        if end is not None:
            try:
                loaded = yaml.safe_load("".join(lines[1:end])) or {}
                if isinstance(loaded, dict):
                    metadata = loaded
                    body = "".join(lines[end + 1 :])
            except yaml.YAMLError:
                # Un frontmatter invalide reste du contenu consultable.
                pass

    # Priorité au titre explicite, puis au premier H1, puis au nom du fichier.
    first_h1 = next(
        (
            match.group(2).strip()
            for line in body.splitlines()
            if (match := _HEADING_RE.match(line)) and len(match.group(1)) == 1
        ),
        None,
    )
    title = str(metadata.get("title") or first_h1 or path.stem).strip()
    relative = path.relative_to(vault_path).as_posix()
    return Note(
        path=relative,
        absolute_path=path,
        fingerprint=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        title=title,
        aliases=_string_list(metadata.get("aliases") or metadata.get("alias")),
        tags=_string_list(metadata.get("tags")),
        body=body,
    )


def is_excluded(relative_path: str, exclude: Sequence[str]) -> bool:
    """Indique si un chemin relatif (posix) est égal à, ou sous, un préfixe exclu."""

    for prefix in exclude:
        prefix = prefix.strip().strip("/")
        if prefix and (relative_path == prefix or relative_path.startswith(prefix + "/")):
            return True
    return False


def iter_notes(vault_path: Path, exclude: Sequence[str] = ()) -> Iterable[Note]:
    """Parcourt les `.md` visibles et non exclus dans un ordre stable entre plateformes."""

    paths = sorted(
        (
            path
            for path in vault_path.rglob("*.md")
            if path.is_file()
            and not any(part.startswith(".") for part in path.relative_to(vault_path).parts)
            and not is_excluded(path.relative_to(vault_path).as_posix(), exclude)
        ),
        key=lambda item: item.relative_to(vault_path).as_posix().casefold(),
    )
    for path in paths:
        yield parse_note(path, vault_path)


def _sections(body: str) -> list[tuple[str, str]]:
    """Sépare le corps selon les titres en conservant leur chemin hiérarchique."""

    headings: list[str] = []
    result: list[tuple[str, str]] = []
    buffer: list[str] = []
    current_heading = ""

    def flush() -> None:
        """Ajoute la section courante lorsqu'elle contient du texte."""

        text = "\n".join(buffer).strip()
        if text:
            result.append((current_heading, text))
        buffer.clear()

    for line in body.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            heading = match.group(2).strip()
            headings[level - 1 :] = [heading]
            current_heading = " > ".join(headings)
        else:
            buffer.append(line)
    flush()
    return result


def _split_text(text: str, limit: int = MAX_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Découpe à une frontière de paragraphe si possible, avec recouvrement."""

    text = text.strip()
    if not text:
        return []
    pieces: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + limit, len(text))
        end = hard_end
        if hard_end < len(text):
            candidates = [
                text.rfind("\n\n", start + limit // 2, hard_end),
                text.rfind("\n", start + limit // 2, hard_end),
                text.rfind(" ", start + limit // 2, hard_end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        next_start = max(start + 1, end - overlap)
        # Évite de commencer au milieu d'un espace ou d'un saut de ligne.
        while next_start < end and text[next_start].isspace():
            next_start += 1
        start = next_start
    return pieces


def chunk_note(note: Note) -> list[Chunk]:
    """Transforme une note en chunks enrichis de leur contexte documentaire."""

    chunks: list[Chunk] = []
    position = 0
    for heading, section_text in _sections(note.body):
        prefix_parts = [f"Titre: {note.title}"]
        if note.aliases:
            prefix_parts.append("Alias: " + ", ".join(note.aliases))
        if note.tags:
            prefix_parts.append("Tags: " + ", ".join(note.tags))
        if heading:
            prefix_parts.append("Section: " + heading)
        prefix = "\n".join(prefix_parts) + "\n\n"
        # Le préfixe compte dans la limite envoyée au modèle d'embedding.
        available = MAX_CHARS - len(prefix)
        if available < 200:
            prefix = f"Titre: {note.title[:200]}\n\n"
            available = MAX_CHARS - len(prefix)
        for piece in _split_text(section_text, available, min(OVERLAP_CHARS, available // 3)):
            text = prefix + piece
            chunks.append(
                Chunk(
                    position=position,
                    heading=heading,
                    text=text,
                    fingerprint=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                )
            )
            position += 1
    return chunks
