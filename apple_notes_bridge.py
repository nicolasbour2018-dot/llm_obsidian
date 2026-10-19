#!/usr/bin/env python3
"""Pont Apple Notes → coffre Obsidian, exécuté sur l'hôte macOS.

Compatible avec /usr/bin/python3 (Python 3.9) et limité à la bibliothèque
standard. Le script lit une copie de la base NoteStore.sqlite (ou, en secours,
passe par JXA/osascript), convertit les notes en Markdown et les importe dans « Boite de réception/Apple Notes » du coffre. Il
peut aussi exposer un mini serveur HTTP local que l'interface (conteneur
Docker) appelle pour déclencher une synchronisation.
"""

from __future__ import annotations

import argparse
import gzip
import hmac
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse


LOGGER = logging.getLogger("apple_notes_bridge")

IMPORT_DIR = Path("Boite de réception") / "Apple Notes"
STATE_RELPATH = Path("0 Second Cerveau") / ".apple-notes-sync.json"
DELETED_FOLDERS = {"recently deleted", "suppressions récentes", "suppressions recentes"}
DEFAULT_PORT = 8591
EXPORT_TIMEOUT = 300
HEADER_LINES = 15

_FORBIDDEN_CHARS = re.compile(r'[/\\:*?"<>|#^\[\]]')
_WHITESPACE = re.compile(r"\s+")
_APPLE_ID_LINE = re.compile(r"^apple-notes-id:\s*(?P<id>.+?)\s*$")

# Script JXA : accès groupé par dossier (une requête Apple Event par propriété),
# beaucoup plus rapide que d'interroger chaque note individuellement.
JXA_SCRIPT = r"""
function run() {
  var Notes = Application('Notes');
  var out = [];
  var seen = {};
  var folders = Notes.folders();
  for (var f = 0; f < folders.length; f++) {
    var folder = folders[f];
    var folderName = '';
    try { folderName = folder.name(); } catch (e) { continue; }
    var lower = folderName.toLowerCase();
    if (lower === 'recently deleted' || lower === 'suppressions récentes') { continue; }
    var notes = folder.notes;
    var ids, names, prot, created, modified;
    try {
      ids = notes.id(); names = notes.name(); prot = notes.passwordProtected();
      created = notes.creationDate(); modified = notes.modificationDate();
    } catch (e) { continue; }
    var bodies = null;
    try { bodies = notes.body(); } catch (e) { bodies = null; }
    for (var i = 0; i < ids.length; i++) {
      if (seen[ids[i]] || prot[i]) { continue; }
      seen[ids[i]] = true;
      var body = '';
      if (bodies !== null) { body = bodies[i]; }
      else { try { body = notes[i].body(); } catch (e) { body = ''; } }
      out.push({
        id: ids[i], name: names[i], body: body || '',
        creationDate: created[i] ? created[i].toISOString() : null,
        modificationDate: modified[i] ? modified[i].toISOString() : null,
        folder: folderName, passwordProtected: !!prot[i]
      });
    }
  }
  return JSON.stringify(out);
}
"""


class BridgeError(Exception):
    """Erreur affichable (en français) sans détail sensible."""


# --------------------------------------------------------------------- export


def export_notes() -> List[Dict[str, Any]]:
    """Exporte les notes via la base SQLite (défaut) ou JXA (APPLE_NOTES_EXPORT=jxa)."""

    if os.environ.get("APPLE_NOTES_EXPORT", "sqlite").strip().lower() == "jxa":
        return export_notes_jxa()
    return export_notes_sqlite()


# Base locale de Notes.app ; ses identifiants sont les UUID qu'obsidian-importer
# a écrits dans les `apple-notes-id`, ce qui rend le dédoublonnage exact.
NOTES_DB = Path.home() / "Library" / "Group Containers" / "group.com.apple.notes" / "NoteStore.sqlite"
# Les dates Core Data comptent les secondes depuis le 1er janvier 2001 (UTC).
_CORE_DATA_EPOCH = 978307200
_TRASH_FOLDER_TYPE = 1
_OBJECT_REPLACEMENT = "￼"


def export_notes_sqlite(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Lit les notes depuis une copie de NoteStore.sqlite et rend leur Markdown."""

    source = Path(db_path or NOTES_DB)
    if not source.is_file():
        raise BridgeError("Base Apple Notes introuvable : Notes.app a-t-il déjà été ouvert ?")
    with tempfile.TemporaryDirectory(prefix="apple-notes-") as tmp:
        # Copie de la base et de son journal WAL : Notes.app peut écrire pendant
        # la lecture, et on ne touche jamais au fichier d'origine.
        try:
            for suffix in ("", "-wal", "-shm"):
                part = Path(str(source) + suffix)
                if part.exists():
                    shutil.copy2(part, Path(tmp) / (source.name + suffix))
        except PermissionError:
            raise BridgeError(
                "Lecture de la base Notes refusée : accordez « Accès complet au disque » au "
                "Terminal dans Réglages Système → Confidentialité et sécurité."
            ) from None
        connection = sqlite3.connect(str(Path(tmp) / source.name))
        try:
            return _read_notes(connection)
        except sqlite3.Error as exc:
            LOGGER.warning("Lecture SQLite impossible : %s", exc)
            raise BridgeError("Le format de la base Apple Notes n'est pas reconnu.") from None
        finally:
            connection.close()


def _read_notes(connection: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Extrait les notes non supprimées et non verrouillées de la base copiée."""

    columns = {row[1] for row in connection.execute("PRAGMA table_info(ZICCLOUDSYNCINGOBJECT)")}

    def pick(alias: str, *candidates: str) -> str:
        """Choisit la première colonne existante : son nom varie selon macOS."""

        for name in candidates:
            if name in columns:
                return "%s.%s" % (alias, name)
        return "NULL"

    query = """
        SELECT n.ZIDENTIFIER, n.ZTITLE1, d.ZDATA,
               {created} AS created, {modified} AS modified,
               {locked} AS locked, {deleted} AS deleted,
               f.ZTITLE2 AS folder, {folder_type} AS folder_type
        FROM ZICCLOUDSYNCINGOBJECT n
        JOIN ZICNOTEDATA d ON d.Z_PK = n.ZNOTEDATA
        LEFT JOIN ZICCLOUDSYNCINGOBJECT f ON f.Z_PK = n.ZFOLDER
        WHERE n.ZTITLE1 IS NOT NULL AND d.ZDATA IS NOT NULL
    """.format(
        created=pick("n", "ZCREATIONDATE3", "ZCREATIONDATE1", "ZCREATIONDATE"),
        modified=pick("n", "ZMODIFICATIONDATE1", "ZMODIFICATIONDATE"),
        locked=pick("n", "ZISPASSWORDPROTECTED"),
        deleted=pick("n", "ZMARKEDFORDELETION"),
        folder_type=pick("f", "ZFOLDERTYPE"),
    )
    attachments = _read_attachments(connection, pick)
    notes = []  # type: List[Dict[str, Any]]
    for identifier, title, data, created, modified, locked, deleted, folder, folder_type in connection.execute(query):
        if locked or deleted or folder_type == _TRASH_FOLDER_TYPE:
            continue
        try:
            text, runs = _decode_note_data(data)
        except Exception as exc:  # une note illisible n'interrompt pas l'export
            LOGGER.warning("Note « %s » illisible : %s", title, exc.__class__.__name__)
            continue
        notes.append(
            {
                "id": identifier,
                "name": title,
                "folder": folder or "Notes",
                "creationDate": _iso(created),
                "modificationDate": _iso(modified),
                "passwordProtected": False,
                "markdown": proto_to_markdown(text, runs, title=title, attachments=attachments),
            }
        )
    return notes


def _read_attachments(connection: sqlite3.Connection, pick: Any) -> Dict[str, Dict[str, Any]]:
    """Indexe les pièces jointes (type, titre, URL) par identifiant."""

    query = (
        "SELECT a.ZIDENTIFIER, a.ZTYPEUTI, {title}, {url}, {summary}, {table} "
        "FROM ZICCLOUDSYNCINGOBJECT a WHERE a.ZTYPEUTI IS NOT NULL"
    ).format(
        title=pick("a", "ZTITLE", "ZTITLE1"),
        url=pick("a", "ZURLSTRING"),
        summary=pick("a", "ZSUMMARY"),
        table=pick("a", "ZMERGEABLEDATA1", "ZMERGEABLEDATA"),
    )
    try:
        rows = connection.execute(query).fetchall()
    except sqlite3.Error:
        return {}  # schéma inattendu : les pièces jointes deviennent de simples marqueurs
    return {
        identifier: {"uti": uti, "title": title, "url": url, "summary": summary, "table": table}
        for identifier, uti, title, url, summary, table in rows
        if identifier
    }


def _iso(value: Any) -> Optional[str]:
    """Convertit une date Core Data en ISO 8601 UTC."""

    if value is None:
        return None
    moment = datetime.fromtimestamp(float(value) + _CORE_DATA_EPOCH, tz=timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _pb_fields(buffer: bytes) -> List[Tuple[int, Any]]:
    """Décode un message protobuf en paires (numéro de champ, valeur brute)."""

    fields = []  # type: List[Tuple[int, Any]]
    position = 0

    def varint() -> int:
        """Lit un entier variable à la position courante."""

        nonlocal position
        shift = result = 0
        while True:
            byte = buffer[position]
            position += 1
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return result
            shift += 7

    while position < len(buffer):
        key = varint()
        number, wire = key >> 3, key & 7
        if wire == 0:
            fields.append((number, varint()))
        elif wire == 2:
            size = varint()
            fields.append((number, buffer[position : position + size]))
            position += size
        elif wire == 1:
            position += 8
        elif wire == 5:
            position += 4
        else:
            raise ValueError("type protobuf non géré : %d" % wire)
    return fields


def _pb_first(fields: List[Tuple[int, Any]], number: int, default: Any = None) -> Any:
    """Renvoie la première valeur d'un champ protobuf."""

    return next((value for field, value in fields if field == number), default)


def _decode_note_data(data: bytes) -> Tuple[str, List[Dict[str, Any]]]:
    """Décompresse ZDATA et renvoie le texte brut et ses « attribute runs »."""

    raw = gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
    # NoteStoreProto(2) → Document(3) → Note : texte (2) et mises en forme (5).
    document = _pb_fields(_pb_first(_pb_fields(raw), 2, b""))
    note = _pb_fields(_pb_first(document, 3, b""))
    text = bytes(_pb_first(note, 2, b"")).decode("utf-8", errors="replace")
    runs = []
    for field, value in note:
        if field != 5:
            continue
        run = _pb_fields(value)
        style = _pb_fields(_pb_first(run, 2, b""))
        checklist = _pb_fields(_pb_first(style, 5, b""))
        link = _pb_first(run, 9)
        attachment = _pb_fields(_pb_first(run, 12, b""))
        attachment_id = _pb_first(attachment, 1)
        runs.append(
            {
                "length": _pb_first(run, 1, 0),
                "style": _pb_first(style, 1, -1),
                "indent": _pb_first(style, 4, 0),
                "done": bool(_pb_first(checklist, 2, 0)),
                "weight": _pb_first(run, 5, 0),
                "link": bytes(link).decode("utf-8", errors="replace") if link else None,
                "attachment": bytes(attachment_id).decode("utf-8", errors="replace") if attachment_id else None,
            }
        )
    return text, runs


# Styles de paragraphe Apple Notes → préfixes Markdown.
_LINE_PREFIX = {0: "# ", 1: "## ", 2: "### ", 100: "- ", 101: "- ", 102: "1. "}
_MONOSPACED, _CHECKLIST = 4, 103


def _inline(text: str, weight: int, link: Optional[str]) -> str:
    """Applique gras/italique/lien en laissant les espaces de bord à l'extérieur."""

    core = text.strip()
    if not core:
        return text
    marks = {1: "**", 2: "*", 3: "***"}.get(weight, "")
    styled = "%s%s%s" % (marks, core, marks)
    if link:
        styled = "[%s](%s)" % (styled, link)
    start = text.index(core[0])
    return text[:start] + styled + text[start + len(core) :]


def render_attachment(info: Optional[Dict[str, Any]]) -> str:
    """Rend une pièce jointe : lien cliquable, ou marqueur explicite si non importable."""

    if not info:
        return ""
    uti = str(info.get("uti") or "")
    title = str(info.get("title") or "").strip()
    url = str(info.get("url") or "").strip()
    if url:
        # Un lien collé dans Notes devient une « carte » : c'est souvent toute l'info.
        return "[%s](%s)" % (title or url, url)
    if uti == "com.apple.notes.table":
        try:
            table = table_to_markdown(info.get("table"))
        except Exception as exc:  # CRDT inattendu : on le signale sans échouer
            LOGGER.warning("Tableau illisible : %s", exc.__class__.__name__)
            table = ""
        return ("\n" + table + "\n") if table else "📋 *Tableau Apple Notes (non importé)*"
    is_scan = uti.startswith("com.apple.paper.doc") or uti == "com.apple.notes.gallery"
    is_image = uti.startswith("public.") and any(
        kind in uti for kind in ("image", "jpeg", "png", "heic", "tiff", "gif")
    ) or uti == "com.compuserve.gif"
    if is_scan:
        label = "📑 *Document scanné%s (resté dans Apple Notes)*"
    elif is_image:
        label = "🖼️ *Image%s (restée dans Apple Notes)*"
    elif uti == "com.adobe.pdf":
        label = "📄 *PDF%s (resté dans Apple Notes)*"
    else:
        label = "📎 *Pièce jointe%s (restée dans Apple Notes)*"
    marker = label % (" : " + title if title else "")
    summary = str(info.get("summary") or "").strip()
    if summary and (is_scan or is_image):
        # Texte reconnu par Notes (OCR) : invisible sinon, mais précieux pour le RAG.
        quoted = "\n".join("> " + line if line.strip() else ">" for line in summary.splitlines())
        return "%s\n\n> [!quote]- Texte reconnu\n%s\n" % (marker, quoted)
    return marker


def table_to_markdown(data: Optional[bytes]) -> str:
    """Décode un tableau Apple Notes (CRDT « MergeableData ») en tableau Markdown."""

    if not data:
        return ""
    raw = gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
    # MergableDataProto(2) → MergableDataObject(3) → entrées (3), clés (4), types (5), UUID (6).
    payload = _pb_fields(_pb_first(_pb_fields(_pb_first(_pb_fields(raw), 2, b"")), 3, b""))
    entries = [_pb_fields(value) for field, value in payload if field == 3]
    keys = [bytes(value).decode("utf-8", "replace") for field, value in payload if field == 4]
    types = [bytes(value).decode("utf-8", "replace") for field, value in payload if field == 5]
    uuids = [bytes(value) for field, value in payload if field == 6]

    def object_index(object_id: bytes) -> int:
        """Index d'objet porté par un ObjectID (champ 6)."""

        return int(_pb_first(_pb_fields(object_id), 6, -1))

    def custom_map(entry: List[Tuple[int, Any]]) -> Tuple[int, List[Tuple[int, bytes]]]:
        """Type et paires (clé, ObjectID) d'une entrée « custom_map » (champ 13)."""

        mapping = _pb_fields(_pb_first(entry, 13, b""))
        pairs = []
        for field, value in mapping:
            if field == 3:
                item = _pb_fields(value)
                pairs.append((int(_pb_first(item, 1, -1)), _pb_first(item, 2, b"")))
        return int(_pb_first(mapping, 1, -1)), pairs

    def target_uuid(object_id: bytes) -> bytes:
        """UUID désigné par une référence : 1re valeur entière de sa custom_map."""

        _, pairs = custom_map(entries[object_index(object_id)])
        return uuids[int(_pb_first(_pb_fields(pairs[0][1]), 2, 0))]

    def dictionary(message: bytes) -> List[Tuple[bytes, bytes]]:
        """Paires (clé, valeur) d'un Dictionary protobuf (éléments = champ 1)."""

        pairs = []
        for field, value in _pb_fields(message):
            if field == 1:
                element = _pb_fields(value)
                pairs.append((_pb_first(element, 1, b""), _pb_first(element, 2, b"")))
        return pairs

    def locations(entry: List[Tuple[int, Any]]) -> List[bytes]:
        """Ordre des lignes/colonnes d'un OrderedSet (champ 16), remplacements inclus."""

        ordering = _pb_fields(_pb_first(_pb_fields(_pb_first(entry, 16, b"")), 1, b""))
        array = _pb_fields(_pb_first(ordering, 1, b""))
        order = [bytes(_pb_first(_pb_fields(value), 2, b"")) for field, value in array if field == 2]
        # « contents » associe l'UUID de position (clé) à l'UUID utilisé par les cellules.
        for key, value in dictionary(_pb_first(ordering, 2, b"")):
            position = target_uuid(key)
            if position in order:
                order[order.index(position)] = target_uuid(value)
        return order

    if "com.apple.notes.ICTable" not in types:
        return ""
    table_type = types.index("com.apple.notes.ICTable")
    root = next(entry for entry in entries if _pb_first(entry, 13) is not None and custom_map(entry)[0] == table_type)
    rows = columns = []  # type: List[bytes]
    cells_entry = None
    for key, object_id in custom_map(root)[1]:
        name = keys[key] if 0 <= key < len(keys) else ""
        target = entries[object_index(object_id)]
        if name == "crRows":
            rows = locations(target)
        elif name == "crColumns":
            columns = locations(target)
        elif name == "cellColumns":
            cells_entry = target
    if not rows or not columns or cells_entry is None:
        return ""

    grid = [["" for _ in columns] for _ in rows]
    for column_key, column_value in dictionary(_pb_first(cells_entry, 6, b"")):
        column_uuid = target_uuid(column_key)
        if column_uuid not in columns:
            continue
        column_entry = entries[object_index(column_value)]
        for row_key, row_value in dictionary(_pb_first(column_entry, 6, b"")):
            row_uuid = target_uuid(row_key)
            if row_uuid not in rows:
                continue
            note = _pb_fields(_pb_first(entries[object_index(row_value)], 10, b""))
            text = bytes(_pb_first(note, 2, b"")).decode("utf-8", "replace")
            # Une cellule Markdown tient sur une ligne et ne doit pas contenir « | ».
            cell = text.replace(_OBJECT_REPLACEMENT, "").replace("|", "\\|").strip().replace("\n", "<br>")
            grid[rows.index(row_uuid)][columns.index(column_uuid)] = cell

    header, body = grid[0], grid[1:]
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(columns)]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def proto_to_markdown(
    text: str,
    runs: List[Dict[str, Any]],
    title: Optional[str] = None,
    attachments: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Rend le texte Apple Notes en Markdown d'après ses styles de paragraphe."""

    # Les longueurs des runs sont exprimées en unités UTF-16 (comme NSString).
    encoded = text.encode("utf-16-le")
    segments = []  # type: List[Tuple[str, Dict[str, Any]]]
    offset = 0
    for run in runs:
        end = offset + 2 * int(run["length"])
        segments.append((encoded[offset:end].decode("utf-16-le", errors="replace"), run))
        offset = end
    if offset < len(encoded):
        segments.append((encoded[offset:].decode("utf-16-le", errors="replace"), {}))

    lines = []  # type: List[Tuple[str, Dict[str, Any]]]
    current = []  # type: List[str]
    for chunk, run in segments:
        # Le caractère U+FFFC marque l'emplacement d'une pièce jointe.
        placeholder = render_attachment((attachments or {}).get(run.get("attachment") or ""))
        parts = chunk.replace(_OBJECT_REPLACEMENT, placeholder).split("\n")
        for index, part in enumerate(parts):
            if part:
                current.append(_inline(part, run.get("weight", 0), run.get("link")))
            if index < len(parts) - 1:
                # Le style d'un paragraphe est porté par le run qui contient son « \n ».
                lines.append(("".join(current), run))
                current = []
    if current:
        lines.append(("".join(current), segments[-1][1] if segments else {}))

    output = []  # type: List[str]
    in_code = False
    for index, (line, run) in enumerate(lines):
        style = run.get("style", -1)
        if index == 0 and title and _normalize_title(line) == _normalize_title(title):
            continue  # le titre est déjà rendu par render_note
        if (style == _MONOSPACED) != in_code:
            output.append("```")
            in_code = not in_code
        if in_code:
            output.append(line)
            continue
        indent = "  " * int(run.get("indent", 0) or 0)
        if style == _CHECKLIST:
            output.append("%s- [%s] %s" % (indent, "x" if run.get("done") else " ", line))
        elif style in (100, 101, 102):
            output.append(indent + _LINE_PREFIX[style] + line)
        elif style in _LINE_PREFIX and line.strip():
            output.append(_LINE_PREFIX[style] + line)
        else:
            output.append(line)
    if in_code:
        output.append("```")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(output)).strip()


def export_notes_jxa() -> List[Dict[str, Any]]:
    """Exporte les notes via osascript (lent, identifiants x-coredata)."""

    try:
        proc = subprocess.run(
            ["osascript", "-l", "JavaScript", "-e", JXA_SCRIPT],
            capture_output=True,
            timeout=EXPORT_TIMEOUT,
        )
    except FileNotFoundError:
        raise BridgeError("osascript est introuvable : ce pont doit tourner sur macOS.") from None
    except subprocess.TimeoutExpired:
        raise BridgeError("L'export Apple Notes a dépassé le délai de 300 secondes.") from None

    # Décodage explicite : la locale de l'hôte n'est pas forcément UTF-8.
    stdout = (proc.stdout or b"").decode("utf-8", errors="replace")
    stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
    if proc.returncode != 0:
        if "-1743" in stderr or "not authorized" in stderr.lower() or "pas autoris" in stderr.lower():
            raise BridgeError(
                "Accès à Notes refusé : autorisez le Terminal (ou Python) dans Réglages "
                "Système → Confidentialité et sécurité → Automatisation → Notes."
            )
        LOGGER.warning("osascript a échoué : %s", stderr.strip()[:500])
        raise BridgeError("L'export Apple Notes a échoué (osascript code %d)." % proc.returncode)

    try:
        notes = json.loads(stdout.strip() or "[]")
    except ValueError as exc:
        LOGGER.warning("JSON d'export invalide (%d octets) : %s", len(stdout), exc)
        raise BridgeError("La sortie de l'export Apple Notes n'est pas du JSON valide.") from None
    if not isinstance(notes, list):
        raise BridgeError("La sortie de l'export Apple Notes est inattendue.")
    return [n for n in notes if isinstance(n, dict) and _is_exportable(n)]


def _is_exportable(note: Dict[str, Any]) -> bool:
    """Indique si une note doit être importée (non protégée, hors corbeille)."""

    if note.get("passwordProtected"):
        return False
    folder = str(note.get("folder") or "").strip().lower()
    return folder not in DELETED_FOLDERS and bool(note.get("id"))


# ------------------------------------------------------------ HTML → Markdown


class _MarkdownConverter(HTMLParser):
    """Convertisseur HTML (format Apple Notes) vers Markdown, volontairement simple."""

    _BLOCKS = {"div", "p", "blockquote", "pre", "table", "tr", "section", "article"}
    _IGNORED = {"style", "script", "head", "title"}

    def __init__(self) -> None:
        """Initialise les piles d'état (listes, liens, styles)."""

        super().__init__(convert_charrefs=True)
        self.buffers: List[List[str]] = [[]]
        self.links: List[Optional[str]] = []
        self.lists: List[Dict[str, Any]] = []
        self.marks: List[Tuple[str, int, int]] = []
        self.heading = 0
        self.ignored = 0

    # Outils de tampon -------------------------------------------------------

    @property
    def out(self) -> List[str]:
        """Tampon courant (celui du lien ouvert le plus interne, sinon la racine)."""

        return self.buffers[-1]

    def _tail(self) -> str:
        """Renvoie les derniers caractères émis (tous tampons confondus)."""

        for buffer in reversed(self.buffers):
            text = "".join(buffer[-4:])
            if text:
                return text
        return ""

    def _newline(self) -> None:
        """Termine la ligne courante si elle n'est pas vide."""

        tail = self._tail()
        if tail and not tail.endswith("\n"):
            self.out.append("\n")

    def _blank_line(self) -> None:
        """Garantit une ligne vide avant le bloc suivant."""

        self._newline()
        tail = self._tail()
        if tail and not tail.endswith("\n\n"):
            self.out.append("\n")

    # Callbacks HTMLParser ---------------------------------------------------

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        """Ouvre une balise : émet le préfixe Markdown correspondant."""

        attributes = {k: (v or "") for k, v in attrs}
        if tag in self._IGNORED:
            self.ignored += 1
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._blank_line()
            self.heading += 1
            self.out.append("#" * int(tag[1]) + " ")
        elif tag in ("b", "strong", "i", "em", "s", "strike", "del"):
            if not self.heading:
                marker = {"b": "**", "strong": "**", "i": "*", "em": "*"}.get(tag, "~~")
                self.out.append(marker)
                self.marks.append((tag, len(self.buffers), len(self.out)))
        elif tag == "a":
            self.links.append(attributes.get("href") or None)
            self.buffers.append([])
        elif tag in ("ul", "ol"):
            self._newline()
            checklist = "checklist" in attributes.get("class", "")
            self.lists.append({"kind": tag, "count": 0, "checklist": checklist})
        elif tag == "li":
            self._newline()
            depth = max(len(self.lists), 1)
            current = self.lists[-1] if self.lists else {"kind": "ul", "count": 0, "checklist": False}
            current["count"] += 1
            classes = attributes.get("class", "")
            checked = "checked" in classes.split() or "checked" in attributes
            checklist = current["checklist"] or "checked" in attributes or "unchecked" in classes
            if checklist:
                prefix = "- [x] " if checked else "- [ ] "
            elif current["kind"] == "ol":
                prefix = "%d. " % current["count"]
            else:
                prefix = "- "
            self.out.append("  " * (depth - 1) + prefix)
        elif tag == "br":
            self.out.append("\n")
        elif tag in self._BLOCKS:
            self._newline()
        elif tag in ("td", "th"):
            tail = self._tail()
            if tail and not tail.endswith(("\n", " ")):
                self.out.append(" | ")
        elif tag == "hr":
            self._blank_line()
            self.out.append("---\n")

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        """Balises auto-fermantes (<br/>, <hr/>) : traitées comme ouvrantes."""

        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        """Ferme une balise : émet le suffixe Markdown correspondant."""

        if tag in self._IGNORED:
            self.ignored = max(0, self.ignored - 1)
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.heading = max(0, self.heading - 1)
            self.out.append("\n")
        elif tag in ("b", "strong", "i", "em", "s", "strike", "del"):
            self._close_mark(tag)
        elif tag == "a" and len(self.buffers) > 1:
            text = "".join(self.buffers.pop())
            href = self.links.pop() if self.links else None
            label = text.strip()
            if href and label and label != href:
                self.out.append("[%s](%s)" % (label, href))
            elif href and not label:
                self.out.append("<%s>" % href)
            else:
                self.out.append(text)
        elif tag in ("ul", "ol"):
            if self.lists:
                self.lists.pop()
            self._newline()
        elif tag == "li":
            self._newline()
        elif tag in self._BLOCKS or tag == "p":
            self._newline()
            if tag == "p":
                self._blank_line()

    def _close_mark(self, tag: str) -> None:
        """Ferme un marqueur gras/italique ; le supprime s'il n'entoure que du vide."""

        for index in range(len(self.marks) - 1, -1, -1):
            if self.marks[index][0] != tag:
                continue
            _, buffer_level, position = self.marks.pop(index)
            buffer = self.buffers[buffer_level - 1] if buffer_level <= len(self.buffers) else self.out
            opening = buffer[position - 1] if 0 < position <= len(buffer) else ""
            inner = "".join(buffer[position:])
            if not inner.strip():
                # Marqueur vide (ex. <b><br></b>) : on le retire.
                if opening:
                    del buffer[position - 1]
                return
            # Les espaces doivent rester à l'extérieur des marqueurs Markdown.
            stripped = inner.rstrip()
            trailing = inner[len(stripped):]
            del buffer[position:]
            buffer.append(stripped)
            buffer.append(opening)
            buffer.append(trailing)
            return

    def handle_data(self, data: str) -> None:
        """Ajoute du texte (espaces insécables normalisés)."""

        if self.ignored:
            return
        text = data.replace("\xa0", " ").replace("\r", "")
        text = text.replace("\n", " ")
        if text:
            self.out.append(text)

    def result(self) -> str:
        """Assemble le Markdown final (liens non fermés inclus)."""

        while len(self.buffers) > 1:
            text = "".join(self.buffers.pop())
            self.out.append(text)
        return "".join(self.out)


def _normalize_title(text: str) -> str:
    """Normalise un titre pour comparer (casse, ponctuation Markdown, espaces)."""

    cleaned = re.sub(r"^#+\s*", "", text.strip())
    cleaned = cleaned.replace("**", "").replace("__", "").strip("*_ ")
    return _WHITESPACE.sub(" ", cleaned).strip().casefold()


def html_to_markdown(html: str, title: Optional[str] = None) -> str:
    """Convertit le HTML d'une note Apple en Markdown (sans dupliquer le titre)."""

    converter = _MarkdownConverter()
    converter.feed(html or "")
    converter.close()
    text = unescape(converter.result())

    # Nettoyage : espaces de fin de ligne, lignes vides multiples.
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip("\n")

    # La première ligne d'une note Apple reprend souvent le titre.
    if title:
        lines = text.split("\n")
        first = next((i for i, line in enumerate(lines) if line.strip()), None)
        if first is not None and _normalize_title(lines[first]) == _normalize_title(title):
            text = "\n".join(lines[first + 1 :]).strip("\n")
    return text


# ---------------------------------------------------------------- utilitaires


def sanitize_filename(name: str, max_length: int = 100) -> str:
    """Assainit un nom pour en faire un nom de fichier/dossier sûr."""

    cleaned = _FORBIDDEN_CHARS.sub(" ", name or "")
    cleaned = _WHITESPACE.sub(" ", cleaned).strip().lstrip(". ")
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip()
    return cleaned or "Sans titre"


def _yaml_quote(value: str) -> str:
    """Encadre une valeur YAML de guillemets doubles."""

    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    return '"%s"' % escaped


def _atomic_write(path: Path, content: str) -> None:
    """Écrit un fichier de façon atomique (fichier temporaire + os.replace)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(tmp_name, str(path))
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _read_frontmatter_id(path: Path) -> Optional[str]:
    """Lit seulement les premières lignes d'une note pour trouver apple-notes-id."""

    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            head = [handle.readline() for _ in range(HEADER_LINES)]
    except OSError:
        return None
    if not head or head[0].strip() != "---":
        return None
    for line in head[1:]:
        stripped = line.strip()
        if stripped == "---":
            break
        match = _APPLE_ID_LINE.match(stripped)
        if match:
            return match.group("id").strip("\"'")
    return None


def scan_vault_ids(vault: Path) -> Dict[str, str]:
    """Recense les notes du coffre portant un apple-notes-id : {id: chemin relatif}."""

    found = {}  # type: Dict[str, str]
    for dirpath, dirnames, filenames in os.walk(str(vault)):
        # Ignore les dossiers cachés (.obsidian, .trash, .git…).
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for filename in filenames:
            if not filename.endswith(".md") or filename.startswith("."):
                continue
            path = Path(dirpath) / filename
            note_id = _read_frontmatter_id(path)
            if note_id and note_id not in found:
                found[note_id] = path.relative_to(vault).as_posix()
    return found


def _is_in_import_dir(rel: str) -> bool:
    """Indique si un chemin relatif se trouve dans le dossier d'import Apple Notes."""

    return rel.startswith(IMPORT_DIR.as_posix() + "/")


def _load_state(path: Path) -> Dict[str, Dict[str, Any]]:
    """Charge l'état de synchronisation (dictionnaire vide si absent ou illisible)."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def render_note(note: Dict[str, Any]) -> str:
    """Construit le contenu Markdown complet (frontmatter + titre + corps) d'une note."""

    title = str(note.get("name") or "Sans titre").strip() or "Sans titre"
    lines = [
        "---",
        "apple-notes-id: %s" % _yaml_quote(note.get("id", "")),
        "apple-notes-folder: %s" % _yaml_quote(note.get("folder") or ""),
        "created: %s" % (note.get("creationDate") or ""),
        "modified: %s" % (note.get("modificationDate") or ""),
        "source: apple-notes",
        "tags: [apple-notes]",
        "---",
        "# %s" % title,
        "",
    ]
    # L'export SQLite fournit déjà le Markdown ; l'export JXA fournit du HTML.
    if note.get("markdown") is not None:
        body = str(note["markdown"])
    else:
        body = html_to_markdown(str(note.get("body") or ""), title=title)
    return "\n".join(lines) + "\n" + (body + "\n" if body else "")


def _title_key(name: str) -> str:
    """Clé de comparaison d'un titre/nom de fichier (assaini, insensible à la casse)."""

    return sanitize_filename(name, max_length=1000).casefold()


# ----------------------------------------------------------------------- sync


def sync(vault: Path, *, dry_run: bool = False, notes: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Synchronise Apple Notes vers le coffre ; renvoie un résumé JSON-sérialisable."""

    vault = Path(vault)
    if not vault.is_dir():
        raise BridgeError("Coffre Obsidian introuvable : %s" % vault)
    if notes is None:
        notes = export_notes()
    notes = [n for n in notes if isinstance(n, dict) and _is_exportable(n)]

    state_path = vault / STATE_RELPATH
    state = _load_state(state_path)
    scanned = scan_vault_ids(vault)
    # Repli pour l'export JXA : il renvoie un identifiant x-coredata au lieu de
    # l'UUID stocké par obsidian-importer. On reconnaît alors les notes déjà
    # importées par leur nom de fichier (notes portant un apple-notes-id).
    imported_titles = {
        _title_key(Path(rel).stem): rel for rel in scanned.values() if not _is_in_import_dir(rel)
    }

    result = {
        "dry_run": bool(dry_run),
        "total": len(notes),
        "created": [],
        "updated": [],
        "skipped": 0,
        "errors": [],
    }  # type: Dict[str, Any]
    planned = set()  # type: set
    new_state = dict(state)

    for note in notes:
        note_id = str(note["id"])
        title = str(note.get("name") or "Sans titre").strip() or "Sans titre"
        modified = note.get("modificationDate") or ""
        try:
            entry = state.get(note_id)
            known_rel = entry.get("path") if isinstance(entry, dict) else None
            known_rel = known_rel or scanned.get(note_id)
            # Le repli par titre ne sert qu'aux identifiants x-coredata de JXA.
            if known_rel is None and note_id.startswith("x-coredata") and _title_key(title) in imported_titles:
                known_rel = imported_titles[_title_key(title)]

            if known_rel is not None:
                path = vault / known_rel
                if not _is_in_import_dir(known_rel) or not path.is_file():
                    # Note importée ailleurs ou supprimée localement : jamais réécrite.
                    result["skipped"] += 1
                    continue
                if not isinstance(entry, dict):
                    # Fichier d'import sans état : on l'adopte sans le réécrire.
                    if not dry_run:
                        new_state[note_id] = {
                            "path": known_rel,
                            "modified": modified,
                            "mtime": path.stat().st_mtime,
                        }
                    result["skipped"] += 1
                    continue
                locally_untouched = path.stat().st_mtime == entry.get("mtime")
                if modified != entry.get("modified") and locally_untouched:
                    if not dry_run:
                        _atomic_write(path, render_note(note))
                        new_state[note_id] = {
                            "path": known_rel,
                            "modified": modified,
                            "mtime": path.stat().st_mtime,
                        }
                    result["updated"].append(known_rel)
                else:
                    result["skipped"] += 1
                continue

            # Nouvelle note : chemin unique dans le dossier d'import.
            folder = sanitize_filename(str(note.get("folder") or "Notes"))
            base = sanitize_filename(title)
            directory = vault / IMPORT_DIR / folder
            candidate = directory / (base + ".md")
            counter = 2
            while candidate.exists() or candidate.as_posix() in planned:
                candidate = directory / ("%s (%d).md" % (base, counter))
                counter += 1
            planned.add(candidate.as_posix())
            rel = candidate.relative_to(vault).as_posix()
            if not dry_run:
                _atomic_write(candidate, render_note(note))
                new_state[note_id] = {"path": rel, "modified": modified, "mtime": candidate.stat().st_mtime}
            result["created"].append(rel)
        except Exception as exc:  # une note en erreur n'interrompt pas la sync
            LOGGER.warning("Note « %s » en erreur : %s", title, exc)
            result["errors"].append("%s : %s" % (title, exc.__class__.__name__))

    if not dry_run:
        try:
            _atomic_write(state_path, json.dumps(new_state, ensure_ascii=False, indent=1, sort_keys=True))
        except OSError:
            result["errors"].append("Impossible d'enregistrer l'état de synchronisation.")
    return result


# --------------------------------------------------------------- serveur HTTP


_SYNC_LOCK = threading.Lock()


def make_handler(vault: Path, token: str, sync_func: Any = None) -> type:
    """Fabrique la classe de gestionnaire HTTP liée au coffre et au jeton."""

    run_sync = sync_func or sync

    class Handler(BaseHTTPRequestHandler):
        """Routes : GET /health (public) et POST /sync (jeton requis)."""

        server_version = "AppleNotesBridge/1.0"

        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            """Envoie une réponse JSON."""

            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            """Vérifie l'en-tête X-Bridge-Token en temps constant."""

            given = self.headers.get("X-Bridge-Token") or ""
            return bool(token) and hmac.compare_digest(given.encode("utf-8"), token.encode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802 (nom imposé par http.server)
            """Traite les requêtes GET."""

            route = urlparse(self.path).path
            if route == "/health":
                self._send_json(200, {"ok": True, "vault": vault.name})
            elif not self._authorized():
                self._send_json(401, {"error": "Jeton invalide."})
            else:
                self._send_json(404, {"error": "Route inconnue."})

        def do_POST(self) -> None:  # noqa: N802
            """Traite les requêtes POST (synchronisation)."""

            parsed = urlparse(self.path)
            if not self._authorized():
                self._send_json(401, {"error": "Jeton invalide."})
                return
            if parsed.path != "/sync":
                self._send_json(404, {"error": "Route inconnue."})
                return
            # Vide un éventuel corps de requête.
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            dry_run = (parse_qs(parsed.query).get("dry_run") or ["0"])[0].lower() in ("1", "true", "yes")
            if not _SYNC_LOCK.acquire(blocking=False):
                self._send_json(409, {"error": "Une synchronisation est déjà en cours."})
                return
            try:
                result = run_sync(vault, dry_run=dry_run)
                self._send_json(200, result)
            except BridgeError as exc:
                self._send_json(500, {"error": str(exc)})
            except Exception:
                LOGGER.exception("Erreur pendant la synchronisation")
                self._send_json(500, {"error": "Erreur interne du pont Apple Notes."})
            finally:
                _SYNC_LOCK.release()

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            """Journal minimal (sans en-têtes ni jeton)."""

            LOGGER.info("%s %s", self.address_string(), format % args)

    return Handler


def serve(
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    *,
    vault: Optional[Path] = None,
    token: Optional[str] = None,
) -> None:
    """Lance le serveur HTTP du pont (bloquant) ; coffre/jeton lus dans l'env par défaut."""

    if vault is None:
        vault_value = os.environ.get("OBSIDIAN_VAULT_PATH")
        if not vault_value:
            raise BridgeError("Coffre non défini : utilisez --vault ou OBSIDIAN_VAULT_PATH.")
        vault = Path(vault_value).expanduser()
    if token is None:
        token = os.environ.get("APPLE_NOTES_BRIDGE_TOKEN", "")
    if not token:
        raise BridgeError("APPLE_NOTES_BRIDGE_TOKEN est obligatoire pour lancer le serveur.")
    server = ThreadingHTTPServer((host, port), make_handler(Path(vault), token))
    LOGGER.info("Pont Apple Notes à l'écoute sur http://%s:%d (coffre : %s)", host, port, Path(vault).name)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# ---------------------------------------------------------------- config/CLI


def load_dotenv(path: Path) -> None:
    """Charge un fichier .env minimal (KEY=VALUE) sans écraser os.environ."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def main(argv: Optional[List[str]] = None) -> int:
    """Point d'entrée en ligne de commande."""

    load_dotenv(Path(__file__).resolve().parent / ".env")
    parser = argparse.ArgumentParser(description="Pont Apple Notes → Obsidian")
    parser.add_argument("--vault", help="Chemin du coffre (défaut : OBSIDIAN_VAULT_PATH)")
    sub = parser.add_subparsers(dest="command")
    serve_parser = sub.add_parser("serve", help="Lance le serveur HTTP local")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    sync_parser = sub.add_parser("sync", help="Synchronise une fois et affiche le résultat")
    sync_parser.add_argument("--dry-run", action="store_true", help="N'écrit rien")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command not in ("serve", "sync"):
        parser.print_help()
        return 2
    vault_value = args.vault or os.environ.get("OBSIDIAN_VAULT_PATH")
    if not vault_value:
        print("Coffre non défini : utilisez --vault ou OBSIDIAN_VAULT_PATH.", file=sys.stderr)
        return 2
    vault = Path(vault_value).expanduser()
    try:
        if args.command == "serve":
            serve(args.host, args.port, vault=vault, token=os.environ.get("APPLE_NOTES_BRIDGE_TOKEN", ""))
        else:
            print(json.dumps(sync(vault, dry_run=args.dry_run), ensure_ascii=False, indent=2))
    except BridgeError as exc:
        print("Erreur : %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
