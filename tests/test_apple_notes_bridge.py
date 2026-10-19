"""Tests du pont Apple Notes et de son client (compatibles Python 3.9)."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import apple_notes_bridge as bridge
import apple_notes_client as client


def _note(note_id: str, name: str, body: str = "<div>Texte</div>", modified: str = "2026-10-01T10:00:00.000Z",
          folder: str = "Notes", **extra: Any) -> Dict[str, Any]:
    """Construit une note exportée factice."""

    note = {
        "id": note_id,
        "name": name,
        "body": body,
        "creationDate": "2026-01-01T10:00:00.000Z",
        "modificationDate": modified,
        "folder": folder,
        "passwordProtected": False,
    }
    note.update(extra)
    return note


class HtmlToMarkdownTests(unittest.TestCase):
    """Conversion HTML Apple Notes → Markdown."""

    def test_headings_bold_italic_and_title_dedup(self) -> None:
        html = "<div><h1>Ma note</h1></div><div><b>gras</b> et <i>ital</i></div><div><h2>Section</h2></div>"
        md = bridge.html_to_markdown(html, title="Ma note")
        self.assertEqual(md, "**gras** et *ital*\n\n## Section")

    def test_bold_title_dedup(self) -> None:
        md = bridge.html_to_markdown("<div><b>Courses</b></div><div>lait</div>", title="Courses")
        self.assertEqual(md, "lait")

    def test_lists_nested_and_ordered(self) -> None:
        html = "<ul><li>a</li><li>b<ul><li>b1</li></ul></li></ul><ol><li>un</li><li>deux</li></ol>"
        md = bridge.html_to_markdown(html)
        self.assertIn("- a\n- b\n  - b1", md)
        self.assertIn("1. un\n2. deux", md)

    def test_links_entities_br(self) -> None:
        html = '<div>Voir <a href="https://ex.com/?a=1&amp;b=2">le site</a> &eacute;t&eacute;<br>ligne 2</div>'
        md = bridge.html_to_markdown(html)
        self.assertEqual(md, "Voir [le site](https://ex.com/?a=1&b=2) été\nligne 2")

    def test_checklist(self) -> None:
        html = '<ul class="checklist"><li class="checked">fait</li><li>à faire</li></ul>'
        self.assertEqual(bridge.html_to_markdown(html), "- [x] fait\n- [ ] à faire")

    def test_blank_lines_compressed_and_empty_bold_removed(self) -> None:
        html = "<div>a</div><div><br></div><div><br></div><div><br></div><div><b><br></b></div><div>b</div>"
        self.assertEqual(bridge.html_to_markdown(html), "a\n\nb")


class SyncTests(unittest.TestCase):
    """Synchronisation avec des notes injectées dans un coffre temporaire."""

    def setUp(self) -> None:
        """Crée un faux coffre."""

        self._tmp = tempfile.TemporaryDirectory()
        self.vault = Path(self._tmp.name) / "Coffre"
        (self.vault / "0 Second Cerveau").mkdir(parents=True)
        (self.vault / "Ailleurs").mkdir()
        (self.vault / "Ailleurs" / "Déjà.md").write_text(
            "---\napple-notes-id: EXISTANT\n---\nContenu local\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        """Supprime le coffre temporaire."""

        self._tmp.cleanup()

    def _files(self) -> List[str]:
        """Liste les fichiers du coffre (chemins relatifs)."""

        return sorted(p.relative_to(self.vault).as_posix() for p in self.vault.rglob("*") if p.is_file())

    def test_dry_run_writes_nothing(self) -> None:
        before = self._files()
        result = bridge.sync(self.vault, dry_run=True, notes=[_note("n1", "Nouvelle"), _note("EXISTANT", "Déjà")])
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["created"], ["Boite de réception/Apple Notes/Notes/Nouvelle.md"])
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(self._files(), before)

    def test_create_dedupe_and_collisions(self) -> None:
        notes = [
            _note("n1", "Titre: A", body="<div><h1>Titre: A</h1></div><div>corps</div>"),
            _note("n2", "Titre: A"),
            _note("EXISTANT", "Autre nom"),
            # Repli par nom de fichier, réservé aux identifiants x-coredata de JXA.
            _note("x-coredata://S/ICNote/p3", "Déjà"),
            _note("p", "Secrète", passwordProtected=True),
            _note("d", "Supprimée", folder="Recently Deleted"),
        ]
        result = bridge.sync(self.vault, notes=notes)
        self.assertEqual(result["total"], 4)
        self.assertEqual(
            result["created"],
            ["Boite de réception/Apple Notes/Notes/Titre A.md", "Boite de réception/Apple Notes/Notes/Titre A (2).md"],
        )
        self.assertEqual(result["skipped"], 2)
        self.assertEqual(result["errors"], [])
        text = (self.vault / result["created"][0]).read_text(encoding="utf-8")
        self.assertIn('apple-notes-id: "n1"', text)
        self.assertIn("source: apple-notes", text)
        self.assertIn("tags: [apple-notes]", text)
        self.assertIn("# Titre: A\n\ncorps\n", text)
        self.assertEqual(text.count("Titre: A"), 1)
        self.assertEqual(
            (self.vault / "Ailleurs" / "Déjà.md").read_text(encoding="utf-8"),
            "---\napple-notes-id: EXISTANT\n---\nContenu local\n",
        )
        state = json.loads((self.vault / bridge.STATE_RELPATH).read_text(encoding="utf-8"))
        self.assertIn("n1", state)
        # Seconde sync identique : rien de nouveau.
        again = bridge.sync(self.vault, notes=notes)
        self.assertEqual(again["created"], [])
        self.assertEqual(again["updated"], [])

    def test_update_only_if_not_locally_modified(self) -> None:
        bridge.sync(self.vault, notes=[_note("n1", "Note", body="<div>v1</div>")])
        path = self.vault / "Boite de réception/Apple Notes/Notes/Note.md"
        changed = [_note("n1", "Note", body="<div>v2</div>", modified="2026-10-02T10:00:00.000Z")]

        dry = bridge.sync(self.vault, dry_run=True, notes=changed)
        self.assertEqual(dry["updated"], ["Boite de réception/Apple Notes/Notes/Note.md"])
        self.assertIn("v1", path.read_text(encoding="utf-8"))

        result = bridge.sync(self.vault, notes=changed)
        self.assertEqual(result["updated"], ["Boite de réception/Apple Notes/Notes/Note.md"])
        self.assertIn("v2", path.read_text(encoding="utf-8"))

        # Modification locale (mtime différent) : la note n'est plus écrasée.
        path.write_text(path.read_text(encoding="utf-8") + "\nmodif locale\n", encoding="utf-8")
        st = path.stat()
        os.utime(str(path), (st.st_atime, st.st_mtime + 10))
        changed3 = [_note("n1", "Note", body="<div>v3</div>", modified="2026-10-03T10:00:00.000Z")]
        result = bridge.sync(self.vault, notes=changed3)
        self.assertEqual(result["updated"], [])
        self.assertEqual(result["skipped"], 1)
        self.assertIn("modif locale", path.read_text(encoding="utf-8"))

    def test_scan_ignores_hidden_dirs(self) -> None:
        (self.vault / ".trash").mkdir()
        (self.vault / ".trash" / "x.md").write_text("---\napple-notes-id: CACHE\n---\n", encoding="utf-8")
        self.assertEqual(bridge.scan_vault_ids(self.vault), {"EXISTANT": "Ailleurs/Déjà.md"})


class ServerAndClientTests(unittest.TestCase):
    """Serveur HTTP du pont et client urllib, sur un port local éphémère."""

    def setUp(self) -> None:
        """Démarre le serveur avec une fonction de sync factice."""

        self._tmp = tempfile.TemporaryDirectory()
        self.vault = Path(self._tmp.name) / "MonCoffre"
        self.vault.mkdir()
        self.calls = []  # type: List[bool]
        self.block = threading.Event()
        self.block.set()

        def fake_sync(vault: Path, dry_run: bool = False) -> Dict[str, Any]:
            self.calls.append(dry_run)
            self.block.wait(5)
            return {"dry_run": dry_run, "total": 0, "created": [], "updated": [], "skipped": 0, "errors": []}

        handler = bridge.make_handler(self.vault, "secret", sync_func=fake_sync)
        self.server = bridge.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        url = "http://127.0.0.1:%d" % self.server.server_address[1]
        self.env = mock.patch.dict(os.environ, {"APPLE_NOTES_BRIDGE_URL": url, "APPLE_NOTES_BRIDGE_TOKEN": "secret"})
        self.env.start()

    def tearDown(self) -> None:
        """Arrête le serveur."""

        self.env.stop()
        self.block.set()
        self.server.shutdown()
        self.server.server_close()
        self._tmp.cleanup()

    def test_health_and_sync(self) -> None:
        self.assertEqual(client.bridge_status(), {"ok": True, "vault": "MonCoffre"})
        self.assertTrue(client.trigger_sync(dry_run=True)["dry_run"])
        self.assertFalse(client.trigger_sync(dry_run=False)["dry_run"])
        self.assertEqual(self.calls, [True, False])

    def test_bad_token(self) -> None:
        with mock.patch.dict(os.environ, {"APPLE_NOTES_BRIDGE_TOKEN": "faux"}):
            with self.assertRaises(client.BridgeError) as ctx:
                client.trigger_sync(dry_run=True)
        self.assertIn("jeton", str(ctx.exception))
        self.assertEqual(self.calls, [])

    def test_concurrent_sync_returns_409(self) -> None:
        self.block.clear()
        first = threading.Thread(target=client.trigger_sync, args=(True,))
        first.start()
        for _ in range(100):
            if self.calls:
                break
            threading.Event().wait(0.02)
        with self.assertRaises(client.BridgeError) as ctx:
            client.trigger_sync(dry_run=True)
        self.assertIn("déjà en cours", str(ctx.exception))
        self.block.set()
        first.join(5)

    def test_unreachable(self) -> None:
        with mock.patch.dict(os.environ, {"APPLE_NOTES_BRIDGE_URL": "http://127.0.0.1:9"}):
            self.assertIsNone(client.bridge_status(timeout=0.5))
            with self.assertRaises(client.BridgeError):
                client.trigger_sync(dry_run=True, timeout=0.5)


class DotenvTests(unittest.TestCase):
    """Parseur .env minimal."""

    def test_load_dotenv_does_not_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("# c\nBRIDGE_T1='a'\nBRIDGE_T2=b=c\nexport BRIDGE_T3=\"x\"\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"BRIDGE_T1": "garde"}):
                bridge.load_dotenv(env_file)
                self.assertEqual(os.environ["BRIDGE_T1"], "garde")
                self.assertEqual(os.environ["BRIDGE_T2"], "b=c")
                self.assertEqual(os.environ["BRIDGE_T3"], "x")


def _varint(value: int) -> bytes:
    """Encode un entier protobuf variable."""

    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _field(number: int, value: Any) -> bytes:
    """Encode un champ protobuf (varint ou bytes/chaîne)."""

    if isinstance(value, int):
        return _varint(number << 3) + _varint(value)
    data = value.encode("utf-8") if isinstance(value, str) else value
    return _varint(number << 3 | 2) + _varint(len(data)) + data


def _note_data(segments: List[tuple]) -> bytes:
    """Construit un ZDATA gzip : segments = (texte, style, poids, lien, coché)."""

    import gzip

    text = "".join(segment[0] for segment in segments)
    runs = b""
    for chunk, style, weight, link, done in segments:
        paragraph = b""
        if style is not None:
            paragraph += _field(1, style)
        if done is not None:
            paragraph += _field(5, _field(1, b"uuid") + _field(2, int(done)))
        run = _field(1, len(chunk.encode("utf-16-le")) // 2)
        if paragraph:
            run += _field(2, paragraph)
        if weight:
            run += _field(5, weight)
        if link:
            run += _field(9, link)
        runs += _field(5, run)
    note = _field(2, text) + runs
    return gzip.compress(_field(2, _field(3, note)))


class SqliteExportTests(unittest.TestCase):
    """Export depuis une base NoteStore synthétique (aucune vraie note lue)."""

    def _make_db(self, directory: Path) -> Path:
        """Crée une base au schéma minimal de Notes.app."""

        import sqlite3

        path = directory / "NoteStore.sqlite"
        db = sqlite3.connect(str(path))
        db.executescript(
            """
            CREATE TABLE ZICCLOUDSYNCINGOBJECT (
                Z_PK INTEGER PRIMARY KEY, ZIDENTIFIER TEXT, ZTITLE1 TEXT, ZTITLE2 TEXT,
                ZNOTEDATA INTEGER, ZFOLDER INTEGER, ZCREATIONDATE1 REAL,
                ZMODIFICATIONDATE1 REAL, ZISPASSWORDPROTECTED INTEGER,
                ZMARKEDFORDELETION INTEGER, ZFOLDERTYPE INTEGER
            );
            CREATE TABLE ZICNOTEDATA (Z_PK INTEGER PRIMARY KEY, ZDATA BLOB);
            INSERT INTO ZICCLOUDSYNCINGOBJECT (Z_PK, ZTITLE2, ZFOLDERTYPE) VALUES (10, 'Perso', 0), (11, 'Corbeille', 1);
            """
        )
        body = _note_data(
            [
                ("Courses 🛒\n", 0, 0, None, None),
                ("Lait\n", 103, 0, None, True),
                ("Pain\n", 103, 0, None, False),
                ("Important", None, 1, None, None),
                (" voir ", None, 0, None, None),
                ("site", None, 0, "https://example.org", None),
                ("\n", None, 0, None, None),
            ]
        )
        rows = [
            (1, "UUID-OK", "Courses 🛒", 1, 10, 0, 0),
            (2, "UUID-LOCK", "Secret", 2, 10, 1, 0),
            (3, "UUID-TRASH", "Jetée", 3, 11, 0, 0),
        ]
        for pk, uuid, title, data_pk, folder, locked, deleted in rows:
            db.execute("INSERT INTO ZICNOTEDATA VALUES (?, ?)", (data_pk, body))
            db.execute(
                "INSERT INTO ZICCLOUDSYNCINGOBJECT (Z_PK, ZIDENTIFIER, ZTITLE1, ZNOTEDATA, ZFOLDER, "
                "ZCREATIONDATE1, ZMODIFICATIONDATE1, ZISPASSWORDPROTECTED, ZMARKEDFORDELETION) "
                "VALUES (?, ?, ?, ?, ?, 0, 0, ?, ?)",
                (pk, uuid, title, data_pk, folder, locked, deleted),
            )
        db.commit()
        db.close()
        return path

    def test_exports_markdown_and_skips_locked_and_trash(self) -> None:
        """Seule la note normale sort, avec checklist, gras et lien en Markdown."""

        with tempfile.TemporaryDirectory() as tmp:
            notes = bridge.export_notes_sqlite(self._make_db(Path(tmp)))

        self.assertEqual([note["id"] for note in notes], ["UUID-OK"])
        note = notes[0]
        self.assertEqual(note["folder"], "Perso")
        self.assertEqual(note["creationDate"], "2001-01-01T00:00:00Z")
        self.assertEqual(
            note["markdown"],
            "- [x] Lait\n- [ ] Pain\n**Important** voir [site](https://example.org)",
        )

    def test_attachments_become_links_or_explicit_markers(self) -> None:
        """Une carte de lien garde son URL ; une image laisse un marqueur."""

        text = "Titre\n￼\n￼\n"
        runs = [
            {"length": 6, "style": 0},
            {"length": 1, "attachment": "A-URL"},
            {"length": 1},
            {"length": 1, "attachment": "A-IMG"},
            {"length": 1},
        ]
        attachments = {
            "A-URL": {"uti": "public.url", "title": "Doc Python", "url": "https://docs.python.org"},
            "A-IMG": {"uti": "public.jpeg", "title": None, "url": None},
        }

        markdown = bridge.proto_to_markdown(text, runs, title="Titre", attachments=attachments)

        self.assertEqual(
            markdown,
            "[Doc Python](https://docs.python.org)\n🖼️ *Image (restée dans Apple Notes)*",
        )

    def test_table_crdt_becomes_markdown_table(self) -> None:
        """Le tableau suit l'ordre des lignes et l'association position → cellule."""

        import gzip

        def oid(index: int) -> bytes:
            """ObjectID désignant une autre entrée."""

            return _field(6, index)

        def custom_map(type_index: int, pairs: List[tuple]) -> bytes:
            """Entrée custom_map (type + paires clé/ObjectID)."""

            body = _field(1, type_index) + b"".join(_field(3, _field(1, k) + _field(2, v)) for k, v in pairs)
            return _field(13, body)

        def dictionary(pairs: List[tuple]) -> bytes:
            """Message Dictionary (éléments clé/valeur)."""

            return b"".join(_field(1, _field(1, k) + _field(2, v)) for k, v in pairs)

        def ordered_set(positions: List[bytes], mapping: List[tuple]) -> bytes:
            """Entrée OrderedSet : ordre des positions puis association position → UUID réel."""

            array = _field(1, b"") + b"".join(
                _field(2, _field(1, i) + _field(2, uuid)) for i, uuid in enumerate(positions)
            )
            return _field(16, _field(1, _field(1, array) + _field(2, dictionary(mapping))))

        uuids = [b"rowpos0", b"rowpos1", b"colpos0", b"row0", b"row1", b"col0"]
        ref = {i: 7 + i for i in range(len(uuids))}  # entrée de référence par UUID
        entries = [
            custom_map(1, [(1, oid(1)), (2, oid(2)), (3, oid(3))]),
            ordered_set([uuids[0], uuids[1]], [(oid(ref[0]), oid(ref[3])), (oid(ref[1]), oid(ref[4]))]),
            ordered_set([uuids[2]], [(oid(ref[2]), oid(ref[5]))]),
            _field(6, dictionary([(oid(ref[5]), oid(4))])),
            _field(6, dictionary([(oid(ref[4]), oid(6)), (oid(ref[3]), oid(5))])),
            _field(10, _field(2, "Nom")),
            _field(10, _field(2, "Lait | bio")),
        ] + [custom_map(0, [(0, _field(2, i))]) for i in range(len(uuids))]
        data = b"".join(_field(3, entry) for entry in entries)
        data += b"".join(_field(4, key) for key in ["UUIDIndex", "crRows", "crColumns", "cellColumns"])
        data += b"".join(_field(5, kind) for kind in ["com.apple.CRDT.NSUUID", "com.apple.notes.ICTable"])
        data += b"".join(_field(6, uuid) for uuid in uuids)
        blob = gzip.compress(_field(2, _field(3, data)))

        self.assertEqual(bridge.table_to_markdown(blob), "| Nom |\n| --- |\n| Lait \\| bio |")

    def test_sqlite_uuid_dedupes_against_existing_import(self) -> None:
        """Une note déjà importée par obsidian-importer (même UUID) est ignorée."""

        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp) / "vault"
            (vault / "3 Ressource").mkdir(parents=True)
            (vault / "3 Ressource" / "Liste.md").write_text(
                "---\napple-notes-id: UUID-OK\n---\n# Liste\n", encoding="utf-8"
            )
            notes = bridge.export_notes_sqlite(self._make_db(Path(tmp)))
            result = bridge.sync(vault, dry_run=True, notes=notes)

        self.assertEqual(result["created"], [])
        self.assertEqual(result["skipped"], 1)


if __name__ == "__main__":
    unittest.main()
