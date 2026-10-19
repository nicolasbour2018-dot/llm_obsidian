from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from chat_backend import ChatBackendError


class AppSmokeTests(unittest.TestCase):
    """Contrôle que l'interface démarre sans dépendre d'un appel réseau."""

    def test_initial_view(self) -> None:
        """La vue initiale affiche le titre, les onglets et le champ de conversation."""

        app_path = Path(__file__).resolve().parents[1] / "app.py"
        with ExitStack() as stack, tempfile.TemporaryDirectory() as brain_dir:
            # Moteur en échec et pont injoignable : l'application doit tout de même démarrer.
            stack.enter_context(patch.dict(os.environ, {"BRAIN_DIR": brain_dir}))
            stack.enter_context(patch("chat_backend.stats", side_effect=ChatBackendError("Index absent")))
            if importlib.util.find_spec("apple_notes_client") is not None:
                stack.enter_context(patch("apple_notes_client.bridge_status", return_value=None))
            app = AppTest.from_file(app_path).run(timeout=15)

        self.assertEqual([title.value for title in app.title], ["🧠 Second Cerveau"])
        self.assertEqual(len(app.chat_input), 1)
        self.assertEqual([tab.label for tab in app.tabs], ["💬 Chat", "🔎 Recherche", "🎲 Redécouvrir"])
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
