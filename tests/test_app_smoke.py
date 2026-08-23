from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class AppSmokeTests(unittest.TestCase):
    def test_initial_view_is_minimal(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(app_path).run(timeout=15)

        self.assertEqual([title.value for title in app.title], ["Obsidian RAG"])
        self.assertEqual(len(app.chat_input), 1)
        self.assertEqual(len(app.button), 0)
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
