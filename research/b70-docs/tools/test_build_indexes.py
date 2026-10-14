#!/usr/bin/env python3
"""Stdlib catalog regressions. Run: python3 tools/test_build_indexes.py"""
from __future__ import annotations

import contextlib
import datetime as dt
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "build_indexes", Path(__file__).with_name("build-indexes.py")
)
assert SPEC is not None and SPEC.loader is not None
INDEXES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INDEXES)


class CatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.root_patch = patch.object(INDEXES, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def write_source(self, path: str, text: str) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def render(self, *, check: bool = False, day: int = 1) -> tuple[int, str]:
        class Clock(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 10, day, tzinfo=tz)

        output = io.StringIO()
        with patch.object(INDEXES, "CHECK_MODE", check), \
                patch.object(INDEXES.dt, "datetime", Clock), \
                contextlib.redirect_stdout(output):
            status = INDEXES.main()
        return status, output.getvalue()

    def generated_bytes(self) -> dict[str, bytes]:
        paths = [self.root / "registry/index.json", self.root / "DOCUMENT-MAP.md"]
        paths.extend(sorted((self.root / "docs/indexes").glob("*.md")))
        return {path.relative_to(self.root).as_posix(): path.read_bytes()
                for path in paths}

    def catalog(self) -> dict[str, dict]:
        data = json.loads((self.root / "registry/index.json").read_text(encoding="utf-8"))
        return {row["path"]: row for row in data["documents"]}

    def test_regeneration_and_check_are_stable_but_detect_source_changes(self) -> None:
        source = "research/2026/late-september-20260930.md"
        self.write_source(source, "---\ndate: 2026-09-30\n---\n\n# Original question\n")
        status, output = self.render(day=1)
        self.assertEqual(status, 0, output)
        first = self.generated_bytes()
        status, output = self.render(day=2)
        self.assertEqual(status, 0, output)
        self.assertEqual(self.generated_bytes(), first)
        status, output = self.render(check=True, day=3)
        self.assertEqual(status, 0, output)

        self.write_source(source, "---\ndate: 2026-09-30\n---\n\n# Revised question\n")
        status, output = self.render(check=True, day=4)
        self.assertEqual(status, 1, output)
        self.assertIn("DRIFT registry/index.json", output)
        self.assertIn("DRIFT docs/indexes/research-index.md", output)
        self.assertEqual(self.generated_bytes(), first, "check must not rewrite outputs")
        status, output = self.render(day=4)
        self.assertEqual(status, 0, output)
        self.assertEqual(self.catalog()[source]["title"], "Revised question")
        status, output = self.render(check=True, day=5)
        self.assertEqual(status, 0, output)

    def test_generated_and_conflict_views_do_not_hide_historical_evidence(self) -> None:
        sources = {
            "research/2026/late-september-20260930.md": (
                "---\ndate: 2026-09-30\n---\n\n<!-- provenance -->\n\n"
                "```markdown\n# Example, not the title\n```\n\n# Late September probe ###\n"
            ),
            "results/2026/09/probe-20260930/README.md": "# Raw probe evidence\n",
            "archive/legacy-20260801/docs/indexes/research-index.md": "# Frozen historical index\n",
            "research/archive/older-20250801.md": "# Older research\n",
        }
        excluded = [
            "docs/indexes/research-index.md",
            "docs/indexes/situation-latest.md",
            "docs/indexes/artifact-index.md",
            "research/PROGRAM.md",
            "research/LEDGER.md",
            "models/STATUS.md",
            "docs/indexes/research-coverage.md",
            "research/2026/probe.sync-conflict-20261001-120000-ABC.md",
            "results/2026/09/probe.sync-conflict-20261001-ABC/README.md",
            "results/2026/09/.artifact-scaffold/README.md",
            "research/.draft.md",
            "research/node_modules/example/README.md",
        ]
        for path, text in sources.items():
            self.write_source(path, text)
        for path in excluded:
            self.write_source(path, "# Derived or duplicate view\n")
        status, output = self.render()
        self.assertEqual(status, 0, output)
        catalog = self.catalog()
        self.assertEqual(set(catalog), set(sources))
        late = catalog["research/2026/late-september-20260930.md"]
        self.assertEqual(late["title"], "Late September probe")
        self.assertEqual(late["date"], "2026-09-30")

        added = "research/2026/newly-discovered-20260929.md"
        self.write_source(added, "# Newly discovered September evidence\n")
        status, output = self.render(check=True)
        self.assertEqual(status, 1, output)
        status, output = self.render()
        self.assertEqual(status, 0, output)
        self.assertEqual(set(self.catalog()), set(sources) | {added})
        self.assertEqual(self.catalog()[added]["date"], "2026-09-29")
        research_view = (self.root / "docs/indexes/research-index.md").read_text(encoding="utf-8")
        self.assertIn(added, research_view)
        self.assertIn("research/2026/late-september-20260930.md", research_view)
        status, output = self.render(check=True, day=2)
        self.assertEqual(status, 0, output)


if __name__ == "__main__":
    unittest.main()
