import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "site_feeds", ROOT / "scripts/build-site-feeds.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


feeds = load()


def make_repo(directory: str, routes: dict[str, object] | None) -> Path:
    root = Path(directory)
    (root / "docs").mkdir()
    (root / "data").mkdir()
    (root / "README.md").write_text("# Readme\n")
    (root / "START.md").write_text("# Start\n")
    (root / "docs/old.md").write_text("# Old\n\n> Status: SUPERSEDED 2026-09-27, see new.md\n")
    (root / "docs/new.md").write_text("# New\n")
    catalog = {"schema_version": "1.0", "records": [{"id": "rec-1"}]}
    (root / "data/benchmarks.v1.json").write_text(json.dumps(catalog))
    if routes is not None:
        (root / "data/routes.v1.json").write_text(json.dumps(routes))
    return root


def route(**over: object) -> dict[str, object]:
    base = {
        "id": "r1",
        "status": "recommended",
        "cards": 1,
        "use_cases": ["chat"],
        "records": ["rec-1"],
        "doc": "docs/new.md",
        "featured": True,
        "sampling": {"presets": [], "source": "https://huggingface.co/x/y"},
    }
    base.update(over)
    return base


class SiteFeedTests(unittest.TestCase):
    def test_banner_marks_doc_historical(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            corpus = feeds.build_corpus(make_repo(d, None))
        status = {doc["path"]: doc["status"] for doc in corpus["docs"]}
        self.assertEqual(status["docs/old.md"], "historical")
        self.assertEqual(status["docs/new.md"], "current")

    def test_valid_routes_pass(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(
                feeds.check_routes(make_repo(d, {"schema_version": "1.0", "routes": [route()]})), []
            )

    def test_missing_record_and_doc_fail(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            errors = feeds.check_routes(
                make_repo(
                    d,
                    {"schema_version": "1.0", "routes": [route(records=["nope"], doc="docs/x.md")]},
                )
            )
        self.assertTrue(any("missing catalog record" in e for e in errors))
        self.assertTrue(any("does not exist" in e for e in errors))

    def test_sampling_needs_model_card_source(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            errors = feeds.check_routes(
                make_repo(
                    d,
                    {
                        "schema_version": "1.0",
                        "routes": [route(sampling={"presets": [], "source": "x"})],
                    },
                )
            )
        self.assertTrue(any("model card source" in e for e in errors))

    def test_exactly_one_featured(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            errors = feeds.check_routes(
                make_repo(d, {"schema_version": "1.0", "routes": [route(featured=False)]})
            )
        self.assertTrue(any("exactly one featured" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
