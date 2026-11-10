import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def load_checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "entry_points", ROOT / "scripts/check-entry-points.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = load_checker()


def make_fixture(directory: str, start: str | None = None) -> Path:
    root = Path(directory)
    (root / "data").mkdir()
    policy = json.loads((ROOT / "data/first-run.v1.json").read_text())
    (root / "data/first-run.v1.json").write_text(json.dumps(policy))
    route = root / policy["primary_route"]
    route.parent.mkdir(parents=True)
    route.write_text("# Reviewed route\n")
    (root / "README.md").write_text((ROOT / "README.md").read_text())
    (root / "START.md").write_text(start if start is not None else (ROOT / "START.md").read_text())
    return root


def start_text() -> str:
    return (ROOT / "START.md").read_text()


class EntryPointTests(unittest.TestCase):
    def test_current_selection_passes(self) -> None:
        self.assertEqual(checker.check(ROOT), [])

    def test_old_artifact_is_rejected_even_with_primary_link_present(self) -> None:
        start = start_text() + "\nSergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16\n"
        with tempfile.TemporaryDirectory() as directory:
            errors = checker.check(make_fixture(directory, start))
        self.assertTrue(any("excluded artifact" in error for error in errors))

    def test_command_drift_fails_instead_of_silently_promoting_speculation(self) -> None:
        start = start_text().replace(" --no-mtp", "")
        with tempfile.TemporaryDirectory() as directory:
            errors = checker.check(make_fixture(directory, start))
        self.assertTrue(any("disable the EXL3 drafter" in error for error in errors))

    def test_turning_prefix_cache_off_is_rejected(self) -> None:
        start = start_text().replace(" --no-mtp", " --no-mtp --cache off")
        with tempfile.TemporaryDirectory() as directory:
            errors = checker.check(make_fixture(directory, start))
        self.assertTrue(any("prefix caching on" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
