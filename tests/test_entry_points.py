import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "entry_points", ROOT / "scripts/check-entry-points.py"
)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class EntryPointTests(unittest.TestCase):
    def fixture(self, directory, start=None):
        root = Path(directory)
        (root / "data").mkdir()
        policy = json.loads((ROOT / "data/first-run.v1.json").read_text())
        (root / "data/first-run.v1.json").write_text(json.dumps(policy))
        route = root / policy["primary_route"]
        route.parent.mkdir(parents=True)
        route.write_text("# Reviewed route\n")
        for name in ("README.md", "START.md"):
            text = (ROOT / name).read_text()
            (root / name).write_text(start if name == "START.md" and start is not None else text)
        return root

    def test_current_selection_passes(self):
        self.assertEqual(checker.check(ROOT), [])

    def test_old_artifact_is_rejected_even_with_primary_link_present(self):
        start = (
            ROOT / "START.md"
        ).read_text() + "\nSergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16\n"
        with tempfile.TemporaryDirectory() as directory:
            errors = checker.check(self.fixture(directory, start))
        self.assertTrue(any("excluded artifact" in error for error in errors))

    def test_command_drift_fails_instead_of_silently_promoting_speculation(self):
        start = (
            (ROOT / "START.md")
            .read_text()
            .replace("vllm.speculative_config=null", "vllm.speculative_config=mtp")
        )
        with tempfile.TemporaryDirectory() as directory:
            errors = checker.check(self.fixture(directory, start))
        self.assertTrue(any("disable the EXL3 drafter" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
