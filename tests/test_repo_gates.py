import importlib.util
import pathlib
import sys
import tempfile
import unittest
from types import ModuleType

ROOT = pathlib.Path(__file__).parents[1]


def load_checker(module_name: str, filename: str) -> ModuleType:
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


todos = load_checker("todos_checker", "check-todos.py")
stdlib_only = load_checker("stdlib_only_checker", "check-stdlib-only.py")
file_sizes = load_checker("file_sizes_checker", "check-file-sizes.py")

BARE = "TO" + "DO"  # spelled split so this test file never trips the marker gate
ISSUE_REF = "TO" + "DO(#12)"


class TodosGateTests(unittest.TestCase):
    def scan(self, text: str) -> list[str]:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source = pathlib.Path(directory) / "notes.md"
            source.write_text(text)
            findings = todos.scan_markers([source])
        assert isinstance(findings, list)
        return findings

    def test_clean_content_passes(self) -> None:
        self.assertEqual(self.scan("nothing to see here\n"), [])

    def test_issue_referenced_marker_passes(self) -> None:
        self.assertEqual(self.scan(f"tracked in {ISSUE_REF}\n"), [])

    def test_bare_marker_fails_with_file_and_line(self) -> None:
        findings = self.scan(f"# {BARE} rework this loop\n")
        self.assertEqual(len(findings), 1)
        self.assertIn(f":1: bare {BARE} marker", findings[0])


class StdlibOnlyGateTests(unittest.TestCase):
    def check(self, text: str) -> list[str]:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source = pathlib.Path(directory) / "sample.py"
            source.write_text(text)
            errors = stdlib_only.check_imports([source])
        assert isinstance(errors, list)
        return errors

    def test_stdlib_and_repo_local_imports_pass(self) -> None:
        self.assertEqual(
            self.check("import json\nfrom collections.abc import Sequence\nimport cookbook_log\n"),
            [],
        )

    def test_unknown_top_level_import_fails_with_module_name(self) -> None:
        errors = self.check("import json\nimport not_a_real_package\n")
        self.assertEqual(len(errors), 1)
        self.assertIn(":2: non-stdlib import: not_a_real_package", errors[0])


class FileSizesGateTests(unittest.TestCase):
    def test_small_file_passes(self) -> None:
        errors = file_sizes.size_offenders([(ROOT / "README.md", 1024)])
        assert isinstance(errors, list)
        self.assertEqual(errors, [])

    def test_oversized_file_is_reported_and_assets_are_exempt(self) -> None:
        errors = file_sizes.size_offenders([(ROOT / "big.bin", 2 * 1024 * 1024)])
        assert isinstance(errors, list)
        self.assertEqual(len(errors), 1)
        self.assertIn("big.bin: 2097152 bytes exceeds the 1 MiB limit", errors[0])
        exempt = file_sizes.size_offenders(
            [(ROOT / "docs" / "assets" / "big.svg", 2 * 1024 * 1024)]
        )
        assert isinstance(exempt, list)
        self.assertEqual(exempt, [])

    def test_line_counting_works_on_tmp_files(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source = pathlib.Path(directory) / "long.py"
            source.write_text("\n" * 1300)
            self.assertEqual(file_sizes.count_lines(source), 1300)

    def test_overlong_python_file_is_reported(self) -> None:
        errors = file_sizes.line_count_offenders([(ROOT / "scripts" / "long.py", 1300)])
        assert isinstance(errors, list)
        self.assertEqual(len(errors), 1)
        self.assertIn("1300 lines exceeds the 1200-line limit", errors[0])

    def test_python_outside_scanned_directories_is_skipped(self) -> None:
        errors = file_sizes.line_count_offenders([(ROOT / "docs" / "long.py", 5000)])
        assert isinstance(errors, list)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
