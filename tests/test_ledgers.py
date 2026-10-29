import importlib.util
import pathlib
import sys
import tempfile
import textwrap
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


ledgers = load_checker("ledgers_checker", "check-ledgers.py")

HEADER = (
    "| ID | Arm | Outcome | Cause | Scope | Revisit when | Evidence | Date |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


def row(identifier="DNR-001", arm="probe arm", outcome="no gain", cause="cost exceeds split",
        scope="two cards", revisit="a cheaper gather exists", evidence="[what worked](WHAT-WORKED.md)",
        date="2026-10-29") -> str:
    return f"| {identifier} | {arm} | {outcome} | {cause} | {scope} | {revisit} | {evidence} | {date} |\n"


class LedgerGateTests(unittest.TestCase):
    def build(self, rows: str, extra_rules: bool = True, extra_index: bool = True) -> list[str]:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = pathlib.Path(directory)
            docs = root / "docs"
            docs.mkdir()
            (docs / "WHAT-WORKED.md").write_text(
                "# index\n" + ("DO-NOT-REPEAT.md\n" if extra_index else ""), encoding="utf-8"
            )
            (docs / "EVIDENCE-LEDGERS.md").write_text(
                "# rules\n"
                + ("DO-NOT-REPEAT.md\n" if extra_rules else "")
                + ("WHAT-WORKED.md\n" if extra_rules else ""),
                encoding="utf-8",
            )
            (docs / "DO-NOT-REPEAT.md").write_text("# ledger\n\n" + HEADER + rows, encoding="utf-8")
            return ledgers.validate(root)

    def test_valid_ledger_passes(self) -> None:
        self.assertEqual(self.build(row()), [])

    def test_unpublished_evidence_is_allowed(self) -> None:
        self.assertEqual(self.build(row(evidence="unpublished lab measurement; cause in source")), [])

    def test_missing_file_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            problems = ledgers.validate(pathlib.Path(directory))
        self.assertTrue(any("missing ledger file" in problem for problem in problems))

    def test_no_table_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = pathlib.Path(directory)
            docs = root / "docs"
            docs.mkdir()
            for name in ("WHAT-WORKED.md", "EVIDENCE-LEDGERS.md", "DO-NOT-REPEAT.md"):
                (docs / name).write_text("no tables here\n", encoding="utf-8")
            problems = ledgers.validate(root)
        self.assertTrue(any("no rejected-arm table" in problem for problem in problems))

    def test_missing_column_fails(self) -> None:
        problems = self.build(row())
        self.assertEqual(problems, [])
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = pathlib.Path(directory)
            docs = root / "docs"
            docs.mkdir()
            for name in ("WHAT-WORKED.md", "EVIDENCE-LEDGERS.md"):
                (docs / name).write_text("DO-NOT-REPEAT.md WHAT-WORKED.md\n", encoding="utf-8")
            (docs / "DO-NOT-REPEAT.md").write_text(
                "# ledger\n\n| ID | Arm | Outcome |\n|---|---|---|\n| DNR-001 | arm | loss |\n", encoding="utf-8"
            )
            problems = ledgers.validate(root)
        self.assertTrue(any("missing columns" in problem for problem in problems))

    def test_bad_id_fails(self) -> None:
        problems = self.build(row(identifier="DNR-1"))
        self.assertTrue(any("ID must match DNR-nnn" in problem for problem in problems))

    def test_duplicate_id_fails(self) -> None:
        problems = self.build(row() + row())
        self.assertTrue(any("duplicate ID" in problem for problem in problems))

    def test_empty_cell_fails(self) -> None:
        problems = self.build(row(cause=""))
        self.assertTrue(any("empty cell" in problem for problem in problems))

    def test_placeholder_fails(self) -> None:
        problems = self.build(row(revisit="TBD"))
        self.assertTrue(any("placeholder" in problem for problem in problems))

    def test_bad_date_fails(self) -> None:
        problems = self.build(row(date="29-10-2026"))
        self.assertTrue(any("date must be YYYY-MM-DD" in problem for problem in problems))

    def test_dangling_evidence_link_fails(self) -> None:
        problems = self.build(row(evidence="[gone](DOES-NOT-EXIST.md)"))
        self.assertTrue(any("evidence link does not exist" in problem for problem in problems))

    def test_external_link_is_not_checked(self) -> None:
        self.assertEqual(self.build(row(evidence="[upstream](https://github.com/ggml-org/llama.cpp/pull/1)")), [])

    def test_evidence_needs_link_or_note(self) -> None:
        problems = self.build(row(evidence="we measured it once"))
        self.assertTrue(any("evidence needs a repository link" in problem for problem in problems))

    def test_missing_cross_reference_fails(self) -> None:
        problems = self.build(row(), extra_index=False)
        self.assertTrue(any("does not link to DO-NOT-REPEAT.md" in problem for problem in problems))

    def test_repository_ledgers_are_valid(self) -> None:
        self.assertEqual(ledgers.validate(ROOT), [])


class SplitRowTests(unittest.TestCase):
    def test_cells_are_stripped(self) -> None:
        self.assertEqual(ledgers.split_row("| a | b |"), ["a", "b"])

    def test_separator_detected(self) -> None:
        self.assertTrue(ledgers.is_separator(ledgers.split_row("|---|---:|")))
        self.assertFalse(ledgers.is_separator(ledgers.split_row("| DNR-001 | x |")))

    def test_pipes_inside_code_split_cells(self) -> None:
        # An unescaped pipe inside a cell splits it, so a row with one produces too many
        # cells and fails the empty-cell check. Authors must avoid or escape such pipes.
        self.assertEqual(ledgers.split_row("| a `x|y` b |"), ["a `x", "y` b"])


if __name__ == "__main__":
    unittest.main()
