#!/usr/bin/env python3
"""Validate the evidence ledgers.

The rejected-arm ledger (`docs/DO-NOT-REPEAT.md`) is the only place this
repository records what did not work. An unreadable or half-empty row is worse
than no row, so this gate requires: a known ID format, unique IDs, every
required column filled, no placeholders, a real date, and evidence links that
resolve to files in this repository.

Exit code 1 with one line per problem. No GPU, no network, stdlib only.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

REJECTED_LEDGER = pathlib.Path("docs/DO-NOT-REPEAT.md")
RULES_DOC = pathlib.Path("docs/EVIDENCE-LEDGERS.md")
ACCEPTED_DOC = pathlib.Path("docs/WHAT-WORKED.md")
LEDGER_FILES = (REJECTED_LEDGER, RULES_DOC, ACCEPTED_DOC)

REQUIRED_COLUMNS = (
    "ID",
    "Arm",
    "Outcome",
    "Cause",
    "Scope",
    "Revisit when",
    "Evidence",
    "Date",
)

ID_PATTERN = re.compile(r"^DNR-\d{3}$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SEPARATOR_PATTERN = re.compile(r":?-{2,}:?")
LINK_PATTERN = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
PLACEHOLDERS = ("tbd", "todo", "fixme", "...", "n/a", "?")
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:")
UNPUBLISHED_NOTE = "unpublished"


def split_row(line: str) -> list[str]:
    """Split one markdown table row into stripped cells."""
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [cell.strip() for cell in body.split("|")]


def is_separator(cells: list[str]) -> bool:
    """True for the `|---|---|` line under a header."""
    return bool(cells) and all(SEPARATOR_PATTERN.fullmatch(cell.replace(" ", "")) for cell in cells)


def table_blocks(text: str) -> list[list[list[str]]]:
    """Return every markdown table in `text` as a list of split rows."""
    blocks: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in text.splitlines():
        if line.strip().startswith("|"):
            current.append(split_row(line))
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def find_ledger_table(text: str) -> tuple[list[str], list[list[str]]]:
    """Locate the header and data rows of the rejected-arm table."""
    for block in table_blocks(text):
        if not block:
            continue
        header = block[0]
        if header and header[0] == "ID":
            rows = [row for row in block[1:] if not is_separator(row)]
            return header, rows
    return [], []


def check_link(target: str, base: pathlib.Path, label: str) -> list[str]:
    """Verify one markdown link target resolves inside the repository."""
    clean = target.split("#", 1)[0].strip()
    if not clean or clean.startswith(EXTERNAL_PREFIXES):
        return []
    if (base / clean).resolve().exists():
        return []
    return [f"{label}: evidence link does not exist: {clean}"]


def check_cells(cells: list[str], index: dict[str, int], label: str) -> list[str]:
    """Reject empty cells, placeholder values, and a malformed date."""
    problems: list[str] = []
    if any(not cell for cell in cells):
        problems.append(f"{label}: empty cell in a required column")
    date = cells[index["Date"]]
    if date and not DATE_PATTERN.match(date):
        problems.append(f"{label}: date must be YYYY-MM-DD, found {date!r}")
    for column in REQUIRED_COLUMNS:
        if cells[index[column]].strip().lower() in PLACEHOLDERS:
            problems.append(f"{label}: placeholder value in {column!r}")
    return problems


def check_evidence(
    cells: list[str], index: dict[str, int], base: pathlib.Path, label: str
) -> list[str]:
    """Require real repository links, or an explicit unpublished-work note."""
    evidence = cells[index["Evidence"]]
    links = LINK_PATTERN.findall(evidence)
    problems: list[str] = []
    if not links and UNPUBLISHED_NOTE not in evidence.lower():
        problems.append(
            f"{label}: evidence needs a repository link or an explicit 'unpublished' note"
        )
    for link in links:
        problems.extend(check_link(link, base, label))
    return problems


def check_row(
    row: list[str],
    index: dict[str, int],
    width: int,
    base: pathlib.Path,
    row_number: int,
) -> tuple[list[str], str]:
    """Validate one data row. Returns its problems and its ID cell."""
    cells = (row + [""] * width)[:width]
    arm = cells[index["Arm"]] or f"row {row_number}"
    label = f"{REJECTED_LEDGER.as_posix()}:{row_number} ({arm[:48]})"
    identifier = cells[index["ID"]]
    problems: list[str] = []
    if not ID_PATTERN.match(identifier):
        problems.append(f"{label}: ID must match DNR-nnn, found {identifier!r}")
    problems.extend(check_cells(cells, index, label))
    problems.extend(check_evidence(cells, index, base, label))
    return problems, identifier


def check_rows(header: list[str], rows: list[list[str]], root: pathlib.Path) -> list[str]:
    """Validate every data row, including ID uniqueness."""
    index = {column: header.index(column) for column in REQUIRED_COLUMNS}
    base = (root / REJECTED_LEDGER).parent
    problems: list[str] = []
    if not rows:
        problems.append(f"{REJECTED_LEDGER.as_posix()}: table has no data rows")
    seen: dict[str, int] = {}
    for offset, row in enumerate(rows, start=2):
        row_problems, identifier = check_row(row, index, len(header), base, offset)
        problems.extend(row_problems)
        if identifier in seen:
            problems.append(
                f"{REJECTED_LEDGER.as_posix()}:{offset}: duplicate ID {identifier} "
                f"(first used on table row {seen[identifier]})"
            )
        else:
            seen[identifier] = offset
    return problems


def check_cross_references(root: pathlib.Path) -> list[str]:
    """The rulebook must point at both ledgers, and the index must point at the rulebook."""
    rules = (root / RULES_DOC).read_text(encoding="utf-8")
    accepted = (root / ACCEPTED_DOC).read_text(encoding="utf-8")
    problems = [
        f"{RULES_DOC.as_posix()}: does not reference {target.name}"
        for target in (REJECTED_LEDGER, ACCEPTED_DOC)
        if target.name not in rules
    ]
    if REJECTED_LEDGER.name not in accepted:
        problems.append(f"{ACCEPTED_DOC.as_posix()}: does not link to {REJECTED_LEDGER.name}")
    return problems


def validate(root: pathlib.Path = ROOT) -> list[str]:
    """Return every problem found in the ledgers; empty means valid."""
    missing = [
        f"missing ledger file: {item.as_posix()}"
        for item in LEDGER_FILES
        if not (root / item).exists()
    ]
    if missing:
        return missing

    text = (root / REJECTED_LEDGER).read_text(encoding="utf-8")
    header, rows = find_ledger_table(text)
    if not header:
        return [
            f"{REJECTED_LEDGER.as_posix()}: no rejected-arm table found "
            "(header must start with 'ID')"
        ]

    absent = [column for column in REQUIRED_COLUMNS if column not in header]
    if absent:
        return [f"{REJECTED_LEDGER.as_posix()}: table header missing columns: {', '.join(absent)}"]

    problems = check_rows(header, rows, root)
    return problems or check_cross_references(root)


def main() -> int:
    problems = validate()
    if not problems:
        print(f"ledgers: OK ({REJECTED_LEDGER.as_posix()})")
        return 0
    for problem in problems:
        print(f"ledgers: {problem}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
