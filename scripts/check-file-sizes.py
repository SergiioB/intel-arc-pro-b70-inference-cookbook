#!/usr/bin/env python3
"""Flag oversized tracked files and over-long Python dev files.

Two gates: no tracked file above 1 MiB except generated assets under
docs/assets/ and results/, and no .py under scripts/, tests/, benchmarks/
above 1200 lines.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAX_BYTES = 1024 * 1024
MAX_LINES = 1200
SIZE_EXEMPT_PREFIXES = ("docs/assets/", "results/")
LINE_COUNT_PREFIXES = ("scripts/", "tests/", "benchmarks/")


def repository_files() -> list[pathlib.Path]:
    """List tracked plus untracked non-ignored repository files."""
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [ROOT / line for line in completed.stdout.splitlines() if line]


def display_path(path: pathlib.Path) -> str:
    """Repo-relative path when possible, plain path otherwise (for tmp tests)."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def size_offenders(entries: list[tuple[pathlib.Path, int]]) -> list[str]:
    """Report entries above MAX_BYTES that are not in the exempt directories."""
    offenders: list[str] = []
    for path, size in entries:
        display = display_path(path)
        if display.startswith(SIZE_EXEMPT_PREFIXES):
            continue
        if size > MAX_BYTES:
            offenders.append(f"{display}: {size} bytes exceeds the 1 MiB limit")
    return offenders


def count_lines(path: pathlib.Path) -> int:
    return len(path.read_text(errors="replace").splitlines())


def line_count_offenders(entries: list[tuple[pathlib.Path, int]]) -> list[str]:
    """Report .py entries under scripts/, tests/, benchmarks/ above MAX_LINES."""
    offenders: list[str] = []
    for path, lines in entries:
        display = display_path(path)
        if not display.endswith(".py") or not display.startswith(LINE_COUNT_PREFIXES):
            continue
        if lines > MAX_LINES:
            offenders.append(f"{display}: {lines} lines exceeds the {MAX_LINES}-line limit")
    return offenders


def main() -> int:
    files = repository_files()
    sizes = [(path, path.stat().st_size) for path in files if path.is_file()]
    counts: list[tuple[pathlib.Path, int]] = []
    for path in files:
        display = display_path(path)
        if display.endswith(".py") and display.startswith(LINE_COUNT_PREFIXES) and path.is_file():
            counts.append((path, count_lines(path)))
    offenders = size_offenders(sizes) + line_count_offenders(counts)
    if offenders:
        for offender in offenders:
            print(f"ERROR: {offender}", file=sys.stderr)
        return 1
    print(f"OK: file sizes and line counts ({len(files)} repository files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
