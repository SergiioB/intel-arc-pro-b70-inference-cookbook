#!/usr/bin/env python3
"""Fail on bare work-item markers that lack an issue reference.

A marker is acceptable only in the form MARKER(#<issue number>), e.g.
``MARKER(#12)`` for GitHub issue 12. Everything else is reported as
file:line. Marker literals are assembled at runtime so this checker never
carries a bare one itself.

patches/ is excluded by design: those files are SHA256-pinned verbatim
sources (docs/FULL-SETUP-COMMANDS.md sections 5 and 11); any edit changes
the pinned hashes and starts a new result generation.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCANNED_SUFFIXES = (".py", ".md", ".js", ".sh")
# The classic three work-marker literals, spelled split so this file stays
# clean under its own gate.
MARKERS = ("TO" + "DO", "FIX" + "ME", "XX" + "X")
MARKER_RE = re.compile(rf"\b({'|'.join(MARKERS)})\b(?!\(#\d+\))")
EXCLUDED_PREFIXES = ("patches/",)


def repository_files() -> list[pathlib.Path]:
    """List candidate files from the git index plus untracked non-ignored files."""
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [
        ROOT / line
        for line in completed.stdout.splitlines()
        if line.endswith(SCANNED_SUFFIXES) and not line.startswith(EXCLUDED_PREFIXES)
    ]


def display_path(path: pathlib.Path) -> str:
    """Repo-relative path when possible, plain path otherwise (for tmp tests)."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def scan_markers(files: list[pathlib.Path]) -> list[str]:
    """Return one file:line finding per bare marker occurrence."""
    findings: list[str] = []
    for source in files:
        display = display_path(source)
        for line_number, line in enumerate(source.read_text(errors="replace").splitlines(), 1):
            match = MARKER_RE.search(line)
            if match:
                findings.append(f"{display}:{line_number}: bare {match.group(1)} marker")
    return findings


def main() -> int:
    findings = scan_markers(repository_files())
    if findings:
        for finding in findings:
            print(f"ERROR: {finding}", file=sys.stderr)
        return 1
    print("OK: no bare work-item markers outside hash-pinned patches/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
