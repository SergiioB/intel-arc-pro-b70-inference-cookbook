#!/usr/bin/env python3
"""Verify every import in scripts/, tests/, and benchmarks/ is stdlib or repo-local.

The runtime contract is Python 3.11+ standard library only (pyproject.toml,
AGENTS.md). Genuine exceptions belong in THIRD_PARTY_ALLOWED with a written
reason; the allowlist is intentionally empty today.
"""

from __future__ import annotations

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCAN_DIRS = ("scripts", "tests", "benchmarks")
SKIP_DIRS = {"__pycache__"}
# name -> written reason. Keep this empty unless a dependency is unavoidable.
THIRD_PARTY_ALLOWED: dict[str, str] = {
    # Offline prompt-generation utility for benchmarks/b70-generate-exact-prompts.py;
    # runs on the lab host only and is not part of the stdlib-only serving runtime.
    "transformers": "offline prompt generator (benchmarks/), no runtime role",
}


def iter_python_files() -> list[pathlib.Path]:
    """Collect the .py files under the scanned dev directories."""
    files: list[pathlib.Path] = []
    for directory in SCAN_DIRS:
        base = ROOT / directory
        if not base.is_dir():
            continue
        files.extend(
            path for path in sorted(base.rglob("*.py")) if SKIP_DIRS.isdisjoint(path.parts)
        )
    return files


def is_repo_local(top_level: str) -> bool:
    """True when the module resolves inside the repository."""
    for base in [ROOT / directory for directory in SCAN_DIRS] + [ROOT]:
        if (base / f"{top_level}.py").is_file() or (base / top_level / "__init__.py").is_file():
            return True
    return False


def imported_modules(tree: ast.AST) -> list[tuple[int, str]]:
    """Yield (line, module) for absolute imports; relative imports are repo-local."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module))
    return found


def check_imports(sources: list[pathlib.Path]) -> list[str]:
    """Return one file:line finding per non-stdlib, non-repo-local import."""
    errors: list[str] = []
    for source in sources:
        display = source.resolve().relative_to(ROOT).as_posix()
        tree = ast.parse(source.read_text(errors="replace"), filename=str(source))
        for line_number, module in imported_modules(tree):
            top_level = module.split(".", 1)[0]
            if top_level in sys.stdlib_module_names or top_level in THIRD_PARTY_ALLOWED:
                continue
            if is_repo_local(top_level):
                continue
            errors.append(f"{display}:{line_number}: non-stdlib import: {module}")
    return errors


def main() -> int:
    errors = check_imports(iter_python_files())
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("OK: scripts/, tests/, benchmarks/ import stdlib or repo-local modules only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
