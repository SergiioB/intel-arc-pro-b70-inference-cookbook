#!/usr/bin/env python3
"""check-links.py — verify every relative link/pointer in the active tree resolves.

Scans .md, .json, .yaml, .py under the active tree. Checks:
  - markdown links [x](relative/path) and inline `path/like/this` pointers
    starting with a known top-level dir.

Exempt: archive/ and results/ (frozen evidence — internal path mentions are
historical record; living registries carry current paths) and research/2026
(dated notes; their repo-prefix paths were mechanically updated 2026-09-15).
node_modules and generated indexes are skipped.

Actions: default run prints violations; exit 1 if any. --summary prints counts only.

Example:
  python3 tools/check-links.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACTIVE = ["docs", "engines", "models", "campaigns", "decisions",
          "registry", "tools", "scripts"]
ROOT_FILES = ["AGENTS.md", "README.md", "DOCUMENT-MAP.md"]
TOP_DIRS = {"docs", "engines", "models", "campaigns", "decisions",
            "registry", "tools", "scripts", "research", "results", "archive"}
MD_LINK = re.compile(r"\]\(([^)\s]+)\)")
TICKED = re.compile(r"`((?:%s)/[^`\s]+)`" % "|".join(sorted(TOP_DIRS)))


def extract(text: str) -> list[str]:
    out = set()
    for m in MD_LINK.finditer(text):
        out.add(m.group(1))
    for m in TICKED.finditer(text):
        out.add(m.group(1))
    return [p for p in out if not p.startswith(("http://", "https://", "#", "mailto:"))
            and not any(c in p for c in "<>*…{")
            and p != "relative/path"
            and not p.endswith("/MM")]


def main() -> int:
    summary = "--summary" in sys.argv
    files, bad = 0, []
    candidates = [ROOT / f for f in ROOT_FILES if (ROOT / f).exists()]
    for d in ACTIVE:
        base = ROOT / d
        if not base.exists():
            continue
        for ext in ("*.md", "*.json", "*.yaml", "*.py"):
            candidates += [p for p in base.rglob(ext)
                           if "node_modules" not in p.parts and ".git" not in p.parts]
    for p in candidates:
        text = p.read_text(encoding="utf-8", errors="replace")
        files += 1
        for ref in extract(text):
            ref = ref.split("#")[0].rstrip("/")
            # citations may carry a line suffix ("path.md:12" / "path.md:70-76") or
            # name another repo ("cookbook/docs/..."); both are valid prose, not links
            ref = re.sub(r":\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*$", "", ref)
            if not ref:
                continue
            if ref.startswith(("cookbook/", "B70-DOCS/", "../")):
                continue
            target = (p.parent / ref).resolve() if not ref.startswith(tuple(TOP_DIRS)) \
                else (ROOT / ref).resolve()
            if not target.exists():
                # allow file-vs-dir and extension-less doc refs
                if not (target.with_suffix(".md").exists() or (target / "README.md").exists()):
                    bad.append((p.relative_to(ROOT).as_posix(), ref))
    if bad:
        if not summary:
            for src, ref in bad[:200]:
                print(f"BROKEN {ref}  (in {src})")
        print(f"FAIL: {len(bad)} broken references across {files} files")
        return 1
    print(f"OK: all relative references resolve ({files} files scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
