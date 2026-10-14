#!/usr/bin/env python3
"""check-frontmatter.py — enforce the front-matter contract on living documents.

Every non-generated .md in the enforced set must carry front-matter with at
least: title, date (YYYY-MM-DD), type, status (living|final|provisional|superseded).
Dated evidence under results/ is exempt (its date lives in the path); research
notes must carry at least date+type (auto-backfilled by --fix from the filename).

Actions:
  --check (default): list violations, exit 1 if any (CI gate).
  --fix: insert minimal front-matter derived from filename/heading/path into
         violating files (never overwrites an existing block).

Example:
  python3 tools/check-frontmatter.py          # gate
  python3 tools/check-frontmatter.py --fix    # mechanical backfill
"""
from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENFORCED = [  # (glob, required keys)
    ("docs/**/*.md", {"title", "date", "type", "status"}),
    ("engines/**/*.md", {"title", "date", "type", "status"}),
    ("models/**/*.md", {"title", "date", "type", "status"}),
    ("decisions/**/*.md", {"title", "date", "type", "status"}),
    ("campaigns/**/*.md", {"title", "date", "type", "status"}),
    ("research/2026/**/*.md", {"title", "date", "type"}),
]
SKIP = {"docs/indexes",}  # generated views
TYPES = {"docs/methods/": "method", "docs/governance/": "governance",
         "docs/operations/": "operation", "docs/reference/": "reference",
         "docs/plans/": "plan", "engines/": "engine-dossier",
         "models/": "model-dossier", "decisions/": "adr",
         "campaigns/": "campaign", "research/2026/": "research"}
DATE_IN_NAME = re.compile(r"(20\d{2})(\d{2})(\d{2})")


def parse_fm(text: str) -> dict:
    if not text.startswith("---"):
        return {}
    out = {}
    for line in text.split("\n")[1:]:
        if line.strip() == "---":
            return out
        m = re.match(r"^([a-z_]+):\s*(.*)$", line.strip())
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def title_of(text: str) -> str:
    for line in text.split("\n"):
        if line.startswith("# "):
            return line[2:].strip()[:100]
    return ""


def targets() -> list[tuple[Path, set]]:
    seen = {}
    for glob, req in ENFORCED:
        for p in ROOT.glob(glob):
            if not p.is_file() or any(s in p.as_posix() for s in SKIP):
                continue
            if p.name.upper() == "README.md" and p.parent == ROOT:
                continue
            seen[p] = seen.get(p, set()) | req
    return sorted(seen.items())


def main() -> int:
    fix = "--fix" in sys.argv
    violations = 0
    for path, req in targets():
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        fm = parse_fm(text)
        missing = req - set(fm)
        if not missing:
            continue
        if not fix:
            print(f"MISSING {sorted(missing)} -> {rel}")
            violations += 1
            continue
        doctype = next((t for p, t in TYPES.items() if rel.startswith(p)), "doc")
        m = DATE_IN_NAME.search(path.name) or DATE_IN_NAME.search(rel)
        date = (f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else
                dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)
                .strftime("%Y-%m-%d"))
        title = fm.get("title") or title_of(text) or rel
        status = "living" if doctype.endswith(("dossier", "method", "governance",
                                               "reference", "operation")) else "final"
        block = [f"title: \"{title}\"", f"type: {doctype}", f"date: {date}",
                 f"status: {status}"]
        block = [k for k in block if not k.split(":")[0] in fm]
        if block:
            path.write_text("---\n" + "\n".join(block) + "\n---\n\n" + text,
                            encoding="utf-8")
            print(f"fixed {rel} (+{len(block)} keys)")
    if violations and not fix:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
