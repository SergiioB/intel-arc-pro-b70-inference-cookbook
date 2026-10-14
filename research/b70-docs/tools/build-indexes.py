#!/usr/bin/env python3
"""build-indexes.py — regenerate every derived index from document front-matter.

Actions:
  - Scans every .md under the active tree (docs/, engines/, models/,
    campaigns/, results/, research/, decisions/, archive/, root README/AGENTS).
  - Reads YAML-ish front-matter (flat keys only); falls back to heuristics:
    filename date (YYYYMMDD), campaign-dir date, section default.
  - Writes registry/index.json (full machine catalog), docs/indexes/*.md
    (research/campaign/decision/engine/doc views), and DOCUMENT-MAP.md.
  - Excludes generated views, sync-conflict copies, and hidden scaffolds;
    historical research and raw-result Markdown remain catalogued.

Output: derived files only; never edits source documents. Deterministic.
Index staleness is a defect: if a document is not listed where you expect,
add front-matter (see tools/check-frontmatter.py) — do not hand-edit these
indexes.

Example:
  python3 tools/build-indexes.py            # regenerate everything
  python3 tools/build-indexes.py --check    # CI gate: exit 1 if drift
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ["docs", "engines", "models", "campaigns", "results", "research",
             "decisions", "archive"]
ROOT_FILES = ["README.md", "AGENTS.md"]
# docs/indexes is the generated-navigation namespace, not source evidence.
GENERATED_DIRS = {"docs/indexes"}
GENERATED_FILES = {"research/PROGRAM.md", "research/LEDGER.md", "models/STATUS.md"}
FM_KEYS = {"title", "type", "status", "date", "updated", "engine", "model",
           "evidence"}
FM_RE = re.compile(r"^([a-z_]+):\s*(.+)$")
DATE_IN_NAME = re.compile(r"(20\d{2})(\d{2})(\d{2})")
ISO_DATE = re.compile(r"^(20\d{2})-(\d{2})-(\d{2})$")

SECTION_TYPE = {  # path-prefix -> default doc type
    "decisions/": "adr", "campaigns/": "campaign", "results/": "campaign-evidence",
    "research/": "research", "docs/methods/": "method",
    "docs/governance/": "governance", "docs/operations/": "operation",
    "docs/reference/": "reference", "docs/plans/": "plan",
    "docs/indexes/": "index", "engines/": "engine-dossier",
    "models/": "model-dossier", "archive/": "archived",
}
TYPE_ORDER = ["living", "final", "provisional", "superseded", "unknown"]


def parse_front_matter(text: str) -> dict:
    if not text.startswith("---"):
        return {}
    out = {}
    for line in text.split("\n")[1:]:
        if line.strip() == "---":
            break
        m = FM_RE.match(line.strip())
        if m and m.group(1) in FM_KEYS:
            out[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return out


def norm_date(raw: str | None) -> str | None:
    if not raw:
        return None
    m = ISO_DATE.match(raw.strip()[:10])
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def heading_title(text: str) -> str | None:
    """Find a Markdown heading, not a YAML delimiter or fenced example."""
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        end = next((i for i, line in enumerate(lines[1:], 1)
                    if line.strip() == "---"), None)
        if end is not None:
            lines = lines[end + 1:]
    fence = None
    for line in lines:
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if marker:
            run = marker.group(1)
            if fence is None:
                fence = run
            elif run[0] == fence[0] and len(run) >= len(fence):
                fence = None
            continue
        if fence is None:
            match = re.match(r"^ {0,3}#{1,6}[ \t]+(.+?)\s*$", line)
            if match:
                title = re.sub(r"[ \t]+#+$", "", match.group(1)).strip()
                if title:
                    return title[:100]
    return None


def is_catalog_source(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part.startswith(".") or part == "node_modules"
           or ".sync-conflict-" in part for part in rel.parts):
        return False
    return (rel.as_posix() not in GENERATED_FILES
            and not any(parent.as_posix() in GENERATED_DIRS
                        for parent in rel.parents))


def derive(path: Path) -> dict:
    rel = path.relative_to(ROOT).as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    fm = parse_front_matter(text)
    doctype = fm.get("type") or next(
        (t for p, t in SECTION_TYPE.items() if rel.startswith(p)), "doc")
    status = fm.get("status") or ("living" if doctype.endswith(("dossier", "method", "governance", "reference", "operation", "engine-dossier"))
                                  else "final" if "archive/" in rel else "unknown")
    date = norm_date(fm.get("date"))
    if not date:
        m = DATE_IN_NAME.search(path.name) or DATE_IN_NAME.search(rel)
        if m:
            date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    if not date and rel.startswith("results/2026/"):
        m = DATE_IN_NAME.search(rel)
        if m:
            date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    if not date:
        stamp = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)
        date = stamp.strftime("%Y-%m-%d")
    title = fm.get("title") or heading_title(text) or rel
    engines = [e.strip() for e in fm.get("engine", "").split(",") if e.strip()]
    return {"path": rel, "title": title, "type": doctype, "status": status,
            "date": date, "updated": norm_date(fm.get("updated")),
            "engines": engines, "model": fm.get("model") or None,
            "evidence": fm.get("evidence") or None}


def scan() -> list[dict]:
    docs = []
    for name in ROOT_FILES:
        p = ROOT / name
        if p.is_file() and is_catalog_source(p):
            docs.append(derive(p))
    for d in SCAN_DIRS:
        for p in sorted((ROOT / d).rglob("*.md")):
            if not p.is_file() or not is_catalog_source(p):
                continue
            docs.append(derive(p))
    return docs


def table(rows: list[dict]) -> str:
    out = ["| date | status | title | path |", "|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["date"], r["path"]), reverse=True):
        out.append(f"| {r['date']} | {r['status']} | {r['title'][:80]} | `{r['path']}` |")
    return "\n".join(out)


CHECK_MODE = "--check" in sys.argv


def write(path: Path, text: str) -> bool:
    new = text if text.endswith("\n") else text + "\n"
    old = path.read_text(encoding="utf-8") if path.exists() else None
    if CHECK_MODE:
        return old != new
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new, encoding="utf-8")
    return old != new


def main() -> int:
    check = CHECK_MODE
    docs = scan()
    by_type: dict[str, list[dict]] = {}
    for d in docs:
        by_type.setdefault(d["type"], []).append(d)

    changed = []
    idx = ROOT / "docs/indexes"
    views = {
        "research-index.md": ("Research notes (research/)", by_type.get("research", [])),
        "campaign-index.md": ("Campaigns (campaigns/ + results/)",
                              by_type.get("campaign", []) + by_type.get("campaign-evidence", [])),
        "decision-index.md": ("Architecture decisions (decisions/)", by_type.get("adr", [])),
        "engine-index.md": ("Engine dossiers (engines/)", by_type.get("engine-dossier", [])),
        "model-index.md": ("Model dossiers (models/)", by_type.get("model-dossier", [])),
        "doc-index.md": ("Methods, governance, operations, reference, plans",
                         [d for d in docs if d["type"] in
                          ("method", "governance", "operation", "reference", "plan")]),
        "archive-index.md": ("Archived (frozen eras — exempt from link checks)",
                             by_type.get("archived", [])),
    }
    for name, (heading, rows) in views.items():
        body = f"<!-- GENERATED by tools/build-indexes.py — do not hand-edit -->\n\n# {heading}\n\n{table(rows)}\n"
        if write(idx / name, body):
            changed.append(f"docs/indexes/{name}")

    reg = {"schema": 1, "documents": sorted(docs, key=lambda d: d["path"])}
    if write(ROOT / "registry/index.json",
             json.dumps(reg, indent=1, ensure_ascii=False)):
        changed.append("registry/index.json")

    counts = {t: len(v) for t, v in sorted(by_type.items())}
    dmap = ["<!-- GENERATED by tools/build-indexes.py — do not hand-edit -->", "",
            "# Document Map", "",
            f"Catalog: {len(docs)} documents. Full machine catalog: `registry/index.json`.",
            "", "## Entry points", "",
            "| Need | Go to |", "|---|---|",
            "| Agent context loading | `docs/governance/agent-operating-system.md`, `tools/situation.py` |",
            "| Blocked/validated paths | `registry/research-ledger.json` |",
            "| Current research questions | `registry/research-program.json`, [research/PROGRAM.md](research/PROGRAM.md) |",
            "| Current model route | `registry/model-status.json`, `models/STATUS.md` |",
            "| Engine state | `engines/<engine>/README.md` — see `docs/indexes/engine-index.md` |",
            "| Campaign evidence | `campaigns/2026/<slug>/` → `results/2026/<MM>/` |",
            "| Decisions | `decisions/` — see `docs/indexes/decision-index.md` |", "",
            "## Document counts by type", ""]
    dmap += [f"- {t}: {n}" for t, n in counts.items()]
    dmap += ["", "## Views", ""] + [f"- [{h}](docs/indexes/{n})" for n, (h, _) in views.items()]
    if write(ROOT / "DOCUMENT-MAP.md", "\n".join(dmap)):
        changed.append("DOCUMENT-MAP.md")

    for c in changed:
        print(("DRIFT " if check else "wrote ") + c)
    if check and changed:
        print("FAIL: indexes out of date — run tools/build-indexes.py")
        return 1
    print(f"OK: {len(docs)} documents catalogued")
    return 0


if __name__ == "__main__":
    sys.exit(main())
