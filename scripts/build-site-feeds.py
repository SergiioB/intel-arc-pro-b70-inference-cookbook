#!/usr/bin/env python3
"""Build and check the machine feeds XeCores reads live from this repository.

  scripts/build-site-feeds.py           write data/docs-corpus.v1.json
  scripts/build-site-feeds.py --check   fail if it is stale, or if data/routes.v1.json is invalid

data/docs-corpus.v1.json: every public doc (README.md, START.md, docs/**/*.md) with
its title, sha256, status and text. A doc whose first lines carry a historical,
superseded or legacy banner is "historical": the site keeps it browsable but does
not use it to answer questions.

data/routes.v1.json: hand-maintained route records (which model + engine to run,
the launcher command, the model card's sampling, the catalog records that measure
it). This script only validates it against data/benchmarks.v1.json.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/docs-corpus.v1.json"
ROUTES = ROOT / "data/routes.v1.json"
CATALOG = ROOT / "data/benchmarks.v1.json"
HISTORICAL = re.compile(
    r"^>\s*(status:\s*)?(historical|superseded|legacy)\b|^status:\s*(historical|superseded)",
    re.I | re.M,
)
STATUSES = {"recommended", "pending", "alternative", "research", "closed"}
USE_CASES = {"chat", "coding", "long-context", "vision"}


def doc_paths(root: Path) -> list[Path]:
    paths = [root / "README.md", root / "START.md"]
    paths += sorted((root / "docs").rglob("*.md"))
    return [p for p in paths if p.is_file()]


def build_corpus(root: Path) -> dict[str, Any]:
    docs = []
    for path in doc_paths(root):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(root).as_posix()
        title = next((m.group(1).strip() for m in re.finditer(r"^#\s+(.+)$", text, re.M)), rel)
        head = "\n".join(text.splitlines()[:15])
        docs.append(
            {
                "path": rel,
                "title": title,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "status": "historical" if HISTORICAL.search(head) else "current",
                "text": text,
            }
        )
    digest = hashlib.sha256("".join(d["sha256"] for d in docs).encode()).hexdigest()
    return {"schema_version": "1.0", "content_sha256": digest, "docs": docs}


def check_routes(root: Path) -> list[str]:
    errors: list[str] = []
    if not (root / ROUTES.relative_to(ROOT)).is_file():
        return errors
    routes = json.loads((root / ROUTES.relative_to(ROOT)).read_text())
    records = {
        r["id"] for r in json.loads((root / CATALOG.relative_to(ROOT)).read_text())["records"]
    }
    if routes.get("schema_version") != "1.0":
        errors.append("routes: schema_version must be 1.0")
    seen = set()
    featured = 0
    for r in routes.get("routes", []):
        rid = r.get("id", "?")
        if rid in seen:
            errors.append(f"routes: duplicate id {rid}")
        seen.add(rid)
        featured += bool(r.get("featured"))
        if r.get("status") not in STATUSES:
            errors.append(f"routes: {rid} has status {r.get('status')!r}")
        if r.get("cards") not in (1, 2):
            errors.append(f"routes: {rid} cards must be 1 or 2")
        for use in r.get("use_cases", []):
            if use not in USE_CASES:
                errors.append(f"routes: {rid} unknown use case {use}")
        for rec in r.get("records", []):
            if rec not in records:
                errors.append(f"routes: {rid} cites missing catalog record {rec}")
        if not (root / r.get("doc", "")).is_file():
            errors.append(f"routes: {rid} doc {r.get('doc')} does not exist")
        sampling = r.get("sampling")
        if sampling is not None and not str(sampling.get("source", "")).startswith(
            "https://huggingface.co/"
        ):
            errors.append(f"routes: {rid} sampling needs a model card source")
        if (
            r.get("status") in {"recommended", "alternative"}
            and sampling is None
            and not r.get("notes")
        ):
            errors.append(f"routes: {rid} has no sampling and no note saying why")
    if featured != 1:
        errors.append(f"routes: exactly one featured route expected, found {featured}")
    return errors


def main(argv: list[str]) -> int:
    corpus = build_corpus(ROOT)
    text = json.dumps(corpus, ensure_ascii=False, indent=1) + "\n"
    if "--check" in argv:
        errors = check_routes(ROOT)
        if not CORPUS.is_file() or CORPUS.read_text(encoding="utf-8") != text:
            errors.append("data/docs-corpus.v1.json is stale: run scripts/build-site-feeds.py")
        for error in errors:
            print(error, file=sys.stderr)
        if not errors:
            print(f"OK: site feeds current ({len(corpus['docs'])} docs)")
        return 1 if errors else 0
    CORPUS.write_text(text, encoding="utf-8")
    print(f"wrote {CORPUS.relative_to(ROOT)}: {len(corpus['docs'])} docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
