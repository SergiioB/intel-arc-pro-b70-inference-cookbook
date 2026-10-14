---
doc_id: DOC-ARCHIVE-INDEX
doc_type: archive_index
title: B70-DOCS Archive
lifecycle: active
created: 2026-08-19
updated: 2026-08-19
last_verified: 2026-08-19
canonical_for:
  - archive-navigation
---

# B70-DOCS Archive

The archive preserves historical, superseded, withdrawn, session, and scratch material. Archived does not mean useless or invalid; each item needs an explicit lifecycle and replacement relationship.

## Rules

- Never archive raw result evidence merely to clean up navigation.
- Keep negative and failed research discoverable.
- A superseded document names its replacement, and the replacement names what it supersedes.
- Important old paths receive a tombstone or compatibility view.
- Scratch material is not indexed as research evidence unless a dossier or campaign explicitly promotes it.
- Potentially sensitive files are excluded from generated public-facing views.

## Current archive areas

| Path | Purpose |
|---|---|
| `archive/scratch/root-20260819/` | Inspected loose root diagnostics and intermediate scripts moved during the observability migration |
| `archive/docs/pre-migration-20260819/` | Pre-migration comprehensive host notebook and guides (superseded claims inside) |
| `archive/legacy-dirs-20260915/` | 2026-09-15 restructure: dissolved `benchmarks/`, `b70-bench-results/`, `runs/`, `studies/` |
| `archive/registry-system-20260830/` | Frozen 2026-08-30 registry YAML system (documents/legacy-results/artifacts/decisions/models/supersessions + migration schemas; generator tool at git `09356e50:tools/research_registry.py`) |
| `archive/legacy-docs-20260915/` | July/Aug session docs formerly loose at `docs/` root + compat redirect stubs (`compat-stubs/`) |
| `archive/root-strays-20260915/` | Top-level QA PNGs (`visual-qa/`) and `AGENTS.md.bak_publish` |
| `docs/archive/` | Existing historical documentation pending typed migration |
| `research/archive/` | Existing historical research pending typed migration |

Frozen-era note (2026-09-15): internal path mentions inside `archive/` and
`results/` are historical record — they are exempt from `tools/check-links.py`;
living registries carry the current paths.

## Root scratch migration — 2026-08-19

The following loose root files were inspected by metadata/type and moved without deletion:

- `--help` (large NDJSON diagnostic output)
- `wait.log`
- `check_params.py`
- `p1.py`, `p2.py`, `p3.py`
- `run_batch1.py`, `run_batch2.py`, `run_batch3.py`
- `mtp_keys.txt`

`mtp_keys.txt` matched no common API-key, bearer-token, private-key, or credential-assignment pattern in a content-safe scan. Its name and contents still warrant conservative treatment, so it remains excluded from generated documentation indexes unless a later evidence review classifies it.
