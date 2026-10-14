---
title: "research/ — dated research notes"
type: reference
status: living
date: 2026-09-15
updated: 2026-09-15
---

# research/

Dated research notes, one topic per note, immutable once written.

## Start here

1. `python3 tools/research.py next` selects one question and its evidence.
2. `PROGRAM.md` shows the three ready focuses and parked questions. JSON authority: `registry/research-program.json`.
3. `LEDGER.md` records scoped outcomes, not a count of adopted wins. JSON authority: `registry/research-ledger.json`.
4. `references/` retains primary-source snapshots and explicit assumption-based prices.
5. `docs/indexes/research-coverage.md` reports historical unlinked notes/results; do not make clearing that backlog the inference objective.

## Layout

- `research/2026/<topic>-<YYYYMMDD>.md` — dated notes (the standard form).
- `research/2026/<topic>/` — multi-file topics (e.g. `quant-format-showdown-20260913/`).
- `references/`, `prs/`, `intel-outreach/`, `archive/` — undated containers
  (external references, upstream PRs, outreach, superseded notes).

## Rules

- New notes go to `research/<current-year>/` with the date in the filename;
  carry front-matter (`tools/check-frontmatter.py` enforces date+type).
- A note is evidence, not authority: current-state claims belong in the
  model/engine dossier or registry, linking to the note.
- A note must close into the typed record: ledger row plus, when the question
  stays open, a `registry/research-program.json` entry. `tools/research.py
  coverage` reports what never made it into a record.
- A `blocked`/`failed` row guards its named path and configuration, not the
  idea class. Re-entry = a named changed variable in the program record.
  Re-measuring a `validated` cell is permitted replication and must say so.
- Outcome and adoption are separate: `validated` answers "did this result hold
  in this scope", never "is this the route" (`models/STATUS.md`).
- Regenerate views: `python3 tools/research.py render` (also `check`).
- Index: `docs/indexes/research-index.md` (generated).
