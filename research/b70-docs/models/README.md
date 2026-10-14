---
title: "models/ — per-model dossiers"
type: reference
status: living
date: 2026-09-15
updated: 2026-09-15
---

# models/

Per-model dossiers: `models/<family>/<model-slug>/` with `README.md`
(route, gates), `serving.md` (verified serving configs and runs),
`artifacts.md` (what exists on disk), `timeline.md` where the history is
load-bearing. Current-state views — update in place, carry `updated:`.

Families: `qwen/` (qwen3.6-35b-a3b, qwen3.8-27b, qwen3.8-flash-next),
`ornith/`, `nemotron/`, `muse/`.

- Fleet snapshot: `models/STATUS.md` (Markdown view of
  `registry/model-status.json`).
- Index: `docs/indexes/model-index.md` (generated).
- Headline numbers never live here without a campaign + results link.
