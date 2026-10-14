---
title: "campaigns/ — campaign narratives"
type: reference
status: living
date: 2026-09-15
updated: 2026-09-15
---

# campaigns/

Human narratives of controlled benchmark campaigns — the interpretation
layer between raw evidence and any published claim.

## Layout

- `campaigns/2026/<slug>/README.md` — one folder per campaign: intent,
  declared variable, cells, findings, links to evidence roots under
  `results/2026/<MM>/`, publication state.
- Registration lives in `registry/campaigns.yaml` (ids, evidence roots,
  tiers). A campaign without a registry entry is not citable.

## Rules

- Every campaign declares one variable and one harness; contamination rules
  per contract §4 (one engine, one server).
- Findings label their statistics (median/peak/sustained, C1 vs Cn, greedy
  synthetic vs recommended sampling) per the measurement-naming rules.
- Publication flows only through
  `docs/governance/public-benchmark-publication.md`; until audit passes,
  numbers stay `PROVISIONAL — NOT FOR PUBLIC HEADLINE`.
