---
title: "SGLang XPU on B70 — engine dossier"
type: engine-dossier
status: living
date: 2026-09-15
updated: 2026-09-15
engine: sglang-xpu
---

# SGLang XPU — living dossier

Exploratory status: no validated serving route on this host yet. One
diagnostic microbench campaign (dense E1 + N256 tile) and run/bench skills
exist; treat any SGLang number as unevidenced until a registered campaign
lands under `results/2026/<MM>/`.

## Known so far

- `research/2026/sglang-xe2-dense-e1-and-n256-tile-20260817.md` — dense
  decode microbench on Xe2; E1 and the 256-tile dispatch path examined
  (pre-restructure path forms preserved in the note).
- Skills: `.agents/skills/sglang-xpu-run/`, `sglang-xpu-bench/` — launching
  and harness scaffolding only; no B70 campaign registered in
  `registry/campaigns.yaml` as of 2026-09-15.

## Rules if a campaign starts

Same gates as any engine: empty-GPU gate, declared engine + harness, cache
guards, power telemetry per `docs/methods/observability-standard.md`, results
root under `results/2026/<MM>/<slug>-<date>/`, campaign narrative under
`campaigns/2026/<slug>/`.
