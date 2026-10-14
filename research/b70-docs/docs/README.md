---
title: "docs/ — governance, methods, operations, reference, plans"
type: reference
status: living
date: 2026-09-15
updated: 2026-10-09
---

# docs/

Typed documentation sections. Nothing loose at the root (2026-09-15 rule).

| Section | Content |
|---|---|
| `governance/` | Agent operating system, **standing rules** (always-on, always-try, closed classes), documentation standard, authority map, evidence lifecycle, publication governance |
| `methods/` | **Normative**: `benchmark-protocol.md` (the protocol), architecture-diagnostic playbook, observability standard, correctness validation, quantization evaluation, speculative decoding, benchmark guide |
| `operations/` | Current host state, storage/VRAM safety, recommended configs, incident & cleanup |
| `reference/` | **Format and lever guide** (which quant/engine/lever per workload), quantization decision record, KV-cache reference (+ quantization deep-dive), engine support matrix, hardware/software stack, megakernel reference, `b70-hardware.json` |
| `plans/` | Dated implementation plans (situation compiler, intelmaxxing site/hub) |
| `indexes/` | GENERATED views (`tools/build-indexes.py`) — research/campaign/decision/engine/model/doc/archive indexes |
| `experiments/`, `production-audit/`, `reproducibility/`, `archive/` | Historical dated material (kept in place; listed in `archive-index`) |

Normative conflict rule: when an older script/report conflicts with
`methods/benchmark-protocol.md`, the protocol wins (contract §2).
