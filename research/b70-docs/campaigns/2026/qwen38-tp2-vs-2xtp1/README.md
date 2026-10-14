---
id: CAM-2026-QWEN38-TP2-VS-2XTP1
doc_id: DOC-CAM-2026-QWEN38-TP2-VS-2XTP1
status: cancelled
title: "Qwen3.8-27B TP2 vs two independent TP1 servers (scoped)"
type: campaign
date: 2026-09-24
updated: 2026-10-09
---

> **Cancelled 2026-10-09, no cell ran.** The pinned artifact
> `ART-QWEN38-GPTQ-G128-MTP-BF16` was removed as quality-defective (RL-087,
> ADR-0010). Stock TP2 measured −45…−71 % (RL-075) and EXL3 TP2 is blocked at
> load (RL-103). The reserved evidence root was released. A future TP2 question
> lives in RQ-06 (custom copy-only collective). The scope below is kept as history.

# Qwen3.8-27B TP2 vs two independent TP1 servers (scoped)

Scope record only: no cells executed, no evidence collected, no numbers. This
campaign executes the ADR-0008 reconsideration trigger: "a protocol-compliant
local TP1 vs TP2 campaign on a pinned image/patch set with quality gate and
energy-aligned power."

- Decision under test:
  [ADR-0008](../../../decisions/adr-0008-dual-b70-independent-default.md)
  (verdict PARTIAL: `topology_confirmed_tp2_externally_reported`)
- External context (E0; their numbers stay theirs): a community dual-B65
  report (vLLM 0.29.0 XPU, Qwen3-30B-A3B GPTQ-Int4) measured TP2 C1 at
  ~51 tok/s against ~157 tok/s aggregate for two independent TP1 workers —
  the same no-P2P, host-staged-collective physics as this host. It motivates
  the campaign; it is never citable as this host's evidence.
- Model route: [Qwen3.8-27B dossier](../../../models/qwen/qwen3.8-27b/README.md)
  (GPTQ INT4 G128 + BF16 MTP4, the single-card flagship, fits one 32 GB card)
- TP2 infrastructure contract: cookbook
  [DUAL-B70-TP2.md](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/docs/DUAL-B70-TP2.md)
  (worker-affinity patch, oneCCL simple thresholds, `SYS_PTRACE`,
  `--ipc=host`, compile-only execution) plus per-worker compile-cache
  isolation (`VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`) per the 2026-09-24
  cookbook note on independent servers
- Evidence root: none (reservation released at cancellation)

## Declared variable

Multi-GPU serving topology for one model on the dual-B70 host:

- two independent TP1 engines (the ADR-0008 default), or
- one TP2 engine (the gated research path, ADR-0008 Decision 2).

Everything else is pinned: image digest, patch set, artifact
`ART-QWEN38-GPTQ-G128-MTP-BF16`, one harness, workload, sampling regime,
power policy.

## Cells

All cells: greedy diagnostic `p512/g128`, warmup discarded (protocol 1.2),
`n=5` with median/range retained (protocol 1.6), ≤55 °C cooldown between
comparable tests (protocol 1.3), empty-GPU gates on both cards, 230 W
configured cap with `energy1_input` deltas per card per cell.

| cell | topology | clients | question |
|---|---|---|---|
| A1 | TP1, card 0 (`ZE_AFFINITY_MASK=0`) | C1 | solo anchor |
| A2 | TP1, card 1 (`ZE_AFFINITY_MASK=1`) | C1 | solo anchor; card asymmetry on X570-F x8/x8 |
| B | TP2, both cards, full infra contract | C1 | per-stream TP2 decode tax vs A1/A2 |
| C | 2×TP1 independent (masks 0/1, separate compile caches, ports 8000/8001) | 2 concurrent C1 | per-stream and aggregate vs A1+A2 |
| D | TP2 | C2 | gate-record: expected blocked under MTP (GDN spec/non-spec mixing); predeclared non-comparable |

## Guards and stop rules

- ADR-0008 Decision 2 gates cell B: `--max-num-seqs 1`, host-staged
  collectives contract applied in full, no stock Docker TP2, no graph capture.
- If cell C aggregate falls below 90 % of the A1+A2 solo median sum, any
  scaling claim halts; investigate host contention before publication.
- A TP2 failure-to-serve in cell B is itself a recorded result (config plus
  failure mode); do not swap patches mid-campaign to rescue it.
- Product metrics never mix (ADR-0008 Decision 5): A1/A2, B, C, and D are
  separate coordinates and are reported separately.

## Publication path

Cookbook topology note plus `benchmark` catalog records only after campaign
validation and claims audit; until then every number from this campaign is
`PROVISIONAL — NOT FOR PUBLIC HEADLINE`. A cell D block outcome publishes at
most as `capability` evidence on the Cn limitation.
