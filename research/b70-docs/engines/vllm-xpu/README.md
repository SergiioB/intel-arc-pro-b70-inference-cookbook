---
title: "vLLM XPU on B70 — engine dossier"
type: engine-dossier
status: living
date: 2026-09-15
updated: 2026-10-09
engine: vllm-xpu
---

# vLLM XPU — living dossier

Serving engine for the Nemotron 3.5 Lightning GPTQ-INT4 G64 + DFlash n7 route
(186.61 tok/s p2048/g128 client post-first, 2026-08-13 harness) and for the
Qwen3.8-27B block-FP8 diagnostic arms. It is **no longer the Qwen3.8-27B
route**: the GPTQ-Int4 + MTP4 champion (102.61 tok/s historical) was removed on
2026-09-27 because its weights carry the in-head arithmetic defect (RL-087,
RL-092). The current dense route is EXL3 on the exl3xpu fork (ADR-0010).

## Current state (as of 2026-09-15; route note 2026-10-09)

- Serving build: `vllm 0.27.2rc1 XPU` (nightly image pinned by digest; record
  the in-container vLLM/kernel versions per contract §7).
- Historical route (removed artifact): `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`,
  MTP4, fp8 KV. Dossier: `models/qwen/qwen3.8-27b/serving.md`. Do not relaunch.
- 2026-09-15 cross-engine sweep (2×B70 TP2 arms): FP8 W8A16 + small-M MTP8
  TP2 holds 57–65 tok/s decode from 32K to 98K ctx where single-card engines
  decay (client post-first, n=3 — `results/2026/09/qw38-ov-mtp-20260914/ctx-sweep-vllm-fp8-tp2.json`).
  GPTQ-Int4 262K probe: 45.7 / 32.6 / 34.5 tok/s at 98K/131K/196K
  (client post-first, n=1 probes — `ctx-sweep-vllm-int4-262k-probe.json`).
- vLLM INT8 (W8A16, 31.6 GB) TP2 launch failed the memory gate
  (29.28 GiB free < 29.39 GiB requested at util 0.97) — open cell, see
  `vllm-int8-attempt.json` + `ctx-sweep-vllm-int8-tp2.json` (empty grid).

## Hard rules (contract §7)

- Pin images by immutable digest; record observed versions.
- GPTQ-INT4 is a best-proven path for **exact validated checkpoints**, not a
  universal format guarantee.
- Prefix caching defaults ON in V1 — verify counters; use the HTTP cache
  guards (contract §9) for every cold cell.
- Single-client measurement gate: one client at a time against C1 servers.
- Dual-card default is independent stacks (`ZE_AFFINITY_MASK`). Stock Docker
  TP2 is forbidden on this host; TP2 only via the registered host-staged
  collectives recipe (ADR-0008).

## Verified capabilities / limits

| Capability | State | Evidence |
|---|---|---|
| GPTQ-Int4 + MTP4 (TP1) | serving leader, 102.61 tok/s p512/g128 client post-first (Run 48, recommended sampling, prefix-on stack, GPU 0, n=5) | model-status, Run 48 sampling |
| Why GPTQ-Int4 leads (mechanism) | W4A8 ties INT8's 363-TOPS DPAS class while halving weight bytes; FP8-W8A16 has no native Xe2 DPAS and lands in the 170-TFLOPS BF16 class with 1-byte weights — INT4 dominates FP8 on both axes | `research/2026/xmx-dpas-rate-classes-20260915.md` |
| FP8 W8A16 reroute (`fp8_gemm_w8a16`) | validated, +34% p512 | RL-003 |
| Native Xe2 small-M FP8 GEMM | blocked (kernel build needed) | RL-004 |
| DFlash2 draft | no-go: XPU K-norm bug #56431 root-caused, partial recovery; retry needs official image with #56431 + vxk#579 + selector-NaN fix | RL-005/RL-027 |
| TP2 stock Docker | blocked | RL-008 |
| INT8 W8A16 TP2 | launch OOM at util 0.97 — open | 2026-09-15 attempt |
| Decode GRF spills | needs `reqd_sub_group_size(16)` bounds | RL-001, `research/2026/xe2-esimd-launch-bounds-2026-08.md` |

## Related

Skills: `.agents/skills/vllm-xpu-run/`, `vllm-xpu-bench/`, `vllm-xpu-profile/`.
Support matrix: `docs/reference/engine-support-matrix.md`.
