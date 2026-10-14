---
title: "llama.cpp SYCL on B70 — engine dossier"
type: engine-dossier
status: living
date: 2026-09-15
updated: 2026-09-15
engine: llamacpp-sycl
---

# llama.cpp SYCL — living dossier

One-card GGUF engine. The workhorse for Q4_K_M-class quant, long-context
capacity, and the only engine with a validated dual-card layer-split path.

## Current state (as of 2026-09-15)

- Build in service: SYCL flashnext `52d4268` (verified in the 2026-09-15
  cross-engine sweep).
- Fleet routes: Qwen3.8-27B UD-Q4_K_M + native draft-MTP (16–38 tok/s
  decode across 512–262K ctx, single card); Q8_0 + draft-MTP (~51 tok/s
  ≤8K ctx). Evidence: `results/2026/09/qw38-ov-mtp-20260914/ctx-sweep-llamacpp-*.json`.
- DeepSeek-V4 K216 IQ3: L5 route `-ncmoe 15 -ts 65,35`, PROVISIONAL best
  9.25 clean / 9.58 patched engine tok/s (re-measured 2026-09-21 on the
  re-downloaded artifact; 8.87 superseded) — decode wall is the fused-IQ3
  `MUL_MAT_ID` MMVQ efficiency gap (5.52x above weight-read floor), not a
  flag problem (`research/2026/dsv4-k216-request-to-output-pipeline-20260825.md`,
  RL-010). The device-ID repair measured +3.54 % end-to-end (ABBA, n=6/build,
  >=10 % gate FAILED) and is withdrawn as a speed lever.

## Mandatory serving defaults (contract §6)

`-ngl 99 -ncmoe 0 -fa on -ctk q8_0 -ctv q4_1 -t 8 --no-mmap`, context always
declared with `-c`. **Never q4_0 K cache** (catastrophic recorded quality
cliff — `docs/reference/kv-cache-quantization.md`).

## Verified capabilities / limits

| Capability | State | Evidence |
|---|---|---|
| Speculative decode (GGUF draft-MTP) | working, Qwen3.8 Q8_0/Q4_K_M | 2026-09-15 ctx sweeps |
| 262K context, Q4_K_M + MTP | 15.95 tok/s @ 262144 (n=1 probe) | `ctx-sweep-llamacpp-q4km-mtp.json` |
| Cold prefill | ~210–250 tok/s (Q4_K_M, no cache-prompt) | `llama-q4km-cold-prefill.json` |
| Dual-card layer split | working (capacity, not 2× per layer) | K216 research; ADR-0008 context |
| `-sm row` | segfault — never | RL ledger |
| `--mmap` on >30 GB GGUF | forbidden (hard-crash class) | contract §5 |

## Known issues

- IQ3/IQ4-class quants omitted from fused SYCL `MUL_MAT_ID` MMVQ **upstream
  at `1692f9e50`** (FALSE for the pinned tree, which carries the device-ID
  cases uncommitted) → per-token D2H expert-ID copies + host GEMV on the
  multi-token path. The fused routed op itself runs 5.52x above its
  weight-read floor — kernel efficiency, not sync, is the K216 wall. Do not
  re-sweep flags; the fix is a grouped IQ3 kernel (RL-011 direction).
- Prompt-cache warm TTFTs in repeated-run harnesses are meaningless as
  prefill: always pair with a `--no-cache-prompt` cold cell when prefill
  matters.

## Related

Skills: `.agents/skills/llamacpp-xpu-run/`. Diagnosis playbook:
`docs/methods/architecture-diagnostic-playbook.md`.
