---
title: "Nemotron 3.5 Lightning 30B-A3B — Optimization Research"
type: research
date: 2026-08-11
status: final
---

# Nemotron 3.5 Lightning 30B-A3B — Optimization Research

**Date:** 2026-08-11
**Model:** NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_K_M.gguf (23.68 GiB)
**Architecture:** `nemotron_h_moe` — hybrid Mamba2-Transformer MoE (46 SSM + 7 attention+MoE layers)
**Hardware:** Intel Arc Pro B70 (32GB, 608 GB/s, Xe2, 256 XMX engines)
**Build:** llama.cpp SYCL `f785fc9ea` (b306) at `~/llama.cpp-sycl-latest/`

## Documents in this folder

| File | Description |
|------|-------------|
| `01-architecture-analysis.md` | Nemotron-H MoE architecture deep-dive: layer types, SSM dims, MoE config, MTP head |
| `02-root-cause-bottleneck.md` | Why decode is 37 t/s not 72: GEMV bandwidth efficiency, dp4a scalar emulation, dispatch overhead analysis |
| `03-kernel-fusion-implementation.md` | Phase 1-3 kernel fusion: conv bias+SiLU, scan dt_bias+D×x, split conv attempt, code changes |
| `04-spec-decoding-comparison.md` | MTP4 vs MTP8 vs DFlash7: acceptance, throughput, tradeoffs |
| `05-mmq-xmx-dispatch-fix.md` | The `__dp4a` scalar emulation discovery and MMQ/XMX dispatch override |
| `06-architect-subagent-report.md` | Full b70-architect analysis: theoretical ceiling, fusion roadmap, Xe2 mapping |
| `07-path-to-75-tps.md` | Concrete next steps: custom GEMV kernel, upstream SYCL, vLLM XPU |
| `evidence/` | Raw benchmark logs, llama-bench outputs, server logs |
| `kernel-fusion-patch.diff` | The complete code diff (5 files, 215 insertions) |

## Headline results

| Config | Before | After | Δ |
|--------|-------:|------:|---|
| No-spec decode (llama-bench tg128, 150W) | 36.83 t/s | 38.18 t/s | **+3.7%** |
| MTP4 decode (HTTP, 230W) | ~52 t/s | ~50-54 t/s | within noise |
| Best achievable (MTP4, 230W) | — | **~54 t/s** | — |
| Qwen MoE 35B-A3B reference | 72.6 t/s | — | target |

## Key findings

1. **The bottleneck is GEMV bandwidth efficiency** (11% for Nemotron vs 36% for Qwen), not kernel launch overhead (only 2ms out of 27ms).
2. **`__dp4a` is scalar-emulated** on Intel Xe2 — 4 scalar INT8 multiplies instead of XMX/DPAS. Fixed by preferring MMQ dispatch.
3. **MTP4 is the best speculative method** (95-100% acceptance, native target head). DFlash7 is worse (75-92% acceptance).
4. **75 t/s requires improving GEMV efficiency** from 11% to 25%+. Needs custom Xe2 GEMV kernel or upstream SYCL rewrite.
