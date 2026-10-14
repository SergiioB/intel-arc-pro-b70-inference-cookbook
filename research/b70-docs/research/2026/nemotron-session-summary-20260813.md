---
title: "Nemotron-3.5-Lightning B70 Optimization — Session Summary (2026-08-13)"
type: research
date: 2026-08-13
status: final
---

# Nemotron-3.5-Lightning B70 Optimization — Session Summary (2026-08-13)

## Current headline (isolated, cite this)

**DFlash on vLLM XPU works.** Isolated C1 n=5, cache off, 150 W, Run 36:

| Cell | Metric | median | range | n |
|---|---|---:|---|--:|
| p2048/g128 | C1 client post-first | **186.61 t/s** | 174.60–201.83 | 5 |
| p8192/g128 | C1 client post-first | 157.92 t/s | 143.50–170.25 | 5 |
| p8192/g1 | cold input (prompt/TTFT) | **7160 t/s** | 7117–7226 | 5 |
| window | DFlash acceptance | **52.0%** | 1830/3521 | — |

Matched-except-speculation vs no-spec n=5 p8192/g128 87.25: **1.81× on that cell**.

Evidence: `results/2026/08/nemotron-dflash-bf16-n7-n5-20260813T082203Z-1407994/`  
LocalMaxxing: `cmsr9po4w000ams01e4fc5qhj` (APPROVED = published self-report).  
HF (canonical **SergiioB**, two i's):
- https://huggingface.co/SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym
- https://huggingface.co/SergiioB/Nemotron-3.5-Lightning-30B-A3B-DFlash-BF16

## Earlier A-grade no-spec (still valid, different recipe)

**21.8 → 93.00 / 87.25 t/s** (p512/p8192 g128, n=5) via XPU graphs.

| Metric | Value | Notes |
|---|---|---|
| Decode p512/g128 n=5 | 93.00 (92.96–93.03) | `nemotron-gptq-graph-n5-20260812T2045Z` |
| Decode p8192/g128 n=5 | 87.25 | same |
| A-grade det n=3 p512 | 92.65 | after `at::zeros` |
| “10k prefill” | 10,371 → 10,349 | **n=3, p8192/g128 TTFT, cold input — not DFlash, not n=5** |

See `research/2026/nemotron-cold-input-vs-10k.md`. **Do not say DFlash hit 10k prefill.**

## Root cause of the weak eager decode (found + fixed)

~950 kernel launches/token (32.2 ms CPU enqueue vs 9.45 ms GPU). Fix: XPU
graph capture enabled by native grouped-topk v2.

## Kernel contributions (PR candidates)

See `research/2026/vllm-xpu-pr-readiness-20260813.md`.

1. **`at::empty` → `at::zeros`** in grouped-GEMM atomic — submit first (kernels).
2. **Native graph-safe grouped-topk** — submit second (vLLM, XPU guard).
3. **B70 SSU B8/W4 JSON** — optional device config.
4. **Muse paged-decode tuple** — ride with (1).
5. **Do not PR** MTP prenorm/concat (still 0%), retracted V1-bypass, W4A8, converters.

## MTP status (unchanged)

0% draft acceptance. Bad draft logits, not a sampler bypass. Issue #52107
retracted. Instrumentation still the missing piece. **DFlash is the working
spec path.**

## DFlash — what actually shipped

1. Official NVFP4 draft dequantized to BF16 (19 MLP/`fc` matrices).
2. Leftover `hf_quant_config.json` renamed (hard fail if left live).
3. `method=dflash` n_spec=7 on the GPTQ target + graphs.
4. Production pinned `Restart=no` for isolated n=5 (n=3 was contaminated).

## Next-model memory

`research/2026/next-model-adaptation-playbook.md`
