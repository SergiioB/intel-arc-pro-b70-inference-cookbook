---
title: "B70 format and lever guide: which quant, engine and lever per workload"
type: reference
status: living
date: 2026-10-09
updated: 2026-10-09
---

# B70 format and lever guide

Use this page to pick a starting route for a workload or a new model. It is a
decision aid, not a result: routes of record live in `registry/model-status.json`,
open questions in `registry/research-program.json`, invariants in
`docs/governance/standing-rules.md`. Update this page when a route, closure or
datapath fact changes. Evidence for every row: `research/2026/b70-untried-levers-synthesis-20261009.md`.

## 1. What the silicon rewards

| Path (per Xe-core per clock) | Rate | Use |
|---|---|---|
| XMX FP16 / BF16 | 2048 ops | Where every current quant decode and prefill ends up after dequant |
| XMX INT8 s8×s8 | 4096 ops (367 TOPS) | The only fully exposed 2× path. Prefill lever; unused by current quant paths |
| XMX INT4 / INT2 int×int | 8192 ops | Hardware-real (2.00× per instruction, ESIMD only). Needs 4-bit activations |
| FP8, MXFP4, NVFP4 | no DPAS | Byte formats only on Xe2 |
| DP4A (vector) | 1024 ops | llama.cpp MMVQ decode with Q8_1 activations, M ≤ 8 |

| Memory fact | Value |
|---|---|
| Pin bandwidth | 608 GB/s, 32 GB per card |
| Best measured local streaming | 381 GB/s, EXL3 M=4 verify (63 %) |
| External streaming references (E0) | ~570 GB/s FP16 GEMV (Puget); 493 GB/s aligned Q6_K (Strata) |
| SLM / GRF | 128 KB per Xe-core; 128 or 256 (large-GRF) registers per thread, sub-groups 16/32 |
| Card-to-card | copy-only P2P 11.1–11.9 GB/s, no peer atomics (RL-059) |

Rules that follow:

1. **Decode** = yield × achieved bandwidth ÷ bytes per token. Pick the fewest
   bytes whose dequant is cheap **and** 16-byte aligned; issue rate, not DRAM, is
   the wall on this card.
2. **Prefill** = compute. Only s8×s8 doubles it. FP8/MXFP4 do not.
3. **MoE** fits the card's economics (few active bytes), but only when experts
   are resident; a miss stream at PCIe rates costs a third of a round.

## 2. Workload regimes — current best, next candidate, closed

| Regime | Current best (route of record) | Next candidate (program) | Closed here |
|---|---|---|---|
| Dense ~27B, C1 decode | EXL3 4bpw + MTP3, exl3xpu, 63.25 tok/s (weights must be restored after the 2026-10-09 cleanup) | **HQ4** (RQ-26): block-128 Hadamard + affine INT4 G128 in a DPAS layout, quality screened against EXL3 first; RQ-20 is its kernel half | Symmetric INT4 W4A16 without rotation; DFlash2 on quantized target; stock TP2 |
| Dense, long context (≥128K) | EXL3 (only route reaching native 262K on one card) | — (TurboQuant KV exists only in newer vLLM) | OV INT4 ≥196K (memory growth, RL-033) |
| Large MoE (Flash-Next class), decode | Strata IQ3_XXS streaming, one card, 48.03 tok/s | **Residency:** two-card `--layer-split` (RQ-18) or expert-pruned Coder on one card (RQ-23, code only) | Host-staged second-card cache; storage tier at 5 GB/s |
| Mid MoE (30B-A3B class), decode | Nemotron GPTQ-INT4 G64 + DFlash n7, vLLM XPU, 186.6 tok/s (historical) | Re-measure + dispatch attribution (RQ-11) | Mamba2 fusion (8 % of device time); post-training MXFP4 (RL-128) |
| Prefill (any) | exl3xpu int8 linears + oneDNN SDPA; Strata `--prefill 4096` | INT8 prompt GEMM for Strata SYCL (RQ-22) | W4A8 prefill (≤1.27×) |
| Agentic / document tasks | Route of record with declared cache state | Prefix reuse (RQ-10), workload-matched speculation (RQ-21) | MTP+ngram stacking (−2 %) |
| Concurrency | EXL3 C16 426 tok/s aggregate (synthetic) | Service levels on real tasks (RQ-05) | — |

## 3. A new model arrives: decide in this order

1. **Fit.** Weights + KV at the target context + 4 GiB reserve ≤ 32 GB? If not,
   can the expert set be resident across two cards (layer split)? Only then
   consider streaming. Check live disk first; three volumes were ≥84 % full on
   2026-10-09.
2. **Architecture class.** Dense → bytes per token rule. MoE → resident active
   bytes. Hybrid attention (GDN/Mamba) → KV only grows on full-attention layers.
3. **Format by engine availability, then quality, then speed:**
   - vLLM XPU: GPTQ/AWQ W4A16, `moe_wna16`, FP8 (tensor/channel/block, online),
     MXFP4/MXFP8, INC W8A8. No NVFP4, no GGUF.
   - exl3xpu: EXL3 trellis only; fp8 KV only.
   - llama.cpp SYCL / Strata: all GGUF types; Strata adds the expert cache and
     MTP. Check block alignment of the chosen type.
   - OpenVINO GenAI: NNCF INT4/INT8 (+AWQ/GPTQ/scale estimation), INT8 or INT4 KV.
4. **Quality gate before any speed cell:** exact-answer battery including the
   in-head arithmetic canary ("14" not "30"). A vendor QAT artifact is preferred
   over a post-training 4-bit export for dense models.
5. **Speculation:** use the model's native MTP first. A separately trained
   drafter must have been trained on the quantized target's states.
6. **Measure** under the benchmark protocol; register through `research.py`.

## 4. Engine pins to check before tuning

| Engine | Local pin (2026-10-09) | Known upstream delta |
|---|---|---|
| exl3xpu | image `21412bdd` (vLLM 0.26.1) | Operator-held; no probe work |
| Strata | v0.1.40.3 / engine 0.1.41 | SYCL port lacks int8 prompt kernels and `--peer-device` |
| llama.cpp SYCL | fleet tree `1692f9e50` (2026-08-14) | Master adds TILE quantized-KV, IQ3_S multicol, grouped-MoE XMX, MXFP4/NVFP4 MMVQ; #29809 MXFP4 MoE open (RL-126) |
| vLLM XPU | `0.27.2rc1` nightly | Main routes TurboQuant 3/4-bit KV on XPU. Current docs list suffix and n-gram speculation; verify they exist in the pinned build (and in exl3xpu's vLLM 0.26.1) before planning |
| OpenVINO GenAI | openvino `2026.5.0-22967` dev, genai `2026.5.0.0-3412` (recorded 2026-09-15) | 2026.4 release notes list MTP, DFlash preview and EAGLE-3 for Qwen3.8 on GPU, and 2026.2 added INT4 KV; confirm the local dev build carries them before testing |
