---
title: "KV Cache Quantization — Definitive Reference"
type: reference
date: 2026-08-03
status: living
---

# KV Cache Quantization — Definitive Reference

**The B70 fleet standard is `q8_0` K + `q4_1` V.** This document explains why,
with the KL-divergence evidence, the theory, and the validation data.

Last updated: 2026-08-03 (fact-checked against llama.cpp #23470). Sources:
[sergiiob.dev KL-divergence analysis](https://sergiiob.dev/posts/kv-cache-quantization-kl-divergence/),
[llama.cpp discussion #23470](https://github.com/ggml-org/llama.cpp/discussions/23470),
[Anbeeld long-context benchmarks](https://anbeeld.com/articles/kv-cache-quantization-benchmarks-for-long-context).

---

## 1. The recommendation

| Setting | Value | Rationale |
|---------|-------|-----------|
| **K cache** | `q8_0` | Keys are sensitive — determines attention routing. KLD ~0.003. |
| **V cache** | `q4_1` | Values are averaged in attention — tolerate aggressive quantization. |
| Flash attention | `on` | **Required** for quantized KV cache to work correctly. |

**Result:** ~50% VRAM reduction vs FP16 KV, with near-lossless quality.

Server flags:
```
-fa on -ctk q8_0 -ctv q4_1
```

---

## 2. KL-divergence evidence (fact-checked 2026-08-03)

KL-divergence measures how much the quantized cache's output distribution diverges
from the FP16 baseline. Lower = better quality preservation.

Numbers below are **measured values** from [llama.cpp discussion #23470](https://github.com/ggml-org/llama.cpp/discussions/23470)
(user `sanmai` on Qwen2.5-7B, user `jkrauss82` expanded table), cross-referenced
with [sergiiob.dev](https://sergiiob.dev/posts/kv-cache-quantization-kl-divergence/).

| K cache | V cache | KL-divergence (mean) | Same top-p % | VRAM vs FP16 | Verdict |
|---------|---------|---------------------|--------------|--------------|---------|
| FP16 | FP16 | 0 (baseline) | 100% | 100% | Reference |
| **q8_0** | **q4_1** | **~0.003** | ~97% | **~50%** | **✅ RECOMMENDED** |
| q8_0 | q4_0 | ~0.0048 | ~96.7% | ~44% | Near-lossless, slightly more aggressive |
| q8_0 | q8_0 | ~0.003 | ~97% | ~75% | Conservative — same K quality, more VRAM |
| q5_0 | q4_1 | ~0.006–0.008 | ~93% | ~44% | Acceptable trade-off (community-validated) |
| q5_1 | q4_1 | ~0.006 | ~93% | ~44% | "Sweet spot" per #23470 consensus |
| **q4_0** | *any* | **≈ 5.5** | **~11.6%** | ~25% | **❌ CATASTROPHIC — model output destroyed** |

### ⚠️ The q4_0 K cliff is catastrophic, not gradual

The `q4_0` K-cache quality cliff is **KLD ≈ 5.5** (not the ">0.03" previously
documented — that was a misattribution). This is a **catastrophic** divergence:
only ~11.6% of top-p tokens match the FP16 baseline. The model output is
essentially destroyed. This cliff is driven by K precision alone — it occurs
regardless of V precision (q4_0 K + f16 V = 5.49, q4_0 K + q4_0 V = 5.51).

### On "never quantize K below 8 bits"

The sergiiob.dev blog states this as a rule, but the llama.cpp community
(#23470 consensus) considers **q5_0 and q5_1 K as acceptable trade-offs** —
KLD ~0.006–0.008, ~93% top-p match. The practical guidance:
- **q8_0 K = safest** (KLD ~0.003, near-lossless)
- **q5_0/q5_1 K = acceptable** (KLD ~0.006–0.008, measurable but minor degradation)
- **q4_0 K = never** (KLD ~5.5, catastrophic)

---

## 3. Why K needs more precision than V

The attention mechanism computes: `softmax(Q·Kᵀ / √d_k) · V`

- **K (Keys)** determine *which tokens to attend to* — the routing decision.
  Quantizing K introduces noise into the attention weights, causing the model
  to attend to wrong tokens. This is a structural error that compounds.
- **V (Values)** are *weighted-averaged* by the attention weights. Quantization
  noise in V is averaged out across many tokens, so the impact on the output
  is much smaller.

This asymmetry means: **spend bits on K, save bits on V.**

---

## 4. VRAM impact — and the critical dense-vs-MoE KV difference

### 4.1 KV scales with ATTENTION size, not total params (key insight)

A common mistake is assuming KV cache scales with total parameter count. It
doesn't — **KV cache scales with the attention layers**, which depend on hidden
size, num heads, and num layers, NOT on FFN/expert size. This has a major
practical consequence on the B70:

**Measured (q8_0 K + q4_1 V, b10222, 2026-08-03):**
- **Dense 27B:** ~6960 MiB KV per 128K context
- **MoE 35B-A3B:** ~1847 MiB KV per 128K context (**3.8× less!**)

MoE 35B-A3B has tiny attention because only 3B params are active per token — the
Q/K/V projections are small. Dense 27B runs full-size attention on all 27B params.
This means: **a 25 GB MoE model can reach 512K context, while a 16 GB dense
model can't pass 256K** — the MoE's KV is so much smaller it more than
compensates for the larger weights.

### 4.2 Per-context KV size by model type (q8_0 K + q4_1 V)

| Context | Dense 27B KV | MoE 35B-A3B KV | Ratio |
|---------|-------------|----------------|-------|
| 32K | ~1.7 GB | ~0.5 GB | 3.8× |
| 128K | ~6.8 GB | ~1.8 GB | 3.8× |
| 256K | ~13.6 GB | ~3.6 GB | 3.8× |
| 512K | ~27.2 GB | ~7.2 GB | 3.8× |

### 4.3 Full VRAM budget table (q8_0 K + q4_1 V, measured)

Fixed overhead: ~1526 MiB (MoE) / ~1957 MiB (dense) for compute buffers + JIT.

| Config | Weights | @128K | @200K | @256K | @512K |
|--------|---------|-------|-------|-------|-------|
| 27B Q4_K_M +MTP +vis | 16 GB | 7.2 GB ✅ | 3.3 GB ✅ | 0.2 GB ⚠️ | OVER |
| 27B Q4_K_M +MTP (no vis) | 16 GB | 8.1 GB ✅ | 4.2 GB ✅ | 1.1 GB ✅ | OVER |
| 27B Q5_K_M +MTP (no vis) | 19 GB | 4.3 GB ✅ | 0.4 GB ⚠️ | OVER | OVER |
| 35B MoE Q4_K_XL | 21 GB | 8.0 GB ✅ | 6.9 GB ✅ | 6.1 GB ✅ | 2.4 GB ✅ |
| 35B MoE Q5_K_M | 25 GB | 4.0 GB ✅ | 3.0 GB ✅ | 2.2 GB ✅ | OVER |

Values = MiB free after model + KV + buffers. ✅ >1 GB headroom, ⚠️ <1 GB (tight).
Note: MoE 35B reaches 512K while dense 27B can't pass 256K — despite larger weights.

---

## 5. Hardware validation (B70, b9853, 2026-07-01)

Validated on B70 hardware. Full per-round data in `research/2026/kv-cache-validation-results.md`.

| Config | Model | Context | Decode t/s | Quality |
|--------|-------|---------|-----------|---------|
| q8_0-q4_1 | Qwen 35B Q5 | 256K | 67.0 | ✅ Clean |
| q8_0-q8_0 | Qwen 35B Q5 | 256K | 67.2 | ✅ Clean (more VRAM) |
| q4_0-q4_0 | Qwen 35B Q5 | 256K | 68.1 | ❌ Degraded output |

Speed is nearly identical across configs (KV quant has minimal compute impact).
The difference is **quality**, not speed.

---

## 6. Configuration drift history

- **Pre-2026-07-01:** Fleet used `q8_0-q8_0` (conservative).
- **2026-07-01:** Switched to `q5_0-q4_1` based on Anbeeld benchmarks (38% savings).
- **2026-08-03:** Upgraded to `q8_0-q4_1` based on sergiiob.dev KL-divergence
  analysis — q8_0 K has half the KL-divergence of q5_0 K at only ~6% more VRAM.
  This is the current fleet standard. All `profiles.json` entries and launcher
  scripts were standardized on this date.
