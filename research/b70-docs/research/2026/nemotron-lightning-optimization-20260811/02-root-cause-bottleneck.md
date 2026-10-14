---
title: "02 — Root Cause: Why Nemotron Decode is 37 t/s, Not 72"
type: research
date: 2026-08-11
status: final
---

# 02 — Root Cause: Why Nemotron Decode is 37 t/s, Not 72

## The gap

| Model | Decode (t/s) | ms/token | Active params |
|-------|------------:|---------:|--------------:|
| Qwen MoE 35B-A3B Q4 | 72.6 | 13.8 | ~3.0B |
| Nemotron 30B-A3B Q4 | 36.8 | 27.2 | ~2.86B |

Nemotron has **fewer** active params but is **2× slower**. Why?

## Hypothesis 1: Kernel launch overhead (INITIAL — refuted)

**Architect's initial estimate:** ~600 dispatches × 22µs = 13ms overhead.

**Measurement after Phase 1+2 fusion** (eliminated 276 dispatches):
- Before: 36.83 t/s (27.2ms)
- After: 37.56 t/s (26.6ms)
- Saved: 0.6ms from 276 dispatches → **~2.2µs/dispatch actual overhead**
- Total dispatch overhead: 600 × 2.2µs = **1.3ms** (only 5% of decode time)

**Conclusion:** Launch overhead is NOT the primary bottleneck. The 22µs estimate
was too high because ggml graph reuse already eliminates most host-side scheduling.

## Hypothesis 2: GEMV bandwidth efficiency (CONFIRMED)

### Bandwidth analysis

```
Nemotron active bytes/token: 1.57 GB weights + 0.19 GB SSM state = 1.76 GB
Qwen active bytes/token: ~3.0 GB (larger model, more active experts)
```

| Model | Bytes/token | t/s | Effective BW | Peak BW | Efficiency |
|-------|------------:|----:|-------------:|--------:|-----------:|
| Nemotron no-spec | 1.76 GB | 37 | 65 GB/s | 608 GB/s | **11%** |
| Nemotron MTP4 | 1.76 GB | 52 | 92 GB/s | 608 GB/s | **15%** |
| Qwen MoE | 3.0 GB | 72.6 | 218 GB/s | 608 GB/s | **36%** |

Nemotron achieves **3.3× worse bandwidth efficiency** than Qwen.

### Why is Nemotron's GEMV efficiency so low?

The 46 Mamba2 layers produce **92 small GEMV operations** (in_proj + out_proj per layer):
- in_proj: 2688 × 10304 (27.7M params, ~13.8 MB at Q4)
- out_proj: 4096 × 2688 (11.0M params, ~5.5 MB at Q4)

Small GEMV matrices achieve poor bandwidth efficiency because:
1. **Low arithmetic intensity** — M=1 means 1 MAC per byte read (memory-bound)
2. **Poor GPU saturation** — small matrices can't fill 32 Xe-cores / 256 XMX engines
3. **L2 cache pressure** — 18 MB L2 vs 13.8 MB per matrix (barely fits)
4. **Sequential dependency** — each layer depends on the previous (no parallelism)

Qwen MoE has fewer but larger GEMVs (attention Q/K/V/O + expert FFNs) that saturate the GPU better.

### The `__dp4a` scalar emulation problem

The SYCL MMVQ kernel (used for Q4_K M=1 GEMV) uses `__dp4a` for INT8 dot products:

```cpp
// dpct/helper.hpp:1859 — emulated as 4 SCALAR INT8 multiplies
inline auto dp4a(T1 a, T2 b, T3 c) {
    res += va[0] * vb[0];
    res += va[1] * vb[1];
    res += va[2] * vb[2];
    res += va[3] * vb[3];
    return res;
}
```

This does NOT use the XMX/DPAS hardware. The 256 XMX engines sit completely idle
during Q4_K decode. The scalar ALUs do the work 10-100× slower than XMX would.

The MMQ kernel (for larger M) DOES use XMX via DPAS (`#if defined(SYCL_USE_XMX)`),
but was never dispatched for M=1 because MMVQ took priority in the dispatch chain.

**Fix applied:** Force MMQ over MMVQ for K-quants when XMX is available.
Result: +3.7% improvement (36.83 → 38.18 t/s). Modest because GEMV is
memory-bound — the XMX compute speedup helps but the bottleneck is memory access.

## Summary: the three bottleneck layers

```
Layer 1 (root):  Small GEMV matrices → poor bandwidth efficiency (11%)
Layer 2 (amplifies):  dp4a scalar emulation → XMX idle → wasted compute cycles  
Layer 3 (additive):  600 kernel dispatches → ~1.3ms launch overhead
```

Fixing Layer 3 (our Phase 1+2 fusion) saved ~0.6ms.
Fixing Layer 2 (MMQ dispatch) saved ~0.6ms.
**Layer 1 remains the dominant bottleneck** and requires a custom GEMV kernel.
