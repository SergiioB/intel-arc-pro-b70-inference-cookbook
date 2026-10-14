---
title: "06 — Path to 75+ t/s"
type: research
date: 2026-08-11
status: final
---

# 06 — Path to 75+ t/s

## Current state

| Metric | Value |
|--------|------:|
| No-spec decode (fused+MMQ, 150W) | 38.2 t/s |
| MTP4 decode (230W) | ~52 t/s |
| Qwen MoE reference | 72.6 t/s |
| Gap to target | ~20 t/s (38%) |

## The fundamental barrier

Nemotron's 46 Mamba2 layers produce **92 small INT4 GEMV operations** that achieve
only **11% of 608 GB/s peak bandwidth**. To reach 75 t/s, the GEMV efficiency
needs to improve from 11% to ~25%.

At 25% efficiency: 1.76 GB × 0.25 / (608 GB/s) = 0.72ms → ~38ms for 53 layers
Wait, that's per-layer. Total: 0.72ms is the total per-token memory time at 100%
efficiency. At 25%: 2.88ms. Adding compute + overhead: ~5-8ms → 125-200 t/s.

Actually, at 25% efficiency, the effective bandwidth is 152 GB/s.
Time per token: 1.76 GB / 152 GB/s = 11.6ms → 86 t/s. With MTP4: ~120 t/s.

So reaching 25% GEMV efficiency would give ~86 t/s no-spec, ~120 t/s MTP4.
Even 15% efficiency (matching the MTP4 level) would give 52 t/s no-spec, 75 t/s MTP4.

## Option 1: Custom Xe2 GEMV kernel (RECOMMENDED)

Write a DPAS-based GEMV kernel optimized for small K-quant matrices.

### Design
- Use `sycl::ext::oneapi::experimental::matrix` (joint_matrix) API for DPAS access
- Each work-group processes a tile of weight rows (e.g., 64 rows)
- Dequantize Q4_K blocks in registers, feed to DPAS as INT8 operands
- Use sub-group reductions for the dot-product accumulation
- Target: 25-35% bandwidth efficiency (2-3× current)

### Effort
- ~500-1000 lines of SYCL kernel code
- 1-2 days development + debugging
- Must verify correctness against CPU reference for all K-quant types

### Expected result
- No-spec: 38 → 60-70 t/s
- MTP4: 52 → 85-100 t/s

## Option 2: Upstream SYCL mmq.cpp rewrite

The current `mmq.cpp` is auto-ported from CUDA via dpct. A native SYCL implementation
using Intel's `joint_matrix` API could be much faster.

### Key files
- `ggml/src/ggml-sycl/mmq.cpp` (3030 lines, dpct-ported)
- `ggml/src/ggml-sycl/vec-dot-q4_K-q8_1.cpp` (quantized dot product)

### Effort
- Multi-week effort (rewrite all quantized matmul kernels)
- High risk (affects all models, not just Mamba)
- Should be upstreamed to llama.cpp

## Option 3: vLLM XPU with native Mamba2 kernels

vLLM has better Mamba abstractions (`MambaMixer2`, dedicated state caches, ReplaySSM)
but lacks optimized Intel XPU selective-state-update kernels.

### Requirements
1. XPU-native `causal_conv1d_update` (width 4, BF16, FP32 accumulate, in-place state)
2. XPU-native `selective_state_update` (d_state=128, head_dim=64, FP32 recurrence)
3. Xe2 subgroup layout (subgroup size 16, 8 d_state fragments per subgroup)
4. An exact supported checkpoint format and XPU dispatch. GPTQ-INT4 is the
   documented B70 XPU path; AWQ, compressed-tensors W4A16, and FP8 require an
   exact architecture/checkpoint load test. Nemotron NVFP4 is not natively
   supported.

### Effort
- 2-3 weeks kernel development
- Plus vLLM integration and testing
- Concurrent batching would give additional gains at Cn > 1

## Option 4: Q4_0 quantization test

Q4_0 has simpler dequantization than Q4_K (no per-block scale lookup). The GEMV
kernel for Q4_0 might achieve higher bandwidth efficiency.

### Effort
- Re-quantize model to Q4_0 (1-2 hours)
- Test llama-bench tg128
- Quality comparison (Q4_0 has lower quality than Q4_K_M)

### Expected result
- Unknown — could be +10% or no change. Worth a quick test.

## Recommended next step

**Test Q4_0** (quickest, Option 4) — if it's significantly faster, the bottleneck
is confirmed as Q4_K dequantization overhead, and a custom Q4_K GEMV kernel is justified.

Then pursue **Option 1** (custom Xe2 GEMV kernel) as the primary path to 75+ t/s.
