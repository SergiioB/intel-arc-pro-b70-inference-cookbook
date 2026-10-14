---
title: "06 — b70-architect Subagent Report (Condensed)"
type: research
date: 2026-08-11
status: final
---

# 06 — b70-architect Subagent Report (Condensed)

**Source:** b70-architect subagent analysis of Mamba2 decode ceiling on B70.
Full output in session agent_670541d5.

## Bottom line

> The B70 is not compute-limited by Mamba2's arithmetic. It is being limited by how
> the current llama.cpp SYCL graph expresses Mamba2 decode: hundreds of short Level
> Zero/SYCL dispatches, repeated materialization of small tensors, and mostly
> vector/scalar FP32 kernels that do not use XMX.

- 37-39 t/s is credible for the current implementation but far below the silicon ceiling
- Realistic no-spec target after purpose-built fused Mamba2 decode: **55-70 t/s**
- 75+ t/s is a stretch goal requiring both Mamba2 fusion AND effective speculation

## Theoretical decode ceiling

### XMX peak is NOT the relevant ceiling
The SSM state is only ~193 MB/token (read+write). At 608 GB/s, that's 0.32ms.
The scan does ~100-200 MFLOP/token — negligible. The bottleneck is implementation
granularity, not arithmetic or bandwidth walls.

### Active-weight bandwidth ceiling
2.86B active params at INT4 ≈ 1.5-1.75 GB/token. Theoretical: 347-405 t/s.
Realistic streamed payload: 2.5-4 GB/token → 150-240 t/s.
Actual 37 t/s is far below even the realistic ceiling.

## Why MTP4 only reaches ~52 t/s

MTP4 changes token time from 27ms to ~19ms — a 1.4× speedup. But this is much less
than the ~4 accepted tokens per verification step might suggest because:

1. The target still evaluates a multi-token Mamba sequence
2. The scan loop is serial over speculative positions
3. The target graph still contains many kernels
4. MTP itself has projection/sampling overhead
5. State checkpoints and accepted-token handling add work

MTP amortizes launch overhead but does not eliminate it.

## Kernel fusion roadmap

| Milestone | No-spec target | With MTP4 |
|-----------|---------------:|----------:|
| Current generic graph | 37-40 t/s | ~52 t/s |
| Conv+state-copy fusion | 43-50 t/s | ~60-70 t/s |
| Fused selective-update | 50-60 t/s | ~70-85 t/s |
| Full Mamba2 decode fusion | **55-70 t/s** | **75-100 t/s** |
| + projection/residual fusion | **65-75 t/s** | **90-105+ t/s** |

## vLLM XPU assessment

vLLM has better Mamba abstractions (`NemotronH`, `MambaMixer2`, ReplaySSM) but
its selective-state-update dispatch exposes only Triton, FlashInfer, and CPU —
no optimized Intel XPU backend. `vllm-xpu-kernels` has Xe2 causal_conv1d under
gated-delta-network work, but no complete Nemotron Mamba2 path.

> vLLM XPU is the better long-term integration target, but it is not presently
> safe to assume it will outperform llama.cpp for Nemotron Mamba2 without
> additional XPU kernel work.

## Xe2 hardware mapping

**Should use XMX/DPAS:** Mamba in/out projections, attention Q/K/V/O, MoE experts
**Should NOT use DPAS:** depthwise conv, SiLU/softplus, exp, state update, C·state reduction

> The solution is not "make SSM_SCAN hit XMX"; it is "make SSM_SCAN one efficient
> vector kernel instead of a chain of generic operations."
