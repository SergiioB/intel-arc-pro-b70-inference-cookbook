---
title: "Nemotron 3.5 Lightning — optimization status"
type: research
date: 2026-08-11
status: final
---

# Nemotron 3.5 Lightning — optimization status

**Updated:** 2026-08-11 22:30 CEST
**State:** active optimisation; provisional; not for public performance claims.

## Model

- Target: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_K_M.gguf`, 23.68 GiB.
- Architecture: `nemotron_h_moe` — hybrid Mamba2-Transformer MoE.
  - 53 blocks: **46 Mamba2/SSM layers** + **7 attention+MoE layers**.
  - MoE: 128 experts, 6 routed + 1 shared per token.
  - SSM: d_inner=4096, d_state=128, dt_rank=64, n_group=8, conv_kernel=4.
  - `nextn_predict_layers = 1` — native MTP head.
- DFlash companion converted to GGUF (1.18 GiB, 6-layer dense qwen3 draft).

## Build

- `~/llama.cpp-sycl-latest/build-sycl/bin/` at `f785fc9ea` (b306).
- Includes: DFlash (#26905), Nemotron MTP (#26725), ssm_conv coalesce (#26612).
- **Custom optimizations applied** (see Code Changes section below).

## Measured evidence (all provisional)

| Config | Cap | Decode t/s | Notes |
| --- | ---: | ---: | --- |
| No-spec llama-bench tg128 (original) | 150W | **36.83 ± 0.03** | baseline |
| No-spec llama-bench tg128 (fused+MMQ) | 150W | **38.18 ± 0.08** | +3.8% |
| MTP4 HTTP p~150/g128 (original) | 230W | **~52** | 95-100% acceptance |
| MTP4 HTTP p~150/g128 (fused+MMQ) | 230W | **~50-54** | within noise |
| MTP8 HTTP | 230W | ~49 | lower acceptance offsets drafts |
| DFlash7 HTTP | 230W | ~42 | 75-92% acceptance |

Qwen MoE 35B-A3B reference (llama.cpp, master 0804, 150W): **72.6 t/s** decode.

## Root-cause analysis (complete)

### The bottleneck is NOT launch overhead
The architect's initial estimate of ~22µs/dispatch × 600 dispatches = 13ms was too high.
Actual per-dispatch overhead is ~8µs (ggml graph reuse already reduces host-side scheduling).
Total dispatch overhead ≈ 2ms out of 27ms/token — only 7%.

### The bottleneck IS GEMV bandwidth efficiency
Nemotron achieves only **11% of peak 608 GB/s bandwidth** on decode (vs Qwen's 36%).
The gap is from the 92 small Mamba GEMV operations (46 layers × in_proj + out_proj):

| Model | Effective BW | Efficiency | Why |
| --- | ---: | ---: | --- |
| Nemotron no-spec | 65 GB/s | 11% | 92 small GEMVs, poor coalescing |
| Nemotron MTP4 | 92 GB/s | 15% | batch amortization (M=5) |
| Qwen MoE | 218 GB/s | 36% | fewer, larger GEMVs |

### `__dp4a` is scalar-emulated on Xe2
The SYCL MMVQ kernel uses `__dp4a` for INT8 dot products, which is emulated as
**4 scalar INT8 multiplies** (not using XMX/DPAS). The 256 XMX engines sit idle.
The MMQ kernel DOES use XMX via DPAS, but was not dispatched for M=1 (MMVQ took priority).
**Fix applied**: force MMQ for K-quants when XMX is available (`ggml-sycl.cpp`).

## Code changes (in llama.cpp-sycl-latest worktree)

### Phase 1: Fused conv bias + SiLU (`ssm_conv.cpp`)
- Conv kernel optionally reads bias from `dst->src[2]`, applies bias + SiLU in-kernel.
- Eliminates `ggml_add` + `ggml_silu` dispatches (2 per layer × 46 = 92 total).
- Graph builder sets `xBC->src[2] = ssm_conv1d_b` (`mamba-base.cpp`).

### Phase 2: Fused scan dt_bias + D×x (`ssm_scan.cpp`)
- Scan kernel optionally reads dt_bias from `dst->src[7]`, ssm_d from `dst->src[8]`.
- Adds dt_bias before softplus, adds D×x skip to output.
- Eliminates `cont(dt)` + `add(dt,bias)` + `mul(x,D)` + `add(y,Dx)` (4 per layer × 46 = 184).
- Relaxed dt contiguity assertion in `ggml.c` (SYCL kernel handles non-contiguous strides).
- Fixed ssm_d indexing: shape is `{1, n_head}`, access `ssm_d[head_idx]` not `ssm_d[warp_idx]`.

### MMQ/XMX dispatch override (`ggml-sycl.cpp`)
- When `SYCL_USE_XMX` is defined, prefer MMQ (XMX/DPAS) over MMVQ (scalar dp4a).
- Fixes scalar `__dp4a` emulation bottleneck for K-quant GEMV.

### Phase 3 (attempted, reverted): Split conv kernel
- Tried to eliminate concat + transpose by reading from conv_state + xBC directly.
- Hit correctness issues (in-place state update conflicts with graph memory management).
- Code exists in `ssm_conv.cpp` (kernel_ssm_conv_split) but is not activated in mamba-base.cpp.

## Path to 75+ t/s

The current ceiling is ~54 t/s (MTP4 230W). Reaching 75 requires improving the
INT4 GEMV bandwidth efficiency from 11% to ~25%+. Options:

1. **Custom Xe2 GEMV kernel** — write a DPAS-based GEMV kernel optimized for
   small K-quant matrices (K=2688-4096). The current MMQ kernel tiles for large M
   and wastes most tiles at M=1. A GEMV-specific kernel using sub-group reductions
   could achieve 2-3× higher bandwidth efficiency.

2. **Upstream SYCL optimization** — the `mmq.cpp` and `vec-dot-q4_K` kernels are
   ported from CUDA via dpct. A native SYCL implementation using Intel's
   `joint_matrix` API or `ext::oneapi::experimental::matrix` could be much faster.

3. **vLLM XPU with native Mamba2** — vLLM has better Mamba abstractions but
   lacks optimized XPU selective-state-update kernels. Would require kernel development.

4. **Different quantization** — Q4_0 might achieve better GEMV efficiency than Q4_K
   due to simpler dequantization (no per-block scale lookup). Worth testing.
