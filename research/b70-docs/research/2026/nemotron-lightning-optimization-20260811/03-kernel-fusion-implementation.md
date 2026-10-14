---
title: "03 — Kernel Fusion Implementation (Phase 1-3)"
type: research
date: 2026-08-11
status: final
---

# 03 — Kernel Fusion Implementation (Phase 1-3)

## Strategy

Extend existing SYCL ops (`GGML_OP_SSM_CONV`, `GGML_OP_SSM_SCAN`) with optional
extra `src[]` pointers. The SYCL kernels detect these and take fused code paths.
No new ggml op types needed — minimal file changes, backward-compatible.

## Phase 1: Fused Conv + Bias + SiLU ✅

**Files:** `ggml/src/ggml-sycl/ssm_conv.cpp`, `src/models/mamba-base.cpp`

**Before (3 dispatches/layer):**
```
ssm_conv(conv_x, conv1d)  →  add(result, conv1d_b)  →  silu(result)
```

**After (1 dispatch/layer):**
```
ssm_conv(conv_x, conv1d)  with src[2] = conv1d_b  →  silu(conv+bias) in-kernel
```

**Kernel change:** After computing `sumf = dot(s, c)`, if `bias_data != nullptr`:
```cpp
sumf += bias_data[channel];
sumf = sumf / (1.0f + sycl::exp(-sumf));  // SiLU
```

**Graph builder change:** Set `xBC->src[2] = model.layers[il].ssm_conv1d_b` after
`ggml_ssm_conv()` call, skip separate `ggml_add` + `ggml_silu`.

**Eliminated:** 2 dispatches × 46 layers = **92 dispatches/token**.

## Phase 2: Fused Scan + dt_bias + D×x ✅

**Files:** `ggml/src/ggml-sycl/ssm_scan.cpp`, `src/models/mamba-base.cpp`, `ggml/src/ggml.c`

**Before (6 dispatches/layer):**
```
cont(dt)  →  add(dt, dt_b)  →  ssm_scan(...)  →  mul(x, D)  →  add(y, Dx)
```

**After (1 dispatch/layer):**
```
ssm_scan(...)  with src[7]=dt_b, src[8]=ssm_d  →  dt_bias + D×x fused in-kernel
```

**Kernel changes in `ssm_scan_f32_group`:**
```cpp
// dt_bias fusion (before softplus):
if (dt_bias != nullptr) {
    dt_val += dt_bias[head_idx];
}

// D×x fusion (after scan output):
if (ssm_d != nullptr) {
    y_val += ssm_d[head_idx] * x_warp[i * stride_x];
}
```

**Critical bug fix:** `ssm_d` has shape `{1, n_head}` — one D value per head, not
per `warp_idx`. Initial code used `ssm_d[warp_idx]` (out-of-bounds read up to 4095
on a 64-element tensor). Fixed to `ssm_d[head_idx]`.

**ggml.c change:** Relaxed `GGML_ASSERT(ggml_is_contiguous(dt))` — the SYCL scan
kernel handles non-contiguous dt via stride parameters. Allows skipping `ggml_cont(dt)`.

**Eliminated:** 4 dispatches × 46 layers = **184 dispatches/token**.

## Phase 3: Split Conv (concat+transpose+state-cpy elimination) ❌ REVERTED

**Goal:** Eliminate 3 more dispatches/layer by reading conv_state + xBC directly.

**Implementation:** New `kernel_ssm_conv_split()` that reads from separate
conv_state and xBC tensors using stride-based indexing, plus in-place state update.

**Issues encountered:**
1. **Assertion failure:** Legacy assertions (`src0->ne[0] == d_conv - 1 + n_t`)
   fired because src[0] was conv_state, not concatenated input. Fixed by moving
   split-path check before assertions.
2. **Non-contiguous stride bug:** xBC is a view into zxBCdt with strides from
   zxBCdt's `nb[]`, not from its own `ne[]`. Initially computed strides from `ne[]`
   (wrong). Fixed to use `nb[] / sizeof(float)`.
3. **In-place state update corruption:** Writing conv_state in-place from within
   the conv kernel corrupted output (graph memory management issue). Setting
   `conv_state_out = nullptr` produced stale state (no update). Both paths broke
   correctness.

**Result:** Code exists in `ssm_conv.cpp` (`kernel_ssm_conv_split`) but is not
activated in `mamba-base.cpp`. Reverted to Phase 1+2 stable path.

**Lesson:** In-place state updates from within ggml compute kernels conflict with
the graph allocator's buffer lifecycle assumptions. State updates must be separate
graph nodes (cpy operations) or the state must be part of the output tensor (like
ssm_scan does with the offset trick).

## Cumulative effect

| Phase | Dispatches eliminated | Measured improvement |
|-------|----------------------:|--------------------:|
| Phase 1 (conv bias+SiLu) | 92 | ~1% (within noise) |
| Phase 2 (scan dt_bias+D×x) | 184 | ~1% (within noise) |
| Phase 1+2 combined | 276 | +0.6ms saved (~2%) |
| MMQ/XMX override (separate) | 0 | +3.7% |

**Why fusion gains were modest:** Per-dispatch overhead is ~2.2µs (not 22µs as
estimated). 276 × 2.2µs = 0.6ms out of 27ms = 2.2%. The real bottleneck is GEMV
memory bandwidth, not dispatch overhead. See `02-root-cause-bottleneck.md`.
