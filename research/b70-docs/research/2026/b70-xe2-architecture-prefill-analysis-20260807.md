---
title: "B70 (Battlemage / Xe2) Architecture & Prefill Optimization Analysis"
type: research
date: 2026-08-07
status: final
---

# B70 (Battlemage / Xe2) Architecture & Prefill Optimization Analysis

> **CORRECTION 2026-09-15:** the inferred peak table below (§ "Theoretical
> INT4 peak": ~734 TOPS INT8 MAC-counted, ~1,468 TOPS INT4) is **superseded**
> by measurement — see `research/2026/xmx-dpas-rate-classes-20260915.md`.
> Effective peaks are ~367 TOPS INT8 / ~183.5 TFLOPS BF16 (the "734" figure
> double-counted), and W4A8 does **not** exceed the INT8 rate class. The
> rest of this note (topology, kernel-stack analysis, prefill levers) stands.

**Date:** 2026-08-07 · **Purpose:** identify the next prefill-throughput levers given
the card's compute architecture and the current vLLM-XPU kernel stack.
**Method:** architecture research (Chips and Cheese, Intel docs, vllm-xpu-kernels
source, vLLM blog, community benchmarks) cross-referenced against our measured
numbers. This doc exists so we don't re-do the web search every session.

---

## Current measured ceiling (the thing we're trying to push)

Model: Qwen3.6-35B-A3B-MoE (256 experts, 8 active + 1 shared, 3B active),
GPTQ-Int4, native `cutlass_grouped_gemm_xe2` path, vLLM nightly
`v0.26.1rc1.dev457`, MTP-4, 150W cap.

| Metric | Value | Constraint |
|--------|------:|------------|
| Prefill p4k (honest, no cache) | 8,153 t/s | compute-bound |
| Prefill p8k (honest, no cache) | 8,393 t/s | compute-bound |
| Decode short/g32 (MTP-4) | 204.6 t/s | bandwidth-bound (608 GB/s) |
| Decode no-spec | 72.6 t/s | bandwidth ceiling |
| Power 150→230W prefill | ±0.2% | NOT a lever (paired A/B, Run 24) |
| Clock locking | impossible | `xe` driver, no sysfs |

**Already ruled out:** power cap above 150W (flat), GT clock locking (driver),
dynamic power boost (refuted). The bottleneck is in the kernel/compute path.

---

## (a) Confirmed architecture facts

### Xe2-core (Battlemage) internal structure
Each **Xe2-core** contains:
- **8 Xe Vector Engines (XVEs)**, each 512-bit wide (shared instruction cache).
- **8 XMX (Xe Matrix Extension) matrix engines**, each **2048-bit wide**
  (4× the vector path). XMX = dedicated systolic array, accessed via the
  **DPAS instruction** (`D = C + A×B`).
- **192 KB shared L1 / SLM** per Xe-core (Intel doc); Chips and Cheese measured
  a 256 KB block serving L1+SLM.
- **64 KB register file per XVE** → up to 8 KB/thread at max occupancy.

### DPAS tile geometry (the load-bearing hardware fact)
From the Intel Graphics Compiler `DPAS.md`:
- `D = C + A×B`, N (execution size) **= 16** on Xe2, systolic depth **SD = 8**,
  repeat count RC = M.
- **OPS_PER_CHAN scales K by precision**: TF32=1, FP16/BF16=2, **INT8/FP8=4**,
  **INT4/INT2=8**. So with SD=8: **K = 64 (INT4)**, 32 (INT8), 16 (FP16).
- Supported operand precisions: **u1, s1, u2, s2, u4, s4, u8, s8, bf16, fp16, tf32**.
  → **INT4 and INT2 are hardware-supported on XMX** (dot8 mode).

### B70 silicon = full BMG-31 (G31)
**32 Xe-cores × 8 XMX engines = 256 XMX engines** (confirmed: Intel datasheet,
ServeTheHome, TechPowerUp, Guru3D). The B60 uses BMG-21 = 20 Xe-cores / 160 XMX.
Render slice = 4 Xe-cores → B70 = 8 slices.

### Clock & bandwidth
- Boost clock **2.8 GHz**, base/graphics **2.28 GHz**, memory 2375 MHz (19 Gbps)
  → **608 GB/s**.
- L2: **18 MB** shared on BMG-31 (~4.3 TB/s). L1: 192 KB/Xe-core, ~256 B/cycle
  delivered (Chips and Cheese).

### Theoretical INT4 peak (inferred — Intel does NOT publish INT4 TOPS)
| Precision | ops/cycle/Xe-core | ×32 cores | ×2.8 GHz | Peak (FMA) |
|-----------|-------------------|-----------|----------|-----------|
| INT8 | 8,192 | 262,144 | 2.8e9 | ~734 TOPS |
| **INT4** | **16,384 (2× INT8)** | 524,288 | 2.8e9 | **~1,468 TOPS** |
| FP16/BF16 | 4,096 | 131,072 | 2.8e9 | 367 TFLOPS |

Intel quotes **367 TOPS INT8 (non-sparse)** = the MAC-counted figure (734/2).
INT4 figure unquoted; treat ~734 TOPS (MAC) as the working estimate. **FLAGGED
as inference, not spec** — the 8192-vs-4096 INT8/cycle discrepancy in Intel's
own docs means all peak numbers could halve if the real figure is 4096.

**Sources:** [Chips and Cheese – Battlemage](https://chipsandcheese.com/p/intels-battlemage-architecture),
[Intel Graphics Compiler DPAS.md](https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/instructions/DPAS.md),
[Intel Arc Pro B70 datasheet (PDF)](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2026-03/datasheet-b70-gpu.pdf),
[oneAPI Optimization Guide – Xe GPU Architecture](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html).

---

## (b) The kernel: cutlass_grouped_gemm_xe2 (vllm-xpu-kernels)

Source: [`vllm-project/vllm-xpu-kernels`](https://github.com/vllm-project/vllm-xpu-kernels)
`csrc/xpu/grouped_gemm/`. Dispatch:
```cpp
if (is_xe2_arch() || is_xe3_arch())
    cutlass_grouped_gemm_xe2(ptr_A, ptr_B, ptr_B_scale, ptr_bias,
                             ptr_D, rows_per_expert, N, K, num_experts);
```
The B70 auto-takes the `cutlass_grouped_gemm_xe2` path.

**Supported weight formats** (Xe2 kernel interface):
- **W16A16** (BF16/FP16, no scales)
- **W8A16 FP8** (e4m3/e5m2 weights, BF16 activations, per-expert FP32 scale)
- **W4A16 INT4** — `B_dtype == at::kChar` + scales; packed nibbles, scales 3D
  `[E, group_num, N]`, **group_size ∈ {32, 64, 128, 256}**
- **MXFP4 (W4A16)** — `kFloat4_e2m1fn_x2` + `uint8_t` scales

**INT4 path detail:** `is_B_int4 = (ElementB==uint8_t) && (ElementS != uint8_t)`;
4-bit weights get byte offsets halved, cast to `int4_t`, compute via
`xe_gemm_4bits<ElementA, ElementS, group_size>`. Tiling via `TiledMMA`
(`mma.tile_mnk()`), persistent-kernel scheduling across experts using
`sycl::atomic_ref` + SLM. **Lowers to DPAS = exercises XMX.**

So: **the W4A16 grouped-GEMM does hit XMX via DPAS** on the B70. Confirmed.

---

## (c) Where the prefill bottleneck sits

Prefill = large M (activation matrix dominates) → **compute-bound**. Bottleneck
hierarchy on the B70:

1. **GEMM mapping efficiency / XMX utilization (primary suspect).** The vLLM
   Arc-Pro-B blog reports **>80% capacity on B60** with the persistent MoE
   kernel, and notes that WITHOUT dynamic computation groups, thread-group
   variance **accumulates to ~15% of total MoE GEMM time**. Risks:
   - Load imbalance across the 8 active experts (`rows_per_expert` varies).
   - Small-M inefficiency: 8 active experts × modest batch may not fill 32
     Xe-cores' worth of tiles.
   - DPAS tile is 16-wide (N=16, SD=8); workgroup tile must align, else
     register-file thrash.
2. **Register file (secondary).** 64 KB / 8 KB-per-thread per XVE. INT4 GEMM
   with group_size 32-256 must hold scales + unpacked weights + accumulators.
   Intel joint-matrix guide **mandates `-ze-opt-large-register-file`** for GEMM
   kernels; if the nightly image wasn't built with it, occupancy is starved.
3. **L1/SLM and L2 (tertiary).** 256-expert MoE = huge weight footprint → poor
   L2 hit rate, weights stream from GDDR6. But prefill is compute-bound, so
   bandwidth is not the wall here (it IS the decode wall).
4. **Memory bandwidth (NOT prefill, IS decode).** 608 GB/s caps decode.

**Best call:** the prefill wall is **XMX utilization via GEMM-tile / group-
scheduling efficiency** + possibly **register pressure** if the build lacks
`-ze-opt-large-register-file`. The blog's own ">80% capacity" implies ~20%
headroom lost to exactly these effects.

---

## (d) Optimization candidates (ranked by expected impact)

1. **Verify the build has `-ze-opt-large-register-file`** (Intel joint-matrix
   guide mandates it for GEMM). Without it, occupancy is register-starved on
   the 64 KB XVE RF. Check the nightly image's IGC flags; if absent, this is the
   single biggest lever. Build vllm-xpu-kernels from source with it if needed.
2. **Confirm persistent-loop + dynamic-computation-group path** (vLLM blog
   Optimizations 1 & 2 — eliminated the 15% MoE-GEMM variance, >80% capacity on
   B60). If our nightly predates these, updating is the biggest lever.
3. **Tune workgroup tile (`mma.tile_mnk()`) to the DPAS-16 systolic geometry**
   and to the 8-active-expert M-distribution. K-blocks of 64 (INT4) are natural.
4. **INT4 group_size**: kernel supports {32,64,128,256}. Our checkpoint is
   group_size=128 (GPTQ default). Larger (256) reduces scale traffic / dequant
   overhead → may free L1/register pressure at small accuracy cost. Smaller
   (64/32) = better accuracy, more overhead.
5. **MXFP4 vs INT4**: kernel supports both. MXFP4 has a dedicated fast
   conversion path (Bitcast-bf16 trick, avoids heavy FP→BF16 instruction
   overhead per the blog). If accuracy permits, MXFP4 may prefill faster.
   (Caveat: Run 13-14 found MXFP4 not competitive at 0.17-xpu — re-test on
   nightly before assuming.)
6. **Attention fusion / graph capture**: ensure oneAPI 2026.0+ so FlashAttention
   is captured inside the SYCL graph (avoids host-device scheduling stalls).
7. **max-num-batched-tokens**: we use 8192 (the Run-20 prefill fix). Test
   16384/32768 — more prefill parallelism may help if the kernel tiles well,
   or hurt if it adds overhead. UNTESTED on XPU.

**Probably dead ends** (don't pursue without new evidence): power cap > 150W
(refuted), clock locking (driver), ngram/Eagle spec on this checkpoint (no
weights / GDN kernel block).

---

## (e) Grounding — other B70 numbers (for context)

| Source | Model | Backend | Prefill | Decode |
|--------|-------|---------|--------:|-------:|
| **This repo (Run 24)** | Qwen3.6-35B-A3B GPTQ-Int4 | vLLM XPU + MTP4 | **8,393 (p8k)** | **204.6** |
| This repo (Run 17) | same, no spec | vLLM XPU native int4 | 9,094 (p2k) | 72.6 |
| PMZFX | Qwen3.6-35B-A3B MoE | vLLM XPU, single B70 | 615 | 54.7 |
| Puget Systems | Qwen3.6-35B-A3B | vLLM, 4×B70 TP4 | — (TTFT 113ms) | 16.3 (1-user) |
| vLLM blog | GPT-OSS-120b MXFP4 | 4×B60 TP4 | — | 1,495 agg |

Our single-B70 prefill (8.4K) is already the top of the public range — the
9,094 no-spec p2k beats PMZFX 615 by 14×. So we're already near the kernel's
practical ceiling; gains from here are single-digit %.

**Sources:** [PMZFX/intel-arc-pro-b70-benchmarks](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks),
[Puget Systems B70 multi-GPU](https://www.pugetsystems.com/labs/articles/intel-arc-pro-b70-multi-gpu-ai-inference-performance/),
[vLLM Arc Pro B blog](https://vllm.com.cn/blog/intel-arc-pro-b).

---

## Items NOT confirmed (flagged for future verification)
- **Intel's official INT4 TOPS for the B70** — not published anywhere. The
  ~1,468 TOPS (FMA) / ~734 TOPS (MAC) figure is inferred from INT8=2×INT4 + DPAS
  dot8 docs.
- **8192 vs 4096 INT8/cycle-per-Xe-core**: Intel's own docs disagree. If the
  real Battlemage figure is 4096, all peak numbers halve. Confirm via a DPAS
  microbenchmark on the running B70.
- **Exact sustained clock under heavy INT4 GEMM load** — boost 2.8 GHz, base
  2.28 GHz; reality is between.
- **Whether the nightly image was built with `-ze-opt-large-register-file`** —
  needs a container inspect (`docker exec` + check the compiled kernel's IGC
  flags, or build from source and compare).

---

## Kernel-research findings (vllm-xpu-kernels, 2026-08-07)

**Key framing:** our nightly (v0.26.1rc1.dev457) already contains the blog's
headline optimizations (persistent-loop single-kernel, dynamic computation
groups) — those wins are banked. The blog's ">80% capacity / 15% variance
eliminated" were measured on **B60**; no B70 numbers are published anywhere.
Next gains are incremental: build flags, kernel-version dispatch heuristics,
attention-kernel coverage — not a new silver-bullet kernel.

### Highest-leverage candidates (ranked)
1. **`-ze-opt-large-register-file`** — Intel mandates it for `joint_matrix`/XMX
   GEMM ([oneAPI guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/joint-matrix.html)).
   Forces large GRF for ALL kernels; the cutlass_xe2 path compiles through IGC
   so it applies. **VERIFY it's in the nightly image's build.** If missing, this
   is the single biggest win on GEMM-heavy MoE prefill (register-starved
   occupancy). Check via `igc` dumps or rebuild the wheel with the flag.
2. **`vllm-xpu-kernels` version** — v0.1.9 "selects better tiling configs,
   reducing wasted EU cycles for small K" (our MoE has small active K). v0.1.12
   removed the legacy `xpu_fused_moe` path (confirms we're on the intended
   cutlass path). Bump to v0.1.12.1+ if the nightly pins older.
3. **Hardware-friendly 4-bit format** — blog: "naive FP4→BF16 incurs too many
   instructions... hardware-friendly format increases efficiency by up to 30%."
   Confirm the `cutlass_grouped_gemm_xe2` int4 variant uses packed/aligned
   loads, not scalar unpack. If our GPTQ-Int4 path uses naive dequant, this is a
   30% lever.
4. **`--max-num-batched-tokens` 16384** — low-effort single A/B. Prefill is
   compute-bound; more parallelism *could* help if not memory-bound. Low
   confidence (MoE routing overhead scales too). Currently 8192 (Run-20 fix).

### Attention kernel coverage (medium impact)
- `VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_full.conf` (216 kernels vs ~13 default)
- `VLLM_PAGED_DECODE_CONFIG=paged_decode_full.conf` (384 kernels)
Default build only pre-compiles Llama/Qwen/DeepSeek shapes; full build avoids
eager fallbacks for non-standard head dims. Our 128/256 expert shapes may
benefit. Set at install/build time.

### Env vars (verified)
- `SYCL_CACHE_PERSISTENT=1` — avoids re-JIT; confirm set (we currently use 0 per
  AGENTS §9.6 for llama.cpp stability — may differ for vLLM).
- `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` — forces Level Zero (required for XMX).
- `VLLM_TRITON_USE_TD=1` — HW 2D block reads for Triton fallback kernels only
  (cutlass_xe2 MoE is SYCL, so only helps non-MoE layers).
- `ZE_AFFINITY_MASK` — multi-tile/TP only; irrelevant single-stream.
- No documented `VLLM_XPU_GRAPH` / capture-mode env var. Graph mode =
  `--compilation-config` (`FULL_AND_PIECEWISE`), not XPU-specific.

### Upstream PRs to watch (vllm-project/vllm-xpu-kernels)
- **#504** `[MoE] Add SYCL reorder_mxfp_scales kernel` (Aug 3) — quantized-scale
  reordering, directly relevant to int4/MoE.
- zufangzhu PRs (Jul): `[MoE] align cutlass_grouped_gemm_interface and add
  a1q_scale` — activation+quant-scale fusion on our exact path.
- **#450** `is_xe2_arch` API (Jul 7) — better B70/Xe2 detection.
- **#507** `remove benchmark configs which don't run on B60` (Aug 6) — signals
  B60/B70 perf-shape divergence is actively tuned.
- Track **intel/intel-xpu-backend-for-triton #6389** (grouped-GEMM autotuning
  against real token distributions) — the next big lever; not landed yet.

### Confirmed dead ends (do NOT pursue)
- Clock locking / power cap — dead on `xe` driver (our A/B matches public consensus).
- Raising capture-size list for prefill — prefill runs eager/piecewise; the
  256/512 graph captures serve decode batches. Adding larger prefill captures
  won't help.
- Triton-based MoE — intel benchmarks show SYCL/cutlass_xe2 outperforms Triton
  ~10× at scale on B60 (1519µs vs 16758µs). Stay on cutlass.
- B70-specific tuning docs — none exist. All public numbers are B60.

**Sources:** [vllm-xpu-kernels repo](https://github.com/vllm-project/vllm-xpu-kernels),
[KERNEL_CONFIGURATION.md](https://github.com/vllm-project/vllm-xpu-kernels/blob/main/KERNEL_CONFIGURATION.md),
[vLLM Arc Pro B blog](https://vllm.ai/blog/2025-11-11-intel-arc-pro-b),
[intel-xpu-backend-for-triton#6389](https://github.com/intel/intel-xpu-backend-for-triton/issues/6389),
[vLLM env vars](https://docs.vllm.ai/en/stable/configuration/env_vars/).
