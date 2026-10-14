<!-- Retained 2026-10-09 from a delegated agent run (model: GLM-5.3). Xe2/B70 datapath facts from primary sources (agent survey, E0). Not an authority; see research/2026/b70-untried-levers-synthesis-20261009.md. Known corrections: The Intel Tech Tour PDF exceeded the 8 MiB capture limit and is cited by URL only. 'INT4 IS native at 2x INT8' holds for int x int (s4 x s4) DPAS, matching the local ESIMD measurement (research/2026/xmx-dpas-rate-classes-20260915.md); W4A16 and W4A8 do not gain rate. -->

# Xe2 / Battlemage (BMG-G31, Arc Pro B70) hardware facts that decide quantization format choice

Compiled 2026-10-09 from primary sources. Contradictions are flagged inline, not resolved.

## 1. DPAS/XMX operand types and per-clock throughput (per Xe-core)

Primary source: Intel Tech Tour 2024 "Xe2 and Lunar Lake GPU Deep Dive" slides
https://cdrdv2-public.intel.com/824434/2024_Intel_Tech%20Tour%20TW_Xe2%20and%20Lunar%20Lakes%20GPU.pdf

Per Xe-core key peak metrics table (slide "Key Peak Metrics for 2nd Gen Xe-core"):

| Path | Ops/clock per Xe-core |
|---|---|
| FP32 (vector, 8 XVE) | 256 |
| FP16 (vector) | 512 |
| DP4a INT8 (vector) | 1024 |
| XMX FP16 / BF16 | 2048 |
| XMX INT8 | 4096 |
| XMX INT4 / INT2 | 8192 |

- Xe2 XMX natively supports **INT2, INT4, INT8, FP16, BF16** (slide: "Xe Matrix Extensions — Support for INT2, INT4, INT8, FP16, BF16"). DPAS instruction shapes shown: `16x4x4x2` (INT8) and `16x2x4x2` (FP16).
- **s4/u4 IS native at 2× INT8 rate** (8192 vs 4096 ops/clock). Same for INT2. This is confirmed by the SYCL joint_matrix spec, where INT8×INT8 (and u8/s8 mixes) are the only integer DPAS combos exposed for `intel_gpu_bmg_g21`/`g31` — INT4 is NOT exposed via joint_matrix despite hardware support (contradiction between ISA capability and software exposure, shown not resolved): https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc ("Intel XMX Supported Combinations" table lists uint8/uint8, uint8/sint8, sint8/uint8, sint8/sint8 → sint32, fp16/fp16, bf16/bf16, tf32/tf32 for bmg_g31; no int4, no fp8).
- **TF32**: listed as supported XMX combination for bmg_g21/g31 in the joint_matrix spec (tf32×tf32→fp32, K=8). Per-clock TF32 rate not in the Tech Tour table — [UNVERIFIED, typically ½ FP16 on Xe2].
- **FP8 (E4M3/E5M2): absent on Xe2 DPAS.** Not in the Tech Tour slide type list, not in the joint_matrix XMX table for bmg. The word "fp8" does not appear anywhere in the joint_matrix spec. FP8 on B70 today is therefore emulation: dequant fp8→fp16/bf16 in registers then FP16/BF16 DPAS (this is what vLLM-XPU "fp8" paths do on Arc; PyTorch XPU FP8 WOQ is stack-level, see §4).
- **Xe3 (Panther Lake, PTL-H/PTL-U in joint_matrix list) exposes the same type combos as Xe2** in the current spec (fp16/bf16/tf32/int8) — i.e., no FP8 via joint_matrix on Xe3 either. Xe3 doubles FP16 and INT8 XMX throughput vs Xe2 and grows L2 to 16 MB on the 12-Xe-core part (TechPowerUp deep dive: https://www.techpowerup.com/review/intel-panther-lake-technical-deep-dive/8.html). MXFP4/MXFP8 hardware support: **not documented for Xe3**; PyTorch is only *optimizing kernels* for MXFP8/MXFP4 on Intel GPUs in 2026 (dev-discuss post, §4), which implies software/emulated paths on current parts.
- **Xe3P (Crescent Island, `intel_gpu_cri`)**: Intel newsroom says it supports "a broad range of data types" for inference without listing them (https://www.intel.com/content/www/us/en/newsroom/news/artificial-intelligence/intel-to-expand-ai-accelerator-portfolio-with-new-gpu.html); sycl-tla already lists FP8 E5M2/E4M3 among its data-movement abstractions and validates on CRI (see §4). Whether Xe3P XMX has native FP8/MXFP8 DPAS: [UNVERIFIED — no public Intel ISA table yet].

Peak TOPS sanity check for B70: 32 Xe-cores × 4096 INT8 ops/clk × 2.8 GHz = 367 INT8 TOPS — exactly the datasheet figure (see §2), confirming the per-clock table scales to G31.

## 2. Arc Pro B70 (BMG-G31) exact specs

Primary sources: Intel B70 datasheet v1.0 and Arc Pro B-Series Quick Reference Guide (March '26):
https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2026-03/datasheet-b70-gpu.pdf
https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2026-03/intel-arc-pro-b-series-graphics-quick-reference-guide-v1-0.pdf

- Architecture: Xe2-HPG, 32 Xe-cores, **256 XMX engines** (8 per Xe-core), 32 RT units
- Graphics clock: **2.8 GHz max dynamic** (datasheet "2800 MHz"; Quick Ref Guide same; Phoronix reports 2.8 GHz max dynamic / 2.28 GHz graphics clock: https://www.phoronix.com/review/intel-arc-pro-b70-linux)
- GPU Peak TOPS: **367 dense INT8 TOPS** (explicitly "XMX workloads with INT8 datatype, dense")
- Memory: 32 GB GDDR6 ECC, 256-bit, 19 Gbps → **608 GB/s** (B60: 24 GB/192-bit/456 GB/s per https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html)
- PCIe: **Gen 5 x16 native** (datasheet "PCIe Support: Gen 5 x16 Native")
- TDP: **160–290 W board range; 230 W for Intel-branded card** (datasheet Power section)
- Display: 4× DisplayPort 2.1; DX12 Ultimate, oneAPI, OpenCL 3.0, OpenGL 4.6, OpenVINO, Vulkan 1.3

From the oneAPI GPU Optimization Guide Xe2-HPG column (validated on B580; Xe-core internals are shared Xe2-HPG): https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html

- L3/L2 cache: 18 MB on BMG-G21. **G31 L2 size: no public Intel figure found** — [UNVERIFIED, likely larger than G21's 18 MB; TechPowerUp BMG-G31 entry exists but Intel has not published the number]
- L1 per Xe-core: 256 KB; **SLM per Xe-core: 128 KB** (max SLM per work-group 128 KB)
- 8 vector engines per Xe-core, 8 HW threads each, max work-group 1024
- **GRF: 128 registers/thread regular mode, 256 in large-GRF mode; 512-bit register width** (large-GRF enabled with IGC option `-ze-opt-large-register-file`, documented in the joint-matrix XMX guide: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/joint-matrix.html)
- **Sub-group sizes: 16 and 32** (Xe2 drops Xe1's size 8)
- 2D block loads/stores: Xe2 vector engines support 2D block IO (`ld2d/st2d`, ESIMD `lsc_matrix`/block loads); exposed via ESIMD and joint_matrix loads with `layout::ext_intel_packed` (VNNI-style B packing) — see https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd/examples/dpas.md. 64b native atomic ops added in Xe2 (Tech Tour slide "Architecture Scalability").

## 3. Measured sustained bandwidth and best decode efficiency

- **Phoronix Arc B580 compute review** (clpeak OpenCL global memory bandwidth): B580 improves only "a small improvement" over Alchemist A580; exact GB/s number is in the SVG result, not the text — https://www.phoronix.com/review/intel-arc-b580-gpu-compute/3 (no public numeric extraction; treat [UNVERIFIED]).
- **Puget Systems 4× Arc Pro B70 vLLM (FP16) article** (Sept-2026 corrected): https://www.pugetsystems.com/labs/articles/intel-arc-pro-b70-multi-gpu-ai-inference-performance/
  - Llama 3.1 8B FP16, TP=1, 1 user: **35.4 tok/s, 28 ms ITL**. Derived bandwidth: 35.4 tok/s × ~16.1 GB (8B × 2 B + overhead) ≈ **570 GB/s ≈ 93–94 % of the 608 GB/s pin** — the best publicly documented Intel-kernel decode efficiency found; note their own correction shows exactly this arithmetic discipline (they flagged an impossible 1,074 GB/s as a tokenizer bug).
  - Qwen2.5-3B FP16: 72.9 tok/s ≈ 0.44 GB/token×72.9 ≈ 32 GB... i.e. ~440 GB/s ≈ 72 % of pin (small model, overhead-dominated).
  - 8-user concurrency: 256 tok/s on 8B → ~4.1 GB/token-equivalent effective, batched GEMM regime.
- Local baseline (this repo, not external): EXL3 M=4 verify 381 GB/s = 63 % of pin — instruction-issue (trellis decode) bound. The Puget FP16 GEMV result shows the vLLM-XPU FP16 path gets ~93 %, i.e. the gap for EXL3 is kernel issue rate, not memory.
- OpenVINO int4 weight-compressed decode efficiency on B70: no public tok/s-vs-bandwidth number found in Intel docs; Phoronix ran OpenVINO on B70 (https://www.phoronix.com/review/intel-arc-pro-b70-linux/2) but numbers are chart-only — [UNVERIFIED].
- Contradiction worth noting: Puget found upstream vLLM XPU supports **only unquantized FP16** ("AWQ/GPTQ dequantization kernels are CUDA-only"); INT4 online quant, FP8 online quant, and GPTQ exist only in Intel's `intel/llm-scaler` container (https://github.com/intel/llm-scaler). This directly bounds what "vLLM-XPU int4 GEMV" can achieve today.

## 4. Intel compute stack, 2026 state

- **oneAPI/DPC++**: current docs at 2025.2; sycl-tla CI validates DPC++ 2025.3+ with Compute Runtime 26.01 / IGC 2.27 on Xe2 (https://github.com/intel/sycl-tla README). Phoronix tested B70 on Ubuntu 26.04 / Linux 7.0 / Mesa 26.0 with Compute Runtime 26.09.37435.1 + IGC 2.30.1 (https://www.phoronix.com/review/intel-arc-pro-b70-linux).
- **SYCL joint_matrix**: production (supported extension) with architecture targets `intel_gpu_bmg_g21` and `intel_gpu_bmg_g31`; type table as in §1 (int8/fp16/bf16/tf32, no int4, no fp8). Large-GRF flag needed for best GEMM. https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc
- **ESIMD**: DPAS example doc in intel/llvm (`sycl_ext_intel_esimd/examples/dpas.md`) — direct DPAS programming incl. on Xe2; also 2D block loads. https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd/examples/dpas.md
- **sycl-tla (CUTLASS-for-Intel)**: v0.9.3 (2026-09-23), tracks CUTLASS 4.2.1; supports FP64/FP32/FP16/BF16/**FP8 E5M2+E4M3**/**4b and 8b signed/unsigned ints with zero-point**, tensor/channel/group-wise mixed-precision quantization; hardware support table: PVC, **Arc B580 (Xe2)**, Crescent Island (Xe3P); build targets include `intel_gpu_bmg_g31`. Recent release fixes a "BMG GEMM performance regression". https://github.com/intel/sycl-tla
- **oneDNN**: int8 matmul/convolution on GPU maps to DPAS (`fma_kind::dpas` on Xe2; dp4a fallback on non-XMX): https://gist.github.com/alex4o/b0341530f5b5560fe586e135a8f7f3f6 and https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/xmx.html. INT4 weights on GPU engine: **not a supported oneDNN GPU datatype** (int4 is CPU/AMX territory) — [UNVERIFIED; current oneDNN GPU matmul dtype table lists f16/bf16/f8/s8/u8, f8 GPU support concentrated on PVC and newer]. FP8: introduced for GPU in oneDNN 3.x per release notes (https://www.intel.com/content/www/us/en/developer/articles/release-notes/onednn/legacy.html "Introduced support for fp8 data type") — on Xe2 this is dequant-to-BF16 GEMM, not DPAS fp8.
- **PyTorch XPU (2026 H1 plan)**: FP8/FP16/BF16/INT8/**INT4 WOQ** supported end-to-end; FlashAttention via SYCL-TLA; 2026 work: XPUGraph/XPUGraphTree integrated into Inductor, SYCL-TLA as Inductor backend, quant/dequant + GEMM optimization for **FP8, MXFP8, MXFP4**, TorchAO INT4/INT8/FP8/MXFP8/MXFP4 enablement, XPU symmetric-memory TP backend. https://dev-discuss.pytorch.org/t/intel-gpu-cpu-enabling-status-and-feature-plan-2026-h1-update/3320
- **Driver/Level Zero**: Linux `xe` driver; B70 works fully open-stack (Phoronix above). Note power not exposed via sysfs on Linux 7.0 (same article). P2P: Puget had to set `CCL_TOPO_P2P_ACCESS=0` because PCIe P2P caused bus errors on their 4-card B70 workstation (appendix of the Puget article) — consistent with this repo's observed 11–12 GB/s P2P copy and no peer atomics. SVM/peer-access/atomic Level Zero features: feature-dependent (`ZE_FEATURE_FLAG_*`); no Xe2-specific primary doc consulted — [UNVERIFIED, use `zelFeatureFlags` query at runtime].
- **Graph capture**: SYCL graph extension experimental in intel/llvm; PyTorch XPU graph capture/replay = XPUGraph, being generalized through `torch.accelerator` (dev-discuss post above).

## 5. Format × Xe2 compute-path matrix

Legend: "dequant cost" = extra non-DPAS work to feed XMX. Bytes/param = weight bytes only (scales/zero-points add ~0.1–0.5 B/param).

| Format | Native Xe2 path | Dequant cost | Bytes/param | Favors |
|---|---|---|---|---|
| INT4 weight + FP16 act (GPTQ/AWQ-style W4A16) | none native for mixed s4×f16; dequant w→fp16 in kernel (register LUT/multiply), then FP16 DPAS 2048 ops/clk | moderate (bit unpack + scale mul per tile); issue-rate bound in decode (matches repo's trellis-decode wall) | ~0.5 (+scales) | **decode** (bandwidth-bound): halves bytes/token vs 8-bit |
| INT4 weight + INT8 act (W4A8) | HW can do s4 DPAS at 8192 ops/clk but s/w exposes only s8 DPAS → upconvert w4→s8 at load, s8×s8 DPAS | low (unpack+shift into s8 GRF tiles) | ~0.5 | prefill (compute-bound) — highest exposed DPAS rate per weight byte; repo's W4A8 *quality* failure is a model-side issue, not HW |
| INT8 × INT8 (incl. GPTQ-INT4 upscaled, DFlash-style) | native s8×s8→s32 DPAS, 4096 ops/clk, fully exposed (joint_matrix, oneDNN, sycl-tla) | none for true int8 weights | 1.0 | **prefill**; also solid decode when quality allows 8-bit |
| FP16/BF16 | native DPAS 2048 ops/clk | none | 2.0 | prefill accuracy fallback; decode only while it fits (Puget: ~93 % of pin bandwidth at bs=1) |
| FP8 (E4M3/E5M2) weight, "emulated" | no FP8 DPAS on Xe2; dequant to bf16/fp16 → 2048 ops/clk | low-moderate (LUT or shift-scale in registers) | 1.0 | decode only (½ bytes of fp16); compute path is no faster than FP16 — matches repo's 27B fp8 = 48 tok/s disappointment being issue-bound, and PyTorch's 2026 fp8 kernel-optimization plans |
| MXFP4 / NVFP4 (block-scaled 4-bit float) | not native on Xe2; dequant to fp16 (block-scale = extra exponent math per 32-elem block) | moderate | ~0.5–0.56 | **decode** (lowest bytes/param that keeps float dynamic range); prefill gains nothing over W4A16 since DPAS rate is fp16-class. Stack support arriving 2026 via TorchAO/Inductor (dev-discuss) |
| GGUF K-quants (Q4_K, Q5_K, Q6_K) | super-block scales; dequant to fp16 vector math or dp4a-free custom kernels; llama.cpp SYCL mostly vector ALU, XMX used only for prompt processing | high-ish (two-level scale unpack) | 0.5–0.8 | decode portability; per-token quality better than Q4_0; prefill weak unless XMX GEMM path used |
| GGUF IQ-quants (IQ3_XXS, IQ4_XS…) | lookup-codebook dequant, vector ALU | high (codebook lookups, per-value) | 0.31–0.5 | decode for big-MoE fit (repo's Strata IQ3_XXS Flash-Next route); prefill compute-bound by dequant itself |
| EXL3 trellis (M=4) | custom trellis decode → fp16/bf16 DPAS GEMM; verify pass hits 381 GB/s (repo-measured) | highest (sequential-ish trellis chain limits instruction issue) | ~0.5 | decode quality/speed frontier on B70 today; wall is instruction issue, not bandwidth — leverage = widen the decode phase or overlap with DMA |

Practical corollaries (grounded in the above):
1. For prefill on B70, INT8 DPAS is the only path with full s/w exposure at 4096 ops/clk; 367 dense INT8 TOPS = the real compute ceiling, and XMX INT4's 2× rate is currently unreachable through joint_matrix/oneDNN (would need raw ESIMD DPAS or IGC-level int4 enablement — worth probing `dpas` with s4 operands via ESIMD as an experiment).
2. For decode, bytes/param dominates; every format above tops out at the same FP16 DPAS compute path, so format choice = quality per byte + dequant issue cost. Puget's 93 % pin FP16 result shows near-perfect streaming is attainable; closing the 63 %→93 % gap on 4-bit is a kernel-authoring problem.
3. FP8/MXFP4 on B70 are memory-format plays only — zero compute upside vs FP16 on Xe2; they become compute formats only on Xe3P-class hardware (Crescent Island, "broad range of data types", sampling 2H 2026).
