<!-- Retained 2026-10-09 from a delegated agent run (model: GLM-5.3). Format x engine kernel landscape on Intel XPU (agent survey, E0 leads). Not an authority; see research/2026/b70-untried-levers-synthesis-20261009.md. Known corrections: Section 12 recommendations were filtered against local closures in the synthesis note: AWQ W4A16 sits in the closed W4A16 family for Qwen3.8-27B (ADR-0010); DFlash/EAGLE-3 retries on quantized targets are pre-rejected by research/2026/qwen38-dflash2-objective-campaign-20260830.md section 12.4. -->

# Quantization × Engine Kernel Landscape on Intel XPU / Arc GPUs — primary-source map (as of 2026-10-09)

Scope: vLLM XPU (+vllm-xpu-kernels), SGLang XPU (+sgl-kernel-xpu), llama.cpp SYCL & Vulkan, OpenVINO/GenAI, IPEX-LLM, IPEX, exllamav3 XPU forks, Strata, mistral.rs, MLC-LLM, AutoRound, Neural Compressor, sycl-tla. Labels: **[main]** = merged in upstream main; **[open PR]**; **[fork]**; **[lead]** = community post, not vendor-verified; **[UNVERIFIED]** = could not confirm from a primary source.

Cross-cutting 2026 status shocks first, then per-engine detail, then KV/spec-decode matrices, 2026 Arc perf reports, and per-regime recommendations for the B70.

---

## 0. Two ecosystem exits (2026) that change planning

- **IPEX-LLM is archived.** `intel/ipex-llm` was archived by the owner on **2026-01-28** ("This repository was archived… now read-only"): https://github.com/intel/ipex-llm . Do not build new routes on it. A community datapoint even before archival: on Arc B580, mainline llama.cpp **Vulkan** beat the IPEX-LLM portable build in tok/s — https://github.com/intel/ipex-llm/issues/12991 **[lead]**.
- **Intel Extension for PyTorch (IPEX) is EOL.** Official page: "Intel® Extension for PyTorch* will reach end of life (EOL) by the end of **March 2026**… Intel CPU and GPU hardware support has been upstreamed into native PyTorch*" — https://pytorch-extension.intel.com/ . The GitHub repo shows the archived banner: https://github.com/intel/intel-extension-for-pytorch/releases . The XPU WoQ int4 (g448/g128) prototypes (e.g. https://intel.github.io/intel-extension-for-pytorch/xpu/2.6.10+xpu/tutorials/llm/int4_weight_only_quantization.html) survive as **native PyTorch fused kernels** — GPTQModel documents "Intel Torch 2.8 fused kernel support for XPU [`Arc` + `Datacenter Max`]": https://github.com/ModelCloud/GPTQModel (kernel matrix section). Practical consequence: vLLM/SGLang XPU now depend on PyTorch-XPU + their own SYCL kernel packages, not IPEX.

---

## 1. vLLM XPU (`vllm-project/vllm` + `vllm-project/vllm-xpu-kernels`) — **[main]**, most active engine

Kernel library: `vllm-xpu-kernels` (moved from IntelLabs to the vllm-project org; wheels on PyPI, latest line v0.1.15.x; requires PyTorch 2.14.0+xpu + oneAPI 2026.1; SYCL/DPC++ + oneDNN): https://github.com/vllm-project/vllm-xpu-kernels . After vLLM RFC #33214 (https://github.com/vllm-project/vllm/issues/33214) vLLM-XPU migrated onto this package — "Installing the latest vLLM for XPU will pull in vllm-xpu-kernels automatically" (README). Release cadence 2026: v0.1.13→v0.1.15 covering DeepSeek-V4, Gemma, GDN spec-decode correctness, Python 3.14 — https://github.com/vllm-project/vllm-xpu-kernels/releases .

### 1a. Quant formats on XPU (native vs emulation)
Authoritative list, `XPUPlatform.supported_quantization` in `vllm/platforms/xpu.py` **[main]** (fetched from main, Oct 2026):
`awq, gptq, moe_wna16, auto_awq, auto_gptq, inc, fp8, deepseek_v4_fp8, mxfp4, mxfp8, fp8_per_tensor, fp8_per_block, fp8_per_channel, online, gpt_oss_mxfp4, modelopt, compressed-tensors`
— source file: https://github.com/vllm-project/vllm/blob/main/vllm/platforms/xpu.py .

- **GPTQ / AWQ int4 W4A16 (sym/asym, group sizes)** — supported; docs hardware matrix marks GPTQ ✅ and AWQ ✅ for "Intel GPU" while Marlin ❌, llm-compressor INT8/FP8 ❌, GGUF ❌, bitsandbytes ❌: https://docs.vllm.ai/en/latest/features/quantization/ . Kernel path = PyTorch-native fused WOQ kernels (Torch 2.8+ upstreamed IPEX work), confirmed by GPTQModel's kernel table ("Intel XPU: TorchFused, TorchFusedAWQ, FP8 Torch, Torch"): https://github.com/ModelCloud/GPTQModel . The repo's own GPTQ-INT4 G64/G128 MoE routes run on this path (registry: Nemotron 186.6 tok/s with DFlash).
- **moe_wna16** — GPTQ/AWQ-format **fused MoE W4A16** on XPU (this is the family the Nemotron/Ornith routes use) **[main]** per the platform list.
- **FP8** — per-tensor / per-channel / per-block (`fp8_per_block`, `deepseek_v4_fp8`), plus **online (dynamic) FP8** from BF16 checkpoints. vllm-xpu-kernels README: "Quantization: FP8, MxFP4 quantization and GEMM" **[main]** — https://github.com/vllm-project/vllm-xpu-kernels (also `csrc/quantization/fp8/`, `csrc/quantization/fp4/mxfp4_quant.cpp`, `csrc/moe/reorder_mxfp_scales.cpp` in the tree). vLLM XPU validated-models table lists BF16 / Online-FP8 / FP8 / MXFP4 (gpt-oss) / Int4 (Qwen3-30B-A3B-GPTQ-Int4): https://docs.vllm.ai/en/stable/models/hardware_supported_models/xpu/ .
- **MXFP4** — `gpt_oss_mxfp4` native (packed e2m1 + ue8m0 block scales) via vllm-xpu-kernels MXFP4 GEMM; gpt-oss-20b/120b are the validated MXFP4 models on Arc Pro B **[main]** (same docs table).
- **MXFP8** — in the supported list **[main]**; no published B-series perf yet.
- **W8A8 INT8** — via **`inc`** (Intel Neural Compressor W8A8) only; llm-compressor INT8 W8A8 is ❌ on Intel GPU (docs matrix above). AutoRound exports W8A16/int8-family checkpoints consumable by the `inc`/`fp8` paths (https://github.com/intel/auto-round).
- **W4A8** — **not** in the XPU list; llm-compressor INT8 (W4A8) row in the docs matrix is CPU-only. Treat as absent on XPU (and this family already failed a quality canary locally).
- **NVFP4 / Marlin / GGUF** — ❌ on Intel GPU (docs matrix). NVFP4 has no native Xe2 tensor-core path; nothing to test.

### 1b. KV cache & attention on XPU
- **KV dtypes:** fp8 (platform `fp8_dtype() == torch.float8_e4m3fn`; the repo's exl3xpu route uses fp8 KV "only dtype"); no int8/int4 KV cache in vLLM XPU. **TurboQuant KV is wired for XPU** **[main]**: `xpu.py` routes any `kv_cache_dtype` starting with `turboquant_` to a dedicated TURBOQUANT attention backend (source above). TurboQuant = Hadamard rotation + Lloyd-Max scalar quant (keys) / uniform (values); merged upstream 2026 via PRs #38479 (base), #39890 (official 3-bit/4-bit grouped modes, merged), #39931 (hybrid models + uniform quant, merged 2026-05): https://github.com/vllm-project/vllm/pull/39890 , https://github.com/vllm-project/vllm/pull/39931 , API docs https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/turboquant/ . (SGLang: still an open request — https://github.com/sgl-project/sglang/issues/21618 .)
- **Attention backends on XPU** (from `xpu.py` selector): **Flash Attention** (vllm-xpu-kernels varlen/cutlass-style chunk-prefill + paged-decode kernels), **Triton Attn** (batch-invariant fallback), **Triton MLA** + **XPU MLA sparse** (DeepSeek), **Torch SDPA** (ViT), **TurboQuant**. Build-time kernel tuples: chunk-prefill (head 64–512, paged/causal/local/sink/lse) and paged-decode (qgroup 8/16, page 16–128) — https://github.com/vllm-project/vllm-xpu-kernels/blob/main/KERNEL_CONFIGURATION.md ; README also lists "XE2 attention variants" and GDN attention. oneDNN fused SDPA is used inside engine plugins (exl3xpu measures oneDNN SDPA ~80–90 TFLOPS at head_dim 256 vs 65–70 for its FA2 path — see §6).

### 1c. Speculative decoding on XPU
vLLM main methods (docs, Sept 2026): **EAGLE-3, MTP, DFlash, draft-model, PARD, MLP, LiLiCorr, n-gram, suffix decoding, DSpark (adaptive verification)** — https://docs.vllm.ai/en/latest/features/speculative_decoding/ . DFlash/DFlash2/DSpark draft-head training via `vllm-project/speculators` v0.5.0 (2026-05/06): https://vllm.ai/blog/2026-05-28-speculators-v050 and https://developers.redhat.com/articles/2026/06/04/speculators-v050-dflash-support-and-online-training . XPU-side kernel evidence: `csrc/ngram_embedding_kernels.cpp` in vllm-xpu-kernels (n-gram spec on GPU) and release notes mentioning "GDN speculative decoding correctness" (gpt-oss MTP) — https://github.com/vllm-project/vllm-xpu-kernels . The repo already runs DFlash n7 (Nemotron, vLLM XPU) and native MTP3 (exl3xpu) — local proof both work on B70.

---

## 2. SGLang XPU (`sgl-project/sglang` + `sgl-project/sgl-kernel-xpu`) — **[main]**

Docs: https://docs.sglang.io/docs/hardware-platforms/xpu (source-only install or `lmsysorg/sglang:*-xpu` Docker; PyTorch 2.13+xpu; optimized for Arc Pro B-Series + Arc B-Series). Official kernel lib **sgl-kernel-xpu** (MIT, C++/SYCL, AOT with optional `USE_SYCL_JIT`, CUTLASS-SYCL headers): https://github.com/sgl-project/sgl-kernel-xpu . The 2025-H2 Intel GPU roadmap issue completed Feb 2026: https://github.com/sgl-project/sglang/issues/8309 .

- **Quant:** documented: BF16 / FP8 checkpoints / **online FP8**, and **native MXFP4 MoE W4A16** — packed e2m1+ue8m0 stays packed ("no dequantization to BF16"; gpt-oss-20b ≈13 GB), **Xe2/BMG-only** grouped GEMM, experts-only (attention/router BF16), plain-SiLU + clamped-swiglu — https://docs.sglang.io/docs/hardware-platforms/xpu#mxfp4-moe-quantization . Kernel surface beyond docs (from sgl-kernel-xpu `benchmark/`): `bench_awq_dequant.py`, `bench_fused_experts_wna16.py`, `bench_moe_wna16_grouped_gemm.py`, `bench_int8_gemm.py`, `bench_fp8_gemm.py`, `bench_fp8_blockwise_gemm.py`, `bench_mxfp4_blockwise_group_gemm.py`, `bench_mxfp8_blockwise_group_gemm.py`, `bench_per_token_group_quant_mxfp4.py` → AWQ W4A16, WnA16 fused-MoE, int8 GEMM, FP8 (tensor/block), MXFP4 **and MXFP8** kernels all exist **[main]**: https://github.com/sgl-project/sgl-kernel-xpu (benchmark dir). GPTQ/AWQ checkpoints load via SGLang's generic quant loaders (GPTQModel documents SGLang GPTQ/AWQ format support incl. Intel XPU: https://github.com/ModelCloud/GPTQModel ).
- **Attention:** `--attention-backend intel_xpu` (pages 32/64/128); kernels: flash-attn fwd, flash-MLA decode/prefill/sparse, GDN, lightning attention. **XPU Graphs [experimental, off by default]:** decode `full` XPUGraph; prefill `tc_piecewise` (torch.compile) or `breakable` segmented capture — https://docs.sglang.io/docs/hardware-platforms/xpu#xpu-graph-experimental .
- **KV:** no int8/int4 KV documented on XPU; TurboQuant KV is an open feature request (issue #21618 above).
- **Spec decode:** EAGLE-2/EAGLE-3, MTP, **DFLASH**, draft-model, n-gram in SGLang docs — https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/speculative_decoding.md ; XPU EAGLE kernels present (`bench_build_eagle_tree.py` in sgl-kernel-xpu).
- **Perf on Arc:** no vendor-published 2026 B-series numbers found; treat SGLang XPU perf claims as **[lead]** until benched locally.

---

## 3. llama.cpp SYCL — **[main]** (Intel-maintained inside ggml-org/llama.cpp)

Docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md .

- **Formats:** all GGUF quants incl. IQ1_S/IQ1_M/IQ2/IQ3/IQ4 families (SYCL support since 2024.4). Q4_0 mul_mat optimized 2025.2 (+21–87% across PVC/Flex/Arc/MTL/ARL — table in SYCL.md). **2026.04–05: mul_mat reorder for Q4_K/Q5_K/Q6_K/Q8_0 + fused MoE.** **2026.09: oneAPI 2026.1 build = 3.1× prefill** on Arc B570 (1331 vs 434 t/s, same code). NVIDIA/AMD SYCL plugins dropped 2026.02.
- **Flash Attention on SYCL [main]:** PR #20190 "[SYCL] support Flash Attention for fp32/fp16/Q4/Q5/Q8" merged **2026-03-08**: https://github.com/ggml-org/llama.cpp/pull/20190 . Enables tensor(row) split + quantized KV. Caveats on Xe2 (all 2026, closed unless noted): B70 long-context prefill regression (https://github.com/ggml-org/llama.cpp/issues/21862), GPU hang with `-fa` + q8_0 KV under server load on B70 (https://github.com/ggml-org/llama.cpp/issues/25692), silently-wrong-answer reports (https://github.com/ggml-org/llama.cpp/issues/27769).
- **KV quant:** `--cache-type-k/v q8_0|q4_0` works through the FA path above, with the Xe2 bug history noted; DeepSeek-V4-class models can misbehave with quantized K on **all** backends (https://github.com/ggml-org/llama.cpp/issues/25382).
- **MXFP4 GGUF (gpt-oss):** native-format support added 2025-08 with NVIDIA collaboration (https://github.com/ggml-org/llama.cpp/discussions/15095); fast paths target CUDA/CPU — SYCL path performance **[UNVERIFIED]** (assume generic dequant fallback until benched).
- **Spec decode:** EAGLE3 merged **2026-06-12** (PR #18039); **DFlash2 merged 2026-08-27** (PR #27342, "local convolution + candidate selector"); classic draft-model + lookup(n-gram). Draft-KV quantization can zero DFlash acceptance (https://github.com/ggml-org/llama.cpp/issues/25725). Public C-API exposure still open (#27469).
- **Intel perf thread:** https://github.com/ggml-org/llama.cpp/discussions/23313 ; B70 tuning thread (Aug 2026) https://github.com/ggml-org/llama.cpp/discussions/27593 **[lead]**.

## 3b. llama.cpp Vulkan on Intel — **[main]**, driver-sensitive

Full GGUF quant support via generic Vulkan kernels; **Mesa 26.1.7 doubled B580 throughput ~30 → ~67 tok/s (Ollama)** — https://github.com/ggml-org/llama.cpp/discussions/27777 **[lead]**; Vulkan beat IPEX-LLM on B580 (https://github.com/intel/ipex-llm/issues/12991 **[lead]**); A770 Vulkan 256K-context runs (https://github.com/ggml-org/llama.cpp/issues/24199 **[lead]**). No Intel-specific cooperative-matrix acceleration documented; treat Vulkan as the fallback backend that sometimes outperforms older SYCL builds purely via driver changes.

---

## 4. OpenVINO / OpenVINO GenAI — **[main]**, very active (2026.4 released 2026-09-16)

- **Weight formats (NNCF `compress_weights`)**: INT8_ASYM; INT4_SYM/INT4_ASYM with INT8 backup ("mixed-precision", `ratio`), **AWQ**, **scale estimation**, **GPTQ**, **LoRA correction**, `backup_mode`; data-free or dataset/synthetic — https://docs.openvino.ai/2026/openvino-workflow/model-optimization-guide/weight-compression.html . 2026.4 NNCF adds **GroupedMatMul support in AWQ & Scale-Estimation** (MoE experts) and ONNX **FLOAT4E2M1/E8M0** (MXFP4 import) — https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html . 2026.4 also adds **MoE weight offload to disk for INT4** (Aug-2026 release note: https://docs.openvino.ai/releasenotes ). GPU runtime = oneDNN. Model coverage now includes **Qwen3.8-27B and Muse Glimmer 30B** on CPU+GPU (release notes).
- **KV cache (GPU plugin):** INT8 KV **default**; **INT4 KV new in 2026.2** — Paged-Attention path quantizes keys BY_CHANNEL (group 16) + values BY_TOKEN; SDPA path BY_TOKEN both (worse accuracy); B580: KV −68% vs FP16 at 8K/17K; B390 iGPU decode 1.28× vs FP16 at 34K ctx — https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/docs/blog/int4_kv_cache_support_in_2026_2.md .
- **Attention:** Paged Attention (recommended) + SDPA on GPU (same blog).
- **Speculative decoding (GenAI):** **MTP for Gemma 4 / Qwen3.5 / Qwen3.6 / Qwen3.8 on CPU+GPU**; **DFlash preview for Qwen models on GPUs** (extended to Qwen3.6 + VLMPipeline); **EAGLE-3 with Top-K tree drafting** (incl. VLMs); Model Server: MTP+DFlash preview, EAGLE-3 — all in the 2026.4 release notes (link above).
- Local baseline already measured: OV int4 Qwen3.8-27B = 28.5 t/s @150W roofline, clean quality (registry).

---

## 5. Strata (`Niko1221/Strata`) — **[main]** (SYCL port experimental)

One-click Qwen3.8-Flash-Next stack; own CUDA engine ported to SYCL in `sycl/` (PR #423 by maxfridbe; experimental in 0.1.39/0.1.40, AOT `bmg-g31`/`bmg-g21`): https://github.com/Niko1221/Strata ; Intel port docs: https://github.com/Niko1221/Strata/blob/main/docs/INTEL_ARC.md and https://github.com/Niko1221/Strata/blob/main/docs/INTEL.md .

- **Formats:** GGUF (IQ1_M/IQ2_XS/Q2_0/Coder + Unsloth UD variants); native expert pack; **~3× llama.cpp SYCL on the same GGUF** (INTEL.md comparison table: 78.2 vs 23–25 tok/s decode, 790 vs ~150 tok/s prompt).
- **KV:** q8_0 KV standard (~20 KiB/token on Flash-Next's 12 full-attn layers), **kv_q4 / kv_q8 / hybrid K8V4** parity-tested (`kv_q4` bench 2026-09-27; `kv_hybrid_parity` 0.1.27), `--kv-resident` host-pinned KV for 256K ctx: https://github.com/Niko1221/Strata/tree/main/bench/results .
- **Spec decode:** base-checkpoint **MTP draft layer** on SYCL — 70–87% acceptance, 3.4 tokens/round → 45–50 tok/s at 2.2K ctx; suffix drafter also present; **multi-card `--layer-split`** (not tensor split).
- **B70 numbers (all primary-repo, community-run on Arc):** Coder IQ1_M all-experts-resident: **78.2 tok/s** short / 66.6 after 128K / 55.9 after 256K; prompt 790–1,117 tok/s; **2×B60 layer-split**: Coder IQ1_M 56.6 tok/s / 428 t/s prompt, Flash-Next IQ2_XS 58.6–61.3 tok/s (https://github.com/Niko1221/Strata/blob/main/bench/results/2026-10-04-community-2x-arc-pro-b60/README.md); streaming case: 11.5k/24.6k experts resident → 30–41 tok/s decode, 980–1,000 t/s prefill. Kernels reach only 130–160 GB/s of the card's ~600 GB/s pin (documented `mmvq_bench` sweep) — i.e. **instruction-issue-bound like EXL3, not bandwidth-bound** (INTEL.md "Speed work" section).

---

## 6. exllamav3 XPU forks — **[fork]** (upstream exllamav3 has no XPU)

- **`sybil-solutions/exl3xpu`** (basis of the repo's champion route): vLLM plugin, ESIMD kernels decode trellis bit-exactly (vector M=1/2/4, DPAS M=3..64), logits top-1 99.63% / KL 9.8e-5 vs CUDA; MTP k=3; fp8 KV 272,570-token pool; **int8 prefill linears via oneDNN s8 GEMM (1.89× on all linears @4K, NLL +0.17%)** and **oneDNN fused SDPA attention (~80–90 TFLOPS @head 256)**; headline C1 91.2 / C16 365.2 prose tok/s, prefill 2,421 t/s @4K → 911 t/s @254K; TP not supported (data-parallel `--dp`): https://github.com/sybil-solutions/exl3xpu .
- **`0xSero/exl3xpu`** + **`0xSero/trellis-serve`** (published attested image `ghcr.io/0xsero/exl3xpu`, local-ai-registry recipes): https://github.com/0xSero/exl3xpu .
- **`Continuum-AI-Corp/OrcaSAQ2-kernel`** — runs searched mixed-precision trellis checkpoints on vLLM or exllamav3's own engine (2/3-bpw EXL3 class): https://github.com/Continuum-AI-Corp/OrcaSAQ2-kernel .
- Related Intel serving stack: `intel/llm-scaler` (the docker base used by exl3xpu; PP=2 across Battlemage cards has open crash reports — https://github.com/intel/llm-scaler/issues/489 ).
- Upstream: `turboderp-org/exllamav3` — no XPU support; all XPU work is fork-side.

---

## 7. mistral.rs & MLC-LLM — no viable Intel-GPU path today

- **mistral.rs**: upstream supports CUDA (FlashAttention V2/V3, PagedAttention), Metal, CPU; **no XPU/SYCL/Vulkan backend** in README/device docs: https://github.com/EricLBuehler/mistral.rs . The enabling layer (candle SYCL) is still an **[open PR]**: "candle Intel GPU / SYCL support" https://github.com/huggingface/candle/pull/3923 . Status: not testable on B70 without building the candle PR yourself.
- **MLC-LLM**: TVM-based; docs list CUDA/ROCm/Vulkan/Metal GPU targets (https://llm.mlc.ai/docs/install/gpu.html); its int4 (q4f16_1) group-quant runs through Vulkan in principle, but I found **no 2026 Intel-Arc validation or benchmark** — **[UNVERIFIED lead]**. Lower priority than the five active stacks above.

## 8. Quantizers & kernel toolkits

- **Intel AutoRound** (`intel/auto-round`, active — 2026/09 news): W2–W8 WNA16, **MXFP4/NVFP4 (2025/09)**, FP8, GGUF (`gguf:q4_k_m`) export; algorithm composition (`--algs awq,signround…`) 2026/08; export formats `auto_round/auto_gptq/auto_awq/llm_compressor/gguf`; deployment on **XPU via vLLM/SGLang/Transformers**: https://github.com/intel/auto-round . Local caveat: two AutoRound W4A16 exports joined the GPTQ canary failure (RL-087) — quality risk on Qwen3.8-27B specifically.
- **Intel Neural Compressor (INC)** — **not archived**, v3.10 (2026/03 news: FP8 KV-cache/attention static quant with AutoRound; 2025/12 NVFP4; 2025/10 MXFP8/MXFP4): https://github.com/intel/neural-compressor . It is the only W8A8-int8 route into vLLM XPU (`inc` in the platform list). Note its docs still point XPU installs at IPEX (now EOL) — pin native PyTorch-XPU instead.
- **GPTQModel** (`ModelCloud/GPTQModel`, v7.5.0 2026-09): GPTQ/AWQ/ParoQuant/GGUF/FP8/EXL3/QQQ export + XPU runtime kernels (TorchFused/TorchFusedAWQ/FP8 Torch): https://github.com/ModelCloud/GPTQModel .
- **sycl-tla** (`intel/sycl-tla`, PyPI 0.9.x): header-only CUTLASS-style SYCL GEMM/fused-epilogue templates, born on PVC/Flex (https://github.com/intel/sycl-tla). Not a user-facing engine — it is the layer below engines; sgl-kernel-xpu already consumes CUTLASS-SYCL headers (`SGL_JIT_CUTLASS_INCLUDE`). Xe2/BMG coverage beyond engine vendored copies: **[UNVERIFIED]**. Community Xe2 XMX GEMM walkthrough for B60: https://github.com/tinafengfun/sycl-xmx-gemm **[lead]**.

---

## 9. Matrix: format → fastest native Intel-GPU kernel (Oct 2026)

| Format | vLLM XPU | SGLang XPU | llama.cpp SYCL | OpenVINO GPU | Strata SYCL |
|---|---|---|---|---|---|
| GPTQ int4 W4A16 (g64/g128) | ✅ torch-fused + `moe_wna16` | ✅ wna16 fused experts / GPTQ loader | n/a | via NNCF GPTQ import | n/a |
| AWQ int4 W4A16 | ✅ (TorchFusedAWQ) | ✅ (awq kernels) | n/a | ✅ NNCF AWQ incl. GroupedMatMul (2026.4) | n/a |
| INT4 (generic sym/asym) | — | — | n/a | ✅ NNCF INT4_SYM/ASYM + INT8 backup | n/a |
| GGUF Q/K/IQ quants | ❌ | ❌ | ✅ (Q4_0/Q4_K/Q5_K/Q6_K/Q8_0 tuned; IQ since 2024.4) | ❌ | ✅ (3× llama.cpp) |
| GGUF MXFP4 (gpt-oss) | n/a (own mxfp4 path) | n/a | generic **[UNVERIFIED]** | ❌ | n/a |
| W8A8 INT8 | ✅ `inc` only | ✅ int8_gemm kernels | n/a | ✅ (static int8) | n/a |
| W4A8 | ❌ | ❌ | n/a | ❌ | n/a |
| FP8 W8A8/W8A16 (tensor/channel/block) | ✅ per-tensor/channel/block + online | ✅ tensor + blockwise | n/a | partial (via ONNX/NNCF) | n/a |
| MXFP4 (OCP) | ✅ gpt_oss_mxfp4 | ✅ MoE W4A16 grouped GEMM (BMG only) | fallback | ✅ FLOAT4E2M1 import | n/a |
| MXFP8 | ✅ in list | ✅ kernels present | n/a | n/a | n/a |
| NVFP4 | ❌ | ❌ | ❌ | ❌ | n/a |
| EXL3 trellis | fork-only (exl3xpu) | ❌ | ❌ | ❌ | ❌ |

## 10. Matrix: KV cache dtypes & spec decode

| Engine | KV options on XPU | Spec decode on XPU |
|---|---|---|
| vLLM XPU | fp8 e4m3; **TurboQuant 3/4-bit grouped [main]** (XPU-routed) | EAGLE-3, MTP, **DFlash/DFlash2/DSpark**, draft, PARD, n-gram, suffix |
| SGLang XPU | bf16/fp8 (no int4/int8 KV; TurboQuant open req) | EAGLE-2/3, MTP, DFLASH, draft, n-gram |
| llama.cpp SYCL | f16, **q8_0/q4_0 via FA [main 2026-03]** (Xe2 bug history) | draft, **EAGLE3 (2026-06)**, **DFlash2 (2026-08)**, lookup |
| OpenVINO GenAI | **INT8 default; INT4 new 2026.2** (Paged-Attn BY_CHANNEL keys) | MTP (Gemma4/Qwen3.5/3.6/3.8), **DFlash preview**, EAGLE-3 top-K |
| Strata SYCL | q8_0, **q4 / hybrid K8V4**, `--kv-resident` | **MTP draft layer (70–87% acc)**, suffix drafter |
| exl3xpu | fp8 KV (plugin, 272K-token pool) | native MTP3 |

---

## 11. 2026 Arc B580/B60/B70 performance reports (decode / prefill by format)

Primary-source or repo-published:
- **B70, Qwen3.8-Flash-Next Coder IQ1_M, Strata SYCL**: 78.2 tok/s decode (19-tok prompt), 55.9 @256K ctx; prompt 790–1,117 t/s; MTP rounds 45–50 tok/s @2.2K ctx — https://github.com/Niko1221/Strata/blob/main/docs/INTEL.md .
- **2×B60 layer-split (Strata)**: Coder IQ1_M 56.6 tok/s / 428 t/s prefill; Flash-Next IQ2_XS 58.6–61.3 tok/s / 436 t/s — https://github.com/Niko1221/Strata/blob/main/bench/results/2026-10-04-community-2x-arc-pro-b60/README.md .
- **B70, llama.cpp SYCL Q4_K_M (27B dense)**: C1 25.0 / C8 56.8 / C16 56.0 aggregate tok/s; prefill 999 t/s @4K, 629 @32K (same-card comparison table in exl3xpu README) — https://github.com/sybil-solutions/exl3xpu .
- **B570, llama.cpp SYCL, oneAPI 2026.1 vs 2025.3**: prompt processing 1,331 vs 434 t/s (3.1×) — https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md (News 2026.09).
- **B580, OpenVINO INT4/INT8/INT4-KV (Llama-3.1-8B)**: KV-size −43%/−68%; B390 decode 1.28× vs FP16 @34K — https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/docs/blog/int4_kv_cache_support_in_2026_2.md .
- **B70, vLLM XPU + AutoRound INT4 (MiniMax M2.7)**: ~83.8 output tok/s @p512/o1536, 24.5K ctx ceiling — https://github.com/steveseguin/b70-optimization-lab/tree/main/repro/minimax-m27-b70-110tps-ubuntu24-20260523 **[lead]**.
- **B580 Vulkan, Mesa 26.1.7**: ~30 → ~67 tok/s (7–8B Q4) — https://github.com/ggml-org/llama.cpp/discussions/27777 **[lead]**; aggregator ~28 tok/s Llama-3.1-8B Q4_K_M llama.cpp — https://runaihome.com/blog/intel-arc-b580-local-ai-guide-2026 **[lead]**.
- Regime picture on Xe2: **decode** winners are the lowest-bits-per-token formats with fused/issue-efficient kernels (EXL3 4bpw ~63 t/s @230W local; Strata IQ1_M 78 t/s MoE), because SYCL matvec kernels sit at 130–160 GB/s (Strata) / 381 GB/s (EXL3) of the 608 GB/s pin — **instruction-issue-bound, not pure BW-bound**. **Prefill** winners are int8/fp8 XMX GEMM paths (exl3xpu int8 linears 1.89×; oneAPI 2026.1 3.1×; vLLM/SGlang fp8+int8 GEMMs).

---

## 12. What to test on B70 next, per regime (excluding EXL3-4bpw / GPTQ-W4A16 / IQ3_XXS already tried)

### Dense decode (Qwen3.8-27B-class, bandwidth/issue-bound)
1. **AWQ W4A16 via vLLM XPU** — the only untried W4A16 *method* with first-class kernels (`awq` + TorchFusedAWQ in GPTQModel matrix; different scale-search from GPTQ/AutoRound). One canary shot is justified despite the family history (RL-087 failures were GPTQ + AutoRound, not AWQ). Ceiling ≈ GPTQ route (~4.9× slower than EXL3 locally → not a champion candidate; this is a quality-diversification test).
2. **EXL3 3bpw or mixed-precision trellis via OrcaSAQ2/exl3xpu `EXL3_ALL_CODEBOOKS`** — same proven kernel stack, 25% fewer bytes/token → theoretical +33% decode if issue-bound stays flat; needs quality gate (2/3-bpw trellis on a 27B is aggressive). Cheap to test: rebuild flags + new artifact.
3. **OpenVINO NNCF INT4-AWQ + INT4 KV (2026.2+)** — quality-clean route already at 28.5 t/s @150W roofline; re-bench @230W with INT4 KV + Paged Attention for the quality-sensitive/long-context lane. (Not a speed champion; a robustness champion.)

### MoE decode (Flash-Next-class)
1. **Strata Coder IQ1_M / Q2_0 with MTP on one B70** — 78 t/s documented all-resident, 70–87% MTP acceptance; swaps the repo's IQ3_XXS 48 t/s route for +40–60% at equal-or-better VRAM headroom (IQ1_M < IQ3_XXS size).
2. **vLLM XPU MXFP4 MoE (gpt-oss-format checkpoints)** — the only fused W4A16-MoE path with vendor CI; validate with gpt-oss-20b (fits trivially), then any Qwen MoE MXFP4 export via AutoRound/llm-compressor format. Watch for: experts-only quant (attention stays BF16) and BMG-only guard in SGLang's variant.
3. **2-card `--layer-split` (Strata) instead of two independent services** for the big MoE — documented 56–61 t/s on 2×B60 with *all experts in VRAM*; on 2×B70 it should clear the single-card 48 t/s IQ3_XXS route by keeping the expert working set resident (P2P copy 11–12 GB/s is enough for layer boundaries, unlike TP2).

### Prefill (compute-bound; XMX int8/fp8)
1. **Online/dynamic FP8 or INC W8A8 int8 on vLLM XPU for the BF16 dense model** — prefill linears in int8/fp8 XMX GEMMs (exl3xpu measured 1.89× on linears; vLLM `online`/`inc` are the stock equivalents). No offline weight-quality loss (dynamic), sidesteps the W4A16 canary entirely.
2. **SGLang XPU prefill graphs (`tc_piecewise` torch.compile + int8/fp8 GEMM kernels)** for batched prefill throughput; untested on B70 publicly → fresh territory, kernels exist **[main]**.
3. **oneAPI 2026.1 rebuild of llama.cpp SYCL** — 3.1× prefill on B570 for free (compiler-only); apply to the Strata/llama.cpp fallback lane before any other work there.

### Long context (256K)
1. **KV compression A/B: TurboQuant 3/4-bit KV (vLLM XPU, merged) vs fp8 KV** — directly attacks the exl3xpu C16 KV-pool wall (~14/16 streams fit); quality gates needed (grouped 4-bit on 27B at 256K is the risk point).
2. **OpenVINO INT4 KV (BY_CHANNEL keys, Paged Attention)** — vendor-validated accuracy-equivalence on INT4-weight models; pairs with the OV int4 route for a 2× KV footprint cut vs INT8.
3. **Strata `--kv-resident` + hybrid K8V4** — documented 56 t/s @256K on Flash-Next; the only route with a measured 256K decode on B70 that keeps quality knobs (K8V4 = safer than q4 both-sides).

Quality-risk ledger for these picks: W4A16 → 2 prior canary failures (GPTQ + AutoRound on this model); W4A8 → 1 failure, and no XPU kernels anyway; ≤2-bpw trellis/IQ1–IQ2 → not yet canaried locally, treat as high-risk; MXFP4 → experts-only quant usually clean, no local datapoint; int8/fp8 prefill-only quant → measured +0.17% NLL (exl3xpu), low risk; INT4-KV/TurboQuant-4bit → vendor-internal validation only on 8B, canary at 27B/256K required.

---

## 13. Status ledger (merged vs open vs fork)

**Merged in main:** vllm-xpu-kernels as vLLM-XPU dep (RFC #33214); XPU supported_quantization list (gptq/awq/moe_wna16/inc/fp8×3/mxfp4/mxfp8/online/modelopt/compressed-tensors); TurboQuant KV 3/4-bit + hybrid (PRs #39890, #39931; XPU routing in `platforms/xpu.py`); DFlash/DFlash2/DSpark in vLLM speculators v0.5.0; SGLang XPU + sgl-kernel-xpu (MoE/FMHA/GDN/MLA kernels; MXFP4 MoE W4A16 BMG-only); llama.cpp SYCL FA (#20190, 2026-03), EAGLE3 (#18039, 2026-06), DFlash2 (#27342, 2026-08), K-quant reorder + fused MoE (2026.04–05), oneAPI 2026.1 3.1× prefill; OpenVINO 2026.2 INT4 KV, 2026.4 MTP/DFlash/EAGLE-3 + NNCF GroupedMatMul AWQ; Strata SYCL port (#423, in-tree but labeled experimental); AutoRound MXFP4/NVFP4; GPTQModel XPU kernels; INC active.

**Open PRs / requests:** candle SYCL (mistral.rs prerequisite) — huggingface/candle#3923; SGLang TurboQuant KV — sglang#21618; llama.cpp spec-decode C API — #27469; vllm-xpu-kernels GLM-5 support — issue #224; llm-scaler PP=2 Battlemage — issue #489.

**Fork-only:** exl3xpu (sybil-solutions / 0xSero), trellis-serve, OrcaSAQ2-kernel, llama.cpp DFlash2-on-Prism community forks (discussion #29387), bryanvine/vllm-xpu & turboquant-xpu (upstreamed equivalents now in main).

**Dead / EOL:** IPEX-LLM (archived 2026-01-28); IPEX (EOL 2026-03, upstreamed to PyTorch); llama.cpp SYCL-for-NVIDIA/AMD (removed 2026-02).

**No Intel path:** Marlin, NVFP4, bitsandbytes, GGUF-in-vLLM, llm-compressor W4A8/W8A8-on-GPU, mistral.rs, MLC-LLM (unvalidated Vulkan **[UNVERIFIED]**).
