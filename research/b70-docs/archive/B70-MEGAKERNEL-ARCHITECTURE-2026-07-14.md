# B70-Native LLM Engine / “Mega-Kernel” Architecture

Date: 2026-07-14
Target: one Intel Arc Pro B70, 32 GiB VRAM
Primary models: Qwen3.6-35B-A3B and Ornith-1.0-35B
Secondary controls: Qwen3.6-27B, Gemma-4-26B-A4B, gpt-oss-20B
Status: design and research complete; execution blocked until reliable access to the B70 host returns

## 1. Verdict

A personalized B70-native engine is technically viable and the Qwen/Ornith 35B-A3B family is the best first target.

The highest-probability design is not one literal transformer-sized C++ kernel. It is:

1. A thin Level Zero runtime with model weights and mutable state permanently resident in VRAM.
2. BMG-AOT-compiled SYCL ESIMD kernels, using DPAS/XMX only for shapes where matrix execution wins.
3. A small model-specific kernel family for decode, prefill, Gated DeltaNet, attention, MoE, MTP and sampling.
4. A model-specialized captured execution graph using SYCL Graph or reusable Level Zero command lists, selected by measurement on the pinned runtime.
5. Operation-level persistent MoE/attention schedulers where useful. A whole-model cooperative kernel remains an experimental research branch, not a production requirement.
6. A llama.cpp/GGUF loader and tokenizer initially, but not the general ggml graph executor in the final fast path.

Intel’s Apache-2.0 `llm-scaler` tree is the strongest current kernel donor. It already contains BMG-specific ESIMD kernels for INT4/FP8 GEMV/GEMM, fused residual/RMSNorm/GEMV, GDN recurrence, paged attention, Qwen MoE, Gemma MoE and vectorized TopK. The custom engine should reuse and specialize these ideas rather than starting from generic SYCL.

The likely result is a substantial model-specific improvement, not an unlimited speedup. For single-token decode:

- Dense 27B remains fundamentally limited by reading almost every quantized weight for every token.
- Sparse 35B-A3B is the better target because only selected experts are read, while its many tiny operations leave much more launch, load-width, routing and synchronization overhead to remove.
- MTP/speculative verification is the only credible way to generate multiple accepted tokens per target-model weight pass and exceed the ordinary one-token memory roofline.

## 2. B70 resources that matter

Official B70 figures:

- Architecture: Xe2-HPG / Battlemage, BMG-G31.
- 32 Xe-cores across 8 render slices.
- 256 Xe Vector Engines.
- 256 XMX matrix engines.
- Maximum graphics clock: 2.8 GHz; published base/reference clock: 2.28 GHz.
- Peak dense INT8 matrix throughput: 367 TOPS at maximum clock.
- 32 GiB GDDR6 on a 256-bit bus.
- 608 GB/s theoretical memory bandwidth.
- PCIe 5.0 x16.
- Typical reference-board power: 230 W; partner range 160–290 W.

Xe2 execution characteristics from Intel’s architecture and oneAPI optimization material:

- Eight Vector Engines and eight XMX engines per Xe-core.
- 64 hardware thread contexts per Xe-core in the documented Xe2 mapping: 2,048 hardware threads across the B70.
- ESIMD maps one explicit-SIMD work-item to one hardware thread and permits direct vector, block-load, SLM and DPAS control.
- Work-groups execute within one Xe-core. Intel documents 256 KiB L1 and up to 128 KiB SLM per Xe-core/work-group; Battlemage measurements indicate L1 and SLM share physical capacity, so the effective split must be profiled.
- Supported ordinary subgroups include 16 and 32; ESIMD uses an explicit logical subgroup of one hardware thread.
- Xe2 adds efficient 2D block loads, larger explicit vectors, hardware named-barrier support in IGC and Level Zero cooperative-kernel support.

Derived arithmetic ceilings at 2.8 GHz, based on Intel’s per-Xe-core Xe2 operations/clock:

- BF16/FP16 matrix: approximately 183.5 TOPS.
- INT8 matrix: approximately 367 TOPS, matching Intel’s published B70 figure.
- INT4/INT2 matrix: approximately 734 TOPS only when **both operands** are sub-8-bit. W4×A8 remains in the approximately 367-TOPS class; W4 expanded to FP16 uses the approximately 183.5-TFLOP/s FP16 path plus conversion overhead.

These derived figures are not application throughput. Batch-1 autoregressive generation is usually constrained by memory traffic, tiny matrix dimensions, reductions and dispatch latency long before peak XMX arithmetic. The natural Xe2 DPAS tile is approximately M8×N16 with K16 for FP16/BF16, K32 for INT8 or mixed low-bit×INT8, and K64 when both sources are low-bit.

GRF and SLM discipline is critical:

- Each thread has 128 regular or 256 large 64-byte GRFs. Large-GRF mode can roughly halve occupancy.
- A measured Xe2 attention regression caused a 4 KiB spill and approximately 33% runtime loss; reject any selected kernel with spills.
- SLM uses 16 banks at four-byte granularity. Pad or skew tiles to avoid stride-16-dword bank conflicts.
- Prefer subgroup shuffles/reductions over full work-group barriers where possible.

Published third-party L2 cache figures conflict between 16 and 24 MiB. Do not encode either value as a design assumption. Query the running device/driver and use cache-counter measurements.

## 3. Critical insight: XMX is not always the fastest path

“Use all XMX engines” is the wrong objective for batch-1 decode.

For M=1 GEMV:

- The operation has too little reuse to feed matrix tiles efficiently.
- Weight traffic dominates.
- Wide, cache-line-aligned ESIMD block loads plus vector dequantization and FP32 accumulation can beat DPAS.

Intel’s current `llm-scaler` source documents the exact BMG failure mode:

- An MoE DPAS path used 16-byte-wide 2D loads and reached about 316 GB/s.
- A decode-specific ESIMD GEMV changed to contiguous 1D loads matching the 64-byte cache line.
- The dense BMG GEMV reference in the same source reaches about 574 GB/s.
- Another BMG FP8 GEMV source claims approximately 97% bandwidth utilization on large shapes.

Therefore the custom dispatcher should use:

| Shape | Preferred execution |
|---|---|
| M=1 decode | ESIMD wide-load GEMV, vector unpack/dequant, FP32 accumulation |
| M=2–4 | Batched ESIMD GEMV or split-K kernels |
| M=8–64 | DPAS/XMX GEMM with 2D block loads and weight-stationary tiling |
| Long prefill | DPAS/XMX Flash Attention and GEMM |
| MoE batch/prefill | Token grouping by expert plus grouped DPAS/XMX GEMM |
| MTP candidate verification | Multi-row DPAS/XMX when enough candidates survive |

The engine’s goal is maximum useful throughput and minimum latency, not maximum XMX occupancy as an isolated metric.

## 4. Best first model: Qwen3.6/Ornith 35B-A3B

The target architecture is unusually suitable for specialization:

- 40 layers.
- Hidden size 2,048.
- Pattern: ten repetitions of three Gated DeltaNet layers followed by one full-attention layer.
- 30 GDN layers and 10 full-attention layers.
- 256 routed experts, top-8, plus one shared expert.
- Expert intermediate size 512.
- 3B advertised active parameters from 35B total.
- GDN: 16 key heads, 32 value heads, dimensions 128/128.
- Full attention: 16 query heads, 2 KV heads, head dimension 256.
- Native MTP training.

Why it is preferable to Qwen-27B dense:

1. Only selected expert weights need to be streamed during each token.
2. The 256-expert/top-8 dimensions are fixed and already targeted by Intel’s ESIMD TopK and INT4 MoE work.
3. The model has many small launch-bound operations: routing, TopK, GDN gates, recurrence, norms and expert reductions.
4. Qwen and Ornith share the same basic text architecture, so one optimized engine can serve two important models.
5. MTP provides a path to transform one-row decode into useful multi-row verification work.

The GDN recurrent state is not small enough to keep across all layers in GRF/SLM. One layer’s state is 32×128×128 elements: about 1 MiB in FP16 or 2 MiB in FP32. Thirty layers require about 30/60 MiB per sequence. The correct design is to tile one head/state block through registers and SLM while updating it, then persist the state in VRAM. Do not claim whole-model on-chip state residency.

## 5. Proposed engine

Working description: B70 model-specialized persistent inference runtime.

### 5.1 Control plane

Initially reuse:

- llama.cpp model metadata/GGUF loading where practical.
- Existing tokenizer, chat template and sampler tests.
- A weight-conversion tool that repacks GGUF or Hugging Face weights into BMG-native tensors once.

Replace the decode executor with:

- Device-only pointer table for all layer tensors.
- Device-resident hidden buffers, GDN state, convolution state and KV cache.
- Fixed-shape execution plans generated from the selected model configuration.
- Direct Level Zero regular command lists first.
- Optional native command-graph extension if supported and faster on the installed runtime.
- No PyTorch dispatcher, Python operator calls or host round trips in the token loop.

### 5.2 Weight format

Use a B70-specific offline-packed INT4 format rather than executing GGUF Q4_K blocks directly in the final engine:

- Symmetric signed INT4.
- Group size 128 as the first implementation, matching Intel’s current INT4 ESIMD kernels and minimizing initial integration risk.
- Evaluate a W4A8 group-32 native format as a second branch: it increases scale metadata but can improve activation quantization quality and maps cleanly to K32 DPAS tiles.
- FP16 scales initially.
- Separate decode layout optimized for 64-byte contiguous block loads.
- Separate prefill layout in the VNNI/DPAS-friendly order.
- Duplicate only performance-critical tensors when the extra VRAM is justified.
- Keep norms, gates and state-sensitive parameters in FP16/BF16 or FP32 as quality testing requires.
- FP32 accumulation for deep MoE and recurrent paths. Intel recently fixed coherent-output failures caused by FP16 DPAS accumulation overflow.

For M=1, unpack INT4 directly into ESIMD vectors and multiply FP16 activations using Vector Engines. For M≥8, test per-token INT8 activation quantization and native INT4×INT8 DPAS, but retain an FP16-activation fallback until KL/perplexity gates pass.

### 5.3 Model-specialized kernel islands

#### Island A: residual + RMSNorm + projections

Fuse:

- Residual add.
- RMS reduction.
- Scale.
- Two or more projections that consume the same normalized hidden vector.

Intel’s source already has `resadd_norm_gemv_fused`, `resadd_norm_gemv2_fused`, INT4 equivalents and multi-GEMV fusion. It states that one avoided BMG launch costs roughly 20–50 microseconds. Qwen’s fixed 2,048-wide hidden vector is ideal for a fully templated kernel.

#### Island B: Gated DeltaNet decode

Fuse as far as correctness and register pressure allow:

1. Q/K/V/Z/A/B projections, sharing the normalized hidden load.
2. Four-token causal-convolution state shift/update.
3. Q/K L2 normalization.
4. Decay and beta gates.
5. Tiled recurrent-state update in FP32.
6. Gated RMSNorm.
7. Output projection.

Do not place all stages in one work-group if doing so reduces occupancy or spills GRFs. Use two or three persistent phases with device-side barriers/counters. Intel already provides `gdn_conv_fused`, sequential-layout variants, norm+GEMV fusion and the recurrent attention components; these are the starting point.

#### Island C: MoE decode

One logical device-side pipeline:

1. Router projection.
2. Softmax + top-8 + renormalization.
3. Expert/task queue creation.
4. Selected expert gate/up projections.
5. SiLU or model-specific activation.
6. Down projection.
7. Routing-weighted reduction.
8. Shared expert.
9. Residual + next-layer normalization.

Important implementation details:

- TopK over 256 experts should be vectorized in one small ESIMD work-group.
- For batch 1, use wide 1D loads and decode GEMV, not the narrow DPAS path.
- For multiple sequences or MTP candidates, group rows by expert and reuse each loaded weight tile with DPAS.
- Use producer/consumer counters so down-projection tiles can begin when an up-projection tile is ready.
- Avoid materializing full routed intermediate tensors when tiled producer/consumer execution is faster.
- Fold routing weights into the down projection or final reduction.

#### Island D: full-attention decode/prefill

Decode:

- Fused Q/K/V projection, query gate and RoPE.
- Quantized-KV paged attention.
- Fused dequantization in the K/V loop.
- Output projection.

Prefill:

- DPAS/XMX Flash Attention with online softmax.
- Query/KV tiling selected for head dimension 256.
- Quantized KV write during attention rather than a separate conversion kernel.

Intel’s current code contains a BMG-specific warning that FP16 `DPAS RepeatCount=4` can produce NaNs while RepeatCount=8 works. Treat every DPAS shape as compiler/driver-sensitive and validate it on the exact installed stack.

#### Island E: LM head + sampling

The 248,320-token vocabulary makes this important:

- Quantized LM-head GEMV.
- Do not store or transfer the complete logits vector if only sampling is needed.
- Fuse temperature, repetition/logit transforms that can be expressed on device, partial top-k reductions and final sampling.
- Return only the chosen token ID and small diagnostic metadata to the host.

#### Island F: MTP/speculative path

MTP is the route to effective speed beyond ordinary one-token weight bandwidth:

1. Produce 2–4 draft candidates from the trained MTP head.
2. Verify candidates as a small M dimension.
3. Use DPAS/XMX when the verified batch shape is large enough.
4. Reuse selected expert weight tiles across candidates routed to the same expert.
5. Commit accepted KV/GDN/conv states transactionally; discard rejected branches without corrupting state.

Report both raw target passes/s and accepted tokens/s. A high apparent speed with poor acceptance is not a win.

## 6. Persistent mega-kernel feasibility

A whole-token persistent launch is theoretically possible through Level Zero cooperative execution, but it is not the recommended production architecture. SYCL root-group synchronization remains experimental, ordinary SYCL/ESIMD work-group barriers are not grid barriers, and an atomic software grid barrier can deadlock unless every work-group is resident.

Level Zero exposes:

- Cooperative kernel launches.
- A query for maximum cooperative work-group count.
- Cooperative-capable queue-group flags.
- Global execution barriers through cooperative execution/runtime support.

The Intel SYCL root-group extension maps conceptually to Level Zero cooperative launches, but its own specification warns that it is experimental and unsuitable as a shipping dependency. IGC contains a global-barrier implementation using a device synchronization buffer, and Xe2-HPG supports hardware named barriers within their supported scope. None of this proves that a whole-model cooperative launch will outperform a well-captured graph on the installed B70 stack.

Candidate topology:

- A cooperative grid sized from `zeKernelSuggestMaxCooperativeGroupCount`.
- Persistent ESIMD workers that consume a device task descriptor.
- One work-group-local SLM arena per resident group.
- Fixed model bytecode or a statically unrolled 40-layer loop.
- Phase boundaries implemented with root/global barriers only where unavoidable.
- Fine-grained producer/consumer counters for MoE and tiled reductions.
- Device-side token loop for N tokens, stopping on EOS, cancellation or host-updated command state.

Why not begin there:

1. BMG currently has documented DPAS shape-specific compiler/runtime faults.
2. Cooperative occupancy may be lower because all work-groups must remain resident.
3. Whole-kernel GRF and instruction-cache pressure can cause spills or compilation failure.
4. Global barriers may cost more than a pre-recorded Level Zero kernel boundary.
5. A hang in one phase can wedge the GPU and complicate recovery.
6. Debugging numerical divergence across 40 layers becomes much harder.

The correct architecture therefore supports both schedulers:

- `graph`: multiple fused ESIMD kernels, recorded/submitted as a Level Zero command list.
- `persistent`: one cooperative token-loop kernel using the same verified microkernels or inlined variants.

Promote persistent execution only if it beats the graph path after launch overhead is already reduced.

## 7. Why one literal whole-model kernel is usually inferior

A transformer-wide monolithic kernel cannot keep the model’s weights on chip; tens of gigabytes still come from GDDR6. It also faces:

- Inter-layer dependencies requiring grid-wide synchronization.
- Register-file spills from trying to inline every algorithm.
- Large instruction footprint and I-cache pressure.
- Dynamic MoE routing.
- Variable context length and paged KV traversal.
- Mutable recurrent, convolution and KV state.
- Sampling-dependent control flow.
- Reduced occupancy under cooperative launch constraints.
- Poor fault isolation and maintainability.

The professional interpretation of “mega-kernel” is a persistent scheduler plus model-specific fused islands, not one enormous unspecialized function.

## 8. Quantitative expectations

### 8.1 Memory roofline

For any path reading `W` gigabytes of unique weight/state data per accepted token:

`ideal tokens/s = 608 / W`

At 75% measured bandwidth:

`practical memory ceiling ≈ 456 / W`

Examples:

| Unique bytes read per token | Ideal | At 75% BW |
|---:|---:|---:|
| 3 GB | 202.7 tok/s | 152.0 tok/s |
| 4 GB | 152.0 | 114.0 |
| 6 GB | 101.3 | 76.0 |
| 10 GB | 60.8 | 45.6 |
| 16 GB | 38.0 | 28.5 |
| 20 GB | 30.4 | 22.8 |

Measure actual bytes using GPU memory counters; do not infer them from file size for MoE models.

### 8.2 Honest target bands

These are engineering targets, not measured promises:

- Qwen/Ornith 35B-A3B, batch-1 custom INT4: first target 1.25× over the best correct local baseline; stretch target 1.5–2.0×.
- MTP effective generation: another 1.2–1.8× only if acceptance and state management are strong.
- Dense Qwen-27B: likely 10–30% kernel/runtime gain; its full-weight memory traffic makes 2× unlikely without lower bits or accepted-token speculation.
- Prefill/concurrent serving: larger gains are plausible because XMX/DPAS can reuse weights across rows.

For the current historical 55–68 tok/s sparse-model range, the evidence supports **65–82 tok/s as the near-term correct single-stream target**. A native packed/fused engine can keep **80–110 tok/s as a stretch band**, but it must first prove that current expert traffic and launch overhead leave that headroom. Existing local MTP on 60+ tok/s MoE was neutral or negative, so do not budget an MTP multiplier for Qwen35/Ornith. Reserve MTP primarily for the bandwidth-bound dense Qwen-27B path unless a new fused MoE verification kernel produces a measured win.

## 9. Existing-engine ranking

| Objective | Best choice |
|---|---|
| Stable daily single-user GGUF inference now | Preserved llama.cpp SYCL b9853 |
| Official one-B70 alternate runtime | OpenVINO/OVMS 2026.2 |
| Kernel source and high-concurrency research base | Intel `llm-scaler` ESIMD + vLLM/SGLang |
| General correctness control | llama.cpp Vulkan/CPU |
| Maximum model-specific ceiling | Thin Level Zero + BMG-AOT ESIMD custom engine |

OpenVINO is valuable for comparison and production features, but its generic graph/runtime layers are not the absolute single-model ceiling. vLLM/SGLang provide continuous batching and useful graph scheduling, but Python/PyTorch/custom-op boundaries are unnecessary for a fixed local model. llama.cpp is the best loader/control implementation but its generic tensor graph and many GGUF quant layouts are not ideal for a single B70-native AOT path.

## 10. Implementation plan

### Phase 0 — recover and freeze

- Positively identify the host/interface.
- Restore reliable SSH and remote recovery.
- Freeze b9853 binary, source SHA, command lines, model hashes and raw results.
- Record driver, kernel, Level Zero, IGC, oneAPI, firmware and power settings.
- Preserve the daily driver untouched.

### Phase 1 — profile the real bottleneck

For Qwen and Ornith:

- Capture per-op GPU time, launch count and gaps.
- Measure GDDR bandwidth, cache hit rate, XMX/Vector Engine utilization, occupancy, GRF spills and SLM usage.
- Account separately for GDN, MoE, full attention, LM head and sampling.
- Calculate actual bytes/token.

Kill criterion: do not design a fused path from elapsed wall time alone.

### Phase 2 — standalone kernel lab

Create independent correctness/performance tests for fixed Qwen35 shapes:

- INT4 GEMV, K=2,048 and relevant projection widths.
- Router + TopK(256,8).
- Expert up/gate and down GEMV.
- GDN recurrent update for 32×128×128 state.
- Head-dim-256 decode and prefill attention.
- Fused residual/RMSNorm/multi-GEMV.

Port/adapt Apache-2.0 kernels from `llm-scaler`, retaining attribution and license headers.

Compile BMG-specific modules AOT with the proven pattern below, but query the exact target accepted by the installed `icpx`/`ocloc`; do not hard-code a web-post alias into the reproducible build:

```text
-O3 -ffast-math -fsycl-device-code-split=per_kernel \
-fsycl-targets=spir64_gen -Xs "-device bmg"
```

Use large/double GRF only for kernels proven to benefit; record spills and occupancy.

Gates:

- Maximum absolute/relative error against FP32 CPU reference.
- Per-layer cosine/KL comparison.
- No NaN/Inf over long deterministic runs.
- No multi-turn state corruption.
- Target at least 500 GB/s for large M=1 weight-streaming kernels before integration.

### Phase 3 — llama.cpp plugin / thin graph path

- Add a model-specialized executor behind an explicit flag.
- Reuse loader/tokenizer/sampling initially.
- Convert weights offline into the BMG-native pack.
- Run fixed fused islands through one in-order queue and a captured SYCL Graph or reusable Level Zero command list. A/B immediate-command-list mode on the exact runtime; do not assume it is faster for sub-10-µs kernels or stable on this host.
- Eliminate host synchronization inside the token loop.

Promotion gate: ≥1.20× correct end-to-end decode improvement and no feature regression required by the target workload.

### Phase 4 — fused Qwen block engine

Implement fixed 40-layer execution:

- 30 GDN blocks.
- 10 attention blocks.
- All 40 MoE blocks.
- Device LM head and sampler.
- Quantized KV path.

Compare graph mode with llama.cpp operation-by-operation and end-to-end.

### Phase 5 — operation-level persistence; optional whole-model cooperative spike

- First implement persistent expert and long-context attention schedulers that never require all logical work-groups to be resident.
- Only then query cooperative support and maximum resident group count for an optional whole-model research spike.
- Implement a two-layer synthetic cooperative loop first.
- Measure global-barrier cost and watchdog/recovery behavior.
- Extend to one GDN+MoE block, then four-layer pattern, then 40 layers.

Kill or pause persistent mode if:

- Cooperative launch reduces memory throughput by >5%.
- Global synchronization consumes >5% of token latency.
- GRF/instruction growth causes spills or occupancy loss.
- It fails to beat the graph-mode fused engine by at least 5–10%.

The graph engine remains a successful deliverable even if persistent mode fails.

### Phase 6 — MTP

- Implement transactional candidate state.
- Test n=2,3,4 separately.
- Batch candidate verification into DPAS-friendly rows.
- Report target passes, acceptance, effective tok/s and quality.

Promotion gate: ≥15% effective end-to-end gain at equal deterministic quality and stable long conversations.

## 11. Verification matrix

Every candidate must pass:

1. CPU reference on kernel microtests.
2. Baseline llama.cpp logits comparison at selected layers.
3. Perplexity and mean KL divergence.
4. 20+ deterministic multi-turn exchanges.
5. Long-context retrieval at 32K and 128K.
6. Tool calls, JSON-schema adherence and coding patches/tests.
7. Vision tests where required.
8. Repeated runs with fixed seed.
9. Batch 1/4/8/16/32 and fresh/shared prefixes.
10. Power at 150/165/180 W, plus thermals and clocks. Use 230 W only as a short one-off maximum-performance characterization, never as the sustained recommendation.
11. GPU reset/recovery test after injected kernel failure.

Log source SHA, compiler/driver versions, exact build flags, exact commands, model/weight hashes, raw output and rollback procedure.

## 12. Immediate next action after host recovery

Do not begin with a new whole-model repository. Use two explicitly different tracks:

### Track A — fastest evidence-backed win

1. In an isolated pinned llama.cpp SYCL fork, reproduce the oneDNN Graph FP16 SDPA path and the quantized-KV bridge represented by PRs #25222/#25312 on Qwen-27B.
2. Start with q8_0/q8_0 at 32K/80K/128K and strict multi-turn/scratch-lifetime gates.
3. Require at least 1.5× attention-path speedup, 15% end-to-end prefill gain, <2% decode regression and <5% VRAM growth.

This is the lowest-risk proof because upstream reports approximately 2.48× at 32K for quantized KV and up to 4.26× at 80K for FP16 KV, although those results are not yet reproduced locally.

### Track B — highest custom decode ceiling

1. Build Intel’s current `llm-scaler` INT4/FP8 ESIMD kernels against the installed BMG stack.
2. Extract the M=1 Qwen35 expert GEMV, fused residual/RMSNorm/GEMV, TopK and GDN kernels into standalone fixed-shape microbenchmarks.
3. Treat approximately 507 GB/s as the credible purpose-built target derived from demonstrated B580 low-bit efficiency; 574 GB/s is an optimistic source-level BMG dense-kernel reference.
4. If the expert path exceeds 500 GB/s and the model profile shows meaningful Amdahl headroom, integrate one complete Qwen35 MoE block behind a specialized llama.cpp path.
5. Only after that block beats baseline correctly should the full graph and any cooperative research scheduler be built.

Track A delivers the practical performance patch. Track B decides whether a professional B70-native decode engine is economically justified.

## 13. Primary sources

- Intel B70 specifications: https://www.intel.com/content/www/us/en/products/sku/245797/intel-arc-pro-b70-graphics/specifications.html
- Intel B70 datasheet: https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2026-03/datasheet-b70-gpu.pdf
- Intel Xe2 architecture: https://cdrdv2-public.intel.com/824434/2024_Intel_Tech%20Tour%20TW_Xe2%20and%20Lunar%20Lakes%20GPU.pdf
- Intel GPU architecture optimization guide: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html
- Intel ESIMD optimization guide: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/optimizing-explicit-simd-kernels.html
- Intel occupancy guide: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/thread-mapping-and-gpu-occupancy.html
- Level Zero programming guide: https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html
- SYCL root-group proposal: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_root_group.asciidoc
- Intel `llm-scaler`: https://github.com/intel/llm-scaler
- INT4 Qwen MoE ESIMD commit: https://github.com/intel/llm-scaler/commit/ea6a0d61bfb9df0a4f447ab9f63d1f01016aa7ac
- Intel/vLLM XPU kernel library: https://github.com/vllm-project/vllm-xpu-kernels
- Xe2 low-bit GEMV/GEMM paper and B580 measurements: https://arxiv.org/html/2508.06753v2
- IGC attention register-spill regression: https://github.com/intel/intel-graphics-compiler/issues/412
- llama.cpp oneDNN Graph SDPA PR: https://github.com/ggml-org/llama.cpp/pull/25222
- llama.cpp quantized-KV oneDNN bridge PR: https://github.com/ggml-org/llama.cpp/pull/25312
- SYCL Graph usage guide: https://github.com/intel/llvm/blob/sycl/sycl/doc/syclgraph/SYCLGraphUsageGuide.md
- Intel PTI `unitrace`: https://github.com/intel/pti-gpu/tree/master/tools/unitrace
- Qwen architecture/model card: https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/README.md
- Mirage mega-kernel project: https://github.com/mirage-project/mirage
- HazyResearch mega-kernels: https://github.com/HazyResearch/Megakernels
- AutoMegaKernel: https://github.com/RightNow-AI/AutoMegaKernel

## 14. Final recommendation

Proceed, but call the first milestone a B70-native fused-block engine, not a whole-model mega-kernel.

The optimal sequence is:

1. Preserve llama.cpp as the stable baseline.
2. Reproduce the Qwen-27B oneDNN SDPA/quantized-KV path as the fastest evidence-backed performance patch.
3. In parallel, transplant and benchmark Intel’s proven BMG ESIMD primitives on the exact Qwen35/Ornith shapes.
4. Build a graph-scheduled Qwen35/Ornith fixed-shape engine only after the expert/GDN microkernels prove sufficient Amdahl headroom.
5. Require ≥1.2× correct end-to-end improvement before expanding the model-specialized engine.
6. Add operation-level persistent MoE/attention schedulers where measured; keep whole-model cooperative execution as an optional research spike.
7. Use MTP primarily for dense Qwen-27B. Revisit MoE MTP only if a fused verification path reverses the current neutral/negative local result.

The project is credible precisely because Intel’s own current source already exposes the required B70-native mechanisms and documented bottlenecks. The opportunity is not inventing an unknown matrix primitive; it is assembling the existing BMG-specific primitives into a model-aware, host-free, launch-minimized execution path that general engines cannot justify maintaining.
