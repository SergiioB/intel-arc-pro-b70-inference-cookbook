# Intel Arc Pro B70 inference roadmap — 2026-07-14

## Executive verdict

The next large gains are not from rebuilding the same July 3 configuration or blindly lowering weight precision. They are most likely from:

1. New llama.cpp SYCL code merged after b9853.
2. Experimental XMX/oneDNN long-context Flash Attention PRs.
3. Matching the quant format to the optimized SYCL MoE path: MXFP4_MOE and NVFP4_MOE.
4. Keeping dense-model MTP, while treating single-GPU MoE MTP as disabled until measured otherwise.
5. Separating single-request latency from aggregate throughput and tuning each independently.
6. Preserving quality with quantitative gates before promoting any profile.

The current q8_0 K / q4_1 V cache is a defensible daily-driver compromise: keys are more quantization-sensitive than values, so the asymmetric split protects attention quality while saving substantial VRAM. It is not assumed optimal until compared against q8/q8, q8/q5_0 and q5_0/q4_1 using perplexity, token-distribution KL, long-context retrieval and real agentic tasks.

## Current known baseline

- GPU: Intel Arc Pro B70, 32 GB GDDR6, approximately 608 GB/s.
- Daily backend: llama.cpp SYCL, F16 kernels.
- Installed build: b9853 / 7af4279f4, built around 2026-07-03.
- Weight baseline: Qwen3.6-27B dense Q5_K_M and Qwen3.6-35B-A3B MoE Q4/Q5 variants.
- KV baseline: q8_0 K / q4_1 V.
- Context target: 128K when practical.
- Power baseline: 165 W was previously the best efficiency point; 230 W produced only about 8% more throughput for +65 W.
- Historical performance: about 67 tok/s single stream on the fastest tested profiles; 145.86 tok/s aggregate at high concurrency.
- Dense Qwen MTP showed approximately +45–50% in prior local tests.
- Single-B70 MoE MTP was not a win because expert routing/kernel-launch overhead dominates.

## Important upstream changes since b9853

Current master is substantially ahead of the installed build. The specific merged changes requiring A/B validation are:

- PR #25042: SYCL environment variables renamed from DISABLE to ENABLE semantics. Old service variables may silently stop doing what we expect.
- PR #25063: DMMV utilization change, K_QUANTS_PER_ITERATION=1. Potential token-generation gain.
- PR #25064: Q2_K reorder path. Relevant only to very low-bit experiments.
- PR #25217: fused top-k MoE. Real B70 validation showed roughly +2–3% MoE decode, neutral dense performance.
- PR #25205: configurable SYCL flash-attention vector thread count, merged 2026-07-14. Must sweep 64/128/256 if the option is exposed for the model/head shape.

The installed build already includes:

- PR #21920 fused SYCL MoE mul_mat_vec_q, including Q/K quants plus MXFP4/NVFP4 expert support. B70 reference result improved Qwen MoE decode from roughly 33.3 to 46.0 tok/s at 16K; the author reported about +24% on MXFP4/NVFP4 expert paths.
- PR #21845 multi-column MMVQ, important for dense MTP.
- PR #23398 Gemma4 MTP support.

## Highest-value experimental branches

Test each in a separate immutable build directory. Never stack them for the first comparison.

### PR #25025 — oneMKL Flash Attention

- Open.
- Expected value: long-context prefill acceleration through XMX/MKL.
- Risk: performance/VRAM/correctness depends on sequence shape and quantized KV path.
- Required tests: backend ops, perplexity, 2K/8K/32K/64K/128K depth, mixed KV, OOM limits.

### PR #25222 — oneDNN SDPA for F16 KV

- Open and blocked at the time of research.
- Reported on B70: 1.21x at short prompt and up to 4.26x at approximately 80K depth.
- Requires GGML_SYCL_DNN=ON.
- Best candidate if F16 KV fits, but F16 KV conflicts with the 128K/32GB memory budget for larger dense models.

### PR #25312 — hybrid oneDNN SDPA for quantized KV

- Open/dirty at the time of research and depends on #25222.
- Highest upside for the actual daily profile because it supports quantized KV.
- B70 reports: Qwen 27B, 32K prompt, q8_0 KV, 332 to 824 prefill tok/s (2.48x); Gemma 31B q8_0, 178 to 440 tok/s (2.48x), with decode maintained around baseline.
- Must not be promoted before FLASH_ATTN_EXT backend tests, perplexity/KL, long multi-turn conversations and OOM validation pass.

## NVFP4, MXFP4 and custom low-bit strategy

### NVFP4

NVFP4 is designed for NVIDIA Blackwell W4A4 tensor cores. A compressed-tensors NVFP4 checkpoint that requires Blackwell/vLLM is not portable to B70. However, NVFP4_MOE stored in GGUF is supported by the merged llama.cpp SYCL fused-MoE path. On B70 it is handled by SYCL kernels/dequantization, not NVIDIA's native Blackwell instruction path.

Therefore:

- Yes, test NVFP4_MOE GGUF on B70.
- Do not expect the advertised Blackwell throughput.
- The potential gain comes from lower expert-weight bandwidth and fused expert dispatch, not native NVFP4 hardware execution.
- Compare Qwen3.6-35B-A3B NVFP4_MOE-Q8 and NVFP4_MOE-Q4_K_M against the same model in Q4_K_XL/Q5_K_XL.

### MXFP4

MXFP4 is the better first FP4 experiment for B70:

- It is an OCP format rather than a Blackwell-only deployment format.
- llama.cpp already supports MXFP4_MOE GGUF on SYCL.
- gpt-oss-20B was post-trained with MXFP4 MoE weights, so its official/native MXFP4 is a better quality control than re-quantizing a BF16 checkpoint.
- Test gpt-oss-20B-MXFP4 first to validate the B70 implementation, then Qwen3.6-35B-A3B-MXFP4_MOE.

### AutoRound INT4 / vLLM XPU

Intel publishes Qwen3.6-35B-A3B INT4 AutoRound models and llm-scaler now has custom ESIMD INT4 MoE kernels. Intel reported INT4 versus FP8 MoE kernel gains ranging around 1.0–1.36x depending on batch, with larger gains at concurrency. This is a second inference stack, not a drop-in llama.cpp quant.

Use current intel/llm-scaler-vllm rather than old b8.1 images. Run it isolated from the daily service. Compare:

- FP16/BF16 reference where it fits.
- FP8 online quantization.
- sym_int4.
- GPTQ/AutoRound INT4 using a packing format the current image actually accepts.
- FP8 KV with calibrated scales, not blind scale=1.0.

Promotion requires tool calling, reasoning-parser, JSON schema and long-context accuracy to pass; historical reports show these can work for chat while failing thinking/tool use.

### TurboQuant KV

TurboQuant is promising for capacity, not the first speed optimization. Public llama.cpp work is not a stable merged SYCL path. Prior reports show about 5x KV compression and near-lossless quality in some architectures, but implementations and backend coverage are experimental. Do not patch production first. Revisit after the quantized oneDNN SDPA path settles because compressed KV without a fast attention kernel can reduce speed.

## Recovery phase — must happen first

1. Recover host access and immediately preserve previous/current boot evidence.
2. Determine whether the failure was suspend, failed resume, network-only, sshd, OOM, GPU reset, kernel panic, power loss or userspace deadlock.
3. Inspect the active ai-node suspend/wake automation. The 07:00 wake job is currently failing and the prior 00:30 suspend job succeeded; this is operationally more important than benchmark work.
4. Make wake monitoring wait for boot/SSH for at least five minutes and report distinct states: packet sent, ping, TCP 22, SSH command, inference endpoint.
5. Add a second recovery path. Preferred: smart plug configured for BIOS restore-on-AC-power; alternatives: router WoL, another always-on LAN host, or hardware remote-management device.
6. Disable or modify automatic suspend until resume reliability is proven for several cycles.
7. Do not benchmark until the machine survives controlled suspend/wake and reboot tests.

## Benchmark protocol

### A. Freeze the baseline

Capture:

- llama.cpp commit, binary hash and build flags.
- oneAPI compiler/runtime, Level Zero, IGC, kernel, firmware and xpu-smi versions.
- service unit, all environment variables and full command line.
- model SHA256, quant metadata and MTP/mmproj file hashes.
- GPU power limit, clocks, temperature, VRAM, ECC and ReBAR state.

Copy the working binary and service profile to an immutable rollback directory.

### B. Build isolation

Create separate builds:

1. baseline-b9853
2. master-current
3. pr25025
4. pr25222
5. pr25312
6. optional current master plus DFlash patches

Use identical compiler/runtime unless the test explicitly varies those. Never combine code and driver updates in the same A/B.

### C. Performance matrix

Models:

- Dense daily driver: Qwen3.6-27B Q5_K_M.
- MoE daily candidate: Qwen3.6-35B-A3B Q4_K_XL or Q5_K_XL.
- MXFP4 control: gpt-oss-20B-MXFP4.
- FP4 comparison: Qwen3.6-35B-A3B MXFP4_MOE and NVFP4_MOE.
- Optional Gemma4 dense model with native MTP to test architecture-dependent gains.

Depths:

- 0/2K, 8K, 32K, 64K, 128K.

Metrics:

- Prefill tok/s and time-to-first-token.
- Decode tok/s and inter-token latency, extracted from server logs rather than mixed wall time.
- Aggregate throughput at parallel 1, 4, 8, 16 and 32.
- Fresh-prefix and shared-prefix cases separately.
- VRAM peak, power, temperature, effective bandwidth, errors and process stability.
- Three to five interleaved repetitions; median plus spread.

### D. KV matrix

Compare on the same build/model/context:

1. f16/f16 reference where it fits.
2. q8_0/q8_0 quality reference.
3. q8_0/q5_0.
4. q8_0/q4_1 current daily profile.
5. q5_0/q4_1 capacity candidate.
6. q4_0/q4_0 only as a stress/capacity point.

The current q8 K / q4_1 V remains the provisional default unless another profile shows either:

- >=5% speed improvement at statistically stable quality, or
- materially higher context/concurrency with no meaningful quality regression.

### E. Speculative decoding

Dense:

- baseline, MTP n=2, n=3 and n=4.
- measure acceptance by workload class, not one synthetic prompt.
- current expectation: n=2 is likely the safest throughput point; n=4 may win on repetitive/code tasks but lose on creative/reasoning prompts.

MoE on one B70:

- default off.
- one verification run after new fused-MoE code.
- promote only if end-to-end wall time improves at 8K, 32K and 64K, not merely acceptance rate.

Gemma4 dense:

- test n=2 and n=4; upstream reports >2x on suitable hardware, but quantized KV has shown acceptance/performance problems and must be isolated.

## Quality gate

Every speed candidate must pass against a higher-precision reference:

1. Perplexity on a fixed corpus.
2. Mean and high-percentile KL divergence of next-token distributions on a fixed prompt corpus.
3. Long-context needle retrieval at 8K/32K/64K/128K.
4. Multi-turn cache consistency.
5. Deterministic coding tasks with executable tests.
6. Tool-call name/argument validity.
7. JSON-schema validity.
8. Multilingual English/Spanish tasks.
9. Vision tasks when mmproj is enabled.
10. Repetition, truncation, NaNs and malformed output detection.

Reject or quarantine a candidate if it fails backend-op tests, produces NaNs/corruption, regresses tool/JSON success materially, or increases KL/perplexity beyond the agreed threshold.

## Promotion criteria

Promote a change to the daily service only if:

- It survives a soak test and repeated service restarts.
- No correctness/backend-op failures occur.
- Quality gates pass.
- Single-stream gain is >=5%, long-context prefill gain is >=15%, aggregate throughput gain is >=10%, or it provides a large context/concurrency gain without quality loss.
- The exact build, flags and rollback command are documented.

## Ordered execution plan

P0 — reliability

1. Recover and diagnose B70 wake failure.
2. Fix suspend/wake and add out-of-band power recovery.
3. Freeze b9853 baseline.

P1 — low-risk software wins

4. Build current master separately.
5. A/B current master vs b9853 on dense and MoE at 0K/32K/128K.
6. Audit renamed SYCL environment variables.
7. Sweep flash-attention vector thread count.

P2 — large long-context wins

8. Test #25025, then #25222, then #25312 separately.
9. Keep the best correct prefill path and validate q8/q4_1 KV on it.

P3 — low-bit model formats

10. Run gpt-oss-20B native MXFP4.
11. Compare Qwen 35B Q4/Q5, MXFP4_MOE and NVFP4_MOE.
12. If MXFP4/NVFP4 wins, evaluate a custom quant with protected attention/router/shared tensors and FP4 routed experts.

P4 — throughput and second stack

13. Tune parallel 1/4/8/16/32 and batching.
14. Isolate current llm-scaler/vLLM and compare FP8 versus AutoRound INT4.
15. Keep llama.cpp as default unless vLLM wins end-to-end and passes agentic quality.

P5 — experimental research

16. Revisit TurboQuant KV after fast quantized attention support stabilizes.
17. Consider custom SYCL kernels only after profiling proves the remaining bottleneck: expert dispatch, attention, dequantization or memory bandwidth.

## Monitoring

The local B70 watchdog is active every six hours. It now tracks llama.cpp and llm-scaler commits plus PRs #25025, #25222 and #25312, and includes MXFP4/NVFP4/FP4/INT4/AutoRound/oneDNN/oneMKL/ESIMD keywords. Alerts are emitted when relevant changes or tracked PR updates appear.
