---
name: intel-xpu-inference-optimisation
description: >
  Hill-climb local LLM serving speed and context capacity on Intel Arc Pro B60/B70
  (Battlemage Xe2, bmg_g31) and Intel XPU GPUs across vLLM XPU, llama.cpp SYCL,
  TensorFold XPU, and exl3xpu: single-user and concurrent decode tok/s, cold prefill
  tok/s and TTFT, speculative decoding (DSpark, MTP, DFlash), KV cache and SWA
  max-context tuning, AOT CUTLASS Xe2 kernels, and energy (J/token) under live VRAM,
  thermal, and correctness gates. Computes Intel Xe2 rooflines first, checks closed
  arms in DO-NOT-REPEAT.md, runs noise-controlled single-variable A/B tests, and
  qualifies every winner with realistic model-card sampling and exact retrieval before
  publishing. Use when optimising, tuning, porting, benchmarking, or writing XPU/SYCL
  kernels for a model on Intel Arc GPUs.
---

# Intel XPU Inference Optimisation (Hill-Climbing on Arc Pro B60/B70 & Xe2)

You systematically hill-climb local LLM serving speed, maximum context, and energy efficiency on Intel Arc / Battlemage Xe2 GPUs (`bmg_g31` and related XPU devices), backed by roofline arithmetic and reproducible evidence.

## Core Loop

```
preflight & DNR check -> roofline ceiling -> baseline -> pick highest (gap x likelihood) lever
       ^                                                                          |
       +--- update ledger + PLAN.md <--- accept/reject (controls + quality) <-----+
```

Read only the reference file needed for the current step:
- `references/ceilings.md` — Measured Intel Arc Pro B70/Xe2 hardware ceilings, looped/MoE active-byte math, SWA `--max-num-batched-tokens` KV formula, and speculative decoding arithmetic.
- `references/levers.md` — Proven Intel XPU levers ordered by measured ROI across vLLM XPU, llama.cpp SYCL, TensorFold XPU, and exl3xpu.
- `references/gotchas.md` — Hard-won Intel XPU, Level Zero, SYCL, Triton, and vLLM traps that silently cost 30–90% performance or crash the GPU.
- `references/methodology.md` — Noise-controlled interleaved A/B, realistic model-card sampling vs greedy diagnostics, cold-cache verification, hardware energy (`energy1_input`), and live `vram_mm` safety guards.
- `references/publishing.md` — Pinning, opt-in kernel gating, sanitization, `data/benchmarks.v1.json` schema, 12-gate CI, and 2x anti-aliased result graphics.
- `scripts/b70_preflight.py` — Read-only host/GPU preflight (PCI devices, `vram_mm` free VRAM, power caps, package temp, `MemAvailable`, disk, active GPU processes).
- `scripts/b70_ceiling.py` — Roofline & SWA-KV ceiling calculator (`examples/inkling2-mini-b70.json` worked example).

---

## Step 0: Preflight & Closed-Arm Gate (Zero GPU Waste)

1. **Read closed arms first.** In this cookbook, read `docs/DO-NOT-REPEAT.md` and `docs/WHAT-WORKED.md`. Never re-run a closed arm unless its **Revisit when** condition is met.
2. **Run read-only preflight:**
   ```bash
   python3 skills/intel-xpu-inference-optimisation/scripts/b70_preflight.py
   ```
   Verify:
   - Target non-display card PCI ID, live free VRAM (`vram_mm`), as-found power cap (never alter caps to fake a benchmark), and package temperature (`<= 55 C` admission).
   - No competing inference server or container on the target GPU.
   - Safe host `MemAvailable` (`>= 20 GiB` remaining under any memory-capped job) and root/data disk headroom (`>= 512 MiB` root reserve).
3. **Inspect the actual checkpoint tensors.** Read `config.json` and safetensors/GGUF tensor headers (`DNR-014`). Never infer weight formats from repo names.
4. **Declare constraints upfront:** target card count (1x vs 2x), context goal, realistic model-card sampling (`temperature`, `top_p`, `reasoning_effort`), and whether lossy/approximate changes are allowed.

---

## Step 1: Price the Intel Xe2 Roofline First

Run `scripts/b70_ceiling.py` before writing code or launching long sweeps (`references/ceilings.md`):

- **Use measured sustained B70 constants, not marketing sheets:**
  - Sustained GDDR6 bandwidth: **~521–528 GB/s** (not 608 GB/s peak).
  - Practical BF16 XMX compute: **~50% of ~157 TFLOPS** (~75–85 TFLOPS effective) on large GEMMs.
  - **No native FP4 or FP8 DPAS on Xe2 Battlemage:** NVFP4 / MXFP8 / INT4 / EXL3 weights save memory bandwidth and unpack to BF16/FP16 in registers (`W4A16`/`W8A16`).
- **Decode ceiling (memory-bound):** `active_bytes_per_token / sustained_BW`. Include dense weights, routed experts (`top_k / n_experts`), loop passes (e.g. Inkling runs 24 stored layers across 48 visits), `lm_head`, and KV bytes at target depth.
- **Speculative ceiling:** divides target bytes by mean accepted length `a`, plus `k` draft steps (draft `lm_head` and verify batch size `M = k+1` often dominate).
- **Prefill ceiling (compute-bound):** `2 * active_params_per_token + attention_flops(context)` divided by practical XMX TFLOPS.
- **KV & SWA ceiling:** separate full-attention layers from Sliding Window Attention (SWA) layers. On vLLM, SWA in-flight block reservation scales with `--max-num-batched-tokens` across every local-attention visit.
- **Impact gate:** before any job > 1 hour or any custom kernel, check whether the best-case removable wall-time share beats the current adopted route by `>= 10%`. If not, stop and document the arithmetic.

---

## Step 2: Non-Negotiable Discipline

1. **One variable per arm, same-session control.** Never combine engine, quant, patch, and drafter in one step without isolating attribution.
2. **Capacity and correctness before speed.**
   - VRAM admission tiers: `< 500 MiB` hard abort; `500–1023 MiB` isolated observation only; `1024–2047 MiB` staged C1; `>= 2048 MiB` preferred C1 performance; `>= 3072 MiB` concurrency.
   - Server boot is not capacity proof; stage a real prompt + generation request and confirm clean `dmesg` and full VRAM restoration afterward.
3. **Separate realistic card sampling from greedy diagnostics.**
   - Headline speed tables use the model card's sampling (`temperature`, `top_p`, default `reasoning_effort`, `n >= 3` seeds) across the full reachable context window (`0`, `8K`, `32K`, `64K`, `128K`, `max`).
   - Greedy (`temperature=0`) is an exactness/parity diagnostic, never a substitute for card-sampling speed.
4. **Never fake cold prefill.** Use unique entropy-first prefixes per request and verify `vllm:prefix_cache_hits_total` delta is `0.0`. Report cold prefill (`prompt_tokens / TTFT`) separately from warm/cached turns and engine-native prefill.
5. **Never force-kill a server mid-prefill.** Drain/abort requests, send `SIGTERM`, wait for exit, and verify baseline VRAM and clean `dmesg` (`xe` CCS resets often follow forced kills mid-batch).
6. **Gate custom kernels behind an opt-in switch.** Default `0` must preserve the exact upstream/stock path until parity and quality gates pass.
7. **Log failures permanently.** Closed arms and root causes go into `docs/DO-NOT-REPEAT.md` (or `PLAN.md`) with a concrete **Revisit when** condition.

---

## Step 3: Proven Intel XPU Lever Order (Highest ROI First)

See `references/levers.md` for measured B70 deltas and citations:

1. **Preserve native checkpoint quantization.** If a model ships native NVFP4/MXFP8 (e.g. Inkling 2 Mini), run it directly in vLLM XPU before converting to GPTQ/INT4/GGUF (`DNR-009`, `DNR-041`).
2. **Eliminate hidden FP32 weight casts & host syncs.** Audit `lm_head` and norm/router ops first — vLLM PR `#69186` (`torch.mm(..., out_dtype=torch.float32)`) delivered **+39–42% decode, -30% J/tok, and +1.22 GiB KV** in 2 lines.
3. **Enable decode graph replay.** Always use `-cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY` on vLLM XPU (never `--enforce-eager` for speed claims) and `GGML_SYCL_ENABLE_GRAPH=1` on llama.cpp SYCL. Persist `TRITON_CACHE_DIR`.
4. **Tune `--max-num-batched-tokens` for SWA context capacity.** On SWA-heavy models in vLLM, `mbt=4096/8192` collapses max context, while `mbt=1024` or `512` can unlock **2x–3x larger context** (Inkling: `106K -> 163K` DSpark, `200K -> 256K / 308.7K KV` no-spec). Report capacity profiles separately from `mbt=2048` speed ladders.
5. **Speculative decoding (`DSpark`, `MTP`, `DFlash`).** Measure per-step verify cost and accepted length `a` separately (`tok/s = a / step_ms`), check greedy parity (`drafted == serial` vs altered hashes), and sweep draft length `k` per depth and concurrency.
6. **Engine-specific quick wins:**
   - **llama.cpp SYCL:** build with `GGML_SYCL_F16=ON` (`3.3x` prefill), keep `GGML_SYCL_ENABLE_OPT=1`, set `SYCL_CACHE_PERSISTENT=0` on DPC++ 2026.1, use `q8_0/q4_0` KV on >= 2026-10-29 builds (`1.8–2.1x` at 64K).
   - **Dual-B70:** default to two independent single-card servers for multi-user throughput; use peer-store TP2 without remote atomics (not stock oneCCL TP2) when optimizing single-stream latency.
7. **Custom AOT `bmg_g31` CUTLASS Xe2 kernels (after config levers plateau).**
   - Compile with `icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg_g31 -options -ze-opt-large-register-file"` against pinned `intel/sycl-tla` + `vllm-xpu-kernels` headers.
   - Fuse SLM route-sort + grouped GEMM + activation (e.g. softcapped SwiGLU) for `M=8`–`2048` verify/prefill batches; verify numerical parity vs exact FP32 reference (`rel_l2 <= 5e-5`) across normal and subnormal scales before end-to-end runs.

---

## Step 4: Qualification & Publication Gate

Before promoting any result (`references/publishing.md`):
1. **Deterministic quality & long-context retrieval suite:** math, structured JSON, state-trace code, multi-turn correction, and multi-depth needle retrieval (e.g. `1.4K`, `8.5K`, `33.8K`, `135K`).
2. **Finished-answer check:** if default high reasoning effort saturates `max_tokens`, verify non-empty final answers at lower effort (e.g. `effort=0.2`) or higher budget.
3. **Full hardware cleanliness:** `dmesg` clean of GPU faults/resets, VRAM restored to baseline, power caps unchanged.
4. **Sanitized, commit-pinned public evidence:** zero private host paths, PCI IDs, credentials, or multi-MB binaries in git. Separate `kind: "benchmark"` records from `kind: "capability"` records in `data/benchmarks.v1.json`, run `python3 scripts/check-repo.py` (all 12 gates PASS), and merge with `--merge` so pinned evidence commits remain in history.
