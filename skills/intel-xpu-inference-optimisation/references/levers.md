# Proven Intel Arc Pro B70 / Xe2 Levers (Ordered by Measured ROI)

Every lever below comes from controlled B70 measurements recorded in this cookbook. Use them as ranked hypotheses, never as unverified promises on a new model.

## Tier 1 — Highest Payoff, Zero or Minimal Code

### 1. Preserve Native Packed Weights (`vLLM XPU` plugins)
- **Measured (Inkling 2 Mini, 2026-11-22):** Serving unmodified native NVFP4 (L0–19) + MXFP8 (L20–23) via `mgaruccio/vllm-inkling2-xpu` reached **3,172 tok/s cold prefill** (36x faster than TensorFold GGUF's 87.6 tok/s at 8K) and **200K–256K context** on one 32 GB B70.
- **Avoid:** Double-quantizing native NVFP4/MXFP8 checkpoints to GPTQ INT4 (`DNR-009`, `DNR-041`), EXL3 (`DNR-035`), or oversized Q8/BF16 GGUFs (`DNR-040`).

### 2. Eliminate Hidden FP32 Weight Casts (`lm_head` / norms)
- **Measured (vLLM PR `#69186`, same-session A/B on Inkling 2 Mini):** Switching `lm_head` from full BF16->FP32 weight recast (~1.72 GiB temporary per step) to `torch.mm(..., out_dtype=torch.float32)` gained **+42.1% short decode (`39.71 -> 56.44 tok/s`), +38.7% 8K decode (`37.49 -> 51.99 tok/s`, 3/3 byte-identical), -30% J/token, and +1.22 GiB KV cache**.
- **Rule:** Always audit large-vocabulary `lm_head`, router, and norm casts before touching GEMM kernels.

### 3. Enable Decode Graph Replay (`FULL_DECODE_ONLY` / `GGML_SYCL_ENABLE_GRAPH=1`)
- **Measured:** Decode graphs cut launch overhead across 48–64+ layers by 1.3x–2.0x (`llama.cpp` Inkling `+34.1%` at 0K, `+26.6%` at 8K, 5/5 greedy identical; `vLLM XPU` `FULL_DECODE_ONLY` mandatory for all speed claims).
- **Important:** Never use piecewise CUDA/XPU graph capture on Intel XPU (causes `UR_RESULT_ERROR_OUT_OF_RESOURCES`); use `-cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY`. Persist `TRITON_CACHE_DIR` across restarts.

### 4. Tune `--max-num-batched-tokens` on Sliding Window Attention (SWA) Models
- **Measured (Inkling 2 Mini, 1x B70):**
  - `mbt=4096 / 8192`: SWA KV reservation explodes -> max context collapses to `~7K`.
  - `mbt=2048`: `106K` (DSpark `u=0.93`) / `200K` (no-spec `u=0.95`) — best prefill speed.
  - `mbt=1024, u=0.95`: unlocks **`163K` (`194.1K` KV, `160K` real prompt @ `46.2 tok/s`) with DSpark** and **`256K` (`273.8K` KV, `256,000` real prompt @ `23.1 tok/s`) no-spec**.
  - `mbt=512, u=0.95`: expands KV pool to **`308,696` tokens**.
- **Rule:** Report `mbt=1024`/`512` capacity profiles separately from `mbt=2048` speed ladders.

## Tier 2 — Engine-Specific & Speculative Levers

### 5. Speculative Decoding (`DSpark`, `MTP`, `DFlash`)
- **Measured (Inkling 2 Mini, `vLLM XPU + PR #69186 + DSpark k=7`):**
  - Stock Triton plugin: card-sampling decode `54.5 -> 57.2 tok/s` at 0K, `39.9 -> 50.1 tok/s` at 64K (`+25.5%`), greedy `56.4 -> 71.6 tok/s`, **8/8 quality scenarios PASS**.
  - With opt-in CUTLASS Xe2 verify: card-sampling decode jumps to **`80.4 / 72.7 / 63.6 / 64.0 / 55.3 tok/s`** (`0–104K`), `effort=0.2` **`99.2 tok/s`**, greedy **`95.1 / 91.7 tok/s`**.
- **Measured (Qwen3.8-27B):** TensorFold + DFlash2 reached `128 tok/s` lossless greedy (`drafted == serial`); EXL3 + MTP3 wins at `>= 16K` and reaches `245K` context (with 27% greedy divergence documented explicitly).
- **Rule:** Always check whether speculation alters greedy hashes; never claim `drafted == serial` without a direct same-prompt hash check.

### 6. llama.cpp SYCL Build & KV Policy
- **Build with `GGML_SYCL_F16=ON` and AOT `-device bmg_g31`:** Ripple-65B pp2048 jumped **218 -> 720 tok/s (3.3x)** with identical quality.
- **KV cache type on builds >= 2026-10-29 (`#31125` SYCL flash-attn):** `q8_0/q8_0` (`22.3 tok/s`) and `q4_0/q4_0` (`23.9 tok/s`) decode **1.8x–2.1x faster at 64K** than `q5_0/q4_1` (`10.5 tok/s`). Default to `q8_0/q8_0`.
- **Runtime flags:** Keep `GGML_SYCL_ENABLE_OPT=1` (`DNR-031`) and `SYCL_CACHE_PERSISTENT=0` on DPC++ 2026.1 (`DNR-037`).

### 7. Dual-B70 Topology Selection
- **Multiple users / aggregate throughput:** Run **two independent single-card servers** (2x throughput, zero collective overhead, independent fault domains).
- **Single-stream speed beyond 1-card roofline:** Avoid stock vLLM TP2 / remote-atomics collectives as an assumed win. Use peer-store TP2 without remote atomics (`TensorFold --tp 2` PR `#3586`: **1.25x–1.5x** on Qwen3.8-27B, `drafted == serial`; or patched `llama.cpp` `GGML_SYCL_TP_P2P=2`).

## Tier 3 — Custom AOT `bmg_g31` CUTLASS Xe2 Kernels (After Config Levers Plateau)

### 8. Native NVFP4 + MXFP8 CUTLASS Xe2 Grouped GEMM + SLM Route-Sort + Fused SwiGLU
- **Measured (Inkling 2 Mini, `INK2_CUTLASS_MOE=1`, 2026-11-22):**
  - Replaced Triton packed MoE (`M=8` DSpark verify and `M=2048` prefill chunks) with a custom AOT `bmg_g31` SYCL CUTLASS Xe2 NVFP4 (`e2m1`+`e4m3`/16, branchless scale decode, `rel_l2 <= 5.0e-5` vs FP32) kernel + `vllm-xpu-kernels` MXFP8 CUTLASS Xe2 + 1-WG SLM route-sort + fused softcapped SwiGLU.
  - **Isolated kernel speedup:** `M=2048` MoE dropped to `3.34 ms` (44 visits `~147 ms`); `M=8` DSpark verify dropped to `0.567 ms` (44 visits `~25 ms`).
  - **End-to-end gain over stock + `#69186` + DSpark:** **+40.5% card-sampling decode at 0K (`57.2 -> 80.4 tok/s`), +28% at 64K (`50.1 -> 64.0 tok/s`), +33%–36% greedy (`71.6/67.3 -> 95.1/91.7 tok/s`), +23.6% cold prefill (`3,172 -> 3,920 tok/s`)**, **8/8 quality/retrieval scenarios PASS**.
- **Build flags that matter on Battlemage:**
  ```bash
  icpx -fsycl -fsycl-targets=spir64_gen \
    -Xs "-device bmg_g31 -options -ze-opt-large-register-file" \
    -O3 -std=c++20 -fPIC -shared \
    -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DCUTLASS_VERSIONS_GENERATED
  ```
- **Safety rule:** Always gate behind an opt-in environment variable (`INK2_CUTLASS_MOE=1`, default `0` = exact stock plugin) and test numerical parity across both normal and subnormal E4M3 scales before end-to-end runs.
