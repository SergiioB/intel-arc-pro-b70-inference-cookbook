# Intel Arc Pro B70 Optimization Playbook (DOs and DON'Ts)

Canonical engineering method distilled from measured B70 campaigns across vLLM XPU, TensorFold XPU, exl3xpu, and llama.cpp SYCL. Read this alongside [WHAT-WORKED.md](WHAT-WORKED.md) and [DO-NOT-REPEAT.md](DO-NOT-REPEAT.md) before starting a port, quant conversion, or kernel change.

**Standalone Agent Skill:** The full executable hill-climbing workflow, 5 Intel Xe2 references (`ceilings`, `levers`, `gotchas`, `methodology`, `publishing`), standalone `PROMPT.md`, and stdlib-only CLI helpers (`b70_preflight.py`, `b70_ceiling.py`) live in their own repository: [`SergiioB/intel-xpu-inference-optimisation-skill`](https://github.com/SergiioB/intel-xpu-inference-optimisation-skill).

## 1. DOs: How to Consistently Reach Peak B70 Speed

1. **Price the roofline before touching a kernel.**
   Compute active bytes/token against sustained B70 memory bandwidth (~521–528 GB/s, not the 608 GB/s marketing ceiling) for decode, and active FLOPs/token against BF16 XMX (~157 TFLOPS peak, ~50% realistic MFU) for prefill. Do not optimize a kernel whose removable wall-time share cannot clear a >=10% end-to-end gate.
2. **Gate by capacity and correctness first, then rank by speed.**
   Always run a fit-first probe at the real target context length with a real prompt + generation request (`min_vram >= 500 MiB` hard abort floor; `>= 1,024 MiB` staged C1; `>= 2,048 MiB` preferred C1 performance). A server load log is never capacity proof.
3. **Preserve native checkpoint quantization before converting.**
   If a checkpoint ships natively packed NVFP4/MXFP8 experts (like Inkling 2 Mini), prefer an engine/plugin that consumes the unmodified weights directly before spending GPU/disk time on GPTQ, AutoRound, or GGUF requantization. Remember: Xe2 Battlemage has no native FP4/FP8 DPAS hardware — packed 4-bit/8-bit weights are bandwidth-saving storage formats unpacked into BF16/FP16 XMX.
4. **Eliminate hidden FP32 weight casts and host round-trips first.**
   On vLLM XPU, the 2-line FP32 `lm_head` fast path ([PR #69186](https://github.com/vllm-project/vllm/pull/69186), `torch.mm(..., out_dtype=torch.float32)`) delivered **+39–42% decode speed**, **-30% J/token**, and freed **+1.22 GiB VRAM** (unlocking 200K context on one B70). Always audit per-step weight casts and CPU fallback graph splits before rewriting matmul kernels.
5. **Keep graph replay on for vLLM XPU (`FULL_DECODE_ONLY`).**
   Never benchmark vLLM XPU with `--enforce-eager` as a speed claim. On Inkling 2 Mini, use `-cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY` and persist `TRITON_CACHE_DIR` across restarts so Triton JIT does not stall first live turns.
6. **Measure across the entire reachable context window with model-card sampling.**
   Every public speed table must cover all standard depths (`0`, `8k`, `32k`, `64k`, `128k`, `256k`) up to the largest proven context, labelling unreachable depths explicitly (`does not fit (largest: N)`). Use model-card sampling and default thinking mode (`>= 3` seeds) for public rates; keep greedy strictly for exactness and parity checks.
7. **Isolate one variable per arm with a same-session control.**
   Never change engine, quant, patch, and speculation in one step. When patching a `uv` hardlinked venv file, break the hardlink first so the shared package cache is untouched.
8. **Verify streaming observer fields and cold-cache counters.**
   On vLLM >= 0.34 with `--reasoning-parser`, streaming reasoning deltas arrive in `delta.reasoning` (not only `delta.reasoning_content`). Count both for TTFT/decode timing, and verify zero `vllm:prefix_cache_hits_total` delta on cold prefill cells with unique entropy-first prefixes.
9. **Use short IPC socket directories on vLLM XPU.**
   Set `VLLM_RPC_BASE_PATH` to a short `/tmp/...` directory so ZeroMQ Unix socket paths never exceed Linux's 107-byte `sun_path` limit when running inside deep campaign directories.
10. **Match multi-card topology to workload.**
    For multiple independent users, run one server per B70 (2x aggregate, zero collective tax). For single-stream decode/prefill beyond one card's bandwidth/FLOP ceiling, use peer-store TP2 without remote atomics (see [DUAL-B70-TP2.md](DUAL-B70-TP2.md)).
11. **Tune `--max-num-batched-tokens` on Sliding Window Attention (SWA) models to unlock 2x–3x larger context.**
    In vLLM's `SlidingWindowManager`, SWA KV reservation scales with `max_num_batched_tokens` across every local-attention layer visit (40 visits in Inkling 2 Mini). Raising `mbt` to `4096`/`8192` collapses Inkling's single-B70 context from 200K down to ~7K, while lowering `mbt` from `2048` to `1024` unlocks **163K with DSpark** (vs 106K) and **256K no-spec** (vs 200K; `308.7K` KV pool at `mbt=512`). Always report `mbt=1024`/`512` capacity profiles separately from `mbt=2048` speed ladders.
12. **Prefer AOT `bmg_g31` CUTLASS Xe2 grouped GEMM over generic Triton for packed MoE prefill and speculative verification (`M=8`–`2048`).**
    Writing a native NVFP4 (`e2m1`+`e4m3`/16) CUTLASS Xe2 grouped GEMM (`sycl-tla` + `icpx -device bmg_g31 -ze-opt-large-register-file`) with a 1-WG SLM route-sort and fused SwiGLU — paired with `vllm-xpu-kernels` MXFP8 CUTLASS Xe2 — lifted Inkling 2 Mini + DSpark `k=7` by **+40% decode (`57.2 -> 80.4 tok/s` card sampling, `71.6 -> 95.1` greedy)** and **+24% cold prefill (`3,172 -> 3,920 tok/s`)** with `rel_l2 <= 5e-5` vs FP32 and **8/8 quality/retrieval scenarios PASS**. Always gate custom kernels behind an opt-in env flag (`INK2_CUTLASS_MOE=1`, default `0`) so the stock path remains reproducible.

## 2. DON'Ts: Closed Traps (See [DO-NOT-REPEAT.md](DO-NOT-REPEAT.md))

- **DON'T convert native NVFP4/MXFP8 MoE checkpoints to GPTQ INT4 or larger Q8/BF16 GGUFs** (`DNR-009`, `DNR-040`, `DNR-041`): double-quantization hurts accuracy without beating native storage density.
- **DON'T infer tensor storage formats from filenames or friendly labels** (`DNR-014`): always read the safetensors/GGUF tensor header census.
- **DON'T mix routes, engines, patches, or metric classes in one table:** keep client cold-input rate (`tokens / TTFT`) separate from engine-native prefill and C1 decode separate from Cn aggregate.
- **DON'T enable prefix caching blindly on vLLM XPU W4A16/speculative stacks:** keep `--no-enable-prefix-caching` until multi-turn corruption checks pass.
- **DON'T disable `GGML_SYCL_ENABLE_OPT` (`=0`) or enable `SYCL_CACHE_PERSISTENT=1` on DPC++ 2026.1 llama.cpp SYCL** (`DNR-031`, `DNR-037`).
- **DON'T treat an unverified third-party headline or single diagnostic pair as an adopted recipe.**

## 3. Repository Structure Rule (No Mixed Information)

Every model family owns exactly one clean slice with strict route separation:
- `docs/<family>/README.md` — **route chooser only** (one row per engine/route, clear status, explicit do-not-mix boundary).
- `docs/<family>/<FAMILY>-<ENGINE>.md` — **one self-contained recipe per engine** (exact pins, launch flags, full-window table, quality/retrieval gate, explicit limits).
- `patches/<family>-<engine>/` — ordered patch stack + `APPLY.md` scoped to one exact base commit.
- `benchmarks/<family>/` — executable launchers with no private paths.
- `results/<family>/<engine>-<YYYYMMDD>/` — sanitized, commit-pinned raw JSONL + `summary.json` + `README.md`.
- `data/benchmarks.v1.json` — single numeric authority (renders `docs/BENCHMARK-CATALOG.md` via `scripts/render-benchmark-catalog.py`).
