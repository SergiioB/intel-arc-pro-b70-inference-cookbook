# Intel XPU Inference Optimisation — Standalone Agent Prompt

Use this prompt with any coding agent or LLM when optimising a local LLM server on an Intel Arc Pro B60/B70 (Battlemage Xe2, `bmg_g31`) or Intel XPU host.

---

You are optimising local LLM inference on Intel Arc / Battlemage Xe2 GPUs (`vLLM XPU`, `llama.cpp SYCL`, `TensorFold XPU`, or `exl3xpu`). Follow this contract strictly:

1. **Read closed arms first:** Inspect `docs/DO-NOT-REPEAT.md` and `docs/WHAT-WORKED.md`. Never repeat a closed arm unless its **Revisit when** condition holds.
2. **Preflight before touching GPUs:** Run `python3 skills/intel-xpu-inference-optimisation/scripts/b70_preflight.py`. Verify free VRAM (`vram_mm`), as-found power caps (do not alter caps), package temperature (`<= 55 C` at admission), `MemAvailable`, disk reserves, and zero competing GPU workloads.
3. **Compute the Xe2 roofline first:** Run `python3 skills/intel-xpu-inference-optimisation/scripts/b70_ceiling.py`. Use measured sustained B70 bandwidth (`~521–528 GB/s`, not `608 GB/s`) and `~50%` of `157 TFLOPS` BF16 XMX. Remember Xe2 has **no native FP4/FP8 DPAS** — NVFP4/MXFP8/INT4/EXL3 unpack to BF16/FP16 in registers. Only run jobs > 1 hour or write custom kernels when the removable wall-time share can beat the adopted route by `>= 10%`.
4. **Apply Intel XPU levers in ROI order:**
   - Preserve native NVFP4/MXFP8 checkpoints before converting to GPTQ/GGUF.
   - Eliminate hidden FP32 weight casts (`lm_head`, e.g. vLLM PR `#69186`).
   - Enable decode graph replay (`FULL_DECODE_ONLY` on vLLM XPU; `GGML_SYCL_ENABLE_GRAPH=1` on llama.cpp SYCL) and persist JIT/Triton caches.
   - Tune `--max-num-batched-tokens` on Sliding Window Attention (SWA) models (`2048` for speed ladders, `1024`/`512` to shrink SWA reservation and unlock 2x–3x context; never `4096`/`8192`).
   - Evaluate speculation (`DSpark`, `MTP`, `DFlash`) with separate `step_ms`, acceptance length `a`, and greedy parity checks.
   - Build opt-in AOT `bmg_g31` CUTLASS Xe2 kernels (`sycl-tla` + `icpx -ze-opt-large-register-file`) gated by an env flag (default `0` = exact stock) only after config levers plateau, verified against FP32 reference (`rel_l2 <= 5e-5`).
5. **Measure honestly:**
   - One variable per arm with same-session controls.
   - Headline public speeds use realistic model-card sampling (`n >= 3` seeds, full reachable context ladder); keep greedy (`temp=0`) strictly as an exactness diagnostic.
   - Count both `delta.reasoning` and `delta.content` on vLLM >= 0.34; verify `vllm:prefix_cache_hits_total` delta `== 0.0` on cold prefill cells with unique entropy-first prefixes.
   - Record hardware energy (`energy1_input` J/tok), live min VRAM, clean `dmesg`, and baseline VRAM restoration.
   - Never force-kill a server mid-prefill; drain and stop cleanly.
6. **Qualify before publishing:** Pass deterministic quality + long-context retrieval tests, sanitize all private host paths/PCI IDs, record separate `benchmark` vs `capability` entries in `data/benchmarks.v1.json`, validate with `python3 scripts/check-repo.py`, and merge with `--merge` so commit-pinned evidence stays intact.
