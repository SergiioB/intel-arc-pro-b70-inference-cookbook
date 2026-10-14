# Arc Pro B70 multi-GPU research — 2026-07-21

## Purpose
Evaluate whether a second B70 should be used as independent capacity or as a logical multi-GPU device, and identify safe candidate optimizations for the existing B70.

## Evidence-first conclusions

### 1. Two B70s are not physically linked
- No NVLink/XeLink cable is used.
- They communicate via PCIe and, depending on engine/runtime, can use direct P2P DMA or host-staged communication.
- VRAM is *not* automatically pooled. Software decides whether to run separate servers, layer-split a single model, or tensor-parallel a model.

### 2. llama.cpp SYCL layer split: capacity, not speed
Source: PMZFX [`multi-gpu.md`](https://raw.githubusercontent.com/PMZFX/intel-arc-pro-b70-benchmarks/master/multi-gpu.md).
- Layer split runs consecutive layer ranges sequentially: GPU 0 computes then hands an activation to GPU 1.
- A model that fits on one B70 should remain on one B70: tested Qwen 27B lost about 4% decode when split.
- It permits larger models: two B70s ran 70B Q4 dense at ~11.5 tok/s and Qwen3-Coder-Next 80B-A3B Q4 at ~43.4 tok/s on a well-wired CPU-direct PCIe 5 x8/x8 system.
- `--split-mode row` was reported to segfault on SYCL; use only layer split if testing.

### 3. vLLM tensor parallel: real parallel compute, but experimental
- Tensor parallelism shards each layer across cards so it can improve aggregate throughput/concurrency and may improve selected dense-model latency.
- It depends heavily on PCIe topology, kernel/xe/NEO/oneCCL versions, and model quantization.
- Public reports include BCS resets, worker crashes and XPU multi-device allocation faults. Do not use it as production on the existing B450 platform without an isolated validation transaction.

### 4. Existing host topology is a poor TP/layer-split platform
Live audit 2026-07-21:
- ASUS TUF B450-PLUS GAMING, Ryzen 7 5700X3D, 32 GiB RAM.
- Current B70 occupies PCIEX16_1. The only second long slot is chipset-connected PCIEX16_2, not CPU x8/x8 bifurcation.
- Therefore a second B70 is useful as an **independent server**, but not an advisable immediate build for single-model multi-GPU performance.
- PSU/case/physical clearance remains unverified.

### 5. Current one-B70 optimization candidates
Live production audit 2026-07-21: llama.cpp source HEAD is `0dc74e332`; the service already pins `ONEAPI_DEVICE_SELECTOR=level_zero:0`, `SYCL_CACHE_PERSISTENT=0`, and regular Level Zero command lists. These are established stability settings, so do not claim an easy global 20–40% gain from generic Reddit flags.

1. **Keep production SYCL for interactive decode.** Current measured production result is already competitive: Qwen3.6-35B decode ~60–70 tok/s depending on model/config. Do not switch to Vulkan based on generic claims.
2. **The 165 W profile is the efficiency optimum for dense Qwen27 MTP.** Existing local sweep measured ~25.6 tok/s at 165 W vs 30.8 at 230 W, with materially better tok/J.
3. **Use one model/server per GPU, not two servers on one GPU.** A second process on an already occupied B70 can collapse performance.
4. **For long context, asymmetric KV (q5_0 K / q4_1 V) is a candidate only after exact live validation.** It can reduce KV VRAM vs q8/q8, but quality/context claims must be tested against the exact model and production server.
5. **`GGML_SYCL_DISABLE_OPT=1` is a MoE-only test candidate, not a global setting.** It can avoid a slot-init failure on some B70 MoE builds but can regress dense models; test it only in an isolated Qwen35/Ornith transaction.
6. **vLLM is for batched/concurrent throughput, not automatically faster single-user decode.** It should be evaluated only inside memory-limited, reversible transactions, never Docker builds on this 30 GiB RAM host.

## Do not act on without validation
- Forced GPU clocks, development Mesa replacement, unknown `GGML_SYCL_*` variables, or a switch to Vulkan.
- Multi-GPU vLLM/TP on the B450 platform.
- Claims from Reddit/SEO sources that contradict local measurements.

## Delegation findings (2026-07-21, 3 subagents)

### The "200 tok/s" Reddit thread
- OP's reference host: **4× B70 on Threadripper Pro + WRX80**, not a consumer platform.
- The "200" in the title refers to **~200 tok/s aggregate at moderate concurrency**, not per-user.
- Practical high-concurrency claim: **965 mean aggregate tok/s at 64 concurrent users**.
- **No independent external replication** of the OP's exact TP4 performance was found.
- Host RAM behavior and PCIe topology are major deployment constraints; do not generalize from WRX80 to B450.

### SYCL vs Vulkan
- Direct Reddit thread recovery was blocked (403/429); cross-checked against public repos.
- Recommended baseline remains **SYCL / Level Zero** for single-stream decode.
- `SYCL_CACHE_PERSISTENT=0` is only needed if the documented load-time crash is hit.
- **Mesa 26.1+ Vulkan with NV_coopmat2 can match or beat SYCL for decode**, but this is not a safe production switch without exact-model validation.
- Multi-process co-tenancy (two servers on one GPU) has backend-specific stability issues.

### Last-14-days Reddit scan
- **Qwen3.6 KV cache quantization**: validate quality with KLD or task eval before dropping below Q8/Q6. Do not assume lower-bit KV is free.
- **vLLM XPU**: upstream merged a pipeline-parallel accuracy fix on Jul 7. Active churn — pin and regression-test a specific revision; do not treat XPU support as turnkey.
- No credible evidence of a "magic flag" that gives +30% on existing SYCL builds without validation.

## Sources
- https://raw.githubusercontent.com/PMZFX/intel-arc-pro-b70-benchmarks/master/multi-gpu.md
- https://raw.githubusercontent.com/PMZFX/intel-arc-pro-b70-benchmarks/master/llm-benchmarks.md
- https://github.com/vllm-project/vllm/issues/41663
- https://www.pugetsystems.com/labs/articles/intel-arc-pro-b70-multi-gpu-ai-inference-performance/
- https://github.com/RagingNoper/qwen36-b70 (OP reproducibility repo)
- https://gist.github.com/urakozz/810b7c642b689f252665f9d7278b763f (commenter's proposed stack patch)
