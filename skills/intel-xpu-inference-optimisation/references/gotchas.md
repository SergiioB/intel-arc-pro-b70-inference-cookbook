# Intel Arc Pro B60/B70 & XPU Gotchas (Hard-Won Traps)

Check this list before debugging a crash, OOM, slow boot, or suspicious speed number on Intel XPU.

## 1. vLLM XPU & Level Zero Runtime Traps

- **Linux 107-byte Unix socket limit (`VLLM_RPC_BASE_PATH`):** Running vLLM from a deep campaign directory makes ZeroMQ IPC socket paths exceed 107 bytes and crash before model load. Always point `VLLM_RPC_BASE_PATH` at a short `/tmp/...` directory created with `mktemp -d`.
- **`uv` hardlink cache corruption when patching venvs:** If a venv was created with `uv --link-mode hardlink`, patching a file in `site-packages/` mutates the shared global `uv` cache. Always copy-unlink-copy (`cp f f.orig && rm f && cp f.orig f`) before running `patch`.
- **Piecewise graph capture fails on Intel XPU:** Only `-cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY` is reliable. Piecewise capture or dynamic per-batch draft lengths fail with `UR_RESULT_ERROR_OUT_OF_RESOURCES`.
- **Prefix caching corruption on W4A16 / speculative vLLM XPU stacks:** Keep `--no-enable-prefix-caching` on vLLM XPU W4A16/speculative stacks until upstream cache-hit bugs (vLLM `#53912` class) are verified fixed with a multi-turn corruption check (corruption was observed even on zero-hit cold requests).
- **Streaming reasoning field name (`delta.reasoning` vs `delta.reasoning_content`):** On vLLM >= 0.34 with `--reasoning-parser`, thinking tokens stream in `delta.reasoning`. Observers that only watch `delta.content` or `delta.reasoning_content` miss TTFT and report broken decode rates.
- **SWA `--max-num-batched-tokens` cliff:** On SWA-heavy models (like Inkling 2 Mini's 40 SWA visits), setting `mbt=4096` or `8192` reserves 10–17 GiB for SWA in-flight blocks and collapses max context to ~7K. Use `mbt=2048` for speed ladders and `mbt=1024`/`512` for max-context profiles.
- **Scoped wrapper losing supplementary groups:** Running `systemd-run --scope` as root without preserving supplementary groups drops `render`/`video`/`docker` access. Use `runuser -u <user> -- ...` so group membership is preserved.

## 2. SYCL / CUTLASS Xe2 / Compiler Traps

- **Missing `cuda_runtime_api.h` when compiling `sycl-tla` headers:** SYCL-TLA headers fall back to CUDA includes unless `-DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DCUTLASS_VERSIONS_GENERATED` are all defined on the `icpx` command line.
- **E4M3 subnormal scales in NVFP4:** Naive float-exponent bit shifts mishandle E4M3 subnormals (`exp == 0`). Use branchless exact E4M3->FP16/FP32 conversion and test both normal and subnormal scales against an FP32 exact reference (`rel_l2 <= 5e-5`).
- **Missing `-ze-opt-large-register-file`:** Without `-Xs "-device bmg_g31 -options -ze-opt-large-register-file"`, Xe2 CUTLASS tiles spill registers to scratch memory and lose 2x–3x throughput.
- **DPC++ 2026.1 `SYCL_CACHE_PERSISTENT=1` (`DNR-037`):** Causes regressions/instability on llama.cpp SYCL; set `SYCL_CACHE_PERSISTENT=0`.
- **`GGML_SYCL_ENABLE_OPT=0` (`DNR-031`):** Disables critical SYCL optimizations; never ship with `=0`.
- **`torch.topk` wide-input bugs on PyTorch XPU:** Watch for segfaults or slow CPU fallbacks on wide vocabularies; check `lm_head` and sampler dispatch explicitly.

## 3. Hardware, VRAM, and Multi-GPU Traps

- **Never identify a B70 by a remembered `hwmon` number:** `hwmonN` indices change across reboots. Always resolve `hwmon` and `vram_mm` through PCI BDF (`/sys/bus/pci/devices/0000:XX:00.0/...`) and pin `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1` + `ZE_AFFINITY_MASK`.
- **Forced container/process kill mid-prefill causes `xe` CCS resets:** Always abort active requests, send `SIGTERM`, wait for drain, and check `dmesg` afterward.
- **PCIe AER link drops mimic kernel crashes:** Always inspect `dmesg` for PCIe AER replay timeouts or `xe` engine resets before blaming model code.
- **Direct peer copies do not imply peer atomics:** Stock vLLM TP2 / oneCCL assumes collective patterns that stall or fail on B70 PCIe; use independent single-card servers for multi-user load or peer-store TP2 without remote atomics.
- **Polling energy wakes idle GPUs:** Read `energy1_input` only at clean phase boundaries (start/end of prefill and decode), not in a tight background loop on an idle card.
- **Host disk exhaustion silently breaks Docker/JIT caches:** Keep Triton, SYCL, and Hugging Face caches on a spacious data disk and enforce a root-filesystem free reserve (`>= 512 MiB`).
