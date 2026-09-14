# The Dense 27B FP8 XPU Kernel Gap

> **Status update 2026-08-09:** the FP8 *linear* gap below is still real, but
> dense vLLM is no longer blocked — dense 27B **GPTQ-INT4** runs on the pinned
> nightly via `XPUwNa16LinearKernel` (~73 t/s C1 synthetic decode, MTP4, fp8 KV
> required for 128K). Only FP8-dtype checkpoints hit the KeyError. See
> README → Dense section.

**Status:** Blocked. vLLM has **no FP8 linear kernel registered for XPU**.

This is the single remaining blocker preventing vLLM from serving dense models
on the Arc Pro B70. It's an upstream vLLM / Intel xpu_kernels gap, not something
patchable in-container.

## The exact error

```
File "/opt/vllm/vllm/v1/.../kernels/linear/__init__.py", line 355, in choose_scaled_mm_linear_kernel
    ...
KeyError: <PlatformEnum.XPU: 4>
```

`choose_scaled_mm_linear_kernel()` is a dispatcher that maps `(platform, fp8_dtype)`
→ kernel implementation. The XPU platform enum exists but has **no entry in the
kernel map** — so the moment vLLM tries to build an FP8 linear layer, it raises
`KeyError`. This is not "slow fallback" — it's a hard absence.

## What we tried (Run 16, Run 19)

- `ThinkingCap-Qwen3.6-27B-FP8-text` (30 GB checkpoint, FP8 quant config)
- `intel/vllm:0.21.0-xpu-int4moe`, `--dtype bfloat16 --enforce-eager`,
  `--max-model-len 4096 --gpu-memory-utilization 0.97`
- Result: `EngineCore failed to start` → `KeyError: PlatformEnum.XPU` in
  `choose_scaled_mm_linear_kernel` during `Fp8LinearMethod.create_weights`.

The older "block-FP8 dequant registry patch" path (Run 16) ran but at ~0.75 t/s
— it dequantizes FP8 → BF16 per-op rather than running a fused FP8 GEMM. Useless.

## Why it matters

Dense 27B is the natural single-user interactive workload (more reasoning
quality per token than MoE 3B-active). Without an FP8 XPU kernel:

- vLLM can't serve dense 27B quantized on the B70 at all.
- llama.cpp SYCL is the **only** working dense path (Q4_K_M @230W = 23 t/s).
- llama.cpp dense + MTP (the GGUF `nextn` layer) pushes ~24–30 t/s — the only
  path past the Q4 baseline.

## How to help (research directions)

1. **Register an XPU FP8 linear kernel** in `choose_scaled_mm_linear_kernel`.
   Intel's `vllm_xpu_kernels` package already has FP8 paths for other ops; the
   linear GEMM registration may be a small PR. Check
   `vllm_xpu_kernels/fused_moe_interface.py` (which already handles `is_fp8`)
   for the pattern.

2. **Test vLLM W4A16 dense** (GPTQ/AWQ non-MoE linear path) on XPU. The MoE
   int4 path works great (this repo's patch 1); the dense linear equivalent
   may work or may hit a similar gap. Untested — needs a run.

3. **OpenVINO Model Server (OVMS)** for dense. Our Jul 5 scout (Run 7–10)
   got ~26–40 t/s wall on chat with the int4-OV dense checkpoint; a clean
   `genai-bench`-style run may be the real vLLM alternative for dense.

4. **Push llama.cpp dense + MTP-4** further at 165W (the documented dense
   efficiency sweet spot, 0.155 t/s/W). The `nextn` layer in the GGUF gives
   ~1.5–1.7× over base; a dedicated sweep may reach ~30 t/s.

## References

- Run 16 (block-FP8 dequant attempt): `docs/CAMPAIGN-LOG.md` A10
- Run 19 (FP8 KeyError confirmation): `docs/CAMPAIGN-LOG.md` A16
- vLLM source: `vllm/v1/.../kernels/linear/__init__..py`
- Intel xpu_kernels: `vllm_xpu_kernels/` package

If you pick this up, open an issue — happy to collaborate / share the raw logs.
