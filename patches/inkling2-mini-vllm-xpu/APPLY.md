# Patch Stack: Inkling 2 Mini on vLLM XPU

- **Target vLLM base:** `vllm-0.34.0rc1.dev1133+g647a76866.xpu` (upstream commit `647a76866b4559a74e27d1c6b55fbffe85c07ce4`, before upstream [vLLM PR #69186](https://github.com/vllm-project/vllm/pull/69186) merged).
- **Plugin:** [`mgaruccio/vllm-inkling2-xpu`](https://github.com/mgaruccio/vllm-inkling2-xpu) pinned at `2e9d28b04ff23f5d9996a221ffaa0ae3a440d8f9` (installed `--no-deps` into the same environment).

## Ordered patch list

1. **Required (`vllm`):** `pr69186-logits-processor.patch` — enables `torch.mm(..., out_dtype=torch.float32)` on XPU for FP32 `lm_head` projection instead of recasting the full BF16 `lm_head` weight (~1.72 GiB) to FP32 on every decode step (+38.7% to +42.1% greedy decode, +1.22 GiB KV cache). Skip once your vLLM XPU build includes merged PR #69186.
2. **Optional opt-in kernel (`vllm-inkling2-xpu`, `INK2_CUTLASS_MOE=1`):**
   - `nvfp4_xe2_ext.cpp` — AOT `bmg_g31` SYCL CUTLASS Xe2 native NVFP4 (`e2m1`+`e4m3`/16) grouped GEMM + 1-WG SLM route-sort + fused softcapped SwiGLU (numerical parity `rel_l2 <= 5.0e-5` vs exact FP32 reference).
   - `moe_cutlass.py` — hybrid dispatcher routing layers 0–19 (NVFP4) to `_inkling_nvfp4_xe2` and layers 20–23 (MXFP8) to `vllm-xpu-kernels` CUTLASS Xe2 grouped GEMM.
   - `vllm-inkling2-xpu-cutlass-optin.patch` — opt-in hook in `vllm_inkling2_xpu/moe.py` gated by `INK2_CUTLASS_MOE=1` (default `0` preserves exact stock plugin behavior).
   - `build-nvfp4-cutlass-xe2.sh` — reproducible `icpx` AOT build & install script (requires `intel/sycl-tla@7b56b936` and `vllm-project/vllm-xpu-kernels@e9b6e4b2` headers).

## Apply 1: Required PR `#69186` patch

If the environment was installed with `uv --link-mode hardlink`, break the hardlink on `logits_processor.py` before patching so the shared package cache is not modified:

```bash
LP=$(python3 -c "import inspect, vllm.model_executor.layers.logits_processor as m; print(inspect.getfile(m))")
cp "$LP" "${LP}.orig" && rm -f "$LP" && cp "${LP}.orig" "$LP"
patch -p1 -d "$(dirname "$(dirname "$(dirname "$(dirname "$LP")")")")" < patches/inkling2-mini-vllm-xpu/pr69186-logits-processor.patch
```

## Apply 2: Optional Native NVFP4 CUTLASS Xe2 extension (`INK2_CUTLASS_MOE=1`)

```bash
source /opt/intel/oneapi/setvars.sh
git clone https://github.com/intel/sycl-tla.git /tmp/sycl-tla && git -C /tmp/sycl-tla checkout 7b56b936b87f799f80dde60c2fd8d11c1128cebb
git clone https://github.com/vllm-project/vllm-xpu-kernels.git /tmp/vllm-xpu-kernels && git -C /tmp/vllm-xpu-kernels checkout e9b6e4b24a69886a8b384edc06e7a631ad282705

VENV=./vllm-inkling-venv \
SYCL_TLA_DIR=/tmp/sycl-tla \
VLLM_XPU_KERNELS_DIR=/tmp/vllm-xpu-kernels \
  patches/inkling2-mini-vllm-xpu/build-nvfp4-cutlass-xe2.sh
```

Enable at runtime with `INK2_CUTLASS_MOE=1`; leave unset (or `0`) to run the exact unmodified stock `vllm-inkling2-xpu` Triton kernels.
