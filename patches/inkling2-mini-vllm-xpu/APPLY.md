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
3. **Optional XPU Vision Tower & FP8 KV (`vllm-inkling2-xpu`, opt-in flags):**
   - `vision.py` + `ops_vision.py` — pure PyTorch XPU port of the 23-layer BF16 `Inkling2VisionEncoder` (+0.55 GiB VRAM, `29 ms` for 2 images, bidirectional SDPA, no CUDA/PTX).
   - `vllm-inkling2-xpu-vision-fp8kv.patch` — wires `model.visual` when `--language-model-only` is omitted and adds `--kv-cache-dtype fp8` (`float8_e4m3fn`) with a reusable 256 MiB BF16 active-page prefill scratch (96% of BF16 prefill speed up to 64K; unlocks `177K`/`323K` KV w/ DSpark+Vision and `569K` KV / `512K` no-spec!).
   - `install-vision-fp8kv-xpu.sh` — breaks `uv` hardlinks safely and installs both modules + patch.
4. **Optional opt-in kernels (`vllm-inkling2-xpu`, `INK2_SYCL_DECODE_ATTN=1`):**
   - `attn_decode_xe2.cpp` — AOT `bmg_g31` SYCL `M=1` paged decode attention.
   - `vllm-inkling2-xpu-sycl-decode-optin.patch` — opt-in dispatch hook gated by `INK2_SYCL_DECODE_ATTN=1` (default `0` preserves exact stock Triton).
   - `build-sycl-decode-attn-xe2.sh` — reproducible `icpx` AOT build & install script.
5. **Optional opt-in kernels (`vllm-inkling2-xpu`, `INK2_HYBRID_PREFIX_SDPA=1` + `INK2_ESIMD_DECODE_ATTN=1`):**
   - `attn_esimd_xe2.cpp` — AOT `bmg_g31` ESIMD/XMX `PagedXmx<M>` decode attention (FP8 E4M3 + BF16 KV, two-pass split-KV, `-ze-opt-large-register-file`).
   - `inkling2-hybrid-sdpa-esimd.patch` — adds both opt-ins to `vllm_inkling2_xpu/attention.py` (chunked oneDNN SDPA over the bias-free prefill prefix, fp32-LSE merged with the Triton relative-bias tail) and `vllm_inkling2_xpu/ops/attention.py` (ESIMD `PagedXmx` decode dispatch: `mql<=8`, GQA group `3`, `head_dim 128`, single request). Both flags default `0` = exact stock behavior.
   - `build-esimd-decode-attn-xe2.sh` — reproducible `icpx` AOT build & install script (also needs `ocloc` on `PATH`).

## Apply 1: Required PR `#69186` patch

If the environment was installed with `uv --link-mode hardlink`, break the hardlink on `logits_processor.py` before patching so the shared package cache is not modified:

```bash
LP=$(python3 -c "import inspect, vllm.model_executor.layers.logits_processor as m; print(inspect.getfile(m))")
cp "$LP" "${LP}.orig" && rm -f "$LP" && cp "${LP}.orig" "$LP"
patch -p1 -d "$(dirname "$(dirname "$(dirname "$(dirname "$LP")")")")" < patches/inkling2-mini-vllm-xpu/pr69186-logits-processor.patch
```

## Apply 2: Optional Native NVFP4 CUTLASS Xe2 extension (`INK2_CUTLASS_MOE=1`)

The AOT `bmg_g31` step also needs `ocloc` on `PATH`. The oneAPI compiler packages do not ship it; install the `intel-ocloc` package from the [compute-runtime release](https://github.com/intel/compute-runtime/releases) that matches your Level Zero driver. Verified end to end with icpx `2026.1.1`, `torch 2.15.0+xpu`, compute-runtime `26.40.40196.8`, and plugin `2e9d28b`; the build, opt-in patch, and module import were checked on 2026-11-23.

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

## Apply 3: Optional XPU Vision Tower + FP8 KV Cache

```bash
VENV=./vllm-inkling-venv patches/inkling2-mini-vllm-xpu/install-vision-fp8kv-xpu.sh
```

Launch with `VISION=1` and/or `KV_DTYPE=fp8` via `benchmarks/inkling2-mini/launch-inkling2-mini-vllm-xpu.sh`. Default (`VISION=0 KV_DTYPE=auto`) preserves `--language-model-only` and BF16 KV.

## Apply 4: Optional Opt-In AOT `bmg_g31` SYCL `M=1` Decode Attention (`+71%` @ 256K!)

Zero external header dependencies beyond Intel oneAPI `icpx` + PyTorch XPU:

```bash
VENV=./vllm-inkling-venv patches/inkling2-mini-vllm-xpu/build-sycl-decode-attn-xe2.sh
```

Enable at runtime with `INK2_SYCL_DECODE_ATTN=1` (default `0` preserves exact stock Triton). Accelerates `M=1` (no-spec) long-context FP8 KV decode by `+19.8%` @ 64K, `+39.3%` @ 128K, and **`+71.0%` @ 256K** (`20.07 -> 34.32 tok/s`) under live XPU graph replay while keeping Triton XMX DPAS on `M>=2`.

## Apply 5: Optional Chunked oneDNN Prefix SDPA + ESIMD XMX Decode (`INK2_HYBRID_PREFIX_SDPA=1`, `INK2_ESIMD_DECODE_ATTN=1`)

**Prerequisites:** requires **Apply 3** (`install-vision-fp8kv-xpu.sh`) and **Apply 4** (`build-sycl-decode-attn-xe2.sh`) to be applied first (`inkling2-hybrid-sdpa-esimd.patch` stacks on their `attention.py` and `ops/attention.py` anchors; `build-esimd-decode-attn-xe2.sh` runs an atomic `patch --dry-run` preflight and exits cleanly without partial edits if prerequisites are missing).

Zero external header dependencies beyond Intel oneAPI `icpx` + PyTorch XPU; `ocloc` must be on `PATH` for the `bmg_g31` AOT device target. The build compiles `_inkling_esimd_attn_xe2` with `-options -ze-opt-large-register-file`, installs it into `vllm_inkling2_xpu/ops/`, and applies the combined opt-in patch:

```bash
source /opt/intel/oneapi/setvars.sh
VENV=./vllm-inkling-venv patches/inkling2-mini-vllm-xpu/build-esimd-decode-attn-xe2.sh
```

Enable at runtime with `INK2_HYBRID_PREFIX_SDPA=1` and/or `INK2_ESIMD_DECODE_ATTN=1` (independent flags, both default `0` = exact stock Triton path; ESIMD decode split defaults `INK2_ESIMD_SWA_SPLITS=16`, `INK2_ESIMD_FULL_SPLITS=128`, fixed per window type for graph-capture safety). Measured on 2026-11-23 in two same-session A/B ladders (24/24 kernel parity, `rel_l2 <= 2.7e-3`):

- **Prefill (`INK2_HYBRID_PREFIX_SDPA=1`):** chunked oneDNN SDPA (8192-token windows) over the bias-free KV prefix + Triton relative-bias tail: `2,706 -> 3,048` tok/s @ 64K (+12.6%), `2,149 -> 2,575` @ 128K (+19.9%), `1,586 -> 2,023` @ 245K (+27.6%) cold-prefill rate; decode path unchanged.
- **Decode (`INK2_ESIMD_DECODE_ATTN=1`):** two-pass ESIMD XMX `PagedXmx` on full-attention layers: `54.1 -> 68.0` tok/s @ 64K, `59.6 -> 76.1` @ 128K, `37.6 -> 56.6` @ 245K raw post-first rate (`+11–12%` / `+19–20%` / `+32–33%` acceptance-normalized step/s) under live XPU graph replay; `-22%` to `-34%` J/token at 128K/245K.

Evidence: `results/inkling2-mini/vllm-xpu-20261123/`; catalog records `inkling2-mini-vllm-xpu-hybrid-prefix-sdpa-20261123` and `inkling2-mini-vllm-xpu-esimd-decode-20261123` in `data/benchmarks.v1.json`.
