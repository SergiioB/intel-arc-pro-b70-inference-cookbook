# Nemotron-3.5-Lightning-30B-A3B on the B70 (vLLM XPU)

Status: **no-spec graph path** (n=5). For DFlash (~186.6 t/s representative)
see [NEMOTRON-DFLASH-B70.md](NEMOTRON-DFLASH-B70.md).

Temperature-0 deterministic replay historically failed on the compiled/XPU-graph
path before the grouped-GEMM `at::zeros` fix. Treat no-spec 93/87 as real at
their coordinates; do not mix with the DFlash table. See the changelog.

## Stack

| Component | Exact tested value |
|---|---|
| Public image | `vllm/vllm-openai-xpu@sha256:1da0a95485455f08588c11080b9718992fd7d434c6a965d74654903a9d999c57` |
| vLLM observed in image | `0.26.1rc1.dev668+g3ee2df303` |
| `vllm-xpu-kernels` observed | `0.1.12.3` |
| Model artifact | published local conversion [`SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym`](https://huggingface.co/SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym) from `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` |
| Patches, in order | `patch_xpu_grouped_topk_native_v2.py` (graph-safe native grouped-topk + `torch.compiler.disable`; still required on this pinned image), B70 SSU JSON `ssu-b70-b8w4/` |
| Optional kernel rebuild | `patches/vllm-xpu-kernels/0001-zero-xe2-grouped-gemm-atomic.py` ([vllm-xpu-kernels#524](https://github.com/vllm-project/vllm-xpu-kernels/pull/524), still open). Do not apply `0002`. |
| Context / scheduler / memory | 16,384 / 8,192 / `gpu-memory-utilization=0.90` |
| Decode path | compiled mode + `VLLM_XPU_ENABLE_XPU_GRAPH=1` (PIECEWISE + FULL decode graphs), `--async-scheduling`, cache explicitly off |

## Why this exists

Eager vLLM decode on this model was **21.8 t/s** — anomalous for a 3B-active
MoE. Profiling found the cause: **~950 kernel launches per token, 32.2 ms/token
of CPU enqueue time vs 9.45 ms of actual GPU work**
(profile: `results/nemotron-gptq-builtin-profile-20260812T171658Z-671822/`).
XPU graph capture collapses the launches — the 4× result below. Do not compare
those 21.8 / 93 t/s cells to Qwen MTP or llama.cpp Nemotron numbers on this page.

## Measured (n=5 decode; graph determinism is a separate caveat)

C1 client post-first decode, exact p512/g128 and p8192/g128, prefix cache
explicitly off, one same-shape warmup discarded, n=5, **150 W configured cap**.
Statistic: median. Do **not** attach the earlier eager/compiled ~89–90 W A/B
to this graph n=5 window.

| Cell | Median (t/s) | Range | vs eager baseline |
|---|---:|---:|---:|
| p512/g128 | **93.00** | 92.96–93.03 | 4.27× |
| p8192/g128 | **87.25** | 87.22–87.31 | 4.01× |

- Eager baseline: 21.79 / 21.74 t/s (same model, no graphs).
- Coherence verified on all measured reps (structured, on-topic, exact 128
  tokens, `finish_reason=length`).
- **Caveat**: identical temperature-0/seed requests occasionally diverge at
  contested near-tie tokens (both continuations coherent). Isolation proved the
  router math is deterministic (full-eager replay passes); the race is in the
  XPU compiled/graph kernel execution. Upstream fix required.
- Prefill: no isolated engine-prefill measurement on this stack. A later
  no-spec n=3 screen showed ~10,349 tok/s from p8192/g128 TTFT; that is a
  **cold input rate**, n=3, and is **not** the isolated DFlash n=5 input
  (7160 at p8192/g1). See `NEMOTRON-DFLASH-B70.md`.

## Launch

```bash
# huggingface-cli download SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym \
#   --local-dir "$HOME/models/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym"
MODEL_DIR="$HOME/models/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym"
bash benchmarks/nemotron35-30a3/launch-nemotron-graph.sh "$MODEL_DIR" 8001
curl -f http://127.0.0.1:8001/health
```

The launcher applies `patch_xpu_grouped_topk_native_v2.py` and the SSU JSON
inside the container before `vllm serve`. The server imports vLLM from
`/workspace/vllm` — the patch targets both source trees.

## Conversion (source → GPTQ INT4 G64 symmetric)

The artifact is a local conversion; there is no official GPTQ. Contract:
4-bit, symmetric, group 64, signed range [-8,7], stored `signed+8` low-first
nibbles, BF16 scales, zero-filled qzeros, `g_idx = arange(K)//64`, norms
(excluded 1D) kept BF16. The published HF GPTQ repo is the reproduction default.
The MTP-head variant exists but is not usable yet (0% draft acceptance).

## LocalMaxxing

Self-reported submission APPROVED (2026-08-12): record `cmsqkxobn00odmr01blm07gi6`,
tokSOut 93 (p512/g128 n=5 median), prefill intentionally omitted, caveat in
notes. Earlier records with a derived prefill field and a duplicate remain on
the platform; the fixed record is the reference.

## Changelog

- 2026-08-13: DFlash isolated n=5 landed — see `NEMOTRON-DFLASH-B70.md`.
  This page remains the no-spec graph recipe.
- 2026-08-12: initial scaffold. Decode n=5 is real at the named cells; compiled
  / graph temperature-0 replay remains an upstream XPU caveat, not a speed
  retraction.
