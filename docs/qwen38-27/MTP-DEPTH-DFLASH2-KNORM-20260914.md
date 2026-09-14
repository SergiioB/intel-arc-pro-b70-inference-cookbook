# MTP Depth Re-confirmation + DFlash2 Root Cause on XPU — 2026-09-14

Single Intel Arc Pro B70 (32 GB), Qwen3.8-27B GPTQ-INT4 sym G128 (MTP draft BF16),
vLLM XPU pinned image `f01e24f6` (vLLM 0.27.2rc1.dev77+gac7509e2b), TP1, one card,
compile-only (torch.compile on, XPU graphs off), prefix cache off, recommended
non-thinking sampling (temp 0.7 / top_p 0.8 / top_k 20 / presence 1.5), entropy-first
unique prompts, 1 warmup discarded per cell, medians over endpoint completion tokens.
Self-reported, n=3 screens — diagnostic, not a records run. Power: both depth arms
ran at a 230 W configured cap; the DFlash2 diagnostic ran at 150 W.

## 1. MTP depth 4 confirmed optimal (single card)

| Depth | p512/g128 decode | p8192/g128 decode | Mean accepted length |
|---|---:|---:|---:|
| MTP4 | **50.78 tok/s** | **46.58 tok/s** | ≈3.3–4.0 |
| MTP6 | 40.87 tok/s (−19.5%) | 42.10 tok/s (−9.6%) | ≈3.3–4.2 |

Per-position acceptance on natural text decays to ≈0.11–0.26 at draft positions 5–6,
so depth 6 pays draft + verify cost for near-zero return. Depth 4 stays the serving
spec. (n=3 screens; both arms shared the same cap and config, so the relative result
is clean; absolute values are not comparable across power caps or host-load states.)

## 2. Why DFlash2 acceptance collapses on XPU — root cause found

Upstream **vLLM PR #56431** (open as of 2026-09-14; kernel fix in
vllm-xpu-kernels#579, also open): the XPU `rms_norm` kernel treats DFlash's stacked
per-layer K-norm weight `[num_layers, head_dim]` as 1-D and applies **layer 0's
weight row to every draft layer**. This corrupts draft context K for layers > 0.

A local port of the per-layer fallback onto the pinned image, plus the earlier
adaptation layer (torch selector walk + SDPA non-causal draft attention), moved the
greedy acceptance diagnostic (three content classes, coherent output on all):

| Metric | 2026-08-30 stack | 2026-09-14 (K-norm fix) |
|---|---:|---:|
| Mean accepted length | ≈0.71–0.85 | ≈1.62–2.26 |
| Mean draft acceptance rate | 0.10–0.12 | 0.089–0.180 |

Recovery is real but **insufficient**: still below matched MTP4 (≈3.3–4.0 accepted
length) and far below the adoption gate. A second defect remains — all-NaN selector
score walks at deeper draft positions (consistent with upstream issue #54928, still
open). Attribution is confounded: target precision (FP8→GPTQ), the K-norm patch, and
the topology (TP2→TP1) changed together versus the August no-go, so the recovery is
a combined-effect bound, not an isolated effect size.

**Practical verdict: DFlash2 on Intel Arc Pro B70 remains not-ready. Keep MTP4.**
Retry condition: an official image containing both #56431 and vxk#579, plus an
upstream resolution of the selector-NaN path (#54928 / #56692).

## 3. New XPU serving traps (all reproduced 2026-09-14)

1. **Parallel inductor compile workers can OOM the Level Zero context at engine
   init** on a secondary card (large context + MTP + GPTQ):
   `UR_RESULT_ERROR_OUT_OF_RESOURCES` during `_compile_to_module`. Fix:
   `TORCHINDUCTOR_COMPILE_THREADS=1`. Load time ~3–4 min single-threaded.
2. **DFlash2 draft path cannot torch.compile on this image** — inductor dies at init
   with a dynamic-shape stride assertion in the draft's context-KV precompute
   (reproduced twice; raw log overwritten by a later failure, recorded as
   recalled-not-retained). Run the DFlash2 diagnostic with `--enforce-eager`.
3. **Eager DFlash2 additionally requires `--max-num-seqs 1`** — at seqs 8 the engine
   dies with `causal_conv1d does not support spec-decode and non-spec tokens in the
   same invocation` (retained log). Working flag set: `--enforce-eager
   --max-num-seqs 1 --async-scheduling --block-size 64 --mamba-ssm-cache-dtype float16`.
4. **Draft-model registry inspection resolves the wrong vllm tree unless the
   container runs with `--workdir /`** — the image's `/workspace/vllm` checkout
   shadows site-packages and any port/overlay under it fails with
   `ModuleNotFoundError` (silent wrong-tree failure).

## Reproducibility notes

- Same-session control: both depth arms measured on one server relaunch each, same
  artifact/image/sampling/cache state; acceptance numbers are server-counter ranges
  transcribed at run time (log-retention slip disclosed in the private evidence tree).
- No cross-cap or cross-host comparisons are claimed. The 2026-09-13 single-card
  campaign (45.7 tok/s MTP4 at 150 W) and this screen (50.8 at 230 W) differ in cap
  and host load — directional only.
