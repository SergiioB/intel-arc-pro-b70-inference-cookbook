---
title: "DeepSeek-V4-Flash-0731 K216 — dual-B70 prefill ubatch (2026-08-25)"
type: research
date: 2026-08-25
status: final
---

# DeepSeek-V4-Flash-0731 K216 — dual-B70 prefill ubatch (2026-08-25)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE**

Artifact: `heath0xFF/DeepSeek-V4-Flash-0731-REAP-K216-GGUF` UD-IQ3_XXS, 82.95 GiB, mmap.
Engine: llama.cpp SYCL `build-sycl-latest-qw38` commit `1692f9e50`.
Placement: dual only, `-sm layer -ncmoe 20 -ts 70,30 -fa on -ctk f16 -ctv f16 -dev SYCL0,SYCL1`.
Evidence: `results/2026/08/dsv4-k216-20260825Tdual2/` (decode ladder) and `results/2026/08/dsv4-k216-20260825T0707Z-prefill/` (ubatch).

`n_ubatch = min(n_batch, -ub)` in this commit. Raising `-ub` without `-b` is a no-op.

## Decode (already measured; not 30 t/s)

C1 p512/g128 engine predicted tok/s, n=3, `-b 256 -ub 256 -c 4096`:

| Config | ncmoe / ts | median tg | median pp |
|---|---|---:|---:|
| L1 | 30 / 78,22 | 2.17 | 7.94 |
| L3 | 20 / 70,30 | 7.26 | 39.16 |
| L5 | 15 / 65,35 | **8.87** (SUPERSEDED 2026-09-21: 9.25 clean / 9.58 patched) | 55.50 |

30 tok/s C1 decode was not reached and is not a realistic target on this IQ3_XXS SYCL MoE path.

## Prefill ubatch on L3

HTTP engine `prompt_per_second`, n=3 after same-shape warmup, unique entropy prefix, `cache_n=0`. Cap 230 W during campaign; 150 W restored.

| Cell | -c / -b / -ub | actual prompt tokens | median pp | min–max |
|---|---|---|---:|---|
| L3 baseline | 4096 / 256 / 256 | ~547 | 39.16 | 30.8–44.5 |
| P512 p512g128 | 4096 / 512 / 512 | 544–550 | 40.26 | 25.4–54.7 |
| P1024 p512g128 | 4096 / 1024 / 1024 | 544–551 | 41.95 | 29.3–62.0 |
| P512 p2048g1 | 4096 / 512 / 512 | 2048–2052 | 80.65 | 77.8–82.6 |
| P1024 p2048g1 | 4096 / 1024 / 1024 | 2047–2054 | **104.55** | 67.9–112.5 |
| P8192 p8192g1 | 8448 / 8192 / 512 | 8066–8075 | 83.72 | 78.4–85.4 |
| P8192u1024 p8192g1 | 8448 / 8192 / 1024 | 8069–8071 | **105.12** | 96.5–110.8 |

p512 mixed with g128 is a bad prefill judge: one tiny tail microbatch plus high variance. Dedicated p2048/g1 and p8192/g1 show the real flag gain: about **1.26–1.30×** from effective ub512 → ub1024. Not 2×.

Post-load free VRAM stayed comfortable on L3 (~5.7–5.9 / 7.6–7.8 GiB). c8448 did not blow KV; DSV4 SWA/compressed caches do not scale like dense full-context KV.

## Do not retry

- `-ts 46,54` or `50,50` with low `-ncmoe` → SYCL1 13.73 GiB alloc fail (C-001 / C-004). Relaxed L0 limits do not fix it.
- `-sm row` → segfault during load (R1 / R2).
- Single GPU (user constraint).
- `--no-mmap` / `--mlock` on this GGUF.
- q4_0 K cache.
- Flipping `ggml_sycl_supports_mmq()` (always false; no IQ3_XXS MMQ at this commit).
- `GGML_SYCL_ENABLE_GRAPH=1` on dual + `MUL_MAT_ID` (capture rejected; 0% expected).

## Remaining prefill wall (kernels, not flags)

Current multi-token `MUL_MAT_ID` still D2H expert IDs + `stream->wait()`, CPU counting sort, per-expert host loop, and full IQ3_XXS→FP16 expand before oneMKL. Ranked leftover work:

1. Device-resident grouped `MUL_MAT_ID` (best first upstreamable change).
2. IQ3_XXS tile-local multi-row GEMM / MMQ without materializing the whole expert as FP16.
3. Fuse routed gate+up packing. Graph capture only after (1).

Keep FA on. FA-off is a secondary p8192 diagnostic, not a 2× path.
