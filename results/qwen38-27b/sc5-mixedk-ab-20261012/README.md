# sc5-mixedk-ab-20261012 — Qwen3.8-27B EXL3 SC_5.00bpw_H6 paired A/B

Public evidence for the `qwen38-27b-sc5` route added in
[`docs/qwen38-27b/EXL3-XPU.md`](../../../docs/qwen38-27b/EXL3-XPU.md)
"Mixed-bit SC_5.00bpw_H6". Recorded 2026-10-12 on one Intel Arc Pro B70
(32 GiB), 230 W configured cap, official-lab self-report.

## What was compared

| Arm | Weights | Image |
|---|---|---|
| A | `turboderp/Qwen3.8-27B-exl3` rev `113cf7ab` (EXL3 trellis 4.00bpw) | `ghcr.io/0xsero/exl3xpu:0.1.0@sha256:21412bdd7535e9c653eeb3d099dce3bc79a83d440def2cd9556111c79c870fa8` |
| B | same repo, branch `SC_5.00bpw_H6`, rev `dfe2f4fd71bf9a1994051d4d47e6460fa96dd723` (mixed-bit K=4–7, ~5.00bpw) | `ghcr.io/sergiiob/exl3xpu-mixedk:20261012c@sha256:7bb0c5cfed8a35da8f4db0034338d8dbe5f53731253d96ea8f1db2b5bddd4a40` (0xsero 0.1.0 + mixed-K patches, `SergiioB/exllamav3` `b70/` @ `244f57e5`) |

Both arms: MTP3, FULL_DECODE_ONLY graphs, fp8 KV, `gpu_memory_utilization`
0.90, `max_model_len` 65,536, prefix caching on. Arm A is the published
upstream image; arm B is the mixed-K patched image — the comparison is each
quant on its route-of-record stack, not an image-isolated delta.

## Protocol

Paired ABBA interleave, 4 server loads (A B B A): block 1 = reps 1–3
(A then B), block 2 = reps 4–5 (B then A); pair *i* uses the same prompt
index on both arms. n=5 measured reps + 1 discarded warmup per arm and cell,
C1, `ignore_eos`, thinking off, recipe sampling (temp 0.7 / top_p 0.8 /
top_k 20 / presence_penalty 1.5). Every measured cell is prefix-cache cold
(`prefix_cache_hits_total` delta 0). xe-event and VRAM guards ran clean
between loads. Draw is the median `energy1_input` delta per cell (~230 W).

## Files

- `summary.json` — compiled canonical numbers: per-arm medians (post-first
  tok/s, wall tok/s, TTFT, MTP acceptance, power), correctness results,
  exclusions.
- `recipe-ab-samples.jsonl` — one sanitized row per request: arm, cell,
  block, rep, warmup flag, exact endpoint token counts, finish reason,
  timing fields, spec-decode counter deltas, energy-derived average W.

## Results (medians, n=5)

| Cell | A plain 4.00bpw | B SC_5.00bpw_H6 | B/A |
|---|---:|---:|---:|
| p524/g128 post-first tok/s | 63.56 | **56.62** | 0.891× |
| p8124/g128 post-first tok/s | 64.34 | **56.82** | 0.883× |

Correctness on arm B: 46/46 mixed-K kernel checks, 6/6 exact-answer canaries
+ 3/3 long prompts (8,161–8,162 tokens). See `summary.json` for the exact
field definitions and caveats.
