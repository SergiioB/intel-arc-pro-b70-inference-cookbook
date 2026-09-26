# EXL3 on one B70 — first evaluation (2026-09-26, screen; not a recipe)

> Status: external route evaluated locally once. No pinned image in this cookbook's
> matrix, no benchmark catalog record, no reproduction contract yet. Outcome decisions
> live in [WHAT-WORKED](WHAT-WORKED.md).

## What it is

[exl3xpu](https://github.com/0xSero/exl3xpu) — a vLLM XPU plugin that decodes EXL3
(trellis) checkpoints with native ESIMD kernels on Battlemage. Evaluated here as an
alternative single-card Qwen3.8-27B route:

- Image: `ghcr.io/0xsero/exl3xpu@sha256:21412bdd…870fa8` (vLLM 0.26.1.dev, plugin
  baked in; attested build).
- Weights: `turboderp/Qwen3.8-27B-exl3` @ `113cf7ab…` (4.00 bpw, 15.7 GiB) — a
  different checkpoint from this repo's GPTQ-INT4 and GGUF arms. MTP3, fp8 KV,
  256K context, vision tower included.
- One B70. Note for dirty cards: the default `gpu_memory_utilization 0.965` exceeds
  this host's GPU0 startup free (~29.5/31.9 GiB); 0.90 with 64K context loaded fine.

## What we measured (lab screen, n=3 cells)

- **Exact-answer battery (greedy, thinking off): the only clean arm on this host.**
  6/6 including `sum(i*i for i in range(4))` = 14 (twice) and 9×7−3 = 60. The serving
  GPTQ-INT4 artifact answers 30 to the first (deterministic); a same-source Q4_K_M
  GGUF answers 14 but fails the arithmetic (59). Scope: 6 completions — a drift gate,
  not a broad quality ranking.
- **Decode class: same as the GPTQ champion, not faster.** Client wall ~58 tok/s at
  p~569/g128 and ~24.7 at p~8223 (prefill included); implied post-first ~65–71 vs the
  champion's 72–73. The author's 91.2 is a different cell (thinking-on prose,
  sustained windows).
- MTP acceptance is content-sensitive here too: random-word filler collapsed it to
  30–50 % (wall rate fell to 10–16 tok/s); natural templates recovered it.

## Why it matters / what's missing

The GPTQ speed program on this host is closed at its byte floor; quality is the open
axis, and EXL3 is currently the only evaluated Qwen3.8 arm that passes the exact-answer
gate. Missing before this could become a recipe: prose-corpus decode campaign,
repeat-stability, long-context needle, and agent-traffic replay — plus a decision on
carrying a second vLLM image generation in the matrix.
