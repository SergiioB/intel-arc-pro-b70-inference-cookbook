---
title: "Nemotron vLLM “10k prefill” — what was actually measured"
type: research
date: 2026-09-15
status: final
---

# Nemotron vLLM “10k prefill” — what was actually measured

**Date:** 2026-08-13  
**Rule:** every Nemotron vLLM input number in this repo is a **cold input rate**
(`prompt_tokens / client TTFT`). None is isolated engine prefill. None is
llama-bench `pp`.

## Cite this

| Claim you want | Number | Cell | n | Cache | Isolation | Use? |
|---|---:|---|--:|---|---|---|
| Isolated DFlash input (Lane 1) | **7160** | p8192/**g1** | 5 | off, 0 hits | production pinned off | **yes** |
| Isolated DFlash decode (representative) | **186.61** | p2048/g128 | 5 | off | production pinned off | **yes** |
| Isolated DFlash long-prompt decode | 157.92 | p8192/g128 | 5 | off | production pinned off | yes, as that cell |
| No-spec graph decode (n=5) | 93.00 / 87.25 | p512 / p8192 g128 | 5 | off | yes | yes, no-spec path |
| No-spec graph input (n=5, decode cell) | 8368 | p8192/**g128** TTFT | 3 | off | yes | notes only; n=3 |
| “10k prefill” | 10371 → **10349** | p8192/**g128** TTFT | 3 | off | A-grade n=3 | **do not headline** |
| 16K-budget n=3 spike | **9874** | p8192/**g1** | 3 | off | isolated | **do not headline** |
| 16K-budget n=5 | 6096 | p8192/**g1** | 5 | off | isolated | notes only; **lost** vs 7160 |

Evidence:
- DFlash n=5: `results/2026/08/nemotron-dflash-bf16-n7-n5-20260813T082203Z-1407994/`
- No-spec n=5: `results/2026/08/nemotron-gptq-graph-n5-20260812T2045Z/`
- A-grade n=3: `results/2026/08/nemotron-gptq-det-a-grade-20260812T2236Z/`

## Why 10,349 is not “the 10k prefill we saw”

1. **Wrong phase name.** 8192 / 0.7916 s = 10,349 is client TTFT on a
   **decode cell** (g128), so the interval includes scheduling + first-token
   work. Protocol name: cold input rate.
2. **n=3, not n=5.** Publication representative input is the isolated DFlash
   p8192/**g1** n=5 median **7160** (TTFT 1.144 s).
3. **It did not reproduce at n=5 even on no-spec.** Same stack, later n=5
   campaign, p8192/g128 TTFT-derived rate = **8368**.
4. **10371 was a rounding error.** Auditor: 8192 / 0.7916 s = 10,349.

DFlash n=5 p8192/g128 TTFT is actually *faster* (~1.041 s → ~7870 cold input
on that decode cell) than the g1 input cell (1.144 s → 7160). Do not mix those
two cells into one “prefill” headline. The Lane 1 input anchors are g1.

## What you may say in public

> On isolated n=5, DFlash cold input at exact p8192/g1 is **7160 tok/s**
> (client TTFT, cache off). An earlier no-spec n=3 screen showed ~10.3k at
> p8192/g128 TTFT; that is a different cell, n=3, and is not isolated engine
> prefill.

Do **not** say “we hit 10k prefill on DFlash.”

## 16K scheduler budget (Run 37) — n=3 won, n=5 lost

Raising `--max-num-batched-tokens` 8192 → 16384 on the same DFlash stack:

- n=3 screen p8192/g1 median **9873.61** (0 hits) looked like +38% vs 7160.
- Fresh isolated n=5 on the same prompts/stack: p8192/g1 median **6095.62**
  (5732–7006), **−14.9%** vs Run 36. Decode p2048/g128 fell to 167.16 vs 186.61.

Treat 9874 as a discarded screen, same class as the old 10.3k cell. Published
input stays **7160** at batched=8192. Evidence:
`results/2026/08/nemotron-dflash-prefill-budget16k-20260813T100159Z-1477342/` and
`results/2026/08/nemotron-dflash-prefill-budget16k-n5-20260813T101252Z-1486488/`.

## Context ceiling (Run 38)

`--max-model-len 120000` loaded (5328 MiB free, KV 295k) and completed
p119872+g32 = **119,904** tokens. That is a **capacity** result. Do not move
its g32 diagnostic rates into the Lane 1 speed card. **128K was not run.**
Evidence: `results/2026/08/nemotron-dflash-ctx-capacity-20260813T103627Z-1501089/`.
