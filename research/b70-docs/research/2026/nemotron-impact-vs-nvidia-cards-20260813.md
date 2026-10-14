---
title: "Impact note — B70 Nemotron DFlash vs 5090 / DGX Spark / H100"
type: research
date: 2026-08-13
status: final
---

# Impact note — B70 Nemotron DFlash vs 5090 / DGX Spark / H100

**Date:** 2026-08-13  
**Rule:** no invented 5090/Spark numbers. Official NVIDIA cards do not publish
matched C1 tok/s for those SKUs on this model.

## What we actually have on the B70

Isolated C1 n=5, cache off, 150 W configured, vLLM XPU, local GPTQ + local
DFlash BF16:

- representative decode **186.61 t/s** at p2048/g128
- p8192/g128 **157.92 t/s** (1.81× vs matched no-spec 87.25)
- p8192/g1 cold input **7160 t/s** (TTFT-derived)
- ~149–160 W cell-window draw

Same card, same target, no-spec graphs: **93 / 87 t/s**.  
Same card, llama.cpp MTP-4 (historical Q4_0, 230 W): **87.4 t/s** (different
engine, different artifact — `not_comparable` as a multiplier).

## What NVIDIA publishes (not a bake-off)

From the official BF16 / NVFP4 / DFlash cards (fetched 2026-08-13):

| Vendor claim | Hardware | Metric class | Why it is not our 186.6 |
|---|---|---|---|
| “40+ TPS/user” interactive | 1× H100, DSpark, concurrency ≤128, NVFP4/W4A16 | **per-user serving at batch** | Different concurrency, spec method, quant, and statistic |
| Fit / 256K memory-bound | 1× H100 or A100 80 GB, BF16 | capacity | Not speed |
| “Up to 1M tokens” | GB200 / B200 / 8× H100 | capacity | Not speed |
| DFlash SPEED-Bench accept length 3.16 | H100 / GB200 / 5090 mentioned as tested | acceptance, not tok/s | We measured 52% token accept at n_spec=7 (~3.6 accepted drafts/step including the verify token) — similar *acceptance class*, not a speed match |
| RTX 5090 | “tested” / GGUF local path named | **no official tok/s** | Cannot compare |
| DGX Spark | DSpark flags / GGUF local path named | **no official tok/s** | Cannot compare |

Fireworks listing metadata (~294 tok/s) is a hosted product number with
unknown hardware, batch, and cache. `not_comparable`.

## Honest impact statement

The result is **not** “B70 beat a 5090.” It is:

1. A **workstation Arc Pro B70 (32 GB, ~150 W)** now serves NVIDIA’s newest
   30B-A3B hybrid with the **official DFlash method** at **186.6 t/s C1
   median** on a real-workload cell.
2. That is **2.00×** the isolated no-spec graph path on the **same card,
   same target, same engine family** at p2048 is not the no-spec headline —
   use p8192 **1.81×** (`matched_except_speculation`) or the no-spec 93 t/s
   p512 as a different cell (`best_cell_cross_result` if mixed).
3. Official NVIDIA docs still treat 5090 / Spark as *supported local
   platforms* without publishing a C1 number we can multiply against.
   Until someone runs the same p2048/g128 C1 cache-off protocol on those
   cards, any “vs 5090” sentence is marketing.

## What is allowed in a post

Allowed:

- B70 C1 186.6 with full coordinates.
- 1.81× vs our no-spec n=5 p8192/g128.
- “NVIDIA publishes 40+ TPS/user on 1× H100 at ≤128 concurrency with
  DSpark — different metric; we are not claiming to beat H100.”
- “No official 5090/Spark tok/s exists on the model card to compare.”

Forbidden:

- “Faster than a 5090 / Spark / H100.”
- “10k prefill.”
- “194 t/s” as the representative number.
- Treating LocalMaxxing APPROVED as independent verification.
