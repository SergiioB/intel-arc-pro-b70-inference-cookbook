# Qwen3.8-27B — Draft INT4 (S + M1)

Runtime patches that **requantize the speculative-draft side** of MTP to
INT4 g128 symmetric (round-to-nearest, not Hessian GPTQ) and route those
linears through `torch.ops._xpu_C.int4_gemm_w4a16`.

Production image: `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
(vLLM `0.27.2rc1.dev77+gac7509e2b`, kernels `0.1.12.3`). The patches fail
closed if `qwen3_5_mtp.py` anchors differ - and that guard is anchor-only:
it knows nothing about topology, so it does not protect TP>1.

> **Do not enable with `--tensor-parallel-size 2`.** Under TP2, target
> verification does not protect the emitted sequence: output can be
> corrupted or non-deterministic even at temperature 0 (issue #9). The
> INT4 draft overlay is a single-card C1 speed keep.

## What is and is not lossless

| Piece | After S+M1 |
|---|---|
| Target body (GPTQ-INT4) | unchanged |
| Target / verify LM head | still FP16/BF16 |
| Draft LM head copy | runtime INT4 g128 RTN |
| Draft MTP 5 linears | runtime INT4 g128 RTN |

Single-card C1 only: the target still verifies. Draft **logits are not**
identical. Acceptance can drop. This is a **speed** keep, not a
quality-parity claim.

## Same-image n=5 (2026-08-18)

Champion `f01e24f6`, MTP4, cache off, MBT 8192, U=0.88, fp8 KV, C1,
configured **230 W**. Timing: client post-first / input÷TTFT. Sampling:
greedy for the 2026-08-18 n=5 tables; at the Qwen-recommended non-thinking
preset (0.7/0.80/20/presence 1.5) the same C1 cell measures **102.61**
(accept 91.8%) — see QWEN38-VLLM-XPU.md §11. Accept from
`vllm:spec_decode_*_total`. Raw: cookbook campaign sibling
`qwen38-pr-n5-20260818T214749Z` + long-ctx `qwen38-pr-long-20260818T221813Z`.

| Cell | BF16 draft (cookbook) | Draft INT4 S+M1 | Δ |
|---|---:|---:|---:|
| p512/g128 median n=5 | 81.20 tok/s (78.76–81.97) | **112.65** (111.58–117.73) | **+38.7%** |
| p8192/g128 median n=5 | 77.52 | **103.63** (99.10–107.70) | **+33.7%** |
| p8192/g1 cold input | 1691.1 | 1696.0 | +0.3% (flat) |
| short agentic 3×5 g256 (**cache off**) | 43.27 | **57.45** | **+32.8%** |
| accept p512 / p8192 | 95.86% / 96.02% | 94.44% / 94.59% | −1.4 pp |
| measured median W p512 | 195.2 | 205.0 | — |

Long-context C1 agentic (g128, 0 crashes, actual first prompts ~8.3K /
~16.5K / ~32.9K): **+37%** at 8K (68.73 vs 50.06), **+26%** at 16K
(70.86 vs 56.29). 32K was isolated C1 (~245 MiB free) and only **+5%** —
not a serving headline.

Do **not** headline +51%. That table compared a different image, patch
list, MBT, and cache mode (`not_comparable`).

The n=3 chat-harness screen (59.1 → 83.9) used ~p530 filler, not exact
p512. It is superseded.

## Generation curve + isolated C1 128K (2026-08-19, Run 42)

Same S+M1 arm, same image, MTP4, cache off, C1, configured **230 W**,
`ignore_eos`, n=5, zero prefix-cache-hit deltas. g128 Lane 1 above was
**not** remeasured. After-load `visible_avail` = **870 MiB** → isolated
C1 capacity only (500–1,023 MiB class). Mid-request free VRAM dipped to
tens of MiB. **Not a serving or preferred-perf 128K claim.**

![S+M1 generation and isolated-C1 128K](../assets/b70-qwen38-draft-int4-ctx-gen.svg)

### Generation length (client post-first tok/s)

g32 is not sustained. g512 is a length diagnostic, not a predeclared
sustained window.

| Cell | median | max | range | accept | median W | n |
|---|---:|---:|---|---:|---:|--:|
| p512/g32 | 101.57 | 102.73 | 89.59–102.73 | 84.9% | 215.5 | 5 |
| p512/g256 | 110.76 | 112.87 | 85.13–112.87 | 89.0% | 195.6 | 5 |
| p512/g512 | 101.20 | 112.71 | 73.24–112.71 | 81.2% | 193.8 | 5 |
| p8192/g32 | 93.92 | 94.64 | 82.50–94.64 | 86.5% | 229.1 | 5 |
| p8192/g256 | 75.75 | 80.33 | 69.00–80.33 | 62.5% | 215.1 | 5 |
| p8192/g512 | 64.96 | 72.22 | 56.81–72.22 | 51.6% | 206.3 | 5 |

### Long-context g128 (isolated C1)

All rows exact tokens, `finish_reason=length`. p130944+g128 completed
131,072 tokens on every rep.

| Cell | median post-first | max | range | input/TTFT | accept | median W | n |
|---|---:|---:|---|---:|---:|---:|--:|
| p16384/g128 | 94.92 | 95.41 | 90.52–95.41 | 1495 | 92.6% | 226.0 | 5 |
| p32768/g128 | 88.25 | 91.59 | 85.00–91.59 | 1269 | 93.8% | 227.6 | 5 |
| p65536/g128 | 76.72 | 79.79 | 76.53–79.79 | 1000 | 94.8% | 228.6 | 5 |
| p98304/g128 | 67.93 | 70.50 | 65.24–70.50 | 827 | 93.4% | 229.0 | 5 |
| p130944/g128 | 62.52 | 65.57 | 58.74–65.57 | 708 | 92.6% | 229.1 | 5 |

Raw: `B70-DOCS/results/qwen38-sm1-ctx-gen-20260819T082145Z/`. Input
column is cold input rate (prompt tokens / client TTFT), not isolated
prefill.

## Prefix-on agentic (2026-08-19, Run 43)

Matched same-image A/B with **prefix caching on** (`enable_prefix_caching:
True` in both server logs). C1, configured **230 W**, client post-first.
After-load 865 / 871 MiB → isolated C1. Comparison:
`matched_except_quant`. Keep the cache-off agentic table above separate.
Do not headline prefix-on as faster decode than cache-off.

Client post-first tok/s, median, C1, prefix cache on, 230 W cap configured:

| Cell | Cookbook BF16 draft | Draft INT4 S+M1 | Δ |
|---|---:|---:|---:|
| short coding 3×5 g256 (3 sessions, 15 turns) | 43.81 | **58.86** | **+34.4%** |
| 8K start g128 (3 sessions, 15 turns) | 48.04 | **66.99** | **+39.4%** |
| 16K start g128 (2 sessions, 8 turns) | 54.40 | **65.92** | **+21.2%** |

Short t5 hits = 1664 tokens/session. Long t2+ hits 4992–18304. 0 crashes.
SVG: [b70-qwen38-draft-int4-agentic-cacheon.svg](../assets/b70-qwen38-draft-int4-agentic-cacheon.svg).
Raw: `B70-DOCS/results/qwen38-agentic-cacheon-20260819T093858Z/`.

## Prefix-on agentic 32K→128K (2026-08-19, Run 44)

Same arms, prefix cache on, C1, 230 W, client post-first g128. After-load
**825 / 816 MiB** → isolated C1. 96K is one session. 128K is a **single
cold t1** (hits=0). Do not headline S+M1 at 128K.

| Cell | Cookbook BF16 draft | Draft INT4 S+M1 | Δ | n |
|---|---:|---:|---:|---|
| 32K start g128 | 46.34 | **61.02** | **+31.7%** | 2 sessions, 6 turns |
| 64K start g128 | 44.83 | **56.20** | **+25.4%** | 2 sessions, 4 turns |
| 96K start g128 | 40.66 | **47.85** | **+17.7%** | 1 session, 2 turns (thin) |
| ~128K t1 only | **43.82** | 37.48 | **−14.5%** | 1 session, 1 turn |

Do not mix with Run 42 synthetic cache-off p130944/g128 **62.52**.
Raw: `B70-DOCS/results/qwen38-agentic-128k-20260819T102259Z/`.

## Prompt-content sensitivity (2026-08-19, important for all MTP numbers)

Same image, same patches (v5 + S+M1), same server config, C1 g128 — only the
prompt family changes:

| Prompt family | p512/g128 median | MTP acceptance |
|---|---:|---:|
| Calibrated real-world Pi prompts (cache ON, prefix on) | **100.2** (92.9–105.1) | 89–96% |
| Same Pi prompts (cache OFF, 2026-08-18 record) | **112.65** | 94.4% |
| Degenerate `INDEX INDEX…` filler (cache ON) | 71.5 | ~44% |

Two separate effects: (1) **prompt content drives MTP acceptance** — the
draft (INT4 or BF16) predicts realistic text far better than degenerate
repetition, so filler-based harnesses understate any MTP stack by ~30% here;
(2) prefix caching ON costs a further ~5–11% at zero hits on this stack.
Never compare decode cells measured with different prompt families, and
treat INDEX-filler numbers as a lower bound. p8192/g1 cold input is
unaffected: 1694 tok/s both ways. Raw:
`B70-DOCS/results/qwen38-pi-isolation-20260819T120424Z/`.

## Required companion

Stack with **GDN mixed-split v5** (`patches/patch_gdn_mixed_split_v5.py`)
if mixed spec + prefill can share an invocation. The original full-buffer
split is OOB on kernels 0.1.12.3.

Pointer wrap (`patch_mtp_ptr_wrap.py`) is optional and **int64-only**.

## Env

- `B70_DRAFT_LMHEAD_INT4=1` — Phase S (patch default off)
- `B70_DRAFT_MTP_INT4=1` — Phase M1 (patch default off)

Set both for the Qwen3.8-27B MTP4 speed keep. Leave unset to recover
BF16-draft acceptance.

## Quality A/B (2026-08-19, keep)

Temperature 0, thinking off, 15-task suite, greedy replay. Same image /
checkpoint / MTP4 as the n=5 speed card. Raw:
`B70-DOCS/results/qwen38-draft-quality-20260819T064654Z/`.

| Arm | Pass | Greedy replay |
|---|---|---|
| BF16 draft (cookbook) | **12/15** | 15/15 |
| Draft INT4 S+M1 | **12/15** | 15/15 |

No C-only regressions. 14/15 greedy SHA match. Shared fails:

- `needle-2k` — both refuse a buried “secret token” (safety), wording differs
- `word-problem` / `reasoning-short` — both truncated at `max_tokens` with
  **identical** SHA; the truncated math is the same on both arms

Passed on both: arithmetic, runnable Python (sum-of-squares, fizzbuzz, empty-mean
fix), JSON object, capitals, prime list, Spanish translation, qwen3_xml tool
call, HTML button snippet.

**Keep the overlay as optional.** Do not make it the default recipe: target
GPTQ quality vs BF16 teacher is still untested (Steve’s 30-vs-14 canary was
the target, not the draft). No logit/KL packet.

## Status

Optional recipe keep on this champion image. Speed **and** this 15-task
`task_quality_tested` A/B: no draft-INT4 regression vs BF16 draft. Prefix-on
agentic also wins vs cookbook MTP4 (Run 43). Not token or KL parity. Not
the default 83.7 LMX row.
