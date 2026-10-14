---
title: "Cookbook PR #1 / #2 review — 2026-08-18"
type: research
date: 2026-08-18
status: final
---

# Cookbook PR #1 / #2 review — 2026-08-18

**n=5 Lane 1 + short agentic + long-ctx 8K/16K/32K is E2 raw evidence.**
Claims audit 2026-08-18: named cells CONFIRMED; 32K PARTIAL (isolated C1);
quality NOT MEASURED. n=3 screen below is **superseded**.

Cookbook master now includes both keeps:

- PR #1 merged `06a14ce` — v5 mixed-split (Cn correctness)
- PR #2 merged `cd241b2` — draft INT4 S+M1 (optional MTP speed)

PRs (fork `popisec`):

- https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/pull/1
- https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/pull/2

Cookbook working setup (champion):

- image `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
- vLLM `0.27.2rc1.dev77+gac7509e2b`, kernels `0.1.12.3`
- Qwen3.8-27B GPTQ-INT4-sym-G128 + BF16 MTP4
- required patches: `patch_mtp_nightly.py` then `patch_mtp_boundary.py`
- cache off, MBT 8192, U=0.88, configured **230 W** (measured draw: not measured)
- timing: client-observed post-first / input÷TTFT, chat `/v1/chat/completions`

Author gates were on **legacy** `2c427ef` / kernels `0.1.12`.

Raw: `results/2026/08/qwen38-pr-ab-20260818T205619Z/`
Live log: `results/_quarantine/qwen38-pr-ab-live.log`

## Architect (fixed files)

`ARCHITECT_VERDICT: OK` on

- `patches/patch_gdn_mixed_split_v5.py` (compact + arange + scatter out **and** `z`)
- `patches/patch_mtp_ptr_wrap.py` (int64 sites only, fail closed)
- `docs/qwen38-27/DRAFT-INT4-S-M1.md` (research-only; no champion +51% / lossless)

Open GitHub PRs were later updated to the fixed algorithm / retracted +51%
lead (`0fa02c5` / `88beb9c`). Local cookbook: `review/pr1-pr2-champion-fixes`
@ `95b0d80`. Cookbook `origin/master` is still the two MTP patches only.

## Why the filed PR #1 algorithm is not the overlay to land

Fused `gdn_attention` on 0.1.12.3 `narrow`s tensors to `num_actual_tokens`.
Global mixed `token_indx` + `num_actual_tokens = group_count` is OOB. v5 is
the contract-correct split. C1 speed is flat (infra, not a keep).

Ptr-wrap site 3 (`dst_ptrs_np` uint64) must stay unwrapped.

## Why the filed PR #2 +51% table is not a cookbook land

117.1 vs 83.7 compares legacy six-patch `2c427ef` (MBT 4096, cache on) to
champion two-patch `f01e24f6` (MBT 8192, cache off). `not_comparable`.

## Same-image 3-arm A/B (n=3, C1 + agentic)

All three arms: same image, same model dir, MTP4, cache off, MBT 8192,
graphs on, configured 230 W, empty-GPU between arms. Prompt tokens are
**actual endpoint counts**, not the cell label.

| Arm | Overlay | Extra env |
|---|---|---|
| A `cookbook-mtp4` | none | none |
| B `research-mtp4-v5` | `patch_gdn_mixed_split_v5.py` | none |
| C `research-mtp4-v5-draftint4` | `patch_champion_stack_overlay.py` (v5 + S + M1) | `B70_DRAFT_LMHEAD_INT4=1,B70_DRAFT_MTP_INT4=1` |

VRAM free after load: A 846 / B 831 / C 748 MiB. C logged the one-time draft
requant: MTP 0.85→0.22 GB, LM head 2.54→0.66 GB.

### C1 screens (n=3, ~p530 / ~p4130, not exact 512/8192)

| Arm | p512/g128 median post-first (t/s) | range | p8192/g1 median input/TTFT (t/s) |
|---|---:|---:|---:|
| A cookbook | **59.101** | 57.649–60.959 | **1747.8** |
| B v5 | **62.737** | 60.691–62.803 | **1746.4** |
| C v5+draft INT4 | **83.944** | 83.782–86.200 | **1757.3** |

C vs A C1 decode: **+42%** on this harness (59.1 → 83.9). Prefill is flat
(±1%). B vs A C1 is +6% n=3 — treat as noise, not a v5 speed keep.

Arm A 59.1 is **below** Run 40’s published cookbook card (83.7 n=5 exact
p512/g128). Arm C 83.9 lands on that card, not on the PR’s 117 t/s cell.
Do **not** write “+42% vs the cookbook 83.7 number.” The matched claim is
“+42% vs this campaign’s cookbook arm on the same chat harness.”

### Agentic (3 sessions × 5 turns, g256, 0 crashes)

| Arm | median all 15 (t/s) | median T2+ (t/s) | median last-turn (t/s) | max turn |
|---|---:|---:|---:|---:|
| A cookbook | **44.33** | 47.46 | 72.16 | 74.76 |
| B v5 | **44.44** | 44.01 | 45.30 | 73.21 |
| C v5+draft INT4 | **59.64** | 60.23 | 95.13 | 96.64 |

C vs A agentic all-turns: **+35%** (44.3 → 59.6). Last-turn 95 t/s is a
**long-prefix peak**, not sustained. B is speed-flat vs A.

### Concurrent mixed probe

Decode-first mix (wait for first decode token, then fire ~4k prefill) on B
and C. A’s first probe fired both as prefills (weaker mix).

| Arm | alive | EngineCore deaths | `GDN mixed split` log lines |
|---|---|---|---:|
| A | 3/3 | 0 | 0 (no overlay) |
| B | 3/3 | 0 | 48 (`prefill=1 decode=0 spec=1 n_ns≈4141 n_sp=5`) |
| C | 3/3 | 0 | 48 (`prefill=1 decode=0 spec=1 n_ns=4140 n_sp=5`) |

v5 is a **correctness** win for mixed batches, not a C1 keep.

## n=5 same-image Lane 1 (exact Run 40 prompts) — 2026-08-18

Raw: `results/2026/08/qwen38-pr-n5-20260818T214749Z/`
Live: `results/_quarantine/qwen38-pr-n5-live.log`
Comparison: `results/2026/08/qwen38-pr-n5-20260818T214749Z/comparison.json`
Timing: client post-first / input÷TTFT. Cache off (`prefix_cache_hits_total`
delta 0 on every C1 cell). C1. Configured **230 W**; measured median W in
the table. Image `f01e24f6`, MTP4, MBT 8192, U=0.88, fp8 KV.

| Arm | p2048/g1 in/TTFT | p8192/g1 in/TTFT | p512/g128 post-first | p8192/g128 | short agentic all-turns | mix alive |
|---|---:|---:|---:|---:|---:|---|
| A cookbook | 1768.8 | 1691.1 | **81.20** (78.76–81.97) | **77.52** | **43.27** | 0/1 (EngineCore dead) |
| B v5 | 1773.1 | 1695.8 | **81.37** (78.73–81.86) | **77.44** | **42.97** | 3/3 |
| C v5+draft INT4 | 1799.5 | 1696.0 | **112.65** (111.58–117.73) | **103.63** (99.10–107.70) | **57.45** | 3/3 |

Accept (`accepted_sum/draft_sum` over n=5): A 510/532 = 95.86% @p512,
507/528 = 96.02% @p8192; B 510/532 = 95.86% / 508/532 = 95.49%; C 510/540
= 94.44% / 507/536 = 94.59%. Measured median W @p512/g128: A 195.2, B 195.2,
C 205.0. Post-load free: A 884 / B 1041 / C 1065 MiB.

C vs A, `matched_exact`, n=5 median:

- p512/g128: **+38.7%** (81.20 → 112.65)
- p8192/g128: **+33.7%** (77.52 → 103.63)
- short agentic (3×5, g256, ~0.9–4.2K growing): **+32.8%** (43.27 → 57.45)
- p8192/g1 cold input: **+0.3%** (flat)

B vs A C1 is speed-flat (+0.2% at p512). v5 is the mixed-batch correctness
keep. Draft INT4 is the C1 + short-agentic speed keep **on this n=5 card**.
This is **not** “+51% vs Run 40 83.7”; the matched cookbook arm here is
81.20, in the Run 40 band.

C one-time draft requant (server.log): MTP 0.85→0.22 GB; LM head 2.54→0.66 GB.

Long-context C1 agentic (`results/2026/08/qwen38-pr-long-20260818T221813Z/`,
g128, 0 crashes, actual first prompts ~8300 / ~16500 / ~32880):

| Arm | 8K median (n=3 sess / 15 turns) | 16K (n=2 / 8) | 32K (n=2 / 6) |
|---|---:|---:|---:|
| A cookbook | 50.06 | 56.29 | 52.14 |
| B v5 | 51.89 | 58.27 | 46.64 |
| C v5+draft INT4 | **68.73 (+37%)** | **70.86 (+26%)** | 54.94 (+5%) |

32K ran with ~245 MiB `visible_avail` — isolated C1 / abort-zone. Do not
headline 32K rates. B 32K last-turn 37.39 is directional only.

## What merged (2026-08-18)

- **PR #1** by popisec — optional Cn overlay. Mixed 3/3 vs cookbook
  EngineCore death. C1 speed-flat.
- **PR #2** by popisec — optional draft INT4 S+M1. Same-image n=5 C1
  **and** short/8K/16K agentic win. Env still defaults off in the
  patches; recipe documents the two env flags. Speed-only.

## Still open

- Token / task-quality / KL parity of INT4 draft vs BF16 draft.
- Independent reproduction (still E2 self-reported).
