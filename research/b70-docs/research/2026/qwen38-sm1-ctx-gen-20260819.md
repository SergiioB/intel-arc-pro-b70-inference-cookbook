---
title: "Qwen3.8 S+M1 generation + isolated C1 128K — 2026-08-19"
type: research
date: 2026-08-19
status: final
---

# Qwen3.8 S+M1 generation + isolated C1 128K — 2026-08-19

**Status:** E2 self-report. Lane 1 S+M1 (112.65 vs 81.20) is the public
overlay card. Run 42 curve/128K is published only as **isolated C1**.
Indexed as **Run 42** in `archive/legacy-docs-20260915/benchmark-history.md`.

## Question

Does draft-INT4 S+M1 (v5 + `B70_DRAFT_LMHEAD_INT4=1` + `B70_DRAFT_MTP_INT4=1`)
complete a generation-length curve and staged long context up to whatever
fits on 32 GB? Target was 128K if the VRAM gate allowed it.

## Stack

- Image `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
- `Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`, MTP4, U=0.88, max-len 131072
- Overlay `scripts/tmp/patch_champion_stack_overlay.py`
- Cache off, C1, `ignore_eos=true`, configured 230 W
- After load: **870 MiB** `visible_avail` → isolated C1 capacity only
- Cleanup: 150 W, empty card 31,919 MiB

## Result

Every new cell is n=5, exact tokens, `finish_reason=length`, zero cache-hit
deltas, zero exclusions. g128 Lane 1 reused from
`results/2026/08/qwen38-pr-n5-20260818T214749Z` (112.65 / 103.63).

Generation: p512 holds ~101–111 tok/s across g32/g256/g512 (one slow g512
sample at 73.2). p8192 decays with output length: 93.92 @ g32 → 75.75 @
g256 → 64.96 @ g512 as acceptance falls 86.5% → 51.6%.

Long context (g128): 94.92 @ 16K → 88.25 @ 32K → 76.72 @ 64K → 67.93 @ 96K
→ **62.52 median / 65.57 max @ p130944**. Acceptance stays 92.6–94.8%.
Cold input/TTFT: 1495 → 1269 → 1000 → 827 → **708**. All five 128K reps
completed 130944+128 tokens. Mid-request free VRAM dipped to tens of MiB.

Do not call 128K preferred-perf or production-safe. Do not mix these client
post-first rates with llama-bench tg.

## Evidence

`results/2026/08/qwen38-sm1-ctx-gen-20260819T082145Z/measure/research-mtp4-v5-draftint4-ctx-gen.json`
