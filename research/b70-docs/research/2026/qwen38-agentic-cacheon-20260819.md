---
title: "Qwen3.8 prefix-on agentic A/B — 2026-08-19"
type: research
date: 2026-08-19
status: final
---

# Qwen3.8 prefix-on agentic A/B — 2026-08-19

**Status:** E2 self-report. Indexed as **Run 43** in `archive/legacy-docs-20260915/benchmark-history.md`.
S+M1 **wins** the matched prefix-on agentic cells vs cookbook BF16-draft MTP4.
Do **not** headline prefix-on as faster decode than the prior cache-off campaign.

## Question

With prefix caching on, does draft-INT4 S+M1 still beat cookbook MTP4 on
multi-turn coding? Publish only if it does.

## Stack

- Image `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
- `Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`, MTP4, U=0.88, max-len 131072
- Cookbook arm: no overlay. S+M1: `patch_champion_stack_overlay.py` +
  `B70_DRAFT_LMHEAD_INT4=1,B70_DRAFT_MTP_INT4=1`
- `enable_prefix_caching: True` in both server logs
- C1, configured 230 W, restored 150 W
- After load: cookbook **865 MiB**, S+M1 **871 MiB** → isolated C1 (500–1,023)
- Timing: client post-first `(completion-1)/(t_end-first)`
- Empty-card after: **32,078 MiB**, cap 150 W

## Result

Prefix-on is proven: short t5 hits = 1664 tokens/session; long t2+ hits
4992–18304. Unique entropy-first user turns keep short t2–t4 at 0 hits.

| Cell | Cookbook BF16 draft | Draft INT4 S+M1 | Δ |
|---|---:|---:|---:|
| short 3×5 g256, n=15 turns | 43.81 | **58.86** | **+34.4%** |
| 8K start g128, 3 sessions / 15 turns | 48.04 | **66.99** | **+39.4%** |
| 16K start g128, 2 sessions / 8 turns | 54.40 | **65.92** | **+21.2%** |

Comparison: `matched_except_quant`. Short measured_avg_W median 192.0 vs 200.9.
Accept 43.3% vs 40.9% (short). 0 crashes both arms.

Prefix-on vs this overlay's **cache-off** agentic (research note only, not a
cookbook delta): short +2.5%, 8K −2.5%, 16K −7.0%. Prefix cache helps TTFT on
hit turns; it does not make decode faster.

Quality was not remeasured here. Prior KEEP 12/15 vs 12/15 still stands.

## Evidence

`results/2026/08/qwen38-agentic-cacheon-20260819T093858Z/`
Harness: `scripts/tmp/qwen38-agentic-cacheon.sh` + `qwen38-agentic-cacheon.py`.
