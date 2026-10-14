---
title: "P4 patch-anchor validation on the champion generation (claims-audit fix 3, source half)"
type: plan
date: 2026-09-22
status: final
---

# P4 patch-anchor validation — 2026-09-22

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Claims-audit fix #3 (`worklogs/CLAIMS-AUDIT-20260921.md`,
defect 5) required proving the two draft-INT4 patch anchors are valid on the **exact champion
source generation** they run against — the headers claim anchors verified against `2c427ef`
(vLLM 0.26.1.dev457) while the champion runs use `f01e24f6` (0.27.2rc1.dev77). This closes that gap
for the source-anchor half. The greedy token-parity half is delivered by the verify runs
(`p4-determinism.py` in every `verify-*` dir).

## Method

Throwaway CPU-only container on `vllm/vllm-openai-xpu@sha256:f01e24f6…` (no GPU claim; XPU device
count zero in the check container — the check only reads source), reading
`vllm/model_executor/models/qwen3_5_mtp.py` and testing the two patch anchor strings **byte-exactly**
as the patch scripts define them (`scripts/tmp/patch_champion_stack_overlay.py` `_DRAFT_S.QWMTP_OLD`
and `_DRAFT_M.QWMTP_FORWARD_OLD`).

## Result — PASS

| Check | Result |
|---|---|
| `QWMTP_OLD` (compute_logits anchor) present verbatim in 0.27.2rc1.dev77 | **YES** |
| `QWMTP_FORWARD_OLD` (load_weights anchor) present verbatim | **YES** |
| `import os` already present (injection path) | NO — patch will inject it at `import torch` as designed |
| B70 markers already in tree (pre-patch state clean) | NO — correct |
| `anchor not found` failures in ANY champion run's server.log | **zero** across `p4-champion-ab-*/v5_*/server.log` |
| Runtime patch application proven | **YES** — `[B70] draft MTP INT4: … 0.85 GB BF16 -> 0.22 GB INT4 (ahorro 630 MB/lectura)` and `[B70] draft LM head INT4: cuantizando lm_head fp16 (248320, 5120) -> INT4 g128 sym` in the draftpair arm's server.log |

Observed in-container versions (recorded for provenance): vLLM `0.27.2rc1.dev77+gac7509e2b`,
`vllm-xpu-kernels 0.1.12.3`.

## What this changes in the audit fix list

- Fix 3 source half: **DONE** — the patches are anchored-verified on the generation they execute on;
  the header claim "anchors verified against 2c427ef" is superseded for the champion runs by this
  record (the anchor strings are stable across both generations).
- Fix 3 parity half: **in flight** — the verify arms carry `p4-determinism.py` greedy fixed-nonce
  probes (2 repeats × 3 tasks per arm) whose sha256 rows are cross-arm comparable.
- Fixes 1 (calibrated cells), 2 (fresh n=5 both cells), 4 (energy/temp telemetry) and 5 (cooldown):
  implemented in `scripts/tmp/p4-verify-arm-20260923.sh`; evidence lands in `verify-*` dirs.

Evidence: `results/2026/09/b70-improvement-probes-20260920/patch-anchor-check-20260922T2207Z/anchor-check.txt`
and `anchor-check-exact.txt`.
