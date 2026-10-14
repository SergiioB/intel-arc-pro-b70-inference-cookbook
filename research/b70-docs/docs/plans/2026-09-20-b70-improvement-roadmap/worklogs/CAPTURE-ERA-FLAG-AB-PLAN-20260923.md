---
title: "Capture-era flag A/B plan — compile-stacked-on-capture, async scheduling, 150-vs-230 W"
type: plan
date: 2026-09-23
status: planned
---

---
doc_id: WL-CAPTURE-ERA-FLAG-AB
doc_type: plan_note
title: Capture-era flag A/B plan (compile × capture × async scheduling × cap)
lifecycle: active
created: 2026-09-23
updated: 2026-09-23
canonical_for:
  - capture-era-flag-ab-design
---

# Capture-era flag A/B plan

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** DESIGN ONLY — nothing below is measured. This
is the last un-started item on the capture-era flag list (architect plan
`ARCHITECT-next-levers-20260921c.md`, "last" entry) plus research-wave-2's #54785
mitigation question. Every hypothesis gets a fixed pass/fail threshold BEFORE the run.

## Why this now

The champion stack's measured +39.6…+48.6 % draft-INT4 lever is **gated on XPU graph
capture** (capture off = inert ≤1.5 %). That makes the capture-era configuration itself a
first-class variable whose interactions we have never measured: torch.compile stacked on
capture, async scheduling under capture, and the power-cap sensitivity of a
capture-dominated decode. Two of these were listed by the architect as "last, expected
flat"; the third (the #54785 clamp cost) is new tonight and is a **safety** question, not
a speed one.

## The four factors, with fixed gates

All cells: champion stack (digest `f01e24f6…` + overlay + `B70_MTP_BF16_DRAFT=1`),
draft-INT4 pair ON (that is the configuration under test), C1, recommended non-thinking
sampling, calibrated p512/g128 **and** p8192/g128, n=5 per arm after one discarded
warmup, `client_post_first_tps` median as the metric, energy telemetry in `power.json`
per arm. Single variable per comparison; ABBA ordering; `cap_before` restored (never 150 W).

| # | Factor | Comparison | Prediction (fixed before measuring) | Gate | Kill if |
|---|---|---|---|---|---|
| A | **torch.compile stacked on capture** | `VLLM_XPU_ENABLE_XPU_GRAPH=1` ± compile (`VLLM_XPU_USE_COMPILE=1` or the generation's equivalent — read the flag from the image before running) | architect: small or flat, because capture already removes launch overhead and compile's remaining win is fusion the graph may not see | **≥5 %** post-first at either cell = adopt; <2 % at both = record as neutral and stop | numerics change (greedy byte-identity must hold) |
| B | **Async scheduling under capture** | `--async-scheduling` on/off at capture on | research wave 2 row 8: V1 already hides scheduler cost behind GPU work ⇒ **flat** | ≥5 % at either cell = adopt | TTFT regression >10 % (async can move cost onto TTFT) |
| C | **150 vs 230 W with the capture-era stack** | cap 150 W vs 230 W (`power1_cap`), everything else identical | architect #5: decode recorded flat (byte-bound), but that record pre-dates the capture + draft-INT4 configuration whose XMX duty is unknown ⇒ **+0–4 % at 230 W** | **statistically flat** (medians within 3 %) = the recorded H5 neutrality extends to the capture era; ≥5 % = the record needs its qualifier widened | draw > cap (that is a telemetry error) |
| D | **#54785 k=3 clamp cost** (safety first) | `num_speculative_tokens` 4 vs 3, capture on, **greedy determinism test as the primary read-out** | **DECIDED 2026-09-23 (`worklogs/SPEC-ECONOMICS-REANALYSIS-20260923.md`): adopt the clamp.** Zero-GPU reanalysis over our measured α band (0.4873–0.58) puts k=3 at **−0.8 % to +1.4 %** of k=4 — i.e. the clamp is nearly free here (wave 2's −2 to −4 % assumed the high-acceptance tail). This run is now a **confirmation measurement**, not an open decision | determinism holds at k=3 (required) and measured cost stays ≥ −2 % ⇒ the clamp becomes the published default; k=4 stays behind `VLLM_XPU_ALLOW_K4_CAPTURE=1` as opt-in-with-hazard | greedy output diverges at k=3 too (then the hazard is broader and the whole capture recipe must be re-examined) |

## Determinism guard (runs in every cell, 15 s)

Wave 2's "spec-canary": before each measured window, 8 identical greedy requests
(`top_k=1`, fixed nonce, identical prompts) must produce **byte-identical output**. Any
divergence invalidates the cell and is itself a finding (record the sha256 set). This is
cheap enough to make mandatory and it is the guard the campaign currently lacks.

## Ordering and budget

D first (it is a safety question and its result changes the default other cells should
use), then C (0.25 d, and it licenses or forbids the 230 W rider), then B (15 min), then
A (compile is the riskiest build change and should run last against a known-good
baseline). Total ≈ 1.5 GPU-days with teardown/cooldown discipline.

Evidence layout: `results/2026/09/capture-era-flag-ab-<UTC>/` with `manifest.txt`,
per-arm `client.log` / `power.json` / `metrics-{before,after}.txt`, the determinism
`canary.json`, and a `comparison.json` summary. Harness: extend
`scripts/tmp/p4-verify-arm-20260923.sh` (counterbalanced ordering + cooldown gate) with
`FACTOR=A|B|C|D` rather than writing a fifth harness.

## What this does NOT cover

- The verify-pass (5th head) INT4 screen — separate lever (D8), its own divergence probe
  runs first and is already dispatched.
- The grouped-IQ3 prefill kernel — separate program (P-3).
- Concurrency effects — aggregate class, deprioritised by the operator.

> **MEASURED CORRECTION 2026-09-23:** the #54785 k=3 clamp recommendation in this file is **REVERSED** — the clamp costs +5.45 % throughput (k4 75.04 vs k3 71.16 tok/s) and the corruption class is not live (spec-canary PASS at k=3 and k=4). Keep MTP4. Details: `worklogs/SPEC-ECONOMICS-20260923.md` §CORRECTION and `clampk3-20260923T022938Z/`.
