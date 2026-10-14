---
title: "Speculation economics — zero-GPU k-sweep verdict from measured acceptances"
type: worklog
date: 2026-09-23
status: final
---

# Speculation economics — the k-sweep is flat (zero GPU cost)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE** as a throughput claim; this is a *decision record* that
kills a planned campaign. Wave-2 research (`RESEARCH-novel-implementations-wave2-20260923.md` idea 3)
predicted the k-sweep would buy ≤1.5 % and proposed this exact reanalysis as its kill probe. Executed
here on tonight's own measured-window acceptance values — no model load, no GPU time.

## Model

Leviathan et al. speedup(γ) = E[γ] / (γ·c_d + c_v), with E[tokens/round] = (1−α^(γ+1))/(1−α),
draft cost c_d = **1.5 ms/token** and verify cost c_v = **28.8 ms/round** (champion anatomy: INT4
drafter ≈0.9 GB/step, target verify ≈17.5 GB → 28.8 ms @ 608 GB/s — consistent with every published
cell to ±3 %). γ = number of draft tokens (MTP depth k). α = measured-window acceptance.

## Inputs — tonight's own measured-window acceptances (bracketed counter diffs)

| Cell | α | source |
|---|---:|---|
| p512 baseline (BF16 drafter) | 0.5435 | `verify-p512-baseline-…T220345Z` |
| p512 draft-INT4 | 0.5581 | `p4-champion-ab-p512-n5-…T214938Z` |
| p8192 baseline | 0.4873 | `p4-champion-ab-p8192-n5-…T213957Z` |
| p8192 draft-INT4 | 0.5676 | same |
| band corners | 0.45 / 0.70 | run-variable band across 6 A/B runs |

## Verdict — k=4 is flat-optimal; the sweep is dead

Speedup over no-spec (γ=1 = 1.000), at each measured α:

| α | g=2 | g=3 | g=4 | g=5 | g=6 | best | g4 gap to best |
|---:|---:|---:|---:|---:|---:|---|---:|
| 0.5435 | 1.135 | 1.179 | 1.177 | 1.154 | 1.122 | g=3 | +0.13 % |
| 0.5581 | 1.143 | 1.193 | **1.196** | 1.176 | 1.145 | g=4 | 0.00 % |
| 0.4873 | 1.105 | 1.126 | 1.110 | 1.080 | 1.044 | g=3 | +1.40 % |
| 0.5676 | 1.149 | 1.203 | **1.209** | 1.190 | 1.160 | g=4 | 0.00 % |
| 0.45 | 1.086 | 1.094 | 1.072 | 1.038 | 1.001 | g=3 | +2.10 % |
| 0.70 | 1.227 | 1.356 | 1.420 | 1.444 | 1.442 | g=5 | +1.68 % |

Deltas vs the champion g=4: **g=5 is −1.5…−3.1 %** and **g=6 −4.0…−6.6 %** at every measured α.
Only the α=0.70 corner (never observed on this stack) would prefer g=5, by 1.7 %. At the measured
band the sweep is worth **≤ +1.4 %** in the best case and negative in the common case.

**→ The MTP depth k-sweep campaign (D6's remaining γ=5/6 candidates) is KILLED at zero GPU cost.**
This confirms RL-026's "MTP4 optimal" independently with tonight's numbers.

## Where the leverage actually is — acceptance

At g=4, finite difference over ±0.025 α: **+8.3 %…+8.8 % throughput per +0.05 acceptance** at all
four measured cells. [CORRECTED 2026-09-29, register row 3: this range conflates two α definitions —
the counter form (accepted/drafted, yield 1+4α) gives **+6.25 % per +0.05** at α=0.55; the
conditional per-position/geometric form gives **+9.25 %**. Use +6.25 % (counter form) as the
pricing constant, which is also the form the D11 counters measure.] Any work that raises α (better drafter quality, KV precision for the draft,
distilled draft) beats depth work by ~6×. The draft-side INT4 lever moved throughput *without*
raising α (0.5563→0.5079 at p512 in the earlier window), which is consistent: it cut c_d.

## Side product — the #54785 determinism clamp is nearly free

Upstream vLLM #54785 reports capture+MTP4 can corrupt greedy logits at bs=1; the deployed mitigation
clamps k→3 under capture. Time/token cost of g=3 vs g=4, time-adjusted on tonight's α:

| α | time/token g3 vs g4 |
|---:|---:|
| 0.5435 | −0.13 % |
| 0.5581 | +0.23 % |
| 0.4873 | −1.38 % |
| 0.5676 | +0.48 % |

**−1.4 %…+0.5 %** — i.e. free, and *faster* at the two lower-α cells. Wave-2 estimated −2…−4 %;
with tonight's c_d/c_v split the clamp is cheaper still. Recommendation: adopt the k=3 clamp together
with the spec-canary (idea 1) before any further publication; the deterministic-output guarantee is
worth ≈0 % throughput here.

Reproduce: `python3` one-liner in the session log 2026-09-23; formulas above.

## CORRECTION 2026-09-23 — the clamp prediction failed; measured before adopt

The k-sweep verdict above **stands** (k=4 flat-optimal; independently reproduced). But the clamp side-product estimate did not survive measurement: the run priced the clamp at **+5.45 % throughput cost** (k4 75.04 vs k3 71.16 tok/s, n=5/arm, champion stack + draft-INT4 pair, C1 p512/g128 — `results/2026/09/b70-improvement-probes-20260920/clampk3-20260923T022938Z/`), not the −1.4…+0.5 % predicted here (the fixed 1.5 ms draft cost understates the k=3 regime). **And the premise for the clamp is absent: the spec-canary PASSED at both k=3 and k=4** (8 identical greedy outputs each, same hash across depths) — the #54785 corruption class is not live on this stack. The earlier **adoption is therefore REVERSED: keep MTP4**. The canary itself (`scripts/tmp/spec-canary-20260923.py`) is adopted as a standing boot/campaign gate. Remains valid: the depth-flatness result and the acceptance-leverage figure (**+6.25 % per +0.05 α, counter form — corrected 2026-09-29**; the earlier "+8.3…8.8 %" conflated α definitions).
