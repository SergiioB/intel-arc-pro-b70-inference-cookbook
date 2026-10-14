---
title: "Speculation-depth economics reanalysis (zero-GPU) — the k=3 safety clamp is nearly free"
type: worklog
date: 2026-09-23
status: final
---

---
doc_id: WL-SPEC-ECONOMICS-REANALYSIS
doc_type: worklog
title: Speculation-depth economics reanalysis — k vs acceptance
lifecycle: active
created: 2026-09-23
updated: 2026-09-23
canonical_for:
  - speculation-depth-economics
  - mtp-k4-vs-k3-clamp-cost
---

# Speculation-depth economics reanalysis (zero-GPU)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** No GPU time; this is arithmetic over the
measured acceptance band. It decides two open questions cheaply and retires one lever.

## Model and its inputs

Standard speculative-decoding emission model (Leviathan et al.; the same arithmetic
research wave 2 used): with draft depth `k` and per-position acceptance `a`, expected
tokens emitted per target round is `E[k,a] = Σ_{i=0..k} a^i`. Round cost from the wave-2
champion anatomy is `T(k) = 28.8 ms verify + 1.5·k ms` (1.5 ms per draft pass = the
INT4 draft read ~0.66 GB at ~440 GB/s effective including its GEMM). Throughput is
`E/T`.

The acceptance values are **measured** (measured-window counter deltas, 2026-09-22
publication-grade runs and the earlier runs): baseline 0.4873 / 0.53 / 0.5435; patched
0.5581 / 0.5676, with the same-host band running 0.55–0.70.

## Result 1 — depth tuning is nearly flat: the k-sweep is retired

| α | k=3 | k=4 | k=5 | k=6 | k=5 vs k=4 | k=3 vs k=4 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.4873 | 55.27 | **54.51** | 53.01 | 51.26 | −2.74 % | +1.40 % |
| 0.5300 | 58.85 | **58.58** | 57.31 | 55.63 | −2.17 % | +0.46 % |
| 0.5581 | 61.36 | **61.51** | 60.46 | 58.86 | −1.71 % | −0.23 % |
| 0.5676 | 62.24 | **62.54** | 61.58 | 60.02 | −1.54 % | −0.48 % |
| 0.5800 | 63.41 | **63.93** | 63.09 | 61.60 | −1.30 % | −0.81 % |
| 0.6500 | 70.48 | 72.58 | **72.77** | 71.88 | +0.27 % | −2.88 % |
| 0.7000 | 76.07 | 79.69 | **81.02** | 80.92 | +1.68 % | −4.54 % |

(tok/s, `T(k) = 28.8 + 1.5k` ms.) **Going deeper than 4 (k=5/6) buys ≤1.7 % and is
negative across our whole measured band.** MTP4 is at or near the optimum and a depth
sweep is not a lever — this independently confirms research-wave-2's finding and retires
D6/"k∈{5,6} screens" for this stack. Sensitivity check: at cheaper draft passes
(0.8–1.1 ms, plausible for the quantized drafter) k=5 is still −0.1 to −0.9 % vs k=4.
The only regime where k>4 wins is α ≥ 0.65 with expensive drafts — not our cell.

## Result 2 — the #54785 k=3 clamp is nearly FREE at our acceptance (decides capture-era factor D)

Same table, k=3 column: within **−0.8 % to +1.4 %** of k=4 across α = 0.4873–0.58
(our measured band), and only −2.9 % / −4.5 % at the optimistic α = 0.65 / 0.70 tail.
Sensitivity: k=3 is −1.9 % to +1.6 % of k=4 as the draft-pass cost varies 0.8–2.2 ms.

**Decision (2026-09-23, executing agent, per "decide rather than defer"):** adopt the
upstream **k=3 clamp as the published default** for the capture-era champion recipe.
It removes the #54785 silent-corruption hazard at `temperature=0` for a predicted cost of
≈0 % at our measured acceptance. k=4 remains available behind
`VLLM_XPU_ALLOW_K4_CAPTURE=1` as an explicitly-labeled opt-in with the hazard disclosed.
This replaces the "weigh it" framing in `CAPTURE-ERA-FLAG-AB-PLAN-20260923.md` factor D:
the plan's factor-D run is now a **confirmation measurement** (is the clamp really ≈0 %
here, and does determinism hold at k=3), not an open decision. The plan's −2 to −4 %
prediction from wave 2 is updated to **−0 to −1.5 % at our α** (wave 2's Leviathan
arithmetic assumed the high-acceptance tail).

## Result 3 — acceptance is the steep lever: it re-ranks the roadmap

| α change | k=4 throughput | Δ |
|---|---|---:|
| 0.4873 → 0.53 | 54.51 → 58.58 | **+7.5 %** |
| 0.55 → 0.60 | 60.64 → 66.25 | **+9.3 %** |
| 0.60 → 0.65 | 66.25 → 72.58 | **+9.5 %** |

**≈ +9.3 % decode per +0.05 acceptance**, versus ≤1.7 % for the entire depth axis. So
drafter-quality work ranks strictly above depth tuning. Practically: the verify-pass
(5th head) INT4 divergence screen is now the correct next decode experiment (it is a
*quality* gate on the verify path, and a pass would stack +6–8 % on top), and
DFlash2-class drafter improvements are the structural bet worth revisiting — while
"k∈{5,6}" and any draft-depth retune are dead as levers.

## Relation to existing records

- D6 (MTP depth beyond 4, recorded LOSS) is now **arithmetically closed**, not just
  policy-closed: its "retune without a changed implementation is banned" rule stands, and
  this shows the retune would not have paid anyway.
- Capture-era factor D (`CAPTURE-ERA-FLAG-AB-PLAN-20260923.md`) decision updated above.
- D8 (verify-pass head INT4) is unchanged as a lever and its divergence screen is
  dispatched; this note only confirms its relative rank.

Script: the arithmetic is 20 lines and reproducible from this table; the sensitivity
sweeps over draft-pass cost 0.8–2.2 ms and α over the measured band.

## REVERSED 2026-09-23 — the clamp is NOT adopted (measured before adopt)

This file's depth-flatness result stands, but its **clamp ADOPTED** clause is reversed by measurement: the clamp costs **+5.45 % throughput** (k4 75.04 vs k3 71.16 tok/s, n=5/arm, `clampk3-20260923T022938Z/`) and the #54785 corruption class is **not live** here (spec-canary PASS at both depths). Keep MTP4; superseded clamp clause → RL-044. `VLLM_XPU_ALLOW_K4_CAPTURE` therefore has no safety counterargument on this stack.
