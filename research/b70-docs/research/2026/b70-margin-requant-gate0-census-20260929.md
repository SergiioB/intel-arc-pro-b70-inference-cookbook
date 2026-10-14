---
title: "Gate0 heldout margin census — the W4A16 low-margin flip generalizes (2026-09-29)"
type: research
date: 2026-09-29
status: provisional
---

# Gate0 heldout margin census — the W4A16 low-margin flip generalizes (2026-09-29)

Status: PROVISIONAL — NOT FOR PUBLIC HEADLINE.
**SUPERSEDED VERDICT (2026-09-29):** the ADVANCE below did not survive the
sequence-level supplement — see the dated correction appended at the end of
this file and `b70-margin-requant-gate0-decision-20260929.md`.
Evidence root: `results/2026/09/b70-margin-requant-gate0-20260929T204352Z/`
(manifest + census.json + per-arm jsonl/npz/logs).
Spec: `b70-sub4bpw-and-margin-requant-plan-20260929.md` Task B; trigger satisfied by
RL-092 = `weights` (`results/2026/09/b70-probe2-30defect-20260929/RESULTS.md`).

## What was run

TaskB Gate0 executed as a bounded CPU census: 10 frozen NEW exact-answer prompts
(disjoint from the canary, the 6-case battery, and the E1 corpus), teacher-forced
step-0 correct/foil logit margins, one batched right-padded fp32 forward per
artifact (BF16 source, armc auto_gptq g128, arext compressed-tensors g128).
Right-pad batching is exact under causality and was validated in-run: the canary
replay reproduces probe2's single-sequence margins to 4 decimals
(+0.4017 / −2.8794 / −2.9730). Total wall ~22 min, 8 threads, ≤26 GB RSS, zero
GPU/container/download/conversion.

## Findings

1. **8/10 heldout positions evaluable** (bf16 margin>0 and argmax=correct).
   h04 (`sum(i*i for i in range(2,7))`=90) and h07 (workers-halving=54) fail on the
   BF16 base itself — non-evaluable, not evidence either way.
2. **2 designated-pair inversions on 2 families, 2 quantizers:** h03
   (len(range(7,63,7))=8; armc +0.33→−1.20 → emits 9) and h06 (17%5+3*8=26; arext
   +0.33→−0.20 → emits the designated foil 40). A third argmax flip on h06@armc
   ('2'→'1', designated margin +0.37) corroborates. ⇒ The 30-defect class is
   **systematic, not canary-specific** — the pre-declared advance criterion is met.
3. **The effect is margin-localized:** both inversions occur on the two
   lowest-margin evaluable rows (+0.330, +0.334); nothing at bf16 margin >~0.9
   flips, and paired Δmargin concentrates at the fragile positions (armc median
   −0.56, arext −0.29 across evaluable rows). High-margin rows (+5…+12) are
   untouched — consistent with correlated-error direction theory (B3), and it
   identifies the target set for a 1/m-weighted margin objective.
4. **Same mechanism signature:** logit-lens shows both heldout inversions are
   established in the L59–L62 terminal swing, matching the canary's L60–L63
   contested branch.

## Consequence for TaskB

Gate0 = **ADVANCE AS SYSTEMATIC CANDIDATE** (cheap signal, explicitly not
statistical proof — 2/16 arm-position cells). This justifies reviewing TaskB for
the next step, which remains forbidden until reviewed: a margin-aware requant
(rotation-fold + 1/m-weighted AutoRound objective) against the B5 ladder
(battery 6/6 kill-gate → replay 3/3 → Δlogprob budget → ≥9/10 tasks). The census
also hands the requant a concrete evaluability instrument: the same heldout set
serves as the post-conversion regression battery, and its evaluable rows are the
positions the objective must protect.

Practical reuse notes for the next stage: `scripts/tmp/gate0_margin_census.py`
generalizes probe2's oracle to batched margin reads (~140–410 s per artifact for
12 prompts — roughly one weight-stream per arm); per-layer hidden states are
retained in `runs/*.hidden.npz` for follow-up first-divergent-layer work.

## Limits

- Single prompt per family cell; margins at one decision position per prompt.
- Designated foils were chosen a priori; h06@armc shows the argmax can flip to a
  non-designated token — the designated-pair metric understates flip incidence.
- No sampling, no generation past step 0, no speed or capacity claims.
- h04/h07 show the BF16 base itself has near-dual failures in the same family —
  worth a note if the battery is later extended.

## Correction appended 2026-09-29 (sequence-level supplement)

The ADVANCE verdict above is **downgraded**: the independent sequence-level
supplement (`results/2026/09/b70-margin-requant-gate0-seqsupp-20260929T212625Z/`,
scoring both complete answer strings per prompt on the same frozen set plus 5
same-first-digit controls) found the frozen gate **NOT MET** at sequence level —
1 designated-pair inversion across 1 family (h03@armc) vs the required ≥2/≥2.
The second step-0 inversion counted here (h06@arext) does not persist at
sequence level (+1.512) and its evaluability is undermined — the bf16 base
itself emits the wrong string (27) on h06, so under strict-greedy evaluability
h06 is not a candidate fail. What survives: one confirmed heldout recurrence
of the flip class beyond the canary — a presence-only signal at n=10, not the
systematic-candidate bar; neither a proof of the correlated-error law nor a
refutation of margin-aware requant. Requant remains forbidden. Full synthesis:
`b70-margin-requant-gate0-decision-20260929.md`; ledger RL-107.
