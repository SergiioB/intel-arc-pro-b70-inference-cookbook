---
title: "TaskB Gate0 decision — step-0 ADVANCE downgraded by the sequence-level supplement (2026-09-29)"
type: research
date: 2026-09-29
status: provisional
---

# TaskB Gate0 decision (2026-09-29)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Single-writer close-out of the
margin-requant Gate0 pair; no GPU, no conversion, no model loads, no re-runs.
All numbers below were re-derived in this pass directly from the retained
fp32 logits (`runs/*.logits.npz`), not copied from the reports.

## Decision

**The step-0 census verdict `ADVANCE_AS_SYSTEMATIC_CANDIDATE` is DOWNGRADED.**
Under the supplementary sequence-level metric the frozen gate is **NOT MET**:
**1** designated-pair inversion across **1** family, against the pre-declared
threshold of ≥2 inversions across ≥2 families. Requant remains forbidden; the
margin-aware program gets no conversion authorization from this evidence.

This is a **small-sample non-result, not a refutation** of the requant
possibility: n=10 heldout prompts, single prompt per family cell. It carries
no statistical claim and no systemic-law causal claim — one confirmed heldout
recurrence of the flip class (h03@armc) beyond the canary is a presence-only
signal, which is what the census report's own caveat already conceded.

## Evidence roots (both CPU-only, both registered RL-107)

| Root | Scope | Report |
|---|---|---|
| `results/2026/09/b70-margin-requant-gate0-20260929T204352Z/` | step-0 designated-pair margin census: 10 frozen heldout + 2 canary replays × {bf16, armc, arext} fp32 | `RESULTS.md`, `census.json`, `manifest.json` (sha `5b261d9e…`) |
| `results/2026/09/b70-margin-requant-gate0-seqsupp-20260929T212625Z/` | sequence-level supplement: same 10 heldout + 5 blind same-first-digit controls + 2 replays; both answer strings scored, first-divergent margins, reconstructed greedy paths | `RESULTS.md`, `seqsupp.json`, `manifest.json` (sha `cf01177a…`) |

Siblings: census note `research/2026/b70-margin-requant-gate0-census-20260929.md`
(correction appended there); splice note
`research/2026/b70-margin-requant-splice-design-20260929.md` — **DESIGN ONLY,
not executed**; plan `research/2026/b70-sub4bpw-and-margin-requant-plan-20260929.md`
Task B. An empty sibling root `results/2026/09/b70-margin-requant-gate0-20260929T2112Z/`
(dir created 23:12Z, zero files) is an aborted intermediate — no evidence
inside, not registered.

## Independent re-derivation (this pass)

Recomputed every arm×prompt cell diagonally from `runs/{arm}-fp32.logits.npz`
(log-softmax over the stored fp32 logits; corr minus foil sums):

- **51/51 cells match `seqsupp.json`** (seq margin, step-0 margin,
  first-divergent margin; max |Δ| < 1e-4). Step-0 values also reproduce the
  census `census.json` rows exactly (h03@armc −1.2036, h06@arext −0.2015).
- **Controls c01–c05 behave as designed:** step-0 margin identically 0.0
  (same first digit), all pair signal at j*=1, **zero** inversions, seq
  margins +5.9 … +13.7.
- **Canary replays are outside the holdout** (validation only, not counted):
  r01 reproduces bf16 +0.4017 / armc −2.8794 / arext −2.9730 and both quants
  emit the foil "30"; r02 bf16 +1.228.

## What the sequence view established

| Metric | Count | Notes |
|---|---|---|
| Step-0 evaluable (census def) | 8/10 heldout | h04, h07 non-evaluable (bf16 base itself misses) |
| Pair-evaluable (frozen def: bf16 seq margin>0 AND step-0 argmax = correct first token) | **13/15** (8 heldout + 5 controls) | h04, h07 out |
| Strict-greedy evaluable (bf16 emits the exact expected string) | **11/15** | additionally drops h06 (bf16 emits 27) and c03 (bf16 emits 32) |
| Sequence-level designated-pair inversions | **1** — h03@armc (+0.3302 → −1.2037) | the only bf16-correct → quant-wrong argmax-path flip on a strictly-evaluable row |
| First-divergent inversions | 2 — h03@armc, h06@arext (−0.2015) | h06 does not persist to sequence level (+1.512) and is not evaluable under strict greedy — its bf16 base answers "27", so h06 is **not a candidate fail** there |
| Frozen gate (≥4 evaluable AND ≥2 inversions across ≥2 families) | **NOT MET** | met only under the weaker first-divergent counting, and even that collapses to 1 under strict-greedy evaluability |

Corroborating detail that cuts against a simple harm story: both quant arms
*repair* two positions the bf16 base fails at string level — h07 (bf16 emits
44 → quants emit the correct 54) and c03 (32 → 33). Paired ΔseqMargin over the
13 pair-evaluable rows is flat (armc mean +0.001 / median +0.39; arext mean
−0.12 / median +0.15 — re-derived here). On h06 all three arms emit three
different wrong strings (bf16=27, armc=17, arext=41): the designated 26-vs-40
pair never was the real emission. Low-margin positions are where the
artifacts *and* the base are fragile; the correlated-error-direction theory is
neither confirmed at the systematic bar nor falsified.

## Instrument caveat now on record (seq_logp blocksum bug)

The executed supplement script (`gate0_seq_margin_review.executed.py`, sha256
`cf339d91…` — the hash recorded inside every run jsonl) stored `seq_logp*` row
fields as **off-diagonal block sums** (`logp[:m, ans_ids].sum()`), e.g. h01:corr
bf16 stored −22.8392 where the diagonal is −0.0178. Verified in this pass:
`per_token` logprobs and the saved npz logits were always correct; every margin
in `seqsupp.json`/RESULTS was recomputed **diagonally from the npz** by
`scripts/tmp/gate0_seq_review_analyze.py` (sha `561cc773…`). Raw artifacts are
preserved untouched; the fixed script (`797dd40c…`) and `hashes.txt` carry the
executed-vs-fixed provenance. Padding correctness is empirical, not assumed:
h06 unpadded-vs-padded logits differ by max |Δ| = 0.0 bitwise, and pad-fill
invariance is 0.0 on all three arms.

## Runtime / provenance honesty

Reported runtimes are **measured self-report, not guaranteed reproducible**:
census elapsed 331.6 / 572.3 / 415.2 s (bf16/armc/arext); supplement smoke
116.4 s + batches 296.3 / 543.5 / 400.4 s (~22.7 min inside the 30-min cap).
Arm file mtimes are sequential within each root (22:51–23:08Z census;
23:31–23:54Z supplement); the earlier probe2 resumed run ended ~16:44Z, so no
cross-campaign overlap is evident. Caveat retained: single-executor timing on
a shared host — if own concurrent work resumed in-window, elapsed figures are
contaminated; nothing here is a purity or speed claim. The census and
supplement were separate executor sessions (supplement re-scored the census's
frozen set), not independent worker replicates — do not cite them as
independent reproductions.

## Consequences

1. **TaskB does not advance on this evidence.** The "systematic candidate"
   bar was not met at sequence level; conversion remains forbidden. Any
   future requant proposal needs a fresh gate — not this census.
2. **The splice note is unchanged in status: DESIGN ONLY.** Its P0 splice
   grid (9 window forwards on retained probe2 npz) is a *separate future
   gate*, unexecuted; a positive P0 would license the next design review,
   never a conversion by itself.
3. **Production unchanged.** EXL3 4bpw remains the primary Qwen3.8-27B route;
   W4A16 remains closed under ADR-0010. Nothing in this evidence reopens it —
   the census downgrade removes the cheapest justification path, and the
   surviving single inversion keeps the door technically open only as a
   research question.
4. The heldout set + controls + retained fp32 npz logits are the reusable
   instrument: any future requant candidate re-scores these same rows at
   near-zero marginal cost.
