---
title: "Cookbook Pain-Point Analysis — 2026-09-14"
type: research
date: 2026-09-14
status: final
---

# Cookbook Pain-Point Analysis — 2026-09-14

Scope: the current serving cookbook (`docs/operations/recommended-configs.md`), the model
dossiers, the catalog/publication flow (`docs/governance/public-benchmark-publication.md`,
XeCores renderer), and how numbers travel. Inputs: this night's campaign work, two
b70-architect reviews, the 2026-09-13/14 claims audits. Verdicts are confirm/deny with
corrections; the architect's corrections are incorporated.

## P1 — Mixed statistic surfaces force re-derivation (CONFIRMED, highest cost)

One cookbook row can mix C1 decode, Cn aggregate, engine-native tg, client post-first,
cold input, and greedy diagnostics — six qualifier sentences were needed in the last X
draft to make one thread honest. Every reader (including future us) re-derives
comparability from prose.

**Fix:** typed columns/views — engine decode | client decode | cold input | Cn aggregate
as distinct surfaces, with sampling and cache badges that cannot be dropped. A
**comparability key** (workload, concurrency, timing definition, cache state, sampling,
statistic, context, power cap) travels with every row; the renderer **rejects or visibly
qualifies** speedup claims across differing keys.

## P2 — Numbers lose identity while traveling (CONFIRMED, one correction)

"102.61 tok/s" requires three documents to learn its cell. Correction from the architect:
the model-status record DOES carry the identity (recommended non-thinking Run48 n=5,
TP2, 230 W, prefix-on) — "greedy-era" is wrong — but the cookbook row itself under-specifies
it, and identity does not survive copying.

**Fix:** every published number carries a one-line identity summary plus an immutable
evidence link (results tree/commit). Copy-paste of the number without the identity line is
a documentation bug, treated like a broken link.

## P3 — Known traps have no user-facing surface (CONFIRMED, scope it)

The traps that cost us nights are real and scattered: `--enforce-eager` = 2.36× decode
penalty (measured 2026-09-13); the DFlash2 draft path cannot torch.compile on the pinned
image (recalled-not-retained — its log was overwritten; recorded in RL-027 blocked
patterns) AND in eager mode still requires `--max-num-seqs 1` (causal_conv1d spec-mix
crash at seqs 8 — retained evidence, RL-015 family) plus `--workdir /` or the registry
resolves the wrong vllm tree; llama.cpp perplexity is non-monotonic on hybrid-GDN
(RL-025, upstream issue filed); q4_0 K cache is prohibited; XPU rms_norm stacked-weight
bug for DFlash/DSpark context path (upstream #56431). They live in
AGENTS.md/ledger/research notes, not where a cookbook user looks.

**Fix (scoped, not a wall of shame):** a "known traps" block attached to the exact
recipe/build/format it applies to, each with resolution status (open / fixed-in-X /
engine-wide rule). Old build-specific failures must not become permanent engine-wide
claims; permanent safety rules (q4_0 K) are labeled as rules, not traps.

## P4 — No use-case decision view (CONFIRMED)

The ecosystem question is "cheapest agent backend / fastest C1 / longest context /
concurrent fleet," but the catalog is per-model. Users must synthesize.

**Fix:** derived views from validated catalog fields only: C1 latency view, concurrent
service view, context-capacity view, reproducibility view (what is independently
reproduced vs self-reported). "Cheapest" requires explicit energy/cost assumptions to be
declared, never implied from tok/s.

## P5 — Baselines re-measured every night (PARTLY CONFIRMED)

Harness drift between nights makes small cross-era deltas ambiguous (45.72 @150 W on
09-13 vs 49.08 [partially PP2-era/mislabeled manifest] vs 50.78 @230 W on 09-14 across
stack eras — the 09-14 anchors also differ in cap, so that trio fails the comparison
gate in every pairing). The architect's correction stands: frozen anchors are
necessary but not sufficient — **when build, card, context, cap, or host conditions
change, a cheap fresh control is mandatory**, and a cross-era delta must never be
narrated as an optimization without a bridge measurement. Power cap is part of the
anchor identity (tonight's audit caught a 230 W MTP arm labeled 150 W — the monitor
CSV is the authority).

**Fix:** keep frozen anchor cells (artifact, prompts, harness, metric definitions) AND a
standard 3-minute fresh-control cell in every campaign manifest; report new results only
against the same-session control, with cross-era rows marked `directional_only`.

## P6 — "Provisional" negatives keep consuming GPU nights (NEW, from the DFlash2 loop)

RL-005 recorded a completed no-go but its label said "provisional," which read as
"maybe retry later" — it took a fresh audit to learn the retry condition. Outcome,
evidence maturity, and retry prerequisite are three different fields.

**Fix (already applied to new entries):** ledger entries separate
`outcome` (validated/failed/blocked) from `publication_state` from `retry_condition`
(named mechanism, not "when upstream improves"). A completed negative does not reopen
because its publication status is provisional.

## What is already good (do not break)

- The ledger's blocked-pattern registry genuinely prevents repeat attempts (it stopped
  the llama.cpp lane re-descents tonight before the operator redirect did).
- The claims-audit culture (canonical-set corrections, burst-semantics notes) is the
  reason the last two X drafts needed no post-cleanup.
- Numbers-travel-up-the-tower is the right law; the fix is identity transport, not a new
  pipeline.

## Priority order

1. P1 comparability keys (blocks honest public comparison; renderer enforcement).
2. P6 ledger field split (cheap; prevents wasted nights).
3. P3 scoped traps (cheap; highest user value per line).
4. P2 identity lines on published rows (cheap; required by P1 enforcement).
5. P5 fresh-control rule (cheap; encode in protocol §1.9 manifests).
6. P4 use-case views (expensive; only after 1–2 land so the views have clean fields).
