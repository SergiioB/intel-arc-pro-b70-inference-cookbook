---
type: plan
date: 2026-09-20
phase: P7
title: "Scheduler + context-aware speculation (chunked prefill, draft-depth economics)"
status: RESEARCH
opened: 2026-09-20
updated: 2026-09-20
ledger: RL-TBD-P7
depends_on: ["P6 for affected routes", "stable kernel paths from P1–P5"]
effort: 3–5 sessions
---

# P7 — Scheduler and context-aware speculation

## Status log

| Date | Status change | Note / evidence |
|---|---|---|
| 2026-09-20 | filed as PLANNED | Roadmap filed from b70-architect draft |
| 2026-09-21 | gate restated by P6's outcome | P7 is gated on P6 so that scheduling is never tuned around a correctness bug. That condition is now **satisfied for the verified envelope**: the ≥196 K failure is an upstream per-generation growth bug (W6) with the envelope declared at ≤163,840, so scheduler work may proceed **within the envelope**; the ≥98 K draft-drop function (O5) remains the one input P7 still wants, because speculation economics at long context depend on it. |
| 2026-09-21 | PLANNED → RESEARCH — the serving lever is measured, not speculative | The concurrency knee measured on the champion stack (worklog P4 §19) **is this phase's subject**: 8 requests per level at C1 p512/g128 gave aggregate 48.45 → **115.70 tok/s** from B=1 to B=4 on the baseline config (**2.39×**) and 61.86 → **140.30** on the draft-pair config (**2.27×**), saturating near 150 at B=8, with every request's output server-reported. So the scheduler/serving lever is **real and large, and it stacks with the model-side lever** — which changes this phase's priority from "tune after kernels" to "the biggest serving number available today". Open within it: where the knee actually turns (B=16/32 with `--max-num-seqs` raised), how the knee interacts with long context (KV grows per sequence) and with the ≥98 K draft-drop function, and the per-request latency budget the deployment must hold. Evidence: `…/p4-champion-ab-p512-n1-…T143215Z|T143819Z/conc-B*.jsonl`. |

## Why this decision (the reasoning)

**The thesis: once kernels are correct and fast, the remaining user-felt gains are in
service behavior — and the cheapest unexplored lever is that speculation economics
change with context, but nobody has made the draft depth a function of context.**

1. **Why scheduler work is this late.** Three prerequisites: correctness (P6 — tuning
   around a bug is banned), stable kernel paths (P1–P5 — otherwise the sweep measures a
   moving baseline), and a declared workload mix. Scheduler tuning is config-space, so
   it is cheap *per experiment* — but only meaningful on a frozen stack.
2. **Why ≥10% on a predeclared service metric.** The fleet's public numbers are C1
   decode; but interactive users feel TTFT, time-to-first-content, and TPOT tails, and
   operators feel Cn aggregate. A predeclared metric with a threshold prevents
   post-hoc metric shopping (the class of error where every sweep "wins" on something).
   The 2026-08-30 lmx/runner overlap incident (28.6 vs 50 tok/s on one server from
   concurrent clients) is the standing reminder that serving measurements are fragile —
   hence "exactly one measurement client for C1" and "never sum per-request rates."
3. **Why context-aware draft depth is the interesting hypothesis.** Draft KV and
   verification cost grow with context while acceptance decays; at some depth the
   drafter costs more than it returns. The OV composite curve already embodies a crude
   version (MTP below boundary, no-MTP above). A *measured* depth-vs-context policy
   (per the P6-approved boundaries) could capture the crossover gains in vLLM routes
   without any new kernel — pure scheduler intelligence. This is the highest
   probability-adjusted value per engineering hour in the phase.
4. **Why chunked prefill budgets are bounded before sweeping.** Chunked prefill trades
   TTFT of long-prompt requests against TPOT of concurrently decoding ones. The bound
   set must come from the declared workload mix, not from a flag sweep; otherwise the
   result is ungeneralizable and the cookbook preset would silently encode one
   workload's assumptions.

**Alternatives considered and rejected.**

- *Skip the phase; C1 is the only metric that matters.* Rejected: mixed-load and Cn are
  real cookbook profiles, and draft-depth economics are free throughput at long context.
- *Tune draft depth inside P4.* Rejected: one variable per gate — P4 changes the draft
  implementation, P7 changes the policy. Coupling them destroys attribution for both.
- *Re-open MTP6/8 depth on the champion.* Closed by RL-026; only a changed workload or
  implementation reopens it (which P4's outcome might — that dependency is explicit).

**What would change this decision.** P1–P5 all withdrawing (the stable-stack baseline
is then today's routes — the phase still runs, just against current winners); the
operator declining mixed-load/Cn cookbook profiles (halve the phase to the
draft-depth-policy question).

## Objective

Optimize service behavior — TTFT/TTFC, TPOT tails, aggregate throughput — after
correctness and major kernel paths stabilize. Not just the best C1 decode sample.

## Baseline and E0 target

Baseline: the stable route produced by earlier phases (record exact identity at entry).
No universal optimal chunked-prefill budget exists in evidence. Target: ≥10%
improvement in a predeclared service metric, with no violation of the other declared
SLOs, correctness, or reserve.

## Engineering work

1. Select the production workload mix and SLOs before sweeping.
2. Bound the candidate prefill-token budgets.
3. Separate cold ingestion from warm-prefix conversation.
4. Measure scheduler interference between long prefill and short decode.
5. Choose draft depth by context and workload economics.
6. Disable/drop draft where its KV or verification cost outweighs acceptance.
7. Keep prefix-cache configuration explicit.
8. Avoid changing kernel build or quantization during this screen.

## Validation and benchmark plan

- Standard C1 five-cell card.
- C2/C4/C8 p2048/g128 homogeneous serving, subject to reserve.
- Mixed load: p8192/g128 ingestion + p512/g128 chat.
- Warm shared-prefix follow-up, separately labeled.
- Long-context draft/no-draft comparisons only at P6-approved boundaries.
- Cn = successful generated tokens ÷ campaign interval; never sum per-request rates.
- Report TTFT, TTFC, TPOT, end-to-end p50/p95, failures, queueing, acceptance.

## Risks and stop criteria

- Stop after two correctness failures or recurring timeouts.
- Reject aggregate wins that violate the interactive p95 SLO.
- Reject C1 claims measured with concurrent clients.
- Stop increasing concurrency before reserve becomes unsafe.
- Do not retune closed MTP-depth cases without a changed workload or implementation.

## Deliverables

- `research/2026/b70-scheduler-context-speculation-<YYYYMMDD>.md`
- `results/2026/<MM>/b70-scheduler-context-speculation-<UTC>/`
- Ledger entry.

## Cookbook integration

Create distinct interactive-C1, mixed-load, and aggregate-serving profiles. Add separate
Cn records and latency tables. A scheduler preset must state its workload assumptions.
