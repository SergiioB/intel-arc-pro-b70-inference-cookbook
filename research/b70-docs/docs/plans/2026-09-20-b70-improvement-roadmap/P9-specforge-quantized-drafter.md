---
type: plan
date: 2026-09-20
phase: P9
title: "SpecForge quantized-target drafter (k≥6, trained on served hidden states)"
status: PLANNED
opened: 2026-09-20
updated: 2026-09-20
ledger: RL-TBD-P9
depends_on: ["P4/P7 stable target", "P8 representation frozen"]
effort: 3–5 capture sessions; 8–20+ training/integration sessions + separately budgeted compute
---

# P9 — SpecForge quantized-target drafter, k≥6

## Status log

| Date | Status change | Note / evidence |
|---|---|---|
| 2026-09-20 | filed as PLANNED | Roadmap filed from b70-architect draft |

## Why this decision (the reasoning)

**The thesis: speculation is the fleet's proven multiplier, the one documented reason
non-native drafters lose here is now understood, and the pipeline to fix that reason is
ground-prepped — but the economics gate is hard and the dependency order is
non-negotiable.**

1. **The evidence chain that motivates the phase.** MTP4 lifted the champion +87%/+55%
   (24.41 → 45.72 at p512; 2026-09-14 addendum) — speculation works on this silicon.
   DFlash2, with every known bug fixed, still closed at exactly half the native head's
   throughput (25.16/25.30 vs 50.78/46.58). Root causes, both quantization-related:
   (a) the drafter was trained against *BF16* target hidden states while the served
   target is quantized — acceptance ~2.0 accepted/step vs ~3.5 for the native MTP head
   (which is BF16 by construction, i.e. quantization-preserving); (b) weight-streaming
   asymmetry — the DFlash drafter streamed a 3.85 GB BF16 5-layer model every cycle, so
   even acceptance parity would not have broken even.
2. **Why training on the *served* target is the whole point.** The closing rule of
   record: a drafter must be trained against (or proven robust to) the exact quantized
   representation it will serve. The SpecForge XPU port is ground-prepped for exactly
   this: repo cloned (`/mnt/<models-disk>/specforge-xpu`, rmacy fork `xpu-training-support`,
   sgl-project#769 lineage), XPU bf16 backward/AdamW smoke passed, 1.36B DSpark
   fwd+bwd+AdamW passed on xpu:0, config schema-validated, training plan written. The
   missing gate is hidden-state capture **from the serving stack** — capture on
   BF16-target states re-creates the numerics mismatch that capped the off-the-shelf
   drafter.
3. **Why the ≥1.5× acceptance entry gate.** Drafter parameters are not free: they add
   streamed bytes every cycle. The DFlash lesson priced at ~2× acceptance needed to
   break even *at parity bytes* — and the practical rule of record demands ≥1.5× the
   native head's acceptance to justify any non-native drafter. At native ≈3.5, the bar
   is ≈5.25 accepted/step on the *same acceptance definition and matched natural-text
   workload* — and even that is necessary-but-not-sufficient: the governing metric is
   ms per emitted output token, not acceptance.
4. **Why k=6 first.** The native MTP head generalized to k=8 for free (56.85 tok/s FP8
   record, accept-len 6.63); the block-trained k=4 drafter collapsed at k≥6 (11–32%).
   A retrained drafter targets k=6 as the first economics-positive depth; going deeper
   only if the economics justify it.
5. **Why this phase is late in the queue.** The capture must come from the *frozen*
   target: P4's subgraph changes and P7's depth policy alter the serving path, and P8
   decides the weight representation. Training against a target that then changes is
   the mismatch mechanism all over again — the phase order is the risk control.

**Alternatives considered and rejected.**

- *Another port of the BF16-trained DFlash artifact.* Explicitly closed; retry trigger
  is a drafter trained on quantized-target states — which is this phase.
- *Native-head depth increases only (MTP6/8 retune).* Depth screen closed at MTP4 for
  the INT4 champion; reopening without a changed implementation is banned (P4/P7 own
  any implementation change).
- *Speculative decoding without draft-quality gates.* Random-token synthetic prompts
  inflate acceptance (sampling policy rule); greedy acceptance is diagnostic only.
  Both are banned as representative evidence here.

**What would change this decision.** P4/P7 delivering most of the available step-time
reduction (the drafter's marginal value shrinks — recompute the economics before
capture); the capture probe failing to match the deployed forward path twice (stop
before training); upstream landing a quantized-target-trained drafter for this
checkpoint (collapse to validation + benchmark).

## Objective

Train a non-native drafter against the exact deployed quantized target, once target
representation and runtime semantics are stable.

## Baseline and E0 target

Historical diagnostic acceptance: ~2.0 (mismatched non-native drafter) vs ~3.5 (native
head) — `results/2026/09/qw38-mtp-depth-20260914/`.

Entry criterion: **≥1.5× native acceptance** (identical definition, matched
natural-text workload; ≈5.25 if native is 3.5) — necessary but not sufficient;
end-to-end draft+verify+bookkeeping per emitted token must beat native MTP:

```
t_output = (t_draft + t_verify + t_bookkeeping) / emitted_output_tokens_per_step
```

## Engineering work

1. Freeze exact target checkpoint, quantization, scales, kernel semantics, and
   hidden-state tap points.
2. Validate XPU capture against the actual serving forward path.
3. Check finite values, positions, sequence boundaries, masks, recurrent state.
4. Hash and version captured datasets.
5. Separate training, validation, and held-out natural-text evaluation.
6. Small training sanity probe before the full job.
7. Train/evaluate k=6 first; go deeper only if economics justify.
8. Measure drafter parameter bytes, KV/state, workspace, verification cost.
9. Keep a native-MTP fallback.

## Validation and benchmark plan

- Capture replay parity before training.
- Held-out acceptance under recommended sampling and natural prompts.
- p512/g128, p8192/g128, p2048/g1, p8192/g1; one representative task workload.
- Long-context tests only at P6-approved boundaries.
- n=3 screen, fresh n=5 finals.
- Acceptance, throughput, latency, quality, reserve reported separately.

## Risks and stop criteria

- Stop before training if capture does not match the deployed target.
- Stop after two capture/finite-value failures.
- Stop if held-out acceptance cannot reach the economic gate.
- Stop if extra drafter bytes violate reserve or reduce useful context.
- Reject random-token or greedy acceptance as representative evidence.
- Reject a higher acceptance figure that loses end-to-end latency.

## Deliverables

- `research/2026/specforge-quantized-target-drafter-<YYYYMMDD>.md`
- `results/2026/<MM>/specforge-quantized-target-drafter-<UTC>/`
- Ledger entry; capture schema, dataset provenance, training config, checkpoint
  identity, matched economics.

## Cookbook integration

A drafter recipe becomes possible only with an exact target–draft compatibility
contract. Never market the drafter as interchangeable across GPTQ, FP8, BF16, or newly
rotated targets without separate evidence.
