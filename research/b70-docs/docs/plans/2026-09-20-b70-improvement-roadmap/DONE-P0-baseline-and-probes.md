---
type: plan
date: 2026-09-21
phase: P0
title: "Baseline compilation, publication inventory, and cheap probes"
status: COMPLETED
opened: 2026-09-20
updated: 2026-09-20
ledger: RL-030
depends_on: []
effort: 2–3 sessions
---

# P0 — Baseline compilation, publication inventory, and cheap probes

## Status log

| Date | Status change | Note / evidence |
|---|---|---|
| 2026-09-20 | filed as PLANNED | Roadmap filed from b70-architect draft |
| 2026-09-20 | PLANNED → RESEARCH | Execution start. Ledger id allocated RL-030. Resource window: GPU 1 (32.6 GB free) only — GPU 0 hosts the operator's live vLLM Qwen3.8-MTP4 service (container `b70-vllm-qw38`, `ZE_AFFINITY_MASK=0`, 27.7 GB resident, interactive client attached) and is **not** available for model loads. Dual-card K216 probes deferred to a window where GPU 0 is free; single-card probes proceed on GPU 1. Worklog: `worklogs/P0-20260920.md`. |
| 2026-09-20 | window re-scoped on live evidence | The operator's service was **not** running at execution start (no inference process, no container; GPU 0 = 1016 MiB desktop use, GPU 1 empty). Both cards passed the §4 gate, so GPU 0 became available for the P4 MTP profiling lane and GPU 1 stayed the operator-probe card. Recorded rather than assumed. |
| 2026-09-20 | probes complete; wording corrections recorded | All seven probe-batch rows are resolved (see `research/2026/b70-improvement-probe-baseline-20260920.md` §2-§3). Two recorded claims were **corrected by measurement**: (1) the agent contract's line "IQ3_XXS is omitted from fused SYCL `MUL_MAT_ID` MMVQ" holds for upstream `1692f9e50` but **not** for the pinned working tree, which already carries the device-ID cases in an uncommitted diff (written 2026-08-26, built, never parity-tested or measured until this phase); (2) the same wait that causes the missing sync is the recorded reason SYCL graphs refuse `MUL_MAT_ID`, so the repair is a graph-capture prerequisite. Baseline manifests, the publication backlog and the A/B evidence are in the research note; `scripts/tmp/` gained no permanent entry (probe harnesses stayed in the pinned/clean build trees). |

| 2026-09-20/21 | RESEARCH → COMPLETED | Close-out: all seven probe-batch rows resolved (two of them by measurement that corrected recorded claims); publication backlog and the unscoped-opportunity list with promotion conditions are in the research note §2.4/§3-§4; RL-030 written with outcome `validated`. **No charts, dossiers or catalog records are produced by this phase** — the phase's own cookbook clause is "internal inventory only, no new speed record implied", and no speed target exists to render. Deliverables: `research/2026/b70-improvement-probe-baseline-20260920.md`, `results/2026/09/b70-improvement-probes-20260920/` (operator A/B + summaries + DPAS sweep with `dpas-rate-probe/FINDINGS.md`), worklogs `P0-20260920.md`, `P0-inventory-20260920.md`, `P0-publication-inventory-20260920.md`, `P10-gap-plan-20260920.md`. Gate to the next phase is satisfied: P1 was decided from these measurements (BLOCKED, target withdrawn). |

## Why this decision (the reasoning)

**The thesis: hours of measurement should gate weeks of kernel work, and this repo has two
expensive scars proving it.**

1. **RL-021 (quarantined MTP prototype).** A draft-step fusion prototype was built and
   timed against a computation that *omitted the actual MTP math* — constant-token
   placeholder, no attention/MLP/state/embedding path, argmax substituted for the
   sampler. Its "1.5 → 0.25 ms/draft token" number entered the old roadmap as if it were
   a measurement. It was fiction that survived into planning until the architect audited
   it. Lesson: a target that hasn't survived arithmetic + a trace is a hypothesis, and
   hypotheses don't justify kernels.
2. **The K216 "2.25×" target.** 8.87 → 20 tok/s means saving 62.74 ms of a 112.74 ms
   token. Spread over 84 sync sites that is ~747 μs/site — but the D2H payload is only
   ~2 KB/token (bandwidth-irrelevant), and if exposed sync overhead is a more typical
   ~50 μs/site, killing all of it buys ~4.2 ms ≈ 1.04×. The gap between 1.04× and 2.25×
   must come from launch serialization and tiny-kernel inefficiency *that nobody has
   measured yet*. P0 Session 2 exists to decide, before any build, whether that time is
   actually removable.
3. **The operator's standing mandate is probe-first.** Cheap falsifying probes before any
   multi-hour run. P0 is that mandate applied to the whole roadmap.

**Why the probe batch contains what it contains.** Each probe resolves a question that
changes a later phase's design, at operator-microbench cost:

- *Graph capture vs `torch.compile`* — the FP8 campaign assumed compiled execution; nobody
  verified Level Zero graph **replay**. Submission overhead is only worth attacking if it
  is exposed; P4 (MTP subgraph capture) design depends on the answer.
- *Level Zero immediate command lists* — the decode-launch-overhead lever for llama.cpp
  SYCL; only pays if CPU submission is on the critical path (P1's trace answers that).
- *Weight-layout / coalescing inspection* — packed INT4/IQ3 access patterns determine
  whether P5 repacking is worth anything before anyone writes a kernel.
- *Per-expert-M distribution* — "p2048 is compute-bound" is a global-prompt-length
  inference; routed MoE may still produce tiny per-expert GEMMs. Determines P3/P5 tiling.
- *Dispatch separation (M=1 / small-M / large-M)* — the FP8 small-M kernel proved the
  pattern pays (it set the FP8 records); whether INT4 shares one kernel for all M is
  unknown and cheap to read off.
- *Pure W4A4/W2A2 DPAS microbench* — the rate-class note leaves OPS_PER_CHAN=8 explicitly
  unmeasured ("do not cite a 2× sub-byte figure either way"). One microbench closes the
  open question and updates the note. It does **not** create an A4 serving path (no
  quality-viable A4 activations exist), and the phase must not pretend otherwise.
- *Register/SLM accounting* — distinguishes "kernel is slow" from "kernel spills";
  changes the fix class entirely.

**Why P0 runs before everything, including the highest-multiplier phase.** Expected value:
P(success) × (multiplier−1) ÷ cost. P0's cost is trivial and it *multiplies the success
probability of every later phase* — it is the cheapest per-unit-of-avoided-waste work in
the whole plan. It also produces the baseline manifests without which "matched A/B" in
P1–P5 is not even definable (power cap, sampling, image digest, placement must be pinned
or the comparison is invalid).

**Alternatives considered and rejected.**

- *Start directly at P1 (highest multiplier).* Rejected: risk profile is exactly RL-021's
  — build first, discover the target was wrong later.
- *Skip the W4A4 microbench (irrelevant for serving).* Rejected as a *knowledge* item: it
  is the one unresolved hardware question in the rate-class law; keeping it open invites
  future "sub-byte 2×" folklore. Kept cheap, clearly fenced as hardware research.
- *Fold P0 into each phase's entry probe.* Rejected: manifests and traces share one setup
  window; doing them once is cheaper than eleven times, and the publication inventory
  (OV/Cascadia catalog gap) has no natural phase owner otherwise.

**What would change this decision.** Nothing plausible — P0 is pure downside-protection.
If the operator grants only one session ever, it should be this one.

## Objective

Establish comparable baselines, the publication backlog, and the cheap architectural
probes that gate every later phase.

## Baseline and required outputs

Baseline sources: `registry/model-status.json`, `registry/research-ledger.json`, and the
source table in the README. No speed target. Required outputs:

- One exact baseline manifest per active target (image digest, runtime/kernel versions,
  local patches, model hashes, quant metadata, KV policy, draft config, power semantics).
- Kernel/dispatch attribution sufficient to estimate critical-path savings for P1–P5.
- A bounded list of previously unscoped opportunities with promotion conditions.
- Publication backlog separated into auditable records and incomplete observations.

## Engineering work

1. Resolve actual engine checkout paths and immutable revisions.
2. Record installed identities (above) for K216, Nemotron, Qwen3.8 MTP routes.
3. Inspect source before profiling to identify the exact operations of interest.
4. Run only the smallest approved operator/trace probes (the probe batch):

| Probe | Question | Promotion condition |
|---|---|---|
| Graph capture vs compilation | Does compiled execution actually replay a device command graph? | Exposed submission time is material and capture is supported |
| Level Zero immediate lists | Is CPU queue submission on the critical path? | Matched probe improves latency without changing work |
| Weight-layout inspection | Are packed INT4/IQ3 accesses coalesced and DPAS-tile compatible? | Demonstrated excess transactions or packing overhead |
| Per-expert-M distribution | Does a large prompt still produce many tiny expert GEMMs? | Tile mismatch explains measurable waste |
| Dispatch separation | Are M=1, small-M verification, and large-M paths incorrectly shared? | A shape-specific operator probe wins |
| Pure W4A4/W2A2 DPAS | Does OPS_PER_CHAN=8 change measured throughput? | Validated instruction emission and comparable microbench |
| Register/SLM accounting | Are spills, occupancy loss, or staging buffers the real limit? | Static + trace evidence identifies a correctable resource issue |

## Validation and benchmark plan

- Operator probes: exact captured shapes, deterministic inputs, numerical reference
  comparisons, verified instruction/dispatch identity.
- Trace p512/g128 and p8192/g128 C1 for the relevant targets; add Nemotron p2048/g128.
- One cold p2048/g1 prefill trace per target where needed.
- Instrumented runs are attribution evidence, not final throughput results; use
  uninstrumented controls to quantify profiler perturbation.

## Risks and stop rules

- Stop a probe after two failures or when its maximum removable time is below the
  predeclared usefulness threshold.
- Do not equate `torch.compile` with graph replay.
- Do not assume 32 workgroups imply one resident workgroup per Xe-core.
- Do not change immediate-command-list settings globally.
- Do not reopen known blocked dual-card graph paths without a changed hypothesis.

## Deliverables

- `research/2026/b70-improvement-probe-baseline-<YYYYMMDD>.md`
- `results/2026/<MM>/b70-improvement-probes-<UTC>/`
- Ledger entry (allocate real id), linking every later phase's entry decision to the
  specific probe result that gated it.

## Cookbook integration

Internal inventory of missing metadata and recipe prerequisites only. No new speed
record is implied. Capability records may be proposed only where public evidence
supports them.
