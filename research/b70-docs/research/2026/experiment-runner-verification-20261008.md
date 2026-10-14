---
title: "Experiment runner verification — policy enforcement exercised end to end (2026-10-08)"
type: research
date: 2026-10-08
status: verified-workflow
---

# Experiment runner verification

Workflow/tooling verification on CPU fixtures only. **No GPU run, no model
load, no benchmark claim.** Every behavior below was exercised through the
real `tools/research.py` dispatch and the `tools/test_experiment.py` suite
(74 tests, all passing on 2026-10-08).

## 1. What was verified

**Unit suite** (`python3 -m unittest discover -s tools -p 'test_*.py'`):
74 tests pass, covering plan schema, question validation, sealing,
assessment gating, registration, interruption, and cleanup.

**End-to-end CLI lifecycle** (temp fixture root, subprocess dispatch):
`run` → sealed run root containing `plan.json`, `manifest.json`,
`environment.json`, `events.jsonl`, `command.stdout`, `cleanup.stdout`;
`assess` → paired-gain gate evaluation with sha256 evidence hashes,
`validated_screen` / `not_adopted`; `record` → `RL-NNN` ledger row plus
`CAM-*` campaign row written under `writer_lock`.

**Negative path**: with one registered run and a portfolio ceiling of 2,
a second run succeeded and a third run exited 2 with
`Unregistered prior run: assess and record it before starting another` —
the registration guard fires before the attempt-limit check, so an
unassessed run can never be silently stacked.

## 2. Enforced policies

- Operator holds: `program.operator_hold` or `question.operator_hold`
  rejects with `Question is on operator hold`.
- Execution budget: `plan.timeout_s` cannot exceed the question's declared
  `execution_budget.{stage}_timeout_s`.
- Minimum useful gain: `comparison.min_relative_gain` cannot be below the
  question's declared gate (0.10 portfolio default).
- Plan identity: `hypothesis` and `changed_variable` must match the
  program question verbatim.
- Attempt ceiling: enforced from `portfolio.max_attempts_per_mechanism`
  (validated positive int, default 2 — same default `research_queue.py`
  uses). Failed exits and changed commands count; matching is on
  `(question_id, hypothesis, changed_variable)`.
- Prior-run registration: any same-question manifest absent from
  `research-ledger.json` blocks the next `run`.
- Writer serialization: `fcntl` lock at `.runtime/experiment.lock`;
  a second writer fails immediately. Same-host only — Syncthing peers
  still need the one-writer operator rule.
- Lifecycle logs: `results/**/{command,preflight,cleanup}.stdout` are
  un-ignored in `.gitignore`; other `*.stdout` stays ignored.
- Sealed evidence: post-seal modification of measurements/checks is
  detected by sha256 and rejected; missing evidence blocks, never passes.

## 3. Fix applied during verification

`run_plan` originally hard-coded `attempts >= 2` while
`research_queue.py` honored the declared `max_attempts_per_mechanism`.
A program declaring 1 would have been bypassed by a direct `run`.
`check_question` now returns the declared limit; regression test
`test_attempt_ceiling_comes_from_declared_portfolio_limit` covers a
declared limit of 1.

Also restored dropped contract rules (verified against
`archive/research-workflow-20261007/AGENTS.md` §20): XPU graph capture
always on for vLLM serving — capture-off numbers are diagnostic-only,
never champion inputs (decisive historically: draft-INT4 inert ≤1.5%
capture-off vs +46% on); and the prefix-cache policy — OFF on
vLLM-XPU W4A16/speculative stacks until the #53912-class cache-hit-path
bugs are fixed, exl3xpu cache-ON allowed under §9 guards, cache ×
speculation a shared-risk axis. The locked per-route spec stack is
model-specific and correctly lives in the model dossier.

Dated "§20 rules"/"§16" citations in worklogs and results are
historically accurate and left in place; AGENTS.md §2 now records the
renumbering and points to the archived contract. Living-authority
citations repointed: engine READMEs (cache guards §8→§9, oneAPI §3→§6),
skill files (empty-GPU gates §8.1→§4, retired "restore 150 W" →
as-found caps), model doc script-promotion refs §10→§14.

## 4. Limits of this verification

- All runs are CPU fixture commands; execution-path correctness on real
  GPU commands (timeouts, SIGTERM handling under real load) is covered by
  test harness simulation, not hardware.
- The writer lock does not coordinate Syncthing peers; concurrent
  cross-host writers can still diverge — mitigated by contract, not code.
- 501 historical result roots remain unregistered; the new guard only
  blocks *new* unregistered stacking. Historical debt is visible in
  `docs/indexes/research-coverage.md`.
- 123 broken doc references exist, all inside pre-existing
  `*.sync-conflict-*` copies; untouched per the no-mass-repair rule.

## 5. Repository safety state

- Baseline commit `02c61d20` (branch `restructure/2026-09-15`) snapshots
  the post-reset tree; bundle copy at
  `~/agent-work/b70-docs-baseline-20261008.bundle`.
- Syncthing folder versioning is disabled; the desktop replica is off.
  Git is the sole recovery path — commit before any further refactor.

## 6. Next ranked experiment

`python3 tools/research.py next` selects **RQ-14** — Flash-Next route
replacement: Strata SYCL on one B70 vs the llama.cpp dual-split
(`d5ea7133741e67743c0e886bb426c0ce8d69cf6c`, v0.1.40.3).
`execution_authorized: false`; hardware off. Next action is source capture
and static dispatch/allocation review — the pinned `docs/INTEL.md`
snapshot is at `research/references/strata-intel-20261008/` (sha256
`020c512a…467b9`, class: external claim, not validated evidence).
Adoption gate unchanged: ≥1.5× decode and ≥3× prefill vs the RL-070
dual-split baseline plus the quality battery, then fresh n=5.
