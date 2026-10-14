---
title: "B70 research reset audit"
type: research
status: final
date: 2026-10-08
---

# Research needs fewer open decisions and stricter win accounting

Scope: root entrypoints, all 16 program questions and 124 ledger rows, workflow
implementations, the recent EXL3 postmortem, external Flash-Next assessment,
hardware/diagnostic authorities, and the older campaign fragments. Older raw
campaigns were sampled through named evidence, not all replayed. The B70 was off;
no inference measurements were performed in this audit.

## 1. What prevented useful progress

| Fault | Evidence | Consequence | Reset |
|---|---|---|---|
| Two apparent entry trees | `~/B70-DOCS` had old fragments and no Git metadata; maintained tree is under `projects/` | An agent could rebuild the wrong history | Old folder points to maintained tree; one `research.py next` command |
| Context grew instead of selecting work | Old contract was 36 KB; `situation compile` returned full ledgers/program as starting files without selecting a question | More reading without a deciding experiment | Compact contract and one-question output with evidence IDs |
| Ready did not mean actionable | RQ-04 shared held RQ-02 tooling but remained ready; RQ-05 and RQ-10 overlapped; RQ-08 pursued an estimated 1–1.5% gain | Resources drifted into stopped or low-value work | Three ready tracks, explicit operator holds, passed prerequisites and bounded attempts |
| Validated mixed unlike outcomes | RL-121 is external static assessment; RL-118 is a bookkeeping correction; RL-069 is an operations diagnosis | Ledger progress could look like speed progress | Keep validation scope and adoption separate; no automatic route promotion |
| Premises survived their correction | Postmortem corrects a ~2x draft-head byte overcount and cross-cap isolated/wall comparisons; active method still prescribed withdrawn Mamba fusion | Mathematical certainty exceeded measurement | Explicit provenance, Amdahl/traffic equations and route-specific conclusions |

## 2. Measurement faults remain important history

1. The older HTTP script substitutes word count for missing completion tokens;
   the OpenVINO script substitutes requested tokens and infers units from value
   magnitude. Those rates cannot support a new matched claim without raw proof.
2. Repeated fixed prefill prompts could reuse cache. Later evidence corrected an
   SSE-chunk rate mistaken for token rate (RL-090). Actual tokens and cache
   counters are mandatory; tool output units are never guessed.
3. RL-048 rejected a promising draft-sampling screen on fresh confirmation.
   RL-054 rejected thermally invalid overlapping measurements. Fresh paired
   confirmation and retained failures are load-bearing.
4. RL-123 records a display-GPU over-allocation followed by reset loops. A failed
   allocation cannot be assumed to fail cleanly; live reserve and ownership gates
   precede an autonomous run.
5. `run2.sh` in the old fragment deletes prior JSON results. New runs instead use
   exclusive UTC directories; historical scripts are retained as evidence, not
   advertised as current execution paths.

## 3. Research that is worth doing now

**RQ-14, route and placement.** External Strata results are a reason for a
quality-gated matched screen against the current Flash-Next placement. The old
assessment says both external routes beat the local placement even while noting
incomparable metrics. Treat that sentence as unsupported adoption language.
The source supports a candidate, not a local win. v0.1.40.3 fixes the Intel setup
path; pin its commit, inspect expert streaming and price storage misses first.

**RQ-10, complete-task latency.** Retained-prefix reuse may save more useful time
than another small decode change. Freeze document/tool tasks and expected answers,
compare cache OFF/ON on the same available route, count total task time and energy.
RQ-05 concurrency follows this task baseline instead of starting another panel.

**RQ-11, active-byte economy.** MoE can reduce per-step weight traffic, but resident
bytes, expert misses, routing and drafter costs still matter. Compare correctly
completed tasks at matched quality. A faster model with worse answers is rejected.

The historical operator stop on EXL3 kernel/image-dependent probes remains in
force. No new kernel is funded without a measured removable wall-time term large
enough to pass the task's gate. The below-threshold shortlist axis is parked.

## 4. Repository decisions

Existing program, ledger, model-status and campaign registries remain the only
state authorities. Historical results stay at their paths. New tooling connects
selection, primary-source snapshots, mathematical pricing, bounded subprocess
execution, paired assessment and registration. A source snapshot is external
content; a mathematical price is an assumption-based bound; a screen is a
candidate. None becomes an adopted route automatically.

The previous workflow is retained in
`archive/research-reset-20261008/workflow-before.tar.gz`. The user authorized
Syncthing send-and-receive; the running service accepted the folder-only change
without restart. Desktop delivery remains pending while that machine is off.

## 5. Source evidence

- [Intel B70 datasheet](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2026-03/datasheet-b70-gpu.pdf)
- [Intel Xe architecture guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html)
- [Strata v0.1.40.3](https://github.com/Niko1221/Strata/releases/tag/v0.1.40.3), commit `d5ea7133741e67743c0e886bb426c0ce8d69cf6c`
- `research/2026/b70-architect-postmortem-20260930.md`, including byte and cross-cap corrections
- `registry/research-ledger.json` RL-048, RL-054, RL-090, RL-118..124 and their named evidence
