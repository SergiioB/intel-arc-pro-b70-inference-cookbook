---
type: plan
date: 2026-09-20
phase: P6
title: "Long-context and mixed-load correctness gates (OV #4483, draft-drop rule, causal_conv1d)"
status: RESEARCH
opened: 2026-09-20
updated: 2026-09-20
ledger: RL-TBD-P6
depends_on: []
effort: 3–6 sessions
---

# P6 — Long-context and mixed-load correctness gates

## Status log

| Date | Status change | Note / evidence |
|---|---|---|
| 2026-09-20 | filed as PLANNED | Roadmap filed from b70-architect draft |
| 2026-09-21 | PLANNED → RESEARCH — **premise refuted: it IS memory exhaustion** | Classification probe on the pinned OV runtime (openvino 2026.5.0-22967, genai 2026.5.0.0-3412, model `/mnt/<sata-disk>/Qwen3.8-27B-int4-gdn8-ov`, GPU 1, 230 W): full protocol **PASSES at 131072 and 163840** (163840: 111 tokens, TTFT 395 s, 14.34 tps) and **FAILS at 196608** with the recorded `CL_OUT_OF_RESOURCES (-5)`. Per-stage VRAM from the probe's own samples: steady state 28.94 GB used / 5,052 MiB free, and **at the failure 34.20 GB used / 44 MiB free with a 16 MiB largest block** — usage grew during the run. **So this is a real capacity failure, not a misclassified one**, which contradicts the phase's opening framing; the honest next steps are capacity-shaped (KV/precision/buffer sizing or declaring a supported context below 196K), not a kernel-config hunt. The light 40-token probe passes at the same 196606 context ⇒ generation length triggers it. Caveats: the probe's rule returned `UNDETERMINED` because its `log.level` property was rejected by the runtime (`Option not found: LOG_LEVEL`), so *what* grew is unattributed; cooldown timed out at 70–71 °C twice. Worklog `worklogs/P6-20260921.md`. |
| 2026-09-21 | follow-up ladder: threshold is **per generation**, not per token | At 196,608 context with u4 KV: one generation of **40** ✓, **64** ✓ (550.8 s, 12.78 tps), **96** tokens ✓ (553.4 s) — but the probe's full mode (**two** generations of 128) fails, and VRAM held its steady state (28.94 GB / 5,052 MiB free) through the single-generation runs instead of climbing to the 34.20 GB seen at failure. So the trigger is the **number of long generations in one process**, not the length of one — a leak-shaped signature. `P6_REPS=2` at g=64 is the decisive test, running now. Worklog addendum. |
| 2026-09-21 | **classification complete: second-generation growth** | The decisive test (`P6_REPS=2`, 64 tokens, 196608, u4 KV) shows generation #1 **PASS** then generation #2 **FAIL** with the same `CL_OUT_OF_RESOURCES` — while a *single* 96-token generation passes and single 64/40-token runs pass. So the trigger is **the second long generation in a process, independent of length**: a per-generation growth / not-returned-allocation signature in the ≥196 K path, not a capacity ceiling and not a misclassification. This also explains the recorded full-protocol failure, since that protocol is warmup + measured = two generations. Upstream-report shape for #4483 is written in the worklog; the phase's next steps are the report plus the draft-drop rule (the ≥98K draft-drop function remains unmeasured and is now independent of this bug). |
| 2026-09-21 | **decision: verified context envelope declared** | Making the call the phase was waiting for: on the OV INT4-GDN8 u4 route, **131,072 and 163,840 are verified working contexts** (full protocol passed at both on 2026-09-21 with the pinned runtime), and **≥196,608 is blocked by an upstream per-generation growth bug** (W6) rather than by fleet configuration. So the supported-envelope statement is *≤163,840 verified; ≥196,608 blocked upstream*, and the fleet should not spend further GPU time trying to make 196 K work locally. What remains in this phase: the **≥98 K draft-drop function** (reserve vs context, measured) and the two mixed-load items (`causal_conv1d` needs the Qwen3.6-35B artifact, which is not on this host). Upstream report text for #4483 is in the worklog. |

## Why this decision (the reasoning)

**The thesis: three correctness/capacity blockers gate both the public long-context
story and any scheduler tuning — and one of them is probably misclassified, which makes
fixing the classification the cheapest win in the phase.**

1. **Why correctness phases exist at all in a speed roadmap.** The repo's own rule: do
   not tune scheduler parameters around a correctness bug. P7 (scheduler) is gated on
   this phase for the affected routes. Beyond sequencing, these three gates cap real
   user-visible capability: the OV composite curve (MTP5 ≤64K → no-MTP ≥98K) is the
   published long-context narrative, and the ≥196K failure plus the draft-drop cliff
   are its documented open edges; Qwen3.6-35B's mixed-load failure blocks a whole model
   route.
2. **Why "CL_OUT_OF_RESOURCES ≠ VRAM OOM" is the entry question.** OpenCL resource
   failures can be workgroup-size, private-memory, or allocation-limit issues with
   *plenty* of VRAM free — and issue #4483 (openvino.genai, filed from this fleet,
   2026-09-15) sits exactly there. If it is misclassified as OOM, the "fix" everyone
   reaches for (shrink context/buffers) is wrong and the real fix (kernel config /
   allocation shape) never happens. So step one is classification with per-stage
   allocation capture, before any fix attempt.
3. **Why the draft-drop rule must become deterministic.** Today the ≥98K draft drop is
   an operational workaround (u8 -14 crash / 3.1× MTP regression recorded). A rule
   derived from *measured reserve and context* (not a magic constant) survives model/
   config changes and is publishable as recipe guidance — the cookbook needs "when does
   MTP turn off" as a function, not folklore.
4. **Why causal_conv1d is isolated before repair.** The Qwen3.6-35B mixed-load failure
   (long prefill + short decode interleaved) smells like a state-layout assumption
   baked into the operator's dispatch boundaries. The repair discipline is the same as
   P1/P2: identify the isolated failing boundary, fix that, preserve recurrent state
   and sequence isolation. Blanking the operator or restructuring the model would be
   the RL-021 class of mistake at runtime scale.
5. **Why capacity probes are labeled separately from speed claims.** 40-token capacity
   probes reaching 256K are *not* a serving claim; a representative speed number needs
   full final sampling (protocol). The cookbook's composite curve already follows this
   labeling discipline; this phase must not regress it by drawing an uninterrupted
   full-protocol curve through capacity-only points.

**Alternatives considered and rejected.**

- *Skip to P7 scheduler tuning and route around the failures.* Rejected: tuning around
  a bug is banned, and mixed-load scheduling is precisely where the causal_conv1d bug
  bites — the tuning would be optimizing against broken behavior.
- *Retire the OV long-context route instead of fixing #4483.* Rejected: it is the
  single-card long-context champion and the public story; also the fix may be small
  once classified.
- *Push to 262,144 tokens because 256K capacity probes passed.* Rejected: 229K/262K are
  later probes, not implied goals; configured context ≠ usable context.

**What would change this decision.** Upstream landing a fix for #4483 (collapse track 1
to validation on the exact artifact); the Qwen3.6 route being retired (drop track 3);
OV GenAI losing the long-context role to another engine (rescope the boundaries work
to that engine).

## Objective

Close correctness and capacity blockers before scheduler tuning complicates the
execution path. Tracks:

1. OV GenAI failure at ≥196K (issue #4483).
2. Draft drop at ~≥98K (make the rule deterministic).
3. Qwen3.6-35B causal_conv1d mixed-load failure.

Target: supported boundary completion and correct mixed-load execution. **No speed
multiplier is assigned** — this is capability/correctness credit.

## Baseline and sources

- `results/2026/09/qw38-ov-mtp-20260914/` (composite curve, gates)
- `models/qwen/qwen3.6-35b-a3b/serving.md` (mixed-load open gate)
- `registry/research-ledger.json`

Established: OV requires draft removal at long context and has a ≥196K resource
failure; the Qwen3.6 mixed-load route is blocked.

## Engineering work

1. Resolve #4483's exact upstream repository, status, affected build, and reproducer
   before claiming any fix.
2. Separate VRAM exhaustion from workgroup/private-memory/allocation resource
   failures (per-stage allocation + kernel config capture).
3. Test any upstream fix on the exact artifact before porting locally.
4. Specify a deterministic draft-drop rule from measured reserve and context.
5. Identify causal_conv1d mixed prefill/decode state-layout assumptions.
6. Repair only the isolated failing operator/dispatch boundary.
7. Preserve recurrent state and sequence isolation.

## Validation and benchmark plan

Capacity progression (total token boundaries): 16,384; 32,768; 65,536; 98,304;
131,072 — then 196,608 only after earlier gates pass. 229,376 / 262,144 are later
probes, not implied goals.

For g128 capacity cells, prompt = boundary − 128 (p98,176/g128; p130,944/g128;
p196,480/g128). Record actual rendered tokens. Capacity probes are labeled separately;
representative speed claims require full final sampling.

Mixed-load reproducer: one long p8192/g128 request + one short p512/g128 request
introduced during prefill; record arrival timing, achieved concurrency, state
correctness, errors, TTFT, TPOT.

## Risks and stop criteria

- Abort immediately below VRAM safety floors.
- Stop after two identical resource failures.
- Do not repeatedly push the context boundary after a deterministic failure.
- Do not infer configured context = usable context.
- Do not classify CL_OUT_OF_RESOURCES as VRAM OOM without evidence.
- Do not tune scheduler parameters around a correctness bug.

## Deliverables

- `research/2026/long-context-runtime-gates-<YYYYMMDD>.md`
- `results/2026/<MM>/long-context-runtime-gates-<UTC>/`
- Ledger entries (distinct per resolved blocker).

## Cookbook integration

Publish supported context boundaries and explicit draft-switching rules. Preserve
composite-series labels ("MTP below boundary; no-MTP above boundary"). Never draw an
uninterrupted full-protocol curve through capacity-only probes or failed regions.
