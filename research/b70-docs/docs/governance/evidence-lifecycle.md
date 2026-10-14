---
title: "B70 Evidence Lifecycle"
status: living
type: governance
date: 2026-09-15
---

---
id: b70-gov-evidence-lifecycle
title: B70 Evidence Lifecycle
document_type: governance
status: canonical
version: "1.0"
effective_date: 2026-08-19
last_reviewed: 2026-08-19
authority: docs/methods/benchmark-protocol.md
scope: Creation, validation, publication, correction, supersession, and retention of B70 evidence
---

# B70 Evidence Lifecycle

## Purpose

This document defines how B70 evidence moves from a planned measurement to a publishable claim and, when necessary, to correction or supersession. It preserves the distinction between what was observed, whether the observation is valid, what the observation supports, and whether the wording is approved for publication.

The [B70 Benchmark Protocol](../methods/benchmark-protocol.md) is normative. Its campaign lanes, hard validity rules, evidence tiers, truth verdicts, and publication gate apply throughout this lifecycle.

## Lifecycle states

| State | Meaning | Permitted use |
|---|---|---|
| `planned` | Scope, lane, coordinates, repetitions, and time budget are declared | Planning only |
| `acquiring` | Warmups and measured observations are in progress | Operational monitoring only |
| `captured` | Raw artifacts and manifest exist; no validity decision yet | Internal inspection |
| `provisional` | Evidence is usable for screening or a narrow working conclusion but has not passed the publication gate | Research and tuning; label `PROVISIONAL — NOT FOR PUBLIC HEADLINE` |
| `validated_internal` | Validity and reconstruction checks passed for the named scope | Internal decisions and qualified documentation |
| `publication_approved` | Exact claim wording and evidence links passed the publication gate | Named public surface and scope |
| `superseded` | Newer evidence or a changed stack replaces the item for current decisions | Historical reference only |
| `invalid` | A methodology defect, contamination, incomplete fixed-length output, or other hard failure makes the observation unusable for the proposed metric | Retain and exclude; never publish as valid |
| `archived` | Retained but no longer actively indexed as current | Historical context |

Evidence tier (`E0`–`E5`) is independent of lifecycle state. For example, an `E2` self-reported campaign can be `publication_approved`, while an `E4` external reproduction can remain provisional if its coordinates do not support the proposed wording.

## Stage requirements

### 1. Plan

Before acquisition, declare:

- the question and result class;
- the smallest protocol lane that answers it;
- engine, build or image, exact model artifact, quantization, context, KV, scheduler, speculation, cache, concurrency, and configured power cap;
- measured cells and actual-token calibration method;
- warmup and repetition policy;
- timing source and statistic;
- correctness scope;
- evidence paths, request count, wall-time estimate, abort conditions, and cleanup.

A missing point remains `not measured`. Do not expand into a Cartesian sweep unless the claim requires the added dimension.

### 2. Acquire

Apply the protocol's empty-GPU preflight, one-engine rule, power restore, same-shape warmup, cache controls, fixed-output checks, telemetry capture, and live-progress requirements. Retain failed, timed-out, early-EOS, contaminated, and excluded observations with reasons.

Raw evidence is append-only. Do not rewrite a sample after acquisition to make it conform to an expected value.

### 3. Capture

A captured campaign has:

- a manifest with provenance and exact configuration;
- raw per-observation output;
- all measured samples and exclusions;
- synchronized cache/spec counters when applicable;
- power, temperature, and VRAM evidence when measured or required;
- calculation inputs for derived statistics; and
- an explicit completion or interruption status.

The run ID and evidence root must remain stable.

### 4. Validate

Validation checks three separate questions:

1. **Measurement validity:** Did the campaign follow the protocol for the named metric?
2. **Correctness scope:** What behavior was actually tested? Use the scopes in the [correctness validation method](../methods/correctness-validation.md).
3. **Claim fit:** Do the observations support the exact proposed wording without changing phase, concurrency, cache state, statistic, or comparison class?

Assign a truth verdict (`CONFIRMED`, `PARTIAL`, `REFUTED`, or `UNCERTAIN`) to the claim, not to the mere existence of a file.

### 5. Compile and present

Derived summaries and visuals must be reproducible from the same numeric source. Preserve raw precision in evidence; round only in presentation. A presentation artifact must keep qualifiers on the same table, chart, card, or post as the number.

The [benchmark history](../../archive/legacy-docs-20260915/benchmark-history.md) indexes campaigns chronologically. It is not a substitute for raw evidence and must not flatten unlike timing sources or C1/Cn metrics into one unlabeled result.

### 6. Approve publication

Publication approval applies to exact wording on a named surface. It requires every gate in [protocol §8](../methods/benchmark-protocol.md), including source preservation, complete coordinates, accurate statistic labels, cache validation, configured-cap versus measured-draw separation, failure accounting, comparison review, and reproducible public prerequisites.

Platform acceptance or an `APPROVED` leaderboard status is `E3` external publication unless the platform independently reruns a comparable or matched protocol. It is not `E4` or `E5` by implication.

## Corrections

When a defect is found:

1. stop repeating the affected claim;
2. mark the observation or claim `invalid` or `superseded` without deleting it;
3. date the correction;
4. identify the old wording and defective assumption;
5. state what remains valid and what is withdrawn;
6. link the old evidence and replacement evidence;
7. recompute dependent summaries from the corrected source; and
8. update every public surface that repeated the claim.

Use the correction format in [protocol §8](../methods/benchmark-protocol.md). Historical tables may remain as recorded when a correction note is adjacent and unambiguous.

## Supersession triggers

Evidence is normally superseded for current recommendations when any material coordinate changes, including:

- engine or kernel build;
- model checkpoint, tokenizer, template, or quantization;
- scheduler, graph/eager, KV, cache, or speculation policy;
- context or concurrency scope;
- measurement definition or timing source; or
- a corrected harness that invalidates the prior method.

Supersession does not mean the historical observation never occurred. It means it no longer controls the current decision.

## Retention and immutability

Retain manifests, raw outputs, monitor logs, summaries, scripts or script identities, and claim audits needed to reconstruct published wording. Do not delete inconvenient valid samples. If a local prerequisite is no longer available, record that availability change separately; do not rewrite the historical manifest.

## Related authorities

- [B70 Benchmark Protocol](../methods/benchmark-protocol.md)
- [Documentation Standard](documentation-standard.md)
- [Metadata and ID Schema](metadata-and-id-schema.md)
- [Correctness Validation](../methods/correctness-validation.md)
- [Observability Standard](../methods/observability-standard.md)
