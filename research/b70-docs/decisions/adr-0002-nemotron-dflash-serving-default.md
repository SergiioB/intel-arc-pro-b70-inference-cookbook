---
title: "ADR-0002: DFlash over native MTP for Nemotron vLLM serving"
status: final
type: adr
date: 2026-09-15
---

---
id: b70-ref-adr-0002-nemotron-dflash-serving-default
title: "ADR-0002: DFlash over native MTP for Nemotron vLLM serving"
document_type: reference
status: canonical
version: "1.0"
effective_date: 2026-08-19
last_reviewed: 2026-08-19
authority: docs/methods/benchmark-protocol.md
scope: Speculative-decoding selection for the measured Nemotron 3.5 Lightning GPTQ vLLM XPU path
adr:
  number: 2
  decision_status: accepted
  truth_verdict: CONFIRMED
  evidence_tier: E2
  validation_state: validated_internal
  publication_state: qualified_self_report
  correctness_state: deterministic_replay_smoke_tested
  operational_state: current_serving_choice
  supersession_state: current
---

# ADR-0002: DFlash over native MTP for Nemotron vLLM serving

## Context

The measured Nemotron 3.5 Lightning GPTQ-INT4 G64 target has both a native-MTP research path and a separate DFlash draft path. On the recorded vLLM XPU stack, native MTP produced zero useful acceptance, while DFlash loaded, drafted, accepted tokens, and completed an isolated C1 Lane 1 card.

## Decision

For the exact measured Nemotron vLLM XPU target, use DFlash rather than native MTP as the current speculative-decoding path.

The selected measured configuration is `method=dflash` with seven speculative tokens. This is an engine-, image-, target-artifact-, and draft-artifact-specific decision; it is not a universal statement that DFlash beats MTP on other models.

## Evidence

- [Benchmark history Run 36](../archive/legacy-docs-20260915/benchmark-history.md) records the isolated C1 n=5 campaign and states that DFlash operated at about 52% window acceptance while native MTP remained at 0%.
- Run 36 summary (`../results/2026/08/nemotron-dflash-bf16-n7-n5-20260813T082203Z-1407994/n5-summary.json`) records the measured cells, cache-off state, timing source, configured cap, acceptance, and power telemetry.
- Run 36 manifest (`../results/2026/08/nemotron-dflash-bf16-n7-n5-20260813T082203Z-1407994/manifest.txt`) records the exact image, target, draft, scheduler, and launch configuration.
- Native no-spec comparison summary (`../results/2026/08/nemotron-gptq-graph-n5-20260812T2045Z/n5-summary.json`) is the matched-except-speculation comparison source used by Run 36.

## Limits

- Evidence is self-reported E2 and not independently reproduced.
- Correctness is limited to deterministic replay/coherence smoke scope, not task-quality or BF16 parity.
- Both target and DFlash draft are local conversions in the recorded campaign.
- The decision does not apply to llama.cpp native MTP, Qwen, Ornith, Muse, or future Nemotron artifacts without a matched screen.

## Consequences

- Current Nemotron vLLM launch recipes should not enable native MTP as the default.
- Any public performance wording must retain C1, cache, timing-source, n, configured-cap, and local-conversion qualifiers.
- A fixed native-MTP implementation must be screened against DFlash at matched anchor cells before replacing this decision.

## Reconsideration triggers

Reopen when native MTP obtains nonzero, stable acceptance on a current image; when an official compatible DFlash or target artifact replaces a local conversion; when task-quality parity fails; or when a matched n=5 comparison on the target workload changes the selection.
