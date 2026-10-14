---
title: "ADR-0003: Provisional MTP1 over MTP4 for Ornith-1.5"
status: final
type: adr
date: 2026-08-21
---

---
id: b70-ref-adr-0003-ornith-provisional-mtp1
title: "ADR-0003: Provisional MTP1 over MTP4 for Ornith-1.5"
document_type: reference
status: accepted
version: "1.1"
effective_date: 2026-08-19
last_reviewed: 2026-08-21
authority: docs/methods/benchmark-protocol.md
scope: Speculative-depth selection for the locally converted Ornith-1.5-35B-A3B GPTQ vLLM XPU artifact (MixedCal-v2)
adr:
  number: 3
  decision_status: accepted_research_default
  truth_verdict: PARTIAL
  evidence_tier: E2
  validation_state: n5_confirmed_c1
  publication_state: self_reported_not_independently_reproduced
  correctness_state: coherent_output_smoke_tested
  operational_state: research_default
  supersession_state: n5_confirmation_complete
---

# ADR-0003: Provisional MTP1 over MTP4 for Ornith-1.5

## Context

The current Ornith-1.5 GPTQ artifact has one MTP hidden layer. Its measured draft acceptance decays sharply after the first proposed position, making depth four pay verification overhead for weak later positions.

## Decision

Use MTP1 rather than MTP4 as the research serving default for the locally converted Ornith-1.5-35B-A3B GPTQ artifact (now MixedCal-v2).

The original n=3 screen is retained as historical evidence. The 2026-08-21 n=5 MixedCal-v2 confirmation at 150 W C1, cache off, kept MTP1 as the winner: MixedCal MTP1 p512/g128 **96.43** vs MTP2 **84.16** vs MTP4 **66.27** (MTP4 slower than no-spec ~70.7). Limits remain: C1 only, greedy diagnostic, no BF16 logit/KL, LocalMaxxing not submitted.

## Evidence

- [Ornith adoption report §7](../research/2026/ornith15-35b-a3b-adoption-20260819.md) records the rationale, the depth-dependent acceptance pattern, and the MTP1 recommendation.
- No-spec C1 screen (`../results/2026/08/ornith15-35b-20260819T1630Z/c1-screen.json`) records the C1, cache-off, client-timed diagnostic baseline.
- MTP4 C1 screen (`../results/2026/08/ornith15-35b-20260819T1630Z/c1-screen-mtp4.json`) records the depth-four diagnostic cell.
- MTP1 speculation configuration (`../results/2026/08/ornith15-35b-20260819T1630Z/spec-mtp1.json`) and MTP4 speculation configuration (`../results/2026/08/ornith15-35b-20260819T1630Z/spec-mtp4.json`) identify the compared modes.
- MTP1 acceptance cells (`../results/2026/08/ornith15-35b-20260819T1630Z/mtp1-accept-cells.json`) and MTP4 acceptance cells (`../results/2026/08/ornith15-35b-20260819T1630Z/mtp-accept-cells.json`) retain the acceptance evidence.

## Status dimensions

| Dimension | Status |
|---|---|
| Measurement | n=5 MixedCal-v2 confirmation (n=3 day-0 screen retained) |
| Concurrency | C1 only |
| Cache | Explicitly off |
| Timing | Client SSE TTFT and post-first rate |
| Correctness | Coherent-output smoke only |
| Quality | Not tested against BF16/reference |
| Capacity | Exact 65K / 131K / 262K C1 completions recorded |
| Publication | Self-reported E2 with limits; LMX not submitted |

## Quantization caveat

The conversion report records RTN fallback for a minority of expert modules. That fallback is part of the provisional artifact lineage, not evidence of GPTQ-calibrated parity. See the accompanying [Ornith RTN fallback correction](corrections-and-supersessions-2026-08-19.md).

## Reconsideration triggers

Reopen after task-quality/reference testing, an engine/image change, native int4moe backport, DFlash2 draft, or Cn testing that changes the user metric. The n=5 MixedCal-v2 MTP-depth A/B is complete.

Additional MixedCal-v2 evidence: `results/2026/08/ornith15-mixedcal-v2-summary/summary.json`, `results/2026/08/ornith15-mtpdepth-power-20260821T075858Z/`, cookbook `ornith15-35a3/CLAIMS.md`.
