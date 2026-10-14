---
status: living
title: "Adoption"
type: model-dossier
date: 2026-08-08
---

---
schema_version: 1
model_id: Qwen/Qwen3.6-35B-A3B
model_slug: qwen3.6-35b-a3b
document: adoption
as_of: 2026-08-19
status: validated_internal
truth_verdict: PARTIAL
evidence_tier: E2
publication_gate: PROVISIONAL_NOT_FOR_PUBLIC_HEADLINE
primary_evidence:
  - ../../../archive/legacy-dirs-20260915/benchmarks/vllm-pi-realworld-and-long-context-2026-08-08.md
  - ../../../docs/methods/benchmark-protocol.md
  - ../../../archive/legacy-docs-20260915/benchmark-history.md
---

# Adoption

## Decision

Adopt for isolated and sequential vLLM XPU use with preserved MTP, subject to the documented stack and boundary patches. Do not claim general mixed-load readiness from the current campaign.

## Completed gates

- GPTQ target with BF16 MTP adaptation loaded and served.
- MTP depth screen and real-world Pi phase testing completed.
- Prefix reuse observed in a resident follow-up flow.
- Deterministic no-spec correctness suite retained.
- Staged long-context ladder reached exact 128K completion under MTP2 and MTP4.

## Open or limited

- Concurrent long-prefill plus speculative-decode robustness on the tested kernel generation.
- Broader Cn latency/throughput characterization under realistic mixes.
- Trusted higher-precision reference parity and scored task-quality evaluation.
- Independent reproduction of E2 measurements.
