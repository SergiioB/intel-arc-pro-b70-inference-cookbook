---
title: "ADR-0006: Sub-1 GiB post-load VRAM reserve is research-only"
status: final
type: adr
date: 2026-09-15
---

---
id: b70-ref-adr-0006-sub-1gib-vram-reserve
title: "ADR-0006: Sub-1 GiB post-load VRAM reserve is research-only"
document_type: reference
status: canonical
version: "1.0"
effective_date: 2026-08-19
last_reviewed: 2026-08-19
authority: docs/methods/benchmark-protocol.md
scope: Operational interpretation of B70 post-load visible VRAM reserve below 1024 MiB
adr:
  number: 6
  decision_status: accepted
  truth_verdict: CONFIRMED
  evidence_tier: E2
  validation_state: operator_safety_policy
  publication_state: capacity_only_with_qualifier
  correctness_state: not_implied_by_fit
  operational_state: research_only
  supersession_state: current
---

# ADR-0006: Sub-1 GiB post-load VRAM reserve is research-only

## Context

A successful model load is not evidence that scheduler allocations, speculative buffers, mixed-token kernels, concurrency, or long-context completion are safe. The benchmark protocol therefore separates an operator-approved research floor from preferred performance and concurrent-serving reserves.

## Decision

Treat post-load `visible_avail` below 1,024 MiB as research-only:

| Post-load reserve | Decision |
|---:|---|
| `<500 MiB` | Abort. |
| `500–1,023 MiB` | Isolated C1 capacity observation only; no mixed traffic, concurrency, or production claim. |
| `1,024–2,047 MiB` | Staged C1 research allowed with stop conditions. |
| `≥2,048 MiB` | Preferred for C1 performance work. |
| `≥3,072 MiB` | Target for mixed or concurrent serving. |

The phrase “sub-1 GiB fits” must not be translated into “production-safe,” “concurrency-ready,” or “128K supported.”

## Evidence

- [Benchmark protocol §12.2](../docs/methods/benchmark-protocol.md) defines the reserve bands and explicitly calls the 500 MiB floor an operator-approved research boundary.
- [AGENTS.md — Post-load VRAM gates](../AGENTS.md) repeats the operational bands and states that launch alone is not a capacity result.
- The archived [pre-migration benchmark history, Run 40](../archive/docs/pre-migration-20260819/benchmark-history.md) records a 997 MiB post-load reserve as an isolated capacity-bound configuration, not a general concurrency-safety result.

## Consequences

- Any result below 1 GiB reserve must state the observed reserve and `isolated C1 capacity observation only` scope.
- Concurrency and mixed-load campaigns require a new configuration with adequate reserve; they cannot inherit a sub-1 GiB capacity result.
- A context claim still requires a real boundary request to complete.

## Reconsideration triggers

Reopen only after the protocol changes based on reproducible allocator and mixed-load evidence with explicit failure margins. A single successful launch or request is not enough.
