---
title: "ADR-0001: q8_0 K and q4_1 V as the llama.cpp fleet KV default"
status: final
type: adr
date: 2026-09-15
---

---
id: b70-ref-adr-0001-llamacpp-kv-fleet-default
title: "ADR-0001: q8_0 K and q4_1 V as the llama.cpp fleet KV default"
document_type: reference
status: canonical
version: "1.0"
effective_date: 2026-08-19
last_reviewed: 2026-08-19
authority: docs/methods/benchmark-protocol.md
scope: Default llama.cpp KV-cache policy for B70 fleet profiles
adr:
  number: 1
  decision_status: accepted
  truth_verdict: CONFIRMED
  evidence_tier: E2
  validation_state: validated_internal
  publication_state: qualified_current_default
  correctness_state: externally_measured_kl_divergence_class
  operational_state: fleet_default
  supersession_state: current
---

# ADR-0001: q8_0 K and q4_1 V as the llama.cpp fleet KV default

## Context

llama.cpp serving needs a KV-cache policy that preserves attention-key precision while reducing VRAM use. K and V do not have equal sensitivity: the retained evidence reports a severe quality cliff for `q4_0` K and a near-lossless measured class for `q8_0` K with `q4_1` V.

## Decision

Use the following as the B70 llama.cpp fleet default:

```text
K cache: q8_0
V cache: q4_1
Flash attention: on
```

This is a llama.cpp KV-cache decision. It does not select model-weight quantization and does not apply to vLLM KV dtype.

## Evidence

- [Benchmark protocol §2](../docs/methods/benchmark-protocol.md) records `q8_0` K + `q4_1` V as the fleet standard and prohibits `q4_0` K.
- [KV-cache quantization analysis](../docs/reference/kv-cache-quantization.md) retains the underlying quality analysis.
- [Current concise KV reference](../docs/reference/kv-cache-reference.md) records the same default and its engine boundary.

## Consequences

- New llama.cpp fleet profiles should use `-ctk q8_0 -ctv q4_1 -fa on` unless a separately evidenced exception is approved.
- `q8_0/q8_0` remains a conservative, higher-VRAM alternative.
- `q5_0` or `q5_1` K with `q4_1` V may be researched as a capacity trade-off.
- `q4_0` K remains prohibited. A model-weight Q4_0 result does not relax this rule.

## Alternatives considered

| Alternative | Disposition |
|---|---|
| `q8_0` K + `q8_0` V | Valid conservative option, but uses more VRAM without improving the reported K-quality class. |
| `q5_0/q5_1` K + `q4_1` V | Acceptable measured trade-off, not the default. |
| `q4_0` K | Rejected because the retained evidence reports a catastrophic quality cliff. |

## Reconsideration triggers

Reopen this ADR if matched quality evidence on the current llama.cpp SYCL stack shows a better K/V pair, if quantized-KV kernel support changes materially, or if the fleet moves away from llama.cpp for the covered profiles.
