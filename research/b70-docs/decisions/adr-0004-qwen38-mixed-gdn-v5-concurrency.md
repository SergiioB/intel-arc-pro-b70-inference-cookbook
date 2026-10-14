---
title: "ADR-0004: Mixed GDN split v5 required for Qwen3.8 speculative concurrency"
status: final
type: adr
date: 2026-09-15
---

---
id: b70-ref-adr-0004-qwen38-mixed-gdn-v5-concurrency
title: "ADR-0004: Mixed GDN split v5 required for Qwen3.8 speculative concurrency"
document_type: reference
status: provisional
version: "1.0"
effective_date: 2026-08-19
last_reviewed: 2026-08-19
authority: docs/methods/benchmark-protocol.md
scope: Required compatibility overlay for concurrent mixed prefill, decode, and speculative Qwen3.8 vLLM XPU batches
adr:
  number: 4
  decision_status: provisional_acceptance
  truth_verdict: CONFIRMED
  evidence_tier: E2
  validation_state: mixed_concurrency_correctness_smoke
  publication_state: operational_only_not_speed_claim
  correctness_state: completion_and_health_smoke_tested
  operational_state: required_overlay_for_covered_concurrency
  supersession_state: pending_upstream_fix
---

# ADR-0004: Mixed GDN split v5 required for Qwen3.8 speculative concurrency

## Context

The XPU fused GDN attention path rejects a batch that contains speculative tokens together with ordinary prefill or decode tokens. Sequential C1 work does not necessarily exercise this mixed step, so a C1 speed screen cannot establish concurrent serving compatibility.

## Decision

For the covered Qwen3.8 vLLM XPU MTP serving stack, apply mixed GDN split v5 whenever concurrent traffic can produce a mixed speculative and non-speculative GDN step.

Patch every imported copy of `_xpu_ops.py`; patching only `/workspace/vllm` is insufficient when EngineCore imports the site-packages copy. Homogeneous C1 batches remain on one fused call.

This is a correctness/availability decision, not a speed keep and not a claim of task-quality parity.

## Evidence

- [Mixed-split v5 report](../research/2026/qwen38-gdn-mixed-split-v5-20260818.md) records the failing fused-kernel condition, patch behavior, import-path failure, and successful mixed probe.
- Raw evidence note (`../results/2026/08/qwen38-gdn-mixed-split-v5-20260818/NOTE.txt`) labels the probe provisional and not a Lane 1 card.
- Server log (`../results/2026/08/qwen38-gdn-mixed-split-v5-20260818/server.log`) records the mixed split execution and completed health state.
- [Upstream issue vllm-xpu-kernels #510](https://github.com/vllm-project/vllm-xpu-kernels/issues/510) tracks the kernel limitation.

## Status dimensions

| Dimension | Status |
|---|---|
| Failure without complete patch | Engine death on mixed concurrent step |
| Covered success | Mixed decode plus prefill request completed; health returned 200 |
| C1 speed | Screened as speed-flat; not replacement evidence for the n=5 card |
| Concurrency performance | Not characterized as an aggregate tok/s claim by the v5 probe |
| Token/task parity | Not measured |
| Long context | 64K–128K gather cost not measured |
| Upstream state | Local overlay pending an upstream fix |

## Consequences

- Production-like Qwen3.8 MTP concurrency must not run the unpatched exclusive fused path.
- A sequential agentic test is insufficient as a release gate; include an actual mixed concurrent probe.
- Do not cite the v5 smoke as a throughput gain.

## Reconsideration triggers

Retire this ADR when the pinned vLLM/XPU-kernels image contains an upstream equivalent and the mixed probe passes without the local overlay, or when the serving stack no longer uses speculative GDN batches.
