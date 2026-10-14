---
title: "ADR-0005: GPTQ as the best-proven practical B70 vLLM path, not a universal optimum"
status: final
type: adr
date: 2026-09-15
---

---
id: b70-ref-adr-0005-gptq-practical-vllm-path
title: "ADR-0005: GPTQ as the best-proven practical B70 vLLM path, not a universal optimum"
document_type: reference
status: canonical
version: "1.0"
effective_date: 2026-08-19
last_reviewed: 2026-08-19
authority: docs/methods/benchmark-protocol.md
scope: Selection language for weight formats on B70 vLLM XPU
adr:
  number: 5
  decision_status: accepted
  truth_verdict: CONFIRMED
  evidence_tier: E2
  validation_state: exact_artifact_paths_proven
  publication_state: qualified_recommendation
  correctness_state: varies_by_artifact
  operational_state: preferred_starting_path
  supersession_state: current
---

# ADR-0005: GPTQ as the best-proven practical B70 vLLM path, not a universal optimum

## Context

Multiple 4-bit names describe different algorithms, packings, group sizes, symmetry choices, activation-order layouts, loaders, and kernels. Hardware support for INT4 does not prove that every INT4 checkpoint reaches the same XPU path or preserves the same quality.

## Decision

Describe compatible GPTQ-INT4 as the **best-proven practical starting path** for B70 vLLM XPU today, but never as the universally optimal quantization for every architecture, checkpoint, workload, or quality target.

A format becomes supported or recommended only after the exact artifact passes architecture registration, loader/kernel dispatch, correctness, capacity, and target-workload gates.

## Evidence

- [Quantization format strategy](../research/2026/quantization-format-strategy.md) states that the exact engine, architecture, packing, scale/zero-point layout, and kernel registration control whether a checkpoint reaches the demonstrated fast path.
- [Quantization Evaluation](../docs/methods/quantization-evaluation.md) and the [B70 Quantization Decision Record](../docs/reference/quantization-decision-record.md) separate architecture support, checkpoint representation, kernel dispatch, quality, capacity, and performance, and retain exact-artifact exceptions without turning them into family-wide support.
- [Nemotron DFlash Run 36](../archive/legacy-docs-20260915/benchmark-history.md) is an exact GPTQ target path with raw E2 evidence.
- [Qwen3.8 Run 40](../archive/legacy-docs-20260915/benchmark-history.md) is a separate exact GPTQ dense path with its own image, patches, KV, and scheduler coordinates.
- [Muse adoption plan](../research/2026/b70-vllm-muse-int4/PLAN.md) records an exact compressed-tensors W4A16 path, demonstrating why GPTQ cannot be called the only possible practical path.

## Required wording

Use:

> Compatible GPTQ-INT4 is the best-proven practical B70 vLLM XPU starting path for the exact architectures and checkpoint layouts that have passed load, correctness, and workload gates.

Do not use:

> GPTQ is universally optimal on B70.

## Consequences

- Repository recommendations must name the exact model repository/revision, group size, symmetry, activation ordering or `g_idx` treatment, engine image, and kernel path.
- AWQ, RTN, compressed-tensors, MLX-derived, MXFP4, and NVFP4 artifacts remain independent candidates, not aliases for GPTQ behavior.
- Successful conversion alone is not support, parity, or performance evidence.

## Reconsideration triggers

Reopen if a different exact format demonstrates broader architecture coverage, matched quality, and matched workload performance on a current pullable B70 vLLM XPU stack.
