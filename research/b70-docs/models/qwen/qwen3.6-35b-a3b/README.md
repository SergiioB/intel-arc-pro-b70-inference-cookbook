---
status: living
title: "Qwen3.6-35B-A3B"
type: model-dossier
date: 2026-08-08
---

---
schema_version: 1
model_id: Qwen/Qwen3.6-35B-A3B
model_slug: qwen3.6-35b-a3b
document: overview
as_of: 2026-08-19
status: validated_internal
truth_verdict: PARTIAL
evidence_tier: E2
publication_gate: PROVISIONAL_NOT_FOR_PUBLIC_HEADLINE
engines: [vllm-xpu]
primary_evidence:
  - ../../../archive/legacy-dirs-20260915/benchmarks/vllm-pi-realworld-and-long-context-2026-08-08.md
  - ../../../results/2026/08/vllm-mtp4-correctness-20260808T182243Z-157102/no-spec/results.json
  - ../../../archive/legacy-docs-20260915/benchmark-history.md
---

# Qwen3.6-35B-A3B

The active B70 route is a preserved-MTP GPTQ-INT4 target on vLLM XPU, with the model's one-layer MTP adaptation retained in BF16. This campaign established real-world Pi-agent flows, exact long-context completion, and a deterministic correctness suite.

MTP4 completed an exact 131,072-total-token request after the boundary patch. MTP2 also completed the exact boundary. These are capacity observations on the recorded stack, not a guarantee for different scheduler, cache, concurrency, or mixed-load conditions.

A resident 32K follow-up reused 30,464 prefix tokens. However, the measured mixed-load invocation that combined long prefill and speculative decode failed in the `causal_conv1d` path; no-spec completed the mixed requests but imposed poor latency on short work. Keep isolated C1, concurrency, and mixed-load claims separate.

See [adoption](adoption.md), [artifacts](artifacts.md), [serving](serving.md), [quality](quality.md), and [timeline](timeline.md).
