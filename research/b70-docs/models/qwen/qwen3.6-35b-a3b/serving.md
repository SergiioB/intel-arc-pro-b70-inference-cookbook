---
status: living
title: "Serving"
type: model-dossier
date: 2026-09-15
---

---
schema_version: 1
model_id: Qwen/Qwen3.6-35B-A3B
model_slug: qwen3.6-35b-a3b
document: serving
as_of: 2026-08-19
status: validated_internal
truth_verdict: PARTIAL
evidence_tier: E2
engine: vllm-xpu
primary_evidence:
  - ../../../archive/legacy-dirs-20260915/benchmarks/vllm-pi-realworld-and-long-context-2026-08-08.md
  - ../../../docs/methods/benchmark-protocol.md
  - ../../../archive/legacy-docs-20260915/benchmark-history.md
---

# Serving

## Proven serving classes

- Isolated/sequential vLLM XPU inference with preserved MTP.
- Prefix-on resident follow-up flow: the recorded 32K follow-up reused 30,464 tokens.
- Exact long-context completion: MTP4 completed p130944/g128, totaling 131,072 tokens; MTP2 also completed the exact boundary.

A server load or reported KV capacity is not the basis for the 128K statement; the completed boundary request is.

## Mixed-load limitation

In the measured mixed invocation, MTP4 failed when long prefill and speculative decode shared a step because of `causal_conv1d`. The no-spec fallback completed all mixed requests, but short-request latency was poor while the long prefill occupied the engine. This means:

- isolated C1 results do not establish mixed-load service quality;
- sequential agentic use does not prove concurrent mixed-step support;
- MTP4 exact-128K capacity does not imply Cn exact-128K capacity.

Use separate workers or scheduling isolation for latency-sensitive short work until the mixed kernel path is validated on the intended image.
