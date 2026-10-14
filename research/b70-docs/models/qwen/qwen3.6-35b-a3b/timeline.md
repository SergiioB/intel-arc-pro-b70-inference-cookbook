---
status: living
title: "Timeline"
type: model-dossier
date: 2026-08-08
---

---
schema_version: 1
model_id: Qwen/Qwen3.6-35B-A3B
model_slug: qwen3.6-35b-a3b
document: timeline
as_of: 2026-08-19
status: validated_internal
truth_verdict: CONFIRMED
evidence_tier: E2
primary_evidence:
  - ../../../archive/legacy-dirs-20260915/benchmarks/vllm-pi-realworld-and-long-context-2026-08-08.md
  - ../../../archive/legacy-docs-20260915/benchmark-history.md
---

# Timeline

| Date | Event |
|---|---|
| 2026-08-08 | Preserved-MTP GPTQ vLLM XPU stack characterized for Pi real-world and long-context work |
| 2026-08-08 | Deterministic no-spec correctness suite retained across ten prompt classes |
| 2026-08-08 | Resident 32K follow-up observed 30,464 reused prefix tokens |
| 2026-08-08 | MTP2 and MTP4 completed the exact 131,072-total-token boundary; MTP4 used p130944/g128 |
| 2026-08-08 | Mixed long-prefill/speculative-decode invocation exposed `causal_conv1d` failure; no-spec fallback completed with poor short-request latency |
| Open | Reference/task parity, independent reproduction, and robust realistic Cn mixed-load card |
