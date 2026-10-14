---
status: final
title: "Qwen3.6 Pi exact 128K matched cache/spec campaign"
type: campaign
date: 2026-08-08
---

---
id: CAM-2026-QWEN36-PI-128K-MATCHED
doc_id: DOC-CAM-2026-QWEN36-PI-128K-MATCHED
status: validated_internal
---

# Qwen3.6 Pi exact 128K matched cache/spec campaign

- Model: [Qwen3.6-35B-A3B dossier](../../../models/qwen/qwen3.6-35b-a3b/README.md)
- Immutable raw evidence: `results/2026/08/vllm-pi-128k-cache-spec-matched-20260808T224033Z-101198` (`../../../results/2026/08/vllm-pi-128k-cache-spec-matched-20260808T224033Z-101198/`)
- Evidence tier: E2 self-reported raw evidence
- Status: qualified capacity observation

The retained evidence records the exact long-context Pi boundary campaign on the preserved-MTP GPTQ-INT4 vLLM XPU route. The dossier states that exact 131,072-total-token requests completed on the recorded patched stack.

Limits: a completed boundary request is a capacity observation, not a guarantee for another scheduler, cache state, concurrency level, mixed workload, image, or artifact. Mixed speculative traffic had a separate recorded failure path; do not generalize this campaign into production-safety or aggregate-throughput claims.
