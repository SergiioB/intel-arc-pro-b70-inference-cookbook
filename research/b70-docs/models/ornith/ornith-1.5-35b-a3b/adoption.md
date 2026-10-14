---
status: living
title: "Adoption"
type: model-dossier
date: 2026-09-15
---

---
schema_version: 1
model_id: ornith-ai/Ornith-1.5-35B-A3B
model_slug: ornith-1.5-35b-a3b
document: adoption
as_of: 2026-08-19
status: provisional
truth_verdict: PARTIAL
evidence_tier: E2
publication_gate: SELF_REPORTED_WITH_LIMITS
primary_evidence:
  - ../../../research/2026/ornith15-35b-a3b-adoption-20260819.md
  - ../../../results/2026/08/ornith15-35b-20260819T1630Z/manifest-start.json
  - ../../../results/2026/08/ornith15-35b-20260819T1630Z/server3.log
---

# Adoption

## Decision

`provisional_lane_complete`: the architecture transfer, local quantization, vLLM XPU load, coherent generation, and a small speculation screen completed. Publication and production adoption remain blocked by the n=3 evidence size and missing quality/reference gates.

## Why the parent recipe transferred

The release audit found the same `Qwen3_5MoeForConditionalGeneration` / `qwen3_5_moe` topology used by Qwen3.6-35B-A3B, including the hybrid three-linear-attention/one-full-attention pattern and one MTP layer. Differences recorded in training/config metadata did not change the serving topology.

## Completed gates

1. Official checkpoint revision and tensor layout inspected.
2. BF16 source downloaded across the available mounts with disk offload used during quantization.
3. GPTQ contract verified after conversion.
4. vLLM XPU selected the target INT4 kernels and served the model.
5. No-spec, MTP4, and MTP1 C1 diagnostic cells completed.

## Open gates

- Fresh n=5 Lane 1 confirmation using exact endpoint coordinates.
- BF16 or trusted higher-precision reference comparison.
- Task-quality testing, especially because calibration fallback was material.
- Staged 64K and 128K completion tests.
- Prefix-on agentic and Cn mixed-load testing.
- Reproducible public image and immutable artifact manifest suitable for a public recipe.
