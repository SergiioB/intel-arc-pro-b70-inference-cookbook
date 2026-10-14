---
status: living
title: "Adoption"
type: model-dossier
date: 2026-09-15
---

---
schema_version: 1
model_id: facebook/Muse-Glimmer-30B
model_slug: muse-glimmer-30b
document: adoption
as_of: 2026-08-19
status: experimental
truth_verdict: PARTIAL
evidence_tier: E2
publication_gate: PROVISIONAL_NOT_FOR_PUBLIC_HEADLINE
primary_evidence:
  - ../../../research/2026/b70-vllm-muse-int4/PLAN.md
  - ../../../results/2026/08/muse-dflash-sweep-manual-20260810T164735/FINAL-TABLE.md
  - ../../../archive/legacy-docs-20260915/benchmark-history.md
---

# Adoption

## Decision

The best characterized route is llama.cpp SYCL with the F16 build, Q4_K_XL target, multimodal projector, and DFlash `n_max=2`. Reacquire and verify artifacts before use because local copies were removed.

The vLLM route remains experimental: it depends on a thin PR #51655 overlay and a local image, and its compressed-tensors artifact is not a stable public serving recipe.

## Completed gates

- llama.cpp text, vision, reasoning-channel, and DFlash smokes.
- Corrected F16-enabled engine card.
- DFlash depth sweep and n=5 confirmation.
- 128K-context load and completed requests through the 32K prompt rung.
- Experimental vLLM text and vision smokes on the overlay.

## Open gates

- Reproducible current artifact acquisition and checksums after local cleanup.
- Public upstream vLLM support and pullable image/overlay recipe.
- Reference and task-quality parity for both quantized routes.
- Full near-boundary 128K request completion; loading a 128K context alone is not enough.
- Vision throughput and broader multimodal quality testing.
