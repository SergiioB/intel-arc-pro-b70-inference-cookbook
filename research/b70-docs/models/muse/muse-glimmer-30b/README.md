---
status: living
title: "Muse Glimmer 30B"
type: model-dossier
date: 2026-09-15
---

---
schema_version: 1
model_id: facebook/Muse-Glimmer-30B
model_slug: muse-glimmer-30b
document: overview
as_of: 2026-08-19
status: experimental
truth_verdict: PARTIAL
evidence_tier: E2
publication_gate: PROVISIONAL_NOT_FOR_PUBLIC_HEADLINE
engines: [llama.cpp-sycl, vllm-xpu]
primary_evidence:
  - ../../../results/2026/08/muse-dflash-sweep-manual-20260810T164735/FINAL-TABLE.md
  - ../../../research/2026/b70-vllm-muse-int4/PLAN.md
  - ../../../archive/legacy-docs-20260915/benchmark-history.md
---

# Muse Glimmer 30B

Muse Glimmer runs on the B70 through two distinct routes:

1. A characterized llama.cpp SYCL multimodal route using Q4_K_XL target weights, `mmproj`, and a DFlash draft. DFlash `n_max=2` was the measured winner at 128K context.
2. An experimental vLLM XPU compressed-tensors W4A16 group-32 overlay tied to a thin Muse support patch. Text and vision smokes passed, but this is not a public pullable production recipe.

The corrected llama.cpp evidence requires `GGML_SYCL_F16=ON`. Results from the earlier F16-disabled build must not be used to characterize the architecture's prefill performance.

Local Muse GGUF and INT4 artifacts were later deleted during storage cleanup; retained evidence describes completed runs, not guaranteed current launch availability.

See [adoption](adoption.md), [artifacts](artifacts.md), [serving](serving.md), [quality](quality.md), and [timeline](timeline.md).
