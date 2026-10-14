---
status: living
title: "Artifacts"
type: model-dossier
date: 2026-09-15
---

---
schema_version: 1
model_id: facebook/Muse-Glimmer-30B
model_slug: muse-glimmer-30b
document: artifacts
as_of: 2026-08-19
status: deleted_local_reacquisition_required
truth_verdict: CONFIRMED
evidence_tier: E2
primary_evidence:
  - ../../../results/2026/08/muse-glimmer-q4kxl-dflash-20260810T143831-2493876/manifest.txt
  - ../../../results/2026/08/muse-dflash-sweep-manual-20260810T164735/MANIFEST.txt
  - ../../../results/2026/08/muse-glimmer-int4-screen-20260812T192845Z-835270/manifest.txt
  - ../../../research/2026/b70-vllm-muse-int4/PLAN.md
---

# Artifacts

## llama.cpp campaign set

- `unsloth/Muse-Glimmer-30B-GGUF` `UD-Q4_K_XL`: 15.88 GB.
- `mmproj-kquant`: 1.40 GB.
- `dflash-kquant`: 1.63 GB.

This three-part set supplied target text weights, vision projection, and DFlash draft respectively. The local files were later deleted; the manifests and raw results remain evidence of the completed campaign, not evidence that the paths still launch.

## vLLM experimental target

The experimental route used compressed-tensors W4A16 group-32 weights with a local Muse support overlay. Artifact naming may include “AWQ,” but the recorded metadata says `compressed-tensors`; use the metadata, not the repository name, when describing quantization.

The vLLM image/overlay is local-only and tied to the PR #51655 work. Reproduction requires reconstructing that exact support layer or using merged upstream support when available.
