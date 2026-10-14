---
status: living
title: "Quality"
type: model-dossier
date: 2026-09-15
---

---
schema_version: 1
model_id: facebook/Muse-Glimmer-30B
model_slug: muse-glimmer-30b
document: quality
as_of: 2026-08-19
status: provisional
truth_verdict: PARTIAL
evidence_tier: E2
highest_correctness_rung: multimodal_coherent_output_smoke_tested
primary_evidence:
  - ../../../results/2026/08/muse-glimmer-q4kxl-dflash-20260810T143831-2493876/manifest.txt
  - ../../../research/2026/b70-vllm-muse-int4/PLAN.md
  - ../../../archive/legacy-docs-20260915/benchmark-history.md
---

# Quality

The llama.cpp campaign observed coherent text and reasoning, a correct `2+2 → 4` response, and a correct vision description of a solid red image. DFlash outputs were coherent at the tested depths. The experimental vLLM overlay separately passed text and vision smokes.

These are smoke-level observations. They do not establish:

- parity to official BF16 for text or vision;
- quality parity between no-spec and DFlash;
- quality parity between llama.cpp Q4_K_XL and vLLM compressed-tensors;
- representative multimodal task performance;
- long-context recall at 128K.

DFlash acceptance and decode speed are not quality metrics. The F16 build correction changes performance interpretation, not the quality evidence level.
