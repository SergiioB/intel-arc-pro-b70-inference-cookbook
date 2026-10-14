---
status: living
title: "Timeline"
type: model-dossier
date: 2026-08-10
---

---
schema_version: 1
model_id: facebook/Muse-Glimmer-30B
model_slug: muse-glimmer-30b
document: timeline
as_of: 2026-08-19
status: experimental
truth_verdict: CONFIRMED
evidence_tier: E2
primary_evidence:
  - ../../../archive/legacy-docs-20260915/benchmark-history.md
  - ../../../results/2026/08/muse-dflash-sweep-manual-20260810T164735/FINAL-TABLE.md
  - ../../../research/2026/b70-vllm-muse-int4/PLAN.md
---

# Timeline

| Date | Event |
|---|---|
| 2026-08-10 | First B70 llama.cpp SYCL text, vision, reasoning, and DFlash campaign completed |
| 2026-08-10 | Missing `GGML_SYCL_F16=ON` identified as a false prefill regression; build corrected and 64K card rerun |
| 2026-08-10 | 128K-context DFlash sweep selected n2; n8 aborted at launch; n=5 short and 8K-prompt cells retained |
| 2026-08-12 | Experimental vLLM compressed-tensors text-only screen recorded on local PR #51655 overlay |
| Later cleanup | Local Muse GGUF and INT4 artifacts deleted; evidence retained |
| Open | Stable upstream vLLM recipe, reacquired artifacts, parity testing, and near-boundary 128K completion |
