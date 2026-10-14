---
status: living
title: "Timeline"
type: model-dossier
date: 2026-08-11
---

---
schema_version: 1
model_id: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B
model_slug: nemotron-3.5-lightning-30b-a3b
document: timeline
as_of: 2026-08-19
status: validated_internal
truth_verdict: CONFIRMED
evidence_tier: E2
primary_evidence:
  - ../../../research/2026/nemotron-session-summary-20260813.md
  - ../../../archive/legacy-docs-20260915/benchmark-history.md
  - ../../../archive/legacy-docs-20260915/compat-stubs/recommended-configs.md
---

# Timeline

| Date | Event |
|---|---|
| 2026-08-11 | Historical llama.cpp Q4_0 requantization established a faster model-weight format than the tested Q4_K_M baseline; retained under `results/2026/08/nemotron-q4_0-test/` |
| 2026-08-11 | Historical Q4_0 MTP4 230 W campaign retained under `results/2026/08/nemotron-q4_0-mtp4-230w/` |
| 2026-08-12 | Local GPTQ-INT4 G64 target and XPU loader/kernel path validated |
| 2026-08-13 | Isolated vLLM GPTQ+DFlash n=5 card completed; earlier n=3 screen superseded |
| 2026-08-13 | Scheduler-budget 16K spike failed fresh n=5 confirmation; 8,192 remained the recipe |
| 2026-08-13 | Capacity ladder completed through 119,904 total tokens; 128K not run |
| Later cleanup | Historical Q4_0 GGUF deleted; evidence retained, launch path removed |
