---
status: living
title: "Quality"
type: model-dossier
date: 2026-09-15
---

---
schema_version: 1
model_id: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B
model_slug: nemotron-3.5-lightning-30b-a3b
document: quality
as_of: 2026-08-19
status: open
truth_verdict: PARTIAL
evidence_tier: E2
highest_correctness_rung: deterministic_replay_smoke_tested
primary_evidence:
  - ../../../results/2026/08/nemotron-dflash-bf16-n7-n5-20260813T082203Z-1407994/correctness-summary.json
  - ../../../research/2026/nemotron-dflash-xpu-first-proof-20260813.md
  - ../../../research/2026/nemotron-lightning-optimization-20260811/08-q4_0-breakthrough.md
---

# Quality

The current GPTQ+DFlash pair passed a deterministic raw-completion replay smoke with exact match in the retained campaign. That establishes repeatability for the tested path, not reference parity or task quality.

Open checks:

- GPTQ target versus a trusted higher-precision reference.
- DFlash versus no-spec quality under matched sampling and prompts.
- Representative reasoning, code, instruction-following, safety, and long-context tasks.
- Quality implications of reconstructing the draft from NVFP4.

The deleted historical Q4_0 GGUF was requantized from Q4_K_M rather than BF16, so it carries an additional quantization-loss caveat. Its historical speed evidence must not be used as quality evidence for the current GPTQ target.
