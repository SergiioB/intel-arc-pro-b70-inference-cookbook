---
status: living
title: "Quality"
type: model-dossier
date: 2026-09-15
---

---
schema_version: 1
model_id: Qwen/Qwen3.8-27B
model_slug: qwen3.8-27b
document: quality
as_of: 2026-08-19
status: provisional
truth_verdict: PARTIAL
evidence_tier: E2
highest_correctness_rung: browser_observed_task_quality
primary_evidence:
  - ../../../results/2026/08/qwen38-quality-20260815/QUALITY-REPORT.md
  - ../../../research/2026/qwen38-27b-day0-adoption-20260814.md
  - ../../../research/2026/qwen38-cookbook-pr1-pr2-review-20260818.md
---

# Quality

The day-0 llama.cpp and vLLM paths passed load, greedy smoke, deterministic replay, and fixed-g128 token-exactness checks. Those rungs did not establish higher-precision reference parity.

A later browser-observed campaign used the exact champion GPTQ artifact with MTP4 for ten Three.js/WebGL/game generations. Solar-system, globe, and minigolf artifacts rendered and supported observed interactions. The no-thinking space-shooter update loop failed after start. Raw WebGL2 shader pages could not be fully judged in the test webview because it lacked WebGL2; their fallback overlays worked. This is task-quality observation, not throughput evidence.

Still open:

- Trusted-reference token/logit parity for the GPTQ target.
- Formal representative task scoring.
- S+M1 INT4 draft/head token, task, or KL parity versus BF16 draft/head.
- DFlash2 correctness beyond coherent-output smoke.

Do not transfer the base artifact's observed quality to the optional INT4 draft/head overlay without the missing matched quality test.
