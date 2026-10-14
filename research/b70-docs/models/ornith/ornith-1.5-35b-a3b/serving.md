---
status: living
title: "Serving"
type: model-dossier
date: 2026-09-15
---

---
schema_version: 1
model_id: ornith-ai/Ornith-1.5-35B-A3B
model_slug: ornith-1.5-35b-a3b
document: serving
as_of: 2026-08-19
status: provisional
truth_verdict: PARTIAL
evidence_tier: E2
engine: vllm-xpu
primary_evidence:
  - ../../../results/2026/08/ornith15-35b-20260819T1630Z/c1-screen.json
  - ../../../results/2026/08/ornith15-35b-20260819T1630Z/c1-screen-mtp4.json
  - ../../../results/2026/08/ornith15-35b-20260819T1630Z/mtp1-accept-cells.json
  - ../../../results/2026/08/ornith15-35b-20260819T1630Z/server4.log
---

# Serving

## Tested stack

The successful path used the `f01e24f6…` vLLM XPU image generation with the recorded MTP nightly and boundary patches, GPTQ target weights, float16 runtime dtype, language-model-only mode, block size 64, cache explicitly off, and a configured 230 W cap. The 32K no-spec load reported 6,024 MiB free and a logged KV capacity of 169,622 tokens.

## Provisional C1 screen

These are client-observed metrics from **n=3 greedy diagnostic** requests. They are not engine-native decode and are not public headline results.

| Mode | Context used for load | Cell | Median client post-first | Acceptance |
|---|---:|---|---:|---:|
| no-spec | 32K | p512/g128 | 69.0 tok/s | — |
| MTP4 | 16K | filler p512/g128 | 65.2 tok/s | 24.8% overall |
| MTP4 | 16K | real prompt | 69.8 tok/s | position profile 81% / 15% / 2.5% / 0.5% |
| MTP1 | 16K | filler p512/g128 | 94.3 tok/s | 80.3% position 0 |
| MTP1 | 16K | real prompt | 94.3 tok/s | 80.3% position 0 |

MTP1 is the current research choice. Do not generalize the measured improvement beyond these cells: context, prompt family, speculation depth, and sample count differ across parts of the screen.

## Serving safeguards

Use exact endpoint token counts, a same-shape warmup, retained samples, explicit cache state, and an empty-GPU preflight. Do not promote the 32K load to a 128K capacity claim; no near-boundary request was completed.
