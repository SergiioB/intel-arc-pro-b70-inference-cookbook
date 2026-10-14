---
status: living
title: "Quality"
type: model-dossier
date: 2026-08-08
---

---
schema_version: 1
model_id: Qwen/Qwen3.6-35B-A3B
model_slug: qwen3.6-35b-a3b
document: quality
as_of: 2026-08-19
status: provisional
truth_verdict: PARTIAL
evidence_tier: E2
highest_correctness_rung: deterministic_replay_suite
primary_evidence:
  - ../../../results/2026/08/vllm-mtp4-correctness-20260808T182243Z-157102/no-spec/results.json
  - ../../../archive/legacy-dirs-20260915/benchmarks/vllm-pi-realworld-and-long-context-2026-08-08.md
---

# Quality

The retained deterministic no-spec suite spans factual, math, code, logic, instruction, summary, creative, safety, ambiguity, and structured-output prompts. It used temperature 0, top_p 1, a fixed seed, and returned token IDs. This is stronger than a single coherent-output smoke, but it remains an internal deterministic suite.

The campaign does not establish:

- token/logit parity against an official BF16 reference;
- scored parity between MTP depths and no-spec;
- broad task-quality equivalence across sampling regimes;
- quality under concurrent mixed-load pressure.

Long-context completion demonstrates capacity and successful request finishing, not long-context recall quality. Evaluate retrieval/recall separately before making a long-context quality claim.
