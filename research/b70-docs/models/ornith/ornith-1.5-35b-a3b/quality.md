---
status: living
title: "Quality"
type: model-dossier
date: 2026-08-22
---

---
schema_version: 1
model_id: ornith-ai/Ornith-1.5-35B-A3B
model_slug: ornith-1.5-35b-a3b
document: quality
as_of: 2026-08-22
status: open
truth_verdict: UNCERTAIN
evidence_tier: E2
highest_correctness_rung: coherent_output_smoke_tested
primary_evidence:
  - ../../../research/2026/ornith15-35b-a3b-adoption-20260819.md
  - ../../../results/2026/08/ornith15-35b-20260819T1630Z/c1-screen.json
  - ../../../research/2026/ornith15-autoround-vs-gptq-20260822.md
  - ../../../results/2026/08/ornith15-ar-headtohead-20260822T0535Z/
  - ../../../results/2026/08/ornith15-ar-parity-20260822/
  - ../../../decisions/adr-0007-ornith-autoround-w4a16-candidate.md
  - artifacts.md
---

# Quality

## Verified

- The locally quantized target loaded and produced coherent responses during the diagnostic campaign.
- Fixed-length performance requests returned the expected output shape in the retained screen evidence.
- AutoRound W4A16 candidate (2026-08-22, n=3): 8/8 greedy probes coherent and
  on-task under the identical serving pattern; **zero RTN fallback**
  (30,720/30,720 expert projections signSGD-tuned; deep-block tuning losses
  cut 70–83%). Speed/KV parity with both GPTQ generations recorded.
- **Reference logprob parity measured (2026-08-22, streaming-BF16 gate,
  315 teacher-forced positions / 12 prompts): AR equal-or-best on all three
  primary metrics vs GPTQ control** — mean |Δlogprob| 0.2006 vs 0.2025,
  top-1 agreement 91.4% vs 90.5%, top-20-union KL 0.277 vs 0.319; GPTQ keeps
  a slight p95-tail edge (0.663 vs 0.702). Evidence:
  `results/2026/08/ornith15-ar-parity-20260822/parity-summary.json`.

## Not verified

- Task-quality parity (instruction following, reasoning, code, multilingual, long-context recall) — logprob parity on 12 short prompts does not establish end-task accuracy, for the original, MixedCal-v2, or AutoRound artifacts.
- Quality equivalence between MTP1 and no-spec under matched sampling.
- n>1 spread of the parity measurement (single deterministic run per side).

The RTN fallback was the central quality risk of the GPTQ lineage: **7,605 / 30,720 = 24.75%** of quantized projection matrices used RTN in the original conversion, including **297 / 768 = 38.7%** in layer 39 (reduced to 10.37% by MixedCal-v2 calibration, eliminated entirely by the AutoRound candidate). None of this proves or disproves degraded output; it bounds what can be claimed.

## LocalMaxxing eval status — 2026-08-20

A fresh authenticated `lmx eval suite list` preflight was attempted before starting the Ornith server. The current LocalMaxxing CLI requested its eval-suite API route and received an application **HTTP 404**. Therefore:

- no Ornith model server was started for this attempt;
- no evaluation questions were sent;
- no score was produced;
- no submission was attempted;
- this is classified as a **LocalMaxxing platform/API blocker**, not a model failure.

Evidence: `../../../results/2026/08/ornith15-lmx-eval-20260820T175635Z/`.

## LocalMaxxing speed-test status — 2026-08-20

The current supported `lmx speed-test run vllm` route was then executed against the proven original GPTQ MTP1 setup without submission. It completed and passed local payload validation:

- C1, one warmup plus three measured requests;
- 32 endpoint prompt tokens and 256 output tokens;
- 85.8 tok/s median client-observed output rate, range 85.6–85.8;
- 52.36 ms median TTFT;
- 70.67% MTP acceptance from counter deltas;
- 129.09 W measured average card draw during the workload at a configured 230 W cap.

The CLI's 611.2 tok/s `tokSPrefill` field is only a TTFT-derived rate from a 32-token prompt. It is not isolated prefill and does not replace the required p2048/p8192 cold-input anchors. This was a speed test, not a task-quality eval, so the quality verdict remains unchanged.

Evidence: `../../../results/2026/08/ornith15-lmx-speed-20260820T180754Z/`.

## Required next check

When LocalMaxxing suite discovery is available, run a bounded non-submitting quality eval against the proven original GPTQ MTP1 setup. Independently, run a matched deterministic suite against a trusted higher-precision reference, then a task-quality suite representative of the intended workload. Keep speed and acceptance evidence separate from the quality verdict.
