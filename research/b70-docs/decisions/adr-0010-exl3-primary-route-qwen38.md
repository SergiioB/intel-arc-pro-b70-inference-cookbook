---
title: "ADR-0010: EXL3 as the primary Qwen3.8-27B route on one B70"
status: final
type: adr
date: 2026-09-27
---

---
id: b70-ref-adr-0010-exl3-primary-route-qwen38
title: "ADR-0010: EXL3 as the primary Qwen3.8-27B route on one B70"
document_type: reference
status: canonical
version: "1.0"
effective_date: 2026-09-27
last_reviewed: 2026-09-27
authority: docs/methods/benchmark-protocol.md
scope: Serving-route selection for Qwen3.8-27B on the single-card B70 host
adr:
  number: 10
  decision_status: accepted
  truth_verdict: ESTABLISHED
  evidence_tier: E2
  validation_state: campaign_measured
  publication_state: PROVISIONAL_not_for_public_headline
  correctness_state: route_measured_end_to_end
  operational_state: primary_route
  supersession_state: current
---

# ADR-0010: EXL3 as the primary Qwen3.8-27B route on one B70

## Context

The prior serving route — `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`
on vLLM-XPU — was removed by operator order (2026-09-26) after the exact-answer
battery showed the code_execution defect ("30" vs correct "14"). The
re-measure campaign `CAM-2026-AUTOROUND-VS-EXL3` then compared every viable
quantized route on one B70: external AutoRound (compressed-tensors), a locally
calibrated AutoRound (auto_gptq, code-heavy corpus, 128×1536×200 iters), the
bf16 base itself (TP2 control), EXL3 4bpw, and OpenVINO int4-ov.

## Decision

1. **EXL3 (exl3xpu engine, turboderp 4bpw artifact) is the primary route** for
   Qwen3.8-27B on one B70.
2. **OpenVINO int4-ov is the alternate route** — clean correctness, best
   150 W decode (~28.5 tok/s, ~414 GB/s effective), with the labeled
   thinking-mode handicap on long structured tasks.
3. **The W4A16 / GPTQ-family symmetric-int4 class is closed** for this model:
   correctness-defective on in-head numeric evaluation (all three independent
   exports answer 30), ~4.9× slower than the removed champion, and unable to
   reach 256K on one card (fp8 KV max ~211K). Do not re-attempt W4A16
   conversions of this model without a new quantization method (RL-087
   blocked-pattern).
4. **Nuance retained:** the defect is narrow — in-head numeric evaluation.
   AutoRound arm C scored 9/10 on host-executed code tasks (RL-088). "W4A16
   is defective" does not generalize to "broken for code authoring".

## Why EXL3 over the alternatives

- Correctness: clean 6/6 battery (code_execution 14/14), stable replay.
- Capacity: the only arm reaching native 262144 on one card — fp8 KV pool
  287,040 tokens @gmu 0.94; 131,098-token completion proven.
- Efficiency: smaller resident footprint (~17.3 GiB weights + leaner overhead)
  leaves ~76K more KV tokens than equal-fp8 AutoRound.
- Headroom: +38–42 % sustained decode at 230 W (power-limited ESIMD kernels);
  5.4× prefix-cache speedup, best of all arms.
- Accepted cost: lower effective bandwidth than OV (~240–355 vs ~414 GB/s) and
  a power-limited 150 W regime — OV remains the steady-150 W alternate.

## Evidence

`results/2026/09/autoround-vs-exl3-20260926/RESULTS.md` (full matrix);
`task2.jsonl`, `cache2.jsonl`, `kv-probes.jsonl`, `*-cappair.jsonl`;
ledger RL-084..RL-088; engine dossier `engines/exl3xpu/README.md`.

## Consequences

- `registry/model-status.json` and `models/STATUS.md` updated to the EXL3
  route; removed GPTQ artifact noted as retired/defective.
- vLLM-XPU remains the engine for W4A16-packaged checkpoints of OTHER models;
  this closure is model-specific (Qwen3.8-27B hybrid graph).
- Publication stays frozen pending operator authorization.
