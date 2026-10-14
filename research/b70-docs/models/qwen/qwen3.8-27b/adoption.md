---
status: living
title: "Adoption"
type: model-dossier
date: 2026-08-27
---

---
schema_version: 1
model_id: Qwen/Qwen3.8-27B
model_slug: qwen3.8-27b
document: adoption
as_of: 2026-08-19
status: validated_internal
truth_verdict: PARTIAL
evidence_tier: E2
publication_gate: PROVISIONAL_NOT_FOR_PUBLIC_HEADLINE
primary_evidence:
  - ../../../research/2026/qwen38-27b-day0-adoption-20260814.md
  - ../../../research/2026/qwen38-27b-drafter-campaign-20260817.md
  - ../../../research/2026/qwen38-cookbook-pr1-pr2-review-20260818.md
---

# Adoption

## Decision

Use vLLM XPU GPTQ target + native MTP4 as the base serving class. The later mixed-split overlay is infrastructure for mixed speculative/non-speculative steps. Treat the S+M1 draft/head INT4 overlay as optional pending quality parity, and keep DFlash2 outside serving defaults.

## Completed gates

- Official release and architecture transfer audited.
- Community Q4_K_M GGUF route loaded in llama.cpp; its converter stripped MTP.
- Self-GPTQ target contract verified and published.
- vLLM no-spec/MTP1/MTP2/MTP4 matrix completed at 131,072 max context.
- Exact 131,072-total-token request completed under all four modes.
- Prefix-on real-world, concurrency, sampling, and browser-observed task campaigns added after the initial day-0 work.

## Open gates

- Independent reproduction of the E2 performance cards.
- Reference parity for the production target.
- Token/task/KL parity for the S+M1 INT4 draft/head overlay versus BF16 draft/head.
- Correct block-diffusion masking and competitive performance for DFlash2.
- Dual-card vLLM TP2 (B450-F at capture time; X570-F dual x8/x8 since 2026-08-27): validated on the registered campaign recipe. External host-staged-collectives recipe exists; see [ADR-0008](../../../decisions/adr-0008-dual-b70-independent-default.md). Independent per-card stacks remain the default.
