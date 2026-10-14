---
status: final
title: "Qwen3.8 four-mode 230 W campaign"
type: campaign
date: 2026-08-19
---

---
id: CAM-2026-QWEN38-4MODE-230W
doc_id: DOC-CAM-2026-QWEN38-4MODE-230W
status: validated_internal
---

# Qwen3.8 four-mode 230 W campaign

- Model: [Qwen3.8-27B dossier](../../../models/qwen/qwen3.8-27b/README.md)
- Decisions: [ADR-0005](../../../decisions/adr-0005-gptq-practical-vllm-path.md) and [ADR-0006](../../../decisions/adr-0006-sub-1gib-vram-reserve.md)
- Corrections: [cache and power interpretation](../../../decisions/corrections-and-supersessions-2026-08-19.md)
- Immutable raw evidence: `results/2026/08/vllm-qwen38-4mode-230w-20260815T170840Z-581426` (`../../../results/2026/08/vllm-qwen38-4mode-230w-20260815T170840Z-581426/`)
- Evidence tier: E2 self-reported raw evidence
- Status: PROVISIONAL — NOT FOR PUBLIC HEADLINE

This campaign records four serving modes on the Qwen3.8 GPTQ-INT4 G128 target with BF16 MTP. The dossier requires greedy diagnostic, recommended non-thinking, and recommended thinking regimes to remain separate.

Limits: the configured 230 W cap is not measured average draw; corrected power comparisons require matched active-request windows. Sub-1 GiB reserve observations are isolated research capacity evidence only. Use the raw summary and tables for exact workload coordinates and do not transfer numbers between sampling regimes.
