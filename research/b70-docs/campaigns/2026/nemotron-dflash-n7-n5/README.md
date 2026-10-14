---
status: final
title: "Nemotron DFlash BF16 n7 isolated C1 n=5"
type: campaign
date: 2026-08-13
---

---
id: CAM-2026-NEMOTRON-DFLASH-N7-N5
doc_id: DOC-CAM-2026-NEMOTRON-DFLASH-N7-N5
status: validated_internal
---

# Nemotron DFlash BF16 n7 isolated C1 n=5

- Model: [Nemotron 3.5 Lightning dossier](../../../models/nemotron/nemotron-3.5-lightning-30b-a3b/README.md)
- Decision: [ADR-0002](../../../decisions/adr-0002-nemotron-dflash-serving-default.md)
- Immutable raw evidence: `results/2026/08/nemotron-dflash-bf16-n7-n5-20260813T082203Z-1407994` (`../../../results/2026/08/nemotron-dflash-bf16-n7-n5-20260813T082203Z-1407994/`)
- Evidence tier: E2 self-reported raw evidence
- Status: validated internally; not independently reproduced

The campaign is the recorded isolated C1 Lane 1 run for the GPTQ-INT4 G64 target with the separate BF16 DFlash draft and seven speculative tokens. The dossier and ADR select this route over native MTP only for the exact recorded stack.

Limits: isolated C1 only; correctness is deterministic replay and coherence smoke rather than task-quality or BF16 parity. Exact workload, cache, timing, power, and build coordinates remain in the raw manifest and summary.
