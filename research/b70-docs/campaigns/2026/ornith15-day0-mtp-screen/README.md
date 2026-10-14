---
status: final
title: "Ornith 1.5 day-0 MTP screen"
type: campaign
date: 2026-08-19
---

---
id: CAM-2026-ORNITH15-DAY0-MTP-SCREEN
doc_id: DOC-CAM-2026-ORNITH15-DAY0-MTP-SCREEN
status: provisional
---

# Ornith 1.5 day-0 MTP screen

- Model: [Ornith 1.5 dossier](../../../models/ornith/ornith-1.5-35b-a3b/README.md)
- Decision: [ADR-0003](../../../decisions/adr-0003-ornith-provisional-mtp1.md)
- Correction: [Ornith RTN fallback](../../../decisions/corrections-and-supersessions-2026-08-19.md#correction-ornith-rtn-fallback-is-part-of-the-artifact-lineage)
- Immutable raw evidence: `results/2026/08/ornith15-35b-20260819T1630Z` (`../../../results/2026/08/ornith15-35b-20260819T1630Z/`)
- Evidence tier: E2 self-reported raw evidence
- Status: PROVISIONAL — NOT FOR PUBLIC HEADLINE

This is an n=3 greedy diagnostic C1 screen of the locally converted GPTQ-INT4 G128 target with BF16 MTP tensors. MTP1 is the provisional candidate because later draft positions weakened in the recorded screen.

Limits: no fresh n=5 confirmation, no BF16/reference parity, no task-quality validation, and no 64K/128K completion ladder. The artifact lineage includes documented per-module RTN calibration fallback.
