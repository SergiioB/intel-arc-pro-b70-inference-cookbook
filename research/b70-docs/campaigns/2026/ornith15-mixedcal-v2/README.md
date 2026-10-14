---
status: final
title: "Ornith 1.5 MixedCal-v2 characterization"
type: campaign
date: 2026-08-21
---

---
id: CAM-2026-ORNITH15-MIXEDCAL-V2
doc_id: DOC-CAM-2026-ORNITH15-MIXEDCAL-V2
status: validated_internal
---

# Ornith 1.5 MixedCal-v2 characterization

- Model: [Ornith 1.5 dossier](../../../models/ornith/ornith-1.5-35b-a3b/README.md)
- Claims lock: cookbook `ornith15-35a3/CLAIMS.md`
- Compiler: `results/2026/08/ornith15-mixedcal-v2-summary/summary.json` (`../../../results/2026/08/ornith15-mixedcal-v2-summary/summary.json`)
- Evidence tier: E2 self-reported raw evidence
- Status: self-reported with limits (C1; LMX not submitted; no BF16 KL)

MixedCal-v2 is the experts-only GPTQ INT4 G128 retarget with mixed-domain
calibration. Conversion win is RTN fallback **10.37% vs 24.76%**. Speed at
150 W is **parity** with the original WikiText GPTQ. Research serve is MTP1.

## Evidence roots (immutable)

| Piece | Root |
|---|---|
| Conversion r4 | `/mnt/<models-disk>/ornith15-quant/mixedcal-v2/runs/mixedcal-v2-20260820T204436Z-r4/` |
| n=5 no-spec | `results/2026/08/ornith15-n5-maxkv-20260821T054323Z/` |
| MTP1 + ladder | `results/2026/08/ornith15-n5-maxkv-resume-20260821T062400Z/` |
| Exact-token capacity | `results/2026/08/ornith15-boundary-fixup-20260821T072410Z/` |
| MTP depth + 150↔230 W A/B | `results/2026/08/ornith15-mtpdepth-power-20260821T075858Z/` |
| DraftINT4 S+M1 | `results/2026/08/ornith15-draftint4-20260821T084132Z/` |
| LMX speed-test MTP1 150 W p32 (not submitted) | `results/2026/08/ornith15-lmx-mixedcal-20260821T090450Z/` |
| LMX + harness 230 W long-prompt (not submitted) | `results/2026/08/ornith15-lmx-prefill230-20260821T095259Z/` | |

## Limits

- C1 only. No Cn serving claim.
- No BF16 logit/KL or task-quality suite.
- Prefill A/B is configured **power cap** 150 vs 230 W, not an independent GT
  clock sweep. Clocks were not pinned; the GPU drew ~152 W vs ~222 W.
- LocalMaxxing MTP1 150 W p32/g256 `cmt2sl6eg0hdcmv01gre5o3ub` APPROVED (`tokSOut` 94.1).
- LocalMaxxing 230 W no-spec long-prompt g128 `cmt2sr6gq0himmv01ogieh0c8` APPROVED (`tokSPrefill` 9780 / `tokSOut` 69.9). The g=1 19k `tokSOut` payload was **not** submitted.
- Hugging Face: `SergiioB/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2`.
