---
title: "ADR-0007: AutoRound W4A16 as the default B70 INT4 converter"
status: final
type: adr
date: 2026-08-22
---

---
id: b70-ref-adr-0007-ornith-autoround-w4a16-candidate
title: "ADR-0007: AutoRound as the default B70 INT4 converter"
document_type: reference
status: accepted
version: "1.2"
effective_date: 2026-08-22
last_reviewed: 2026-08-22
authority: docs/methods/benchmark-protocol.md
scope: INT4 conversion route for Ornith-1.5-35B-A3B on vLLM XPU (B70), experts-only W4A16 sym G128 with MTP BF16
adr:
  number: 7
  decision_status: accepted_default_converter
  truth_verdict: PARTIAL
  evidence_tier: E2
  validation_state: reference_parity_passed_equal_or_best
  publication_state: not_published
  correctness_state: logprob_parity_measured_vs_bf16
  operational_state: default_converter_for_new_conversions
  supersession_state: ornith_serving_swap_pending_n5
---

# ADR-0007: AutoRound W4A16 as the default B70 INT4 converter

**Authorization:** converter decision accepted by the operator on 2026-08-22
("use AutoRound moving forward") after the completed head-to-head campaign.

## Context

The Ornith-1.5 GPTQ conversion lineage carries calibration debt in the form
of GPTQModel's low-observation RTN fallback: 24.76% of expert projections
untuned in the original (Aug-19) conversion, still 10.37% in the current
MixedCal-v2 reference. AutoRound signSGD tuning was run with the identical
dynamic scope, calibration corpus (MixedCal-v2, sha `e88ccd5f…`), and packing
(auto_gptq, sym, G128, desc_act=false, MTP BF16) so the converter is the only
changed variable.

## Decision

1. **AutoRound is the default converter for new B70 INT4 conversion work**
   (routed MoE and dense), effective 2026-08-22. New conversions start from
   the retained recipe: auto-round 0.14.2 (pinned), `quantize_and_save`
   streaming save, forced `is_model_patched` where the model type is absent
   from auto_round's MODEL_CONFIG, `AR_WORK_SPACE` on a ≥90 GiB-free volume,
   24 GiB container cgroup cap plus host watchdog. GPTQModel remains the
   documented fallback for targets where autograd or host memory blocks
   AutoRound; the blocker must be recorded in the conversion manifest.
2. The **artifact format is unchanged** — AutoRound emits the same
   `auto_gptq`/gptq packing (qweight/qzeros/scales, desc_act=false), so
   ADR-0005 (GPTQ packing as the best-proven practical B70 vLLM path) is not
   superseded: this decision changes the converter, not the format.
3. **Existing serving routes do not switch automatically.** MixedCal-v2 GPTQ
   remains the Ornith serving/publishing/LocalMaxxing reference until the
   AutoRound artifact passes an **n=5 confirmation set** at the declared cap
   (the reference-parity requirement was met on 2026-08-22, see below) plus
   upload/publication authorization. The AR candidate is not uploaded and not
   submitted anywhere.
4. **Reference parity (measured 2026-08-22, post-authorization addendum):**
   teacher-forced against the BF16 source on 315 positions / 12
   out-of-calibration prompts — mean |Δlogprob| 0.20059 (AR) vs 0.20250
   (GPTQ), top-1 agreement 0.9143 vs 0.9048, top-20-union KL 0.27747 vs
   0.31911. **AR equal-or-best on all three primary metrics**; GPTQ retains
   only a slight p95-tail edge (0.663 vs 0.702). Evidence:
   `results/2026/08/ornith15-ar-parity-20260822/`.
5. Rationale: serving is a verified drop-in (same `--quantization gptq` path,
   same KV footprint 169,622 tok @ 32K, speed inside n=3 spread of both GPTQ
   generations, MTP1 works with the same patches at 78.5% pos0 vs 80.3%), so
   the only remaining question is accuracy — and removing untuned weights is
   structurally safer than retaining them.

## Consequences

- **Local artifact state (2026-08-22, operator-directed):** both local GPTQ
  MixedCal-v2 copies were deleted for storage recovery; the artifact remains
  available on public Hugging Face (re-download is the recovery path, API
  verified at deletion time). The AutoRound artifact is consequently the only
  local Ornith INT4 artifact and the local serving candidate; if GPTQ serving
  is required for an n=5 cross-check against the published reference,
  re-download it first (manifests:
  `results/2026/08/ornith15-ar-parity-20260822/*-deletion-manifest.txt`).
- Any future Ornith INT4 conversion for quality-sensitive serving starts
  from the AutoRound recipe (auto-round 0.14.2 pinned, `quantize_and_save`
  streaming, forced `is_model_patched`, `AR_WORK_SPACE` on the NTFS volume,
  24 G cgroup cap + host watchdog); see
  `research/2026/ornith15-autoround-vs-gptq-20260822.md` §3 for the full
  host-compatibility ledger.
- The behavioral probe (8/8 coherent on both artifacts) is not acceptance
  evidence for parity claims.
- Conversion cost: ~3.5 h wall on this host (vs ~6.4 h GPTQ).

## Evidence

- `research/2026/ornith15-autoround-vs-gptq-20260822.md` (campaign narrative,
  three-way fallback ladder, results)
- `results/2026/08/ornith15-ar-headtohead-20260822T0535Z/` (AR screens, MTP1 accept,
  probes, scripts, conversion log)
- `results/2026/08/ornith15-ar-headtohead-control-20260822T0551Z/` (control probe)
- Conversion evidence: `/mnt/<disk>/runs/{ar-smoke-SUCCESS,ar-r1}`;
  smoke artifact preserved at `/mnt/<other-disk>/ar-artifacts/`
- Control baselines: `results/2026/08/ornith15-35b-20260819T1630Z/c1-screen.json` (original,
  n=3) and MixedCal-v2 n=5 dossier numbers (70.74 / 96.43 @ 150 W)
