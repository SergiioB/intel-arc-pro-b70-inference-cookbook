---
title: "Corrections and Supersessions for Historical B70 Claims"
status: final
type: adr
date: 2026-08-19
---

---
id: b70-report-corrections-and-supersessions-2026-08-19
title: Corrections and Supersessions for Historical B70 Claims
document_type: report
status: canonical
version: "1.0"
effective_date: 2026-08-19
last_reviewed: 2026-08-19
authority: docs/methods/benchmark-protocol.md
scope: Registry-ready correction and supersession records for cache, power, quantization, and artifact-state claims
correction_set:
  registry_state: ready_for_ingestion
  truth_verdict: CONFIRMED
  evidence_tier: E2
  publication_state: correction_required_where_repeated
  preservation_state: retain_old_evidence
  registry_edit_state: applied_2026-08-19
---

# Corrections and Supersessions for Historical B70 Claims

This file is the human-readable source for correction and supersession content now ingested into `archive/registry-system-20260830/supersessions.yaml`. The source document itself does not mutate historical evidence; registry application is recorded separately and preserves the old observations.

## Registry-ready summary

| Correction ID | Subject | Old disposition | Replacement disposition |
|---|---|---|---|
| `correction:qwen-prefill-cache:2026-08-19` | Historical Qwen HTTP prefill with repeated prefixes | `invalid` for uncached prefill and power-gain claims | Use entropy-first unique prompts with zero cache-hit delta |
| `correction:dense-power-window:2026-08-19` | Dense mode power means from unequal monitor windows | `superseded` for mode-to-mode interpretation | Use matched active-request windows |
| `correction:ornith-rtn-fallback:2026-08-19` | Ornith GPTQ calibration lineage | `partial` artifact, not fully GPTQ-calibrated parity | Retain fallback disclosure and require quality validation |
| `correction:nemotron-q4-0-evidence-path:2026-08-19` | Nemotron model-weight Q4_0 result | `superseded` when cited from logs or prose alone | Use campaign JSON plus submission record; keep requantization caveat |
| `correction:artifact-availability-state:2026-08-19` | Uploaded, locally present, and deleted states | `invalid` when collapsed into one availability flag | Track remote publication and local presence independently |

## Correction: historical cache-contaminated Qwen prefill

**Correction ID:** `correction:qwen-prefill-cache:2026-08-19`  
**Truth verdict:** `CONFIRMED`  
**Evidence tier:** `E2`  
**Affected metric class:** HTTP cold input/prefill  
**Old evidence disposition:** `invalid` for uncached prefill and any derived power-scaling claim

> **Correction — 2026-08-19**
>
> Earlier Qwen HTTP prefill sweeps reused a constant filler prefix and changed only a suffix. Later repetitions reused prefix-cache blocks, so the reported “steady-state” values did not measure uncached prefill. The observed p8192 jump from 8,640 to 42,733 tok/s was cache reuse, not a fivefold prefill improvement. The associated claim that 230 W improved prefill by 16–22% is withdrawn.
>
> Remains valid: the server ran and prefix caching accelerated repeated-prefix requests. Withdrawn: uncached prefill rates from the affected repeated-prefix cells and power conclusions derived from them.

**Old evidence and defect record:**

- [Benchmark history Run 24, methodology fix](../archive/legacy-docs-20260915/benchmark-history.md)
- [Benchmark protocol §1.5, prefix-cache contamination](../docs/methods/benchmark-protocol.md)

**Replacement method:** entropy at the beginning of every prompt, exact rendered-token preservation, and a zero `prefix_cache_hits` delta over the measured window. Later vLLM wording must call `prompt_tokens / client TTFT` a cold input rate unless a server-native phase timer isolates prefill.

**Supersession rule:** any registry claim sourced from `b70-mtp-prefill-sweep.py` or `b70-mtp-grid-bench.py` must be marked `invalid` for uncached prefill unless a specific observation independently proves unique prefixes and zero cache-hit delta.

## Correction: dense power-window interpretation

**Correction ID:** `correction:dense-power-window:2026-08-19`  
**Truth verdict:** `CONFIRMED`  
**Evidence tier:** `E2`  
**Affected metric class:** measured average card draw by MTP mode  
**Old evidence disposition:** `superseded` for mode-to-mode comparison

> **Correction — 2026-08-19**
>
> The historical dense campaign means of 195 W, 197 W, 146 W, and 146 W used unequal monitor coverage. The no-spec and MTP1 windows included heavy full-context prefill, while the other windows covered shorter or interrupted work. They cannot support a claim that MTP depth caused those power differences.
>
> A matched active-request-window A/B measured no-spec 149.9 W, MTP4 151.0 W, MTP1 156.1 W, and MTP2 153.3 W mean on the declared workload. All four modes were within a 6 W band. The retained maximum values are 0.5-second interval averages, not instantaneous peaks.

**Old evidence:** [Benchmark history Run 31 power table and adjacent correction](../archive/legacy-docs-20260915/benchmark-history.md)

**Replacement evidence:**

- Matched power summary (`../results/2026/08/dense27-matched-power-20260809T214116Z/summary.json`)
- Matched power manifest (`../results/2026/08/dense27-matched-power-20260809T214116Z/manifest.txt`)

**Remains valid:** the historical campaign-monitor readings occurred over their recorded windows.  
**Withdrawn:** interpreting their differences as a causal MTP-mode power ranking.

## Correction: Ornith RTN fallback is part of the artifact lineage

**Correction ID:** `correction:ornith-rtn-fallback:2026-08-19`  
**Truth verdict:** `CONFIRMED`  
**Evidence tier:** `E2`  
**Affected subject:** correctness and quantization provenance  
**Old evidence disposition:** `partial`

> **Correction — 2026-08-19**
>
> The local Ornith-1.5 GPTQ conversion did not complete every expert projection matrix through the same calibrated GPTQ path. The canonical quantizer CSV records RTN fallback for **7,605 of 30,720 expert projection matrices (24.755859%)**: 2,535 each for `gate_proj`, `up_proj`, and `down_proj`; layer 39 contains 297 of 768 fallbacks (38.671875%). Therefore the artifact must not be described as fully GPTQ-calibrated or quality-equivalent to BF16 on the strength of its load, coherence, or speed screen.
>
> RTN here is GPTQModel's per-module low-observation calibration fallback. It is not an XPU runtime fallback, failed INT4 packing, or an unquantized BF16 substitution. Remains valid: the saved artifact passed its stated tensor contract, loaded on the recorded vLLM XPU image, produced coherent output, and supported the provisional MTP screen. Not established: task-quality parity, logit/KL parity, or the quality impact of the RTN-fallback matrices.

**Evidence:**

- [Ornith adoption report, failure ledger](../research/2026/ornith15-35b-a3b-adoption-20260819.md)
- [Ornith adoption report, open gates](../research/2026/ornith15-35b-a3b-adoption-20260819.md)
- Ornith starting manifest (`../results/2026/08/ornith15-35b-20260819T1630Z/manifest-start.json`)
- Derived RTN summary with canonical source hash (`../results/2026/08/ornith15-35b-20260819T1630Z/derived/rtn-fallback-summary.json`)

**Required registry fields:** `quantization_provenance=GPTQ_with_RTN_fallback`, `quality_state=not_validated_against_reference`, `performance_state=provisional_n3`, `publication_state=not_for_public_headline`.

## Correction: Nemotron Q4_0 evidence path

**Correction ID:** `correction:nemotron-q4-0-evidence-path:2026-08-19`  
**Truth verdict:** `CONFIRMED` for the named recorded cells  
**Evidence tier:** `E2` local raw evidence; platform publication is not independent reproduction  
**Affected subject:** model-weight Q4_0 throughput and provenance  
**Old evidence disposition:** `superseded` when supported only by a server log, prose summary, or unmatched baseline

> **Correction — 2026-08-19**
>
> The Nemotron Q4_0 claim concerns **model weights**, not KV-cache K quantization. The artifact was requantized from Q4_K_M with `--allow-requantize`; it was not produced directly from BF16. Cite the retained campaign JSON and submission record, preserve the extra quality-loss caveat, and do not generalize the result to other models or to `q4_0` K cache.
>
> Remains valid at the recorded cells: 68.0 tok/s no-spec at tg128/150 W and 87.4 tok/s median native-MTP4 at 230 W, n=5, on the recorded llama.cpp SYCL build. Withdrawn: universal “Q4_0 is optimal on Xe2” wording, quality-equivalence implications, and any use of this result to permit q4_0 K cache.

**Canonical evidence path for the claim:**

- Raw Q4_0 test root (`../results/2026/08/nemotron-q4_0-test/`)
- MTP4 Q4_0 server evidence (`../results/2026/08/nemotron-q4_0-mtp4-230w/server.log`)
- LocalMaxxing submission record (`../results/_quarantine/nemotron-q4_0-lmx-result.json`)
- [Conversion provenance and caveat](../research/2026/nemotron-lightning-optimization-20260811/08-q4_0-breakthrough.md)

**Comparison classification:** `matched_except_quant` only where the exact source documents establish matched coordinates; otherwise `directional_only`.

## Correction: artifact uploaded and deleted states are independent

**Correction ID:** `correction:artifact-availability-state:2026-08-19`  
**Truth verdict:** `CONFIRMED` for the dated states below  
**Evidence tier:** `E2` repository provenance  
**Affected subject:** artifact availability  
**Old evidence disposition:** `invalid` when one boolean is used for remote publication and local presence

> **Correction — 2026-08-19**
>
> “Uploaded,” “present locally,” and “deleted locally” are separate status dimensions. A remote Hugging Face upload does not prove the artifact remains on a given local volume, and a local deletion does not prove a remote repository was deleted. Registry ingestion must preserve each dated state independently.

**Dated states supported by retained evidence:**

| Artifact or class | Remote publication state | Local state | Evidence |
|---|---|---|---|
| `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` | Uploaded 2026-08-15; 16 files recorded | Do not infer from upload record; consult current host state/live storage | [Archived pre-migration Run 40](../archive/docs/pre-migration-20260819/benchmark-history.md) |
| Muse local GGUFs and `muse-glimmer-awq-int4` checkpoint | No deletion of a remote repository is established by the cited record | Deleted locally 2026-08-14 | [Archived pre-migration artifact note](../archive/docs/pre-migration-20260819/AGENTS.md) |
| Ornith-1.0 35B Q5 GGUF | Remote state not established by the deletion record | Deleted locally 2026-08-14; stale launcher path remained | [Archived pre-migration model inventory](../archive/docs/pre-migration-20260819/AGENTS.md) |
| Ornith-1.5 local GPTQ conversion | Remote publication not established in the adoption report | Present in the 2026-08-19 host snapshot | [Current host state](../docs/operations/current-host-state.md) |

**Required registry dimensions:** `remote_repository_state`, `remote_state_observed_at`, `local_presence_state`, `local_state_observed_at`, `host_or_volume`, `provenance_state`, and `last_verified`. Unknown fields must remain `unknown` rather than being inferred.

## Registry ingestion rules

1. Preserve all old evidence links and assign `invalid` or `superseded`; do not delete historical observations.
2. Keep truth verdict, evidence tier, lifecycle state, publication state, correctness state, and artifact availability as independent dimensions.
3. Do not upgrade LocalMaxxing `APPROVED` to independent reproduction.
4. Do not rewrite raw result JSON, historical manifests, benchmark history, or registries during ingestion.
5. Apply correction IDs to every registry claim or artifact entry that repeats the affected wording.
