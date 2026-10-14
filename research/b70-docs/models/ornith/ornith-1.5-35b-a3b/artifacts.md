---
status: living
title: "Artifacts"
type: model-dossier
date: 2026-08-24
---

---
schema_version: 1
model_id: ornith-ai/Ornith-1.5-35B-A3B
model_slug: ornith-1.5-35b-a3b
document: artifacts
as_of: 2026-08-21
status: provisional
truth_verdict: CONFIRMED
scope: artifact_contract_and_calibration_fallback
evidence_tier: E2
primary_evidence:
  - ../../../research/2026/ornith15-35b-a3b-adoption-20260819.md
  - ../../../results/2026/08/ornith15-35b-20260819T1630Z/supervise.stdout
  - ../../../results/2026/08/ornith15-35b-20260819T1630Z/download.log
---

# Artifacts


> **Route closed — 2026-08-24 (user decision: quality rejected).** All local
> Ornith artifacts were removed: source shards (`/mnt/<models-disk>/Ornith-1.5-35B-A3B-shards`),
> AutoRound W4A16 artifact + conversion workspaces (including the 80G loop image,
> `ar-artifacts`, `ornith15-quant`), the Ornith 1.0 mmproj GGUF, the HF cache stub,
> and the `~/bench-ornith-vs-qw38` repo. The `bench-report.service` was stopped and
> disabled; the autoround loop mount and its fstab entry were torn down.
> Measured evidence is retained under `results/ornith15-*` and
> `campaigns/2026/ornith15-*`. The HF-public MixedCal-v2 repo remains re-downloadable.
> Removal log: `results/2026/08/cleanup-ornith-removal-20260824T121500Z/`.

## Source and derived target

- Source: `ornith-ai/Ornith-1.5-35B-A3B`, recorded revision `fbb995a79eedd569a5edc5f2af9644c0fa1124fc`.
- Derived B70 target: `Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16`.
- Reported derived size: 24.42 GB.
- Quantizer: gptqmodel 7.3.2; 4-bit, symmetric, group size 128.
- Contract check: **30,720 `qweight` tensors**, **785 BF16 `mtp.*` tensors**, and **0 MTP quantization leaks**.

## Exact RTN fallback accounting

The quantizer log contains **7,605 RTN fallbacks out of 30,720 quantized projection matrices: 24.75%**. The fallback count is exactly balanced across the three expert projections:

| Projection | RTN fallback modules |
|---|---:|
| `gate_proj` | 2,535 |
| `up_proj` | 2,535 |
| `down_proj` | 2,535 |
| **Total** | **7,605 / 30,720 = 24.75%** |

Layer 39 alone records **297 fallbacks out of 768 expert projection matrices = 38.7%**.

## What “calibration fallback” means

This was a **per-module fallback inside the GPTQ quantizer**. When the calibration-driven GPTQ solve could not use a module's collected activation statistics successfully, the quantizer emitted that module with round-to-nearest (RTN) weight quantization instead. It is not a runtime CPU fallback, kernel fallback, unquantized BF16 substitution, or failed tensor. The module still appears in the verified GPTQ contract, but it did not receive the intended calibration-aware GPTQ optimization.

Because 24.75% of the quantized projection matrices used RTN, this artifact remains **provisional** until reference and task-quality tests determine whether the fallback distribution is acceptable.

## MixedCal-v2 candidate

Published on Hugging Face at
`SergiioB/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2` (public,
revision `ed26281b`, six shards, API revision `89306782` verified 2026-08-22).
Contract check: **30,720 expert `qweight`**, **785 MTP tensors**,
**0 MTP quantized**, **0 forbidden qweight**. RTN fallback
**3,186 / 30,720 = 10.37%** (layer 39: 75/768). Calibration: 128 × 1,536 mixed-domain
tokens, seed `15035260819`, corpus SHA-256 `e88ccd5f…`. Original GPTQ and BF16
source were not overwritten. See [MixedCal-v2 conversion](mixedcal-v2-conversion.md).

**Local presence state (2026-08-22):** both local copies were **deleted at
operator direction** to free storage — the working copy under
`/mnt/<models-disk>/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16` (24.45 GB) and
the conversion-workspace copy inside the 80 GiB
`ornith-mixedcal-v2-ext4.img` loop image (img deleted whole; only the
artifact and 2.8 MB of offload residue were inside; conversion evidence
retained in `results/2026/08/ornith15-mixedcal-v2-summary/`). Deletion manifests
with hashes: `results/2026/08/ornith15-ar-parity-20260822/{gptq-local,mixedcal-workspace}-deletion-manifest.txt`.
Recovery path: re-download from the public HF repo above. The only local
Ornith INT4 artifact is now the AutoRound candidate (ADR-0007).

Provisional promotion: **quality MixedCal-v2** (lower RTN, coherent A/B);
**speed parity at 150 W**; serve **MTP1**; optional local DraftINT4 overlay;
230 W if the operator wants the ~9.7k cold-input class.
