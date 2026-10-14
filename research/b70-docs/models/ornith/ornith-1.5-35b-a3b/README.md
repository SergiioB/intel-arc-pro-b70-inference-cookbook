---
status: living
title: "Ornith 1.5 35B-A3B"
type: model-dossier
date: 2026-08-24
---

---
schema_version: 1
model_id: ornith-ai/Ornith-1.5-35B-A3B
model_slug: ornith-1.5-35b-a3b
document: overview
as_of: 2026-08-22
status: self_reported
truth_verdict: PARTIAL
evidence_tier: E2
publication_gate: SELF_REPORTED_WITH_LIMITS
engines: [vllm-xpu]
primary_evidence:
  - ../../../research/2026/ornith15-35b-a3b-adoption-20260819.md
  - ../../../results/2026/08/ornith15-mixedcal-v2-summary/summary.json
  - ../../../results/2026/08/ornith15-n5-maxkv-20260821T054323Z/
  - ../../../results/2026/08/ornith15-n5-maxkv-resume-20260821T062400Z/
  - ../../../results/2026/08/ornith15-boundary-fixup-20260821T072410Z/
  - ../../../results/2026/08/ornith15-mtpdepth-power-20260821T075858Z/
  - ../../../results/2026/08/ornith15-draftint4-20260821T084132Z/
  - ../../../results/2026/08/ornith15-lmx-mixedcal-20260821T090450Z/
  - ../../../results/2026/08/ornith15-lmx-prefill230-20260821T095259Z/
  - ../../../research/2026/ornith15-autoround-vs-gptq-20260822.md
  - ../../../results/2026/08/ornith15-ar-headtohead-20260822T0535Z/
  - ../../../results/2026/08/ornith15-ar-parity-20260822/
  - ../../../decisions/adr-0007-ornith-autoround-w4a16-candidate.md
---

# Ornith 1.5 35B-A3B


> **Route closed — 2026-08-24 (user decision: quality rejected).** All local
> Ornith artifacts were removed: source shards (`/mnt/<models-disk>/Ornith-1.5-35B-A3B-shards`),
> AutoRound W4A16 artifact + conversion workspaces (including the 80G loop image,
> `ar-artifacts`, `ornith15-quant`), the Ornith 1.0 mmproj GGUF, the HF cache stub,
> and the `~/bench-ornith-vs-qw38` repo. The `bench-report.service` was stopped and
> disabled; the autoround loop mount and its fstab entry were torn down.
> Measured evidence is retained under `results/ornith15-*` and
> `campaigns/2026/ornith15-*`. The HF-public MixedCal-v2 repo remains re-downloadable.
> Removal log: `results/2026/08/cleanup-ornith-removal-20260824T121500Z/`.

Ornith 1.5 is a `qwen3_5_moe` hybrid GDN MoE with the same measured topology as Qwen3.6-35B-A3B: 40 layers, 256 experts with 8 active, and one MTP layer. The day-0 B70 route is a locally produced GPTQ-INT4 symmetric group-128 target with its MTP tensors retained in BF16.

MixedCal-v2 is the current research artifact: same GPTQ INT4 G128 experts-only contract as the original WikiText GPTQ, mixed-domain calibration, RTN fallback 10.37% vs 24.76%. n=5 confirmation at 150 W is **speed parity**. MTP1 remains the research speculation default. Exact-token capacity completed through 262,016+g128 no-spec. MTP2/MTP4 n=5, paired 150↔230 W prefill A/B, and DraftINT4 S+M1 n=5 are measured. Hugging Face:
`SergiioB/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2`. LocalMaxxing MixedCal MTP1 @150 W: 94.1 tok/s n=3 p32/g256. LocalMaxxing no-spec @230 W long-prompt: `tokSPrefill` **9556.4** (2899 tokens, n=5). Both validate-local valid, **not submitted**.

## Current dossier state

| Area | State |
|---|---|
| Release | Complete official BF16/FP8/NVFP4/GGUF/MLX families; no official GPTQ/AWQ found in the recorded release audit |
| B70 artifact | **AutoRound W4A16 candidate is the only local INT4 artifact** (workspace loop, contract 30720/0/785, zero RTN fallback, reference parity passed — ADR-0007). GPTQ MixedCal-v2 is HF-public only (local copies deleted 2026-08-22, re-downloadable); BF16 source + shards retained immutable |
| Serve | vLLM XPU `f01e24f6` nightly, WNA16, cache off, language-model-only |
| Speculation | MTP1 selected. MTP2 mid; MTP4 slower than no-spec. Optional DraftINT4 overlay screened 106.27 p512 vs 96.43 BF16 draft |
| Performance | n=5 no-spec instance-median MixedCal p512/g128 **70.74** / MTP1 **96.43** tok/s @150 W C1. Prefill lever is 230 W (LMX `tokSPrefill` **9556.4** at 2899 tokens). Self-reported E2 |
| Quality | AutoRound candidate: reference logprob parity vs BF16 measured — equal-or-best vs GPTQ on mean Δ / top-1 / KL (315 positions); zero RTN fallback. Task-quality suite still not run for any artifact |
| Capacity | Exact p65408, p130944, p262016 completions recorded |

See [adoption](adoption.md), [artifacts](artifacts.md), [MixedCal-v2 conversion method and status](mixedcal-v2-conversion.md), [serving](serving.md), [quality](quality.md), and [timeline](timeline.md).
