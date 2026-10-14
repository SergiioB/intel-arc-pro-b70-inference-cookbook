---
status: living
title: "Adoption"
type: model-dossier
date: 2026-09-01
---

---
schema_version: 1
model_id: Qwen/Qwen3.8-Flash-Next
model_slug: qwen3.8-flash-next
document: adoption
as_of: 2026-09-01
status: research_foothold
truth_verdict: PARTIAL
evidence_tier: E2
---

# Adoption

Skill: `.agents/skills/qwen38-flash-next-adoption/SKILL.md`.

## Outcomes (2026-09-01)

- `official_tensor_inventory_complete` (index names + API totals; per-tensor headers still E0/E1)
- `native_llamacpp_correctness_foothold` / `correctness_smoke_tested` (`B70-FLASH-OK` / `703` / `Paris`)
- `kernel_shape_dispatch_ok_not_model_class` — synthetic `XpuFusedMoe` H=2560 I=640 G=128 topk=10 on `b70-pinned:f01e24f6`
- `blocked_missing_architecture_support` — no `vllm.models.qwen4_exp` in the pinned image; overlay QSA ctor is `NotImplementedError`
- `blocked_missing_xpu_kernel` — QSA sparse attention; packed W2 MoE
- `blocked_artifact_or_storage` — no official shards; `/` 2.1G; models2 13G
- `autoround_exact_model_support_missing`
- `blocked_capacity_or_bandwidth` — W4 experts-first dual residency vs mixed-serving ceiling
- Not `native_vllm_model_class_ready`, not `autoround_slice_passed`, not `publication_ready`

- `production_candidate_pending_n5` **cleared for the two leader cells** (n=5 fused-FP32 c8 p512 decode 23.38; n=5 fused-F16 c16 p9096 cold input 594.49). Ordinary ≤55 °C loaded-idle was not reachable on GPU1 pkg.

## What was not done

Download of 360 GB, AutoRound, GGUF requant, native MTP, grouped IQ3_S/IQ4_NL GEMM (SYCL MMQ compile-disabled; F16+XMX already owns large-ne12). GPU ID grouping is in (`+0.81%` prefill n=3). No public cookbook/XeCores publish.
