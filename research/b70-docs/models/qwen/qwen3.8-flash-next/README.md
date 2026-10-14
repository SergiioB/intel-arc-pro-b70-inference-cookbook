---
status: living
title: "Qwen3.8-Flash-Next"
type: model-dossier
date: 2026-09-01
---

---
schema_version: 1
model_id: Qwen/Qwen3.8-Flash-Next
model_slug: qwen3.8-flash-next
document: overview
as_of: 2026-09-01
status: research_foothold
truth_verdict: PARTIAL
evidence_tier: E2
publication_gate: PROVISIONAL_NOT_FOR_PUBLIC_HEADLINE
engines: [llama.cpp-sycl]
primary_evidence:
  - ../../../results/2026/08/qwen38-flash-next-dual-b70-20260831T130711Z/HANDOVER.md
  - ../../../results/2026/08/qwen38-flash-next-dual-b70-20260831T130711Z/OVERNIGHT-20260901.md
  - ../../../results/2026/08/qwen38-flash-next-dual-b70-20260831T130711Z/KERNEL-ACCEPTANCE-20260901.md
  - ../../../results/2026/08/qwen38-flash-next-dual-b70-20260831T130711Z/GPTQ-ABI-FREEZE-20260901.md
  - ../../../results/2026/08/qwen38-flash-next-dual-b70-20260831T130711Z/CONVERSION-GATES-20260901.md
  - ../../../results/2026/08/qwen38-flash-next-dual-b70-20260831T130711Z/N5-CONFIRM-20260901.md
  - ../../../results/2026/08/qwen38-flash-next-dual-b70-20260831T130711Z/GPU-GROUP-20260901.md
  - ../../../results/2026/08/qwen38-flash-next-dual-b70-20260831T130711Z/COOKBOOK-DRAFT-QWEN38-FLASH-NEXT.md
  - ../../../research/2026/qwen38-flash-next-b70-adoption-20260826.md
---

# Qwen3.8-Flash-Next

Official checkpoint: `Qwen/Qwen3.8-Flash-Next` (`qwen4_exp`, ~180B tensors, 131 BF16 shards). Live HF HEAD observed 2026-09-01: `de4b8e4d43b917e7706784d8bb445c9af86a3540`. Skill pin `f5d08274…` is stale for downloads.

**Runnable B70 route today:** dual-card llama.cpp SYCL on the community AtomicChat AD-4.27bpw Q4_K_M M64 GGUF. That is a correctness/capacity foothold, not a vLLM serving recipe.

**Not the route:** AutoRound W4A16 GPTQ → vLLM-XPU. Conversion is stopped (`blocked_artifact_or_storage`, `blocked_missing_architecture_support` for QSA, no official shards locally). Do not requantize the GGUF.

Do not mix these numbers with Qwen3.8-27B GPTQ/MTP or with vLLM client post-first.

See [adoption](adoption.md), [artifacts](artifacts.md), [serving](serving.md), [timeline](timeline.md).
