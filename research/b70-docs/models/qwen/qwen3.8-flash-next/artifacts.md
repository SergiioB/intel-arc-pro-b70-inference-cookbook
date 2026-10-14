---
status: living
title: "Artifacts"
type: model-dossier
date: 2026-09-01
---

---
schema_version: 1
model_id: Qwen/Qwen3.8-Flash-Next
model_slug: qwen3.8-flash-next
document: artifacts
as_of: 2026-09-01
status: research_foothold
truth_verdict: CONFIRMED
evidence_tier: E2
---

# Artifacts

| Artifact | Role | State |
|---|---|---|
| `Qwen/Qwen3.8-Flash-Next` BF16, 131 shards, ~360 GB | Official source for a future AutoRound/vLLM path | **Not local.** HEAD `de4b8e4d43b917e7706784d8bb445c9af86a3540` (API 2026-09-01). Metadata-only older pin `f5d08274…` in `results/2026/08/qwen38-flash-next-adoption-20260826T133339Z/metadata/` |
| `AtomicChat/Qwen3.8-Flash-Next-GGUF` `Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64` rev `142262902a46f7daed19c79d0771534c8106ad59` | llama.cpp target | **Local.** 33 shards, 94,525,394,976 B / 88.03 GiB. Mixed: gate/up **IQ3_S**, down **IQ4_NL**, PLE `per_layer_token_embd` **Q5_1**. Headline Q4_K_M is mixed. No usable MTP tensors |
| AutoRound W4A16 GPTQ (intended) | vLLM-XPU serving | **Not built.** ABI freeze: `results/2026/08/qwen38-flash-next-dual-b70-20260831T130711Z/GPTQ-ABI-FREEZE-20260901.md` |

Community GGUF is not official-lineage GPTQ. Do not treat IQ3_S/IQ4_NL as group-128 INT4.
