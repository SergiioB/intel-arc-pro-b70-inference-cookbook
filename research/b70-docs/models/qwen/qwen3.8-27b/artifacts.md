---
status: living
title: "Artifacts"
type: model-dossier
date: 2026-09-15
---

---
schema_version: 1
model_id: Qwen/Qwen3.8-27B
model_slug: qwen3.8-27b
document: artifacts
as_of: 2026-08-19
status: validated_internal
truth_verdict: CONFIRMED
evidence_tier: E2
primary_evidence:
  - ../../../research/2026/qwen38-27b-day0-adoption-20260814.md
  - ../../../research/2026/qwen38-27b-drafter-campaign-20260817.md
  - ../../../research/2026/qwen38-cookbook-pr1-pr2-review-20260818.md
  - ../../../research/2026/qwen38-dflash2-smoke-20260819.md
---

# Artifacts

| Artifact | Quantization role | State |
|---|---|---|
| `Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` | Production **target weights**: GPTQ INT4 sym G128; native MTP tensors BF16 | Base vLLM target |
| S+M1 draft/head overlay | Requantizes the MTP head and LM head to INT4 | Optional experiment; not target conversion |
| Qwen3.8 DFlash2 draft | Separate diffusion-style draft for `method=dflash` | Research only |
| `unsloth/Qwen3.8-27B-GGUF` Q4_K_M | llama.cpp target; MTP stripped by converter | Day-0 llama.cpp route |
| `johannrplaster/Qwen3.8-27B-Uncensored-int4-AutoRound` | External AutoRound INT4 W4A16 g128 used by humble-b70-llm TP2 | Not a local serving artifact; different checkpoint from the GPTQ target. See [`research/2026/humble-b70-llm-dual-tp2-20260824.md`](../../../research/2026/humble-b70-llm-dual-tp2-20260824.md). |

The base GPTQ artifact contract records 400 qweight tensors, 15 BF16 MTP tensors, zero MTP leaks, and 19.56 GB.

The optional S+M1 overlay changes only draft/head storage: the MTP head is reported at approximately 0.85 GB BF16 and 0.22 GB INT4; the LM head at approximately 2.54 GB and 0.66 GB. These figures do not describe a second quantization of the already-GPTQ target transformer.

Canonical target: [SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16](https://huggingface.co/SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16).
