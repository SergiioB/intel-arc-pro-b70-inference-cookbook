---
status: living
title: "Adoption"
type: model-dossier
date: 2026-09-15
---

---
schema_version: 1
model_id: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B
model_slug: nemotron-3.5-lightning-30b-a3b
document: adoption
as_of: 2026-08-19
status: validated_internal
truth_verdict: PARTIAL
evidence_tier: E2
publication_gate: SELF_REPORTED_NOT_INDEPENDENTLY_REPRODUCED
primary_evidence:
  - ../../../research/2026/nemotron-lightning-optimization-20260811/10-vllm-gptq-readiness-20260812.md
  - ../../../research/2026/nemotron-dflash-xpu-first-proof-20260813.md
  - ../../../results/2026/08/nemotron-dflash-bf16-n7-n5-20260813T082203Z-1407994/manifest.txt
---

# Adoption

## Decision

Use the current vLLM XPU GPTQ+DFlash route for active work. Treat the llama.cpp Q4_0 work as historical evidence only.

## Completed gates

- Local GPTQ-INT4 G64 symmetric target converted and loaded through the XPU GPTQ path.
- Separate DFlash draft reconstructed from official NVFP4 material into BF16 and loaded with `method=dflash`.
- Isolated C1 n=5 speed card completed with prefix caching off and retained raw evidence.
- Deterministic raw-completion replay matched within the measured smoke.
- Staged capacity requests completed through 119,904 total tokens at `--max-model-len 120000`.

## Limits

- Both target and draft are derived artifacts; speed is not official-quant parity.
- Native MTP showed 0% acceptance in the recorded work and is not the adopted speculation route.
- The 119,904-token completion is not a 128K result.
- Independent reproduction and task-quality/reference parity remain open.
