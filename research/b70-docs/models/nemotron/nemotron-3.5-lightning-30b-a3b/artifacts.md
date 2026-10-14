---
status: living
title: "Artifacts"
type: model-dossier
date: 2026-09-15
---

---
schema_version: 1
model_id: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B
model_slug: nemotron-3.5-lightning-30b-a3b
document: artifacts
as_of: 2026-08-19
status: mixed_availability
truth_verdict: CONFIRMED
evidence_tier: E2
primary_evidence:
  - ../../../archive/legacy-docs-20260915/compat-stubs/recommended-configs.md
  - ../../../research/2026/nemotron-lightning-optimization-20260811/08-q4_0-breakthrough.md
  - ../../../results/2026/08/nemotron-q4_0-test/
  - ../../../results/2026/08/nemotron-q4_0-mtp4-230w/
  - ../../../results/2026/08/nemotron-dflash-bf16-n7-n5-20260813T082203Z-1407994/manifest.txt
---

# Artifacts

## Current vLLM pair

| Role | Artifact | State |
|---|---|---|
| Target | GPTQ-INT4 G64 symmetric, local path recorded as `/mnt/<models-disk>/nemotron-lightning-gptq-sym64-20260812` | Current target route |
| Draft | `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-DFlash-BF16`, reconstructed from official NVFP4 | Current DFlash route |

Published repository locations recorded by the campaign:

- [GPTQ target on Hugging Face](https://huggingface.co/SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym)
- [BF16 DFlash draft on Hugging Face](https://huggingface.co/SergiioB/Nemotron-3.5-Lightning-30B-A3B-DFlash-BF16)

## Historical llama.cpp Q4_0

The historical target was `/mnt/<models-disk>/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf`, requantized from Q4_K_M with `--allow-requantize`. It was deleted during storage cleanup. Requantization from an already quantized source can add quality loss.

Correct retained evidence links:

- Q4_0 conversion and no-spec test evidence (`../../../results/2026/08/nemotron-q4_0-test/`)
- Q4_0 MTP4 230 W evidence (`../../../results/2026/08/nemotron-q4_0-mtp4-230w/`)

These directories prove historical measurements. They do not restore the deleted GGUF and must not be presented as a current launch path. Model-weight Q4_0 is also distinct from KV-cache `q4_0 K`; the latter is not an approved quality configuration.
