---
status: living
title: "Artifacts"
type: model-dossier
date: 2026-09-15
---

---
schema_version: 1
model_id: Qwen/Qwen3.6-35B-A3B
model_slug: qwen3.6-35b-a3b
document: artifacts
as_of: 2026-08-19
status: available_local
truth_verdict: CONFIRMED
evidence_tier: E2
primary_evidence:
  - ../../../archive/legacy-dirs-20260915/benchmarks/vllm-pi-realworld-and-long-context-2026-08-08.md
  - ../../../archive/legacy-docs-20260915/compat-stubs/recommended-configs.md
  - ../../../results/2026/08/vllm-mtp4-correctness-20260808T182243Z-157102/no-spec/results.json
---

# Artifacts

The campaign uses a Qwen3.6-35B-A3B GPTQ-INT4 target with the checkpoint's MTP adaptation preserved in BF16. The target and draft/head portions are one serving artifact contract: quantized target transformer weights plus explicit higher-precision exclusions for MTP.

The retained benchmark identifies the serving stack by immutable image digest `2c427ef…`, vLLM `0.26.1rc1.dev457+gc810e5ee9`, XPU kernels 0.1.12, and the boundary/nightly patch set. Reproduction must pin the full stack rather than relying on a floating image tag.

Artifact availability and local mount paths are operational state, not benchmark evidence. Before launch, verify the configured path exists, inspect the quantization metadata, and confirm that MTP tensors were preserved as intended. Do not substitute a similarly named GGUF or stripped-MTP artifact and carry over these vLLM results.
