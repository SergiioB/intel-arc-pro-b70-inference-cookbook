---
status: living
title: "Nemotron 3.5 Lightning 30B-A3B"
type: model-dossier
date: 2026-09-15
---

---
schema_version: 1
model_id: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B
model_slug: nemotron-3.5-lightning-30b-a3b
document: overview
as_of: 2026-08-19
status: validated_internal
truth_verdict: PARTIAL
evidence_tier: E2
publication_gate: SELF_REPORTED_NOT_INDEPENDENTLY_REPRODUCED
engines: [vllm-xpu, llama.cpp-sycl]
primary_evidence:
  - ../../../research/2026/nemotron-session-summary-20260813.md
  - ../../../research/2026/nemotron-dflash-xpu-first-proof-20260813.md
  - ../../../results/2026/08/nemotron-dflash-bf16-n7-n5-20260813T082203Z-1407994/n5-summary.json
---

# Nemotron 3.5 Lightning 30B-A3B

The current B70 route is **vLLM XPU with a GPTQ-INT4 symmetric group-64 target and a separate BF16 DFlash draft**. It is not the historical llama.cpp Q4_0 route.

The isolated C1 n=5 campaign at a configured 150 W cap measured 186.61 tok/s at p2048/g128 and 157.92 tok/s at p8192/g128 using client-observed post-first timing. Cold input at p8192/g1 was 7,160.1 prompt tokens/client TTFT second; that is not isolated engine prefill. DFlash window acceptance was 52.0%.

The older locally requantized Q4_0 GGUF produced useful llama.cpp evidence, but that file was deleted during storage cleanup and is **not a currently serveable artifact**. See [artifacts](artifacts.md) and [serving](serving.md) for the separation.

Additional dossier pages: [adoption](adoption.md), [quality](quality.md), and [timeline](timeline.md).
