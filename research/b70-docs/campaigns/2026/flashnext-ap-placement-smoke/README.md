---
status: validated_internal
title: "Flash-Next AP-Q4_K_XL corrected placement load and smoke"
type: campaign
date: 2026-09-23
---

---
id: CAM-2026-FLASHNEXT-AP-PLACEMENT-SMOKE
status: validated_internal
---

# Flash-Next AP-Q4_K_XL corrected placement load and smoke

- Model artifact: Signal-3.8-Flash-Next AP-Q4_K_XL GGUF (`qwen4exp`), 94.2 GiB.
- Engine: local qwen4exp-capable llama.cpp SYCL tree, matched oneAPI 2026.1.1 build, dual B70 layer split 49/51.
- Changed variable: placement only. PLE plus expert tensors from blocks 0–5 and 42–47 were assigned to CPU; remaining layers followed the recorded 49/51 split.
- Header/source inventory: before this placement, weight bytes alone were approximately 34.59 GiB on GPU0 and 32.28 GiB on GPU1, above the ~31.85 GiB visible capacity of either card. Corrected estimated device weights are 26.68/24.38 GiB before KV and compute buffers.
- Load-only evidence: `results/2026/09/flashnext-ap-placement-smoke-20260923T113433Z/` (`../../../results/2026/09/flashnext-ap-placement-smoke-20260923T113433Z/`); server reached `/health` in about 90 seconds with 3372/7050 MiB free.
- Bounded correctness evidence: `results/2026/09/flashnext-ap-placement-correctness-20260923T114633Z/` (`../../../results/2026/09/flashnext-ap-placement-correctness-20260923T114633Z/`); exact-token `B70-FLASH-OK`, arithmetic `703`, and `Paris` passed; no DEVICE_LOST/OOM/segmentation signatures were found.
- Cleanup: 32051/32615 MiB free, no server/container remained, caps stayed at the found 180/230 W.
- Limits: **load-feasibility and three fixed smoke prompts only**. No prefill/decode timing, MTP, broad task-quality, or cookbook-recipe speed comparison was run. The CPU PLE/expert placement changes the recipe and its throughput cost is unknown. Performance follow-up requires both cards <=55 C and empty process/container/VRAM gates.
