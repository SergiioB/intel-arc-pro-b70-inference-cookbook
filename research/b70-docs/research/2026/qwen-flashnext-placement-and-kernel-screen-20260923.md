---
title: "Qwen3.8 kernel screen and Flash-Next placement correction"
type: research
date: 2026-09-23
status: provisional
---

# Qwen3.8 kernel screen and Flash-Next placement correction

> **PROVISIONAL — NOT FOR PUBLIC HEADLINE.** This note closes two separate investigations. The Qwen3.8-27B kernel dispatch screen did not promote. The Flash-Next placement discriminator loaded and passed a three-prompt smoke, but no prefill/decode timing was measured.

## Qwen3.8-27B: fused QK-RoPE dispatch

The local treatment enabled the existing `fused_qk_rmsnorm_rope_gate` Triton path in the pinned champion generation. The corresponding XPU dispatch is already upstream in vLLM #56096; this was a local old-image test, not a novel upstream fix.

Corrected post-warmup analysis uses n=6 per arm. At p512/g128, client post-first median was 82.58 tok/s baseline vs 73.52 treatment (-10.97%). At p8192/g128 it was 70.54 vs 78.78 (+11.68%). Both sample ranges overlap. The p2048/g1 cold-input control was -0.73%. MTP acceptance changed by -0.0663 and +0.102 at the two decode cells, beyond the predeclared ±0.03 guard. Greedy canaries passed within arm, but logs do not trace actual fused-kernel dispatch.

The campaign’s temperature gate was invalid: package readings were about 69–78°C while its helper accepted stable temperatures above the protocol maximum of 55°C. Treat the screen as diagnostic/no-promotion evidence only. Do not call it a speed gain, do not file a duplicate #53989 dispatch PR, and do not reuse the built-in summary that counted warmups. Use the corrected analysis:
`results/2026/09/qkrope53989-abba-20260923/screen/RESULTS.md` (`../../results/2026/09/qkrope53989-abba-20260923/screen/RESULTS.md`) and `comparison-authoritative.json` (`../../results/2026/09/qkrope53989-abba-20260923/screen/comparison-authoritative.json`).

## Flash-Next AP-Q4_K_XL: placement, not a demonstrated loader fix

The AP-Q4_K_XL GGUF has 1,224 tensors, 48 repeating blocks, 63.28125 GiB of MoE expert tensors, and a 26.8222 GiB PLE table. Mirroring the pinned llama.cpp `-ts 49,51` layer-split source (`src/llama-model.cpp:1467-1492`) assigns blocks 0–24 to GPU0 and blocks 25–47 plus output to GPU1. With PLE on CPU but expert weights left on the devices, the header-derived weight totals are ~34.59 GiB on GPU0 and ~32.28 GiB on GPU1, each above the ~31.85 GiB visible card budget before runtime allocations. Previous overrides only moved early-layer experts, which did not relieve the GPU1 tail.

A one-shot placement discriminator kept PLE on CPU and moved expert tensors from blocks 0–5 and 42–47 to CPU. This estimates ~26.68 GiB of weights on GPU0 and ~24.38 GiB on GPU1 before KV/compute buffers. With the matched oneAPI 2026.1.1 build, server health was reached in ~90 seconds; post-load reserve was 3,372 MiB / 7,050 MiB. A separate serialized correctness smoke returned the expected `B70-FLASH-OK`, `703`, and `Paris`. No allocator/device-loss/segfault error appeared in the server log. Cleanup returned to ~32,051 / 32,615 MiB free VRAM and no process/container.

Conclusion: corrected placement is **load-feasible and bounded-smoke coherent** on this artifact/build. The earlier six failures used materially overfilled placement and therefore did not establish a source-code or driver bug. This does not establish broad quality, reference parity, MTP operation, performance, or recipe superiority. CPU expert/PLE offload may cost throughput; that impact is unmeasured. Evidence: `load probe` (`../../results/2026/09/flashnext-ap-placement-smoke-20260923T113433Z/`), `correctness smoke` (`../../results/2026/09/flashnext-ap-placement-correctness-20260923T114633Z/`).

## Next gate

A controlled Flash-Next performance screen remains unrun. Before it, both package sensors must be <=55°C, both GPUs must pass the empty-process/container/VRAM gates, and the corrected placement must retain the required post-load reserve. Use the existing cookbook/campaign protocol for a same-shape warmup, one-client cold-input and decode cells, cache guards, telemetry, and cleanup. Do not start MTP pairing, a broad sweep, or an upstream allocation report until this first matched placement screen identifies a reproducible problem or gain. No GPU test was run after the smoke because both packages rose to 57°C.
