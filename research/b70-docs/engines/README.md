---
title: "engines/ — per-engine living dossiers"
type: reference
status: living
date: 2026-09-15
updated: 2026-10-09
---

# engines/

One living dossier per inference engine on the host. Mirrors the
`models/<family>/<slug>/` pattern: current state, verified capabilities and
limits, hard rules, and links to dated evidence — numbers always trace to a
campaign under `campaigns/2026/` + `results/2026/<MM>/`.

| Engine | Dossier | One-line state (2026-10-09) |
|---|---|---|
| EXL3-XPU | [exl3xpu/README.md](exl3xpu/README.md) | **Primary Qwen3.8-27B route** (ADR-0010): trellis 4bpw + MTP3, 262K on one card, 63.25 tok/s p512/g128. The M=4 kernel probe program (RQ-01…RQ-04) is operator-stopped (RL-122) |
| Strata SYCL | [strata-sycl/README.md](strata-sycl/README.md) | **Primary Qwen3.8-Flash-Next route** (ADR-0011): IQ3_XXS GGUF + VRAM expert cache on one card, 48.03 tok/s decode, 989 tok/s 8K prefill; quality gates open |
| vLLM XPU | [vllm-xpu/README.md](vllm-xpu/README.md) | Nemotron GPTQ-INT4 + DFlash n7 route and FP8 diagnostic arms. The Qwen3.8-27B GPTQ route was removed (RL-087); TP2 gated (ADR-0008) |
| llama.cpp SYCL | [llamacpp-sycl/README.md](llamacpp-sycl/README.md) | GGUF workhorse (fleet KV `q8_0` K + `q4_1` V); local tree ~2 months behind master (misses TILE-KV, grouped-MoE XMX, MXFP4 MoE work; RL-126) |
| OpenVINO GenAI | [openvino-genai/README.md](openvino-genai/README.md) | Alternate Qwen3.8-27B route: INT4 clean at 28.5 tok/s p512 @150 W (RL-086 correction); ≥196K is per-generation memory growth (RL-033) |
| SGLang XPU | [sglang-xpu/README.md](sglang-xpu/README.md) | Exploratory; no validated serving route yet |
| PyTorch XPU | [torch-xpu/README.md](torch-xpu/README.md) | Kernel-repro/profiling substrate, not a serving engine |
| Cascadia | [cascadia/README.md](cascadia/README.md) | Separate engine (Rust server + IR surgery); PP2 −0.26 %, RPC split −46 % (RL-051, RL-083) |

Cross-engine support summary: `docs/reference/engine-support-matrix.md`.
Which format and engine to pick per workload regime: `docs/reference/b70-format-and-lever-guide.md`.
Engine comparison evidence lives in campaigns (e.g. the 2026-09-15 INT8-class
Qwen3.8-27B sweep, `results/2026/09/qw38-ov-mtp-20260914/`).
