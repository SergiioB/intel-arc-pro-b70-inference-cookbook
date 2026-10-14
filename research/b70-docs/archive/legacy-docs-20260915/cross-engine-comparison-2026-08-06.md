# B70 Cross-Engine Comparison — All Engines, MoE + Dense (2026-08-06)

**Hardware:** Intel Arc Pro B70 32GB (32,656 MiB), Ryzen 7 5700X3D, 30GB RAM, 608 GB/s bandwidth
**Methodology:** Single-stream unless noted. Decode = engine rate. Prefill = pp2048+. All MEASURED, no estimates.

## Master Comparison Table — Single-Stream (150W unless noted)

| Engine | Model | Type | Quant | Decode t/s | Prefill t/s (p2k) | Max Ctx | Notes |
|--------|-------|------|-------|-----------|-------------------|---------|-------|
| **vLLM 0.21 XPU + MTP** | Qwen3.6-35B-A3B | MoE | GPTQ-Int4 | **126** | **7,535** | 128K | 4 patches, MTP ceiling |
| **vLLM 0.21 XPU (no spec)** | Qwen3.6-35B-A3B | MoE | GPTQ-Int4 | 72.6 | **9,094** | 128K | 230W; native int4 v4 |
| vLLM 0.21 XPU (no spec) | Qwen3.6-35B-A3B | MoE | GPTQ-Int4 | 65-68 | 7,1-8.4K | 128K | 150-180W |
| vLLM 0.17 XPU (Triton) | Qwen3.6-35B-A3B | MoE | GPTQ-Int4 | 57.9 | 5,349 | 32K | graphs ON, pre-patch |
| vLLM 0.17 XPU MXFP4 | Qwen3.6-35B-A3B | MoE | MXFP4 | 10.4 | 1,738 | 32K | 7 patches, not competitive |
| **llama.cpp b10255** | Qwen3.6-35B-A3B | MoE | Q4_K_XL | **72.6** | **2,128** | **512K** | Production-grade |
| llama.cpp b10255 | Qwen3.6-35B-A3B | MoE | Q5_K_M | ~68 | ~2,000 | 256K | Best quality |
| llama.cpp b10255 | Ornith-1.0-35B | MoE | Q5_K_M | 78.7 | 1,590 | 256K | Fastest decode |
| OVMS (OpenVINO) | Qwen3.6-35B-A3B | MoE | OpenVINO IR | 58.4 | — | — | Lower decode |
| — | — | — | — | — | — | — | — |
| **llama.cpp b10255** | ThinkingCap 27B | Dense | Q4_K_M+MTP | **29** | ~936 | 128K | MTP-4; only working dense |
| **llama.cpp b10255** | ThinkingCap 27B | Dense | Q6_K+MTP | ~27 | ~800 | 128K | Production quality |
| llama.cpp b10255 | ThinkingCap 27B | Dense | Q4_K_M (no MTP) | 21.3 | 936 | 256K | Base decode |
| vLLM XPU | ThinkingCap 27B | Dense | FP8 block | ❌ 0.75 | — | 2K | No XPU FP8 kernel |
| OVMS (OpenVino) | ThinkingCap 27B | Dense | OpenVino IR | 29.8 | — | — | Beats llama.cpp dense |
| SGLang XPU | — | — | — | ❌ | — | — | No XPU image; build OOMs |

## MoE vs Dense — Why the Engine Winner Flips

### MoE 35B-A3B (3B active params)
- Decode bottleneck: Memory bandwidth (~3GB/token from active experts)
- Winner: vLLM XPU + MTP (126 t/s) — 1.8x over llama.cpp
- vLLM advantage: Native int4 grouped-GEMM + MTP speculative + PIECEWISE graphs
- llama.cpp advantage: Simpler ops, no patches, 512K context (vLLM limited to 128K practical)

### Dense 27B (27B active params)
- Decode bottleneck: Memory bandwidth (~19GB/token, all weights)
- Winner: llama.cpp (29 t/s with MTP-4) — the ONLY working path
- vLLM failure: No block-FP8 XPU kernel. Dequant patch = 0.75 t/s (BF16 matmul tax)
- OVMS note: OVMS beats llama.cpp on dense decode (29.8 vs 20.8 base), but no MTP
- The gap: Until Intel ships native XPU block-FP8 scaled-MM, dense on vLLM is academic

## Multi-User (Concurrency) — Where vLLM Dominates

| Users | vLLM agg t/s | vLLM TPOT | llama.cpp agg t/s | llama.cpp TPOT |
|-------|-------------|-----------|-------------------|----------------|
| 1 | 64 | 20ms | 113 | 15ms |
| 4 | 225 | 41ms | 135 | 51ms |
| 8 | 424 | 60ms | 160 | 87ms |
| 16 | **694** | 88ms | 147 | 188ms |

vLLM continuous batching: aggregate scales near-linearly, latency stays flat.
llama.cpp --parallel: saturates at ~85-92 t/s, latency degrades linearly.
Crossover: vLLM wins aggregate at >=8 users. Below that, llama.cpp is faster per-user.

## Power Scaling — MoE and Dense Want Opposite Settings

| Power | MoE (vLLM) | MoE (llama.cpp) | Dense (llama.cpp) | Dense temp |
|-------|-----------|----------------|-------------------|------------|
| 150W | **126** | 69 | 22 | 71C |
| 180W | ~120 | ~71 | 27 | 75C |
| 230W | 115 ❌ | — | **30** | 79C |

MoE: self-limits to ~140W. Above 150W = zero gain or regression.
Dense: scales +30% from 150-230W but hits 79C. Sweet spot = 180W.

## Context Scaling — vLLM MTP + Prefix Caching

| Context | Prefill t/s | Decode t/s | Cold TTFT | Warm TTFT |
|---------|------------|------------|-----------|-----------|
| 4K | 5,423 | 121 | 0.7s | — |
| 20K | 7,325 | 116 | 2.6s | — |
| 65K | 4,418 | 105 | 14.3s | — |
| **128K** | **3,064** | **93** | **40s** | **1.4s** |

Warm TTFT (prefix caching) makes 128K interactive for multi-turn document chat.

## SGLang Status
- No XPU Docker image exists. Building from source compiles entire kernel stack.
- Docker build on B70 = cascading OOM (30GB RAM). OOM watchdog reboots machine.
- Not testable without >=64GB RAM host.

## Engine Maturity on Intel Arc XPU (2026-08-06)

| Engine | MoE | Dense | MTP/Spec | Multi-user | Maturity |
|--------|-----|-------|----------|------------|----------|
| **llama.cpp SYCL** | 72 t/s | 29 t/s (MTP) | MTP-4 | saturates | Production |
| **vLLM XPU 0.21** | 126 t/s (MTP) | 0.75 t/s | MTP (patched) | excellent | Near-prod (4 patches) |
| **OVMS/OpenVino** | 58 t/s | 30 t/s | none | unknown | Research |
| **SGLang XPU** | no image | no image | none | none | Not feasible |

## Sources
- llama.cpp: benchmark-history.md Runs 1-22
- vLLM: research/vllm-021-campaign-20260806.md (A0-A15)
- OVMS: docs/b70-engine-sweep-results-2026-07-05.md
- Concurrency: results/vllm-mxfp4-concurrency-20260805.json + llamacpp-concurrency-20260805.json
