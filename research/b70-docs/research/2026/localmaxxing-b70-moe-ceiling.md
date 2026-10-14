---
title: "Localmaxxing B70: The 244 tok/s MoE Ceiling"
date: 2026-09-30
status: verified
---

# Localmaxxing the B70: 244.5 tok/s Ceiling

The Intel Arc Pro B70 hits a hard pin bandwidth ceiling at 608 GB/s. Dense 27B models max out at ~95 tok/s because they must read ~13.5 GB of weights per token.

To break 100 tok/s on a single B70, we must read fewer bytes. Mixture of Experts (MoE) reduces active weights to ~1.5–2 GB per token.

## The Max Speed Record (2026-09-30)

**Hardware:** 1x Intel Arc Pro B70 32GB (GPU1, 200W cap, 2400 MHz limit).
**Champion Model:** NVIDIA-Nemotron-3.5-Lightning-30B-A3B (GPTQ-INT4 G64 + BF16 DFlash n7 speculation).
**Baseline Model:** Qwen3.8-27B (EXL3 4bpw + MTP3 speculation).

### 1. Raw Speed (p512/g128, greedy)
- **MoE (Nemotron):** 244.5 tok/s (median, n=6)
- **Dense (Qwen3.8):** 95.1 tok/s (median, n=3)
- **Result:** 2.57x gain.

### 2. Real Tasks (Natural EOS, task sampling)
- **MoE (Nemotron):** 95.2 tok/s (ttft: 0.05s)
- **Dense (Qwen3.8):** 61.1 tok/s (ttft: 0.17s)
- **Result:** 1.56x gain.

### 3. Aggregate Throughput (p2048/g128)
- **MoE C16 (16 concurrent streams):** 291.2 tok/s total (~63.3 tok/s per stream).

## The Bottleneck Shift
At 244.5 tok/s, the bottleneck is no longer memory bandwidth. It is instruction dispatch. The K216 decode pipeline requires 84+ host synchronizations per token. The hardware has the compute (363 TOPS INT8), but the CPU/driver overhead limits further scaling. To push past 250 tok/s, we need a fused grouped dequant GEMM and an on-device router.

**Verdict:** Nemotron-3.5-Lightning-30B is the local maximum for single-card B70 performance today.
