---
title: "KV Cache Quantization Analysis"
type: research
date: 2026-09-15
status: final
---

# KV Cache Quantization Analysis

## Summary

The B70 documentation mandates q8_0-q8_0 KV cache as "required." External benchmarks
(Anbeeld 2026) show this is overkill. q5_0-q4_1 asymmetric quantization achieves
92.65% of bf16 tail precision while using 38% less VRAM than q8_0-q8_0.

## How KV Cache Quantization Works

llama.cpp quantizes key and value tensors in the attention cache independently.
Supported types on b9851: f32, f16, bf16, q8_0, q5_0, q5_1, q4_0, q4_1, iq4_nl.

Each type uses block-wise scalar quantization with block size 32:

| Type | Bits/value | Scale bits | Effective bits | Levels | SNR (uniform) |
|------|-----------|------------|----------------|--------|---------------|
| f16 | 16 | 0 | 16.0 | 65536 | 98.1 dB |
| q8_0 | 8 | 16 | 8.5 | 256 | 49.9 dB |
| q6_0 | 6 | 16 | 6.5 | 64 | 37.9 dB |
| q5_0 | 5 | 32 | 6.0 | 32 | 31.9 dB |
| q4_0 | 4 | 16 | 4.5 | 16 | 25.8 dB |

## Memory Scaling

KV cache VRAM scales linearly with context length and proportionally to effective bits.

Relative to bf16 (baseline = 100%):

| KV Config | % of bf16 | vs q8_0-q8_0 | 99.9% Tail Precision |
|-----------|-----------|--------------|----------------------|
| bf16 / bf16 | 100.0% | 1.88x | 100.00% |
| q8_0 / q8_0 | 53.1% | 1.00x (baseline) | 94.61% |
| q8_0 / q5_1 | 45.3% | 0.85x | 94.21% |
| q6_0 / q5_0 | 37.5% | 0.71x | 93.29% |
| **q5_0 / q5_0** | **34.4%** | **0.65x** | **92.70%** |
| **q5_0 / q4_1** | **32.8%** | **0.62x** | **92.65%** |
| q5_0 / q4_0 | 31.3% | 0.59x | 91.39% |
| q4_0 / q4_0 | 28.1% | 0.53x | 89.84% |

## Asymmetric K/V: Why It Works

The attention mechanism is asymmetric: keys determine *which* tokens to attend to
(via Q·K^T), while values determine *what* information to propagate (via w·V).

Key precision matters more because:
1. Key errors affect attention weight computation (softmax of Q·K^T/√d)
2. Small per-element errors in K get amplified by the dot product
3. V errors affect the weighted sum output, which averages over many elements

The Anbeeld data confirms: q5_0-q4_1 (q5 K, q4 V) performs virtually identically
to q5_0-q5_0 (q5 K, q5 V) — 92.65% vs 92.70% tail precision. The extra bit on V
buys almost nothing.

## The Quality Cliff

q4_0-q4_0 sits at 89.84% tail precision — a 32% increase in tail KL divergence
vs q5_0-q5_0. This is the threshold where:
- Tool calls begin to fail (wrong JSON structure, hallucinated parameters)
- Code generation quality drops
- Long-context retrieval degrades

The community consensus ("night and day" for q4_0) aligns with this cliff.

## B70-Specific Implications

### Current state (q8_0-q8_0 KV):
- Qwen 27B MTP Q5: 19GB + 13GB KV = 32GB at 128K (TIGHT, borderline OOM)
- Qwen 35B Q5: 25GB + 5GB KV = 30GB at 128K (1.9GB free)

### With q5_0-q4_1 KV (0.62x current KV):
- Qwen 27B MTP Q5: 19GB + 8.0GB KV = 27.0GB at 128K (comfortable!)
- Qwen 27B MTP Q5: 19GB + 12.5GB KV = 31.5GB at 200K (fits!)
- Qwen 35B Q5: 25GB + 3.1GB KV = 28.1GB at 128K (3.8GB free!)
- Qwen 35B Q5: 25GB + 6.2GB KV = 31.2GB at 256K (fits!)

This doubles the usable context for our two flagship models at negligible quality cost.

## Recommendation

**Change the mandatory KV cache from q8_0-q8_0 to q5_0-q4_1.**

The quality trade: -1.96% tail precision for +100% context on Qwen 35B Q5_K_M
and +72K headroom on Qwen 27B MTP. This is the correct trade for real-world use.

Conservative users can use q6_0-q5_0 (93.29% tail, 37.5% size) as a middle ground.
