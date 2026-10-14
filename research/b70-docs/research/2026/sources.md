---
title: "Research Sources"
type: research
date: 2026-09-15
status: final
---

# Research Sources

## Primary Source: Anbeeld KV Cache Quantization Benchmarks

**URL**: https://anbeeld.com/articles/kv-cache-quantization-benchmarks-for-long-context

**Date**: 2026 (exact date TBD, published after llama.cpp rotation feature)

**What was tested**:
- Model: Qwen 3.6 27B (same architecture as our Qwen 27B MTP)
- Weight quants: Q5_K_S and IQ4_XS
- Context lengths: 64K and 128K
- Hardware: RTX 3090 (24GB), Ryzen 7 5700X3D
- Backend: BeeLlama.cpp v0.1.2 (llama.cpp fork with DFlash + TurboQuant/TCQ)
- Metrics: Perplexity (Wikitext), KL Divergence (mean, 90%, 95%, 99%, 99.9%), prefill throughput

**Key findings applied to B70**:

1. **q5_0-q4_1 is the practical sweet spot** (32.8% of bf16 KV, 92.65% 99.9% tail precision)
   - Nearly identical quality to symmetric q5_0-q5_0 (92.70%) at 4.7% less VRAM
   - The asymmetric allocation (q5 K, q4 V) is nearly free — K precision dominates tail quality

2. **q8_0-q8_0 is overkill** (53.1% of bf16 KV, 94.61% tail precision)
   - Only +1.96% tail precision vs q5_0-q4_1 for 62% more VRAM
   - The asymmetric q8_0-q5_1 (45.3%, 94.21%) gets nearly the same quality

3. **q4_0-q4_0 drops to 89.84%** tail precision
   - 32% increase in tail KL divergence vs q5_0
   - This is where tool calls, JSON, and code generation start breaking

4. **Weight quant matters**: Lower weight quants (IQ4_XS) tolerate more KV quantization
   - Our Q5_K_M models need MORE careful KV quantization
   - Our Q4_K_M models can tolerate more aggressive settings

5. **The preset ladder** (from Anbeeld §8):
   - q8_0-q5_1 (45.3%, 94.21%): Best precision/size at high end
   - q5_0-q5_0 (34.4%, 92.70%): Good precision/size, relatively safe
   - q5_0-q4_1 (32.8%, 92.65%): Best default if VRAM-constrained
   - q4_0-q4_0 (28.1%, 89.84%): Memory saving with measurable precision loss

6. **q6_0 follow-up** (§12-13): q6_0-q5_0 at 37.5% gives 93.29% tail precision
   - Better than q5_0-q5_0 for only +3.1% size
   - q6_0-q5_1 at 39.1% gives 93.50% — the fidelity tier

## Secondary Source: Reddit r/LocalLLaMA Discussion

**URL**: https://www.reddit.com/r/LocalLLaMA/comments/1tp9d1w/kv_cache_quant_benchmarks_q5_q6_are_underrated/

**Title**: "KV cache quant benchmarks — q5, q6 are underrated"

**Key community consensus**:
- q5 and q6 KV quants are underappreciated
- Most users default to q8_0 out of caution, but the quality gap is much smaller than assumed
- Asymmetric K/V allocation is the most underused optimization

## Tertiary Sources

### NVIDIA DGX Spark KV Cache Benchmarks
**URL**: https://forums.developer.nvidia.com/t/kv-cache-quantization-benchmarks-on-dgx-spark-q4-0-vs-q8-0-vs-f16-llama-cpp-nemotron-30b-128k-context/365138

- Tested Nemotron-30B on DGX Spark with 128K context
- Confirmed q8_0 is near-lossless, q4_0 has measurable degradation
- Independent verification of the q4_0 quality cliff

### llama.cpp GitHub Issue #21295
**URL**: https://github.com/ggml-org/llama.cpp/issues/21295

- q5_0 KV performance bug report (slower than q4_0 and q8_0 on Vulkan)
- Confirms q5_0 is a supported but less-tested KV cache type

## Weight Quantization Quality Sources

### Reddit r/LocalLLaMA: Qwen 3.6 27B Quantization Benchmark

**URL**: https://www.reddit.com/r/LocalLLaMA/comments/1tr9vzn/qwen3627b_quantization_benchmark/

**Date**: ~June 2026

**Key findings**:
- Community uses llama-perplexity KL divergence against BF16 baseline
- Q6-Q8 range: near-lossless (KLD < 0.04)
- Q5_K_M: sweet spot (KLD ~0.06-0.07), widely considered optimal balance
- Q4_K_M: measurable drift (KLD ~0.08-0.10), but acceptable for most tasks
- Qwen 3.6 27B described as "notoriously resilient" to quantization
- Factors like context window size and MTP matter more than Q5→Q6 in practice

**Community consensus**:
- Q5_K_M is the sweet spot for dense models on 24-32GB hardware
- The practical difference between Q5_K_M and Q6_K is "subtle" in real coding tasks
- Q4_K_M is standard for 24GB cards, but Q5_K_M preferred when VRAM allows

### Unsloth GGUF Calibrated Quants

**Qwen 3.6 27B MTP**: https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF
**Gemma 4 26B A4B**: https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF

**Key findings**:
- Unsloth uses importance matrices (imatrix) for calibrated quantization
- Gemma 4 26B Q4_K_M: Mean KLD ~0.016-0.018 (exceptionally well-calibrated)
- Gemma 4 26B perplexity at Q4: ~4.7-5.4

### llama.cpp Perplexity and KL Divergence Tools

**URL**: https://github.com/ggml-org/llama.cpp

**Methodology**:
- `llama-perplexity` tool with `--kl-divergence-base` and `--kl-divergence` flags
- Generates BF16 logit baseline (~37 GB for LLaMA 3 on WikiText-2)
- Computes KL divergence between quantized model outputs and baseline
- Lower KLD = higher fidelity to original; 0 = identical distributions
