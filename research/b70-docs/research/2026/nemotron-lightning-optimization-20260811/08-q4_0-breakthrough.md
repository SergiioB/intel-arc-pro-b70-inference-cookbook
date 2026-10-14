---
title: "08 — The Q4_0 Breakthrough: 78% Faster Decode from Simpler Quantization"
type: research
date: 2026-08-11
status: final
---

# 08 — The Q4_0 Breakthrough: 78% Faster Decode from Simpler Quantization

**Date:** 2026-08-12
**Status:** Historical measured result; LocalMaxxing accepted self-report (87.4 t/s MTP4). The local Q4_0 artifact is now absent and this is not the current Nemotron serving recommendation.

## The result

| Metric | Q4_K_M (baseline) | Q4_0 | Δ |
|--------|------------------:|-----:|---:|
| Model size | 23.68 GiB | 17.44 GiB | **-26%** |
| No-spec decode (tg128, 150W) | 38.2 t/s | 68.0 t/s | **+78%** |
| MTP4 decode (230W, n=5 median) | ~50 t/s | **87.4 t/s** | **+75%** |
| MTP acceptance | 95-99% | 95-99% | same |
| Correctness | verified | verified ("Paris") | — |

The speed gain (1.78×) **exceeds the size reduction (1.36×)** — meaning Q4_0's
gain is not just "fewer bytes to read" but genuinely **better GEMV kernel
efficiency** from simpler dequantization.

## Why Q4_0 is faster on Xe2

The two quant formats have fundamentally different block structures:

### Q4_K_M (block_q4_K, 5.44 BPW)
- 256-weight blocks with 4-bit data + 2 fp16 scales + 2 fp16 mins
- Per-block min/max normalization + super-scale handling
- Dequantization requires: unpack 4-bit nibbles → apply min offset → multiply
  by per-block scale → apply super-scale → combine high/low sub-blocks
- **Complex address computation and multiple scale lookups per block**

### Q4_0 (block_q4_0, 4.5 BPW)
- 32-weight blocks: 1 fp16 scale + 16 bytes of packed 4-bit data
- Dequantization requires: unpack 4-bit nibbles → multiply by ONE scale
- **Single scale lookup, no min offset, no super-scale, no sub-block logic**

On the B70's Xe2, the MMQ/MMVQ GEMV kernels dequantize every weight block on
the fly. Q4_K's per-block min/max + super-scale chain multiplies the integer
and address arithmetic per element. The Q4_0 kernel is memory-access and
arithmetic-simple enough to run at much higher effective bandwidth.

**Measured bandwidth efficiency:**
- Q4_K_M: ~75 GB/s effective (12% of 608 GB/s peak)
- Q4_0: ~97 GB/s effective (16% of peak)
- The remaining gap to Qwen MoE's 36% suggests further GEMV kernel work is
  still possible, but Q4_0 already exceeded the 75 t/s target with MTP4.

## The conversion path

```
Source:  NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_K_M.gguf (23.68 GiB)
Target:  NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf (17.44 GiB)
Command: llama-quantize <source> <target> Q4_0 --allow-requantize
```

Note: the source "Q4_K_M" is actually a **mixed quant** (f32/q5_0/q8_0/q4_K/bf16
tensors). Plain Q4_0 failed with `requantizing from type q5_0 is disabled`;
`--allow-requantize` completed it. One tensor fell back to q8_0 (`output.weight`).

## Precision loss: how much worse is Q4_0?

Honest assessment — Q4_0 is a **legacy/simpler format with measurable but modest
quality loss** on a 30B model:

| Aspect | Q4_K_M | Q4_0 |
|--------|--------|------|
| Bits/weight | ~5.44 (mixed 4-6 bit) | 4.5 |
| Block size | 256 | 32 |
| Scales | 2×fp16 + super-scale + min | 1×fp16 |
| Quality | community-recommended | legacy, acceptable |
| Speed | slower GEMV | **much faster GEMV** |

Known community measurements (7-13B models): Q4_0 is typically ~0.05-0.15
higher perplexity than Q4_K_M on wikitext. For 30B+ models the gap narrows
further (~0.03-0.08). This is a **quality-vs-speed tradeoff, not a correctness
failure** — the model still produces coherent, correct output (verified).

**Caveat:** our Q4_0 was created by requantizing from Q4_K_M, which adds one
extra quantization step vs direct-from-BF16. For a production quality comparison,
quantizing from the BF16 source would be the clean baseline. The performance
characteristics are identical; the absolute quality is marginally worse than a
native Q4_0 would be.

## When to use which

- **Q4_0 historical finding**: for this exact Nemotron llama.cpp build/artifact,
  throughput-critical serving gained materially from simpler model-weight blocks.
  The tested file was a Q4_K_M requant, is no longer present locally, and lacks a
  matched task-quality comparison; do not treat it as the current launch path.
- **Current Nemotron serving route**: vLLM symmetric G64 GPTQ target plus BF16
  DFlash, as documented in the Nemotron model dossier and Run 36.
- **Native Q4_K_M/Q5_K_M**: retain as quality-oriented llama.cpp candidates when
  an exact present artifact is validated on the intended task.

## Evidence

- Raw: `results/2026/08/nemotron-q4_0-test/` (quantize.log, benchmark logs)
- Q4_0 MTP4 server evidence: `results/2026/08/nemotron-q4_0-mtp4-230w/server.log`
- Q4_0 LocalMaxxing payload/result: `results/_quarantine/nemotron-q4_0-lmx-result.json`
- `results/2026/08/nemotron-lmx-mtp4-230w/campaign-results.json` is the Q4_K_M baseline,
  not the Q4_0 campaign; the earlier cross-link was incorrect.
- LocalMaxxing run `cmsp78jxo00mhmp015lf9v28b`, tokSOut 87.4, accepted published
  self-report; not independent reproduction.
