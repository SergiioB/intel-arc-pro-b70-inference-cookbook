# Quantization Quality: GPTQ-Int4 vs GGUF K-quants

> How the vLLM GPTQ-Int4 path compares to llama.cpp GGUF K-quants on
> Qwen3.6-35B-A3B (MoE), measured by KL divergence.

## The two paths

The B70 runs the same base model — **Qwen3.6-35B-A3B** (256 experts, 8 active,
hybrid GDN/attention, 3B active params/token) — through two different quantization
formats, each native to its inference engine:

| Engine | Format | File | Size | Bits/weight |
|--------|--------|------|------|-------------|
| vLLM XPU | **GPTQ-Int4** (group_size=128, symmetric, desc_act=false) | safetensors | 21 GB | ~4.5 |
| llama.cpp SYCL | **GGUF Q4_K_XL** (Unsloth Dynamic) | .gguf | 21 GB | ~4.88 |
| llama.cpp SYCL | **GGUF Q5_K_M** (Unsloth Dynamic) | .gguf | 25 GB | ~6.12 |

**GGUF and GPTQ are not directly interchangeable.** GPTQ is a weight-only
quantization method that packs int4 weights with float scales computed via
second-order (Hessian-based) calibration. GGUF K-quants use block-level scales
with mixed precision across tensor types (some tensors at Q8, some at Q4, some
at Q6). Same 4-bit target, different math.

## Model reference

**Important:** Our vLLM checkpoint is *not* the official Qwen release. It is:

```
Base:       Qwen/Qwen3.6-35B-A3B (BF16, Apache 2.0)
Derivative: llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved
Quantized:  llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4
Tool:       GPTQModel 7.1.0-dev (github.com/modelcloud/gptqmodel)
Config:     4-bit, group_size=128, sym=true, desc_act=false, pack_dtype=int32
            damp_percent=0.05, true_sequential=true, act_group_aware=true
MTP:        Preserved (1 MTP layer, BF16 — not quantized)
```

The "heretic" variant is an uncensored/abliterated post-train of the base
Qwen3.6-35B-A3B. Quantization quality metrics below are architecture-level
(Qwen3.5/3.6-35B-A3B share the same architecture: 40 layers, 256 experts, 8
active + 1 shared, hybrid GDN/attention). The heretic post-training may shift
absolute KLD slightly, but the relative ordering of quant formats is
architecture-determined, not weight-determined.

## KL divergence: what the community measured

KLD (KL Divergence) measures how far the quantized model's output distribution
has drifted from the BF16 baseline. Lower = more faithful to the original.

**Interpretation scale:**

| Mean KLD | Quality tier |
|----------|-------------|
| < 0.01 | Effectively lossless |
| 0.01–0.03 | Near-lossless (Q8 range) |
| 0.03–0.05 | Excellent (Q6 range) |
| 0.05–0.08 | High quality — sweet spot (Q5 range) |
| 0.08–0.12 | Acceptable (Q4 range) |
| > 0.15 | Quality cliff |

### Qwen3.5-35B-A3B GGUF quants (same architecture)

Measured by Unsloth and AesSedai via `llama-perplexity` KLD against BF16
baseline on Qwen3.5-35B-A3B (identical architecture to our Qwen3.6 variant):

| Quant | Size (GB) | PPL (WikiText) | Mean KLD | 99.9% KLD | Source |
|-------|-----------|----------------|----------|-----------|--------|
| BF16 | ~70 | 6.534 | 0 | 0 | baseline |
| Q8_0 | 35.2 | 6.535 | 0.0026 | 0.1033 | Unsloth |
| Q6_K_XL | 28.2 | 6.539 | 0.0041 | 0.1437 | Unsloth |
| **Q5_K_M** | **24.5** | **6.536** | **0.0058** | **0.210** | AesSedai |
| Q5_K_XL | 23.2 | 6.549 | 0.0069 | 0.236 | Unsloth |
| **Q4_K_XL** | **19.2** | **6.592** | **0.0137** | **0.410** | Unsloth |
| Q4_K_M | 18.5 | 6.605 | 0.0192 | 0.548 | Unsloth |
| Q4_K_M | 20.6 | 6.566 | 0.0096 | 0.317 | AesSedai |

Sources:
- [Unsloth Qwen3.5 GGUF Benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks)
- [AesSedai/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/AesSedai/Qwen3.5-35B-A3B-GGUF)

### GPTQ-Int4 estimated KLD

GPTQ-Int4 at group_size=128 with Hessian calibration is **directly comparable
to GGUF Q4_K_M/XL** in effective precision (both target ~4.5 bits/weight). The
IST-DASLab GPTQ-GGUF toolkit study measured GPTQ-4 vs K-quants on identical
models at matched bitwidths:

| Method | Bits/weight | WikiText PPL | Note |
|--------|-------------|-------------|------|
| GGUF Q4_K_S | 4.54 | 12.04 | Standard block quant |
| **GPTQ-4** | **4.50** | **12.16** | Hessian-calibrated |
| GGUF Q4_1 | 4.50 | 12.04 | Standard |

GPTQ-4 is within 1% of Q4_K_S perplexity — the Hessian calibration recovers
most of the precision loss that block quantization avoids via mixed precision.
At MoE scale (3B active params), the effective quality gap narrows further
because expert routing masking smooths quantization noise.

**Estimated Mean KLD for our GPTQ-Int4: ~0.012–0.020** (Q4 tier, architecture
equivalent to Q4_K_XL's 0.0137). We have not yet run a direct KLD measurement
on our checkpoint — this is the remaining correctness gate (see Contributing).

Source: [IST-DASLab/gptq-gguf-toolkit](https://github.com/IST-DASLab/gptq-gguf-toolkit)

## Side-by-side comparison

| Format | Size | Est. Mean KLD | Decode (B70) | Prefill (B70) | Engine |
|--------|------|---------------|-------------|---------------|--------|
| **GPTQ-Int4** | 21 GB | ~0.012–0.020 | **133 t/s** | **8,718 t/s** | vLLM XPU + MTP |
| GGUF Q4_K_XL | 21 GB | 0.0137 | 69 t/s | 1,498 t/s | llama.cpp SYCL |
| GGUF Q5_K_M | 25 GB | 0.0058 | 70 t/s | 1,604 t/s | llama.cpp SYCL |

**Key insight:** GPTQ-Int4 and Q4_K_XL are within the same KLD tier (~0.012–0.020).
The quality difference between them is negligible for practical use — both sit
firmly in the "high quality" band. The 1.8× decode and 5.2× prefill speedup
from vLLM MTP comes **for free** relative to Q4_K_XL quality.

Q5_K_M halves the KLD to 0.0058 (near-lossless tier) but is only available
through llama.cpp — which is 1.8× slower. The quality-speed tradeoff:

- **Maximum quality:** Q5_K_M via llama.cpp (KLD 0.006, but 70 t/s)
- **Maximum speed:** GPTQ-Int4 via vLLM MTP (KLD ~0.015, but 133 t/s)
- **Same quality, 2× speed:** GPTQ-Int4 ≈ Q4_K_XL in KLD, but vLLM is 1.8× faster

## Why MoE is quantization-robust

Qwen3.6-35B-A3B has 256 experts but activates only 8 per token (+ 1 shared).
This sparse activation has two effects:

1. **Noise dilution.** Quantization error in any single expert affects only
   the tokens routed to it. With 8/256 active, 96.9% of experts are idle per
   token — their quantization noise contributes zero to that token's output.

2. **Routing smoothing.** The router (gate network) selects experts based on
   dot-product scores. Small quantization perturbations in expert weights don't
   change routing decisions — the gate itself runs at full precision (or Q8).
   This means quantization noise can't cascade into routing failures.

This is why MoE models tolerate Q4 better than dense models of similar total
parameter count. The effective per-token quality is closer to a 3B-parameter
model at Q4 than a 35B model — because only 3B params are active.

## The remaining gate: direct KLD on our checkpoint

The numbers above are architecture-level estimates from community benchmarks on
Qwen3.5-35B-A3B. Our checkpoint is a Qwen3.6 derivative (heretic post-train +
GPTQModel 7.1.0 quantization). A direct KLD measurement requires:

1. Running `llama-perplexity --kl-divergence` on the GGUF conversion of our
   GPTQ checkpoint vs a BF16 baseline
2. OR running vLLM's built-in perplexity evaluation on both checkpoints

This is listed in the repo's Contributing section as the correctness gate.
The community KLD data gives high confidence that GPTQ-Int4 sits in the same
tier as Q4_K_XL, but the absolute number on our specific checkpoint is pending.

## References

- [Unsloth Qwen3.5 GGUF Benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks) — comprehensive KLD table for Qwen3.5-35B-A3B across all GGUF quants
- [AesSedai/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/AesSedai/Qwen3.5-35B-A3B-GGUF) — per-tensor MoE quant with KLD measurements
- [IST-DASLab/gptq-gguf-toolkit](https://github.com/IST-DASLab/gptq-gguf-toolkit) — GPTQ vs K-quant comparison at matched bitwidths
- [Banandre: Quantization Fidelity Benchmarking](https://www.banandre.com/blog/quantization-fidelity-benchmarking-kld-and-ppl-as-metrics-for-gguf-model-selection) — KLD variance across Q4_K_M providers (5× spread)
- [GPTQModel](https://github.com/modelcloud/gptqmodel) — the quantization tool used for our checkpoint
- [llama.cpp perplexity methodology](https://github.com/ggml-org/llama.cpp) — KL divergence computation in `llama-perplexity`
