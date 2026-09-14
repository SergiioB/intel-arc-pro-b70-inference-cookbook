# Quantization Format Strategy for Intel Arc Pro B70

> Which quant formats work on Intel Arc B70 (Battlemage / Xe2), why, and
> which is optimal. Last updated: 2026-08-07.

## TL;DR

**GPTQ-Int4 is the optimal format for vLLM XPU on MoE models.** It is not a
compromise — INT4 is the format Intel's XMX engines are built to accelerate.
This is Intel's equivalent of NVIDIA's NVFP4 on Tensor Cores. The 133 t/s
single-stream decode we measure is the hardware running its native fast path.

For dense models, llama.cpp GGUF (Q4_K_M / Q5_K_M) is the only working path.
vLLM XPU has no native block-FP8 kernel for dense models.

## What the terms mean

People conflate four independent things. They are not the same:

| Term | What it is | Who made it |
|------|-----------|-------------|
| **INT4** | Data format: 4-bit integers (0–15) | Generic, not vendor-owned |
| **GPTQ** | Algorithm that computes INT4 weights via Hessian calibration | Frantar et al. (2022) |
| **AWQ** | Alternative INT4 algorithm (activation-aware) | Lin et al. (2023) |
| **MXFP4** | Data format: 4-bit floats (E2M1) with FP8 block scales | OCP standard (Intel co-author) |
| **NVFP4** | NVIDIA's proprietary 4-bit float variant | NVIDIA |
| **FP8 (E4M3)** | Data format: 8-bit floats | OCP standard |
| **Heretic** | Post-training abliteration (uncensor) applied to model weights | Community (llmfan46) |

**GPTQ is not "the Intel format."** GPTQ is a quantization *method* that
produces INT4 *weights*. Intel's XMX hardware accelerates the INT4 *format*.
Any INT4 weights (GPTQ, AWQ, RTN) would hit the same XMX fast path. GPTQ is
the best-calibrated INT4 method available, so it produces the highest-quality
INT4 weights.

**Heretic is not a quantization format.** It is a post-training weight
modification (abliteration of refusal directions) applied *before*
quantization. Our model lineage:

```
Qwen/Qwen3.6-35B-A3B (BF16, stock)
  → llmfan46 heretic (uncensored/abliterated)
    → Native-MTP-Preserved (MTP layers kept intact)
      → GPTQ-Int4 (quantized to 4-bit integers via Hessian calibration)
```

## Format comparison on B70

### Formats that work

| Format | Engine | Single-stream decode | Prefill | Quality (KLD) | Status |
|--------|--------|--------------------:|--------:|--------------|--------|
| **GPTQ-Int4** | vLLM XPU | **133 t/s** | **8,718 t/s** | ~0.012–0.020 | ✅ Production MoE path |
| GGUF Q4_K_XL | llama.cpp SYCL | 69 t/s | 1,498 t/s | 0.0137 | ✅ Production llama.cpp path |
| GGUF Q5_K_M | llama.cpp SYCL | 70 t/s | 1,604 t/s | 0.0058 | ✅ Best quality path |
| GGUF Q6_K | llama.cpp SYCL | ~20 t/s (dense) | — | ~0.004 | ✅ Dense reasoning tier |
| GGUF Q8_0 | llama.cpp SYCL | ~18 t/s (dense) | — | ~0.003 | ✅ Near-lossless |
| MXFP4 | vLLM XPU | 10.4 t/s | 1,738 t/s | ~0.027 | ⚠️ Loads, correct output, slow |
| FP8 (block) | vLLM XPU | 0.75 t/s | — | — | ⚠️ Dequant fallback only |

### Formats that don't work on B70

| Format | Why not | Fix path |
|--------|---------|----------|
| NVFP4 | NVIDIA proprietary, Intel silicon doesn't implement it | N/A — wrong hardware |
| FP8 (native) | No XPU block-FP8 scaled MM kernel (`KeyError: PlatformEnum.XPU`) | Upstream `xpu_kernels` contribution |
| GPTQ-Int8 | Would load via vLLM, but ~42 GB — exceeds 32 GB VRAM | Not possible on single B70 |
| INT8 (W8A8) dense | XMX supports INT8, but vLLM has no XPU INT8 linear kernel for dense | Needs kernel registration |

## Why GPTQ-Int4 is the Intel optimal path

### Hardware: XMX engines are integer-first

The B70 has 256 XMX (Xe Matrix Extension) engines. These are the Intel
equivalent of NVIDIA Tensor Cores, but with a different design priority:

| | NVIDIA Tensor Cores | Intel XMX Engines |
|---|---|---|
| Native fast path | FP4 / FP8 / FP16 / TF32 | **INT4 / INT8 / INT16 / FP16** |
| Proprietary 4-bit | NVFP4 (float) | None (uses open OCP MXFP4) |
| Integer acceleration | Good but not primary | **Primary design target** |

Intel designed XMX around integer math. INT4 grouped GEMM (the exact operation
GPTQ-Int4 weights need) is the single fastest compute path on this silicon.
This is not a workaround — it is the hardware's intended use.

### Why MXFP4 is slower despite being "more native"

MXFP4 (OCP microscaling FP4) was co-authored by Intel and the B70 XMX does
support it. But our MXFP4 measurement (10.4 t/s) is not limited by the quant
format — it is limited by the model architecture:

- Qwen3.6-35B-A3B is a **hybrid model**: 30 GDN (Gated DeltaNet / linear
  attention) layers + 10 full attention layers
- The GDN layers run through Triton FLA (Flash Linear Attention) kernels
- These kernels have CUDA-specific fast paths but **generic fallbacks on XPU**
- 30 of 40 layers are bottlenecked by unoptimized Triton → 10.4 t/s

The MoE expert path with MXFP4 runs fine on XMX. If the GDN bottleneck were
fixed (native SYCL FLA kernels), MXFP4 could theoretically compete. But even
then, MXFP4 has 2× worse KLD than GPTQ-Int4 at the same bitwidth.

### Why INT4 beats FP4 at the same bitwidth

4 bits can encode:

| Format | Exponent bits | Mantissa bits | Values representable | Precision |
|--------|:---:|:---:|---|---|
| INT4 | 0 | 4 | 16 evenly-spaced integers | Uniform, high |
| FP4 (E2M1) | 2 | 1 | 16 values with large dynamic range | Non-uniform, low |

INT4 uses all 4 bits for precision. FP4 burns 2 bits on the exponent (dynamic
range), leaving only 1 mantissa bit — terrible precision per value. GPTQ's
Hessian calibration compensates for INT4's limited dynamic range by computing
optimal per-group float scales. The result: INT4 + learned scales beats FP4 +
fixed exponent at the same size.

Unsloth measured this directly on Qwen3.5-35B-A3B (same architecture):

| Format | Mean KLD | Verdict |
|--------|----------|---------|
| Q4_K_XL (INT4) | 0.0137 | High quality |
| MXFP4 (FP4) | 0.0272 | 2× worse — retired from dynamic quants |

## NVIDIA vs Intel quantization strategy

The two vendors chose opposite 4-bit philosophies:

**NVIDIA:** Tensor Cores are float-first. They created NVFP4 (proprietary) to
maximize FP4 throughput. INT4 works but is not the primary path. NVIDIA's
software stack (TensorRT, CUTLASS) is optimized around FP formats.

**Intel:** XMX engines are integer-first. INT4/INT8 are the native fast paths.
They co-authored the open OCP MX standard (MXFP4) rather than creating a
proprietary format. But the silicon's speed lives in INT4.

This means:
- **On NVIDIA:** FP4 formats (NVFP4, MXFP4) are competitive with or better than INT4
- **On Intel:** INT4 (GPTQ, AWQ) is strictly faster than FP4 (MXFP4)
- Porting NVIDIA-optimized quant recipes to Intel without considering this is
  a performance trap

## What would change the picture

| Development | Impact | Likelihood |
|-------------|--------|------------|
| Native XPU block-FP8 kernel | Unblocks dense 27B on vLLM (currently 0.75 t/s → could be 20-30 t/s) | Medium — Intel xpu_kernels team aware |
| Native SYCL FLA kernels for GDN | MXFP4 MoE decode 10→50+ t/s (removes Triton bottleneck) | Low — FLA is niche |
| Non-hybrid model (pure attention MoE) | MXFP4 would not have the GDN bottleneck | Model-dependent |
| OCP standardization of INT4 microscaling | Could unify the INT4/FP4 paths | Speculative |

The FP8 dense kernel is the highest-impact gap. Everything else is either
already optimal (MoE on GPTQ-Int4) or architectural (GDN bottleneck).

## Sources

- [Unsloth Qwen3.5 GGUF Benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks) — KLD per quant
- [AesSedai/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/AesSedai/Qwen3.5-35B-A3B-GGUF) — MoE-specific KLD
- [IST-DASLab/gptq-gguf-toolkit](https://github.com/IST-DASLab/gptq-gguf-toolkit) — GPTQ vs K-quant at matched bitwidths
- [GPTQModel](https://github.com/modelcloud/gptqmodel) — quantization tool
- [OCP Microscaling Formats (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-specification-pdf-html)
