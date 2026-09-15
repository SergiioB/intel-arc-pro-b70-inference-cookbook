# Quantization and Drafters on Intel Arc Pro B70

How to choose a weight format and a speculative-decoding drafter on Xe2, based
on measured evidence from this cookbook. One rule drives both sections: **the
quantized representation you serve is part of every downstream decision —
kernels, capacity, and drafter acceptance.**

## 1. Weight formats — what measured best

From the Qwen3.8-27B same-source format showdown
([Qwen3.8 evidence](qwen38-27b/QUANT-FORMAT-SHOWDOWN-B70.md)):

| Format | Decode (27B dense, one B70) | Verdict |
|---|---:|---|
| AutoRound/GPTQ INT4 sym G128 (W4A16), vLLM | 24.4 tok/s spec-off; **45.7 with native MTP4** | serving leader |
| GGUF Q4_K_M, llama.cpp SYCL | 20.0 tok/s | llama.cpp workhorse |
| GGUF mixed Q4+Q6/Q8 "rescue" | 18.9 tok/s | negative — costs decode, buys nothing |
| GGUF Q5_K_M | 17.0 tok/s | quality-lean |
| GGUF Q8_0 | 13.6 tok/s | capacity/quality-lean, not speed |

Rules that generalize:

- **Decode tracks streamed weight bytes; long-prompt prefill is byte-flat**
  (kernel-bound) — format choice prices decode, not prefill.
- **Selective up-precision of "sensitive" tensors is not a free lunch** on
  hybrid-GDN: the default preset's own mixture already allocates precision
  well. Measured: +10.7% bytes → −5.8% decode, no quality gain.
- **Execution mode is part of the format decision**: `--enforce-eager` costs
  2.36× decode on the pinned vLLM XPU image. Never benchmark or serve eager.
- Quality claims on hybrid linear-attention (GDN) models need logit parity,
  not perplexity — PPL is non-monotonic in precision on this class
  ([llama.cpp#28879](https://github.com/ggml-org/llama.cpp/issues/28879)).

## 2. Drafters — the native MTP head wins on quantized single-card serving

Measured closure on Qwen3.8-27B GPTQ-INT4 (one B70, 230 W, all known bugs
fixed — see [MTP-DEPTH-DFLASH2-KNORM-20260914.md](qwen38-27b/MTP-DEPTH-DFLASH2-KNORM-20260914.md)):

| Drafter | Acceptance (tok/step) | p512/g128 | p8192/g128 |
|---|---:|---:|---:|
| **Native MTP4** (1 layer, shares embed/head) | ≈3.3–4.0 | **50.78** | **46.58** |
| DFlash2 k=7 (5-layer 2B BF16 drafter) | ≈1.4–2.2 | 25.16 | 25.30 |

Why the heavier learned drafter loses on this class of setup:

1. **Drafter acceptance is conditioned on the target's hidden-state
   distribution.** Drafter checkpoints are trained against full-precision
   targets; a quantized serving target (GPTQ-INT4, worse FP8-decode) shifts
   that distribution and the published advantage disappears — measured
   acceptance lands *below* the native head's, which is quantization-exempt
   when its tensors are excluded from quantization.
2. **Cycle-cost asymmetry.** Speculative throughput ≈ accepted length ÷ cycle
   time. The native MTP drafter's marginal cost is ~one shared-weight layer;
   DFlash2 streams 3.85 GB and a selector walk every cycle. Break-even needs
   ~2× the native acceptance; it measures ~0.6×.
3. **Deep proposals decay.** Per-position acceptance drops to ≈0.1–0.3 by
   draft positions 5–6 on natural text — long proposal lists pay up-front
   draft cost for near-zero return. Depth 4 is the measured optimum; depth 6
   loses 10–20%.

**The rule for any future drafter on this host:** it must be trained against
(or proven robust to) the exact quantized representation it will serve, and
demonstrate ≥1.5× the native head's acceptance on held-out natural text
before any throughput campaign. A BF16-trained artifact port does not qualify.

Fair scoping: heavier drafters (DFlash2-class) were designed for, and published
on, BF16 targets with H200-class verify throughput and batch serving. Nothing
here disputes that regime — it is simply not single-card quantized Xe2 serving.

## 3. Related pages

- [Qwen3.8-27B GPTQ recipe](qwen38-27b/QWEN38-VLLM-XPU.md) — the serving route
  these decisions produce, including the DFlash2 research history and traps.
- [Draft INT4 overlay](qwen38-27b/DRAFT-INT4-S-M1.md) — quantizing the native
  MTP head itself (the sanctioned drafter-cost reduction).
- [Image and patch matrix](IMAGE-AND-PATCH-MATRIX.md) — quantization-specific
  patch compatibility.
