---
title: "XTQ: a rate-distortion-first 4-bit weight format co-designed for Xe2 (B70) decode"
type: research
date: 2026-10-11
status: provisional
---

# XTQ — rate-distortion-first weight format for the B70

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Design and kill ladder. No number here is a
B70 measurement unless it cites a ledger row.

## 1. What mathematics allows

After a two-sided random Hadamard transform (incoherence), weight entries behave like
i.i.d. Gaussians. For a Gaussian source the rate-distortion bound is

$$D(R) = \sigma^2\,2^{-2R}, \qquad D(4) = 0.0039\,\sigma^2 .$$

No format beats it at equal rate. What each quantizer class loses at fixed rate:

| Quantizer class (fixed rate) | MSE / D(R) | Bits lost vs bound | Source |
|---|---|---|---|
| Uniform scalar, 4 bit, optimal step | 2.95× | 0.78 | Max (1960) table, 0.01154 |
| Lloyd-Max scalar, 4 bit | 2.43× | 0.64 | Max (1960) table, 0.009497 |
| Entropy-coded uniform scalar (high rate) | 1.42× | 0.25 | πe/6 |
| E8P lattice VQ (QuIP#), 2 bit | 1.42× | 0.25 | QTIP Table 1: 0.089 vs 0.0625 |
| Trellis-coded (QTIP, L=16), 2 bit | ~1.10× | ~0.07 | QTIP Table 1: 0.069 |

Measured on the dense Qwen3.8-27B (13-layer prefix, RL-170): scalar INT4 with a one-sided
rotation sat at 2.9× EXL3's final-logit KL. That matches the scalar row. EXL3 (a QTIP-family
trellis) is already within about 0.07 bit of the bound on weight MSE.

Consequence: at 4.0 bpw, only a trellis-class quantizer can match EXL3 on weight MSE.
Lower final-logit KL than EXL3 at equal bits can come only from three places:

1. **The objective.** Round to minimise the model's output KL, not each layer's
   proxy error tr(ΔW H ΔWᵀ). Kronecker-factored full-model Hessians (YAQA, Tseng et al.
   2025) reported about one-third lower KL than layer-local LDLQ [UNVERIFIED on this model].
2. **Bit allocation.** If tensor t contributes s_t·c_t·2^{-2R_t} to the KL, reverse
   water-filling gives the optimal rates
   R_t = R̄ + ½·log₂(s_t c_t / G), where G is the geometric mean of s_t c_t over all
   weights. At equal average bpw, KL falls by the ratio
   (arithmetic mean)/(geometric mean) of s_t c_t. That ratio is measurable from the
   screen's per-tensor Hessian-weighted errors before building anything.
3. **Better incoherence** (learned rotations) where weights stay non-Gaussian after the
   random transform.

## 2. What the B70 allows

Decode streams weights: tok/s ≈ effective GB/s ÷ bytes per token. EXL3 reaches 381 GB/s
logical at 4.0 bpw against a 608 GB/s pin (RL-119). The limit is instruction issue in the
trellis decode. Per card at 4 bpw, the pin rate is 1216 G weights/s, which leaves a budget of
a few vector-lane operations per weight for decode plus the M-row dot product.

So a B70-perfect format must meet three conditions:

- decode each weight in a few Xe2 ALU ops (shift/mask from a 16-byte-aligned stream, one
  cheap code evaluation; no per-weight table gather if SLM gather is slower);
- produce decoded values directly in the DPAS B-operand register layout (VNNI fp16
  pairs), so the M=4 verify product runs on XMX and the vector units only decode;
- keep the stream in 16-byte-aligned tiles that one sub-group (16 lanes) consumes without
  shuffles.

## 3. XTQ definition (candidate)

- Two-sided block-128 random sign-Hadamard per linear (same as EXL3's incoherence).
- Bitshift trellis, state L = 16 bits, K = 4 bits per weight (rates 3/4/5 allowed per
  tensor), tiles of 16×16 weights, code function f(state) → fp16 **chosen from candidates
  for Xe2 issue cost at equal MSE** (QTIP reports 1MAD, 3INST and HYB within ~3 % MSE of
  each other, so code choice is a speed decision, not a quality decision).
- Rounding: Viterbi with block LDLQ feedback (QTIP), then optionally a KL-Hessian objective.
- Per-tensor rate from reverse water-filling on measured sensitivities, average 4.0 bpw.
- Kernel: decode into DPAS operand layout; M = 1 on vector FMA, M = 4 on DPAS.

## 4. Kill ladder (cheapest first)

| Step | Measurement | Kills XTQ if |
|---|---|---|
| K0 | Screen (RQ-28 prefix): scalar INT4 with two-sided RHT vs EXL3 | — (settles whether incoherence, not the grid, explains R4) |
| K1 | Per-tensor Hessian-weighted errors → water-filling AM/GM ratio | ratio < 1.10 (allocation adds < 10 % KL) — allocation dropped, format continues |
| K2 | CPU Viterbi MSE of each Xe2 candidate code on Gaussian data, L = 16 | best Xe2 code > 1.05× EXL3-code MSE |
| K3 | Compile-only Xe2 ISA price of decode-into-DPAS-layout per code | no code below EXL3's per-weight decode issue by ≥ 1.4× |
| K4 | GPU1 decode microbench, ring > L2, real shapes | < 550 GB/s logical at 4.0 bpw |
| K5 | XTQ quantizer on real tensors vs EXL3 weight error, then the screen | KL > 1.0× EXL3 at 4.0 bpw |
| K6 | Serving integration, paired decode | < 1.30× EXL3 in any confirmation pair |

Related: `research/2026/b70-r4-rotated-int4-design-20261011.md`, ledger RL-119, RL-167..RL-172.
Primary sources: QTIP arXiv 2406.11235; QuIP# arXiv 2402.04396; Max, "Quantizing for
minimum distortion", IRE Trans. IT 6(1), 1960.
