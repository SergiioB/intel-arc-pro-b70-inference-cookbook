---
title: "XMX DPAS rate classes on B70: sub-8-bit weights buy bandwidth, not TOPS"
type: research
status: final
date: 2026-09-15
---

# XMX DPAS rate classes on B70: sub-8-bit weights buy bandwidth, not TOPS

Supersedes the peak-inference table in
`research/2026/b70-xe2-architecture-prefill-analysis-20260807.md` (which
estimated ~734 TOPS INT8 MAC-counted and ~1,468 TOPS INT4 with an explicit
unverified flag — both now measured wrong for effective throughput).

## Claim audited

> "INT4 is the Intel-optimized format. The B70's XMX engines have native
> INT4 grouped GEMM support — this is Intel's equivalent of NVIDIA's NVFP4."

**Verdict: false as a throughput claim; true only as operand support.**
`s4/u4` are native DPAS operand types, but W4A8 does **not** increase
arithmetic throughput over W8A8 on Xe2. NVFP4 on Blackwell buys 2× math rate
over FP8; INT4 on Xe2 buys zero extra TOPS over INT8.

## Evidence — operator ngen microbench (2026-09-15, single B70)

Private operator-built matmul kernels via Intel `ngen` (the assembler
oneDNN uses to JIT kernels), with guaranteed DPAS emission. **Not a
registered campaign — operator-provided measurement, unverified by repo
protocol; consistent with DPAS instruction geometry and the corrected
theoretical model below.**

| Kernel | Operands | Rate |
|---|---|---:|
| bf16 | 16-bit × 16-bit | 170 TFLOPS |
| s8 | 8-bit × 8-bit | 363 TOPS |
| w8a4 | 8-bit × 4-bit | 363 TOPS |
| w8a2 | 8-bit × 2-bit | 363 TOPS |
| w4a8 | 4-bit × 8-bit | 363 TOPS |

## Corrected theoretical model

256 XMX engines (32 Xe-cores × 8), 256 INT8 MACs/clk/XMX (128 BF16), MAC =
2 ops, 2.8 GHz: **367.0 TOPS INT8 / 183.5 TFLOPS BF16** peaks. Measured
363 / 170 = **98.9% / 92.6% of peak** (effective clocks / utilization).
Intel's quoted 367 TOPS INT8 is the deliverable class; the earlier
"734 MAC-counted" interpretation was double counting.

Why mixed widths tie: DPAS selects `OPS_PER_CHAN=4` whenever **either**
integer operand is 8-bit, so 8×8, 8×4, 8×2, and 4×8 all execute at the same
K=32 / SD=8 geometry → one rate class. **Pure sub-8×sub-8 (4×4 / 2×2,
`OPS_PER_CHAN=8`) — measured 2026-09-21:** an ESIMD `xmx::dpas` sweep at identical
shape (M=4, N=16, nacc=1), identical chain depth (`reps=1`, nsub=1024) gives
**s4×s4 = 33.91 TOPS vs s8×s8 = 16.96 TOPS = 2.00×** — exactly the `K_instr`
32 → 64 effect at unchanged per-instruction cost. That is an **instruction-efficiency**
fact, not an application ceiling: every absolute number in that probe sits far below the
363/170 anchors above (issue-limited harness; its best verified bf16 cell is 58.6 TFLOPS).
It stays irrelevant for LLM serving where activations are 8/16-bit (no quality-viable A4
path). So a 2× sub-byte *per-instruction* figure is now measured and citable **with those
caveats**; a 2× *application* figure is still not implied by anything. Probe + raw CSVs:
`results/2026/09/b70-improvement-probes-20260920/dpas-rate-probe/`.

## Implications for this fleet

- **Decode (bandwidth-bound):** INT4 weights ≈ half the weight streaming of
  INT8 → the only place INT4 wins. Expect well under 2× end-to-end: KV
  traffic, activations, scales, and any load-time expansion don't shrink.
  Cross-check: OV INT4-GDN8 no-MTP 22.4 vs OV INT8 no-MTP 18.7 tok/s (+20%),
  `results/2026/09/qw38-ov-mtp-20260914/`.
- **Prefill / compute-bound:** W4A8 gains nothing (same rate class); with
  load-time expansion to 8/16-bit it can be *slower*. Expected and observed:
  OV INT4 vs INT8 prefill parity.
- **FP8 W8A16 has no native Xe2 DPAS path** — it dequants to BF16 and lands
  in the half-rate class (~183.5 peak) with 1-byte weights. INT4 W4A8
  (363-class, 0.5-byte) **dominates FP8-W8A16 on both axes** — the mechanism
  behind GPTQ-Int4 being the serving leader over every FP8 arm measured.
- **Runtime caveat:** INT4-compressed OpenVINO/oneDNN weights do not
  guarantee w4a8 DPAS dispatch; kernels may expand to INT8/FP16 at load or
  JIT. If an INT4 build shows prefill *regression* vs INT8, inspect the
  generated kernel / trace before blaming the format.

## Corrections applied

- `docs/reference/megakernel-comprehensive-reference.md` §5: rate-class law
  added; stale B450-F-era topology line fixed.
- `AGENTS.md` §17 rule 3: throughput-model sentence added.
- `engines/vllm-xpu/README.md`, `engines/openvino-genai/README.md`: mechanism
  notes (bandwidth lever, not TOPS; FP8 → BF16 rate class).
- Banner on the superseded 2026-08-07 peak table.
