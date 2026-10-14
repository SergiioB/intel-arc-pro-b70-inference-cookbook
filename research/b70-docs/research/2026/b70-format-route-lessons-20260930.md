---
title: "B70 format and route lessons — dense vs MoE, what the hardware is best at (2026-09-30)"
type: research
date: 2026-09-30
status: provisional
---

# B70 format & route lessons (2026-09-30)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Private lessons note. Sources are cited by evidence
path; numbers carry their cell and cap.

## 0. The one law everything obeys

```
decode tok/s  =  yield × achievable_bandwidth / bytes_per_token
```

- B70 pin floor: **608 GB/s**, 32 GB, one card. Achieved in the EXL3 M=4 verify pass: **381 GB/s**
  (63 %); reached **636 GB/s** when the trellis decode work is skipped (B_FAKE ablation,
  `results/2026/09/b70-m4-flag-ablation-20260930T143631Z/ABLATION-RESULT.json`).
- yield = tokens accepted per round. EXL3 native MTP3 measured **2.93–3.16** of a theoretical 4.0.
- No engine, format, or flag escapes this law. Every "immense" lever must move one of the three terms:
  **bytes** (quantization, MoE sparsity), **bandwidth** (second card — unbuilt), or **yield** (speculation — bounded).

## 1. What the B70 hardware is best at (Xe2 / BMG-G31)

| Resource | Value | Consequence |
|---|---|---|
| XMX DPAS INT8 | ~363 TOPS class | **Prefill / grouped GEMM is its strongest op** |
| XMX DPAS BF16/FP16 | ~170 TFLOPS class | EXL3 decodes trellis→fp16, so it runs here |
| Sub-8-bit int operands | tie INT8 (no extra TOPS) | INT4 is a **bandwidth** lever, never a compute multiplier |
| Native FP8 DPAS | **absent** | FP8 runs in the BF16 half-rate class |
| GDDR6 | 608 GB/s, 32 GB | **Decode is streaming**: winner = fewest bytes/token |
| Vector FP32 | M≤2 GEMV path | small-M decode tail |

## 2. Dense vs MoE — the measured difference on this host

| | Dense 27B @4bpw (EXL3) | MoE 30B-A3B @4bpw |
|---|---|---|
| Weight bytes per token | ~13.5 GB (whole model) | ~1.5–2 GB (active experts + shared) |
| Theoretical ceiling @608 GB/s, yield 3 | ~111 tok/s | ~750–1000 tok/s |
| **Measured on this host** | **75.7–81.0 tok/s C1** (solo, 230 W, GPU1) / **426 aggregate C16** | **186.6 tok/s** (Nemotron-3.5-30B-A3B, GPTQ-INT4 G64 + DFlash n7, p2048/g128 n=5 client post-first, 150 W — historical metric, re-measure owed) |

**Why MoE makes the B70 shine:** the card's 608 GB/s is fixed; a MoE converts it into ~7× fewer bytes
per token, so the same bandwidth yields ~2.5× the measured tokens. 32 GB fits a 30B MoE at 4 bpw
comfortably, and the 363-TOPS INT8 class accelerates grouped prefill.

**What limits MoE here:** dispatch, not bandwidth. The K216 lesson: 84+ host synchronizations per token
and a grouped GEMM at 1.79 TFLOPS vs ~46 for a comparable dense GEMM (~25× gap). MoE on B70 needs an
on-device router and a fused grouped dequant GEMM before its ceiling is reachable.

## 3. Format verdict — is EXL3 still the best?

**Dense (Qwen3.8-27B): yes, and it is not close.**
- Only route passing quality (6/6 battery, 9/10 host tasks, correct canary) AND reaching native 262,144
  context, with a measured 5.4× prefix-cache win.
- Reproduces its upstream's published table within **1–7 %** at every C1–C16 cell on this box, and beats
  it at C8 (+1.5 %) and C16 (+4 %): `results/2026/09/b70-exl3-concurve-compare-20260930T1449Z/PROTOCOL-COMPARE.json`.
- MTP3 native, int8 prefill (1.89×), oneDNN fused SDPA (+10–18 % long-context prefill) — all active.

**MoE: EXL3 is not the natural format.** The reason is exact and measured: EXL3's cost is
**instruction issue** — trellis state-build + codebook decode + operand moves ≈ **336 instructions per
512 B tile-group** (~8.6 G instr/pass at M=4). In a dense model that is amortized against large streaming
reads (still costing 37 % of bandwidth); in a MoE the tiles are smaller and grouped per expert, so the
per-byte instruction cost is worse. Native INT4/INT8 (GPTQ/AWQ-family) feeds DPAS directly with the
dequant fused into the GEMM — which is why the fastest MoE measured here uses GPTQ INT4, not trellis.

## 4. W4A16 vs EXL3 on the 27B — were the W4A16 numbers better? No.

| | EXL3 4bpw (current) | W4A16 (AutoRound-ext) |
|---|---|---|
| p512/g128 @180 W | 60.60 | **64.49** (+6 %) |
| p8192/g128 @180 W | **59.83** | 46.34 (**−23 %**) |
| 230 W | **75.7–81.0** (today, solo) | historical 83.7 — **different artifact + client post-first statistic, never reproduced** |
| Quality | **passes** | **fails the canary (answers “30” not “14”); defect is in the weights (RL-092)** |
| Capacity | **262,144 native** | 256K unreachable |
| Prefix cache | **5.4× measured** | corruption → cache OFF |

The single historical number that looked better was measured differently, on a different artifact, at a
different cap. At matched conditions W4A16 is +6 %/−23 % and quality-defective. **EXL3 wins every axis.**

## 5. Always-on rules learned the hard way

1. **Verify MTP is active per route** (server log + acceptance counters). EXL3 = MTP3 (accept ≈2.9–3.16);
   W4A16 lane = MTP4. A measurement without MTP is not comparable to the rest of the record.
2. **Verify the effective card from the engine process.** Upstream `serve.py` derives `ZE_AFFINITY_MASK`
   from its own `--gpu` argument and **overrides** the container env — pass the card through the wrapper
   and confirm with `vram_mm` / process env before measuring.
3. **One client per server.** Two clients split the same tokens; the "C1 gap" that looked like a
   hardware deficit was exactly this (per-lane shares summed to the upstream number).
4. **Record caps and clocks per campaign.** Operating point since 2026-09-30: **200 W both cards,
   2400 MHz ceiling both** (GPU0 was 180 W/2400, GPU1 230 W/2800 as-found).
5. **Instruction issue is the current kernel frontier**, not DRAM: B_FAKE recovered +30/+38/+68 % GB/s and
   pushed qkv to the pin floor; A_ONCE ≈ +2 % (activations are not the problem); PF256 −29 %.

## 6. Where the immense numbers actually are

- **Batching**: demonstrated on this route — C16 = 426 aggregate vs 76 C1 (**5.6×**), verified solo.
- **MoE**: demonstrated by class — 186.6 vs ~76 (**2.5×**), re-measurement owed on today's harness.
- **Dense single-stream**: bounded — VERIM4 (instruction-issue rewrite) ceiling ~76–79; past ~95 tok/s on
  one card requires the unbuilt custom TP2.

Related: `research/2026/b70-next-program-design-20260930.md`,
`research/2026/b70-adversarial-closure-review-20260930.md`,
`research/2026/b70-kernel-forensics-next-20260930.md`,
`results/2026/09/b70-m4-flag-ablation-20260930T143631Z/`.
