---
title: "HQ4: a 4-bit dense format designed for Xe2 decode - EXL3 incoherence without the trellis (2026-10-09)"
type: research
date: 2026-10-09
status: provisional
---

# HQ4 — Hadamard-incoherent affine INT4 in a DPAS-operand layout

**PROVISIONAL - NOT FOR PUBLIC HEADLINE.** Design and pricing only. No GPU, no
download and no conversion ran for this note. Program question: RQ-26. Ledger: RL-130.

## 1. The two facts this design is built on

| Fact | Evidence |
|---|---|
| Quality on Qwen3.8-27B comes from incoherence plus code shape. The bit count is not what fails. EXL3 4bpw (block-128 Hadamard + trellis) passes the battery. Unrotated GPTQ/AutoRound sym INT4 G128 and Q4_K_M fail it at the same ~4 bpw. | RL-087, RL-092, ADR-0010; battery case "14 vs 30" |
| The EXL3 trellis is what costs speed. The kernel moves 381 GB/s. With the trellis decode step stubbed out (B_FAKE, wrong results by construction), the three M=4 cells rose 18-21 % to 476-542 GB/s, and the route-level anchor is 636 logical-byte GB/s. About 330 of ~336 instructions per 512 B are state reconstruction, codebook arithmetic and operand moves. | RL-119, `results/2026/09/b70-m4-flag-ablation-20260930T143631Z/`, `research/2026/b70-small-m-quant-layout-static-spec-20261009.md` |
| A Hadamard rotation alone fixes most of the INT4 error. On real 27B blocks, rotated GPTQ W4-G128 cut block-output KL to 0.40× (GDN) and 0.58× (attention) at identical bytes, and reduced reconstruction error on every tensor. | RL-067, `results/2026/09/qwen27-rotation-t1t2-20260925T1812Z/RESULTS.md` |

Inference: keep the half of EXL3 that is cheap on Xe2 (sign flips + block-128 Hadamard)
and replace the half that is expensive (trellis decode) with a code the hardware
unpacks in a few instructions. Nobody has measured this combination here.
Previous work tested only:
- rotation with GPTQ, at layer level;
- cheap INT4 kernels, without a quality-passing artifact (RQ-20 deferred for that reason).

## 2. Format definition

| Field | Choice | Why on Xe2 |
|---|---|---|
| Input transform | Per-linear random signs `su` × block-128 Hadamard on K (as EXL3 `suh`); optional offline-folded residual rotation H_5120 where it folds (T1/T2 fold map) | Block-128 FWHT is register-local; its cost is not separated in the B_FAKE run, so F1 times the wraps on their own; 17408 = 136 × 128 so `down_proj` is covered with no composite Hadamard |
| Output transform | Signs `sv` × block-128 Hadamard on N (as EXL3 `svh`), only where measured to help | Same cost class; arm in the quality screen |
| Weight code | 4-bit, one FP16 scale per 128 inputs (group = Hadamard block), symmetric; level shape is a screen axis: **uniform** (default), **E2M1-shaped** (arithmetic remap, no LUT) | Uniform decodes with the FP16 magic-number trick: two bit-ops per pair of weights; no gather, no LUT |
| Rounding | GPTQ (desc_act off) and AutoRound sign-rounding, both done in the rotated basis | AutoRound is Intel's and runs on XPU; GPTQ was 4× better than RTN in T2 |
| Escape hatch | Any tensor that fails its own error gate is stored as INT8 per-channel (native DPAS type) | Average must stay ≤ 4.5 bpw (above that EXL3 at 381 GB/s wins on bytes) |
| Byte layout | Nibbles pre-arranged in DPAS B-operand (VNNI) tile order, 64-byte aligned; scales in a separate aligned plane | One 2D block load per tile, no shuffles; the alignment lesson from Strata Q6_K (2.3–4.7×) |
| Activations | A16 (FP16 DPAS) by default; s8 activation × s4 weight integer DPAS is a separate ablation | Keeps verification numerics unchanged; the mixed s8×s4 DPAS mode needs its own ISA proof |
| MTP head / LM head / embeddings | Unchanged from the reference route (LM head stays FP16: INT4 head failed, RL-041) | Head quantization is a closed class |

Bytes: 4.125 bpw on body linears ≈ 12.6 GB vs EXL3 12.23 GB (+3 %).

## 3. Price

`research/references/rq26-hq4-price.json`, `research.py price`:

- Assumptions: 550 GB/s at 4.125 bpw against EXL3's 381 GB/s at 4.0 bpw, on an assumed 75 % share of round time, at equal MTP3 acceptance.
- Result: **1.27× decode** (63.25 → ~80 tok/s class at 230 W).
- Isolated kernel ceiling at pin rate: 1.60×.
- The 8K prefill also gains if the integer-DPAS ablation passes. That is not priced here.

## 4. Kill ladder (cheapest first; each stage can close the question)

| Stage | Work | GPU | Gate (pre-declared) |
|---|---|---|---|
| F0 quality screen | Download BF16 source (55.6 GB) and EXL3 4bpw (16.9 GB) to SN770. Regenerate calibration/eval hidden states (the D11 set is gone). On ≥4 blocks (2 GDN, 2 attention incl. layers 30/31, one early, one late): EXL3 reconstructed via its bit-exact `ref.py` vs HQ4 arms (uniform/E2M1 × GPTQ/AutoRound × out-transform on/off) | none (CPU) | Best HQ4 arm block-output KL **≤ 1.0×** EXL3's on every block, top-1 ≥ EXL3 − 0.1 pt, within 4.5 bpw |
| F1 kernel | RQ-20 kernel extended with block-128 Hadamard wraps, VNNI layout, M = 1/4/8 | one card, isolated | ≥ 500 GB/s effective at M = 4 on real 27B shapes, bitwise equal to a CPU reference, ISA shows no scratch spills and 2D block loads |
| F2 whole model | Convert all body linears; fake-quant KL vs BF16 on held-out text; exact-answer battery incl. the 14-vs-30 case, replay, long/multi-turn | conversion on one card | KL ≤ EXL3's whole-model KL; battery identical pass set to EXL3 |
| F3 route | vLLM XPU quant-method plugin (own code; exl3xpu kernels untouched), MTP3, graph capture on; paired cells vs EXL3 4bpw on the same image | one card | ≥ 1.20× decode in every confirmation pair, acceptance within 0.05, battery pass |

Stop rule: F0 failing on any block, after one rounding-recipe retry, closes HQ4 for
this model and records the measured KL gap. That gap tells us how much of EXL3's
quality comes from the trellis.

## 5. What this is not

- Not the stopped EXL3 kernel program (RL-122). EXL3 is used only as the quality
  reference and as the paired baseline.
- Not W4A4/s4×s4 decode (priced negative, B9), not QuaRot→INT3 (closed), not
  requant from a quantized artifact (RL-113..115), not DSM/M5X/Q3DT (different ABIs,
  `research/2026/b70-invented-quant-abis-20261009.md`).
- Nearest literature: QuaRot/SpinQuant (rotation + uniform INT4, W4A4 focus),
  QuIP#/QTIP (rotation + lattice/trellis codes), HIGGS-class rotated scalar grids.
  HQ4 differs in where the hardware cut falls. It drops the vector/trellis code because
  Xe2 decode pays for it in ALU work. It keeps only transforms measured as cheap on this card.

## 6. Storage sequencing

SN770 has 160 GB free. F0 needs 72.5 GB and F2 a further ~14 GB. RQ-25's route
repack needs ~76 GB and does not fit at the same time. Order:
1. F0 → F2.
2. Delete the BF16 source (it can be downloaded again, revision recorded).
3. Then the RQ-25 repack.
