---
title: "Run-4 subagent report: format-theory (2026-10-12)"
type: research
date: 2026-10-12
status: provisional
---

# FormatTheory — 4-bit weight-format mathematics for the dual-B70 dense Qwen3.8-27B route

Lane: FormatTheory (CPU-only, no GPU used). Date 2026-10-11 (local) / 2026-10-12 UTC.
Contract: `AGENTS.md` §3,4,8,9,11,13,14,17. All commands below are unsealed CPU smokes.

Raw outputs (never on `/`): `/mnt/<nvme-disk>/b70-run4-20261011/FormatTheory/`
(`gauss_rows.json`, `real_tensors.json`, `e8rvq_real.json`, `mixed_precision.json`,
`k1_waterfill.json`, `format_theory_table.json`, `format_theory_table.md`, `*.log`).
Scratch code: `scripts/tmp/format_theory_20261011/` (this lane owns it).

---

## 0. Reproduce

```bash
cd ~/B70-DOCS
P=scripts/tmp/format_theory_20261011
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 /usr/bin/python3 $P/selfcheck.py
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 --setenv=OMP_NUM_THREADS=4 \
  /usr/bin/python3 $P/gauss_rows.py                # part 1  (seed 20261011, 2e7 samples)
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 --setenv=OMP_NUM_THREADS=6 \
  /usr/bin/python3 $P/real_tensors.py              # part 2  (EXL3 ref.py + candidates)
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 /usr/bin/python3 $P/quant_extra.py
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 --setenv=OMP_NUM_THREADS=6 \
  /usr/bin/python3 $P/e8rvq_slice.py               # E8RVQ4 on 256-row tensor slices
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 /usr/bin/python3 $P/mixed_precision.py
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 /usr/bin/python3 $P/waterfill_k1.py \
  --layers results/2026/10/RQ-27-20261011T202824666303Z-94c266d8dd/raw/layers.jsonl --arm r4_g64
/usr/bin/python3 $P/bits_arith.py                  # final table
```

Seeds: every script pins `SEED = 20261011` (`gauss_rows.py:21`, `real_tensors.py:29`,
`e8rvq_slice.py:27`, `quant_extra.py:26`, `mixed_precision.py:37`,
`waterfill_k1.py:136`). Seeds fix the random sign-Hadamard and the Gaussian draw;
the quantizers themselves are deterministic. No measurement here is a GPU
measurement; the only measured B70 numbers used are cited from RL-119 / the RQ-27
smoke and are marked.

Verification (`selfcheck.py`, all passing, log in `format_theory_*`):
`H^T H - I` max 5.96e-8; `||R_out W R_in||_F/||W||_F - 1` 1.94e-7 → the two-sided
block-128 sign-Hadamard is orthonormal, so a relative Frobenius error measured in the
rotated basis equals the original-basis error **exactly** (this is why every
"rel MSE" below is basis-independent); E8 nearest-point equals exhaustive brute force
on all 64 test vectors (max difference 0.00e0); empirical `G(E8) = 0.07170` vs the
published 0.0717; Lloyd-Max 4-bit MSE agrees between an exact Gaussian centroid
iteration and an independent 8e6-point grid EM (9.5010e-3 both).

---

## 1. Part 1 — Monte-Carlo per-weight MSE, i.i.d. N(0,1)

`gauss_rows.py:29` (2.0e7 samples, chunk 2^21, `SEED=20261011`). D(R)=2^-2R, so
D(4)=2^-8=3.90625e-3. Raw: `gauss_rows.json`. Quantizer implementations:
`quant.py:78` (uniform optimal step), `:106` (GPTQ sym grid, generalised bits),
`:167` (Lloyd-Max, exact levels `:45`), `:179` (NF4), `:154` (INT8 per row),
`:232` (plain-E8 stand-in), and the real E8P12 codec `scripts/tmp/e8rvq_20261011.py`.

| quantizer (fixed rate, no entropy coding) | bpw | Gaussian MSE | x D(4) |
|---|---|---|---|
| uniform INT4, optimal step Δ=0.335201 | 4.0 | 1.153683e-2 | **2.953** |
| GPTQ sym grid G128 (fp16 scale) | 4.125 | 1.225728e-2 | **3.138** |
| GPTQ sym grid G64 | 4.25 | 1.056266e-2 | **2.704** |
| Lloyd-Max 4-bit | 4.0 | 9.501160e-3 | **2.432** |
| NF4 per-group 128 | 4.125 | 9.137158e-3 | **2.339** |
| NF4 per-group 64 | 4.25 | 8.464044e-3 | **2.167** |
| E8RVQ4 (E8P12 codebook, 2-stage RVQ, codec `fit_scales`) | 4.002 | 9.2434e-3 (rows of 256) / 9.5758e-3 (rows of 5120) | **2.366 / 2.451** |
| E8RVQ4, wider per-row scale search (1.2, 0.9) | 4.002 | 8.6202e-3 (rows of 5120) | **2.207** |
| plain E8 lattice, rate-4 volume scale (entropy-coded convention) | 4.0 | 4.782310e-3 | **1.224** |
| reference: entropy-coded uniform scalar, high rate (πe/6) | 4.0 | – | 1.423 |
| reference: QTIP Table 1, trellis L=16 at 2 bits, and E8P at 2 bits | 2.0 | 0.069 / 0.089 | 1.10 / 1.42 [UNVERIFIED here, cited from the design note] |
| INT8 per-output-row (control) | 8.0 | 1.376487e-4 | 0.0352 |

Findings.

1. **The lead's arithmetic reproduces exactly.** uniform INT4 optimal step = 2.953x D(4)
   (design note / Max 1960 table says 2.95x); Lloyd-Max = 2.432x (2.43x). Both were
   recomputed here from first principles (fixed-range quadrature + golden section on Δ,
   and an exact centroid iteration), not quoted. The optimal step is Δ=0.3352 (verified
   by a direct MSE-vs-Δ scan: 0.32→1.1733e-2, 0.335→1.1543e-2, 0.35→1.1692e-2).
2. **The Gaussian gap of the fixed-rate scalar class is ~2.4-3.0x, and the whole
   scalar class lives between 2.17x and 3.14x.** A *nonuniform* 16-level grid fitted
   for N(0,1) (Lloyd-Max, NF4) beats uniform-step and the GPTQ absmax grid by 0.16-0.35
   bits at equal rate; that is free at the format level (no data dependence) but needs a
   16-entry dequant LUT rather than `(code-8)*scale`.
3. **E8RVQ4 (the QuIP#-style E8P12+RVQ2 format) does not deliver the lattice's
   advantage at fixed rate.** 2.37-2.45x D(4) with the codec's per-row fp16 scales,
   i.e. only 1.25-1.33x better than the scalar GPTQ grid and *worse than the NF4 G64
   scalar grid* at comparable bpw. A plain E8 lattice nearest-point quantizer at the
   rate-4 volume scale would sit at 1.224x D(4) — the loss is the fixed-rate indexed
   codebook + shaper/side-information, not the lattice geometry. This is why E8RVQ4 is
   killed by the mathematics of its own contract (lead decision 2026-10-11) and not
   merely by its kernel.
4. NaN/infinity: none; all quantizer outputs finite. The **plain-E8 stand-in with
   per-row absmax/RMS scales** (`quant.py:232`) is an *infinite* lattice and therefore
   **not rate-matched** (it reports 0.32x and 0.001x D(4) — i.e. below the R-D bound,
   which is impossible at 4 bits). Those rows are excluded from every comparison; they
   are kept in the JSON only as the granular-gain control that produced G(E8)=0.0717.

### 1.3 E8RVQ4 scale-convention study (why the number moves)

The E8RVQ4 contract fixes one fp16 scale pair per **output row** (`s[2, out]`). Three
scale grids were measured on the same i.i.d. Gaussian data (rows of 5120, 32-64 rows):

| scale rule | row width / rows | MSE | x D(4) | log |
|---|---|---|---|---|
| codec `fit_scales` (per-row, grid centred 0.971 both stages) | 5120 / 64 | 9.6711e-3 | 2.476 | `e8_scalegrid_check.log` |
| codec `fit_scales` | 256 / 1280 | 9.2367e-3 | 2.365 | `e8_scalegrid_check.log` |
| strict per-row search, best (1.2, 0.9) x row/residual RMS | 5120 / 32 | 8.6202e-3 | 2.207 | `e8_sweep_perrow2.log` |
| strict per-row search at the codec's own point | 5120 / 32 | 9.6330e-3 | 2.466 | `e8_sweep_perrow.log` |

Records, so nobody repeats them: **two wider-search numbers were withdrawn** because
their stage-2 scale was per-8-vector (1.296x, 1.646x D(4)) — that is +2 bpw of side
information and not the contract. The contract-per-row numbers are the ones above.
Conclusion for the record: the E8RVQ4 number is scale-grid sensitive (2.45x vs ~2.16x
between the codec grid and a wider per-row search) but stays **≥ 2.1x D(4)** for
strictly per-row side information, i.e. never within 1.4x of EXL3's 1.51x D(4).

---

## 2. Part 2 — real tensors (layer 2), EXL3, and the two-sided RHT

`real_tensors.py:136` measures, on `model.language_model.layers.2.linear_attn.in_proj_qkv.weight`
[10240,5120] and `...layers.2.mlp.down_proj.weight` [5120,17408], the relative Frobenius
weight MSE of each candidate RTN quantizer, in the raw basis and after a two-sided
block-128 random sign-Hadamard (`real_tensors.py:69`, seed 20261011). The EXL3 4 bpw
reconstruction is bit-exact via the pinned reference
`/mnt/<nvme-disk>/b70-k3-build-20260929/src/exl3xpu/ref.py`
sha256 `3f614b7c91510d8520202e85dd0a08255370349a4d5ae4e7601edfa6897d1780`
(`real_tensors.py:82`, geometry copied from `scripts/tmp/r4_r0_20261011.py:648`,
`ref.weight_orig` = `ref.py:133`).

**EXL3 4 bpw relative weight MSE: 5.890234e-3 (in_proj_qkv), 5.832744e-3 (down_proj).**
In RD currency that is **1.508x / 1.493x D(4)**, i.e. EXL3 recovers ~0.30 bit of the
0.64-bit Lloyd-Max and 0.78-bit uniform-scalar losses at 4 bits. This is the number
every candidate below is compared against.

| candidate (RTN, two-sided RHT basis) | bpw | qkv rel MSE | down rel MSE | qkv xEXL3 | down xEXL3 |
|---|---|---|---|---|---|
| EXL3 4 bpw (pinned ref.py, own basis) | 4.03 | 5.890e-3 | 5.833e-3 | 1.000 | 1.000 |
| NF4 G64 | 4.25 | 8.4593e-3 | 8.4611e-3 | **1.436** | **1.451** |
| Lloyd-Max, G128 absmax scale | 4.125 | 8.8307e-3 | 8.8311e-3 | 1.499 | 1.514 |
| E8RVQ4 (codec, 256-row slice) | 4.002 | 9.6926e-3 | 9.6289e-3 | **1.645** | **1.651** |
| Lloyd-Max, G64 RMS scale | 4.25 | 9.1731e-3 | 9.1737e-3 | 1.557 | 1.573 |
| Lloyd-Max, G128 RMS scale | 4.125 | 9.3262e-3 | 9.3288e-3 | 1.583 | 1.599 |
| GPTQ sym grid G64 (fp16 scale) | 4.25 | 1.05615e-2 | 1.05627e-2 | 1.793 | 1.811 |
| uniform INT4, single global optimal step | 4.0 | 1.16995e-2 | 1.15935e-2 | 1.986 | 1.988 |
| GPTQ sym grid G128 (fp16 scale) | 4.125 | 1.22644e-2 | 1.22629e-2 | 2.082 | 2.102 |
| (same candidates without the RHT: GPTQ G128 | 4.125 | 1.28938e-2 | 1.28271e-2 | 2.189 | 2.199) | |
| (same candidates without the RHT: GPTQ G64 | 4.25 | 1.09632e-2 | 1.08983e-2 | 1.861 | 1.868) | |

Findings.

1. **The RHT really does Gaussianise these tensors.** Measured on real weights,
   GPTQ G128 gives 1.22644e-2 (qkv) / 1.22629e-2 (down); the i.i.d.-Gaussian Monte
   Carlo gives 1.225728e-2. Agreement to 0.06 % / 0.05 %. The same holds for G64
   (1.05615e-2 vs 1.056266e-2) and for every other grid in the table. **Consequence:
   any fixed-rate quantizer's Gaussian MSE number is a valid predictor of its real
   weight MSE after the two-sided RHT** — no per-tensor Gaussianity check is needed.
2. The RHT's *weight-MSE* benefit over the raw basis is small and grid-dependent:
   1.05x for G128 and 1.04x for G64. So the R4-vs-EXL3 quality gap (2.9-3.6x in KL) is
   **not** explained by weight-MSE magnitude; it must be explained by the *structure*
   of the error (how it interacts with the Hessian) — see §3. That is a separate,
   measurable claim from K0's arms.
3. **The best 4-bit weight-only candidate measured here is a nonuniform scalar grid,
   not a lattice**: NF4 G64 = 1.44x EXL3, Lloyd-Max G128-absmax = 1.50x, E8RVQ4 = 1.65x,
   GPTQ G64 = 1.79x, GPTQ G128 = 2.09x. In bits, switching the grid from
   uniform-absmax to a nonuniform 16-level grid is worth `0.5*log2(1.0562e-2/8.4593e-3)`
   = **0.16 bits** at G64 and `0.5*log2(1.2264e-2/8.8307e-3)` = **0.24 bits** at G128;
   EXL3's remaining advantage over the best nonuniform scalar is
   `0.5*log2(1.443)` = **0.26 bits** (4.25 bpw − 0.26 = 3.99 bpw equivalent). So of the
   0.53-bit gap between the campaign's uniform INT4 G128 and EXL3, **half is the grid
   shape and half is the trellis**, and the grid half is data-independent, free at
   format level, and costs no extra bits — it only needs a 16-entry dequant LUT in the
   kernel instead of `(code - zero) * scale`.

---

## 2.5 Acceptance table (machine-generated, `format_theory_table.md`)

Produced by `bits_arith.py` from the JSONs; `relMSE` columns are the two-sided-RHT
(the candidate arm's own basis) relative Frobenius weight errors, `xEXL3` is measured
against the pinned ref.py reconstruction of the same tensor. Rows marked
`NOT_RATE_MATCHED` (plain infinite-lattice stand-ins) and `UNSCALED_unitvar_BROKEN`
(the Lloyd-Max grid applied without a scale) and `*_control` are kept for provenance
and must not be quoted as formats.

```
| quantizer | bpw | Gauss MSE | xD(R) | gdn_in_proj_qkv relMSE | xEXL3 | mlp_down_proj relMSE | xEXL3 | GB/s @1.30x | GB/s @1.44x | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| e8_plain_rate4_ref | 4.000 | 4.7823e-03 | 1.224 | - | - | - | - | 495.3 | 548.6 | possible at both gates |
| e8rvq4_codec | 4.002 | 9.2557e-03 | 2.369 | 9.6926e-03 | 1.646 | 9.6289e-03 | 1.651 | 495.5 | 548.9 | possible at both gates |
| e8rvq4_rowscale_search | 4.002 | 8.6202e-03 | 2.207 | - | - | - | - | 495.5 | 548.9 | possible at both gates |
| e8rvq4_standin_absmax4_NOT_RATE_MATCHED | - | 1.2640e-03 | 0.324 | 2.6222e-03 | 0.445 | 3.2627e-03 | 0.559 | - | - | - |
| e8rvq4_standin_rms_NOT_RATE_MATCHED | - | 2.6024e-06 | 0.001 | 8.2295e-06 | 0.001 | 8.2322e-06 | 0.001 | - | - | - |
| exl3_4bpw | 4.030 | - | - | 5.8902e-03 | 1.000 | 5.8327e-03 | 1.000 | 499.0 | 552.8 | possible at both gates |
| gptq_sym4_g128 | 4.125 | 1.2257e-02 | 3.138 | 1.2264e-02 | 2.082 | 1.2263e-02 | 2.102 | 510.8 | 565.8 | possible at both gates |
| gptq_sym4_g128_fp32scale_control | - | 1.2258e-02 | 3.138 | - | - | - | - | - | - | - |
| gptq_sym4_g64 | 4.250 | 1.0563e-02 | 2.704 | 1.0562e-02 | 1.793 | 1.0563e-02 | 1.811 | 526.3 | 582.9 | possible at both gates |
| gptq_sym4_g64_fp32scale_control | - | 1.0564e-02 | 2.704 | - | - | - | - | - | - | - |
| gptq_sym_scalar_match_exl3 | 4.630 | - | - | 5.8902e-03 | 1.000 | 5.8327e-03 | 1.000 | 573.3 | 635.1 | possible at 1.30x; DEAD at 1.44x |
| int8_row | 8.000 | 1.3765e-04 | 0.035 | - | - | - | - | 990.6 | 1097.3 | DEAD-by-arithmetic |
| lloyd_max4_UNSCALED_unitvar_BROKEN | 4.000 | 9.5012e-03 | 2.432 | 5.4643e+01 | 9276.813 | 1.1930e+02 | 20453.348 | 495.3 | 548.6 | (defect row) |
| lloyd_max4_g128_absmax | 4.125 | - | - | 8.8307e-03 | 1.499 | 8.8311e-03 | 1.514 | 510.8 | 565.8 | possible at both gates |
| lloyd_max4_g128_rms | 4.125 | - | - | 9.3262e-03 | 1.583 | 9.3288e-03 | 1.599 | 510.8 | 565.8 | possible at both gates |
| lloyd_max4_g64_rms | 4.250 | - | - | 9.1731e-3 | 1.557 | 9.1737e-03 | 1.573 | 526.3 | 582.9 | possible at both gates |
| nf4_g128 | 4.125 | 9.1372e-03 | 2.339 | 9.1328e-03 | 1.551 | 9.1391e-03 | 1.567 | 510.8 | 565.8 | possible at both gates |
| nf4_g64 | 4.250 | 8.4640e-03 | 2.167 | 8.4593e-03 | 1.436 | 8.4611e-03 | 1.451 | 526.3 | 582.9 | possible at both gates |
| uniform_int4_opt | 4.000 | 1.1537e-02 | 2.953 | 1.1699e-02 | 1.986 | 1.1594e-02 | 1.988 | 495.3 | 548.6 | possible at both gates |
```

Ranking in the currency of the acceptance table (weight MSE per bit, real tensors):
**NF4 G64 1.44x EXL3 < Lloyd-Max-G128-absmax 1.50x < E8RVQ4 1.65x < GPTQ G64 1.79x
< uniform-global 1.99x < GPTQ G128 2.09x**. No fixed-rate scalar or lattice format
measured here reaches EXL3's weight MSE at ≤ 4.6 bpw except by paying the bits the
arithmetic §4 shows are unaffordable.

---

## 3. Part 3 — does the scalar-vs-rate-distortion explanation hold?

Measured KL from the sealed-context unsealed screen
`results/2026/10/rq27-r0-prefix-smoke-20261011-UNSEALED/smokeL11.out:74-79`
(13-layer prefix, layers 0-12, no Hessian defect this time):
bf16 0.0, exl3 0.00122, gptq_plain 0.00438, r4_g128 0.00411, r4_g64 0.00353,
r4_fp (bf16 storage, quantization off) 0.00013.

| arm | weight-MSE ratio to EXL3 (this lane, RTN proxy) | measured KL ratio to EXL3 | KL/weight ratio |
|---|---|---|---|
| exl3 | 1.00 | 1.00 | – |
| r4_g64 (= GPTQ grid G64) | 1.79 (1.79/1.81) | **2.89** (2.94 after subtracting the 0.00013 storage floor) | 1.61 |
| r4_g128 (= GPTQ grid G128) | 2.08 (2.08/2.10) | **3.37** (3.64 floor-subtracted) | 1.62 |
| gptq_plain (= GPTQ grid G128, no rotation) | 2.09 (2.19 unrot proxy) | **3.59** (3.90) | 1.64 |

So the ordering is reproduced (G64 better than G128, both far above EXL3), and the
magnitudes are in the same regime, but **KL grows ~1.6-1.7x faster than weight MSE.**
That is expected and not a defect: the KL is Hessian-weighted (`tr(ΔW H ΔWᵀ)`), the
screen's arms differ from this lane's RTN in rounding (GPTQ error feedback) and in the
fold, and the per-tensor sensitivities are not uniform. **Conclusion: the
scalar-vs-rate-distortion explanation holds as the *dominant* term** — a fixed-rate
uniform scalar grid at 4 bits is 2.95-3.14x D(4) while EXL3 is 1.51x D(4), a 0.6-0.8
bit structural gap, which no rotation, fold or group size can remove (measured
above: RHT buys 1.04-1.05x in weight MSE; G64 vs G128 buys 1.16x).

**Prediction for E8RVQ4.** Its measured weight-MSE ratio to EXL3 is 1.645-1.651.
Using the conservative "KL proportional to weight MSE" reading it scores ~1.65x EXL3;
using the screen's observed 1.6x superlinearity it scores ~2.6x EXL3. The gate is
≤ 1.10x. There is no remaining mechanism in the format that closes a factor 1.5 in
weight MSE, so **E8RVQ4 fails the quality gate by construction** — consistent with the
lead's kill. [INFERENCE for the 2.6x figure: the 1.6x superlinearity is measured on
three arms only, on the folded R4 path, and is not a law.]

## 4. Part 4 — bit and bandwidth arithmetic (`bits_arith.py:57`)

Model (RL-119 + FormatReview update): EXL3 decode streams **381 GB/s logical at 4.0 bpw**;
decode time tracks streamed bits, so a candidate at `b` bpw must sustain
`381 * G * b / 4.0` GB/s logical to beat EXL3 by `G`. `G = 1.30` (route decode gate) and
`G = 1.44` (raised kernel gate). Pin = **608 GB/s**; `b*` = the bpw at which the
requirement equals the pin.

| candidate | bpw | GB/s needed @1.30x | GB/s needed @1.44x | verdict |
|---|---|---|---|---|
| any 4.0 bpw format | 4.000 | 495.3 | 548.6 | possible at 1.30x (81% of pin); 90% of pin at 1.44x |
| E8RVQ4 (2 fp16 scales / row) | 4.002 | 495.5 | 548.9 | same; and killed on quality anyway |
| GPTQ sym G128 | 4.125 | 510.8 | 565.8 | possible at 1.30x; 93% of pin at 1.44x |
| NF4 G128 / Lloyd-Max G128-absmax | 4.125 | 510.8 | 565.8 | as above — **best quality per bit measured here** |
| GPTQ sym G64 / NF4 G64 | 4.250 | 526.3 | 582.9 | possible at 1.30x; 96% of pin at 1.44x |
| **scalar format sized to match EXL3's weight MSE** | **4.63** (4.505/4.512 + 0.125 scale overhead) | **573.2** | **634.9 > 608** | **DEAD at the 1.44x kernel gate**; 94% of pin at 1.30x |
| INT8 per-row (W8A16) | 8.000 | 990.6 | 1097.4 | DEAD-by-arithmetic |

`b*` values: at G=1.30, any format above **4.91 bpw** is dead by arithmetic; at
G=1.44, above **4.43 bpw**. The scalar path needs 4.63 bpw to match EXL3's weight MSE,
so at the raised bar it is dead, and at the 1.30x bar it needs 94 % of the pin (i.e.
essentially all of the memory bandwidth of the card, while EXL3 achieves 63 %).
Conclusion: **the scalar route cannot be fixed by bits**; equivalently ~4.4-4.6 bpw is
the ceiling any fixed-rate scalar format can be pushed to while keeping a decode
advantage, and its weight MSE there is at best EXL3-equal. A 3.0-3.2 bpw scalar format
would clear the 1.30x bar with room (371-396 GB/s), but its weight MSE is 4-8x worse.

---

## 5. Stock-path mixed-precision table (lead request)

`mixed_precision.py` (measured, `mixed_precision.json`): INT4 G128 RTN and INT8
per-output-channel RTN on the two-sided-RHT basis, on eight sampled tensors that span
both block types and depths 1/3/31; the 407 layer-linears (24.699 G params) come from
the checkpoint headers only (`mixed_precision.py:68`). Ranking is a **declared input**:
this lane has no activation Hessian, so the primary order is the structural prior
(mlp.down_proj → attention output projection → gate/up → in_proj/qkv), plus a
measured-INT4-MSE order and a parameter-count order for the spread.

Measured inputs: INT4 G128 rel MSE 1.2211e-2 … 1.2257e-2 (spread **0.4 %** — the
format side `c_t` is uniform across tensors after the RHT); INT8 rel MSE 7.75e-5 …
8.97e-5. INT4-only parameter-weighted model rel wMSE = **1.2253e-2**.

| k tensors at INT8 | params INT8 | bpw | predicted rel wMSE | vs INT4-only | GB/s for 1.62x | stock kernel speed vs EXL3 |
|---|---|---|---|---|---|---|
| 0 | 0.00 G | 4.125 | 1.2253e-2 | 1.000 | 636.5 | 1.669 |
| 8 | 0.71 G | 4.240 | 1.1902e-2 | 0.971 | 654.3 | 1.624 |
| 16 | 1.43 G | 4.356 | 1.1550e-2 | 0.943 | 672.1 | 1.581 |
| 32 | 2.85 G | 4.587 | 1.0848e-2 | 0.885 | 707.8 | 1.501 |
| 64 | 5.70 G | 5.049 | 9.4430e-3 | 0.771 | 779.1 | 1.364 |
| 128 | 7.78 G | 5.384 | 8.4226e-3 | 0.687 | 830.8 | 1.279 |
| 407 (all) | 24.70 G | 8.125 | 8.4625e-5 | 0.007 | 1253.7 | 0.847 |

Reading:
- The **1.62x anchor survives only 8 tensors** (0.71 G params, bpw 4.24, predicted
  weight MSE −2.9 %). At 64 tensors (bpw 5.05) the predicted weight MSE improves 23 %
  while the anchor decays to 1.36x — still above the 1.30x route gate, but the
  improvement is worth only 0.19 bit of distortion for 1.05 extra bpw spent.
- Arithmetic warning (recorded in the JSON `anchor_note`): `381 * 1.62 * 4.25/4 =
  636 GB/s > 608 GB/s pin` — the measured 1.62x stock W4A16 advantage is **not**
  explainable as a pure 4.25-bpw byte stream at EXL3's achieved logical rate, so the
  GB/s column is relative, and the honest speed statement is the bpw ratio:
  `speed vs EXL3 = 1.62 * 4.25 / bpw` (last column). Under that model the 1.30x gate is
  kept up to **5.30 bpw** and the 1.62x anchor up to exactly 4.25 bpw.
- Mixed precision is therefore a weak lever *if the ranking is by param count* (the
  predicted gain tracks the INT8 parameter fraction). It is a strong lever only if the
  per-tensor sensitivity spread is large — **which it is: see §5b.**

### 5b. Mixed precision re-run on the harness's MEASURED sensitivities (2026-10-12)

HarnessFmt's weighted prefix (`/mnt/<nvme-disk>/b70-run4-20261011/harness-fmt/weighted-resume13/raw/layers.jsonl`,
13 layers, 81 linears, 4.944 G weights, arms bf16/exl3/int4_g64_rht) supplies the real
per-tensor sensitivity `u_t = tr_dW_H_dWt / numel` (`k1_analyze.py`). Ranking by
`u_t` descending and moving tensors to INT8 (`D_t` from 4 to 8 bits at +4 bpw on that
tensor, INT8 gap `gap8 = 1.376487e-4/2^-16` measured by this lane):

| k tensors at INT8 | weight fraction | bpw | proxy gain | sensitivity mass covered | GB/s @1.30x | GB/s @1.44x |
|---|---|---|---|---|---|---|
| 0 | 0.000 | 4.125 | 1.000 | 0.000 | 510.8 | 565.8 |
| 4 | 0.031 | 4.248 | 1.335 | 0.254 | 526.0 | 582.7 |
| 8 | 0.058 | 4.358 | 1.527 | 0.349 | 539.7 | 597.8 |
| 16 | 0.107 | 4.553 | 1.972 | 0.499 | 563.8 | **624.5 (dead)** |
| 24 | 0.170 | 4.804 | 2.724 | 0.641 | **594.8 (98 % of pin)** | 658.9 (dead) |
| 32 | 0.306 | 5.351 | 4.727 | 0.798 | **662.6 (dead)** | 733.9 (dead) |
| 81 | 1.000 | 8.125 | 83.858 | 1.000 | 1006.1 (dead) | 1114.4 (dead) |

So with the measured order the lever is **much** stronger than the param-count table:
8 tensors (5.8 % of the weights, +0.23 bpw) yield a 1.53x proxy gain; 16 tensors
(+0.43 bpw) 1.97x; 24 tensors (+0.68 bpw) 2.72x but at 98 % of the pin, i.e. only
usable at the 1.30x gate. At the raised 1.44x gate the ceiling is ~8 tensors.
**And the measured order contradicts the structural prior**: the most sensitive are
`in_proj_z / in_proj_qkv` and `q/k/v` of the earliest layers, while `mlp.down_proj`
(early layers) and `attn.o_proj` are the *least* sensitive under this metric — the
opposite of what §5 assumed. Caveat: with a 13-layer prefix the sensitivity decays with
depth, so part of the ordering is prefix geometry; the full 64-layer ranking must come
from the all-layer run. The metric is a sum of layer-local weighted errors (see §6).

## 6. K1 of the XTQ design note — reverse water-filling (`waterfill_k1.py`)

Derivation (implemented at `waterfill_k1.py:45-134`, verified numerically):
with `D_t(R_t) = n_t u_t 2^(-2 R_t)`, `u_t = s_t c_t` the per-weight KL at rate 0,
and `sum_t n_t R_t = B N`, the KKT condition gives
`R_t = R̄ + ½ log₂(u_t / GM(u))`, `D* = N · GM(u) · 2^(-2 R̄)`,
`D_uniform = N · AM(u) · 2^(-2 R̄)`, so **gain = AM(u)/GM(u)**.

Validation (`k1_waterfill.json`): (a) the closed form's KKT conditions were checked by
20 000 random weighted bit transfers — maximum fractional improvement found: **0.0**
(no perturbation improves it); (b) for lognormal spreads AM/GM matches
`exp(σ²/2)` to <1 %; (c) the **realizable** integer-rate gain on the measured scalar
gap curve is smaller than AM/GM:

| log-sensitivity σ | AM/GM (theory) | AM/GM (measured) | realized integer-rate gain @4.0 bpw avg |
|---|---|---|---|
| 0.10 | 1.005 | 1.005 | 1.000 |
| 0.25 | 1.032 | 1.031 | 1.001 |
| 0.50 | 1.133 | 1.134 | 1.077 |
| 0.75 | 1.325 | 1.314 | 1.249 |
| 1.00 | 1.649 | 1.628 | 1.545 |
| 1.50 | 3.080 | 3.152 | 2.978 |

So the K1 gate (AM/GM ≥ 1.10 → keep the allocation) corresponds to a **per-weight
log-sensitivity standard deviation σ ≥ 0.44**, and the integer per-tensor bits
(XTQ: 3/4/5) recover ~85-95 % of the continuous gain for σ ≥ 0.5.

Provisional reading, **superseded by the CLOSED run below** (`--layers .../RQ-27-.../raw/layers.jsonl
--arm r4_g64`, 2 layers × 6 linears, `rel_fro` = weight-only proxy, not Hessian-weighted):
AM/GM = 1.100 — at the gate, with no integer allocation gain (1.000). That number is a
weight-only proxy over two layers and must not be quoted in place of the closed one.

**Structural result that matters more than the provisional number:** the *format* side
of the sensitivity is uniform — measured INT4 G128 rel MSE varies by 0.4 % across
tensor classes after the RHT (§5) — so `u_t = s_t c_t` with `c_t ≈ const` and
**K1's AM/GM is the AM/GM of the per-tensor Hessian sensitivity alone.**

**K1 is now CLOSED on the harness's real weighted prefix** (2026-10-12;
`k1_real_analysis.json`, `k1_analyze.py`, source
`/mnt/<nvme-disk>/b70-run4-20261011/harness-fmt/weighted-resume13/raw/layers.jsonl`,
13 layers × 81 linears × 3 arms, 4.944 G weights per arm):

| arm | AM/GM | σ_log | r_t median | R* (unconstrained) | gain, continuous R≥0 | gain, integer 2..7 | gain, XTQ 3/4/5 |
|---|---|---|---|---|---|---|---|
| int4_g64_rht | **5.854** | 1.88 | 1.6004e-3 | [-1.02, 7.09] | 5.781 | 4.436 | **2.461** |
| exl3 | **5.330** | 1.83 | 1.6498e-3 | [-0.35, 7.03] | 5.320 | 4.560 | **2.495** |

- The **negative unconstrained rates are inherent** to the Gaussian R-D form (no rate
  floor; `D_t(0) = n_t u_t` = "send nothing"), so the valid numbers are the R≥0 and
  integer columns. HarnessFmt reproduced the integer 2..7 numbers (4.4363 / 4.5602)
  with the same command — the pipeline is reproducible across lanes.
- **K1's gate (AM/GM ≥ 1.10) passes by a wide margin**, and even the format-realizable
  allocation (XTQ's 3/4/5-bit rate set, average 4.0 bpw) is worth **2.46x** in the
  metric. Rate map for int4_g64_rht/exl3: 31 tensors at 3 bits, 12 at 4, 38 at 5.
  Robustness: trimming the bottom 1 %/5 %/10 % of tensors by u gives AM/GM 5.25/4.12/3.56,
  so the result is not a single-outlier artifact.
- **Risk that must travel with the number**: at a fixed average bpw any gain has to come
  from *taking* bits somewhere. The 3/4/5 allocation degrades **41 % of the weights**,
  by up to **3.24x** per tensor (4→3 bits). The metric is a *sum* of layer-local
  Hessian-weighted errors; the model's KL is plausibly dominated by a few tensors, so a
  sum-optimal allocation can lose KL while winning the proxy. Recommend implementing the
  rate map as a screen arm (K0/K5 comparator), never as a default, and capping the
  degradation of the top-N most sensitive tensors.
- Cross-lane consistency: the weighted relative error r_t median 1.60e-3 (int4_g64_rht)
  vs this lane's independent plain-Frobenius G64 relative error 1.06e-2 on the same
  layer-2 tensors — the H-weighted metric is 6.6x more forgiving, as expected for a
  GPTQ-shaped error; the two lanes do not contradict each other.
- K1 does not change the format conclusion: it is a *bit allocation* lever over whatever
  format is chosen (worth ≤ 0.65 bits of effective rate at the XTQ rate set), and it is
  independent of the §1-§4 finding that no fixed-rate scalar/lattice format matches
  EXL3's weight MSE below ~4.5 bpw.

---

## 7. Caveats / status

- All candidate numbers are **RTN** (round-to-nearest), no Hessian, no GPTQ error
  feedback. The screen's arms use GPTQ+LDLQ, which is why §3 compares *ratios*, not
  absolute values.
- E8RVQ4 real-tensor numbers use the first 256 output rows of each tensor
  (`e8rvq_slice.py:28`, `SLICE_ROWS=256`) because the codec encodes ~5-7 k 8-vectors/s
  (≈1.9e-5 s/weight): a whole-tensor pass is hours. Slice RMS vs full-tensor RMS:
  0.01619/0.01573 (qkv) and 0.01092/0.01097 (down) — within 3 %, and the Gaussian
  Monte-Carlo value is reproduced by the slice to 0.1 %.
- [UNVERIFIED] QTIP Table 1 rows are cited from
  `research/2026/b70-xtq-format-design-20261011.md` §1 and were not re-measured here.
- [UNVERIFIED] The KL superlinearity factor 1.6-1.7x in §3 rests on three arms from one
  13-layer prefix screen; it is used only as a direction.
- The mixed-precision ranking is a declared structural prior, not a Hessian measurement
  (§5); the measured INT4-MSE ranking is deliberately included so the spread is visible.
- `lloyd_max4` as first written assumed unit variance and gives nonsense (5e1-1e2 rel
  MSE) on real weights; the scale-normalised version (`quant.py:131`) is the one used in
  §2 (`quant_extra.py` records both, the raw rows are retained in the JSON).
- No GPU was touched by this lane; no sealed run, no registry edit, no push.

## 9. Conclusions and recommendation

1. **E8RVQ4 is killed by weight MSE, robustly.** Contract-compliant (per-output-row
   fp16 scale pair, 4.0 bpw) it reaches 2.21-2.45x D(4) on i.i.d. Gaussian and
   1.65x EXL3's weight MSE on the two real tensors. The gate needs ≤ 1.10x. Neither
   the codebook (E8P12 measured G(E8)=0.0717) nor a wider per-row scale search
   (2.207x vs the codec's 2.45x) moves it. Under the screen's observed 1.6x KL
   superlinearity its predicted KL is ~2.6x EXL3, under the most conservative
   reading 1.65x. Recommend closing E8RVQ4 in the ledger with this number, and
   keeping E8DecodePrice compile-only.
2. **The scalar route cannot be repaired with bits.** A scalar GPTQ grid needs
   4.505/4.512 bpw to match EXL3's weight MSE on these tensors; with the G128 scale
   overhead that is 4.63 bpw, which at the raised 1.44x kernel gate requires 635 GB/s
   > the 608 GB/s pin (DEAD-by-arithmetic) and at the 1.30x gate requires 573 GB/s =
   94 % of the pin. The measured ceiling where any fixed-rate scalar keeps a 1.30x
   decode advantage is 4.91 bpw.
3. **Highest-value cheap lever found: replace the uniform symmetric grid with a
   nonuniform 16-level grid (NF4-style or Lloyd-Max) under the same two-sided RHT and
   the same per-group absmax/RMS scale.** Measured worth 0.16-0.24 bits of weight MSE
   (1.4x) at zero bit cost, i.e. about half of the remaining gap to EXL3, with a
   decode that is *cheaper* than the current one if the 16-entry table lives in
   registers (one lane-wise select/permute instead of sub+mul+fma) — this is the
   cheapest possible Xe2 decode and does not need trellis state, DPAS layout work or
   a re-quantized checkpoint structure. Recommend XeTrellisCode price this
   "LUT-scalar-16" decode as the floor case against which the trellis decode must
   show its ≥1.4x.
4. **K1 passes, with a caveat that must travel with it.** On HarnessFmt's real weighted
   prefix (13 layers × 81 linears × 4.944 G weights) AM/GM = **5.854** (int4_g64_rht) /
   5.330 (exl3), and even the format-realizable XTQ allocation (3/4/5 bits, average 4.0)
   is worth **2.46x / 2.50x** in the layer-local Hessian-weighted metric (integer 2..7:
   4.44x / 4.56x). The unconstrained optimum has negative rates, so only the R≥0 and
   integer columns are valid. Caveat: any fixed-average gain requires taking bits from
   somewhere — the 3/4/5 map degrades 41 % of the weights by up to 3.24x, and the metric
   is a sum of layer-local errors while the KL is not; implement the rate map as a screen
   arm, not as a default. Value ≈ 0.65 bits of effective rate.
5. **The same measured sensitivities make mixed precision on the stock path a real
   lever** (§5b), and they contradict the structural prior used earlier: the most
   sensitive tensors are early-layer `in_proj_z/in_proj_qkv` and `q/k/v`, not
   `mlp.down_proj`. 8 tensors at INT8 (+0.23 bpw, 5.8 % of weights) = 1.53x proxy gain;
   16 tensors (+0.43 bpw) = 1.97x; 24 (+0.68 bpw, 98 % of pin) = 2.72x and only at the
   1.30x gate. At the 1.44x gate the ceiling is ~8 tensors. Re-ranking on the full
   64-layer run is the next step.

## 10. Scripts index (this lane)

| file | purpose |
|---|---|
| `quant.py` | all quantizers + E8 nearest-point + `rel_mse` |
| `selfcheck.py` | orthonormality, E8 vs brute force, G(E8), Lloyd-Max cross-check |
| `gauss_rows.py` | part 1 Monte-Carlo rows (2e7 samples, seed 20261011) |
| `gauss_mc.py` | part 1 original driver incl. the scalar bpw curves (slow, kept for provenance) |
| `real_tensors.py` | part 2: RHT, candidates, EXL3 ref.py decode, bpw-to-match scan |
| `quant_extra.py` | scale-normalised Lloyd-Max rows added to `real_tensors.json` |
| `e8rvq_slice.py` | E8RVQ4 codec measurement on 256-row tensor slices |
| `mixed_precision.py` | §5 stock-path mixed-precision table |
| `waterfill_k1.py` | §6 K1 water-filling (formula + synthetic + real-schema reader) |
| `k1_analyze.py` | §6 CLOSED K1 on the harness's real weighted prefix (AM/GM, constrained+integer allocations, risk metrics, §5b mixed precision by measured sensitivity) |
| `bits_arith.py` | §4 arithmetic + the final comparison table (`format_theory_table.md`) |
