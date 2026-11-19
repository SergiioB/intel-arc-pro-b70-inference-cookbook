---
title: "Quant and EXL3 path: what a better EXL3 kernel or a better weight format is worth on the B70, priced"
type: research
status: draft
date: 2026-11-19
agent: quantpath
---

# Quant and EXL3 path, Qwen3.8-27B on the B70

Research, arithmetic and one CPU-only measurement. No GPU job, no build, no registry edit. Labels: **measured**
(file on this host, or a CPU read of a model file today), **quoted** (primary source opened today, URL in §6),
**estimated** (formula given), **lead** (community post), **NOT VERIFIED** (not opened or not run).

## 1. Answer first

1. Do not fund more exl3xpu kernel work. Best case is +3–8 % serial (32.7 → 33.8–35.2 tok/s) and +13–23 % MTP3
   (83.2 → 94–102). That is still below TensorFold on one card (127.8). It does not touch exl3xpu's non-repeatable output.
2. MTP3 is capped by the drafter, not the kernel. Yield is 2.90 tokens a round (survival 0.79/0.65/0.46), the hard
   limit is 4, and a round costs 1.20 serial steps. So the ceiling is 3.3x even with perfect acceptance.
3. EXL3 weights inside TensorFold are a wash at 4.0 bpw: 116–130 drafted against 128 today. The byte saving (−9 %)
   pays for the trellis decode (+7–20 % weight time). They pay only at 3.0–3.5 bpw: 125–154 drafted on one card.
4. The real lever is fewer bytes with today's decode. TensorFold's int4 kernel turns each nibble into an fp16
   subnormal for DPAS (the Xe2 matrix instruction) in about 1 ALU op per weight and streams at 88 % of pin (the
   card's 608 GB/s peak). Any sub-4-bit format must keep that.
5. Priced at 77–88 % of pin, a 3.55 bpw format with that decode gives 38.8–40.0 serial and 148–152 drafted on one
   card (+16–19 %). On two cards it gives 58–59 serial and 181–183 drafted (+10–11 %). At 131K it gives about +8 %.
6. No existing format is both cheaper to decode than EXL3 and proven at 3–3.5 bpw on this model. Codebook and VQ
   formats (AQLM, VPTQ, GPTVQ) decode by gather. Ternary needs training that does not exist for this model.
   Entropy coding is capped: the MLX 4-bit codes on disk carry **3.71 bits** of entropy (measured), at most 7 % saving.
7. Proposal: "B70-native" bit-plane affine. Per-tensor widths 3/4/6 chosen by sensitivity, group 128, fp16
   scale and bias, and the same nibble-to-subnormal DPAS path. About 3.55 bpw. Rotation only if needed.
8. The riskiest assumption is quality, not speed. It can be tested with zero engine code. A 3-bit affine code is
   exactly representable in MLX 4-bit storage, so we can quantize with AutoRound, repack, serve on unchanged
   TensorFold, and run the 329-item battery. This costs about 1 day and 4–5 GPU hours.
9. Pass bar for that test: battery ≥ 279 (stock 282 − 3), net flips ≤ 3, the 14-vs-30 canary right, drafted ==
   serial, and DFlash2 yield ≥ 5.1 tokens a round. Only then write the bit-plane kernel (2–3 days, gate ≥ 0.95x
   int4 GB/s).
10. Better drafter for exl3xpu: park. It prices at 96–139 tok/s, fixes nothing on repeatability, and needs DFlash2
    support in a vLLM 0.26 fork. That support is NOT VERIFIED.

## 2. Ranked table

One card unless marked. Today's route is TensorFold MLX 4-bit g64 + DFlash2: **measured** 32.4 serial / 127.8
drafted on one card, and 50.7 / 165.5 on two cards. Priced columns use the model in §5.1, which reproduces those
four numbers within 0.6 %.

| # | Option | Eff. bpw | Priced 1 card serial / drafted (2 cards) | Quality evidence | Decode class on Xe2 | Cost | Risk | First experiment |
|---|---|---|---|---|---|---|---|---|
| 1 | **B70-native bit-plane affine, mixed 3/4/6, g128** (§4) | 3.55 (est.) | 38.8–40.0 / 148–152 (58–59 / 181–183) | None for this model at ~3.5 bpw. Risk evidence below | cheap integer: shift, mask, OR into fp16 subnormal, then DPAS | Test 1 d; kernel 2–3 d; quantizer pipeline 3–5 d; engine wiring 5–8 d | **High** on quality, low-medium on kernel | E1 (§4.6): AutoRound 3-bit and mixed 3/4 in an MLX 4-bit container, battery on TF |
| 2 | **EXL3 3.0–3.5 bpw inside TensorFold** | 3.15–3.62 | 3.0: 37.6–40.3 / 136–154 (57–60 / 171–185); 3.5: 33.9–36.2 / 125–141 | Measured: 3.0 bpw 6/6 canaries x2, tasks 9/10 (RL-094). Quoted: QTIP 3-bit Llama-2-70B ppl 3.26 vs FP16 3.12 | computed code: window extract + int32 multiply + byte sum + FMA (about 0.33 issue/weight, RL-176) | 10–20 d | Medium-high: decode cost unproven in TF's DPAS path | E0: battery x3 for turboderp 3.00bpw on exl3xpu; `XPU_EXL3_PROBE` in TF (5 min) |
| 3 | MLX/affine 4-bit **g128** asym | 4.25 | 34.2 / 133.8 (52.8 / 169.5) | None at g128 for this model. **Measured risk:** W4 g128 *symmetric* GPTQ/AutoRound exports flip the 14-vs-30 canary (RL-084/092/117) | cheap integer (today's kernel, compiled for g128) | 2–3 d | Low-medium | RTN or AutoRound g128 asym, repacked; battery |
| 4 | exl3xpu kernel work (option a) | 4.12 | 33.8–35.2 / MTP3 94–102; C=8 ≤ ~370 | Battery 286–287 (measured) but not repeatable | trellis (unchanged) | 10–20 d | High: 6+ prior variants gave ≤ +10 % on one family (RL-101/102/112/184) | none recommended (§3.1) |
| 5 | exl3xpu + DFlash2 drafter (option c) | 4.12 | drafted 96–119; with (a) 120–139 | as #4 | trellis | 10–15 d | High: support in the fork NOT VERIFIED; exactness unfixed | grep the image for `DFlash2DraftModel` (CPU) |
| 6 | Low-rank + quantised residual (LQER/CALDERA class) on top of #1 | +0.1–0.2 | −2 to −4 % against #1 | Quoted: LQER "nearly-lossless W4A8" | extra small GEMV per linear | +3–5 d | Medium | only if E1 Arm B fails narrowly |
| 7 | VQ 2–2.3 bpw (AQLM, PV-tuning, VPTQ, GPTVQ) | ~2.25 | bytes alone give 58 / 203. The real number is lower: gather decode will not stream at 88 % | Quoted: QTIP Table 4 AQLM 81.5 vs QuIP# 186 tok/s (7B, RTX 6000 Ada); no 27B number | gather (AQLM 1x16 codebook ~1 MiB > 128 KB SLM); small LUT for GPTVQ | weeks + CUDA GPU-days | High | park |
| 8 | 2:4 sparse + 4-bit | ~3.25–3.5 | ≤ #1 | Quoted: SparseGPT "generalizes to semi-structured (2:4 and 4:8)", no numbers | scatter into a dense operand; no sparse DPAS known on Xe2 (NOT VERIFIED) | weeks + recovery training | High | park |
| 9 | Entropy-coded 4-bit codes | ≥ 4.21 | ≤ +7 % on bytes, minus decode | **Measured:** code entropy 3.71 bits | sequential (Huffman/ANS) | — | — | closed by measurement |
| 10 | Ternary / 2-bit QAT (BitNet, ParetoQ) | 1.6–2.25 | bytes alone give 58+ / 200+ | Quoted only for models trained that way; no Qwen3.8-27B QAT int release found | cheap | pretraining scale | — | closed: no artifact; training is out of reach on two B70s |

How to read the table: options 1 and 2 are the only ones that move single-stream speed by more than 5 % with
evidence. Option 1 is faster if its quality holds. Option 2 has the better quality evidence at 3 bits.

## 3. Part 1: the EXL3 kernel question

### 3.1 Option (a): improve exl3xpu's own decode kernels

**Why the earlier stop happened, and whether the reason still holds.** RL-122 (2026-10-07) recorded the operator's
order: "I dont want to try the 0xsero one. completely discard that. use both gpus for best recipes if needed, stop
everything." It closed the kernel-probe program (RQ-01 VERIM4, RQ-02 geometry sweep, RQ-03 residual split). The
reason was priority, not a technical failure. RL-180 (2026-10-12) reopened exl3xpu source work only for mixed-K
serving, as a new image tag, and kept RQ-01/02/03 closed. AGENTS.md §7: "Operator-stopped EXL3 kernel work stays stopped."

The priority reason still holds, and it now has a technical side too. Since then, TensorFold has beaten exl3xpu at
short context (drafted 127.8 vs 83.2) and on exactness (8/8 vs 2/16). exl3xpu's three remaining wins are
250K on one card, 8 streams and cold prefill (routecmp2 §1). They come from fp8 K/V, int8 prefill linears and
vLLM batching. None of them comes from the decode kernel. The operator can lift the hold. This report advises against it.

**Time split, measured.**

| Fact | Value | Source |
|---|---|---|
| Serial token | 30.6 ms (32.7 tok/s), 13.18 GB a token = 431 GB/s effective (71 % of pin) | routecmp §3, §6 |
| M=1 linears only | 29.44 ms over 13.12 GB = 446 GB/s at GPU0 180 W; about 25.6–26.8 ms at 230 W (estimate: GPU1 at 230 W runs these kernels 10–15 % faster, RL-119) | RL-098 |
| M=4 verify pass | 34.46 ms = 381 GB/s (62.6 %) at 180 W. Family spread 10.6–83.5 % of pin | RL-098 |
| Trellis decode share at M=4 | Removing the decode chain (`EXL3_DEBUG_B_FAKE`, wrong output by design) speeds the kernels up +15 to +33 % at 230 W and +30 to +68 % at 180 W. That is 38 % fewer instructions | `results/2026/09/b70-m4-flag-ablation-20260930T143631Z/RESULTS.md` (RL-119) |
| Non-GEMM overhead at M=4 | "at the big gate_up shape about half of the production M=4 call is non-GEMM overhead, not trellis decode" | RL-184 |
| Clean-bench ceiling | A DPAS mul1 kernel in an isolated bench: 496–503 GB/s at M=4 (82 % of pin) | RL-183 |

So at M=1 the trellis decode costs about 8 % (TensorFold-class kernels reach 535 GB/s against about 495). At M=4 it
costs 15–33 %, and the per-call overhead around it (Hadamard kernels, split-K, torch dispatch) costs about as much again.

**Ceiling at 88 % of pin (535 GB/s), estimated.**

- Serial: `t = 13.18 GB / 535 GB/s + non-weight = 24.6 ms + (3.8 to 5.0 ms) = 28.4–29.6 ms` → **33.8–35.2 tok/s**
  (+3–8 %). Non-weight is 30.6 minus the 25.6–26.8 ms kernel estimate.
- MTP3 drafted: the round today is `3.05 tokens (RL-090) / 83.16 tok/s = 36.7 ms`. The M=4 pass is about
  34.46 × 0.88 ≈ 30 ms at 230 W. At 535 GB/s plus 4 % for 4 rows it would be 25.6 ms. That saves 4.4–6 ms, so the
  round becomes 30.7–32.3 ms → **94–102 tok/s** (+13–23 %). The ledger's own price was ×1.26 (RL-098: "63 →
  ~79 tok/s ... NOT reachable by kernel work alone" for 85–95).
- 8 streams: 49.6 tok/s per stream ÷ ~3 tokens a round ≈ 60 ms a round. The same 5–6 ms saving → **≤ ~370 tok/s
  aggregate** (+≤10 %) against 337. At 16 streams the round is longer and the share smaller: **≤ ~345** against 321.
  Estimates. The weight share at C=8 was never measured.

**What limits MTP3 at 2.5x (measured inputs, arithmetic).** Per-position survival on the serving route is
0.79 / 0.65 / 0.46 (RL-093), so the yield is 1 + 0.79 + 0.65 + 0.46 = 2.90 tokens a round. MTP3 can never
exceed 4. A round costs 36.7 / 30.6 = 1.20 serial steps, partly because the eager draft takes about 7.8 ms
(RL-187). Graph-capturing it did not help: RL-187 gave 0.96x / 0.91x, RL-188 parity. 2.90 / 1.20 = 2.42x, against
2.26–2.54x measured. Perfect acceptance gives 4 / 1.20 = 3.3x. The DFlash2 tree gives 5.39 tokens at 1.36 serial
steps = 3.94x. **The limit is the drafter's depth, not the kernel.**

**Cost, risk, stop.** 10–20 days. High risk. The ledger has RL-101 (MB=4: +0.4 pin points), RL-102 (+8–10 % on one
family), RL-112 (−3 to −6 %), RL-184 (K-specialised builds 0.58–0.92x) and RL-122 (stop). **Stop fact:** if one
fused decode-chain variant does not cut the weighted M=4 pass by ≥ 5 % in isolated timing (RQ-01's own gate), stop.
It also does nothing for repeatability: with MTP off, 1 of 8 prompts gave one text over 3 repeats (routecmp §3).
That is an engine property. Split-K reduction order is a candidate cause (hypothesis, NOT VERIFIED).

### 3.2 Option (b): EXL3 weights inside TensorFold

**What upstream carries.** `$TF/src-tp3586/zig/kernels/xpu/qwen27/decode.cl:209` defines `exl3_gemv`. It is a
speed probe on random trellis bits (`qwen27_main.zig:446-488`, `XPU_EXL3_PROBE`) for the mul1 codebook at 4, 3
and 2 bits, T = 1 and 8 rows. It decodes each weight with window extraction across two 32-bit words through
`sub_group_shuffle`, one 32-bit multiply (`s * 0x83DCD12D`), a 4-byte sum and one FMA. Then it does scalar fp32
FMAs per row. There is no DPAS, so it is not a verify kernel. It has no record of a run on this host: I searched
B70-DOCS and `/data/*/work` for its output lines. NOT MEASURED.

TensorFold core has the EXL3 format and a reference decoder (`zig/src/core/exl3_format.zig`, per
`docs/recipes/exl3.md`). CUDA serves dense Qwen3.8-27B EXL3 packs, including a 3.50 bpw pack (CHANGELOG #1475).
Metal has a projection kernel. On GB10 the TensorFold kernel reaches "221-226 GB/s of a 237 GB/s streaming
ceiling" at 6 bits, and "2 bits is where the decode itself is the work" (recipe, quoted from the local tree).

**Decode cost per weight on Xe2.**

- TensorFold int4 today: one 32-bit word gives two fp16 lanes per shift-and-mask
  (`(w0 >> 4k) & 0x000F000F`, `gemm.cl:116`). Each nibble becomes an fp16 subnormal q·2^-24 that goes straight into
  `intel_sub_group_f16_f16_matrix_mad_k16`. The scale is applied per group. The bias is applied by one small DPAS
  on group sums (`BIAS_TILE`). About 1 ALU op per weight. Weight kernels run at 88 % of pin (README §2).
- EXL3 mul1: about 0.334 issues per weight against 0.206 "needed for pin rate" at 4.0 bpw (RL-176, local arithmetic).
  At 3 bpw the budget per weight shrinks by 3/4 while the decode stays the same, so the gap widens.
- Calibration from measurements: a decode cost D per token, constant in bits. **D ≈ 2.2 ms** at M=1 (the RL-183
  bench kernel against TF's 535 GB/s, plus about 0.5 ms for fused Hadamard). **D ≈ 4 ms (serial) / 7 ms (16-row
  verify)** for exl3xpu-class kernels (B_FAKE share).

**Priced tok/s (estimated, §5.1 model, EXL3 effective bpw = body b + 6-bit head):**

| EXL3 width | Eff. bpw / GB | 1 card serial | 1 card drafted | 2 cards serial | 2 cards drafted |
|---|---|---|---|---|---|
| 4.0 | 4.10 / 13.14 | 31.0–32.8 | 116.5–130.0 | 48.8–51.0 | 155.0–166.5 |
| 3.5 | 3.62 / 11.61 | 33.9–36.2 | 125.4–141.2 | 52.5–55.1 | 162.6–175.3 |
| 3.0 | 3.15 / 10.09 | 37.6–40.3 | 135.7–154.4 | 56.7–59.8 | 171.1–185.2 |
| *today, MLX 4.5* | *4.50 / 14.42* | *32.4 (measured)* | *127.8 (measured)* | *50.7* | *165.5* |

At 131K the weight term is 37 % of the drafted round (PATHFINDER §2), so every gain above shrinks to about a third.

**DFlash2 exactness.** The weight format does not affect it, as long as the EXL3 kernel keeps TensorFold's contract:
"the bits of a row do not depend on the number of rows, on its position, or on the tiling" (`docs/xpu/NAVIGATION.md` §5).
That rules out atomics and split-K with a free reduction order. The trellis decode itself is deterministic.
What may change is acceptance: DFlash2 yields 5.39 tokens a round on the MLX target, and a different quantised
target may yield a little less. Measure it. Do not assume it.

**Two cards.** EXL3 tiles split at 128 boundaries are exact (RL-104: column split bitwise, row split exact to fp32
reassociation). TensorFold's `--tp 2` would need the same.

**Cost, risk, stop.** 10–20 days: loader wiring, a decode-into-DPAS kernel row-invariant from M=1 to 16, widths
3/4/6 and the half-bit windows of 3.5, fused Hadamard in and out, a prompt kernel, TP split. Medium-high risk.
**Stop facts:** (1) if turboderp 3.00/3.50 bpw on exl3xpu scores below 284 on the 329 battery (3-run mean, against
287 for 4.0), there is no quality reason to pay for the kernel. (2) If the first DPAS kernel at 3 bits does not
reach a time per weight ≤ 0.80x today's int4 g64 kernel at M=1 and M=16, the byte saving does not survive.

### 3.3 Option (c): a better drafter for exl3xpu

- The DFlash2 drafter on disk (`Qwen3.8-27B-DFlash2`, `DFlash2DraftModel`, 5 layers, block 8, bf16, 3.85 GB
  measured file size) is tagged `vllm` on its card. vLLM's DFlash support is in vllm ≥ 0.20 (vLLM blog, from a
  search summary; not opened). Whether exl3xpu's vLLM 0.26 fork registers `DFlash2DraftModel`, and whether it runs
  on XPU, is **NOT VERIFIED**.
- Priced (estimated): linear block of 8 with yield 4.3–5.0 tokens (no tree; the TF tree gets 5.39). Round = verify
  at M=9 (31–34 ms) + bf16 drafter read (3.85 GB / 500 GB/s ≈ 7.7 ms) + about 3 ms host ≈ 42–45 ms →
  **96–119 tok/s**. With option (a) at 88 % of pin: 36 ms → 120–139.
- It does not fix exactness. exl3xpu cannot use the second card for one stream either (RL-103: "exl3: tensor
  parallel sharding not supported yet").
- Cost 10–15 days. High risk. **Stop fact:** the image lacks the architecture class, or acceptance on the exl3 target
  is < 4.3 tokens a round.

## 4. Part 2: a format built for this card ("B70-native")

### 4.1 Requirements, as numbers

- Decode = yield × achieved bandwidth ÷ bytes a token (format guide §1). Achieved bandwidth today is 535 GB/s in
  the weight kernels. Do not give that up.
- Issue budget: at 88 % of pin and b bpw, the kernels must decode 535e9 × 8 / b weights a second. That is 0.95e12/s
  at 4.5, 1.21e12/s at 3.55 and 1.43e12/s at 3.0. Today's ~1 ALU op per weight fits. EXL3's does not fit fully at 4.0
  (RL-176), and fits less as bits fall.
- Quality: 329 battery within run-to-run spread (about 3 items), plus the in-head arithmetic canary (14, not 30).
- Prefill wants int8 operands (s8×s8 DPAS is the only 2x XMX path, format guide §1). K/V stays int8 or bf16.
  Exact speculation is independent of the format.

### 4.2 Design

| Part | Choice | Why |
|---|---|---|
| Code | Unsigned b-bit integer per weight, b per tensor ∈ {3, 4, 6} (8 for rare outliers) | One kernel template. Any code below 1024 is an exact fp16 subnormal q·2^-24, so the existing DPAS path takes it unchanged |
| Bit allocation | Per-tensor widths by measured sensitivity (final-logit KL per tensor, greedy under a byte budget), the method EXL3-SC uses (RL-179), applied to affine codes. Target code mix: 80 % of parameters at 3, 15 % at 4, 5 % at 6 (head and the most sensitive) | Allocation, not codebook, carried most of SC5's gain (RL-179: "the differences come from the body allocation") |
| Groups | 128 along K. Asymmetric: fp16 scale + fp16 bias per group per output column = 0.25 bpw. Fall back to g64 (+0.25) for 3-bit tensors only if E1 needs it | TF kernels already compile for g64 or g128. Upstream W4A8 prefill is hard-coded to g128 (README §3, H8) |
| Storage | **Bit planes**: 3 = 2-bit plane + 1-bit plane, 6 = 4 + 2, 4 = today's nibble plane. Each plane uses today's 2D-block tile layout (`intel_sub_group_2d_block_read_32b_8r16x1c`, pitch N×4) | Every plane is 32-bit aligned with no straddling windows. A k-step is one block read per plane |
| Kernel decode | Per 32-bit pair of fp16 lanes: `q = ((P2 >> 2t) & 0x00030003) \| (((P1 >> t) & 0x00010001) << 2)` → `as_half2` subnormal → DPAS fp16, fp32 accumulators. Scale per group on the partials (`SCALE_AT`), bias by `BIAS_TILE` | About 5 ops per 2 weights at 3 bits against about 2 today (estimate). Inside the 1.21e12 w/s budget |
| Row invariance | Unchanged summation order (K slices from 0, then `((t0+t1)+t2)+t3`, then bias) | Keeps drafted == serial |
| Prefill | The same codes, centered, as int8 DPAS operands with per-token int8 activations (W3/W4A8) | 2x XMX. Prompt only, so decode exactness is unaffected (prompt K/V bits change once; measured 7/8 outputs changed on the g64 W4A8 arm, PATHFINDER row 5) |
| Rotation | None in phase 1. Phase 2 only if E1 fails narrowly: block-128 sign+Hadamard on K, fused into the RMSNorm and SwiGLU epilogues | Online rotation changes the DFlash2 tap path if folded into the residual. Keep it off until it is needed |
| Head / embedding | Head at 6 bits (in the 5 %); embedding not read per token | |

**Expected bpw:** 0.80×3 + 0.15×4 + 0.05×6 = 3.30 code bits + 0.25 = **3.55 bpw**, 11.38 GB a token
(estimated). Priced: **38.8–40.0 serial / 148–152 drafted on one card, 58–59 / 181–183 on two cards**. The ranges
are 85–88 % of pin with the yield held at 5.39.

### 4.3 Honest quality risk

- The 4-bit codes on disk use the full range: the 16 levels hold 1.0–12.3 % each and the entropy is 3.71 bits
  (measured). So dropping to 3 bits by rounding alone will lose quality. Learned rounding (AutoRound/GPTQ) and
  allocation have to recover it.
- Scalar codes are worse than trellis codes at the same bits. QTIP's own table puts its 3-bit trellis at Llama-2-70B
  ppl 3.26 against FP16 3.12, and the next-best vector methods at 3.35–3.36 (quoted). I opened no primary source for
  scalar GPTQ/AutoRound 3-bit numbers on a modern 27B: NOT VERIFIED.
- This model's history: W4 g128 symmetric GPTQ and AutoRound exports invert the canary margin by about −3.3 logits
  and answer 30 (RL-084/092/117). EXL3 and Q4_K_M answer 14. A homemade INT4-pure AutoRound arm was 12.7x worse than
  EXL3 on block error (RL-161; harness flawed, optimiser not Intel's).
- Guess, labelled as such: uniform 3-bit passes the battery bar with probability under 25 %; mixed 3/4/6 at about
  3.55 bpw with probability 40–50 %. That is why the test comes before any kernel work.

### 4.4 Why not the alternatives as the base

- EXL3/QTIP/CPTC computed codes: better rate-distortion (CPTC "lowest KL divergence ... by up to 27 %" against QTIP
  codebooks, quoted). But every one decodes with ≥ 2 multiplies per weight plus window extraction. That puts them
  in the decode class that already costs 8–20 % at 4 bits here. CPTC's 64-byte table would fit in one register
  per subgroup (16 lanes × 4 bytes), which makes it the best computed-code candidate for Xe2 if option 2 is ever built.
- VQ/AQLM: gather from codebooks larger than SLM.
- Sparsity: no sparse DPAS on Xe2 that I could find, and it needs recovery training.
- Entropy coding: ≤ 7 %, sequential decode.
- QAT/ternary: no artifact. A third-party Qwen3.8-27B QAT exists only in NVFP4 (lead), which has no DPAS on Xe2.

### 4.5 What it would take to ship

1. Quantizer pipeline: AutoRound with a per-tensor `layer_config`, a sensitivity pass, and a repack to bit planes
   (3–5 days; CPU + one card).
2. TensorFold: `w4k1_*` and `w4k_dsplit` templated over planes, a loader for per-tensor widths, the g128 build,
   and W3A8 prefill (5–8 days).
3. Gates per AGENTS.md §9 and §11, then a route decision (b70-route-decision skill). Upstream acceptance is not
   assumed: the maintainer's rule is exact changes only, and a new weight artifact is a new model, not a token change.

### 4.6 Cheapest experiment for the two riskiest assumptions

**Assumption Q (quality at about 3.5 bpw affine). Experiment E1: no engine code, about 1 day, 4–5 GPU hours.**

1. Quantize `$DATA/sata1/models/Qwen3.8-27B-bf16` (59 GB, on disk) with Intel `auto-round` (pin the version; RL-117
   ran 0.14.2 on this host with patches) on card 1. Asymmetric, group 64, `iters` 200, 512 × 2048 calibration
   tokens of general text, disjoint from the battery. Two arms:
   - **A:** uniform 3-bit.
   - **B:** 3-bit, with 4-bit for about 25 % of parameters (down_proj, attention o/v, GDN out_proj, layers 0–1 and
     62–63), plus the head at 4.
   RL-117's precedent for this step: 64 blocks at about 70 s each, peak RAM 25.8 GB, VRAM 8.65 GB. That is about
   1.5–2.5 h per arm. Run under a memory cap with ≥ 20 GiB MemAvailable left.
2. Repack on CPU into an MLX 4-bit g64 container: codes 0–7 go into nibbles, `scale` stays, `bias = −scale × zp`.
   This is exact up to the bf16 rounding of the bias. Output goes to `$DATA/samsung980/work/quantpath-<date>/`.
3. Serve on the same TensorFold binary, drafter and flags as the route (`src-tp3586`, `XPU_ATTN_PREF=256`, one card).
   Run the 329 battery (`scripts/tmp/breakthrough_20261118/tp_battery.sh`, graded offline). Run drafted == serial on
   the 8 frozen prompts. Record DFlash2 tokens a round at 0 and 8K with `depth_bench.py`.
4. **Pass:** total ≥ 279 and net flips ≤ 3 against TF stock (282, 2026-11-13), JSON 55/55, canary 14,
   drafted == serial 8/8, tokens a round ≥ 5.1. **Fail:** total ≤ 276 or a canary flip. Then close scalar 3-bit at
   that allocation and go to rotation (phase 2) or option 2.

Speed does not move in E1 by design: storage stays 4.5 bpw. It tests quality and DFlash2 acceptance only.

**Assumption K (bit-plane decode stays at TF speed). Experiment E2: run only if E1 passes. 2–3 days of code,
about 15 GPU minutes.**

1. Add a 3-bit bit-plane variant of `w4k1_d64` and `w4k_dsplit` to `gemm.cl`, fed with random planes and scales
   (the `exl3Probe` pattern), plus a CPU reference.
2. Time M=1 and M=16 on gate|up (5120 → 34816), down (17408 → 5120) and GDN in_proj_qkv shapes, card 1, through the queue.
3. **Pass:** logical GB/s, counting the scale and bias bytes, ≥ 0.95x the int4 kernel's in the same run (about
   ≥ 508 GB/s). Bit-exact against the CPU reference. M=1 rows equal to M=16 rows. **Fail:** < 0.90x. Then re-price
   with the measured penalty, or stop.

**Experiment E0 for option 2 (in parallel, cheap).**

1. Run `XPU_EXL3_PROBE=1` on the dev runner (5 min, one card). A pass is informative; a fail is weak, because the
   probe is a non-DPAS kernel.
2. Download turboderp `3.00bpw` (13.8 GB per RL-089) to `$DATA/samsung980`. Run the 329 battery 3 times on the
   mixed-K exl3xpu image (K=3 kernels exist, RL-100/182) against 4.0's 287.

## 5. Arithmetic

### 5.1 Pricing model (inputs measured; outputs estimated)

```
W   = 25.64e9 linear params read per token (14.42 GB at 4.50 bpw, routecmp §6)
one card:  serial(b)  = 1 / (3.8 ms + 26.9 ms × k)            26.9 = 14.42 GB / 535 GB/s; 3.8 = 30.9 − 27.1 (README §2)
           drafted(b) = 5.39 / (11.1 ms + 30.9 ms × k)        round 42.0 = 30.9 weight + 11.1 rest (PATHFINDER §2, §9)
two cards: serial(b)  = 1 / (6.15 ms + 13.55 ms × k)          19.7 ms/token measured (TP-ROOT-CAUSES, job 110)
           drafted(b) = 5.39 / (17.2 ms + 15.45 ms × k)       32.65 ms round (PATHFINDER §9)
k = (b_eff / 4.5) × (0.88 / pin_fraction); EXL3 adds D (serial) and D16 (round), halved on two cards
EXL3 b_eff = (24.37e9 × b + 1.27e9 × 6) / 25.64e9            6-bit head
```

Check against measured: 32.6 / 128.3 / 50.8 / 165.1 vs 32.4 / 127.8 / 50.7 / 165.5. Script:
`/tmp/claude-1000/-home-sergio-B70-DOCS/802da85a-4a0f-4cc9-9f41-a5cc466983a0/scratchpad/price.py` (scratch;
the formulas above are the record).

| Format | Eff. bpw | GB/token | 1c serial / drafted | 2c serial / drafted |
|---|---|---|---|---|
| MLX 4b g64 (today) | 4.50 | 14.42 | 32.6 / 128.3 | 50.8 / 165.1 |
| affine 4b g128 asym | 4.25 | 13.62 | 34.2 / 133.8 | 52.8 / 169.5 |
| 4b g128 symmetric | 4.12 | 13.22 | 35.1 / 136.7 | 53.8 / 171.9 |
| IQ4_XS-class bytes | 4.20 | 13.46 | 34.6 / 135.0 | 53.2 / 170.5 |
| affine 3b g64 | 3.50 | 11.22 | 40.4 / 153.4 | 59.9 / 184.5 |
| B70-native 3.55, 88 % / 85 % | 3.55 | 11.38 | 40.0 / 151.9 ; 38.8 / 148.3 | 59.4 / 183.4 ; 58.1 / 180.8 |
| affine 3b g128 | 3.25 | 10.42 | 43.1 / 161.3 | 62.8 / 190.1 |
| 2-bit class, bytes only | 2.25 | 7.21 | 58.0 / 203.0 | 77.4 / 216.2 |

The drafted column assumes the yield stays at 5.39 tokens a round and the decode stays TF-class. Both are
assumptions that E1 and E2 test.

### 5.2 Entropy measurement (measured today, CPU only)

`python3 -I nib_entropy.py <Qwen3.8-27B-MLX-4bit>` read 22 packed uint32 weight tensors of layers 3, 31 and 62
(MLP, attention, GDN). Zero-order entropy of the 4-bit codes was 3.67–3.72 bits per tensor, **3.709 bits overall**.
Level frequencies ran from 0.0102 to 0.1225. Upper bound of the saving from entropy coding the codes:
(4 − 3.71) / 4.5 = 6.4 % of bytes. That is before paying for a sequential decoder.

## 6. Sources (opened 2026-11-19 unless marked)

| Source | Claim quoted | Use |
|---|---|---|
| QTIP, arXiv 2406.11235 (abs + HTML) https://arxiv.org/abs/2406.11235 | "QTIP introduces a spectrum of lookup-only to computed lookup-free trellis codes"; HYB: "Setting Q=9 gives a 2KiB codebook, which fits in L1 cache", "amortized 2 instructions per weight"; 3INST "can be implemented in 3 ALU instructions"; codes "≤4 hardware instructions per weight on NVIDIA GPUs"; matvec "at over 80% of peak memory bandwidth". Table 5 (Llama 2, ctx 4096, Wikitext2): FP16 7B 5.12 / 70B 3.12; QTIP 4-bit 5.17 / 3.16; 3-bit 5.28 / 3.26; 2-bit 5.86 / 3.70; QuIP# 3-bit 5.41 / 3.35; AQLM 3-bit 5.38 / 3.36. Table 4 (7B, RTX 6000 Ada): FP16 55.9, AQLM 81.5, QuIP# 186, QTIP 2/3/4-bit 188/161/140 tok/s | trellis quality and decode class. Table values were read through a fetch tool's text alignment: check the PDF before citing externally |
| EXL3 format doc v1.4.4 (via R6) https://raw.githubusercontent.com/turboderp-org/exllamav3/v1.4.4/doc/exl3.md | "EXL3 is a variant of QTIP"; "procedural codebook ... tail-biting trellis" | format identity |
| turboderp/Qwen3.8-27B-exl3 https://huggingface.co/turboderp/Qwen3.8-27B-exl3 | Branches: plain 2.00–6.00 (incl. 3.00, 3.50); SC_ 1.40–6.00; SCP_ 1.75–6.00. Quality only as chart images | artifacts exist; numbers NOT VERIFIED |
| TensorFold `docs/recipes/exl3.md` (local tree `src-tp3586`) | mul1: "x = s * 0x83DCD12D mod 2^32; h = 1024 + (the sum of x's four bytes); value = h * fp16(0x1EEE) + fp16(0xC931)"; "6-bit ... 221-226 GB/s of a 237 GB/s streaming ceiling"; "2 bits is where the decode itself is the work" | decode cost |
| MLX `mx.quantize` docs https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html | modes "affine", "mxfp4", "mxfp8", "nvfp4"; affine bits 1,2,3,4,5,6,8, groups 32/64/128, "we also save s and β which are the returned scales and biases" | affine overhead = 32/g bits |
| mlx-lm LEARNED_QUANTS.md https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/LEARNED_QUANTS.md | DWQ "fine-tunes non-quantized parameters (including quantization scales and biases)" with the unquantized model as teacher, default 1024 samples, "works best distilling to lower precision, anywhere from 2-bit to 4-bit"; dynamic quant default 5 bits sensitive / 4 rest; no quality numbers | MLX mixed-bit / DWQ: tooling exists, no numbers |
| GPTQ, arXiv 2210.17323 https://arxiv.org/abs/2210.17323 | "in approximately four GPU hours"; "3 or 4 bits per weight"; "negligible accuracy degradation" (OPT/BLOOM era) | W3/W4 class |
| AWQ, arXiv 2306.00978 https://arxiv.org/abs/2306.00978 | "Protecting only 1% salient weights can greatly reduce quantization error" | W4 class |
| AutoRound, arXiv 2309.05516 https://arxiv.org/abs/2309.05516 | "optimize rounding values and weight clipping in just 200 steps"; "near-lossless 4-bit quantization in most scenarios" | E1 tool |
| RedHatAI/Qwen3.8-27B-INT4 https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4 | W4A16 GPTQ with AWQ smoothing, 512 samples × 4096; recovery IFEval 99.67 %, MMLU-Pro 98.81 %, GSM8K-Platinum 101.07 %, MATH-500 99.52 %, GPQA-D 98.49 %, AIME25 98.69 % | public best W4 for this model |
| Qwen3.8-27B discussion #192 https://huggingface.co/Qwen/Qwen3.8-27B/discussions/192 | **lead**: wiki ppl BF16 7.993, AWQ INT4 8.037 (KLD 0.0335), AutoRound INT4 7.946 (KLD 0.0459), FP8 8.029 (0.0117), QUASAR NVFP4 QAT 8.247 (0.0682); one DGX Spark | W4 PTQ spread on this exact model |
| AQLM, arXiv 2401.06118 https://arxiv.org/abs/2401.06118 | "Pareto optimal ... when compressing to less than 3 bits per parameter" | VQ |
| PV-Tuning, arXiv 2405.14852 https://arxiv.org/abs/2405.14852 | first Pareto-optimal Llama-2 quantization "at 2 bits per parameter" (paraphrase of abstract) | VQ fine-tuning |
| VPTQ, arXiv 2409.17066 https://arxiv.org/abs/2409.17066 | "reduces model quantization perplexity by 0.01-0.34 on LLaMA-2"; "4.41-7.34 on LLaMA-3 over SOTA at 2-bit"; "1.6-1.8x increase in inference throughput compared to SOTA" | VQ |
| GPTVQ, arXiv 2402.15319 https://arxiv.org/abs/2402.15319 | "between 3 and 11 hours to process a Llamav2-70B model"; "VQ leads to improved latency compared to using a 4-bit integer format" (mobile CPU) | small-codebook VQ |
| QuIP#, arXiv 2402.04396 https://arxiv.org/abs/2402.04396 | randomized Hadamard "faster and has better theoretical properties"; E8 lattice "optimal 8-dimension unit ball packing" | rotation evidence |
| FLUTE, arXiv 2407.10960 https://arxiv.org/abs/2407.10960 | LUT kernel "2-4x faster than existing GEMM kernels" at batch < 32, g128; 1.5–2x end-to-end (LLaMA3) | LUT decode on GPU |
| HIGGS, arXiv 2411.17525 https://arxiv.org/abs/2411.17525 | "data-free ... Hadamard rotations and MSE-optimal grids"; "outperforms ... NF4" | data-free alternative |
| LQER, arXiv 2402.02446 https://arxiv.org/abs/2402.02446 | "combines quantization and low-rank approximation"; "nearly-lossless W4A8 quantization" | low-rank residual |
| SparseGPT, arXiv 2301.00774 https://arxiv.org/abs/2301.00774 | "at least 50% sparsity in one-shot"; "generalizes to semi-structured (2:4 and 4:8) patterns" | sparsity |
| DFloat11, arXiv 2504.11651 https://arxiv.org/abs/2504.11651 | "around 30% model size reduction while preserving bit-for-bit identical outputs" | entropy coding (bf16) |
| LazyWeights, arXiv 2611.17108 https://arxiv.org/abs/2611.17108 | Huffman-coded GPTQ/K-quant codes "5.7-31.4% smaller"; decoder "between DRAM and L2" as a "0.21 mm^2 ASIC" | entropy coding needs hardware |
| ParetoQ, arXiv 2502.02631 https://arxiv.org/abs/2502.02631 | "notable learning transition between 2 and 3 bits"; ternary/2/3-bit "generally exceeds 4-bit" in size-accuracy | QAT only |
| BitNet b1.58 2B4T, arXiv 2504.12285 https://arxiv.org/abs/2504.12285 | "native 1-bit ... 2-billion parameter scale", "4 trillion tokens" | ternary needs pretraining |
| CPTC, arXiv 2611.22019 https://arxiv.org/abs/2611.22019 | decoder "multiplies its state by two coefficients and adds the top bits of the products to index a 64-byte table"; "within 0.07 bits of the ideal" at 4 bits; KL "by up to 27%" below QTIP codes; Trainium 4 | newest computed code (2026-11) |
| Leech lattice 2-bit, arXiv 2609.02652 https://arxiv.org/abs/2609.02652 | "1.38x perplexity and 14.7 MMLU points at 4B" | 2-bit lattice, quality cost |
| Disaggregated quantization, arXiv 2609.26333 https://arxiv.org/abs/2609.26333 | separate prefill weights "matching or exceeding its accuracy at 2-3-bit decode"; uses Qwen3.8-27B GGUF decoders | prefill/decode split (NVFP4 prefiller: no DPAS here) |
| DFlash / vLLM (search summary only) https://vllm.ai/blog/2026-05-28-speculators-v050 | DFlash in vLLM via PR #38300, vllm ≥ 0.20 | NOT VERIFIED (page not opened) |
| Search: Qwen3.8-27B QAT | no official QAT INT4 found; third-party QUASAR NVFP4 QAT | one search, not exhaustive |

Local evidence used: routecmp `REPORT.md` §3/§6; routecmp2 `REPORT.md` §1/§7; `TP-ROOT-CAUSES.md` (job 110); README §2;
PATHFINDER §2/§3/§9; ADR-0013 decision 3; ledger RL-084, 090, 092, 093, 094, 098, 100–104, 112, 117, 119, 122, 161,
176, 179–184, 187–191; `docs/reference/b70-format-and-lever-guide.md` §1; TF `gemm.cl:84-124`, `decode.cl:208-250`,
`qwen27_main.zig:446-488`, `docs/xpu/NAVIGATION.md` §5; `~/HOST-FACTS.md` (18 Nov 23:45) for the models on disk.

## 7. Not verified, not tested

1. Every tok/s in this file except the four route baselines and the exl3xpu cells is an estimate from §5.1. No new
   GPU measurement was made.
2. `exl3_gemv` has never run on this host as far as the evidence shows. Its per-weight cost is not measured.
3. The decode-cost calibration D (2.2 / 4 / 7 ms) is inferred from RL-183 (bench kernel, M=4) and RL-119
   (B_FAKE, M=4). At M=16 in a TF-class kernel it is unmeasured. At 3 bits in any DPAS kernel on this card it is
   unmeasured (RL-100's 42.4 tok/s at 3 bpw was a whole-route number on an untuned K=3 path).
4. DFlash2 acceptance on any target other than MLX 4-bit g64 is unmeasured.
5. Quality: no 329-battery result exists for any sub-4-bit artifact of this model. EXL3 3.0 has only 6/6 canaries
   and 9/10 tasks (RL-094). turboderp's KLD charts are images and were not read. No primary 3-bit scalar
   GPTQ/AutoRound number on a ~27B model was opened.
6. Whether MLX 4-bit g64 itself passes the 14-vs-30 canary was not checked in this session.
7. Whether exl3xpu's vLLM fork supports `DFlash2DraftModel` was not checked; the vLLM DFlash page was not opened.
8. Sparse DPAS on Xe2: absence was not confirmed from Intel documentation.
9. The QTIP table values came through a text extraction; confirm them in the PDF before quoting them publicly.
10. TensorFold's g128 build for the Qwen3.8-27B MLX path is assumed from NAVIGATION ("compiled for group 64 or 128"); not built.
11. Bias rounding in the E1 repack (`bias = −scale × zp` stored in bf16) adds a small error that has not been quantified.
