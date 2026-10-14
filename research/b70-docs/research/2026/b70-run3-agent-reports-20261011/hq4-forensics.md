---
title: "Run-3 subagent report: hq4-forensics (2026-10-11)"
type: research
date: 2026-10-11
status: provisional
---

# HQ4 F0 screening harness — forensic audit (2026-10-11)

Scope: `scripts/tmp/hq4_f0_20261010.py` (RL-150), `scripts/tmp/rq26_f0_driver_20261011.py` (RL-157),
`scripts/tmp/hq4_states_20261010.py`; runs
`results/2026/10/RQ-26-20261010T231243348252Z-1e1f43f8d1/` (RL-150) and
`results/2026/10/RQ-26-20261011T122026925511Z-0d5b5ff8b7/` (RL-157; checkpoints under
`/mnt/<nvme-disk>/b70-run2-20261011/rq26/RQ-26-20261011T122026925511Z-0d5b5ff8b7/`).
Method notes: all heavy probes ran under `systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 -p CPUQuota=200%`
(lead-order compliant; no GPU, no daemon). Probes live in `/tmp/hq4forensics/` (probe1_metric.py, probe2_attl03.py,
probe4_ychan.py + probe1_output.txt, attl03_deltas.pt). After the lead's 17:40Z xe-reset order, only sub-second /
sub-0.5 GB analysis ran (attl03_deltas.pt stats, RL-150/157 JSON reads, one 1.9 s synthetic Adam demo).

---

## Q1 — RTN-INT3 sanity arm reads block-delta KL ≈ −2e-20 on att_l03 despite rel_fro 0.24-0.26 weights

**Verdict: HARNESS BUG — metric blindness (softmax saturation), NOT a weight-install or forward bug.**
The quantized weights ARE installed and DO perturb the block output by 14.5 %; the primary metric is
provably insensitive to that perturbation on this block.

### The weights reach the forward (install path is sound)

- Install path: `job_sanity` quantizes each of the 7 att linears (`rq26_f0_driver_20261011.py:454-464`:
  `wq, _s = rtn_int3_quant(w0)`; `sd_q[f"{n}.weight"] = wq.to(torch.bfloat16)`), then
  `build_layer(txt_cfg, layer_idx, sd_q)` + `run_layer` (`:466-468`). `build_layer` fails closed on wrong keys:
  `load_state_dict(sd, strict=False)` with hard asserts on unexpected keys and non-rotary missing keys
  (`hq4_states_20261010.py:211-213`). Keys used are the module's own relative names
  (`self_attn.q_proj.weight` etc., `hq4_f0_20261010.py:90`), so no key-space mismatch is possible without an
  assertion firing. No module reuse/caching: a fresh `Qwen3_5DecoderLayer` is constructed per arm
  (`hq4_states_20261010.py:210`), `sd_q = dict(sd)` is a fresh dict, and tensors are materialized copies
  (never views of `sd_bf16`), so no shared-tensor aliasing.
- Forward path is faithful: probe2 (`/tmp/hq4forensics/probe2_attl03.py`) rebuilt the layer from the **saved
  wq tensors the sealed job wrote to NVMe** (`arms/att_l03/rtn-int3/*.wq.safetensors`), ran the identical
  `build_layer`/`run_layer`, and reproduced the sealed KL **bit-for-bit in 6 significant figures**:
  repro `kl_mean = -2.034e-20` vs sealed `kl_fp32 = -2.0342040164188262e-20`
  (`results/.../sanity_control.json` → per_block.att_l03).
- The perturbation is real: on the saved repro pair (`/tmp/hq4forensics/attl03_deltas.pt`, from
  2026-10-11 19:35Z capped run; re-analyzed today in <3 s):
  `rel_err_delta = ||d_arm−d_ref||/||d_ref|| = 0.1447`, `cos_min = 0.9690`, `cos_med = 0.9933`.
  A 14.5 % delta error is present in the block output; the forward sees the weights.
- The in-run fp64 fix round also read −2.0342038e-20 (`sanity_control.json` kl_fp64) — the ~0 is not fp32
  cancellation, exactly as the driver's own rule then concluded (`"rule": "harness broken..."`).

### The code path that hides it: `top1_kl` standardize+softmax on the delta

`delta_metrics` (`rq26_f0_driver_20261011.py:235-240`) hands `d_ref`, `d_arm` to the imported
`metrics_vs`/`top1_kl` (`hq4_f0_20261010.py:355-364`):

```python
pa = torch.softmax(a / a.std(-1, keepdim=True), dim=-1)
pb = torch.softmax(b / b.std(-1, keepdim=True), dim=-1)
kl = (pa * (log pa − log pb)).sum(-1)
```

Measured on the repro pair (today's capped analysis of attl03_deltas.pt) and on the RL-157 checkpoint tensors
(probe4, `/tmp/hq4forensics/probe4_ychan.py`):

- **One channel dominates the delta**: channel 3994 is `argmax(std(d_ref))` on **100.0 % of the 832 eval
  positions** and holds **94.1 % of ||d_ref||** (att_l03 is the *writer* layer of this massive-activation
  channel; it holds 94.2 % of ||x_eval|| too).
- After standardization the top-1 z-gap over the other 5119 channels is **66.2 nats median / 44.2 min** →
  `softmax(z)` is one-hot at ch3994: `p_top1 ≈ 1 − e^{−44}`.
- The INT3 arm never flips the argmax (`top1 = 1.0000`), so both distributions are the *same* one-hot up to
  fp rounding; KL collapses to ±1e-18..1e-24 rounding noise per position, mean −2.03e-20. Negative values
  are possible because the estimator subtracts two rounded `log p`s of near-identical rows — a KL estimate
  must be ≥ 0, so negative output is itself a red flag for a degenerate comparison.
- Cross-check (probe1, `/tmp/hq4forensics/probe1_output.txt`): a *synthetic* EXL3-magnitude delta error
  (10 % of ||d||, Gaussian) injected on att_l03 moves the delta metric by only **−5.4e-22** — the metric
  cannot see perturbations of this size on this block, independent of any arm.

**Minimal fix** (metric, not the sanity arm): the sanity control must gate on a non-saturating statistic —
relative delta error `||d_arm−d_ref||/||d_ref||` (fp32, no softmax) with a floor ~0.05, plus the tail
final-logit KL; keep softmax-KL only as a secondary. See Q5.

---

## Q2 — RL-150: every arm AND EXL3 read ~0 on att_l03, att_l31, gdn_l01, gdn_l30

**Verdict: HARNESS BUG — same single cause as Q1 (softmax saturation on the massive channel), applied to the
block OUTPUT y instead of the delta. Not a different bug.**

- RL-150's worker scored `metrics_vs(y_bf16, y_q)` — softmax of standardized **output** rows
  (`hq4_f0_20261010.py:540-547`, math at `:355-364`). Measured numbers (RL-150 `block_*.json`): EXL3 and all
  8 arms read |KL| ≤ 4e-12 on att_l03/att_l31/gdn_l01/gdn_l30 (e.g. att_l03: EXL3 3.6e-26, arms −6e-27…4e-26;
  gdn_l01: 1.8e-29…−5e-31) while the same arms show `cos_min` 0.994-0.9998 → outputs ARE perturbed; the metric
  is blind, not the arms dead.
- Same channel, same mechanism (probe4, all 6 blocks, capped run today):
  - gdn_l01: ch3994 holds 98.0 % of ||y||, argmax on 100.0 % positions
  - att_l03: 96.5 % of ||y||, argmax 100.0 %
  - gdn_l30: 77.9 %, argmax 99.9 % | att_l31: 77.5 %, argmax 99.9 %
  - gdn_l62: 35.1 %, argmax 94.4 % | att_l63: 25.4 %, argmax only 32.7 %
- probe1 concentration: standardized-y top-1 z-gap is **50.5-67.3 nats** on the four dead blocks vs
  **12.4 (gdn_l62) / 2.6 (att_l63)** on the two live ones, whose KL indeed responded
  (EXL3 3.8e-4 / 2.4e-3 in RL-150). Synthetic 10 %-of-||d|| perturbation → output-KL response
  1e-13…1e-27 on the dead blocks vs 0.19 / 1.16 on the live ones (probe1_output.txt).
- RL-157's move to the **delta** (`delta_metrics`, `rq26_f0_driver_20261011.py:235-240`) removes the shared
  input carrier of ch3994 and recovered 5/6 blocks — sanity KL 4.7e-5 / 0.230 / 0.152 / 0.380 / 0.255
  (`sanity_control.json`) — but att_l03's *delta itself* is 94 % ch3994 (Q1), so it stays blind there.
  Same cause, different tensor; the two runs are two views of one defect.

**Minimal fix:** replace the softmax-standardization metric with the Q5 set; do not "repair" per block.

---

## Q3 — Homemade AutoRound loss traces rise over 100 iterations

**Verdict: REAL DEFECT in the in-file recipe (`hq4_f0_20261010.py:308-333`, re-declared at iters=100 in the
RL-157 driver `:287-322`) — it is not Intel auto-round's algorithm.**

Evidence from the sealed RL-157 traces (8 AR jobs × 6 tensors, `results/.../jobs_arm_*_ar-*.json`,
`ar_loss_trace`): **all 48 traces are monotone rising — minimum at iter 1 (the initial RTN grid), maximum at
iter 100 (the returned state)** — e.g. gdn_l01 `linear_attn.in_proj_qkv`: uniform 2.60e3 → 5.79e4 (22x),
e2m1 2.53e3 → 2.25e4 (9x); gdn_l30 `in_proj_z`: 7.82e3 → 3.67e5 (47x). Every tensor then exceeds
rel_fro 0.25 and escapes to INT8 (esc 6/6, block_bpw 8.125 in all 8 jobs). A 1.9 s synthetic micro-demo of the
verbatim update math (today, <100 MB, compliant) reproduces it: loss 4.91e5 (iter 1) → 4.80e5 (iter 10) →
7.39e5 (iter 100) with median scale drifting to 0.93-0.97 of init.

What Intel auto-round actually does (sources vendored in /tmp/hq4forensics/ from github.com/intel/auto-round;
paper arXiv 2309.05516):

| element | Intel auto-round (evidence) | homemade recipe (evidence) |
|---|---|---|
| optimized variable | per-weight **rounding offset V** added before round: `round_ste(tensor/scale + v)` — `/tmp/hq4forensics/ar_int.py:211,233`; V init 0 (`wrapper.py:220-222`) | **group clip scale s** only; no V at all (`hq4_f0_20261010.py:314-331`) |
| bounds | V bounded ±0.5 level by construction; min/max **scale multipliers clamped every step** into `minmax_scale_bound=(0.0,1.0)` (`wrapper.py:76,297-299`) | only `clamp_min(1e-6)` — unbounded above (`hq4_f0_20261010.py:331`) |
| optimizer | **SignSGD fixed** (`signround_q.py:836-838` returns SignSGD; update `param.add_(sign(d_p), alpha=-lr)`, `sign_sgd.py:388`) | Adam(0.9, 0.999), whose normalized step is ±lr **regardless of gradient magnitude** |
| lr | `1.0/iters` default (`signround_cfg.py:229-240`; 2/iters for ≤3 bits), **linear decay start_factor 1.0 → end_factor 0.0** over iters (`signround_q.py:435,708`) | constant 5e-3 (`hq4_f0_20261010.py:98,331`) |
| returned state | **best-loss checkpoint kept** (`signround_q.py:441,554-555,583`) | last state returned (`hq4_f0_20261010.py:332`) |

**Exact defect:** the recipe applies constant-lr Adam to the per-(row,group) *scale*. Adam's step is
`lr·m̂/√v̂ ≈ ±5e-3` per iteration per scale, while `s_init = max|w_group|/7.5 ≈ 1e-3…1e-2` — each step moves
the whole grid by a fraction of itself, momentum keeps the sign consistent, and with no box, no decay and no
best-state retention the scale walks monotonically away from init; the true loss is piecewise-constant in s
(the STE gradient treats the integer level `u` as constant and ignores level flips), so the walk cannot track
loss and diverges 10-50x by iter 100, pushing rel_fro past the 0.25 escape threshold. The correct variable —
the bounded per-weight rounding offset V with signed-gradient descent, decayed lr and best-loss export — was
never implemented.

**Minimal fix:** port `sign_round`'s actual parameterization: optimize V ∈ [−0.5, 0.5] via SignSGD on
`round_ste(w/s + V)`, keep scales fixed or as multipliers clamped to a [0.5, 2]·init-style box, set
lr = 1/iters with linear decay to 0, and export the best-loss iterate. (Owed after `CPU GO`: the capped A/B/C
demonstration on the real gdn_l01 `mlp.down_proj` tensor with the sealed run's saved xs —
`/tmp/hq4forensics/probe3_ar.py` already implements all three variants; ~5-8 min CPU, see §Owed.)

---

## Q4 — Were the "AutoRound-uniform beat EXL3" arms on att_l63 / gdn_l62 actually INT4?

**Verdict: HARNESS-LEVEL MISREADING (real numbers, wrong conclusion) — NO, they were not INT4. Every
AutoRound-uniform arm that beat EXL3 escaped 100 % of its tensors to INT8 at 8.125 bpw — double EXL3's bit
budget and above the declared 4.5 bpw cap. RL-151's "AutoRound-uniform beat EXL3 10-16x" DOES NOT SURVIVE.**

From RL-150 `block_att_l63.json` / `block_gdn_l62.json` (`arms.<arm>.{metrics.kl_mean, block_bpw, per_tensor}`):

| block | arm | kl_mean | block_bpw | INT8 escapes | per-tensor bpw |
|---|---|---|---|---|---|
| att_l63 | exl3 (4bpw artifact, `hq4_f0_20261010.py:410` asserts K=4) | 2.435e-3 | ~4 | 0/– | – |
| att_l63 | ar-uniform-h0 | 1.531e-4 | **8.125** | **7/7** | all 8.125 (post-escape rel_fro ≈ 0.009) |
| att_l63 | ar-uniform-h1 | 1.495e-4 | **8.125** | **7/7** | all 8.125 |
| att_l63 | gptq arms (4 arms) | 3.8-5.1e-3 | 4.125 | 0/7 | all 4.125 |
| att_l63 | ar-e2m1-h0/h1 | 2.4e-2/3.6e-2 | 5.083 | 1/7 | mixed |
| gdn_l62 | exl3 | 3.758e-4 | ~4 | 0/– | – |
| gdn_l62 | ar-uniform-h0 | 3.914e-5 | **8.125** | **6/6** | all 8.125 |
| gdn_l62 | ar-uniform-h1 | 3.514e-5 | **8.125** | **6/6** | all 8.125 |
| gdn_l62 | gptq arms (4 arms) | 1.2-1.6e-3 | 4.125 | 0/6 | all 4.125 |
| gdn_l62 | ar-e2m1-h0/h1 (the only INT4-pure AR arms) | 4.786e-3 / 3.983e-3 | 4.125 | **0/6** | all 4.125 (rel_fro 0.19-0.25) |

- The "winners" (1.53e-4/1.50e-4 vs EXL3 2.43e-3 → "0.06x"; 3.91e-5/3.51e-5 vs 3.76e-4 → "0.09-0.10x",
  i.e. the 10-16x claim recorded in `registry/research-ledger.json:1826,1834` RL-151 and echoed in RL-153
  `:1860`) are INT8-escaped arms: per-tensor `escaped_int8: true` on **every** linear, `bpw: 8.125`, post-escape
  rel_fro ≈ 0.009 (INT8 per-channel error) vs EXL3's 4-bit error. That is a bit-budget win, not a recipe win.
- The same recipe at true INT4 (ar-e2m1, 0 escapes, 4.125 bpw) is **12.7x WORSE than EXL3** on gdn_l62
  (4.786e-3 vs 3.758e-4) — consistent with the Q3 defect (diverged scales → near-worst grid).
- RL-151's note "AR ran at iters=10" understates it: at iters=10 the scale walk had already pushed every
  tensor past the 0.25 escape threshold (RL-157 driver docstring `:6-7` says the same for all AR-uniform
  tensors). The 4.5 bpw cap (`BPW_MAX`, `hq4_f0_20261010.py:97`) should have disqualified these arms from the
  "usable blocks" comparison; RL-150's own block JSONs carry the escape data that proves it.

**Minimal fix:** any arm-level comparison must filter on `block_bpw ≤ 4.5` (or report INT4-pure arms only);
the registry rows RL-151/RL-153 should be annotated that the "winning class" was INT8 at 8.125 bpw.

---

## Q5 — Is KL over softmaxed standardized hidden channels a sound quality proxy?

**Verdict: UNSOUND as a primary screen metric for this model.** Three independent failures measured:

1. **Saturation / blindness** (Q1/Q2): with one 90 %+ energy channel, softmax(std(row)) is one-hot with a
   44-66 nat gap; a 14.5 % block-delta error reads −2e-20 nats and a 10 % synthetic error reads −5e-22
   (probe1/probe2). Sensitivity is a property of the *block's activation anatomy*, not the arm — arms are
   rankable only on the two late blocks.
2. **top-1 agreement saturates at exactly 1.0** on 4/6 blocks (RL-150 block JSONs) — no discriminative power
   where the massive channel is argmax.
3. **The estimator can be negative** (−2e-20, −6e-27, …): a KL estimate must be ≥ 0; negative output means it
   is measuring fp rounding of identical one-hots.

Additionally it is scale-invariant per row (immune to the error component that changes row magnitude, not
shape) and uses only batch row 0 (832 of 1664 positions, `hq4_f0_20261010.py:358-359`).

**Recommended metric set for the re-screen (in priority order):**

1. **PRIMARY — relative error norm of the block delta, fp32, no softmax**:
   `rel = ||(y_arm − x) − (y_bf16 − x)||_F / ||y_bf16 − x||_F` over all positions, both batch rows (report
   mean per-position relative error too). Cheap, deterministic, unsaturable, and it is the quantity the
   quantizers themselves minimize (GPTQ/AWQ objective class). On att_l03 it reads 0.1447 where the KL metric
   read 0; it separates INT4 (rel ~0.10-0.25 per tensor) from INT8 (~0.009) at a glance, which would have
   caught Q4 immediately. Gate: sanity arm must exceed ~0.05 on every block (INT3 measured 0.14+).
2. **Anchor to what the format serves — tail final-logit KL + next-token argmax agreement** (RL-157's
   `chunked_logit_metrics`, `rq26_f0_driver_20261011.py:260-284`, fp32 chunks; no positions×vocab matrix held)
   on ≥ 2 blocks, ideally all 6 tails, plus one whole-model perplexity on held-out text before any adoption
   claim. Vocab logit rows do not exhibit the single-channel one-hot anatomy (they are the only place where
   "KL of softmaxed rows" is meaningful).
3. Keep `cos_min` as a cheap tertiary (it does respond: 0.969 for INT3 att_l03) but never as a gate.
4. If a hidden-state KL must be kept for continuity with RL-067: mask the top-k dominant channels (e.g. drop
   ch3994 and any channel >10 % of row energy) before standardization, add temperature so the post-mask z-gap
   is < ~5 nats, and compute a symmetrized KL in fp64 so the estimate cannot go negative. Still secondary.

---

## Owed after lead's `CPU GO` (not run: >0.5 GB / >30 s cap now in force)

1. Real-tensor A/B/C of the AutoRound fix (Q3 confirmation on sealed data; probe already written):
   `systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 -p CPUQuota=200% env OMP_NUM_THREADS=2 \
   python3 /tmp/hq4forensics/probe3_ar.py` — variants: (a) homemade verbatim, (b) scale box-clamped,
   (c) SignSGD on bounded V with decayed lr + best-state; tensor gdn_l01 `mlp.down_proj` with the sealed
   run's saved `xs` rows; ~5-8 min CPU, ~1.2 GB peak. The synthetic micro-demo and the sealed 48-trace record
   already establish the mechanism; this run only quantifies the fix's gain on real data.

## Artifact map

- `/tmp/hq4forensics/`: probe1_metric.py + probe1_output.txt, probe2_attl03.py, probe3_ar.py (owed),
  probe4_ychan.py, attl03_deltas.pt (d_ref/d_arm of the exact att_l03 sanity repro), vendored auto-round
  sources (autoround.py, signround_q.py=quantizer.py, signround_cfg.py, wrapper.py, ar_int.py=data_type/int.py,
  ar_qbase.py, ar_utils.py, adam_round.py, sign_sgd.py), tree.json.
- Verified this session, read-only: RL-150 block_*.json (6), RL-157 sanity_control.json + jobs_arm_*_ar-*.json
  (8) + jobs_sanity_*.json, registry/research-ledger.json RL-151/153 rows, b70-run2-workflow-report-20261011.md.
