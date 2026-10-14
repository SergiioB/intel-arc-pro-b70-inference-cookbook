---
title: "As-served quality of int8 prefill on exl3xpu; EAQ / TensorFold assessment; next EXL3 levers"
type: research
date: 2026-10-14
status: completed
---

# As-served quality of int8 prefill on exl3xpu, EAQ / TensorFold, next levers

Ledger: RL-211 (EAQ / TensorFold), RL-212 (v1, invalid), RL-213 (v2, material).
Evidence: `results/2026/10/asserved-kl-int8-prefill-20261014/` (PLAN.md, PLAN-v2.md, v2/, sources/).
Scripts: `scripts/tmp/asserved_kl_20261014/`. Raw logprobs and panels: `/data/nvme/archive/asserved-kl-20261014/`
(manifest and sha256 in the result root's `ARCHIVE_POINTER.txt`).

## 1. EAQ (MiaAI) and TensorFold

### What the post and its sources claim

- **The X post** (2026-10-13): "EAQ vs stock EXL3, same size, 37 % lower KL, measured on TensorFold."
- **The linked page** says something different: 41 % / 54 % lower "error" (undefined) against Mia's own
  public GLM-5.3-Flash quant, not against stock EXL3.
- **Availability:** in validation, recipe unpublished, no files. The target is a MoE model (GLM-5.3-Flash).
- **The same author's published predecessor** (same format, new calibration) gained 2–18 % KL as served,
  measured against the FP8 release.
- **Most of that served error is TensorFold's own:** TensorFold re-quantizes the BF16 dense layers to q4
  at load, and that q4 floor is 0.0676 of 0.0903 KL on wiki. "Engine-aware" means calibrating the
  experts around that engine-side error.

### Verdict

- **Not adoptable.** There is no artifact or recipe, and the numbers disagree between the post and the page.
- **The evaluation may hide a known trap.** Its test set is workload-matched and held out from the same
  distribution as the calibration data. Turboderp's SC4 is the same idea, a same-size workload-calibrated
  EXL3. On our harness SC4 was 0.75x plain on chat but 1.15x worse on prose (RL-179). An in-domain-only
  evaluation would not have caught that.
- **Even taken at face value it does not beat SC5.** A 37 % gain lands about 0.63x plain KL. SC5 is
  0.42x / 0.16x.
- **TensorFold port: rejected.**
  - TensorFold 1.0.9 is a Zig engine with Metal and CUDA kernels, and has no SYCL or Level Zero backend.
  - Qwen3.8-27B is qualified on Metal only. Its own README says the 27B decodes slower than mlx_lm
    without the drafter and reads prompts at about half mlx_lm's speed.
  - SC5 without speculation already runs at 99.5 % of its byte floor (RL-202).

## 2. The engine-side error on exl3xpu: int8 prefill

exl3xpu serves EXL3 tensors as stored. Its engine-side approximations are:
- W8A8 int8 prefill (`EXL3_INT8_PREFILL=1` in model.yaml, on by default);
- the fp8 KV cache (the only dtype, cannot be toggled);
- oneDNN SDPA.

RL-179's KL was measured on reference-decoded weights, so the served error was unmeasured.

### v1 (RL-212): four arms, wikitext, chat and a 16K panel

v1 ran plain and SC5, each with int8 and with fp16 prefill. It is **invalid under its own predeclared
determinism rule**: repeated passes had to agree to a KL of at most 1e-6, and they measured 5e-4 to
5e-3. The bound was not lowered.

Observations that held up:
- **Repeat drift:** repeated passes on one server are bit-identical up to a token index that is a multiple
  of 16 (160 to 800), then drift.
- **fp16 drift:**
  - inside the first 1024-token chunk: about 2e-5 KL;
  - in chunks with a cached prefix: about 1e-3 to 2e-3, concentrated on a few hard tokens (NLL 7–11 nats
    in every arm).
- **int8 drift:** 30–50x fp16's drift in the first chunk.

### v2 (RL-213): SC5 only, 18 held-out chat rows, two passes per load

- **Panel:** 44,473 scored positions; tail mass 0.0007, so the coarse KL is accurate here.

| Pair | KL lower bound | top-1 | ΔNLL (nats) |
|---|---:|---:|---:|
| fp16 → int8 (pass 1 / pass 2) | 0.00099 [0.00092, 0.00107] / 0.00099 | 0.9893 | +0.00065 / +0.00115 |
| fp16 pass 1 → pass 2 (noise floor) | 0.00004 | 0.9985 | 0.0000 |
| int8 pass 1 → pass 2 | 0.00081 | 0.9907 | +0.00049 |
| fp16, v1 load → v2 load | 0.00003 | 0.9987 | 0.0000 |

- **Decision (PLAN-v2 rule):** valid (noise 4e-5 ≤ 0.1·E), K = 0.70 with lower CI 0.65 → **material**.

### What it means

- On chat, int8 prefill moves SC5's served distribution by about 0.001 KL. That is 70 % of SC5's whole
  weight-quantization KL to BF16 (0.00141).
- It is still about 1/10 of the as-served distance between plain and SC5 (0.0101). The NLL cost is small
  (+0.0007 to +0.0011 nats).
- **Most of int8's divergence is its own run-to-run non-determinism.** int8 pass-to-pass drift is 0.00081;
  fp16 is reproducible even across server loads (3e-5). That is a different defect from a systematic
  quality loss, and possibly a fixable one.

### Consequence

SC5's served quality advantage is partly spent by the default int8 prefill. Two options are open:
- **Option 1:** `EXL3_INT8_PREFILL=0` for the quality tier. The engine README prices the cost at about
  23 % slower omp-replay TTFT (p50 3.78 → 4.65 s). That price is not re-measured here, and it matters more
  at the 196K SC5 profile.
- **Option 2:** make int8 prefill deterministic first (lever B below).

No launcher change is made by this note.

## 3. Ideas for mixed-K EXL3 and Qwen3.8-27B, ranked

Engine facts these rely on:
- hybrid model, 48 GDN + 16 gated full-attention layers, hidden 5120;
- per-tensor K from 2 to 7, block-128 sign+Hadamard around every linear;
- `lm_head` at K=6: 0.89 GiB, 7 % of plain's decode bytes;
- MTP3 verify round 46.7 ms, of which the eager draft is 7.8 ms (RL-185);
- fp8 KV: 16 layers × 2 × 1024 B = 32 KiB per token.

| # | Lever | Mechanism | Ceiling or prediction | Cost | Status |
|---|---|---|---|---|---|
| A | int8 prefill off for the SC5 tier | config | removes ~0.001 KL on chat; measured cold TTFT 1.63x / 1.54x / 1.34x at 8K / 32K / 131K (RL-215) | done | opt-in launcher mode (b5e7332, unreleased) |
| B | Deterministic int8 prefill | find the order-dependent reduction in the s8 path; bit-compare repeated calls on fixed input | **withdrawn (RL-214):** every prefill op is bitwise deterministic in isolation. The drift is upstream input differences amplified ~2x by int8 rounding, and a deterministic int8 path would keep the same ~0.001 KL | — | closed |
| C | Attention-K/V graft (SC5 → plain) | copy SC5's K=6/7 `k_proj` / `v_proj` (16 layers, ~50 MB more) into plain 4.0; each EXL3 tensor carries its own trellis + signs + scale; the mixed-K image already serves per-member K | if those tiny tensors carry a large share of SC5's gain: a quality upgrade at ~0.99x plain speed, the niche SC4 missed. Hypothesis; cumulative hidden-state MSE in RL-179 `layers.jsonl` is not monotone and cannot attribute it | CPU artifact assembly + one RL-179-harness run (~30 min GPU) + canaries | new |
| D | In-graph MTP draft | run the 3 draft steps inside the captured decode graph instead of eagerly (config capture failed: RL-187 / RL-188) | eager 7.8 → ~2 ms of 46.7 ms ≈ +12–14 % decode on plain and SC5; the only speed lever above the 10 % SC5 re-entry gate | kernel / runner engineering | new, gated by §17 |
| E | Suffix / n-gram speculation for agent code edits | copy-heavy tool traffic complements MTP | RQ-21; local ngram alone −31 % on non-copy text | needs the RQ-10 frozen corpus (CPU-only) | gated |
| F | Rotated 4-bit KV for the 16 attention layers | Hadamard-domain int4 KV, QuaRot-style; the plain q4_0-K cliff was unrotated | at 196K: 6.4 GB of KV read per token vs 13–16 GB of weights, so up to ~1.3x long-context decode; ~0 at 8K | attention kernel + quality battery | new, long-context only |
| G | Measure fp8-KV error | reference harness with simulated fp8 KV | the last unmeasured engine-side error | ~30 min | new |
| — | K/V co-merge, narrow-N split-K, shared `had_in`, K=4 draft head | already priced (RL-202) | +7–9 % combined, SC5 ≤ 0.93x plain | — | closed |
| — | Lower-bit `lm_head` or a sparse-vocab verify | changes outputs | ≤ 1.2 % bytes | — | rejected (not exact) |
| — | TensorFold port, EAQ adoption | — | — | — | rejected / watch |

### Recommended order (updated after RL-214, RL-215)

1. A is done: fp16 prefill costs 1.63x / 1.54x / 1.34x / 1.28x cold TTFT at 8K / 32K / 131K / 190K. By the
   predeclared rule it is an opt-in launcher mode (`--prefill fp16`, b70-launcher b5e7332, unreleased),
   not the SC5 default.
2. C. Cheap, and the only quality lever at plain speed.
3. G.
4. Optional: locate the upstream request-to-request input difference (per-layer hidden-state dump of two
   identical requests). It would make serving reproducible, but it is not a quality fix.

D is the speed line, and needs an attribution-backed plan first. E waits on RQ-10. F only matters if the
196K profile becomes the primary use.
