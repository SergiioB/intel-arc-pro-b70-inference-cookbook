# SC5 served quality: int8 vs fp16 prefill (2026-10-14)

Record: `qwen38-27-exl3xpu-sc5-int8-prefill-kl-20261014` (capability, not a ranking row).

## Question

The SC5 KL screen ([`sc5-kl-screen-20261012`](../sc5-kl-screen-20261012/README.md)) measured weights
decoded with the reference code, not the served engine. The exl3xpu serve config turns on W8A8 int8
prefill by default (`EXL3_INT8_PREFILL=1`). This folder measures how much that changes SC5's
next-token distribution as served, against fp16 prefill on the same server image.

## Setup

- **Card:** one Intel Arc Pro B70, `ghcr.io/sergiiob/exl3xpu-mixedk:20261014@sha256:933eda54…`.
- **Model:** `turboderp/Qwen3.8-27B-exl3` @ `SC_5.00bpw_H6` rev `dfe2f4fd`.
- **Serving:** fp8 KV, oneDNN attention, XPU graphs. MTP and prefix caching are off, because only
  prompt logprobs are read. Prefill chunk 1024. A cache-hit delta of 0 was verified on every pass.
- **Outputs:** `/v1/completions` with `prompt_logprobs` top-128. The observed token's logprob is always
  returned.
- **Panel (v2):** all 18 rows of turboderp's held-out `qbench_prompts_gen.json` trace with at least
  1,024 response tokens, scored on response positions: 44,473 per pass. Built by
  `scripts/prep_panels_v2.py`.
- **Passes:** two full passes per server load, so the noise floor is measured on the same tokens.
- **Estimator:** a coarse KL lower bound, using the reference's top-128 tokens plus one tail bucket.
  Grouping can only lower KL. The tail mass here is 0.0007, so the bound is tight. NLL is exact.
  CIs are 2,000 bootstraps of 256-position blocks. Code: `scripts/analyze.py`, `scripts/analyze_v2.py`.

## Result (v2, decision panel)

| Pair | KL lower bound | Top-1 agreement | ΔNLL (nats) |
|---|---:|---:|---:|
| fp16 → int8 prefill, pass 1 | 0.00099 [0.00092, 0.00107] | 98.93 % | +0.00065 [+0.00029, +0.00102] |
| fp16 → int8 prefill, pass 2 | 0.00099 [0.00092, 0.00106] | 98.93 % | +0.00115 [+0.00080, +0.00149] |
| fp16 pass 1 → pass 2 (noise floor) | 0.00004 | 99.85 % | 0.00000 |
| int8 pass 1 → pass 2 | 0.00081 | 99.07 % | +0.00049 |
| fp16, v1 load → v2 load (cross-load) | 0.00003 | 99.87 % | 0.00000 |

Predeclared rule (`PLAN-v2.md`):
- **Validity:** the fp16 noise floor (4.2e-5) must be at most 0.1× the effect. It is, so the result is valid.
- **K:** the effect divided by SC5's weight-quantization KL on held-out chat (0.00141). K = 0.70, lower
  CI 0.65. That is at least 0.25, so the effect is **material**.

### Reading it

On chat, int8 prefill moves SC5's served distribution by about 70 % as much as SC5's whole
quantization error against BF16.
- It is about one tenth of the as-served distance between plain 4.00bpw and SC5 (0.0101 KL in the
  v1 diagnostic).
- The NLL cost is small: +0.0007 to +0.0011 nats.
- Generation (decode) is unchanged. Decode steps use the small-batch fp16 path either way. The int8 path only runs
  for prompt chunks above 128 tokens.

## Non-determinism

Two identical requests to the same server are bit-identical up to a token index that is a multiple
of 16 (160–800 in v1), then drift. int8 drift is 30–50× fp16 drift in the first chunk.

The isolated probe (`determinism/`) found every prefill op bitwise repeatable over 30 runs:
- the EXL3 int8 and fp16 linears on real SC5 tensors, K=4..7;
- `torch.matmul`;
- oneDNN Graph SDPA and FA2, with and without a prefix;
- the FLA gated-delta-rule chunk kernel.

The FLA, causal-conv and attention paths contain no atomics. A one-ulp input change moves int8 outputs
about 2× as much as fp16 outputs.

So the drift comes from small input differences between requests, upstream of these ops, amplified by
int8 rounding. A deterministic int8 path would not remove the ~0.001 KL: it would freeze one
realization of the same rounding error.

## v1 (kept, invalid by its own rule)

`PLAN-v1.md` covered four arms (plain and SC5, each with int8 and fp16) on wikitext, chat and a
16K row. It predeclared that repeated passes must agree to a KL of at most 1e-6. They measured 5e-4 to
5e-3, so v1 is invalid. Its numbers (`v1-analysis.*`) are diagnostic only, and the bound was not
lowered after the fact.

## Consequence

b70-launcher 0.6.10 adds an opt-in `--prefill fp16` mode for EXL3 recipes. Its cold-prompt cost is
measured in [`sc5-prefill-ttft-20261014`](../sc5-prefill-ttft-20261014/README.md). The default stays
int8.

## Not covered

- One card, one artifact (SC5), one chat panel.
- fp8 KV error is not measured; the image has no other KV dtype.
- Task-level quality is not measured.
- The raw prompt-logprob dumps (~250 MB) are not published; their sha256 values are in
  `raw-sha256.txt`.
