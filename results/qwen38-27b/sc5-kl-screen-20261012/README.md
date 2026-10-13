# Qwen3.8-27B EXL3 KL screen — 2026-10-12

Weight-level quality screen of three EXL3 quants against the native BF16 model.
One run on one Arc Pro B70. No serving engine and no speed figures.

## Arms

| Arm | Artifact |
|---|---|
| `bf16` (reference) | Qwen3.8-27B BF16 weights, native BF16 head |
| `exl3` (plain 4.0bpw) | `turboderp/Qwen3.8-27B-exl3` 4.00bpw, the route-of-record artifact |
| `exl3_sc4` | branch `SC_4.00bpw_H5` @ `4acd9ad5af224ec9e8815a54d71a033f378a7e9d` |
| `exl3_sc5` | branch `SC_5.00bpw_H6` @ `dfe2f4fd71bf9a1994051d4d47e6460fa96dd723` |

The `*_bf16head` arms are a diagnostic that runs the same quantized body with
the shared BF16 head. They move each KL figure by about 6 % at most, so the
differences come from the body bit allocation.

## Method

- Harness: [`fmt_screen_20261011.py`](fmt_screen_20261011.py) (sha256
  `f4d915df8c22a8ee22ba5f70f6ab03a1f566992fee0a33ec51f90a6cef051aaa`), run in
  `vllm/vllm-openai-xpu:v0.30.0@sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d`
  as a plain PyTorch environment (torch `2.13.0+xpu`, transformers `5.16.1`).
  FP32 math, deterministic mode, all 64 layers.
- EXL3 weights are decoded bit-exactly with the exl3xpu `ref.py` reference
  (sha256 `3f614b7c…`, source commit `15ded2f3`), then multiplied in FP32.
  `raw/exl3_decode_checks.json` proves the decode for every K used.
- Each quant runs with its own quantized head (decoded to BF16, FP32 matmul).
- Metric: mean final-logit KL(P_bf16 ‖ P_quant) over scored positions, plus
  top-1 agreement with the BF16 argmax. Scoring code: harness lines 1655–1720.
- Eval A: frozen wikitext-2-raw-v1 test slice, 2 × 832 tokens, 1,664 scored
  positions.
- Eval B: turboderp's held-out chat trace `qbench_prompts_gen.json` rows 0–1
  (sha256 `6070481d…`), response positions only, 1,531 scored positions. It
  was not used to calibrate these numbers. Stored input ids already include
  the chat template.
- Identity and data coordinates: `raw/identity_start.json`,
  `raw/measurement_validity.json`. Full log: `raw/run_stdout.log`.

## Results (native heads)

| Arm | KL A (wikitext) | KL B (held-out chat) | Top-1 A | Top-1 B |
|---|---:|---:|---:|---:|
| plain 4.0bpw | 0.01391 | 0.00876 | 94.65 % | 96.41 % |
| SC_4.00bpw_H5 | 0.01602 | 0.00653 | 94.29 % | 96.80 % |
| SC_5.00bpw_H6 | **0.00582** | **0.00141** | **96.57 %** | **98.11 %** |

SC5 KL relative to plain 4.0bpw: 0.418× on eval A and 0.161× on eval B. SC4
is worse than plain on wikitext (1.152×) and better on chat (0.746×).
Source: `raw/arms_all.json` (`kl_mean`, `top1_vs_bf16`,
`heldout_response_slice`), ratios in `raw/gate.json`.

## Limits

- One run, one card, two small eval sets. A screen, not a task benchmark.
- KL is measured on the weights through a reference decode. It is not
  measured through the served exl3xpu engine, fp8 KV cache or MTP.
- `raw/gate.json` shows `evaluated: false`: the harness has no adoption gate
  for this comparison. The numbers are reported, not gated.
