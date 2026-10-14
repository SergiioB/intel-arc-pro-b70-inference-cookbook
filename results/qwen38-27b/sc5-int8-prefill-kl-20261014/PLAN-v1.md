# PLAN: as-served quality of int8 prefill on exl3xpu (plain 4.00bpw and SC5), 2026-10-14

## Question

RL-179's KL figures (plain 0.01391 / 0.00876, SC5 0.00582 / 0.00141, wikitext / chat) were measured
on reference-decoded weights, not on the served engine. The served route turns on W8A8 int8 prefill
(`EXL3_INT8_PREFILL: "1"` in model.yaml). The engine README gives its cost as NLL +0.17 % and 97.2 %
top-1 agreement with fp16 prefill, measured on plain 4.0 only. SC5's own top-1 vs BF16 is 96.6–98.1 %.
So int8 prefill could erase a material share of SC5's served quality advantage. That is also the only
engine-side approximation on exl3xpu that an "engine-aware" quant (MiaAI EAQ post, 2026-10-13) could
target, apart from the fp8 KV cache, which cannot be toggled.

## Arms

On the identical frozen panels, one card (GPU1), C1, prompt logprobs only (max_tokens 1, greedy):

| Arm | Artifact | `EXL3_INT8_PREFILL` |
|---|---|---|
| plain_int8 | turboderp 4.00bpw @113cf7ab, `<models-dir>/b70-exl3-qwen38-4bpw-20260926` | 1 (route) |
| plain_fp16 | same | 0 |
| sc5_int8 | turboderp SC_5.00bpw_H6 @dfe2f4fd, `<models-dir>/qwen38-27b-exl3-SC_5.00bpw_H6` | 1 (route) |
| sc5_fp16 | same | 0 |

The image is `ghcr.io/sergiiob/exl3xpu-mixedk:20261014` for all four arms. Route settings are kept
(fp8 KV, oneDNN attention, graphs FULL_DECODE_ONLY). Diagnostic deviations, the same in every arm:
- MTP off: prompt logprobs come from the target only.
- Prefix caching off.
- `max_num_batched_tokens` 1024: bounds full-vocab prompt-logit memory.
- gmu 0.85, `max_model_len` 32768, `max_logprobs` 128.
- `EXL3_DRAFT_VOCAB` unset.

## Panels

`<work-dir>/panels.json`, sha256 `bda3c237...53e3c`, built by
`scripts/tmp/asserved_kl_20261014/prep_panels.py`:
- **P1:** 4 × 2048-token wikitext-2 test windows. The corpus sha256 `5f1bea06...` is the same as RL-179's.
- **P2:** 4 qbench held-out chat rows (trace sha256 `6070481d...`, the same as RL-179's), response positions only, top-128.
- **P3:** one 16,384-token wikitext window, top-32.

Each arm passes every panel once (rep1); P1 then gets a second pass (rep2) as the determinism floor.

## Metrics

Computed by `analyze.py`, written before the run:
- **Coarse KL lower bound:** P's top-k tokens plus one tail bucket. The tail mass is reported.
- **Top-1 agreement.**
- **Exact NLL** on the observed tokens.
- **95 % block-bootstrap CIs.**
- The reference distribution in each pair is the fp16-prefill arm.

## Decision rule (predeclared)

Let **R** be SC5's int8 NLL increase divided by the as-served fp16 NLL gap between plain and SC5. Let
**K** be SC5's int8 KL lower bound divided by RL-179's SC5 KL on the same panel class (P1, P2).
- **Material:** R ≥ 0.25 or K ≥ 0.25 on any panel. Recommend `EXL3_INT8_PREFILL=0` for the SC5 quality
  tier, then price its TTFT cost in a separate run. Reopen an engine-side quality question.
- **Negligible:** R < 0.10 and K < 0.10 on every panel. Close "engine-aware quantization" for exl3xpu:
  the served engine adds nothing a quant could compensate for, apart from the fp8 KV cache (unmeasured).
- **Otherwise minor:** record it, no route change.

The rep1-vs-rep2 KL must be ≤ 1e-6, or the run is invalid (non-deterministic).

## Impact gate (AGENTS.md section 21)

Expected wall time is 45–75 min, mostly 4 server loads with about 180 s of kernel compile each.
- **Best case:** a config-only quality gain for the SC5 tier, or a closed line with evidence.
- **Comparisons:** the adopted route itself, and turboderp SC5, the best public EXL3 artifact, which is
  what is being measured.
- **No speed claim** is made or possible.

## Safety

- GPU1 only, under `flock /tmp/b70-gpu1.lock`. The empty-GPU gate runs before each load.
- Container cap `--memory=14g`, to keep at least 20 GiB MemAvailable (34 GiB at plan time; another
  session's CPU KL job holds about 11 GB).
- Containers are labelled `b70.owner=AsServedKL`. Only owned containers are removed.
- Power caps are recorded as found and not changed.
- Timeout 25 min per arm. Stop on any xe reset in dmesg.
