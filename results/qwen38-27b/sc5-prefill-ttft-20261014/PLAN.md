# PLAN: price fp16 prefill for the SC5 quality tier (2026-10-14)

## Why

RL-213 found int8 prefill materially changes SC5's served distribution: chat KL 0.00099, K = 0.70.
The determinism probe (`results/2026/10/int8-determinism-20261014/`) found every prefill op bitwise
repeatable in isolation:
- `E.linear` int8 and fp16;
- `torch.matmul`;
- oneDNN SDPA and FA2;
- the FLA GDN chunk kernel.

A one-ulp input change moves int8 outputs about 2x as much as fp16 outputs. The serving drift is
therefore small upstream input differences between requests, amplified by int8 rounding.

That means making int8 deterministic would **not** fix the quality loss. A deterministic int8 output is
one realization of the same rounding error, so its distance to fp16 stays around the measured
0.001 KL. The remaining fix is fp16 prefill (`EXL3_INT8_PREFILL=0`) for the SC5 tier. This run prices it.

## Arms

| Arm | Image | Model | Config | Prefill |
|---|---|---|---|---|
| A | `ghcr.io/sergiiob/exl3xpu-mixedk:20261014@sha256:933eda54...` | SC5 `<models-dir>/qwen38-27b-exl3-SC_5.00bpw_H6` | launcher qwen38-27b-sc5 profile | `EXL3_INT8_PREFILL=1` (as served) |
| B | same | same | same | `EXL3_INT8_PREFILL=0` |

The launcher profile is: serve config `models/qwen3.8-27b-exl3-4.00bpw`, `max_model_len` 196608,
gmu 0.94, MTP3, fp8 KV, `UR_L0_USE_COPY_ENGINE=0`.

Diagnostic deviations, the same in both arms:
- prefix caching off, so every request is cold;
- `EXL3_DRAFT_VOCAB` as the launcher sets it, or unset if the launcher unsets it (the runner must copy
  the launcher's exact env/--set list).

GPU1, power cap as found (230 W expected, recorded, not changed).

## Cells

Cold prompts: 32 fresh random token ids followed by wikitext ids, exact N tokens. Each request is
streamed with `max_tokens` 1, and TTFT is the time to the first text chunk.
- **Order:** ABAB server loads, four loads in total.
- **Per load:** one discarded warmup per shape, then 8192 ×3, 32768 ×3, 131072 ×2.
- **On the second load of each arm only:** one capacity/TTFT request at N = 190,000, which must complete.
- **Metric:** cold-input rate = prompt_tokens / TTFT.
- **Thermal:** sustained regime at the as-found cap. Each load logs package temperature and energy
  before and after its cells.

## Decision (predeclared)

R(N) is the median TTFT of B divided by A, from interleaved arms, per N.
- **Recommend fp16 prefill for SC5** (launcher recipe env change, operator release decision) if all
  three hold:
  - R(32768) ≤ 1.40;
  - the 190K request completes in arm B with free VRAM ≥ 500 MiB under load;
  - greedy canaries pass 6/6 in arm B.
- **If R(32768) > 1.40:** offer it as an opt-in "max quality" toggle instead of the default.
- **If B fails the 190K request:** fp16 prefill is allowed only at the 131K safe profile; record that.

The plain 4.00bpw speed route is not changed in either case.

## Safety

Same guards as RL-212/213:
- GPU1 lock, empty-GPU gate, MemAvailable ≥ 34 GiB before load, `--memory=14g`;
- label `b70.owner=AsServedKL`, only owned containers removed;
- dmesg reset abort, 60-min hang guard per load;
- nothing left running.
