# SC5 cold-prompt cost of fp16 prefill (2026-10-14)

Record: `qwen38-27-exl3xpu-sc5-prefill-ttft-20261014` (capability, not a ranking row).

## Question

fp16 prefill removes the int8-prefill shift in SC5's served output
([`sc5-int8-prefill-kl-20261014`](../sc5-int8-prefill-kl-20261014/README.md)). This folder measures
how much slower it reads cold prompts on the launcher's SC5 profile, and whether the 190K profile
still completes.

## Setup

- **Card and image:** one Intel Arc Pro B70 at a 230 W cap (as found; ~227–230 W measured average per
  load window; package temperature up to 82 °C, logged for the last three loads). Image `ghcr.io/sergiiob/exl3xpu-mixedk:20261014@sha256:933eda54…`.
- **Model:** `turboderp/Qwen3.8-27B-exl3` @ `SC_5.00bpw_H6` rev `dfe2f4fd`.
- **Launch:** the launcher's `qwen38-27b-sc5` launch, reproduced exactly (`launcher_equivalence.md`):
  - context 196,608, gmu 0.94;
  - native MTP3, fp8 KV, prefill chunk 4096;
  - XPU graphs, copy engine off.
- **Deviations from the launcher:**
  - prefix caching off, so every request is cold;
  - `EXL3_INT8_PREFILL` set explicitly: 1 for arm A (the default) and 0 for arm B.
- **Order:** ABAB over four server loads.
- **Cells per load:** one discarded warmup per shape, then 8,192 ×3, 32,768 ×3 and 131,072 ×2.
- **190K:** one 190,000-token request on each arm's second load, without a warmup.
- **Prompts:** 32 fresh random token ids, then wikitext-2 token ids, to an exact length. Every request
  asserts `usage.prompt_tokens`.
- **Metric:** streamed with `max_tokens` 1. TTFT is the client time to the first text chunk. Cold-input
  rate = prompt tokens / TTFT. Client: `scripts/ttft_client.py`; summary: `scripts/summarize_ttft.py`.

## Result

| Prompt tokens | int8 TTFT (median, s) | fp16 TTFT (median, s) | fp16 / int8 | int8 cold-input tok/s | fp16 cold-input tok/s | Samples per arm |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 2.919 | 4.766 | 1.63 | 2,807 | 1,719 | 6 |
| 32,768 | 13.871 | 21.352 | 1.54 | 2,362 | 1,535 | 6 |
| 131,072 | 88.55 | 118.76 | 1.34 | 1,480 | 1,104 | 4 |
| 190,000 | 158.2 | 202.1 | 1.28 | 1,201 | 940 | 1 (single observation) |

- **Spread:** about 0.1 s within each cell (raw rows in `ttft.jsonl`).
- **190K capacity:** the 190K request completes on both arms. Minimum free VRAM during it was 884 MiB
  with int8 and 1,053 MiB with fp16. The int8 path keeps extra int8 scratch.
- **Correctness:** fp16 passes the 6/6 greedy exact-answer canaries and 3/3 long prompts
  (`canary_B.log`).

Predeclared rule (`PLAN.md`): fp16 becomes the SC5 default only if fp16/int8 ≤ 1.40 at 32,768. It is
1.54, so fp16 prefill ships as an **opt-in** launcher mode, and the default stays int8.

## Launcher 0.6.10

`b70 launch qwen38-27b-sc5 --prefill fp16` adds `--set env.EXL3_INT8_PREFILL="0"` to the serve
command. Without `--prefill`, the command is unchanged. Real launches through an installed 0.6.10
showed:
- `EXL3_INT8_PREFILL=0` in the container's process environment with fp16, and `=1` by default;
- a correct smoke answer from both (`launcher-0.6.10/`).

## Not covered

- C1 cold prompts only. Warm, prefix-cached turns are not timed.
- Decode speed is not re-measured. Decode does not use the int8 path.
- One card, 230 W.
