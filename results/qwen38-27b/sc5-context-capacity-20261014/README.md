# SC5 context-capacity ladder — 2026-10-14

Purpose: the highest **valid** context for `Qwen3.8-27B-exl3` `SC_5.00bpw_H6`
(exl3xpu-mixedk route, fp8 KV, native MTP3) on one Arc Pro B70 — staged real
prompt+output ladder with retrieval + multi-turn + exact-answer checks.
**Capacity only: this folder contains no performance result.**

Record: `qwen38-27-exl3xpu-sc5-ctx-capacity-20261014`.

## Headline

- **Safe validated: context 131,072** — stage 130,872 occupied (cold prompts
  130,829 / 130,798 tokens + real outputs + warm `CODE=/WORD=` follow-up), exact
  needle retrieval at depths 0.5 and 0.9, exact multi-turn answers; **2,327 MiB
  free** under sustained load at `gpu_memory_utilization` 0.90 (≥2 GiB class).
- **Observed (capacity observation only, NOT a recommendation):** 196,438
  occupied at `gmu 0.94` — 1,069 MiB free (<2 GiB); all five rungs 16K / 32K /
  64K / 131K / 196K passed retrieval + multi-turn + exact checks.
- **Engine budget (adjudicated by the engine, not a validation):** max seq len
  **168,000** at gmu 0.90 (6.23 GiB KV), **206,400** at gmu 0.94 (7.5 GiB),
  ~235K at 0.965 (derived). The 168,000 top cell was **not completed** (two
  client-side harness timeouts, retained). 245,760+ would require gmu ≈ 0.982
  (~600 MiB free) — below any operating floor; **240K / 262K / 270K are
  unreachable on SC5** (270K also exceeds the 262,144 model limit). The
  FP8 KV pool is one shared token budget (`max_num_seqs` 16): pool capacity is
  not context. Plain 4.00bpw (14.9 GiB vs SC5's 20.09 GiB) is a separate route
  that reaches native 262,144 — do not merge the two.

## What this is not

- **Not performance:** single-sample capacity cells only; the recorded TTFT and
  rate fields are diagnostics. Sampling was frozen greedy (temperature 0,
  top_p 1.0, thinking off, max_tokens 256, natural stop) — not the recipe
  sampling used by the speed rows.
- **Cache counters missing (explicit):** this image exposes no
  `vllm:prefix_cache_*` counters and no `usage.prompt_tokens_details`; cold
  cells are zero-hit **by construction only** (unique entropy + distinct word
  sequence per request). Warm-cell cache behavior is evidenced by warm-vs-cold
  TTFT ratio at equal conversation length only.

## Failures retained

- Two budget aborts at engine init: `max_model_len` 262,144 > 206,400 (gmu 0.94)
  and 175,000 > 168,000 (gmu 0.90) — these produced the budget verdicts above.
- One SC5 weight-upload **host fault** on a Server B load: `Engine memory CAT
  error [18]: class=bcs` → engine reset → `Timedout job` → Xe device coredump at
  safetensors shard 2/3 (known class; 1 fault in 6 loads that day, copy-engine
  env already set). See `serverB_*` files.
- Two top-rung client timeouts (204,800 and 167,424 not verified; the server
  logged 200 OKs for 167,424 but the responses were lost — no claim).

## Host state

One Intel Arc Pro B70 (GPU1, `0000:0f:00.0`), 230 W cap as found and never
changed (card0 180 W as found); sustained thermal regime during the ladder
(pkg 82 °C, no cooldown between stages — capacity claim only); guard clean
before and after; nothing left running; VRAM returned to the empty-card
baseline.

## Files

| file | what |
|---|---|
| `PLAN.md` | pre-committed protocol + 3 amendments (budget verdicts, cache-telemetry gap, fault + retries) |
| `commands.log` | exact commands for both servers, the ladder, and cleanup |
| `REPORT.md` | full report: route identity, budget table, per-stage results, cleanup |
| `ladder_A.jsonl` / `ladder_A-summary.json` | gmu 0.94 ladder raw rows + independent adjudication |
| `ladder_B.jsonl` / `ladder_B-summary.json` | gmu 0.90 recommendation cells (131,072 pass; 167,424 unverified) |
| `serverB_fault_dmesg.txt`, `serverB_devcoredump_head.txt`, `serverB_load_hang_logtail.txt` | host-fault evidence |
| `asfound_power_vram.txt`, `asfound_power_after.txt` | as-found caps/VRAM before and after (unchanged) |

Method provenance (see `REPORT.md`): needle plant + question verbatim from
upstream exl3xpu `tests/test_needle.py`, noise-doc sizing from `bench/sweep.py`,
warm multi-turn pattern from `exl3xpu-mixedk` `tests/multiturn.py`, request and
counter method from `scripts/bench/ladder.py`, provenance read-back discipline
from `scripts/bench/prov.py`. Files copied from the originating session root of
2026-10-14.
