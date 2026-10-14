---
title: "Protocol — Exact Live Qwen27 MTP Vision Baseline"
type: doc
date: 2026-07-18
status: final
---

# Protocol — Exact Live Qwen27 MTP Vision Baseline

**Date:** 2026-07-18  
**Target:** the already-running `qwen27-mtp-q5-256k-vision-180w` production profile  
**Raw evidence root:** `/mnt/<other-disk>/inference-models/baselines/`  
**Runner:** `~/inference/scripts/live_exact_baseline.py`

## Objective

Measure the exact production service through its authenticated OpenAI-compatible streaming API without restarting or reconfiguring it. Establish current TTFT, prompt processing, engine decode, streaming decode, wall throughput, MTP acceptance, energy, 4K prefill, bounded 32K context behavior, and vision correctness.

This protocol measures the live endpoint and its current template/sampling/runtime overhead. It is not a `llama-bench` kernel microbenchmark and must not be compared as though it were one.

## Fixed configuration gate

The runner aborts unless all of the following remain true before and after every request:

- `llama-profile.service` is active/running with its original PID.
- Exactly one `llama-server` process exists.
- `active-profile` is `qwen27-mtp-q5-256k-vision-180w`.
- The API reports alias `active`, context 131,072, and health `ok`.
- The only slot is idle before the next request.
- The power cap is exactly 180 W.
- At least 2 GiB visible VRAM, 20 GiB available RAM, 20 GiB on `/mnt/<models-disk>`, 10 GiB on `/`, and 10 GiB in the experimental workspace remain.
- GPU temperature is below 82 °C.
- No new xe error/fault/reset or OOM-kill line appears in the kernel log.

The runner has no server stop/start/restart, process-kill, power-write, mount, download, conversion, or model-file mutation path.

## Workload

Requests are sequential and use streaming with usage/timing metadata, `cache_prompt=false`, and fixed seeds.

| Stage | Workload | Output cap | Purpose |
|---|---|---:|---|
| Warmup | Short GPU-memory-bandwidth explanation | 64 | Discarded; removes residual first-request effects |
| Text 1 | Legal explanation | 384 | Diverse steady-state decode |
| Text 2 | Paged-KV systems explanation | 384 | Diverse steady-state decode |
| Text 3 | Fault-tolerant service design | 384 | Diverse steady-state decode |
| Text 4 | Speculation versus batching comparison | 384 | Diverse steady-state decode |
| Prefill gate | Deterministic evidence ledger, target 4,096 raw tokens | 32 | API prompt-processing rate and TTFT |
| Context gate | Deterministic evidence ledger, target 32,768 raw tokens | 32 | Bounded long-context behavior and recall smoke |
| Vision | Existing `b70-launcher.png` fixture | 256 | Logo description, transcription, colors, central symbol |

The raw tokenizer target excludes chat-template overhead. The authoritative prompt size is `timings.prompt_n` returned by llama-server.

## Metrics and interpretation

- **TTFT:** local monotonic time from request start to the first non-empty `content` or `reasoning_content` streaming delta. It includes local HTTP, scheduling, template/tokenization, prompt evaluation, and first decode.
- **Prompt rate:** server `timings.prompt_per_second` using `timings.prompt_n` and `prompt_ms`.
- **Engine decode:** server `timings.predicted_per_second`; this is the primary generation-rate metric.
- **Streaming decode:** `(completion_tokens - 1) / (stream end - first output)`; an end-to-end approximation that includes local streaming overhead.
- **Wall throughput:** completion tokens divided by total request wall time; includes prefill and therefore is not pure decode.
- **MTP acceptance:** `draft_n_accepted / draft_n` when proposals are reported.
- **Power/energy:** differences in xe card/package cumulative energy counters over each whole request. Short tests can have counter/timing noise.

## Promotion and abort rules

This is a baseline, not an optimization candidate, so it has no speed-promotion decision. The run is valid only if:

1. all requests return usage and timing data;
2. the fixed configuration gate passes after every request;
3. no xe/OOM fault appears;
4. the 4K and 32K requests remain within 2% of their raw tokenizer target;
5. vision output correctly identifies the dominant `B70 INFERENCE` logo characteristics;
6. raw SSE, per-test JSON, prompts, metadata, and final summary remain under the approved ext4 workspace.

Any abort leaves the production process untouched and requires a health/PID/power/profile restoration check before further work.

## Execution addendum

The core run completed at `/mnt/<other-disk>/inference-models/baselines/qwen27-live-exact-20260718T231705+0200/`. Because the nominal output caps produced empty final content, bounded 768-token quality checks and a final 1,024-token text/vision gate were added. The 32K quality check returned the exact final record normally. Text and vision still returned empty content at 1,024 tokens. After the final vision request, `MemAvailable` was 19.86 GiB, below the 20 GiB guard, so the extension stopped and no further generation was attempted. See `live-exact-baseline-20260718.md` for results and evidence classification.
