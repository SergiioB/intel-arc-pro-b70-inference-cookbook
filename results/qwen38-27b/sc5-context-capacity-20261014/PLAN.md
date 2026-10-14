# SC5 context-capacity ladder — pre-committed plan (2026-10-14)

Agent: ContextCapacity. Result root: `results/2026/10/ctx-capacity-sc5-20261014/`.
Assignment: own context-capacity experiments + resulting recipe settings for the
dense27B EXL3 route on our SC5 image; NO launcher/registry edits; proposed recipe
values go to LauncherUniform / parent.

## Question (one)

What is the highest context that our **tested route** — dense27B **SC5**
(turboderp `SC_5.00bpw_H6`, mixed K 4-7, 20.09 GiB) on image
`ghcr.io/sergiiob/exl3xpu-mixedk:20261014@sha256:933eda54bb51a2e8a6669a70f5a2d7696fabd12a5f8b7c469df02fdfecc79d3a`
— validates with **FP8 KV**, staged **real prompt + real output**
(16K→32K→64K→128K→192K→240K→262144), retrieval + multi-turn + correctness,
and what `max_model_len` should the recipe freeze at?

## Route distinctions (frozen; never conflated)

| Route | Artifact | Weights | Route config (upstream 0xSero `models/qwen3.8-27b-exl3-4.00bpw/model.yaml`) |
|---|---|---|---|
| **plain 4bpw (upstream 0xSero)** | turboderp 4.00bpw @113cf7ab | 14.9 GiB | `max_model_len 262144`, `kv_cache_dtype fp8`, `gmu 0.965`, `max_num_seqs 16`, `max_num_batched_tokens 4096`, MTP3 (`qwen3_5_mtp n=3`), prefix cache on + `mamba_cache_mode align`, env `EXL3_DRAFT_VOCAB / EXL3_KV_BLOCK_EXACT=1 / EXL3_ONEDNN_ATTN=1 / EXL3_INT8_PREFILL=1`, FULL_DECODE_ONLY graphs; proven 261,920-tok prompt completion (pool 287,040 tok @ gmu 0.94) and "needle 6/6 at 131K/240K" |
| **SC5 (this experiment)** | turboderp `SC_5.00bpw_H6` @dfe2f4fd | 20.09 GiB | same route config; launcher builder picks gmu 0.94 at ctx≥131072, 0.90 below; recipe currently frozen `ctx 65536 / ctx_safe 16384` ("longer is unverified") |
| **Flash-Next** | different model entirely (MoE 125B-A6B; Strata GGUF route) | — | NOT this experiment, NOT proof for it |

Hybrid model (upstream model.yaml notes): 64 layers = 48 GDN + 16 gated full
attention, hidden 5120, 4 KV heads, head dim 256 → FP8 KV = 2·16·4·256 = 32 KiB/token
(+ block rounding). KV pool capacity ≠ usable context: pool is sized at startup
from gmu, `max_model_len` is a separate limit, and prompt+output must both fit.

270K aim: **not attempted — model/route limit is 262,144 native** (route config
`max_model_len 262144`, launcher `--ctx` capped at 262144); memory budget for SC5
is below plain's pool. Documented, per "aim 270K only if model limit AND memory permit".

## GPU window (contract)

Parent granted the current window; explicit START/STOP messages to Main. m2
(FlashNextKL) gets the GPU only after my STOP. One owned server at a time, one
campaign harness. GPU1 = card1 `0000:0f:00.0` / renderD129 (research card).
**As-found power caps recorded, never changed** (card0 hwmon5 = 180 W,
card1 hwmon6 = 230 W as found 2026-10-13T23:30Z; no change → nothing to restore).
No driver/power/PCI changes. Host RAM kept light (m2 streams 258 GB of BF16 reads
on CPU; prompts are MB-scale).

## Servers (existing pipeline only)

1. **Server A (ladder, capacity observation + candidate freeze):** launcher
   pipeline (`b70 launch qwen38-27b-sc5 -e exl3 --gpus 1 --port 8123 --no-test`).
   The installed launcher clamps `--ctx` to the recipe's `ctx_max` 65536 (its
   "longer is unverified" freeze), so the experiment axes are passed as declared
   `--extra` overrides (last `--set` wins in `scripts/serve.py`):
   `--extra '--set vllm.max_model_len=262144 --set vllm.gpu_memory_utilization=0.94 --set served_model_name=qwen3.8-27b-exl3'`
   → `max_model_len 262144`, gmu 0.94 (the builder's own ≥128K rule value), fp8 KV,
   MTP3, prefix cache on, FULL_DECODE_ONLY, the route envs — all other axes are the
   recipe/route defaults. Declared axes: max_model_len (the experiment), gmu 0.94
   (recipe's own long-ctx rule), served-model-name (cosmetic, so upstream adapters
   run verbatim). Nothing else is changed, silently or otherwise.
2. **Server B (recommendation, only if Server A leaves <2 GiB live free or its top
   rung is <2 GiB class):** same pipeline with the one declared gmu axis
   (`--extra '--set vllm.gpu_memory_utilization=0.92'`), `--ctx` = frozen candidate,
   validate one full staged set at the frozen context with measured ≥2 GiB free.

## Frozen protocol (sampling / tasks / stop conditions)

- Sampling: greedy, temperature 0.0, top_p 1.0, `enable_thinking: false`,
  max_tokens 256, no stop strings, no `ignore_eos` (natural real outputs),
  streaming with `include_usage`. Prompt cache: ON (route default) — cold cells
  require **zero** `vllm:prefix_cache_hits` delta (unique ≥64-char entropy tag at
  prompt start, inside the token budget); the warm multi-turn cell declares and
  requires a cache hit. All other axes = route/recipe defaults, nothing silent.
- Stages S ∈ 16384, 32768, 65536, 131072, 196608, 245760, 262144 = target total
  occupied context (prompt + output). Prompt target = S − 256 tokens, built from
  the upstream `bench/sweep.py` noise-document method (sized via the server's own
  `/tokenize`, exact), with the upstream `tests/test_needle.py` needle plant and
  question text verbatim.
- Per stage (method = upstream `tests/test_needle.py` + our
  `exl3xpu_mixedk_20261012/tests/multiturn.py`, timing/counter method =
  `scripts/bench/ladder.py` `chatreq` + `scripts/bench/prov.py` read-back):
  1. **cold retrieval depth 0.5** (secret-code needle, exact-match answer),
  2. **cold retrieval depth 0.9**,
  3. **warm multi-turn** on cell 2's conversation: code re-recall + first-word
     recall + format compliance (multi-turn/correctness at depth).
- Actual token counts from `usage.prompt_tokens/completion_tokens` (SSE chunks and
  words are never tokens). Per request: TTFT, client post-first tok/s, engine
  prefill/decode from `/metrics` deltas, spec acceptance (diagnostic only),
  cache-hit delta, VRAM `visible_avail` before/after (debugfs), finish/error.
- Stage PASS = all three cells complete `finish=stop` with correct exact answers,
  cold cache delta = 0, warm `cached_tokens` > 0, prompt tokens as recorded.
- **Probe gate:** stage 16K must pass before deeper stages (smallest falsifying
  probe first). Per-stage stop on first failure: retain the failure + one
  time-boxed diagnosis; retry only on a changed mechanism. Commitment deadline =
  my GPU window (m2's pending request): stop the ladder at the deadline and report
  partial stages as `not measured`.
- **Capacity classes (parent rule):** live free VRAM after cache allocation
  dictates the class — ≥2048 MiB → recommendation class; 1024–2047 MiB → staged C1
  research / **capacity observation, NOT a recommendation**; <500 MiB → abort.
  Configured gmu is never claimed as reserve.
- Correctness gates after the ladder: `canary.py` 6/6 exact + 3/3 long
  (existing `prompts_decoded.json`), `multiturn.py` PASS. Verbatim upstream
  `tests/test_needle.py` cross-check at the frozen context and at 131072
  (comparability with the upstream "needle 6/6 at 131K/240K" claim).

## Request count / runtime estimate

Server A: 7 stages × 3 requests = 21 + 1 warmup (discarded) + canary 9 + multiturn 4
≈ 35 requests. Prefill dominates: ~2–5 min per top-rung cold cell → ladder ≈ 45–75 min
after a ~6–8 min load. Server B ≈ 20–30 min. Total ≈ 2 h, bounded by m2's GPU request.

## Deliverable

- Best validated completion (actual counts, evidence rows) and a **lower safe
  recommendation** (`ctx`/`ctx_safe`/gmu/max_model_len) with measured ≥2 GiB free,
  proposed to LauncherUniform + parent (NO edits by me).
- Failures, uncertainties (pool-vs-context, gmu sensitivity), plain4bpw vs SC5
  footprint distinction, Flash-Next exclusion. Cleanup proof + as-found state.

## Amendment 1 (2026-10-13T22:06Z) — Server A load FAILED (retained), budget established

First launch (max_model_len 262144, gmu 0.94) failed at engine init — retained log
`~/.local/state/b70-launcher/logs/qwen38-27b-sc5-exl3-8123-1791928791309532460.log`:

> `ValueError: To serve at least one request with the model's max seq len (262144),
> 9.29 GiB KV cache is needed, which is larger than the available KV cache memory
> (7.5 GiB). Based on the available memory, the estimated maximum model length is
> 206400.` (+ "Add 3 padding layers, may waste at most 6.25% KV cache memory")

This IS the budget answer at gmu 0.94 for SC5: **engine max seq len 206,400** —
the 245,760 and 262,144 rungs are budget-infeasible on SC5 at this gmu (9.29 GiB
needed vs 7.5 GiB available). At the upstream route gmu 0.965 the engine would get
~8.3 GiB → est. max ≈ 235K, still below the 240K rung; 240K+ needs gmu ≈ 0.982
(≈600 MiB free = isolated-observation/abort class). Plain 4bpw (14.9 GiB) is a
separate route with a larger pool and reaches 262K. Per parent: bounded ladder
stop at budget; impossible stages not run.

Changed mechanism for retry (protocol: retry only on a changed mechanism):
`max_model_len 262144 → 206000` (below the engine's own 206,400 estimate, leaving
room for the warm multi-turn growth of ~320 tokens above each stage). Ladder rungs
become 16384, 32768, 65536, 131072, 196608, 204800; all stages now inside budget.
Declared axes unchanged otherwise (gmu 0.94, served-name cosmetic).

## Amendment 2 (2026-10-13T22:2xZ) — cache telemetry absent on this fork; criterion revised

The running fork (exl3xpu image, vLLM V1 engine) exposes `vllm:spec_decode_*` counters
but **no `vllm:prefix_cache_*` counters**, and chat usage carries no
`prompt_tokens_details.cached_tokens` (both null in every row). Zero-hit on cold cells
is therefore guaranteed by construction only (unique entropy tag + distinct word
sequence per request — no shared prefix above the tag), which is weaker than a
counter-verified zero delta: flagged as an uncertainty. Warm-cell cache state is now
adjudicated by warm-vs-cold TTFT ratio at equal conversation length (<0.34x = hit;
observed 0.97 s warm vs 6.7-11.6 s cold at 16K), plus any telemetry that exists.
Multi-turn CORRECTNESS is judged independently of cache telemetry (CODE=/WORD= exact).
This applies to Server B onward; Server A rows retain the raw criterion fields
(`code_ok`, `word_ok`, `cache_hit`, `ttft_s`) for identical re-adjudication.

Server A provenance read-back (live): max_model_len 206,000; GPU KV cache size 206,000
tokens; gmu 0.94; fp8 KV; MTP3 (spec counters live: 72 draft / 71 accepted tokens early);
prefix cache ON (mamba align, experimental warning logged); free VRAM 2366 MiB idle after
load but **1069 MiB under sustained long-context load** (pool allocated) → the honest
live reserve for class assignment is the loaded value.

## Amendment 3 (2026-10-13T23:5xZ) — rung 204,800 harness failure; Server B load fault + single retry

- **Rung 204,800 on Server A: not completed (harness wall-time, not a model failure).**
  Two attempts timed out (3600 s cap during the run; 1500 s on the standalone retry) with
  no completed cell: the exact-targeting binary search issues ~15 `POST /tokenize` probes
  of 300K-600K-token strings per prompt build and dominates wall time at >196K targets.
  Server A closes at its highest PASSED stage 196,608 per Main ("close A at highest
  passed"). Builder replaced by one tokens-per-word calibration + <=4 ratio corrections
  for Server B (mechanism change, documented above `build_prompt_v2`).
- **Server B (gmu 0.90, max_model_len 175,000) load attempt 1: HOST FAULT, retained.**
  Engine hung at safetensors shard 2/3 for 20+ min; kernel: `Engine memory CAT error
  [18]: class=bcs` -> `Engine reset: engine_class=bcs` -> `Timedout job ... in python3`
  -> `Xe device coredump` (GPU1, 0000:0f:00.0). Identical signature to findings §6
  (2026-10-13 SC5 load hangs; RL-178 class) — it struck despite
  `UR_L0_USE_COPY_ENGINE=0 + SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0`, so the 2026-10-13
  4/4 screen does not prove the fault gone. Evidence: `serverB_fault_dmesg.txt`,
  `serverB_devcoredump_head.txt`, `serverB_load_hang_logtail.txt`.
  Recovery: `b70 stop` (container only, never touched other workloads); VRAM returned to
  the 32,615 MiB empty-card baseline; driver engine reset cleared the fault.
- **Retry (same config; infra-fault recovery, time-boxed 720 s) is the single retry.**
  If it faults again, B is closed with NO >=2 GiB recommendation and the blocker is the
  answer. No further attempts this window (m2/GPU priority).
