# SC5 context-capacity ladder — results (ContextCapacity, 2026-10-14)

Result root: `results/2026/10/ctx-capacity-sc5-20261014/`. Plan + amendments: `PLAN.md`.
Exact commands: `commands.log`. Method provenance: `runner.py` docstring (upstream
exl3xpu `tests/test_needle.py` + `bench/sweep.py` needle/sizing method, our
`exl3xpu_mixedk_20261012/tests/multiturn.py` warm multi-turn pattern,
`scripts/bench/ladder.py` request/counter method, `scripts/bench/prov.py` read-back).

## Route identity (never conflated)

| | artifact | weights | image | route config |
|---|---|---|---|---|
| **SC5 (this run)** | turboderp `SC_5.00bpw_H6` @dfe2f4fd (mixed K 4-7, mtp 4-bit head) | 20.09 GiB (18.53 GiB checkpoint shards) | `ghcr.io/sergiiob/exl3xpu-mixedk:20261014@sha256:933eda54bb51a2e8a6669a70f5a2d7696fabd12a5f8b7c469df02fdfecc79d3a` | upstream 0xSero `models/qwen3.8-27b-exl3-4.00bpw/model.yaml`: fp8 KV (only dtype), MTP3, prefix cache ON + `mamba_cache_mode align`, `max_num_seqs 16`, `max_num_batched_tokens 4096`, FULL_DECODE_ONLY graphs, env `EXL3_DRAFT_VOCAB / EXL3_KV_BLOCK_EXACT=1 / EXL3_ONEDNN_ATTN=1 / EXL3_INT8_PREFILL=1` |
| plain 4bpw (NOT run here) | turboderp 4.00bpw @113cf7ab | 14.9 GiB | upstream `ghcr.io/0xsero/exl3xpu:0.1.0` | same route config incl. `gmu 0.965`, `max_model_len 262144`; proven 261,920-tok prompt completion (pool 287,040 tok @ gmu 0.94) and upstream "needle 6/6 at 131K/240K" |
| Flash-Next | different model (MoE) | — | Strata GGUF route | out of scope; nothing here proves anything about it |

Model = hybrid Qwen3.5 (64 layers: 48 GDN + 16 full attention, hidden 5120, 4 KV
heads, head dim 256) → FP8 KV ≈ 32 KiB/token theoretical, ≈37.2 KiB/token with
64-token blocks + 3 padding layers (engine log). **Pool capacity ≠ context**: the
pool is a single token budget shared by up to 16 sequences; `max_model_len` bounds
one sequence; prompt AND output must both fit.

## Budget verdicts (engine's own numbers, retained failures)

| config | available KV | engine max seq len | note |
|---|---|---|---|
| SC5 @ gmu 0.94 | 7.5 GiB | **206,400** (log: "estimated maximum model length is 206400") | 262,144 needs 9.29 GiB → Server A attempt 1 aborted at init (retained) |
| SC5 @ gmu 0.90 | 6.23 GiB | **168,000** (175,000 needs 6.48 GiB → B attempt 2 aborted, retained) | the ≥2 GiB-reserve config |
| SC5 @ gmu 0.965 (upstream route value) | ~8.3 GiB (derived) | ≈235K (derived) | still below the 240K rung |
| SC5 @ 245,760+ | needs gmu ≈ 0.982 | — | ≈600 MiB free → below operating floor; **240K/262K/270K not achievable on SC5, by live bytes** |
| plain 4bpw (separate route) | bigger pool (5.2 GiB less weights) | 262,144 native proven | unchanged |

270K is out of scope on model/route grounds as well (`max_model_len` native ceiling
262,144; launcher `--ctx` cap 262,144).

## Server A — ladder at gmu 0.94 (CAPACITY OBSERVATION class)

Declared axes: `max_model_len 206000` (budget-derived), `gmu 0.94` (the launcher
builder's own ≥128K rule value), `served_model_name=qwen3.8-27b-exl3` (cosmetic, so
upstream adapters run verbatim). Everything else = recipe/route default. Read back
from the live server: `max_model_len 206000`, `GPU KV cache size: 206,000 tokens`,
fp8 KV, MTP3 (spec counters live), prefix cache ON. GPU1 230 W as-found (never
changed); sustained thermal regime (pkg 82 °C / VRAM 80 °C mid-run, no cooldown
between stages — noted; capacity claim only, **no perf claims**).

Frozen protocol: greedy, thinking off, max_tokens 256, natural stop, real
noise-document prompts sized to stage−256 with the upstream needle plant + question
verbatim; per stage = 2 cold retrieval cells (depth 0.5/0.9, unique entropy tag) +
1 warm multi-turn follow-up (CODE= / WORD= exact). Stage pass = all three cells
complete with exact answers.

| stage | occupied context (max) | needle 0.5 | needle 0.9 | multi-turn code/word | warm TTFT vs cold | free VRAM |
|---|---|---|---|---|---|---|
| 16,384 | 16,146 | hit (16,140+6) | hit | ✓/✓ | 0.97 s vs 6.7 s (0.14x) | 1069 MiB |
| 32,768 | 32,598 | hit (32,523+6) | hit | ✓/✓ | 1.38 s vs 15.1 s (0.09x) | 1069 MiB |
| 65,536 | 65,365 | hit (65,292+6) | hit | ✓/✓ | 2.26 s vs 35.8 s (0.06x) | 1069 MiB |
| 131,072 | 130,899 | hit (130,826+6) | hit | ✓/✓ | 3.48 s vs 98.4 s (0.04x) | 1069 MiB |
| 196,608 | **196,438** | hit (196,364+6) | hit | ✓/✓ | 4.65 s vs 203.9 s (0.02x) | 1069 MiB |
| 204,800 | not completed | — | — | — | — | — |

- **All five rungs PASS** (retrieval exact at both depths at every rung; multi-turn
  exact at every rung; independent adjudication in `ladder_A-summary.json`).
- **Highest validated occupied context: 196,438 tokens** (real prompt 196,364 +
  real outputs + warm follow-up). Class: **capacity observation only** — free VRAM
  under sustained load was 1069 MiB (< 2 GiB target). TTFT/rate figures are
  diagnostics only (no perf claims at <1 GiB-class reserve; also single samples).
- Rung 204,800 (inside the 206,000 budget): **not completed — harness wall-time
  failure, not a model failure** (exact token targeting probed `/tokenize` with
  300K-600K-token strings ~15x per prompt build; two attempts timed out, retained).
  Builder fixed afterwards (1 calibration + ≤4 corrections) for Server B.
- Cache telemetry: this fork exposes **no `vllm:prefix_cache_*` counters and no
  `usage.prompt_tokens_details`**. Cold zero-hit is by construction only (unique
  entropy + distinct word sequence per cell) — flagged uncertainty. Warm cache
  behavior evidenced by warm-vs-cold TTFT ratio at equal length (table above).
- Correctness context: this exact image digest carries the standing battery
  (6/6 exact canaries + 3/3 long, runtime record 2026-10-14); this window adds
  in-context correctness at every depth (exact needle codes + exact CODE=/WORD=
  multi-turn format answers).

## Server B — ≥2 GiB recommendation cell (gmu 0.90)

Config read back live: `max_model_len 168,000`, `GPU KV cache size: 168,000 tokens`
(available KV 6.23 GiB), gmu 0.90, fp8 KV, MTP3, prefix cache ON — all route defaults
except the declared `max_model_len`/`gmu`/served-name axes.

| stage | occupied context (max) | needle 0.5 | needle 0.9 | multi-turn code/word | warm vs cold TTFT | free VRAM (loaded) |
|---|---|---|---|---|---|---|
| 131,072 | 130,872 | hit (130,829+6) | hit (130,798+6) | ✓/✓ | 3.45 s vs 100.5 s (0.034x) | **2327 MiB** |
| 167,424 | not verified | — | — | — | — | — |

- **Stage 131,072 PASSES at gmu 0.90 with 2327 MiB free under sustained load
  (>= 2 GiB) → RECOMMENDATION CLASS.** This is the freeze anchor: real prompt
  130,829/130,798 tokens + real outputs + warm follow-up, exact retrieval at both
  depths, exact multi-turn answers, cache-warm follow-up (TTFT ratio 0.034).
- Stage 167,424 (the full-budget cell): **not verified — harness/client wall-time
  failure, retained** (the server log shows the two chat completions returning
  200 OK server-side after the client pipeline was killed at its 1500 s cap, so
  the responses could not be checked; no row, no claim). Same client-side failure
  class as rung 204,800 on Server A. Per the pre-committed rule (no retry loops)
  B closes here.
- Measured budget chain for SC5 (engine-adjudicated, both retained failures):
  **gmu 0.90 → 6.23 GiB KV → max seq len 168,000; gmu 0.94 → 7.5 GiB → 206,400;
  gmu 0.965 → ~8.3 GiB → ~235K (derived)**. 240K/262K/270K unreachable on SC5
  within any acceptable free-VRAM floor.

## Host fault (retained)

Server B load attempt 1 (gmu 0.90 / 175,000) died to the known SC5 weight-upload
fault class despite the copy-engine mitigation: `Engine memory CAT error [18]:
class=bcs` → `Engine reset` → `Timedout job ... in python3` → Xe device coredump,
hanging at safetensors shard 2/3 for 20+ min. Evidence: `serverB_fault_dmesg.txt`,
`serverB_devcoredump_head.txt`, `serverB_load_hang_logtail.txt`. The 2026-10-13
4/4-with-env screen does not prove this fault gone (5 loads observed today, 1 fault).
Recovery: container stop only; VRAM returned to the 32,615 MiB empty-card baseline;
driver engine reset cleared it.

## Cleanup / state

- Servers stopped (`b70 stop`), no containers left, GPU1 back to empty-card VRAM.
- Power caps untouched (card0 180 W, card1 230 W as found → unchanged, nothing to
  restore). No driver/power/PCI changes. `guard_after.log` appended at close.
- Uncertainties: (1) zero-hit cold-cache not counter-verifiable on this fork;
  (2) 196,608→206,400 gap unvalidated (harness wall-time); (3) single-sample cells
  (capacity protocol, not speed); (4) BCS fault rate on SC5 loads remains open.

## Proposed recipe values for `qwen38-27b-sc5` (proposed only — NOT applied)

| field | current | proposed | basis |
|---|---|---|---|
| `ctx` (default) | 65,536 | **65,536 (unchanged)** | fine for everyday use |
| `ctx_safe` | 16,384 ("not a cap" per user) | **131,072** | fully validated TODAY at gmu 0.90 with 2327 MiB free (>=2 GiB): retrieval + multi-turn + exact answers at 130,872 occupied |
| `ctx_max` | 65,536 | **168,000** | engine budget at gmu 0.90 (est max seq len; pool 168,000 tok). Top cell (167,424) staged validation is OWED — label in ctx_note, do not present 168K as validated |
| `ctx_note` | "…8,162-token prompts… longer is unverified" | rewrite | see wording below |
| builder gmu rule | 0.90 below 131K, 0.94 at >=131K | **for SC5: keep 0.90 at long ctx** (or >=0.93 only for explicitly-observation launches) | at gmu 0.94 the 20.09 GiB SC5 footprint leaves only 1069 MiB free under load (observation class); 0.90 leaves 2327 MiB |

Proposed `ctx_note` wording:
> SC5 (20.09 GiB): fp8 KV pool is one shared token budget (max_num_seqs 16); pool capacity != context. Engine budget at gmu 0.90: max seq len 168,000 (6.23 GiB KV). Validated 2026-10-14: 131,072 occupied (prompt+output+multi-turn, exact retrieval at 0.5/0.9 depths) with 2327 MiB free; capacity scope observed to 196,438 occupied at gmu 0.94 (1069 MiB free = observation class, not a recommendation); 206,400 is the gmu-0.94 budget ceiling, 240K+/262K/270K not achievable on SC5. Plain 4bpw (14.9 GiB) is a separate route: native 262,144 (proven 261,920-tok completion). Cache counters absent on this image — cold cells are zero-hit by construction only.

Proposed values are for LauncherUniform/parent to apply through the recipe channel —
I changed nothing in the launcher or any registry.
