---
title: "RQ-11 one-round attribution: blocked-as-found 2026-10-10 (foreign engine loop holds both cards)"
type: research
status: blocked-as-found
date: 2026-10-10
---

# RQ-11 one-round attribution — BLOCKED-AS-FOUND, 2026-10-10

**No measurement exists.** No server was started, no trace was captured, and every
share number reported upstream of this note is a schema-required void placeholder
(0.0), not data. The four buckets (routed-expert GEMM, dispatch/router, draft,
host waits+launch) remain UNMEASURED.

## What blocked it (exact commands and outputs)

Foreign engine loop — another session's `arm.sh`/`llama-stream` automation, uid 1000,
never started by this campaign, therefore not killable under the never-kill rule —
held engine-class clients on the cards throughout this ask:

- `pgrep -af 'llama-server|llama-bench|vllm serve|strata|sglang'` (grep -v dockerd),
  run ~20:0x CEST: PID 3704504 (`.../llama-stream-stack-r3/build/bin/llama-server -m
  Qwen3.8-27B-UD-Q4_K_S.gguf -ngl 99 --port 19870 --spec-type draft-mtp
  --spec-draft-n-max 1`).
- PID chain across ~40 min: 3704504 → 3713194 → 3742308 (`--spec-draft-n-max 4`) →
  3747406 → 3851017 (19:11:21Z, `--port 35053`, `-ctk q8_0`): the loop re-arms under
  new PIDs faster than the ~5.5 min vLLM boot (RL-137 behavior, observed first-hand).
- `sudo cat /sys/kernel/debug/dri/1/clients`: llama-server rows on renderD129;
  `sudo cat /sys/kernel/debug/dri/0/clients`: llama-server PID 3747406 rows on
  renderD128 as well (both cards held).
- `sudo cat /sys/kernel/debug/dri/1/tile0/vram_mm`: `visible_avail: 20492MiB`
  (later 30008 MiB between loop iterations) vs `visible_size: 32656MiB` — below the
  31000 MiB empty-card bar while the foreign model is resident.
- 20-minute background watch (40 polls × 30 s, 18:50:59Z–19:10:32Z): llama-server
  process count 2–3 at ALL 40 polls, never 0; `visible_avail ≥ 31000 MiB` appeared
  only as isolated single-poll readings (32615 at 18:50:59Z and 18:55:30Z; 32403 at
  19:01:01Z) — the seconds-long gap between loop iterations. The lead's window
  condition (zero engine-class clients on dri/1 AND ≥ 31000 MiB, sustained ≥ 15 min,
  re-checked immediately before boot) is unmet in kind.

Escalation answer (lead, this ask): no operator authorization exists to kill another
session's work; the owner decision is surfaced and unanswered. Directive: if no
window opens, do NOT fire the sealed anchor plan into a known-held card (a
preflight-certain refusal would consume a sealed attempt for zero information);
record blocked-as-found with this evidence and the re-entry condition below; consume
ZERO sealed attempts.

## Declaration required by the ask

- Still-warm vs fresh relaunch: **fresh relaunch was the call** — no RQ-11 server was
  ever started (the anchor plan `campaigns/2026/rq11-nemotron-v030-anchor/plan.json`
  is authored and launch-verified but has never executed, blocked by this same loop),
  so no warm server exists.
- Capture: would have been ON (`VLLM_XPU_ENABLE_XPU_GRAPH=1`, no enforce-eager);
  profiler surface verified in the pinned image
  `vllm/vllm-openai-xpu@sha256:fc0e112a...`: `--profiler-config` (arg_utils.py:1737),
  ProfilerConfig at config/profiler.py:39 (`profiler='torch'`, `torch_profiler_dir`
  absolute, `with_stack=True` default, gzip traces, `delay_iterations`/`max_iterations`
  for a bounded `/start_profile` window; `capture_torch_profiler` exists as a
  capture-time fallback). MoE expert kernel for the expert-GEMM bucket:
  `XPUExpertsWNA16` → `vllm_xpu_kernels.fused_moe_interface.xpu_fused_moe(is_int4=True)`
  (fused_moe/oracle/int_wna16.py:89-94, :1034). DFlash narrows graph capture to
  FULL_DECODE_ONLY (v1/worker/gpu/spec_decode/dflash/speculator.py:122-139), and the
  draft bucket is window-matched from CPU-track `DFlash*` frames, name patterns only
  secondary. If capture-on traces prove kernel-blind under graph replay, the named
  fallback is ONE `--enforce-eager` relaunch labeled diagnostic-only (AGENTS §7) —
  not exercised, no such trace exists.

## Prepared execution path (resume = run as-is, no re-authoring)

1. Verify window: `pgrep -c -f llama-server` == 0, dri/1 clients engine-free,
   `visible_avail >= 31000` — twice, 60 s apart.
2. Fresh relaunch on GPU1 (render node resolved live via
   `/dev/dri/by-path/pci-0000:0f:00.0-render`, gid 990), pinned digest, flags per
   `research/2026/v0300-nemotron-gptq64-dflash-n7-serve-source-20261010.md` plus
   `--profiler-config '{"profiler":"torch","torch_profiler_dir":"/work/traces"}'`
   with the traces dir bind-mounted to this directory; utilization clamped by the
   live free-minus-4-GiB rule as in the anchor plan.
3. One discarded warmup request; then POST `/start_profile` `{"max_iterations": 2}`,
   ONE C1 greedy request (unique entropy prefix, p128, g1, ignore_eos → exactly one
   DFlash decode round), POST `/stop_profile`; trace lands here.
4. `python3 parse_attribution.py <trace.json.gz> --dump-names` — parser compiled and
   smoke-tested on a synthetic trace this ask (shares sum 1.000000; draft windows
   take precedence over name patterns; host = wall − device-busy; residual named via
   the emitted kernel histogram). Final bucket names are to be grounded in the real
   trace's kernel histogram before any share is quoted.

## Re-entry condition (verbatim from the lead)

"anchor plan and attribution are authored and launch-verified (serve flags pinned
in-image, parse_attribution.py written); execution needs one >= 15-minute GPU1 window
free of engine-class clients; ZERO sealed attempts consumed; resume = execute the
existing plan as-is, no re-authoring."

## Retention

- This note + `parse_attribution.py` (validated) in this directory.
- Anchor plan: `campaigns/2026/rq11-nemotron-v030-anchor/plan.json` (+ workload.json,
  adapter `scripts/tmp/rq11_anchor_20261010.py` with a proven fail-closed preflight).
- Watch log (raw poll output): retained in the actor task output file of this session;
  the 40-poll summary is quoted above. Caps as-found throughout: 180 W (0000:0b:00.0),
  230 W (0000:0f:00.0); no cap writes, no processes started, no daemon state changes.
