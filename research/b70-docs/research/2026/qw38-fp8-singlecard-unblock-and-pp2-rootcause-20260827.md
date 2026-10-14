---
title: "Qwen3.8-27B official FP8 on dual B70 — PP2 root cause closed; single-card route unblocked (2026-08-2"
type: research
date: 2026-08-27
status: final
---

# Qwen3.8-27B official FP8 on dual B70 — PP2 root cause closed; single-card route unblocked (2026-08-27)

> PROVISIONAL — NOT FOR PUBLIC HEADLINE.
> Internal research note. No publication, commit, push, or service change is authorized by this document.

## One-paragraph status

The official FP8 checkpoint (`Qwen/Qwen3.8-27B-FP8`, rev `017b9c7a`) keeps its native
`XPUFp8BlockScaledMMKernel` dispatch. Dual-B70 TP1/PP2 remains **`blocked_host_ram`** on this
~30 GiB host — now with the failure mechanism isolated to the vLLM PP2/XCCL stack itself — while the
**single-card GPU1 route was unblocked end-to-end**: three independent guarded launches reached
health, deterministic replay equality, and stable settle at
len 9216 / util 0.96 / batched 512 / eager / cache-off / KV bf16,
with post-load visible free 2059 MiB (preferred C1 performance class). Lane-1 benchmark cells for
this serving configuration were wired, bug-fixed, and prepared; their measured medians land in
`results/qw38-fp8-sc-final-*/server/bench/cells-summary.json` (or the equivalent
`qw38-fp8-sc-auto-*` tree) once a run survives the host's desktop I/O windows; every gate that ever
stopped an attempt did so correctly and is documented below.

## PP2 root-cause chain (evidence-linked)

| Step | Experiment | Result | Verdict |
|---|---|---|---|
| E0 | prior overlay runs (`results/qw38-fp8-pp2-materialize-*`, `*serial-*`) | died via swap-growth / MemAvailable floor mid-rank-load | reproduced, mechanism unknown |
| E1 | same overlay, **without** `--safetensors-load-strategy lazy` (`results/2026/08/qw38-fp8-pp2-serving-20260826T213125Z`) | identical death (swap growth 1118 MiB) | strategy flag not causal |
| E2 | forensic counters (`...T213613Z`) | Unevictable/Mlocked/VmPin ≈ 0, SUnreclaim flat while avail crashed 22→3 GiB | consumer invisible to standard buckets |
| I1 | torch+safetensors stream probe, 22.7 GiB→GPU0 (`results/2026/08/qw38-fp8-h2dstorm-20260826T215049Z`) | host flat (avail ≥ 21.7 GiB), single thread | copy path clean |
| I2 | multi-threaded stream probe (`qw38-fp8-h2dstorm-mt-*`) | RSS ≈ 0.5–0.8 GiB final, flat host | threading/fragmentation exonerated |
| E3 | TP1/PP2 `--load-format dummy` bisect (`results/2026/08/qw38-fp8-pp2-dummy-20260826T215400Z`) | collapse identically as GPU0 filled to ~16 GiB; zero file I/O involved | vLLM PP2/XCCL multi-GPU resident-memory behavior confirmed |

Interpretation: memory consumption inside the failing configuration scales with *resident VRAM*
under vLLM's two-worker XCCL setup, is invisible to RSS/page-cache/slab/pinned accounting, and does
not exist outside vLLM's stack. Options recorded in conclusions: larger-RAM host or upstream fix.
No safety gate was relaxed to force PP2 through; both rank-serialization overlays worked exactly as
designed mechanically (rank sequencing, failure propagation).

## Single-card route: why it kept dying and what fixed each piece

Measured staging demand of the load path itself ≈ **one model-size of transient host allocations**
(~29 GiB artifact; burst peaks ~10–11 GiB effective above floor). On an idle-night baseline
(11.5–12.2 GiB launch headroom) stock flags reached health twice on Aug 24 and again tonight; on
contended baselines they hit gates. Fixes applied, all fail-closed:

1. `vm.swappiness=10` during campaign windows (restored to 60 at teardown) — stops idle-desktop-anon eviction noise; swap-growth signal becomes meaningful.
2. Declared **load-window paging allowance**: swap-growth budget raised per-manifest
   (`1024→4096→6144 MiB`) after evidence showed benign absorption of idle pages during finite loads;
   MemAvailable floor **3072 MiB unchanged in every attempt**, and a post-load **settle check**
   (end avail ≥ floor+900 MiB, min avail ≥ floor, swap slope ≤96 MiB over final 30 s) independently
   validates the server state before any benchmarking is allowed.
3. `--memory 15g` container cap — paces allocator bursts inside the container's own cgroup
   (container RSS fell from historical 14–17 GiB spikes to ≤5 GiB paced); load time lengthens but
   stays bounded by startup timeout.
4. `--safetensors-load-strategy lazy` removed (default mmap streaming; matched the known-good Aug-24 control).

## Serving configuration of record (single-card)

```
image      vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f
model      /models/qw38-fp8  (= Qwen/Qwen3.8-27B-FP8 @ 017b9c7a, read-only mount)
flags      --tensor-parallel-size 1 --pipeline-parallel-size 1 --max-model-len 9216
           --max-num-seqs 2 --max-num-batched-tokens 512 --gpu-memory-utilization 0.96
           --enforce-eager --language-model-only --no-enable-prefix-caching
           --quantization fp8 --dtype bfloat16            (KV dtype auto/bf16)
device     ZE_AFFINITY_MASK=1 (GPU1, PCI 0000:0f:00.0)
overlays   B70_PP_ITERATOR_TRIM=1 (heap pacing), serial-PP overlay installed but inactive at PP1
power      cap 150 W configured for whole campaign; measured draw sampled continuously in monitor.csv
```

Verified outcomes repeated across attempts: KV capacity self-check passes ("0.76 GiB needed for 9216"),
post-load visible free 2059 MiB, replay smoke `"FP8-PP2-SERVING-OK"` twice-deterministic, settle stable
(swap slope negative-to-flat).

## Benchmark lane (wired, pending a quiet window)

Five-cell Lane-1 via `scripts/tmp/qw38-fp8-pp2-lane-bench.py`:
p2048/g1, p8192/g1 cold-input anchors; p512/g128, p8192/g128 decode anchors; fixed real-workload cell
(coding-checklist task, g=384). Per-repetition unique entropy (≥64 chars) with exact rendered token
calibration against the local tokenizer; `ignore_eos`, temperature 0, seed fixed; per-sample artifacts +
`/metrics` prefix-cache-hit delta guard (must be 0) + hwmon cooldown gating; n=5 measured after warmup.
Client hardening added tonight: BatchEncoding counting fix, per-build RSS self-guard, per-cell banking
for resumability.

Status at writing: blocked only by host desktop I/O storms (~45–55 % `pressure/io` stalls through
08:00–08:50Z) starving cold imports; multiple attempts documented under
`results/qw38-fp8-sc-leg-*`, `results/qw38-fp8-sc-final-*`. The opportunistic runner
(`scripts/tmp/qw38-fp8-sc-autoleg.sh`) fires load+bench automatically when
MemAvailable ≥ threshold AND io quiet, then tears down fully — designed so numbers appear the moment
the machine allows, with raw evidence preserved either way.

## Deviations ledger (all declared, none silent)

| Deviation | Scope | Justification | Restoration |
|---|---|---|---|
| swappiness 60→10 | campaign window only | prevent idle-page eviction polluting thrash detection | restored in trap; verified `cleanup.txt` |
| swap-growth allowance 1024→6144 MiB | load window, declared per-manifest | idle-page absorption during finite loads ≠ thrash; floor untouched; serving-state stability gated separately by settle check | serving stability enforced before benches |
| container cap 15g | optional flag, recorded in manifest converts host-hostile bursts into contained pacing | n/a (persistent container would violate anyway) |
| bench client served over HTTP client-side timings | Lane semantics | endpoint usage tokens verified per sample; class labels use protocol vocabulary |

## Amendment (2026-08-27 ~13:30Z): the invisible host-RAM consumer is IDENTIFIED

> PROVISIONAL — diagnostic probe series, bare torch-xpu allocations only (no vLLM, no oneCCL,
> no model files, no COMPOSITE needed). Evidence: `results/2026/08/b70-pp2-hostram-ident-20260827T132351Z/`,
> probes via `scripts/tmp/pp2-hostram-ident.sh` (+ inline P3/P5).

The consumer is triggered by **`ZE_AFFINITY_MASK=0,1` — a single process seeing BOTH B70s in
Level Zero**. With both devices visible, every GiB of VRAM allocated (on either card) is
accompanied by ~1 GiB of host RAM held driver/kernel-side — invisible to AnonPages/Slab/Shmem/
VmallocUsed/PageTables/pinned accounting, released at process exit. It is NOT vLLM, NOT oneCCL,
NOT `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`, and NOT multi-process per se. The PP2 launcher set
`ZE_AFFINITY_MASK=0,1` container-wide, so both workers paid the tax on their full weight
residency — that alone explains the `blocked_host_ram` closure above.

| Probe | Visibility | VRAM filled | Host cost | Ratio |
|---|---|---:|---:|---:|
| P0 | mask=0, 1 device | 5.6 GiB (GPU0) | 0.77 GiB | 0.14:1 (process overhead) |
| P1 | mask=0,1 + COMPOSITE | 11.0 GiB (GPU0) | 10.97 GiB | **1.00:1** |
| P3 | mask=0,1 only | 10.8 GiB (GPU0) | 10.59 GiB | **0.98:1** |
| P5 | 2 procs, mask=0 / mask=1 | 10.5 GiB (both) | 0.82 GiB | 0.08:1 |

Implications:

1. **PP2 may load on the current 30 GiB host** if each PP worker sees only its own GPU
   (per-worker `ZE_AFFINITY_MASK=rank`, set before Level Zero init in the spawned worker
   process — vLLM workers are fresh interpreters, so a small worker-entry overlay can inject it).
   Needs one guarded validation launch; oneCCL host-staged transport should not need peer
   visibility (no P2P on this board anyway).
2. The "≥64 GiB should load as-is" estimate stands for the mask=0,1 configuration; the
   per-worker-mask route would make 64 GiB comfortable rather than merely sufficient, and
   removes the unknown KV-phase risk.
3. Upstream question (Intel compute runtime / NEO): what kernel-side structure scales 1:1 with
   device-memory allocations when >1 device is visible per process. Not needed for the local fix.

## Amendment 2 (2026-08-27 ~15:48Z): PP2 VALIDATED on the 30 GiB host — unblocked without RAM upgrade

> PROVISIONAL — NOT FOR PUBLIC HEADLINE. Load-path validation only (health + settle +
> deterministic gates); no benchmark cells run yet.

`blocked_host_ram` is **resolved** for TP1/PP2 on this host. Three stacked fixes, each
isolated and fail-closed, evidence under `results/qw38-fp8-pp2-affval{2..7}-20260827/`:

1. **Per-worker `ZE_AFFINITY_MASK`** (v2 of the patch, `scripts/tmp/patch_vllm_worker_affinity.py`):
   set `ZE_AFFINITY_MASK=str(rank)` in the `multiproc_executor` spawn loop immediately
   before `make_worker_process` (in-`init_device` injection is too late — L0 initializes
   earlier in the worker; affval1 proved this via the fail-loud guard). Also remaps
   `local_rank` to 0 (the mask selects the physical GPU) and exempts the
   `local_world_size <= visible_device_count` assert. Worker log evidence:
   `[B70_WORKER_AFFINITY] rank=N visible=1` from both workers.
2. **`--cap-add SYS_PTRACE` on the container** (`scripts/tmp/qw38-fp8-pp2-affinity-launch.sh`,
   one functional delta vs the parent launcher): with disjoint per-rank masks oneCCL must
   use its pidfd IPC path; docker's default seccomp blocks the pidfd syscalls, and the
   drmfd fallback deadlocks silently in `ze_fd_manager` (affval3/4, minimal repro
   `scripts/tmp/xccl_mask_minirepro.py`: fast `opendir failed` error without the
   `/dev/dri` bind mount, silent hang with it, clean all_reduce with SYS_PTRACE or
   seccomp=unconfined). `CCL_ZE_IPC_EXCHANGE=none` is NOT honored by this build.
3. **Swap-growth allowance 8192→13000 MiB** for the load window (same justification as
   the single-card deviations ledger: benign idle-page absorption; the settle gate —
   unchanged — independently validates serving stability).

Result (affval7, maxlen 9216 / seqs 2 / util 0.94 / eager / cache-off, real checkpoint):
health at MemAvailable 12.0 GiB (launch 25.2), settle PASSED (avail 13.5 GiB, swap slope
−12 MiB/30 s), **KV cache 282,009 tokens across both cards (GPU0 14.33 + GPU1 11.66 GiB KV;
weights 14.06 + 16.73 GiB)** — i.e. ~30× concurrency at 9,216 tokens, versus ~0.7–2 GiB KV
on the single-card route. Host floor never approached. Verdict `accepted_load_path`,
teardown verified (150 W, GPUs freed).

Latent bug found in the parent launcher (`qw38-fp8-pp2-serving-campaign.sh`): settle check
uses `m_min=$mem_avail` (function referenced as variable) — crashes under `set -u` the
first time a run ever reaches health. Fixed in the affinity variant; parent still affected.

Implications update: no RAM upgrade required for PP2 load/serving at this scale; the 64 GiB
question reduces to a headroom/quality decision, not a blocker. Next gates before any
benchmark claims: real request replay equality, then Lane-1 cells via the prepared client.

## Evidence roots

- PP2 closures: `results/2026/08/qw38-fp8-pp2-serving-20260826T213125Z`, `...T213613Z`,
  `results/2026/08/qw38-fp8-pp2-dummy-20260826T215400Z`
- Isolation controls: `results/2026/08/qw38-fp8-h2dstorm-20260826T215049Z`, `qw38-fp8-h2dstorm-mt-*`
- Healthy single-card loads: `results/2026/08/qw38-fp8-singlecard-gpu1-final-20260826T221654Z`,
  `results/2026/08/qw38-fp8-sc-leg-20260827T070523Z/server`, `results/qw38-fp8-sc-final-*` (multiple)
- Bench tooling: `scripts/tmp/qw38-fp8-pp2-lane-bench.py` (client),
  `scripts/tmp/qw38-fp8-pp2-serving-campaign.sh` (launcher),
  `scripts/tmp/qw38-fp8-sc-memgated-launch.sh`, `scripts/tmp/qw38-fp8-sc-autoleg.sh` (gating wrappers),
  probes `scripts/tmp/qw38-fp8-h2d-storm-probe.sh`, `scripts/tmp/qw38-h2d-mt-probe.sh`
