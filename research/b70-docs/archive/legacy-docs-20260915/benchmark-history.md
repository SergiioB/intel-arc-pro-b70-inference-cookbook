---
doc_id: DOC-COMPAT-BENCHMARK-HISTORY
doc_type: compatibility_view
title: B70 Benchmark History
lifecycle: active
updated: 2026-08-19
canonical_for:
  - benchmark-history-compatibility
---

# B70 Benchmark History

The maintained compact campaign view is `docs/indexes/campaign-index.md` (`indexes/campaign-index.md`), backed by `registry/campaigns.yaml` (`../registry/campaigns.yaml`). Canonical campaign coverage is intentionally incremental: only high-confidence campaigns with explicit identities are promoted there.

The complete pre-migration chronological run log remains at `archive/docs/pre-migration-20260819/benchmark-history.md` (`../archive/docs/pre-migration-20260819/benchmark-history.md`). Use it for runs not yet represented in the canonical campaign registry. Historical entries and corrections remain evidence, but their original ordering and wording do not make them current recommendations.

## Current canonical campaign records

- Muse Glimmer Q4_K_XL DFlash (`../campaigns/2026/muse-glimmer-q4kxl-dflash/README.md`)
- Nemotron DFlash n7 isolated C1 n=5 (`../campaigns/2026/nemotron-dflash-n7-n5/README.md`)
- Ornith 1.5 day-0 MTP screen (`../campaigns/2026/ornith15-day0-mtp-screen/README.md`)
- Ornith 1.5 MixedCal-v2 characterization (`../campaigns/2026/ornith15-mixedcal-v2/README.md`)
- Qwen3.6 exact-128K matched cache/spec (`../campaigns/2026/qwen36-pi-128k-matched/README.md`)
- Qwen3.8 four-mode 230 W (`../campaigns/2026/qwen38-4mode-230w/README.md`)

## Authority and limits

- Raw `results/` evidence remains immutable and authoritative for observations.
- Campaign records provide bounded interpretation and explicit status.
- Corrections and supersessions live in `registry/supersessions.yaml` (`../registry/supersessions.yaml`) and `decisions/corrections-and-supersessions-2026-08-19.md` (`../decisions/corrections-and-supersessions-2026-08-19.md`).
- Unknown historical coordinates remain `unknown`; they are not inferred during migration.
- A run is not publication-approved merely because it appears in either history view.

## Session entries — 2026-08-27

- **Qwen3.8-27B official-FP8 PP2 root-cause closure + single-card unblock** — PP2 dual-B70
  `blocked_host_ram` confirmed under full instrumentation and explained: not load-strategy,
  not xe copy path (isolated torch H2D storms flat), not pinned/slab/tmpfs; consumption
  tracks resident VRAM inside vLLM's TP1/PP2 XCCL stack including a `--load-format dummy`
  bisect with zero checkpoint I/O. Single-card FP8 route unblocked on GPU1 at
  len 9216 / util 0.96 / batched 512 / eager / cache-off with deterministic replay equality;
  Lane-1 benchmark cells launched under declared load-window paging allowance
  (MemAvailable floor 3072 MiB never relaxed) — see campaign REPORT.md for measured cell
  medians if completed, else honest partial-state record.
  Evidence: `results/qw38-fp8-pp2-serving-*`, `results/qw38-fp8-h2dstorm-*`,
  `results/qw38-fp8-pp2-dummy-*`, `results/qw38-fp8-singlecard-gpu1-final-*`,
  `results/qw38-fp8-sc-leg-*`, `results/qw38-fp8-sc-final-*`, `results/qw38-fp8-sc-auto-*`.
  PROVISIONAL — NOT FOR PUBLIC HEADLINE.

## Session entries — 2026-08-24

- **Qwen3.8-27B Q4_K_M dual-B70 layer-split vs single-GPU (llama.cpp SYCL)** — screens n=3 +
  finalist n=5 + HTTP cold-input/real cells; layer-split decode 26.03 ± 0.01 t/s vs control
  25.78 ± 0.01 (+1.0%), long cold prefill −1.7%; row-split SIGSEGV (negative diagnostic).
  Evidence: `results/qw38-dualcard-20260824T123000Z/` (`../results/qw38-dualcard-20260824T123000Z/MANIFEST.md`) ·
  Analysis: `research/dual-b70-stack-research-20260824.md` (`../research/dual-b70-stack-research-20260824.md`).
  PROVISIONAL — NOT FOR PUBLIC HEADLINE.
- **vLLM official FP8 on B70** — native load proven (`XPUFp8BlockScaledMMKernel`, 28.51 GiB);
  single-card capacity-blocked at 16K ctx; PP=2 blocked by host-RAM staging (watchdog-guarded,
  no kernel OOM). Evidence: `results/qw38-fp8-20260824T130000Z/`.
- **Cascadia two-stage pipeline across both B70s (OpenVINO ov-runtime)** — Qwen2.5-1.5B INT4
  shards on GPU.0/GPU.1; end-to-end serving verified through OpenAI API; e2e median 70.0 t/s
  vs 85.8 single-stage control (−18 % small-model pipeline tax). Doubles as the OpenVINO
  Battlemage datapoint. Evidence: `results/cascadia-dualgpu-20260824T130000Z/` +
  `research/dual-b70-stack-research-20260824.md` §6. PROVISIONAL.
