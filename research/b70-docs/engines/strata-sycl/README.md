---
title: "Strata SYCL on B70 — engine dossier"
type: engine-dossier
status: living
date: 2026-10-09
updated: 2026-10-09
engine: strata-sycl
---

# Strata SYCL — living dossier

Niko1221/Strata's own inference engine (CUDA original, `sycl/` port for Intel
Arc). **Primary serving route for Qwen3.8-Flash-Next on one B70 since
2026-10-09** (ADR-0011), replacing the llama.cpp AP-Q4_K_XL dual-card split.
Not llama.cpp: it reads GGUF expert weights into its own native pack, keeps a
VRAM expert cache, and streams cache misses from a pinned host mirror.

## Current state (2026-10-09)

- Engine: Strata v0.1.40.3 SYCL (engine string 0.1.41, revision class
  `d5ea713`), Docker image `strata-sycl-dev` on the isolated daemon
  `unix:///run/b70-strata-docker.sock` (data-root on the T-FORCE volume).
- Artifact: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` GGUF (two shards, hashes in
  `results/2026/10/rq14-strata-matched-screen-20261009/ARTIFACTS.sha256`) +
  native pack `iq3xxs` + `mtp-rt`.
- Config of record: `results/2026/10/rq14-strata-matched-screen-20261009/serve-flashnext-iq3xxs.json`
  (`--stream-experts --expert-cache auto --spec 4 --spec-min-p 0.5 --kv int8
  --prefill 4096 --max-context 10240 --vram-reserve-mib 256`).
- Measured (RL-125, n=5, one card): 48.03 tok/s p512/g128 engine decode-wall;
  989.11 tok/s 8K prefill; exact-answer battery 3/3.
- Residency: 14,544 of 24,576 IQ3_XXS experts (1.75 MB each) in a 23.6 GiB VRAM
  cache, the rest in 16.4 GiB pinned host RAM (59 %).

## Verified capabilities / limits

| Capability | State | Evidence |
|---|---|---|
| One-card decode and prefill | 2.22× decode, 5.1–5.3× prefill vs the RL-070 dual-split anchors | RL-125 |
| Second card as expert cache (`--expert-cache-device1`) | Host-staged, 0 expert bytes by DMA; slower (33.44 tok/s, 111.86 tok/s prefill) | RL-125 |
| `--peer-device` (second GPU as expert tier) | Stubbed in the SYCL port (CUDA calls) | Strata INTEL.md capture `research/references/levers-synthesis-20261009/sha256-a394b896…` |
| `--layer-split` across two cards | Untested here. External 2×B60 run held 100 % of experts resident (IQ2_XS 55–61 tok/s) | RQ-18; capture `sha256-ca6f0ea9…` |
| Fused int8 prompt kernels (upstream #136) | Not built in the SYCL port; prefill GEMM runs FP16 through oneMKL | Strata INTEL.md capture; RQ-22 |
| Quality beyond 3 exact cases | Open: representative tasks, long/multi-turn, replay drift | ADR-0011; RQ-17 |

## Hard rules learned

- Discard one same-shape warmup on a cold engine: the first request exits 2
  with `--spec needs the device residency table`.
- Answer-bearing prompts need `gen >= 64`; the model thinks first.
- `STRATA_VERIFY_NO_HOST=1` is required whenever any expert lives outside VRAM;
  without it logits turn NaN on xe (Strata INTEL.md).
- Large arenas go through Strata's alias-checked allocator; the xe aliased-page
  class reproduced here as a GT reset on the display GPU when a 20 GiB arena was
  probed outside the engine (RL-123). Never allocate beyond free − 4 GiB.
- The image entrypoint swallows non-engine commands; pass `--entrypoint`.
- Restore as-found caps at cleanup (180 W / 230 W on 2026-10-09), not 150 W.
- 256 MiB VRAM reserve is below Strata's 1,024 MiB recommendation for ≤32K;
  revisit before longer-context cells.

## Related

Route decision: `decisions/adr-0011-flashnext-strata-one-card-route.md`.
Evidence: `results/2026/10/rq14-strata-matched-screen-20261009/`.
Source math: `research/2026/flashnext-strata-source-math-20261009.md`.
External assessment: `research/2026/flashnext-external-routes-assessment-20261007.md`.
