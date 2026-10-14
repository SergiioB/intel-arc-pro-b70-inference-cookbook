---
title: "PyTorch XPU on B70 — engine dossier"
type: engine-dossier
status: living
date: 2026-09-15
updated: 2026-09-15
engine: torch-xpu
---

# PyTorch XPU — living dossier

Not a serving engine on this host. Role: kernel/patch development and
profiling substrate — root-cause and A/B substrate behind the vLLM XPU
kernel fixes (DFlash2 K-norm #56431, vxk #597/#579, selector-NaN RL-029)
and custom-kernel experiments.

## What it is used for

- Minimal reproducers for vLLM XPU kernel bugs (eager vs `torch.compile`
  A/Bs; see RL-028 dynamic-shape stride assert and the DFlash2 fixes).
- Microbench of Xe2 primitives (DPAS launch bounds, small-M GEMM) feeding
  `research/2026/xe2-esimd-launch-bounds-2026-08.md` and the megakernel
  references (`docs/reference/megakernel-comprehensive-reference.md`).
- Skills: `.agents/skills/torch-xpu-run/`, `torch-xpu-bench/`,
  `torch-xpu-profile/` (unified runtime, VTune/unitrace paths).

## Standing constraints

- oneAPI activation before any SYCL/L0 work (contract §6);
- power telemetry via `xe` hwmon energy counters, never `xpu-smi`;
- any perf claim goes through the benchmark protocol like an engine cell —
  torch microbenchmarks are diagnostic evidence, never headline numbers.
