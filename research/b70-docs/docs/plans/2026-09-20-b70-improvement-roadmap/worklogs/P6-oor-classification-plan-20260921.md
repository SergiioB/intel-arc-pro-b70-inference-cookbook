---
title: "P6 entry step — OV GenAI ≥196K CL_OUT_OF_RESOURCES (-5): classification plan (per-stage allocation capture)"
type: plan
date: 2026-09-21
phase: P6
status: READY-TO-RUN (probe prepared, not run — see §6)
entry_step: "classification with per-stage allocation capture, before any fix attempt"
ledger: RL-TBD-P6
artifact: scripts/tmp/p6-ov-oor-probe-20260921.sh
---

# P6 entry step — classify the OV GenAI ≥196K resource failure

Roadmap step: `docs/plans/2026-09-20-b70-improvement-roadmap/P6-long-context-gates.md` §Engineering work (1)(2); risk rule "Do not classify CL_OUT_OF_RESOURCES as VRAM OOM without evidence" (:128).

## 1. What the recorded failure actually says (on disk, verbatim)

**The failure is recorded on disk** — error text + context lengths:

- `results/2026/09/qw38-ov-mtp-20260914/ctx-sweep-ov-int4-gdn8-u4-max.json`:7 — length **196608**, full protocol (warmup + n=1, `max_new_tokens=128`), INT4-GDN8, no-MTP, u4 KV, `cache_size=0`:
  `RuntimeError('Exception from src/inference/src/cpp/infer_request.cpp:224:\nException from src/plugins/intel_gpu/src/runtime/ocl/ocl_stream.cpp:395:\n[GPU] clFinish, error code: -5 CL_OUT_OF_RESOURCES\n\n')`
- `…/ctx-sweep-ov-int4-gdn8-u4-max.json:10` — length **262144**, same config, from the plugin's error-translation path `ocl_common.hpp:62` ("[GPU] CL_OUT_OF_RESOURCES exception. Due to a driver bug, any subsequent OpenCL API call may cause the application to hang…"). The stored string is truncated at 300 chars by the runner (`scripts/tmp/ov-mtp-20260914/ctx_sweep_ov_int4_u4_max.py`:85); the untruncated text is not on disk.
- `results/2026/09/qw38-ov-mtp-20260914/ov-nomtp-ceiling.json`:29 — **229376**, light 40-token probe, same `-5` class. `ov-u4-maxctx.json` records 196608/229376/262144 *passing* at 40 tokens (5041/4358/3812 MiB free after the runs).
- Earlier `-5` instances (same class, other configs): `ctx-sweep-ov-int4-gdn8.json:159` (65536, no-MTP u8), `ov-ceiling-confirm.json:20` (131072 with MTP5), `kv-u8-draft-f16.json:30` (98304), `ov-int8-draft-kv-test.json:35`.
- Working boundary, same runner + protocol as the failing cell: `ctx-sweep-ov-int4-gdn8-u4-full.json` — 512…**131072 all ok** (decode 15.9 tok/s @131072).
- Filing: `research/2026/ov-genai-mtp-longctx-wall-upstream-draft-20260915.md`:20-21 (environment), `:36` (the `-5` 196608/262144 row), `:72` — **filed as openvino.genai#4483 on 2026-09-15**. Only the draft + URL are on disk; the filed body and thread are upstream-only. Public restatements: `cookbook:docs/qwen38-27b/ENGINE-COMPARISON-CTX-20260914.md:9-10,252`; `docs/reference/engine-support-matrix.md`:44.

**Not on disk** (the probe must produce it): any per-stage allocation record, any DEBUG-level OV/L0 log, and any VRAM reading *at the moment of the 196K failure*. `SUMMARY.md:79-80` asserts "~33.4 GB in use during prefill, 814 MiB free", but no file in the campaign root contains those numbers for that run (nearest provenance is post-load dumps: `baseline-int8.json:8,38`, `mtp-gate/int4-gdn8-mtp5.json:8`). Treat that sentence as **unverified** until re-measured; it is exactly the number that would decide OOM vs not-OOM.

## 2. The three hypotheses and the observable that separates each

**(a) True device OOM** (VRAM exhausted, or no contiguous block big enough). Observable: failure-moment `visible_avail` near 0 and/or largest free `order-N` block < the outstanding request; an allocation-path error (`cannot allocate` / `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`) rather than only `clFinish -5`; the 40-token probe at the same length failing too. Pass for (a): free < ~1 GiB at failure *and* control failure. Fail: plenty free with multi-GiB order blocks → (a) rejected for that attempt.

**(b) Workgroup/private-memory limit inside a kernel.** Observable: `-5` arrives asynchronously at `clFinish` with **no allocation error anywhere**; the last stage is a *kernel execution* of a named layer (OV DEBUG log / profiling trace); the same nominal context passes in the light probe (fewer resident buffers, same prefill shapes) while the heavier request fails. Pass: kernel-stage failure, no allocation signal. Fail: the failing call is an allocation.

**(c) Allocation-size/shape limit or driver/runtime resource cap.** Observable: the requested single buffer exceeds a reported cap (`GPU_DEVICE_MAX_ALLOC_MEM_SIZE` / clinfo `CL_DEVICE_MAX_MEM_ALLOC_SIZE`, captured per attempt) or the ZE validation layer flags an oversized/invalid allocation; failure in the allocation path while free VRAM is large. Pass: cap < attempted size, or explicit oversized-allocation rejection. Fail: no cap approached, no allocation complaint.

Shared discriminator: **(a) vs (b)/(c) is decided by measured free VRAM + free-order map at failure**; **(b) vs (c) by which stage fails** — kernel execution vs allocation — from the OV DEBUG/profiling log plus L0 validation/loader traces.

## 3. Per-stage capture this runtime supports (knobs verified in installed bytes)

Runtime identity, read from `/mnt/<models-disk>/.ov-genai-venv` (not guessed): `libopenvino.so.2650` → **2026.5.0-22967-38a575de45f**; `libopenvino_genai.so.2650` → **2026.5.0.0-3412-8fa12145144**; wheel metadata `openvino 2026.5.0.dev20260901` / `openvino-genai 2026.5.0.0.dev20260901`; Python 3.14 (`pyvenv.cfg`). Same build as the upstream draft (`:20-21`), so the reproducer targets the build that failed.

Exact run line the probe uses per attempt (also written to `attempt-<tag>.cmd` in the run dir):
`cd <run_dir> && ZE_AFFINITY_MASK=1 OPENVINO_LOG_LEVEL=DEBUG ZE_ENABLE_VALIDATION_LAYER=1 ZE_ENABLE_LOADER_DEBUG_TRACE=1 NEO_CACHE_PERSISTENT=0 NEO_CACHE_DIR=/mnt/<sata-disk>/ov-cache-p6-20260921 /mnt/<models-disk>/.ov-genai-venv/bin/python -u probe_ov_oor.py <full|light> <len> > attempt-<tag>.log 2>&1`

- **OV logging** — `OPENVINO_LOG_LEVEL=DEBUG` (literal in the installed `libopenvino_genai.so.2650`) plus in-process `core.set_property(dev, ov.properties.log.level(ov.properties.log.Level.DEBUG))` (`Level` enum in `openvino/_pyopenvino/properties/log.pyi`). Proves: which stage (load / prefill / decode) and which primitive was executing when `-5` arrived.
- **Plugin allocation + stage accounting** — `ov.properties.intel_gpu.memory_statistics` (getter), `device_total_mem_size`, `device_max_alloc_mem_size` (`openvino/_pyopenvino/properties/intel_gpu/__init__.pyi`; key strings `GPU_MEMORY_STATISTICS`, `GPU_DEVICE_TOTAL_MEM_SIZE`, `GPU_DEVICE_MAX_ALLOC_MEM_SIZE` present in the installed GPU plugin). Proves plugin-tracked allocation totals per stage and the single-allocation ceiling → separates (c).
- **Per-layer timings** — `ov.properties.enable_profiling(True)` (`PERF_COUNT` in the plugin). Proves the last layer to run before failure (names the kernel for (b)).
- **Level Zero** — `ZE_ENABLE_VALIDATION_LAYER=1`, `ZE_ENABLE_LOADER_DEBUG_TRACE=1` (loader writes `ze_loader.log`; both strings in `libze_loader.so.1.28.2`), optional `ZE_ENABLE_TRACING_LAYER=1` (`ZET_ENABLE_API_TRACING_EXP` = deprecated spelling). Proves whether the failing L0 call is a memory allocation (→ (a)/(c)) or command-list/kernel execution (→ (b)).
- **Device capability snapshot** — `/usr/bin/clinfo` (installed) → `CL_DEVICE_MAX_MEM_ALLOC_SIZE`, `CL_DEVICE_LOCAL_MEM_SIZE`, `CL_DEVICE_MAX_WORK_GROUP_SIZE`, `CL_DEVICE_MAX_WORK_ITEM_SIZES`, `CL_DEVICE_GLOBAL_MEM_SIZE`. Proves the candidate hard limits (b)/(c) and their values; the `-5` came through the OCL stack (`intel-opencl/libigdrcl.so`, error prefix `ocl_stream.cpp`).
- **Driver cache/debug** — `NEO_CACHE_PERSISTENT=0`, `NEO_CACHE_DIR=<out>`, `NEO_LOCAL_MEMORY_ALLOCATION_MODE` (all three strings in `libze_intel_gpu.so.1.14.37020`; semantics not asserted). The driver's `igdrcl.config` / `neo.config` / `igdrcl_dumped.config` names exist in the installed driver but the lookup path was not established — the probe records a dump if one appears and claims nothing more.
- **Kernel context** — `sudo -n dmesg -T` snapshot before/after each attempt: did `xe` log a fault/reset around the failure?

## 4. VRAM-vs-limit reading plan (authorities = hwmon + xe debugfs, not xpu-smi)

- **Free VRAM**: `/sys/kernel/debug/dri/0000:0f:00.0/tile0/vram_mm`. GPU 1 is `0000:0f:00.0` (`mtp-gate/int4-gdn8-mtp5.json:5-6`: "device": "GPU.1", "pci": "0000:0f:00.0"; hwmon `hwmon6`, `name=xe`). `visible_avail` = free MiB; the `order-N free:` lines give the largest contiguous blocks. The probe takes the **full dump** at gate and per attempt entry, plus a 2 s background sampler **during** each attempt, so the last sample before process exit is the failure-moment reading, stored beside the exact command and timestamps.
- **Temperature/power**: `/sys/bus/pci/devices/0000:0f:00.0/hwmon/hwmon6` (`temp2_label=pkg`, `temp3_label=vram`, `power1_cap`). Gate: `visible_avail ≥ 31000 MiB`, pkg ≤ 65 C, no docker containers, no compute processes — the repo's known-good lane pattern (`SUMMARY.md:50-51`).
- **Contiguity, why the full order map is kept**: on the empty card at 2026-09-21 11:57 the largest free block was `order-22 free: 16384 MiB, blocks: 1` (of 32615 MiB free) — a single allocation > ~16 GiB would fail with plenty of VRAM free. The probe therefore records the whole `order-N` map, not just `visible_avail`, at entry and during the attempt; that is what separates "no VRAM" from "no contiguous block".
- **Sufficiency per attempt**: `entry_visible_avail` + largest free order block are written into the manifest next to the length and previous attempts' peak growth, so "free VRAM at entry was sufficient" is a measured comparison, never an assertion. xpu-smi deliberately unused.

## 5. Prior art

- `registry/research-ledger.json` (31 entries, checked 2026-09-21): **no entry** covers #4483 or this classification. Closest is `RL-015` (Qwen3.6-35B mixed-load) — a different failure.
- `research/2026/ov-genai-mtp-longctx-wall-upstream-draft-20260915.md` — filing draft (issue text `:15-68`), filed `:72`. Names suspects ("dynamic allocation / temporary-memory pressure or a shape-dependent GPU kernel fault", `:48-56`), tests none. Filed issue body is upstream-only.
- Cookbook `cookbook:docs/qwen36-35a3/MOE-INT4-THREE-ENGINE-CEILING-20260917.md:134-136` — closest prior *reading*: for Qwen3.6-35B MOE INT4, "OpenVINO 131K is a kernel-level FAIL, not a KV-cache size problem … CL_OUT_OF_RESOURCES inside a oneDNN prefill primitive … the ceiling is OV's oneDNN prefill allocation path". Different model, no allocation capture, not this artifact.
- Campaign restatements: `SUMMARY.md:74-89`; `cookbook:docs/qwen38-27b/ENGINE-COMPARISON-CTX-20260914.md:252`.

## 6. What blocks running the probe now

- **GPU 1 occupancy is transient**: at 2026-09-21 ~11:47 a root vLLM container (`p4champ`, image `f01e24f6c7ff`, PIDs 211419/212402) held `/dev/dri/renderD129` (`usage: 22979166208` / `visible_avail: 10741 MiB`); by ~11:55 it had exited (`usage: 42000384`, `visible_avail: 32615 MiB`, no docker containers) but the card was still at `pkg 69.00 C` — above the probe's 65 C gate — with `power1_cap=230 W`. The probe gates and cools on its own; the only requirement is not starting it while another lane is running.
- **Root partition is 97% full (8.2 GB free)** and this model's OV compile cache was 25.66 GB before pruning (`results/2026/09/qw38-ov-mtp-20260914/CLEANUP-20260920.md`). The probe points the compile cache off-root (`/mnt/<sata-disk>/ov-cache-p6-20260921`, 178 GB free); default cache location would re-fill `/`.
- Nothing else missing: model (25 GB, `/mnt/<sata-disk>/Qwen3.8-27B-int4-gdn8-ov`, same inode as the campaign's `/mnt/<models-disk>/...`), venv, runner protocol, evidence root all verified.

Run: `bash scripts/tmp/p6-ov-oor-probe-20260921.sh` (header carries OUTPUT/SAFETY).
