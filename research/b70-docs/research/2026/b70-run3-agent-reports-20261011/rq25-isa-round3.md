---
title: "Run-3 subagent report: rq25-isa-round3 (2026-10-11)"
type: research
date: 2026-10-11
status: provisional
---

# RQ-25 — ISA round 3: large-GRF arms (256 GRF), 0-spill verdicts, numerics proof, bench prep

Date 2026-10-11. Compile-only (docker `strata-sycl-dev:latest`, `--memory=3g --cpus=2`, **no** `/dev/dri`); no GPU touched this round. Package root: `scripts/tmp/rq25_soa_20261011/`. This round's sources are byte-identical to round 2 (verified: `build_offline_r3.sh:22` "Arms produced (sources UNTOUCHED from round 2)").

## 1. The large-GRF option the toolchain honours

The working flag is **`-cl-intel-256-GRF-per-thread`**, passed (a) to ocloc as `-options` on `-spirv_input` for the ISA arms, and (b) to icpx's AOT backend as `-Xsycl-target-backend "-device bmg -options \"…\""` for the bench executable (`build_offline_r3.sh:9-20,47,64-75,92-95`). Dead ends probed and recorded in the same header: `-cl-intel-256-GRF` (silently ignored on the SPIR-V path), IGC-internal `-forceTotalGRFNum` / `-ForceLargeGRFNum4RQ` / `-VISAGRFBumpUpNumber` / `-TotalGRFNum` (accepted, ineffective), and there is **no** SYCL header attribute for GRF in this oneAPI (`grep -ri grf include/` = 0 hits). Toolchain: icpx 2026.1.0, ocloc 26.35.39758.10 (`ISA_ARTIFACTS_R3.json` "toolchain").

## 2. Per-arm ISA table (round 3)

All dumps live under `scripts/tmp/rq25_soa_20261011/isa/r3/`. "gu" = the `GuLaunch<E,8>` gate/up kernel (the timed kernel); line numbers are from the per-kernel asm dumps `kernels/<arm>/gu_kernel.asm` (re-grep-verified today by this writer, host grep).

| arm | GRF / SIMD | ocloc spill report | ISA spill traffic (`a32.ss` sites) | q-plane load widths (gu) | dump paths |
|---|---|---|---|---|---|
| **AoS-128** (incumbent control) | 128 / 32 | "SIMD32 allocated 128 regs and spilled around 4" (`rq25_current_128.ocloc.log`, sole warning line) | 13 sites — lines 77 78 88 96 117 1228 1364 1374 1388 2186 2239 2272 2274 | 0× d32x4 (16 B); 3× `load.ugm.d32x2` (8 B) at lines 171 1445 2341; 1× d32x4t (line 113); 2× d32x2t; 10× d16u32 (2-B d-plane scales) | `isa/r3/rq25_current_128.{disasm,ze_info,ocloc.log}`, `isa/r3/kernels/rq25_current_128/gu_kernel.asm` |
| **AoS-256** | 256 / 32 (`rq25_current_256.ze_info` `grf_count: 256`, `simd_size: 32`) | no spill line in `rq25_current_256.ocloc.log` → 0 spills | **0** `a32.ss` sites | unchanged widths: 3× d32x2 at 158 1411 2277; 1× d32x4t (97); 2× d32x2t; 10× d16u32 | `isa/r3/rq25_current_256.{disasm,ze_info,ocloc.log}`, `isa/r3/kernels/rq25_current_256/gu_kernel.asm` |
| **SoA-256** | 256 / 32 (`rq25_soa_256.ze_info:9,15`) | no spill line in `rq25_soa_256.ocloc.log` → 0 spills | **0** `a32.ss` sites | **3× `load.ugm.d32x4.a64`** (16 B q-plane) at lines 110? no — verified sites **145, 1346, 2163** (line 110 is the d32x4**t** variant); 0× d32x2; 1× d32x4t (110); 2× d32x2t; 10× d16u32 | `isa/r3/rq25_soa_256.{disasm,ze_info,ocloc.log}`, `isa/r3/kernels/rq25_soa_256/gu_kernel.asm` |

Example 16-B load (SoA-256 gu, line 145): `send.ugm (32|M0) r126 r4 null:0 0x0 0x08803580 {I@1,$8} // wr:4+0, rd:8; load.ugm.d32x4.a64`.

Round-2 context (why round 3 existed): the r2 SoA arm reached d32x4 but spilled ("around 8", 6 stores + 10 fills) and the r2 incumbent spilled ("around 7"); the 0-spill ISA precondition therefore blocked timing (`research/2026/b70-run2-workflow-report-20261011.md:53-57`). Note the r3 dev-image rebuild of the incumbent spills "around 4" (vs ~7 on the r2 host-built icpx 2026.1.1) — the incumbent still spills under every compiler tried, which is itself part of the A/B story.

**0-spill precondition verdict: HOLDS for SoA-256 (0 spills, and 16-B q-plane loads retained). HOLDS for AoS-256 (0 spills). Does NOT hold for AoS-128 (control, spills ~4).** No live-range reduction was needed — the GRF bump alone removed every spill site.

**Occupancy consequence of 256 GRF:** grf_count 256 doubles the per-thread register file reservation, so resident threads per EU halve versus the 128-GRF arms (Xe2 EU register file is a fixed per-EU budget; the `grf_count: 256` in each arm's `ze_info` is the compiler-confirmed setting). With a memory-bound gu kernel at E≤6 columns this trades occupancy for spill elimination — exactly what the three-arm timing is designed to measure.

### Executable embedded-image verification (bench arms are real)

`rq25_bench_g256` (AOT, both TUs linked per `build_offline_r3.sh:92-95`) embeds **both** gu kernels at 256 GRF. Verified today on host by `clang-offload-extract` + `ocloc disasm` (extracted images archived at `isa/r3/exe_unbundle/`):
- `im.0` → `GuLaunch<1,8>` (SoA): `grf_count: 256`, 0 `a32.ss`, 3× d32x4 — `exe_unbundle/embedded_soa_gu.asm`.
- `im.2` → `GuLaunch<0,8>` (AoS): `grf_count: 256`, 0 `a32.ss`, 3× d32x2 — `exe_unbundle/embedded_aos_gu.asm`.
- Combined `grf_count` census: `exe_unbundle/embedded.ze_info` (2× `grf_count: 256`).

## 3. Numerics proof (CPU, offline, real expert tensors)

Round 3 does not change any arithmetic (sources untouched), so the round-2 proof carries; its artifacts re-verified present today:

- Ring built from **real expert gate/up tensors** of the pack (`native_experts.txt`, layers 35/36/41/44/45/47, 427 slots, 546,560 rows, 510.8 MiB/layout) at `/mnt/<nvme-disk>/b70-run2-20261011/rq25/ring-20261011/` (`README.md:117-129`).
- `cpu_selftest.json` (read today): `align_violations: 0`, `roundtrip_row_mismatches: 0` (SoA ring is a pure byte permutation of the GGUF bytes), **`prep_sites_checked: 43,724,800`, `prep_word_mismatches: 0`**, `pass: true`. The per-(block,part) decode words `MultiW` fully determine the dequantized values and the float pipeline after them is shared code (`README.md:157-163`), so **SoA dequantized values are bitwise equal to AoS dequantized values** on real expert tensors.
- The `words` cell re-proves this on device before any timing (now run on BOTH executables — see run_bench.sh).

## 4. Prepared GPU bench (prepared, NOT run)

Driver: `scripts/tmp/rq25_soa_20261011/run_bench.sh` (updated this round; `bash -n` clean). Contract now implemented:

- **Three arms**: `aos128` = incumbent via `offline-build/rq25_bench` (128 GRF); `aos256` + `soa256` via `isa-devimg-r3/rq25_bench_g256` (both kernels 256 GRF, embedded-image-verified above). Existence-checked at start (`run_bench.sh:42-43`).
- **Preflight refusing contaminated GPU1**: `gpu_guard_20261011.sh` (fails on any non-desktop process holding `/dev/dri/renderD129`) runs before any cell and before/after EVERY cell (`run_bench.sh:102-103, 78, 90`); exit 1 aborts without killing.
- **GPU1 only**: `export ONEAPI_DEVICE_SELECTOR=level_zero:1` (`run_bench.sh:36-37`) plus the binary's own positive PCI match on 0000:0f:00.0 (`src/rq25_bench.cpp`, `README.md:186-187`).
- **Allocation budget**: live `visible_avail` read per cell from `/sys/kernel/debug/dri/0000:0f:00.0/tile0/vram_mm` via `sudo -n` (`run_bench.sh:48,51-55,80-81`); bench refuses allocations > visible_avail − 4 GiB; unreadable budget aborts the cell.
- **Protocol**: one discarded warmup per (E, arm), then **3 interleaved triples** (aos128, aos256, soa256) per shape E=1..6 (`run_bench.sh:116-125`); ring 510.8 MiB > L2 asserted at runtime; raw per-cell JSON + `cells.tsv`; energy1_input and vram snapshots per cell.
- **f64 reference gate**: `cpu_check --gate-dots` after all cells, tolerance ≤ 5e-3 relative (declared, `README.md:164-169`).

Exact command (execution step, after the registry authorizes and GPU1 is free):

```bash
cd ~/B70-DOCS
bash scripts/tmp/rq25_soa_20261011/run_bench.sh \
    /mnt/<nvme-disk>/b70-run2-20261011/rq25/ring-20261011 \
    /mnt/<nvme-disk>/b70-run2-20261011/rq25/cells-r3 3
```

(`RQ25_BUILD` / `RQ25_BUILD256` override the two binaries; `3` = interleaved triples per shape, the default.)

## 5. Files this round added/changed

- `isa/r3/` (from the killed attempt, re-verified): three arms' disasm/ze_info/ocloc logs, per-kernel asm, `census_r3.txt`, `ISA_ARTIFACTS_R3.json`, `SHA256SUMS_R3`, `rq25_bench_g256`.
- NEW `isa/r3/exe_unbundle/` — embedded images `im.0`/`im.2`, `embedded_soa_gu.asm`, `embedded_aos_gu.asm`, `embedded.ze_info` (host-side extraction, this round).
- UPDATED `run_bench.sh` — 3-arm protocol, preflight guard, `ONEAPI_DEVICE_SELECTOR`, per-cell budget hard-fail, PAIRS default 3.

Everything above is compile-only/CPU; no GPU cell was run; the 1.5× GB/s registry gate remains unmeasured by design.
