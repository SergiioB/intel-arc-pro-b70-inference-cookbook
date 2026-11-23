# Patch Stack: Hy4-Flash on Strata/XeStrata SYCL, two Arc Pro B70s

- **Target engine base:** `Niko1221/Strata` release `v0.2.20` (`strata-hy4-sycl` 0.1.66, upstream commit `a34979fc94f3d126b9bb8210f4ad311345688b52`).
- **Model packs (not patched):** `strata-engine/Hy4-Flash-Strata` @ `995bac227ab7e0d221772be641e2b5954a9fac06` — `Hy4-Flash-SQ-64.strata` + `Hy4-Flash-Engram-Q8.strata` (hashes in [the recipe](../../docs/hy4-flash/HY4-FLASH-STRATA-SYCL.md#1-pinned-stack)).
- **Patch:** `strata-hy4-b70-opt.patch` — git diff against `v0.2.20`, 8 files under `engine/src/hy4flash/sycl/`, ~617 insertions / ~59 deletions. SHA-256 in `SHA256SUMS` (`1f55fecc…`).

## What the patch changes

1. **Prefill page-capacity fix (`sy_peer.h`, `ensure_pf_capacity`).** `alloc_p = max(P + 128, P + P/4 + 128)` and `pf_cap_p = alloc_p - 64`: 64 guaranteed-mapped tail rows absorb the 8-row `joint_matrix_load` overread (up to 7 rows past `ntok` when `ntok % 8 != 0`). This is the root-cause fix for the `ccs` engine page fault (`Faulted Address 0x…aa0000` family) that wedged the grouped-prefill path at 28K in round 1. **Standalone correctness fix — safe to upstream unconditionally.**
2. **Clean-exit drain (`sy_serve.h`, `sy_peer.h`).** Before `hy4s_exit_fast(0)`, the process now drains `Q()`, `GpuExperts::wait_all()`, `vcache.wait_copy()` (new method), and `peers.release()` (drains every peer queue and calls `ext_oneapi_disable_peer_access` for enabled mappings). `probe_p2p` disables peer access on a failed/exceptional probe instead of leaving a dangling cross-context mapping; `SyPeerExperts::release()` drains `Q()` first. Eliminates the `xe` `bcs` engine resets and the following-launch hang seen on stock v0.2.20 exits.
3. **`--peer-vram-experts-gb` (`sy_main.cpp`, `sy_serve.h`, `sy_graph.h`).** Asymmetric peer expert budget. `-1` (default) falls back to `--vram-experts-gb` — v0.2.20 behavior bit-for-bit. `HY4S_PEER_VRAM_EXPERTS_GB` env is also honored. Adds `--help` output (none existed).
4. **Grouped XMX SG16 peer prefill (`sy_peer.h`, `sy_gpuexp.h`/`.cpp`, `sy_common.h`).** New `gx_eval_grouped()` exports the existing gather/GEMM1+SwiGLU/GEMM2/reduce kernels for an arbitrary queue. `SyPeerExperts::launch_prefill_grouped` plans pairs/items on the host (same layout as `GpuExperts::run`), launches on each peer's in-order queue **before** `gx->run`, and `finish_prefill_grouped` folds the `T*1536` partial sums into `zout_f` after it. Each peer expert weight is read once across all T tokens; no 256-token tile syncs; no host-RAM weight streaming; P2P or pinned-host bounce for input/output copies. `HY4S_PEER_PF_LEGACY=1` restores the exact stock path. Non-XMX peers still get the grouped evaluator (SIMT variant). Scratch grows lazily; OOM falls back to legacy tiles. `dev_caps()` is refactored into `probe_dev_caps(device)` so peer capabilities are probed per device.
5. **SG16 peer decode (`sy_vram_cache.h`, `sy_peer.h`).** The five `vcache_detail` kernels take a trailing `sycl::queue&` (existing callers unchanged). `SyPeerExperts::plan_decode` builds dedup `VramExpertGroup`s identical to the primary cache; `launch_decode` runs `k_quant_q8_1_g64` + `k_vcache_gemm{1,2}_{q8,fp}<NT>` + the weighted reduce on the peer queue — same math as the primary GPU path. `HY4S_PEER_DEC_SG32=1` restores legacy SG32; non-SG16 peers keep it automatically.
6. **`HY4S_STEP_TIMING=1` (`sy_graph.h`).** Per-layer stderr line `gx_ms peer_pf_ms vcache_ms cpu_exp_ms peer_dec_ms`. Adds explicit drains when enabled — diagnostics only, off by default, zero impact unset.

## Apply and build (GPU-free)

```bash
git clone https://github.com/Niko1221/Strata.git strata-v0220
cd strata-v0220
git checkout a34979fc94f3d126b9bb8210f4ad311345688b52   # v0.2.20
git apply /path/to/strata-hy4-b70-opt.patch             # or: git am < the patch
bash ./engine/src/hy4flash/sycl/build_hy4_sycl.sh bmg   # sources oneAPI setvars itself
```

The `bmg` target compiles with `icpx -O3 -std=c++17 -fiopenmp -fsycl -fsycl-targets=intel_gpu_bmg_g21,intel_gpu_bmg_g31,spir64 -DHY4S_DISABLE_SG8 -DHY4S_ENABLE_SG16_XMX` plus oneMKL, producing `build-hy4-sycl-bmg/strata-hy4-sycl` (~19 MiB). Verified with Intel oneAPI `icpx 2026.1.1` + oneMKL 2026.1; the R2 measured binary has SHA-256 `6d85f1adaf7174b8e4bfcb0bb8c98fdf5c95723d8df4cde4cbaa1035e8c05e1c`.

Two environment notes:

- **`ocloc`:** only needed if you separately build AOT OpenCL kernels — `build_hy4_sycl.sh` itself does not invoke it. The oneAPI compiler packages do not ship it; install `intel-ocloc` from the compute-runtime release matching your Level Zero driver.
- **`libumf.so.1`:** the SYCL Level Zero adapter fails to enumerate devices without the UMF runtime library. Ensure `<oneapi>/umf/<ver>/lib` (plus `compiler/latest/lib` and `mkl/latest/lib`) is on `LD_LIBRARY_PATH` at runtime — the launcher in [`benchmarks/hy4-flash/`](../../benchmarks/hy4-flash/launch-hy4-flash-strata-sycl.sh) does this for you when `ONEAPI_ROOT` is set.

A clean apply + compile proves **source compatibility only**. It does not prove runtime correctness, stability, or speed — those are the measured receipt in [`results/hy4-flash/strata-sycl-20261123/`](../../results/hy4-flash/strata-sycl-20261123/README.md).

## Runtime opt-outs and diagnostics (same binary, env-selected)

```bash
HY4S_PEER_PF_LEGACY=1 HY4S_PEER_DEC_SG32=1  # exact v0.2.20 legacy paths (control arm)
HY4S_STEP_TIMING=1                        # per-layer phase timing to stderr (diagnostics)
HY4S_SYNCALL=1                            # sync-everything probe (diagnostics; does NOT fix nondeterminism)
HY4S_PEER_PINNED=1                        # force pinned-host staging (no-op where P2P probe never passes)
HY4S_XMX_I8=1  /  STRATA_CPUEXP_ACT=q8    # measured dead levers — see DO-NOT-REPEAT DNR-043/044
```

Suggested upstream split when proposing this work: (1) `pf_cap_p` overread fix alone, (2) grouped XMX prefill + SG16 decode, (3) `--peer-vram-experts-gb`, (4) `HY4S_STEP_TIMING` instrumentation — or one PR with (1) as the lead commit.
