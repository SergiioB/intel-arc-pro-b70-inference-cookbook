---
title: "Upstream filings — dual-B70 host-RAM and oneCCL findings (2026-08-27)"
type: research
date: 2026-08-27
status: final
---

# Upstream filings — dual-B70 host-RAM and oneCCL findings (2026-08-27)

> **FILED 2026-08-28** (user-authorized):
> - Filing 1 → https://github.com/intel/compute-runtime/issues/986
> - Filing 2 → https://github.com/uxlfoundation/oneCCL/issues/217
> - Filing 3 (vLLM PR) — deferred until compute-runtime answers; see bottom.
>
> Evidence:
> `results/2026/08/b70-pp2-hostram-ident-20260827T132351Z/` (probes), `results/qw38-fp8-pp2-affval*/`
> (vLLM runs), `scripts/tmp/xccl_mask_minirepro.py`, `scripts/tmp/pp2-hostram-ident.sh`.

## Filing 1 — intel/compute-runtime issue

**Title:** Multi-device visibility (ZE_AFFINITY_MASK=0,1) costs ~1 GiB host RAM per GiB of
device memory allocated (driver-side, invisible to meminfo buckets)

**Body:**

### Summary

On a system with two Arc Pro B70 (xe driver, kernel 7.0, Level Zero runtime
`libze_intel_gpu.so.1.15.39122` / intel-opencl-icd `26.27.39122.11`), any process in which
Level Zero can see **both** GPUs (`ZE_AFFINITY_MASK=0,1`) consumes approximately **1 GiB of
host RAM per 1 GiB of device memory it allocates**, on either GPU. With a single visible
device (`ZE_AFFINITY_MASK=0` or `1`) the same allocations cost almost no host RAM. The host
memory is not attributable to any standard accounting bucket and is released only when the
process's DRM contexts are torn down.

This makes multi-GPU serving on modest-RAM hosts fail: loading a 29 GiB FP8 model across two
32 GB B70s through two worker processes (each seeing both GPUs, as vLLM configures) exhausts
30 GiB of host RAM, while the identical load with per-process single-device visibility
succeeds with ~2 GiB of host overhead.

### Reproducer (60 s, no ML framework state, no files read)

```bash
docker run --rm --device /dev/dri -v /dev/dri:/dev/dri:ro \
  --entrypoint python3 <any torch-xpu image> - <<'EOF'
import os, time
# container env: ZE_AFFINITY_MASK selects visibility
import torch
assert torch.xpu.device_count() >= 1
ts = []
for i in range(10):                       # 10 GiB of device memory on xpu:0
    t = torch.empty(1 << 30, dtype=torch.uint8, device="xpu:0")
    t.zero_()                             # commit pages on device
    ts.append(t)
print("HOLDING")                          # inspect /proc/meminfo on the host NOW
time.sleep(60)
EOF
```

Watch host `/proc/meminfo` `MemAvailable` while the process holds the tensors.

### Measured matrix (host MemAvailable delta while tensors held)

| Process visibility | Device memory held | Host RAM cost | Ratio |
|---|---:|---:|---:|
| `ZE_AFFINITY_MASK=0` (1 GPU visible) | 5.6 GiB (GPU0) | 0.77 GiB | ~0.14 (process overhead) |
| `ZE_AFFINITY_MASK=0,1` | 11.0 GiB (GPU0) | 10.97 GiB | **1.00** |
| `ZE_AFFINITY_MASK=0,1` (+`ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`) | 11.0 GiB (GPU0) | 10.97 GiB | **1.00** |
| 2 processes, masks `0` and `1` | 10.5 GiB (both GPUs) | 0.82 GiB | ~0.08 |

All allocations landed on GPU 0 in the affected runs (GPU 1 nearly untouched), so the cost
tracks allocations, not per-device context creation.

### Forensics (while 10.9 GiB VRAM held under mask 0,1)

- Worker RSS (VmHWM): ≤ 2.6 GiB — the host memory is not process anonymous memory.
- `AnonPages` 121 MiB, `Cached` 80 MiB, `Slab` 0.83 GiB, `Shmem` 4 MiB, `VmallocUsed`
  144 MiB, `HugePages_Total` 0, `Unevictable`/`Mlocked`/`VmPin` ≈ 0 — **no visible bucket
  accounts for the ~10 GiB**; `/proc/buddyinfo` confirms the pages are genuinely allocated.
- Fully recovered after the container exits (DRM context teardown).
- Reproduces with zero file I/O and with the allocations never copied from host buffers
  (`t.zero_()` only), so this is not page cache, not safetensors staging, not pinned
  transfer buffers.

### Questions

1. What kernel/driver structure scales ~1:1 with `zeDriverAllocDeviceMemory` when the
   process sees more than one device? (Candidates we could not confirm: cross-device
   sharing backings, SVM/import tables.)
2. Is there a supported way to disable it (env or allocation flag)?
3. If intended, can it be documented — currently it makes multi-GPU XPU processes look like
   a host-memory leak.

Environment: 2 × Intel Arc Pro B70 (`8086:e223`), `xe` KMD, kernel 7.0.0, Level Zero
`1.15.39122`, containers via docker `--device /dev/dri` + read-only `/dev/dri` bind mount,
AMD-IOMMU enabled, resizable BAR active (32 GB aperture per card).

## Filing 2 — intel/oneccl issue

**Title:** Disjoint per-rank ZE_AFFINITY_MASK deadlocks ZE IPC exchange (drmfd path);
`CCL_ZE_IPC_EXCHANGE=none` not honored

**Body:**

### Summary

When two ranks on one node are each restricted to a single, different GPU via
`ZE_AFFINITY_MASK` (rank0 → `0`, rank1 → `1`), oneCCL initialization hangs indefinitely in
the ZE IPC/fd exchange. The pidfd exchange path is reported unsupported under Docker's
default seccomp profile (`pidfd is not supported, fallbacks to drmfd exchange mode`), and
the drmfd fallback never completes. `CCL_ZE_IPC_EXCHANGE=none` is acknowledged by config
(`value of CCL_ZE_IPC_EXCHANGE changed to be none (default:pidfd)`) but the drmfd fallback
warning still appears and the hang persists — `none` appears not to be honored on this path.

With `--cap-add SYS_PTRACE` (or `--security-opt seccomp=unconfined`), the pidfd path works
and everything completes normally — so the exchange itself is functional; the failure is
the silent drmfd fallback.

Additionally, without a `/dev/dri` bind mount into the container (`--device /dev/dri` only,
so `/dev/dri/by-path` is absent), the exchange fails fast instead with
`oneCCL: ze_fd_manager.cpp:144 init_device_fds: EXCEPTION: opendir failed: could not open
device directory`.

### Minimal reproducer (~60 s)

```python
# two processes on one node: RANK=0 ZE_AFFINITY_MASK=0 python m.py & RANK=1 ZE_AFFINITY_MASK=1 python m.py &
import os, datetime, torch, torch.distributed as dist
rank = int(os.environ["RANK"])
torch.xpu.set_device(0)
dist.init_process_group("xccl", init_method="env://", world_size=2, rank=rank,
                        timeout=datetime.timedelta(seconds=60))
t = torch.zeros(8, device="xpu:0") + rank + 1
dist.all_reduce(t)   # hangs here without SYS_PTRACE; completes with it
dist.destroy_process_group()
```

### Observations (oneCCL as shipped in vllm/vllm-openai-xpu image, level-zero 1.15.39122)

- Default container caps: `CCL_WARN ... pidfd is not supported, fallbacks to drmfd exchange
  mode` then silence; all_reduce never completes (verified >15 min).
- `--cap-add SYS_PTRACE`: all_reduce completes correctly.
- `CCL_ZE_IPC_EXCHANGE=none`: warning `value ... changed to be none (default:pidfd)`
  printed, drmfd fallback warning still printed, hang persists.
- Without `/dev/dri` bind mount: immediate `ze_fd_manager.cpp:144 init_device_fds:
  opendir failed` exception instead of a hang.

### Requests

1. Honor `CCL_ZE_IPC_EXCHANGE=none` (skip the exchange entirely — hosts without P2P lose
   nothing), or
2. fail fast with a clear error when the drmfd exchange cannot complete, instead of hanging,
3. document the `SYS_PTRACE`/seccomp requirement for the pidfd path in containers, and the
   requirement for a `/dev/dri/by-path` bind mount for the drmfd path.

## Filing 3 — vLLM PR (DEFERRED, not filed today)

Per-worker `ZE_AFFINITY_MASK` pinning for XPU workers (spawn-time env + `local_rank=0`
remap + `local_world_size` assert exemption), opt-in via env. Deferred until Filing 1 is
answered: if the driver-side host-RAM behavior is fixed or made disableable upstream, the
vLLM-level workaround is unnecessary. Local patch of record:
`scripts/tmp/patch_vllm_worker_affinity.py` (+ `scripts/tmp/qw38-fp8-pp2-affinity-launch.sh`
docker flags: `--cap-add SYS_PTRACE`).
