---
title: "Night prep — unitrace/pti-gpu build on nvme + root disk reclaim (CPU-only)"
type: worklog
date: 2026-09-25
status: final
---

# Night prep 2026-09-25 — unitrace build + root disk reclaim

Prep for tonight's 22:00 GPU research night (Phase 4, O13 tooling). All work
CPU-only: no model loads, no servers, no kernels run. The only GPU-adjacent
action was Level Zero *enumeration* (`sycl-ls`) under the tracer as the binary
verification — zero VRAM allocation, zero compute.

## 1. unitrace / pti-gpu build — COMPLETE

### Binary

- Launcher: `/mnt/<nvme-disk>/builds/pti-gpu-build/install/bin/unitrace`
  - sha256 `c771fafc41eb57897861aac6165ed9b8c4af9be8f4bd57abe544bf72ac9d235d`
- Tracer library (the actual LD_PRELOAD payload):
  `/mnt/<nvme-disk>/builds/pti-gpu-build/install/lib/libunitrace_tool.so`
  - sha256 `2938ba5e64014ad83313481eb085157f53db0a0a1271c0f1a439a7e5c18975f3`
- Build log: `/mnt/<nvme-disk>/builds/pti-gpu-build/build.log`
  (configure -> build -> install, finished `2026-09-25T17:59:44Z BUILD COMPLETE OK`)
- Build script (rerunnable): `/mnt/<nvme-disk>/builds/pti-gpu-build/do-build.sh`
- Smoke evidence: `/mnt/<nvme-disk>/builds/pti-gpu-build/smoke/sycl-ls.1439935.json`

### Versions and configuration

| Component | Value |
|---|---|
| pti-gpu | 0.49.28, git `8f063593943dc7f97b068a110843a9e1887008de` (2026-09-10) |
| Build config | `BUILD_WITH_L0=1 OPENCL=1 ITT=1 XPTI=1 MPI=0 OMP=1 PERFETTO=1` (printed by `--help`) |
| Toolchain | cmake 4.2.3, gcc 15.2.0, in-build Python 3.12.13 |
| System L0 | loader `libze_loader.so.1.28.2` at `/usr/lib/x86_64-linux-gnu/` |
| Runtime seen | NEO 26.05.037020, UR adapter 20.2.0 (from `sycl-ls`) |

Notes: MPI tracing deliberately disabled (vLLM does not use MPI; drops the
`unitrace_mpi` shim and the find_package(MPI) dependency). Perfetto protobuf
output ON (perfetto_trace.pb.cc built; JSON remains the default format).
Everything built out-of-tree on nvme; the root filesystem received zero build
bytes (total build footprint 572 MB on nvme).

### Runtime environment requirement

The launcher is self-contained, but `libunitrace_tool.so` needs
`libxptifw.so`, which is only in the oneAPI compiler dir. Either source
oneAPI:

```bash
source /opt/intel/oneapi/setvars.sh --force
```

or set the minimal path (verified working):

```bash
export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2026.1/lib
```

### Verification performed (no workload)

1. `unitrace --help` — runs, prints the build-config banner above. RC=0.
2. `unitrace --chrome-call-logging --device-timeline /bin/true` — clean exit,
   "No event of interest" (expected for a no-op).
3. `unitrace --chrome-call-logging sycl-ls` — 81 traceEvents, 80 Level Zero
   host API calls across 14 distinct `ze*` APIs including `zeInitDrivers` (5),
   `zeDriverGet` (8), `zeDeviceGet` (10), `zeDeviceGetProperties` (9).
   Device enumeration traced; both B70s enumerate normally.

### Invocation for tonight — vLLM tracing

Host-launcher mode (verified pattern; wraps any host command):

```bash
source /opt/intel/oneapi/setvars.sh --force
/mnt/<nvme-disk>/builds/pti-gpu-build/install/bin/unitrace \
  --chrome-call-logging --chrome-kernel-logging --device-timeline \
  -o /mnt/<nvme-disk>/traces/vllm-phase4.json \
  -- <command and args>
```

Container mode for the pinned vLLM image (NOT yet validated — first thing to
check tonight; see "What remains"):

```bash
mkdir -p /mnt/<nvme-disk>/traces
docker run --rm --device /dev/dri \
  -v /mnt/<nvme-disk>/builds/pti-gpu-build/install:/unitrace:ro \
  -v /opt/intel/oneapi/compiler/2026.1/lib:/xptifw:ro \
  -e LD_LIBRARY_PATH=/xptifw \
  -e LD_PRELOAD=/unitrace/lib/libunitrace_tool.so \
  -e UNITRACE_ChromeCallLogging=1 \
  -e UNITRACE_ChromeKernelLogging=1 \
  -e UNITRACE_DeviceTimeline=1 \
  -e UNITRACE_TraceOutputDirPath=1 -e UNITRACE_TraceOutputDir=/traces \
  -v /mnt/<nvme-disk>/traces:/traces \
  vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f \
  <entrypoint args>
```

Known risks for container mode (in order):

1. **libstdc++ ABI**: the tool .so was linked by host gcc 15; if the vLLM
   container's libstdc++ is older, preload fails with `GLIBCXX_x.y not found`.
   Fix: prepend host `/usr/lib/x86_64-linux-gnu/libstdc++.so.6` to
   `LD_PRELOAD`, or rebuild unitrace inside a container matching the image.
2. `libxptifw.so` mount is mandatory (handled above via `/xptifw`).
3. If only kernel-level timing is needed, `UNITRACE_ChromeKernelLogging=1`
   alone is the lower-overhead start; add call logging after it works.

### What remains (smallest steps)

- Validate the container-mode LD_PRELOAD line above against the pinned image
  (risk #1 is the only likely blocker). Everything else is done.
- The original root-owned source snapshot at `~/pti-gpu-build`
  (32 MB) was left untouched; the build lives entirely on nvme at
  `/mnt/<nvme-disk>/builds/pti-gpu-build/` (source copy + build
  dir + install prefix + log). Delete the /home copy only with explicit
  authorization.

## 2. Root disk safe reclaim — COMPLETE (target met: 2.3 GiB -> 19.8 GiB free)

### Before / after

| | total | used | free |
|---|---:|---:|---:|
| BEFORE (2026-09-25T17:53Z) | 228,804,210,688 | 223,956,992,000 | 2,494,361,600 (2.32 GiB, 99%) |
| AFTER (2026-09-25T20:04Z) | 228,804,210,688 | 205,275,209,472* | 21,211,607,040 (19.8 GiB, 91%) |

Freed total: ~18.7 GB. (*intermediate reading; df -h after: 214G size, 20G avail.)

### Exactly what was freed

| Category | Command / action | Freed |
|---|---|---:|
| `~/.cache/cascadia/ov-cache` (OpenVINO/oneDNN kernel+model compile cache; 136 hash-named `.blob`/`.cl_cache` files; no process held them open; rebuild = one-time recompile on next Cascadia import) | `rm -rf` | 16,323,643,974 B |
| `~/.cache/pip` | `pip cache purge` (662 files) | 1,106.4 MB |
| `~/.cache/neo_compiler_cache` (NEO compiler cache) | `rm -rf` | 440,943,464 B |
| systemd journal | `journalctl --vacuum-size=200M` | 414.7 MB |
| apt lists + archives | `apt-get clean` + `rm -rf /var/lib/apt/lists/*` | ~280 MB |
| `~/.cache/uv` | `uv cache clean` | ~116 MB |
| docker build cache | `docker builder prune -af` | 0 B (cache was empty) |
| dangling images | none removed — see below | 0 B |
| `/tmp` >7 days | nothing found; `/tmp` is tmpfs (RAM), 0 disk impact | 0 B |

### Protections verified

- Pinned image intact after cleanup:
  `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
  — it is the ONLY dangling/untagged image on the host (tag stripped when a
  newer `:latest` was pulled), so per instructions NO image was removed. All
  5 images survive: the pinned digest, `b70-vllm-qkrope53989:f01e24f6`,
  `b70-vllm-kernels-0.1.15.4:f01e24f6`, `vllm/vllm-openai-xpu:latest`,
  `alpine:latest`.
- Not touched (outside documented-safe categories, listed for the next
  decision): `/var/lib/snapd` 12 GB; `/var/log` rotated syslog/kern.log ~390 MB
  beyond journal; `~/.cache/{opencode,BraveSoftware,tracker3,huggingface}`
  ~1.9 GB; `~/.cache/huggingface` also contains model-adjacent artifacts and
  was excluded deliberately.

### Largest remaining root consumers (context, no action taken)

`/home` 128 GB (models/trees, off-limits), `/var/lib/containerd` 38 GB (all
five live images), `/var/lib/snapd` 12 GB, `/opt` 19 GB (oneAPI).

## 3. Gates on exit

- No inference process or container was started at any point (CPU-only prep).
- GPU power state untouched; no `b70-gpu1` operations.
- Power caps untouched (no research work performed, 150 W baseline not altered).
- Original `~/pti-gpu-build` source preserved unchanged.
