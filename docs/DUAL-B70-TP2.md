# Dual-B70 Multi-GPU Infrastructure (TP2 / PP2)

This page is the infrastructure and topology authority for serving one vLLM
model across two Intel Arc Pro B70 cards with tensor or pipeline parallelism.
It defines worker isolation, Docker permissions, oneCCL algorithm selection,
verification, and the collective-only repro.

It is **not** a model recipe or benchmark authority. Model checkpoints, image
digests, kernel builds, serve flags, context limits, performance numbers, power
measurements, and evidence belong on model-specific pages and in the generated
benchmark catalog.

- Qwen3.8 family routes: [qwen38-27b/README.md](qwen38-27b/README.md)
- Qwen3.8 FP8 W8A16 TP2 recipe and evidence:
  [qwen38-27b/FP8-TP2-W8A16.md](qwen38-27b/FP8-TP2-W8A16.md)
- Image and model patch compatibility:
  [IMAGE-AND-PATCH-MATRIX.md](IMAGE-AND-PATCH-MATRIX.md)
- Published numeric records: [BENCHMARK-CATALOG.md](BENCHMARK-CATALOG.md)

For two independent single-card servers, do not use TP2 or PP2. Run one engine
per card, with `ZE_AFFINITY_MASK=0` and `ZE_AFFINITY_MASK=1` respectively.

Each independent engine also needs its own compile cache. The cookbook
container layout keeps the vLLM and inductor caches inside each container,
which satisfies this by default. Mounting one shared cache directory into both
servers is the known-bad layout: a community dual-B65 deployment (vLLM 0.29.0
XPU, two TP=1 workers sharing one vLLM cache) reported the second worker
loading the model and then segfaulting while restoring compiled artifacts;
separate `VLLM_CACHE_ROOT` and `TORCHINDUCTOR_CACHE_DIR` per worker fixed it.
Not reproduced on the cookbook dual-B70 hosts; recorded as a community report
that the default per-container layout already avoids.
Per-*engine* cache persistence is safe and valuable: bind-mounting one engine's
cache volume (its own `VLLM_CACHE_ROOT`/inductor dir) into successive
containers of the same model, image, and profile skips the compile step on
restart — on the author's host a 27B-class TP2 cold boot dropped from
~3.5 minutes to ~1 minute. One volume per engine; never shared between
concurrently running engines.

## Infrastructure status

Spawn-time worker affinity is validated on the author's dual-B70 host. The four
oneCCL settings below were also confirmed serving on two additional dual-B70
hosts through cookbook
[issue #8](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/issues/8),
including a host that crashed without them despite using the affinity patch and
matching driver packages. Treat both parts as required infrastructure.

Current model-specific numeric cards supersede the earlier capability-only
state. This page deliberately carries no benchmark values; follow the model
recipe or catalog links above.

## Failure without the infrastructure fix

```text
RuntimeError: oneCCL: ze_call.cpp:28 do_call: EXCEPTION: ze error at
zeMemOpenIpcHandle, code: ZE_RESULT_ERROR_INVALID_ARGUMENT
```

The failure occurs at the first `all_reduce` during memory profiling, before the
server becomes healthy. The same error reproduces in a two-process `xccl`
all-reduce without vLLM, which isolates it from model code.

## Root cause

### 1. Spawn-time device visibility

A container-level `ZE_AFFINITY_MASK=0,1` is not equivalent to one mask per
worker. Both subprocesses can otherwise initialize Level Zero while seeing both
cards. Changing the mask later inside `init_device()` is too late.

`patches/patch_vllm_worker_affinity.py` injects
`ZE_AFFINITY_MASK=<rank>` before each worker is spawned, so each rank starts with
one visible device. This also avoids exposing both cards' allocations to every
worker process.

### 2. oneCCL algorithm selection on non-P2P topology

Above a small message threshold, oneCCL's large SYCL all-reduce path exchanges
peer device-memory handles. On desktop dual-card layouts without GPU P2P, the
`xe` driver rejects the resulting `zeMemOpenIpcHandle` call.

Raising the simple-algorithm thresholds keeps practical model collectives on
simple/tmp-buffer algorithms that do not open peer device memory. This is the
same workaround class Intel documented for dual B60 on a platform without P2P:
[intel/llm-scaler#594](https://github.com/intel/llm-scaler/issues/594).

## Required infrastructure contract

### Container environment

```bash
-e ZE_AFFINITY_MASK=0,1 \
-e B70_WORKER_AFFINITY=1 \
-e CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296 \
-e CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296 \
-e CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296 \
-e CCL_SYCL_ALLTOALL_TMP_BUF=1
```

`4294967296` is Intel's documented workaround value. A lower value is only
acceptable when the operator proves every collective remains below it.

Do not add `CCL_ZE_IPC_EXCHANGE`, `CCL_ATL_TRANSPORT`, `FI_PROVIDER`, or
`CCL_ATL_SHM` overrides. They select how handles are exchanged, not whether the
failing peer-device IPC algorithm is selected, and some combinations return the
runtime to the failing path.

### Docker flags

```bash
--device /dev/dri --ipc=host --cap-add SYS_PTRACE
```

`SYS_PTRACE` permits oneCCL's cross-rank pidfd exchange under Docker seccomp.
`--privileged` and `--security-opt seccomp=unconfined` are not required.

### Worker-affinity patch

Apply `patches/patch_vllm_worker_affinity.py` at container startup and fail
closed if it does not print its success line. Model-specific patches must come
from the selected model recipe and
[IMAGE-AND-PATCH-MATRIX.md](IMAGE-AND-PATCH-MATRIX.md); this infrastructure page
does not define their order.

### Parallelism and graph mode

Use either:

```text
--tensor-parallel-size 2
```

or:

```text
--pipeline-parallel-size 2 --tensor-parallel-size 1
```

The affinity patch covers both worker topologies. XPU graph capture has been
refused for multi-GPU execution on the tested image generations, so TP2/PP2 uses
compile-only execution. The model recipe remains authoritative for all other
serve flags.

The contract was independently validated on a four-B70 host: TP2 and TP4
tool-calling both completed after the worker-affinity patch and all four oneCCL
settings were applied; both topologies crashed during warmup without them.
PP4 tool-calling also passed without MTP. See
[issue #11](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/issues/11#issuecomment-5507921002).

## Verification

Expected worker lines:

```text
[B70_WORKER_AFFINITY_SPAWN] rank=0 mask=0
[B70_WORKER_AFFINITY_SPAWN] rank=1 mask=1
[B70_WORKER_AFFINITY] pid=... rank=0 visible=1
[B70_WORKER_AFFINITY] pid=... rank=1 visible=1
```

oneCCL may also warn that each process sees one device while the communicator
contains two UUIDs. That is expected with per-worker masks. A valid verification
requires successful collectives followed by a healthy model endpoint; warning
text alone is not proof of success.

## Collective-only isolation repro

Run this inside the same container with `/dev/dri`, `SYS_PTRACE`, and the oneCCL
threshold environment configured:

```python
# xccl_minirepro.py
import datetime
import os

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
torch.xpu.set_device(0)  # each process sees one GPU through its spawn-time mask
dist.init_process_group(
    "xccl",
    init_method="env://",
    world_size=2,
    rank=rank,
    timeout=datetime.timedelta(seconds=60),
)
t = torch.zeros(8, device="xpu:0") + rank + 1
dist.all_reduce(t)
print(f"[mini rank={rank}] allreduce OK mean={t.float().mean().item()}", flush=True)
dist.barrier()
print(f"[mini rank={rank}] DONE", flush=True)
```

```bash
ZE_AFFINITY_MASK=0 RANK=0 WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29517 python xccl_minirepro.py &
ZE_AFFINITY_MASK=1 RANK=1 WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29517 python xccl_minirepro.py
```

Expected result: both ranks report `allreduce OK mean=3.0` and `DONE`.

## Topology notes

- `pcie_acs_override` is not required. The failure is a user-space Level Zero
  peer-memory open, not kernel PCIe routing.
- The relevant class is two CPU-attached cards without a shared P2P-capable
  switch. Host-staged collectives are therefore part of the infrastructure
  contract.
- One TP2/PP2 engine occupies both cards. Independent single-card engines remain
  a separate deployment topology.
- Do not infer a scaling factor from topology alone. Performance and power
  claims require a model-specific matched card.

Provenance: Intel workaround
[intel/llm-scaler#594](https://github.com/intel/llm-scaler/issues/594), cookbook
confirmations and debugging trail
[issue #8](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/issues/8),
and the independently developed
[humble-b70-llm](https://github.com/JP-devv/humble-b70-llm) host-staged route.

## Host-RAM mirroring of device allocations (NEO #953 / #1019)

**Symptom.** On a TP2 host, `free` Available drops by tens of GiB at model
load and again at KV-cache sizing, while the serving process RSS accounts for
far less. Under memory pressure the engine dies with an out-of-memory that
names the *device* even when VRAM is clearly free — host RAM is what died.

Two stacked regressions produce this; they are independent and additive:

1. **Peer GTT pinning (NEO 26.09 → 26.34).** Every large device-USM
   allocation on one card was eagerly pinned into the peer card's GTT —
   ~1 GiB host-pinned per 16 MiB of segment on a 2-GPU topology. Fixed in
   NEO 26.35 (compute-runtime#953). Check the **container's** NEO, not the
   host's: serving images ship their own copy in
   `/usr/lib/x86_64-linux-gnu/` (strings `libze_intel_gpu.so`); the host ICD
   is irrelevant inside a container.
2. **Deferred-backing export (kernel 7.1+ + NEO).** Kernel commits
   (populate-on-dma-buf-export, from 7.1-rc1) make every dma-buf export of a
   NEO deferred-backing BO allocate a real host-RAM page table. TP2
   peer-imports every peer allocation, so the KV cache gets an ~1:1 host-RAM
   mirror (compute-runtime#1019, open).

**Detection.** The mirroring is invisible in process RSS. The honest host
budget is:

```bash
awk '/^(VmRSS|Shmem):/{print} /^Rss:|^Shmem:/{s+=$2} END{print "sum "s" kB"}' \
  /proc/$(pgrep -f "vllm serve")/status; free -m | awk '/^Mem:/{print $7}'
```

If `free` available dropped by far more than RSS + Shmem across a boot, the
mirroring is active. The tell during boot is a step at KV-cache sizing.

**Fix (tested).** Pass to the TP2 container:

```text
-e NEOReadDebugKeys=1 -e EnableDeferBacking=0
```

`EnableDeferBacking=0` removes the deferred-backing BOs the kernel change
acts on (the old eager backing path is unaffected). Cost is real: our
n=160 A/B (same image/model/profile, flag on/off) measured a ~5-6%
decode-median regression, direction consistent across all eight
categories (sign test p≈0.004). Upstream reports zero cost; on our
stack we did not reproduce that. Treat the flag as a memory-for-speed
trade: set it when the mirroring threatens host RAM, leave it unset
when it does not. The flag must be
set in **every** launch branch: launchers that split `TP=2` from `TP=1`
carry separate env blocks, and setting it on the single-GPU path only has no
effect on TP2 containers.

**Scope.** Single-GPU containers do not peer-import and are unaffected.
Verified on the author's dual-B70 host (kernel 7.2.9, container NEO 26.27)
on two TP2 stacks: host-RAM used at steady state fell from the mid-30s GiB
with invisible mirroring to ~17 GiB fully accounted, with boot healthy and
the decode cost stated above applies.

**Trade table (author's host, dual-B70 TP2, n=160 harness):** mirroring
off costs ~5-6% decode median and buys back ~19 GiB of host RAM that the
mirror consumed. On a host where that RAM is not scarce, running without
the flag is the faster tested configuration; the flag is the stability
switch, not a free win.
