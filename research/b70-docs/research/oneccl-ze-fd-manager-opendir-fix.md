# oneCCL `ze_fd_manager opendir failed` in Docker — vLLM XPU on Arc Pro B70 (TP=1)

Error: `RuntimeError: oneCCL: ze_fd_manager.cpp:144 init_device_fds: EXCEPTION: opendir failed: could not open device directory`
Site: `vllm/v1/worker/xpu_worker.py init_device()` → `torch.distributed.all_reduce(torch.zeros(1).xpu())`

## Root-cause chain (verified)

1. oneCCL IPC exchange defaults to `pidfd`. Docker's default seccomp profile blocks
   `pidfd_getfd` unless the container has `CAP_SYS_PTRACE`
   (https://github.com/moby/moby/blob/master/profiles/seccomp/default_linux.go —
   `pidfd_getfd`/`ptrace`/`kcmp` only allowed `Includes: Caps: [CAP_SYS_PTRACE]`;
   confirmed by https://github.com/moby/moby/issues/45622).
2. `ccl_comm::init_ipc_exchange_mode` (src/comm/comm.cpp) then falls back to `drmfd`
   mode — and a logic bug means even an explicit `CCL_ZE_IPC_EXCHANGE=sockets` is
   silently coerced to `drmfd` when pidfd is unsupported
   (https://github.com/uxlfoundation/oneCCL/issues/213 — filed on 2× Arc Pro B70 + vLLM).
3. `drmfd` mode runs `opendir("/dev/dri/by-path")` in `ze_fd_manager.cpp:144
   init_device_fds`. With only `--device /dev/dri`, the by-path dir does not exist in
   the container → the exact exception.
4. vLLM runs this warm-up all_reduce even at world_size==1 (single GPU), so oneCCL
   init is forced on a box that never issues a collective
   (https://github.com/vllm-project/vllm/issues/52386).

NOTE: `CCL_ZE_IPC_EXCHANGE=sockets` does NOT work around this (silently overridden, oneCCL #213).

## Ranked fixes

### 1. Mount `/dev/dri/by-path` into the container (canonical, zero downside)
```bash
docker run ... --device /dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path \
  <vllm-xpu-image>
```
- Intel's documented run command for every XPU container (intel/vllm Docker Hub,
  intel/ai-containers, IPEX): `--shm-size 10g --net=host --ipc=host --privileged
  -v /dev/dri/by-path:/dev/dri/by-path --device /dev/dri`
- Named as working workaround in oneCCL #213 on the same B70 hardware.
- Requires `/dev/dri/by-path` populated on the HOST (udev does this). If the host
  lacks it, create the symlinks (pattern from
  https://github.com/stevezau/media_preview_generator/commit/736d6c0):
  ```bash
  # host, run once (or as container entrypoint):
  mkdir -p /dev/dri/by-path
  for n in /dev/dri/renderD*; do
    pci=$(basename "$(readlink -f /sys/class/drm/$(basename $n)/device)")
    ln -sf ../$(basename $n) /dev/dri/by-path/pci-$pci-render
  done
  ```
Sources:
- https://hub.docker.com/r/intel/vllm (Start a container)
- https://github.com/intel/ai-containers/blob/main/vllm/0.17.0-xpu.md
- https://intel.github.io/ai-containers/pytorch/
- https://github.com/uxlfoundation/oneCCL/issues/213

### 2. `--cap-add SYS_PTRACE` so oneCCL stays on its default pidfd mode
```bash
docker run ... --device /dev/dri --cap-add SYS_PTRACE <image>
# older Docker engines whose seccomp allowlist predates pidfd_getfd may also need:
#   --security-opt seccomp=unconfined
```
- Exact workaround from oneCCL #213 ("--cap-add SYS_PTRACE --security-opt seccomp=unconfined
  to make pidfd work"), same B70 + vLLM setup.
- Default profile allows pidfd_getfd when CAP_SYS_PTRACE present (moby default_linux.go;
  moby#45622: "--cap-add=SYS_PTRACE makes it work" on 20.10 and 24.x).
- `--privileged: true` is the blunt variant that also resolves it (ipex-llm #12990).
Sources:
- https://github.com/uxlfoundation/oneCCL/issues/213
- https://github.com/moby/moby/issues/45622
- https://github.com/intel/ipex-llm/issues/12990

### 3. Skip the oneCCL warm-up at world_size==1 (upstream fix; for TP=1 this is the "true" fix)
- Merged upstream: PR #52389, commit f94666b ("[Bugfix][XPU] Skip oneCCL warm-up
  all_reduce when world_size == 1"), fixes issue #52386. Guard:
  ```python
  if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1 \
     and torch.distributed.is_xccl_available():
      torch.distributed.all_reduce(torch.zeros(1).xpu())
  ```
- For a pinned nightly that still fails: either move to a nightly containing f94666b
  (merged ~2026-08; check `git merge-base --is-ancestor f94666b ac7509e2b` in a vllm
  checkout) or patch the installed file in the image:
  ```bash
  # inside image at /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/xpu_worker.py
  # guard the warm-up all_reduce with a world_size>1 check (7-line diff, see PR)
  ```
- Sitecustomize-style no-patch alternative for TP=1: none exists via env; `CCL_ZE_ENABLE=0`
  only converts the failure into "ze_data was not initialized" (#52386).
Sources:
- https://github.com/vllm-project/vllm/issues/52386
- https://github.com/vllm-project/vllm/pull/52389 / commit f94666b60d4c58ec0807d22c837cfae322a1dde9
