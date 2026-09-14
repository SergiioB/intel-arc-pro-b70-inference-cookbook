#!/usr/bin/env python3
"""Per-worker ZE_AFFINITY_MASK injection for vLLM-XPU TP2/PP2 on dual-B70.

WHEN TO USE: dual-B70 serving (TP2 or PP2) on the pinned vLLM-XPU image
(vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f).

ROOT CAUSE:
1) Discrete Arc GPUs on consumer PCIe topologies without ACS lack direct P2P aperture
   routing (zeMemOpenIpcHandle fails with ZE_RESULT_ERROR_INVALID_ARGUMENT if both GPUs
   are exposed in the same Level Zero driver context).
2) Exposing both GPUs to a single process under the xe driver causes ~1 GiB host RAM
   per 1 GiB VRAM driver duplication.
3) Setting ZE_AFFINITY_MASK inside init_device is too late (Level Zero initializes
   upon subprocess start/import).
4) Per-worker masks alone are NOT sufficient on every dual-GPU host: above a small
   message size oneCCL's SYCL allreduce uses the "large" multi-kernel algorithm,
   which performs a Level Zero IPC exchange of peer ranks' temp buffers on every
   collective ("perform IPC exchange every time",
   oneCCL src/coll/algorithms/allreduce/sycl/allreduce_large_sycl.hpp) — the exact
   zeMemOpenIpcHandle call xe rejects on non-P2P boards. The CCL_SYCL_*_SIMPLE_THRESHOLD
   variables below pin every realistic TP2 message on the simple/tmp-buffer
   algorithms, which never open peer device memory. Intel's workaround for the
   identical 2x B60 failure: intel/llm-scaler#594. Dual-B70 confirmations:
   intel-arc-pro-b70-inference-cookbook issue #8.

WHAT THIS PATCH DOES:
1) vllm/v1/executor/multiproc_executor.py: injects ZE_AFFINITY_MASK=rank at worker spawn time.
2) vllm/v1/worker/xpu_worker.py: remaps local device index to 0 (since mask selects physical GPU)
   and verifies single visible device.
3) vllm/v1/worker/mamba_utils.py: wraps pointer offsets safely for high 64-bit addresses.

REQUIREMENTS:
- Run container with: --cap-add SYS_PTRACE (or seccomp=unconfined) and --ipc=host
- Env (required on non-P2P dual-GPU hosts — Intel workaround, llm-scaler#594):
    CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296
    CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296
    CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296
    CCL_SYCL_ALLTOALL_TMP_BUF=1
- Leave CCL_ZE_IPC_EXCHANGE / CCL_ATL_TRANSPORT / FI_PROVIDER / CCL_ATL_SHM at their
  defaults: they only choose how IPC handles are exchanged, not whether device IPC is
  used, and some overrides push oneCCL back into the failing path.
- Full guide (root cause, launch, verification, isolation repro): docs/DUAL-B70-TP2.md
"""
import os
import sys
from pathlib import Path

SP = Path("/opt/venv/lib/python3.12/site-packages")
EXECUTOR = SP / "vllm/v1/executor/multiproc_executor.py"
WORKER = SP / "vllm/v1/worker/xpu_worker.py"
MAMBA = SP / "vllm/v1/worker/mamba_utils.py"

# --- 1. Patch multiproc_executor.py ---
EXECUTOR_ANCHOR = (
    "            for local_rank in range(self.local_world_size):\n"
    "                global_rank = global_start_rank + local_rank\n"
)

EXECUTOR_INJECT = EXECUTOR_ANCHOR + '''\
                if os.environ.get("B70_WORKER_AFFINITY", "1") == "1" and (
                    (
                        int(self.vllm_config.parallel_config.pipeline_parallel_size) == 2
                        and int(self.vllm_config.parallel_config.tensor_parallel_size) == 1
                    )
                    or (
                        int(self.vllm_config.parallel_config.pipeline_parallel_size) == 1
                        and int(self.vllm_config.parallel_config.tensor_parallel_size) == 2
                    )
                ):
                    # Each worker interpreter inherits its own single-GPU mask
                    os.environ["ZE_AFFINITY_MASK"] = str(global_rank)
                    print(
                        "[B70_WORKER_AFFINITY_SPAWN] "
                        f"rank={global_rank} mask={global_rank}",
                        flush=True,
                    )
'''

# --- 2. Patch xpu_worker.py ---
WORKER_ANCHOR = (
    "    def init_device(self):\n"
    "        # In DP mode, XPU workers see all visible devices.\n"
)

WORKER_INJECT = '''\
    def init_device(self):
        import os as _os
        if _os.environ.get("B70_WORKER_AFFINITY", "1") == "1" and (
            (
                int(self.parallel_config.pipeline_parallel_size) == 2
                and int(self.parallel_config.tensor_parallel_size) == 1
            )
            or (
                int(self.parallel_config.pipeline_parallel_size) == 1
                and int(self.parallel_config.tensor_parallel_size) == 2
            )
        ):
            import torch as _torch
            _n = _torch.accelerator.device_count()
            _rank = int(getattr(self, "rank", 0))
            print(
                f"[B70_WORKER_AFFINITY] pid={_os.getpid()} rank={_rank} "
                f"visible={_n}",
                flush=True,
            )
            if _n != 1:
                raise RuntimeError(
                    f"B70_WORKER_AFFINITY: rank {_rank} sees {_n} devices; "
                    "spawn-time mask injection failed"
                )
            self.local_rank = 0
            _os.environ["LOCAL_RANK"] = "0"
        # In DP mode, XPU workers see all visible devices.
'''

WORKER_ASSERT_ANCHOR = (
    "            assert parallel_config.local_world_size <= visible_device_count, (\n"
)

WORKER_ASSERT_INJECT = '''\
            assert (
                (
                    os.environ.get("B70_WORKER_AFFINITY", "1") == "1"
                    and (
                        (
                            int(parallel_config.pipeline_parallel_size) == 2
                            and int(parallel_config.tensor_parallel_size) == 1
                        )
                        or (
                            int(parallel_config.pipeline_parallel_size) == 1
                            and int(parallel_config.tensor_parallel_size) == 2
                        )
                    )
                )
                or parallel_config.local_world_size <= visible_device_count
            ), (
'''

# --- 3. Patch mamba_utils.py (Pointer wrapping for >= 2**63) ---
MAMBA_ANCHOR = (
    "                for state_type_idx, state in enumerate(kv_caches):\n"
    "                    # Base address\n"
    "                    self.state_base_addrs[idx] = state.data_ptr()\n"
)

MAMBA_INJECT = (
    "                for state_type_idx, state in enumerate(kv_caches):\n"
    "                    # Base address\n"
    "                    # B70_PTR_WRAP_BASE: XPU device pointers can sit at >= 2**63;\n"
    "                    # store two's-complement-wrapped so int64 tensors hold\n"
    "                    # the exact same bits (kernels only add small offsets).\n"
    "                    _b70_ptr = state.data_ptr()\n"
    "                    self.state_base_addrs[idx] = (\n"
    "                        _b70_ptr - (1 << 64) if _b70_ptr >= (1 << 63) else _b70_ptr\n"
    "                    )\n"
)

MAMBA_BT_ANCHOR = (
    "        for i, bt in enumerate(block_tables):\n"
    "            self.block_table_ptrs[i] = bt.data_ptr()\n"
)

MAMBA_BT_INJECT = (
    "        for i, bt in enumerate(block_tables):\n"
    "            _b70_bt_ptr = bt.data_ptr()  # B70_PTR_WRAP_BT\n"
    "            self.block_table_ptrs[i] = (\n"
    "                _b70_bt_ptr - (1 << 64) if _b70_bt_ptr >= (1 << 63) else _b70_bt_ptr\n"
    "            )\n"
)

MAMBA_CP_ANCHOR = (
    "                src_ptrs_np[offset] = copy_spec.start_addr\n"
    "                dst_ptrs_np[offset] = state[dest_block_id].data_ptr()\n"
)

MAMBA_CP_INJECT = (
    "                _b70_src = copy_spec.start_addr  # B70_PTR_WRAP_CP\n"
    "                _b70_dst = state[dest_block_id].data_ptr()\n"
    "                src_ptrs_np[offset] = (\n"
    "                    _b70_src - (1 << 64) if _b70_src >= (1 << 63) else _b70_src\n"
    "                )\n"
    "                dst_ptrs_np[offset] = (\n"
    "                    _b70_dst - (1 << 64) if _b70_dst >= (1 << 63) else _b70_dst\n"
    "                )\n"
)


def patch_file(path: Path, anchor: str, inject: str, marker: str) -> bool:
    if not path.exists():
        print(f"[B70_WORKER_AFFINITY_INSTALL] Optional file {path.name} not present, skipping", flush=True)
        return True
    src = path.read_text()
    if marker in src:
        print(f"[B70_WORKER_AFFINITY_INSTALL] already installed: {path.name}", flush=True)
        return True
    n = src.count(anchor)
    if n != 1:
        print(
            f"[B70_WORKER_AFFINITY_INSTALL] FATAL: anchor count {n} != 1 "
            f"in {path}",
            flush=True,
        )
        return False
    path.write_text(src.replace(anchor, inject, 1))
    print(f"[B70_WORKER_AFFINITY_INSTALL] patched {path.name}", flush=True)
    return True


def main() -> int:
    ok1 = patch_file(EXECUTOR, EXECUTOR_ANCHOR, EXECUTOR_INJECT, "B70_WORKER_AFFINITY_SPAWN")
    ok2 = patch_file(WORKER, WORKER_ANCHOR, WORKER_INJECT, "end B70_WORKER_AFFINITY")
    ok3 = patch_file(
        WORKER,
        WORKER_ASSERT_ANCHOR,
        WORKER_ASSERT_INJECT,
        "or parallel_config.local_world_size <= visible_device_count",
    )
    ok4 = patch_file(MAMBA, MAMBA_ANCHOR, MAMBA_INJECT, "B70_PTR_WRAP_BASE")
    ok5 = patch_file(MAMBA, MAMBA_BT_ANCHOR, MAMBA_BT_INJECT, "B70_PTR_WRAP_BT")
    ok6 = patch_file(MAMBA, MAMBA_CP_ANCHOR, MAMBA_CP_INJECT, "B70_PTR_WRAP_CP")
    all_ok = ok1 and ok2 and ok3 and ok4 and ok5 and ok6
    if all_ok:
        print("All worker affinity patches installed successfully.", flush=True)
    return 0 if all_ok else 3


if __name__ == "__main__":
    sys.exit(main())
