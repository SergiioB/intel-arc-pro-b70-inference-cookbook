#!/bin/bash
# Container entrypoint for Bolt TP2 on two B70s (cookbook docs/DUAL-B70-TP2.md contract).
# Applies the cookbook's patch_vllm_worker_affinity.py, then requires its two core parts:
# spawn-time ZE_AFFINITY_MASK per worker and the per-worker device remap. On vLLM 0.31.0
# the patch's other anchors are gone (device-count assert rewritten; Mamba pointer wrap
# is upstream in mamba_utils.py), so its all-installed line can't appear; check markers.
set -u
python3 /patches/patch_vllm_worker_affinity.py
# resolve like the `vllm` CLI does (site-packages); in the image WORKDIR a bare
# `python3 -c` would import the /workspace/vllm source tree instead
V=$(cd / && python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))" 2>/dev/null)
echo "[tp2-entry] vllm at $V"
grep -q B70_WORKER_AFFINITY_SPAWN "$V/v1/executor/multiproc_executor.py" \
  || { echo "FATAL: affinity spawn patch missing" >&2; exit 90; }
grep -q '\[B70_WORKER_AFFINITY\] pid=' "$V/v1/worker/xpu_worker.py" \
  || { echo "FATAL: affinity worker patch missing" >&2; exit 91; }
grep -q 'value - (1 << 64)' "$V/v1/worker/mamba_utils.py" \
  || { echo "FATAL: no Mamba pointer wrap in this vLLM" >&2; exit 92; }
# Bolt routes 9 experts/token; remap_hidden_states has no TopK=9 kernel: pad with zero weight
python3 /patches/patch_xpu_moe_topk_pad.py || { echo "FATAL: topk pad patch failed" >&2; exit 93; }
echo "[tp2-entry] affinity patch core installed; Mamba pointer wrap upstream"
exec vllm "$@"
