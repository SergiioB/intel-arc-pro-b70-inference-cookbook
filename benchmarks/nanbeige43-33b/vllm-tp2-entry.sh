#!/bin/bash
# Container entrypoint for Nanbeige4.3-33B TP2 on two B70s (docs/DUAL-B70-TP2.md contract).
# Applies patches/patch_vllm_worker_affinity.py and requires its two core parts:
# spawn-time ZE_AFFINITY_MASK per worker and the per-worker device remap.
set -u
python3 /patches/patch_vllm_worker_affinity.py
V=$(cd / && python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))" 2>/dev/null)
echo "[nb-tp2-entry] vllm at $V"
grep -q B70_WORKER_AFFINITY_SPAWN "$V/v1/executor/multiproc_executor.py" \
  || { echo "FATAL: affinity spawn patch missing" >&2; exit 90; }
grep -q '\[B70_WORKER_AFFINITY\] pid=' "$V/v1/worker/xpu_worker.py" \
  || { echo "FATAL: affinity worker patch missing" >&2; exit 91; }
echo "[nb-tp2-entry] affinity patch core installed"
exec vllm "$@"
