#!/usr/bin/env bash
# Serve Inkling 2 Mini UD-Q4_K_XL on one Arc Pro B70 with the exact (bit-exact) default of the recipe.
#
#   LLAMA_BIN=<tree>/build-sycl/bin MODEL=<dir>/Inkling-2-Mini-UD-Q4_K_XL.gguf \
#     benchmarks/inkling2-mini/launch-inkling2-mini-llamacpp.sh [extra llama-server args]
#
# Every variable is explained in docs/inkling2-mini/INKLING2-MINI-LLAMACPP.md.
# The measured cells added: --no-cache-prompt --cache-ram 0 (cold prompts).
set -euo pipefail
: "${LLAMA_BIN:?directory with llama-server}"
: "${MODEL:?path to Inkling-2-Mini-UD-Q4_K_XL.gguf}"
GPU=${GPU:-0}
PORT=${PORT:-8050}
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null; set -u

export GGML_SYCL_ENABLE_GRAPH=1          # SYCL graph replay: decode +20-34 %, same greedy text
export GGML_SYCL_XMX_GATHER_TYPES=32767  # keep patch 0002 (NVFP4 fused prefill GEMM) off: bit-exact default
export SYCL_CACHE_PERSISTENT=0           # the persistent kernel cache crashes this DPC++ release (DNR-031)
export ZE_AFFINITY_MASK=$GPU             # one card

exec "$LLAMA_BIN/llama-server" -m "$MODEL" -ngl 99 -lm none -c 49152 -t 8 -np 1 --jinja \
  --host 127.0.0.1 --port "$PORT" "$@"
