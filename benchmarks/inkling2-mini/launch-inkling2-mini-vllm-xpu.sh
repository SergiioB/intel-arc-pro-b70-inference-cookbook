#!/usr/bin/env bash
# Serve unmodified thinkingmachines/Inkling-2-Mini on one Intel Arc Pro B70 with vLLM XPU + vllm-inkling2-xpu.
#
# Usage:
#   VLLM_BIN=/path/to/venv/bin/vllm MODEL=/path/to/Inkling-2-Mini \
#     benchmarks/inkling2-mini/launch-inkling2-mini-vllm-xpu.sh
#
# Profiles (override via env):
#   Balanced daily (90K ctx, ~4 GiB free VRAM):   CTX=90112  GPU_MEM_UTIL=0.88
#   Max single-card context (200K ctx, ~1.8 GiB): CTX=200000 GPU_MEM_UTIL=0.95
#   Optional DSpark k=7 (up to 106K ctx):         DSPARK=1 CTX=106496 GPU_MEM_UTIL=0.93
set -euo pipefail

: "${VLLM_BIN:=vllm}"
: "${MODEL:?path or repo id for thinkingmachines/Inkling-2-Mini@ed40e7da}"
GPU="${GPU:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_NAME:-inkling-2-mini}"
CTX="${CTX:-90112}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
MAX_BATCHED="${MAX_BATCHED:-2048}"
DSPARK="${DSPARK:-0}"

RPC_DIR="$(mktemp -d /tmp/ink2-ipc-XXXXXX)"
trap 'rm -rf "$RPC_DIR"' EXIT

export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ZE_AFFINITY_MASK="$GPU"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_RPC_BASE_PATH="$RPC_DIR"

EXTRA=()
if [[ "$DSPARK" == "1" ]]; then
  EXTRA+=(--speculative-config '{"method":"dspark","num_speculative_tokens":7}')
fi

exec "$VLLM_BIN" serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --language-model-only \
  --dtype bfloat16 \
  --max-model-len "$CTX" \
  --max-num-batched-tokens "$MAX_BATCHED" \
  --max-num-seqs 1 \
  --block-size 64 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --no-enable-prefix-caching \
  -cc.mode=none \
  -cc.cudagraph_mode=FULL_DECODE_ONLY \
  --reasoning-parser inkling \
  --enable-auto-tool-choice \
  --tool-call-parser inkling \
  "${EXTRA[@]}"
