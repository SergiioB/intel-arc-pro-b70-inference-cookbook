#!/usr/bin/env bash
# Serve unmodified thinkingmachines/Inkling-2-Mini on one Intel Arc Pro B70 with vLLM XPU + vllm-inkling2-xpu.
#
# Usage:
#   VLLM_BIN=/path/to/venv/bin/vllm MODEL=/path/to/Inkling-2-Mini \
#     benchmarks/inkling2-mini/launch-inkling2-mini-vllm-xpu.sh
#
# Verified 1x B70 operating profiles (override via env):
#   1. Fastest interactive (106K ctx, ~2.5 GiB free):  DSPARK=1 INK2_CUTLASS_MOE=1 CTX=106496 MAX_BATCHED=2048 GPU_MEM_UTIL=0.93
#   2. Long-context DSpark (163K ctx, ~1.5 GiB free):  DSPARK=1 INK2_CUTLASS_MOE=1 CTX=163840 MAX_BATCHED=1024 GPU_MEM_UTIL=0.95
#   3. Balanced stock daily (90K ctx, ~4.0 GiB free):  CTX=90112  MAX_BATCHED=2048 GPU_MEM_UTIL=0.88
#   4. Full-speed 200K no-spec (~1.8 GiB free):        CTX=200000 MAX_BATCHED=2048 GPU_MEM_UTIL=0.95
#   5. Max 256K no-spec BF16 (273.8K KV, ~1.65 GiB):   CTX=262144 MAX_BATCHED=1024 GPU_MEM_UTIL=0.95
#   6. Vision + DSpark + FP8 KV (163K / 177.6K KV):    VISION=1 KV_DTYPE=fp8 DSPARK=1 INK2_CUTLASS_MOE=1 CTX=163840 MAX_BATCHED=2048 GPU_MEM_UTIL=0.92
#   7. Vision + DSpark + FP8 KV 256K (323.4K KV):      VISION=1 KV_DTYPE=fp8 DSPARK=1 INK2_CUTLASS_MOE=1 CTX=262144 MAX_BATCHED=1024 GPU_MEM_UTIL=0.95
#   8. Max 512K no-spec FP8 KV (569.5K KV):            KV_DTYPE=fp8 CTX=524288 MAX_BATCHED=1024 GPU_MEM_UTIL=0.95
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
VISION="${VISION:-0}"
KV_DTYPE="${KV_DTYPE:-auto}"
INK2_CUTLASS_MOE="${INK2_CUTLASS_MOE:-0}"
INK2_SYCL_DECODE_ATTN="${INK2_SYCL_DECODE_ATTN:-0}"

RPC_DIR="$(mktemp -d /tmp/ink2-ipc-XXXXXX)"
trap 'rm -rf "$RPC_DIR"' EXIT

export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ZE_AFFINITY_MASK="$GPU"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_RPC_BASE_PATH="$RPC_DIR"
export INK2_CUTLASS_MOE
export INK2_SYCL_DECODE_ATTN

EXTRA=()
if [[ "$VISION" == "1" ]]; then
  EXTRA+=(--limit-mm-per-prompt '{"image":4}')
else
  EXTRA+=(--language-model-only)
fi
if [[ "$KV_DTYPE" != "auto" ]]; then
  EXTRA+=(--kv-cache-dtype "$KV_DTYPE")
fi
if [[ "$DSPARK" == "1" ]]; then
  EXTRA+=(--speculative-config '{"method":"dspark","num_speculative_tokens":7}')
fi

exec "$VLLM_BIN" serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
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
