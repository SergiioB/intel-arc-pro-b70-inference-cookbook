#!/usr/bin/env bash
# B70 Nemotron-3.5-Lightning vLLM launcher — XPU-graph decode config (~93 t/s).
#
# WHEN TO USE: serve the local symmetric GPTQ INT4 G64 conversion with the
#   compiled + XPU-graph config that raised decode from ~21.8 to ~93/87 t/s.
# WHAT IT DOES: applies the native grouped-topk router patch (v2, graph-safe)
#   + the B70 SSU tuning JSON inside the pinned public image, then starts vLLM
#   with graphs, async scheduling and the exact tested flags.
# OUTPUT: OpenAI-compatible endpoint on PORT; Docker logs in the named container.
# SAFETY: refuses another inference process; does not change host power.
#
# NOTE: decode is verified (n=5) and coherent, but temperature-0
#   deterministic replay does not hold on this stack (XPU compiled-kernel FP
#   race).
#
# Example:
#   bash benchmarks/nemotron35-30a3/launch-nemotron-graph.sh /path/to/model 8001
set -euo pipefail

MODEL_DIR=${1:?usage: $0 MODEL_DIR [PORT]}
PORT=${2:-8001}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
IMAGE='vllm/vllm-openai-xpu@sha256:1da0a95485455f08588c11080b9718992fd7d434c6a965d74654903a9d999c57'
SERVED='Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym'
CONTAINER='b70-nemotron-graph'
PATCH="$ROOT/patches/patch_xpu_grouped_topk_native_v2.py"
SSU_DIR="$ROOT/patches/ssu-b70-b8w4"

if [ ! -d "$MODEL_DIR" ]; then
  echo "Model directory not found: $MODEL_DIR" >&2
  exit 1
fi
if pgrep -af '[l]lama-server|[l]lama-bench|[v]llm serve|[v]llm-xpu' >/dev/null; then
  echo 'Refusing to launch: another inference process is active.' >&2
  exit 1
fi

GID=$(getent group render | cut -d: -f3)
docker run --name "$CONTAINER" --rm --network host \
  --device /dev/dri --group-add "$GID" -v /dev/dri:/dev/dri:ro \
  --ipc host --shm-size 16g \
  -v "$MODEL_DIR:/model:ro" \
  -v "$PATCH:/patch_v2.py:ro" \
  -v "$SSU_DIR:/ssu:ro" \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 -e SYCL_CACHE_PERSISTENT=0 \
  --entrypoint bash "$IMAGE" \
  -lc "python /patch_v2.py; cp /ssu/*.json /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/ops/configs/selective_state_update/ 2>/dev/null || true; exec vllm serve /model --served-model-name $SERVED --dtype bfloat16 --quantization gptq --max-model-len 16384 --max-num-seqs 1 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 --no-enable-prefix-caching --async-scheduling --host 0.0.0.0 --port $PORT" >/dev/null 2>&1 &
echo "launched $CONTAINER on :$PORT (model $SERVED)"
