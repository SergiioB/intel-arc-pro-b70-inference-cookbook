#!/usr/bin/env bash
# B70 Nemotron-3.5-Lightning vLLM launcher — DFlash n_spec=7 (~186.6 t/s E2).
#
# WHEN TO USE: serve the published GPTQ INT4 G64 target with the published
#   BF16 DFlash draft on one B70. Isolated n=5 evidence is in
#   docs/nemotron35-30a3/NEMOTRON-DFLASH-B70.md.
# WHAT IT DOES: applies native grouped-topk v2 (vllm#52159) + B70 SSU B8/W4
#   inside the pinned public image, then starts vLLM with XPU graphs, async
#   scheduling, cache off, and method=dflash. Kernel at::zeros (kernels#524)
#   is optional and is NOT applied here (needs a kernels rebuild).
# OUTPUT: OpenAI-compatible endpoint on PORT.
# SAFETY: refuses another inference process; does not change host power.
#
# Example:
#   bash benchmarks/nemotron35-30a3/launch-nemotron-dflash.sh \
#     /path/to/gptq-target /path/to/dflash-bf16 8001
set -euo pipefail

MODEL_DIR=${1:?usage: $0 MODEL_DIR DRAFT_DIR [PORT]}
DRAFT_DIR=${2:?usage: $0 MODEL_DIR DRAFT_DIR [PORT]}
PORT=${3:-8001}
# Serving default is the completed capacity limit. The published 186.6 / 7160
# speed card was measured at 16384 — set MAX_MODEL_LEN=16384 to reproduce it.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-120000}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
IMAGE='vllm/vllm-openai-xpu@sha256:1da0a95485455f08588c11080b9718992fd7d434c6a965d74654903a9d999c57'
SERVED='nemotron-gptq-dflash7'
CONTAINER='b70-nemotron-dflash'
PATCH="$ROOT/patches/patch_xpu_grouped_topk_native_v2.py"
SSU_DIR="$ROOT/patches/ssu-b70-b8w4"
NUM_SPEC=7

if [ ! -d "$MODEL_DIR" ]; then
  echo "Model directory not found: $MODEL_DIR" >&2
  exit 1
fi
if [ ! -d "$DRAFT_DIR" ]; then
  echo "Draft directory not found: $DRAFT_DIR" >&2
  exit 1
fi
if [ ! -f "$DRAFT_DIR/model.safetensors" ]; then
  echo "Draft weights missing: $DRAFT_DIR/model.safetensors" >&2
  exit 1
fi
if [ -e "$DRAFT_DIR/hf_quant_config.json" ]; then
  echo "ABORT: $DRAFT_DIR/hf_quant_config.json is live. Rename it or vLLM will treat the draft as NVFP4." >&2
  exit 1
fi
if pgrep -af '[l]lama-server|[l]lama-bench|[v]llm serve|[v]llm.entrypoints' >/dev/null; then
  echo 'Refusing to launch: another inference process is active.' >&2
  exit 1
fi

SPEC=$(printf '{"method":"dflash","model":"/draft","num_speculative_tokens":%s}' "$NUM_SPEC")
GID=$(getent group render | cut -d: -f3)
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run --name "$CONTAINER" --rm --network host \
  --device /dev/dri --group-add "$GID" -v /dev/dri:/dev/dri:ro \
  --ipc host --shm-size 16g \
  -v "$MODEL_DIR:/model:ro" \
  -v "$DRAFT_DIR:/draft:ro" \
  -v "$PATCH:/patch_v2.py:ro" \
  -v "$SSU_DIR:/ssu:ro" \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 -e SYCL_CACHE_PERSISTENT=0 \
  --entrypoint bash "$IMAGE" \
  -lc "set -e; python /patch_v2.py; \
    cp /ssu/*.json /workspace/vllm/vllm/model_executor/layers/mamba/ops/configs/selective_state_update/ 2>/dev/null || true; \
    cp /ssu/*.json /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/ops/configs/selective_state_update/ 2>/dev/null || true; \
    exec vllm serve /model \
      --served-model-name $SERVED \
      --dtype float16 --quantization gptq \
      --max-model-len $MAX_MODEL_LEN --max-num-seqs 1 --max-num-batched-tokens 8192 \
      --gpu-memory-utilization 0.90 --no-enable-prefix-caching \
      --language-model-only --async-scheduling \
      --speculative-config '$SPEC' \
      --host 0.0.0.0 --port $PORT"
echo "launched $CONTAINER on :$PORT (model $SERVED)"
