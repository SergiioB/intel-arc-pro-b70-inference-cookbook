#!/usr/bin/env bash
# B70 vLLM 128K mode launcher — no-spec/MTP1/MTP2/MTP4, cache on/off.
#
# WHEN TO USE: launch any mode from the matched 128K cache/spec matrix.
# WHAT IT DOES: applies both current patches in order inside the pinned public
#   image, then starts vLLM with the exact tested context and scheduler flags.
# OUTPUT: OpenAI-compatible endpoint on PORT; Docker logs in the named container.
# SAFETY: refuses another inference process and does not change host power.
#
# Example:
#   bash benchmarks/qwen36-35a3/launch-vllm-128k-mode.sh /path/to/model mtp2 on 8000
set -euo pipefail

MODEL_DIR=${1:?usage: $0 MODEL_DIR MODE CACHE [PORT]}
MODE=${2:?usage: $0 MODEL_DIR MODE CACHE [PORT]}
CACHE=${3:?usage: $0 MODEL_DIR MODE CACHE [PORT]}
PORT=${4:-8000}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMAGE='vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97'
MODEL='Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4'
CONTAINER="b70-${MODE}-cache-${CACHE}"
SPEC="/tmp/${CONTAINER}-spec.json"

if [ ! -d "$MODEL_DIR" ]; then
  echo "Model directory not found: $MODEL_DIR" >&2
  exit 1
fi
if pgrep -af '[l]lama-server|[l]lama-bench|[v]llm serve|[v]llm-xpu' >/dev/null; then
  echo 'Refusing to launch: another inference process is active.' >&2
  exit 1
fi

case "$MODE" in
  no-spec)
    printf '%s\n' '{}' > "$SPEC"
    SPEC_ARG=''
    ;;
  mtp1|mtp2|mtp4)
    N=${MODE#mtp}
    printf '{"method":"mtp","num_speculative_tokens":%s}\n' "$N" > "$SPEC"
    SPEC_ARG='--speculative-config "$SPEC"'
    ;;
  *)
    echo 'MODE must be no-spec, mtp1, mtp2, or mtp4.' >&2
    exit 2
    ;;
esac

case "$CACHE" in
  on) CACHE_ARG=--enable-prefix-caching ;;
  off) CACHE_ARG=--no-enable-prefix-caching ;;
  *)
    echo 'CACHE must be on or off.' >&2
    exit 2
    ;;
esac

sudo docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
RENDER_GID=$(stat -c '%g' /dev/dri/render* | sort -u | sed -n '1p')

sudo docker run -d --restart unless-stopped --name "$CONTAINER" -p "$PORT:8000" \
  --device /dev/dri --group-add "$RENDER_GID" \
  -v /dev/dri:/dev/dri:ro \
  -v "$MODEL_DIR:/model:ro" \
  -v "$ROOT/patches/patch_mtp_nightly.py:/patch_mtp.py:ro" \
  -v "$ROOT/patches/patch_mtp_boundary.py:/patch_boundary.py:ro" \
  -v "$SPEC:/spec.json:ro" \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0 \
  -e B70_MTP_BF16_DRAFT=1 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash "$IMAGE" -lc \
  "set -e
python /patch_mtp.py
python /patch_boundary.py
SPEC=\$(cat /spec.json)
exec vllm serve /model \\
  --quantization gptq \\
  --dtype float16 \\
  --max-model-len 131072 \\
  --gpu-memory-utilization 0.85 \\
  --port 8000 \\
  --max-num-seqs 64 \\
  --max-num-batched-tokens 8192 \\
  --enable-auto-tool-choice \\
  --tool-call-parser qwen3_coder \\
  $CACHE_ARG \\
  --served-model-name $MODEL \\
  --language-model-only \\
  $SPEC_ARG"

cat <<EOF
Container: $CONTAINER
Image:     $IMAGE
Mode:      $MODE
Cache:     $CACHE
Logs:      sudo docker logs -f $CONTAINER
Health:    curl -f http://127.0.0.1:$PORT/health
Models:    curl -s http://127.0.0.1:$PORT/v1/models
Stop:      sudo docker rm -f $CONTAINER
EOF
