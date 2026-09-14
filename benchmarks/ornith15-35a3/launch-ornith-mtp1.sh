#!/usr/bin/env bash
# Launch Ornith-1.5-35B-A3B MixedCal-v2 on vLLM XPU (Intel Arc Pro B70).
# Default: MODE=mtp1, DRAFT_INT4=1, prefix cache off. Does not change host power.
#
#   bash benchmarks/ornith15-35a3/launch-ornith-mtp1.sh /path/to/model 16384 8000
#   BOUNDARY=1 bash benchmarks/ornith15-35a3/launch-ornith-mtp1.sh /path/to/model 131072 8000
#   DRAFT_INT4=0 MODE=mtp1 bash ...   # BF16 MTP draft
#   MODE=no-spec bash ...            # no speculation
set -euo pipefail

MODEL_DIR=${1:?usage: $0 MODEL_DIR [CTX] [PORT]}
CTX=${2:-16384}
PORT=${3:-8000}
MODE=${MODE:-mtp1}
CACHE=${CACHE:-off}
BOUNDARY=${BOUNDARY:-0}
UTIL=${UTIL:-0.85}
DRAFT_INT4=${DRAFT_INT4:-1}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
MODEL='Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2'
CONTAINER="b70-ornith-${MODE}-c${CTX}"
SPEC="/tmp/${CONTAINER}-spec.json"

if [ ! -d "$MODEL_DIR" ]; then
  echo "Model directory not found: $MODEL_DIR" >&2
  exit 1
fi
if pgrep -af '[l]lama-server|[l]lama-bench|[v]llm serve|[v]llm-xpu' >/dev/null; then
  echo 'Refusing to launch: another inference process is active.' >&2
  exit 1
fi
if [ -n "$(docker ps -q 2>/dev/null)" ]; then
  echo 'Refusing to launch: a container is already running.' >&2
  exit 1
fi

case "$MODE" in
  no-spec)
    printf '%s\n' '{}' > "$SPEC"
    SPEC_INLINE=''
    ;;
  mtp1|mtp2|mtp4)
    N=${MODE#mtp}
    printf '{"method":"mtp","num_speculative_tokens":%s}\n' "$N" > "$SPEC"
    SPEC_INLINE='--speculative-config "$(cat /spec.json)"'
    ;;
  *)
    echo 'MODE must be no-spec, mtp1, mtp2, or mtp4.' >&2
    exit 2
    ;;
esac

case "$CACHE" in
  off) CACHE_ARG=--no-enable-prefix-caching ;;
  on) CACHE_ARG=--enable-prefix-caching ;;
  *) echo 'CACHE must be on or off.' >&2; exit 2 ;;
esac

PRE="exec vllm serve /model"
PATCH_MOUNT=()
ENV_EXTRA=()
APPLY=""
if [ "$BOUNDARY" = 1 ]; then
  APPLY="${APPLY}python /patch_boundary.py; "
  PATCH_MOUNT+=(-v "$ROOT/patches/patch_mtp_boundary.py:/patch_boundary.py:ro")
fi
if [ "$DRAFT_INT4" = 1 ] && [ "$MODE" != "no-spec" ]; then
  APPLY="${APPLY}python /patch_s.py; python /patch_m1.py; "
  PATCH_MOUNT+=(
    -v "$ROOT/patches/patch_draft_lmhead_int4.py:/patch_s.py:ro"
    -v "$ROOT/patches/patch_draft_mtp_int4.py:/patch_m1.py:ro"
  )
  ENV_EXTRA+=(-e B70_DRAFT_LMHEAD_INT4=1 -e B70_DRAFT_MTP_INT4=1)
fi
if [ -n "$APPLY" ]; then
  PRE="${APPLY}exec vllm serve /model"
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
RENDER_GID=$(stat -c '%g' /dev/dri/render* | sort -u | sed -n '1p')
KV_EXTRA=${KV_EXTRA:-}

docker run -d --restart unless-stopped --name "$CONTAINER" -p "$PORT:8000" \
  --device /dev/dri --group-add "$RENDER_GID" \
  -v /dev/dri:/dev/dri:ro \
  -v "$MODEL_DIR:/model:ro" \
  -v "$SPEC:/spec.json:ro" \
  "${PATCH_MOUNT[@]}" \
  "${ENV_EXTRA[@]}" \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash "$IMAGE" -lc \
  "set -e
$PRE --quantization gptq --dtype float16 --max-model-len $CTX \
  --gpu-memory-utilization $UTIL $KV_EXTRA --kv-cache-dtype auto \
  --port 8000 --host 0.0.0.0 --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --block-size 64 $CACHE_ARG --served-model-name $MODEL \
  --language-model-only --trust-remote-code $SPEC_INLINE"

cat <<EOF
Container: $CONTAINER
Image:     $IMAGE
Mode:      $MODE
Ctx:       $CTX
Cache:     $CACHE
Boundary:  $BOUNDARY
DraftINT4: $DRAFT_INT4
Logs:      docker logs -f $CONTAINER
Health:    curl -f http://127.0.0.1:$PORT/health
Stop:      docker rm -f $CONTAINER
EOF
