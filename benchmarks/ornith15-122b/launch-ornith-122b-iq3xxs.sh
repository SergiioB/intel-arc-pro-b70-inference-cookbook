#!/usr/bin/env bash
# Ornith-1.5-122B Unsloth UD-IQ3_XXS on llama.cpp SYCL (two Arc Pro B70).
# Does not change host power. Pass the GGUF path; no private defaults.
#
#   bash benchmarks/ornith15-122b/launch-ornith-122b-iq3xxs.sh /path/to/Ornith-1.5-122B-UD-IQ3_XXS.gguf [CTX] [PORT]
set -euo pipefail

GGUF=${1:?usage: $0 GGUF_PATH [CTX] [PORT]}
CTX=${2:-32768}
PORT=${3:-8000}
IMAGE='ghcr.io/ggml-org/llama.cpp:server-intel-b11429@sha256:3353d968ed8987612b7859828d14d205413567baafed00a90d71d0c23e63d7ba'
CONTAINER="b70-ornith122-c${CTX}"

if [ ! -f "$GGUF" ]; then
  echo "GGUF not found: $GGUF" >&2
  exit 1
fi
SIZE=$(stat -c%s "$GGUF")
if [ "$SIZE" -lt 49000000000 ]; then
  echo "GGUF looks incomplete ($SIZE bytes); expected 49515667712." >&2
  exit 1
fi
if pgrep -af '[l]lama-server|[v]llm serve' >/dev/null; then
  echo 'Refusing to launch: another inference process is active.' >&2
  exit 1
fi
if [ -n "$(docker ps -q 2>/dev/null)" ]; then
  echo 'Refusing to launch: a container is already running.' >&2
  exit 1
fi

MODEL_DIR=$(cd "$(dirname "$GGUF")" && pwd)
MODEL_FILE=$(basename "$GGUF")

exec docker run --rm --name "$CONTAINER" \
  --device /dev/dri \
  -v /dev/dri:/dev/dri:ro \
  --group-add "$(stat -c '%g' /dev/dri/renderD128 2>/dev/null || echo 0)" \
  -e ZE_AFFINITY_MASK=0,1 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
  -p "${PORT}:8080" \
  -v "$MODEL_DIR":/models:ro \
  "$IMAGE" \
  -m "/models/${MODEL_FILE}" \
  --host 0.0.0.0 --port 8080 \
  -c "$CTX" -np 1 -ngl 99 \
  --split-mode layer --tensor-split 50,50 \
  -fa on \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  -b 512 -ub 512 -t 6 -tb 14 \
  --temp 0.6 --top-p 0.95 --top-k 20
