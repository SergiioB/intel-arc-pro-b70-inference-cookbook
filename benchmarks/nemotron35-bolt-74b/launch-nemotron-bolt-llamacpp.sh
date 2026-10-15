#!/usr/bin/env bash
# Launch Nemotron 3.5 Bolt 74B-A9B (NVIDIA 24GB GGUF + mmproj) on one Intel
# Arc Pro B70 with the patched llama.cpp SYCL image.
#
# WHEN TO USE: serve the recipe in docs/nemotron35-bolt-74b/NEMOTRON-BOLT-B70-LLAMACPP.md
#   (C1, 64K context, no drafter).
# WHAT IT DOES: runs llama-server from the image with the measured flags and
#   GGML_SYCL_GROUPED_MAX_ROWS=0. The cookbook patch is already in the image.
# OUTPUT: OpenAI-compatible endpoint on PORT; logs via `docker logs -f b70-nemotron-bolt`.
# SAFETY: refuses another inference process; does not change host power.
#
#   bash benchmarks/nemotron35-bolt-74b/launch-nemotron-bolt-llamacpp.sh /path/to/model-dir 8080
#   GPU=1 bash ...                                   # second card
#   IMAGE=llama.cpp-sycl-b70:b11669-bolt bash ...    # your own build of the same patch
set -euo pipefail

MODEL_DIR=${1:?usage: $0 MODEL_DIR [PORT]}
PORT=${2:-8080}
GPU=${GPU:-0}
IMAGE=${IMAGE:-ghcr.io/sergiiob/llama.cpp-sycl-b70:b11669-bolt@sha256:c1b871ee87d2790e92605e89d8fa40c41a61afa128bdce55da0f9fba22e3d042}
GGUF='NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf'
MMPROJ='mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf'
CONTAINER='b70-nemotron-bolt'

for file in "$GGUF" "$MMPROJ"; do
  if [ ! -f "$MODEL_DIR/$file" ]; then
    echo "Missing $MODEL_DIR/$file (download it from nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF)" >&2
    exit 1
  fi
done
if pgrep -af '[l]lama-server|[l]lama-bench|[v]llm serve|[v]llm.entrypoints' >/dev/null; then
  echo 'Refusing to launch: another inference process is active.' >&2
  exit 1
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
RENDER_GID=$(stat -c '%g' /dev/dri/render* | sort -u | sed -n '1p')

docker run -d --restart unless-stopped --name "$CONTAINER" -p "$PORT:8080" \
  --device /dev/dri --group-add "$RENDER_GID" \
  -v /dev/dri:/dev/dri:ro \
  -v "$MODEL_DIR:/model:ro" \
  -e GGML_SYCL_GROUPED_MAX_ROWS=0 \
  -e ONEAPI_DEVICE_SELECTOR="level_zero:$GPU" \
  "$IMAGE" \
  -m "/model/$GGUF" --mmproj "/model/$MMPROJ" \
  -ngl 999 --fit off -c 65536 -np 1 -b 8192 -ub 4096 \
  --flash-attn on -ctk q8_0 -ctv q4_1 -t 8 --jinja --metrics \
  --host 0.0.0.0 --port 8080

cat <<EOF
Container: $CONTAINER (image $IMAGE, card $GPU)
Health:    curl -f http://127.0.0.1:$PORT/health
Logs:      docker logs -f $CONTAINER
Stop:      docker rm -f $CONTAINER
EOF
