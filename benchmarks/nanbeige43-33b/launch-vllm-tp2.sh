#!/usr/bin/env bash
# Launch Nanbeige4.3-33B (compressed-tensors W4A16 g128) on two Arc Pro B70 cards,
# vLLM-XPU 0.31.0, tensor parallel 2.
#
# WHEN TO USE: you need more than the one-card KV ceiling (46,848 tokens at fp8 KV
#   and 0.92 utilisation) and can pay the TP2 collective cost.
# WHAT IT DOES: pinned image + patches/patch_vllm_worker_affinity.py, oneCCL
#   thresholds pinned (docs/DUAL-B70-TP2.md), `vllm serve ... --tensor-parallel-size 2`.
# OUTPUT: OpenAI-compatible endpoint on 127.0.0.1:PORT, logs via `docker logs`.
# SAFETY: refuses another inference process or container; never changes host power caps.
#
#   bash benchmarks/nanbeige43-33b/launch-vllm-tp2.sh /path/to/w4a16-dir 8110
#   MAX_LEN=131072 bash benchmarks/nanbeige43-33b/launch-vllm-tp2.sh /path/to/w4a16-dir
set -euo pipefail

MODEL_DIR=${1:?usage: $0 MODEL_DIR [PORT]}
PORT=${2:-8110}
MAX_LEN=${MAX_LEN:-32768}
UTIL=${UTIL:-0.92}
IMAGE=${IMAGE:-vllm/vllm-openai-xpu@sha256:95ac815038c4b3537b173798bfe3b89bdf525a52bd653cd14d4deb01f1062be7}
CONTAINER=${CONTAINER:-nb-vllm-tp2}
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ENTRY="$REPO_ROOT/benchmarks/nanbeige43-33b/vllm-tp2-entry.sh"

[ -f "$MODEL_DIR/config.json" ] || { echo "Missing $MODEL_DIR/config.json" >&2; exit 1; }
for file in "$ENTRY" "$REPO_ROOT/patches/patch_vllm_worker_affinity.py"; do
  [ -f "$file" ] || { echo "Missing $file" >&2; exit 1; }
done
if pgrep -af '[l]lama-server|[l]lama-bench|[v]llm serve|[v]llm.entrypoints' >/dev/null; then
  echo 'Refusing to launch: another inference process is active.' >&2
  exit 1
fi
if docker ps --format '{{.Names}} {{.Image}}' | grep -v "^$CONTAINER " | grep -Eqi 'vllm|llama|sglang|exl3|model_server'; then
  echo 'Refusing to launch: another inference container is running.' >&2
  exit 1
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
RENDER_GID=$(stat -c '%g' /dev/dri/render* | sort -u | sed -n '1p')

docker run -d --name "$CONTAINER" --network host \
  --device /dev/dri --group-add "$RENDER_GID" \
  --ipc=host --shm-size 8g --cap-add SYS_PTRACE \
  -v "$MODEL_DIR:/model:ro" \
  -v "$REPO_ROOT/patches:/patches:ro" \
  -v "$ENTRY:/nb-tp2-entry.sh:ro" \
  -e ZE_AFFINITY_MASK=0,1 \
  -e B70_WORKER_AFFINITY=1 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296 \
  -e CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296 \
  -e CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296 \
  -e CCL_SYCL_ALLTOALL_TMP_BUF=1 \
  -e CCL_KVS_IFACE=lo -e FI_TCP_IFACE=lo -e VLLM_HOST_IP=127.0.0.1 \
  -e HF_HUB_OFFLINE=1 \
  --entrypoint bash "$IMAGE" /nb-tp2-entry.sh serve /model \
  --served-model-name nanbeige43-33b --host 127.0.0.1 --port "$PORT" \
  --trust-remote-code --tensor-parallel-size 2 --dtype bfloat16 \
  --kv-cache-dtype fp8 --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$UTIL" --enable-prefix-caching

cat <<EOF
Container: $CONTAINER (image $IMAGE, cards 0+1, TP2, max length $MAX_LEN)
Ready:     "[B70_WORKER_AFFINITY] ... visible=1" on both ranks, then the API starts
Health:    curl -f http://127.0.0.1:$PORT/health
Logs:      docker logs -f $CONTAINER
Stop:      docker rm -f $CONTAINER
EOF
