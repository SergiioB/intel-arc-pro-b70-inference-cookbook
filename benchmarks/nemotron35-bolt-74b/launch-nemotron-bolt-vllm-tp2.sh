#!/usr/bin/env bash
# Launch Nemotron 3.5 Bolt 74B-A9B (AutoRound W4A16, int4, group 128) on two
# Intel Arc Pro B70 cards with vLLM XPU 0.31.0, tensor parallel 2.
#
# WHEN TO USE: serve the recipe in docs/nemotron35-bolt-74b/VLLM-TP2-AUTOROUND-W4A16.md
#   (TP2, 65536 context by default, up to 4 sequences).
# WHAT IT DOES: runs the pinned vLLM XPU image with tp2-entry.sh as entrypoint.
#   The wrapper applies patches/patch_vllm_worker_affinity.py and
#   patches/patch_xpu_moe_topk_pad.py, checks both, then starts `vllm serve`.
#   oneCCL is pinned to loopback (CCL_KVS_IFACE=lo, FI_TCP_IFACE=lo).
# OUTPUT: OpenAI-compatible endpoint on 127.0.0.1:PORT (host network, no API key);
#   logs via `docker logs -f b70-nemotron-bolt-tp2`.
# SAFETY: refuses another inference process or container; does not change host power.
#   Load time depends on disk; about 12-15 min on our USB SSD.
#
#   bash benchmarks/nemotron35-bolt-74b/launch-nemotron-bolt-vllm-tp2.sh /path/to/model-dir 8000
#   MAX_LEN=262144 bash ...    # model maximum (opt-in: a 250K prompt takes about 35 min to read)
set -euo pipefail

MODEL_DIR=${1:?usage: $0 MODEL_DIR [PORT]}
PORT=${2:-8000}
MAX_LEN=${MAX_LEN:-65536}
IMAGE=${IMAGE:-vllm/vllm-openai-xpu@sha256:95ac815038c4b3537b173798bfe3b89bdf525a52bd653cd14d4deb01f1062be7}
CONTAINER='b70-nemotron-bolt-tp2'
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ENTRY="$REPO_ROOT/benchmarks/nemotron35-bolt-74b/tp2-entry.sh"

if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "Missing $MODEL_DIR/config.json (download SergiioB/Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym)" >&2
  exit 1
fi
for file in "$ENTRY" "$REPO_ROOT/patches/patch_vllm_worker_affinity.py" "$REPO_ROOT/patches/patch_xpu_moe_topk_pad.py"; do
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
  -v "$ENTRY:/tp2-entry.sh:ro" \
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
  --entrypoint bash "$IMAGE" /tp2-entry.sh serve /model \
  --served-model-name nemotron-bolt --host 127.0.0.1 --port "$PORT" \
  --trust-remote-code --tensor-parallel-size 2 --dtype bfloat16 \
  --kv-cache-dtype fp8 --attention-backend TRITON_ATTN \
  --max-model-len "$MAX_LEN" --max-num-batched-tokens 8192 --block-size 64 \
  --max-num-seqs 4 --gpu-memory-utilization 0.85 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --override-generation-config '{"temperature":1.0,"top_p":0.95}' \
  --enable-prefix-caching --mamba-cache-mode align

cat <<EOF
Container: $CONTAINER (image $IMAGE, cards 0+1, TP2, max length $MAX_LEN)
Ready:     log shows 'Using XPUExpertsWNA16' on both ranks, then the API starts
Health:    curl -f http://127.0.0.1:$PORT/health
Logs:      docker logs -f $CONTAINER
Stop:      docker rm -f $CONTAINER
EOF
