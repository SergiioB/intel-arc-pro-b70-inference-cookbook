#!/usr/bin/env bash
# B70 MTP4 exact-128K nightly launcher — patched Qwen3.6 MoE serving.
#
# WHEN TO USE: reproduce the completed p130944/g128 MTP4 boundary.
# WHAT IT DOES: applies the BF16-draft and partial-final-group patches, then
#   launches the pinned vLLM XPU nightly at 131072 context and budget 8192.
# OUTPUT: Docker logs from container b70-mtp4-128k; HTTP endpoint on PORT.
# SAFETY: refuses a competing inference process; does not change host power.
#
# Example:
#   bash benchmarks/qwen36-35a3/launch-mtp4-128k-nightly.sh /path/to/model 8000
set -euo pipefail

MODEL_DIR=${1:?usage: $0 MODEL_DIR [PORT]}
PORT=${2:-8000}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMAGE='vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97'
MODEL='Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4'
SPEC=/tmp/b70-mtp4-128k-spec.json

if pgrep -af 'llama-server|llama-bench|vllm serve|vllm-xpu' >/dev/null; then
  echo 'Refusing to launch: another inference process is active.' >&2
  exit 1
fi
if [ ! -d "$MODEL_DIR" ]; then
  echo "Model directory not found: $MODEL_DIR" >&2
  exit 1
fi

printf '%s\n' '{"method":"mtp","num_speculative_tokens":4}' > "$SPEC"
sudo docker rm -f b70-mtp4-128k >/dev/null 2>&1 || true
RENDER_GID=$(stat -c '%g' /dev/dri/render* | sort -u | sed -n '1p')

sudo docker run -d --restart unless-stopped --name b70-mtp4-128k -p "$PORT:8000" \
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
  'set -e; python /patch_mtp.py; python /patch_boundary.py; SPEC=$(cat /spec.json); exec vllm serve /model --quantization gptq --dtype float16 --max-model-len 131072 --gpu-memory-utilization 0.85 --port 8000 --max-num-seqs 64 --max-num-batched-tokens 8192 --enable-prefix-caching --served-model-name Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4 --language-model-only --speculative-config "$SPEC"'

printf 'Container started. Follow startup:\n  sudo docker logs -f b70-mtp4-128k\n'
printf 'Health endpoint after startup:\n  curl -f http://127.0.0.1:%s/health\n' "$PORT"
