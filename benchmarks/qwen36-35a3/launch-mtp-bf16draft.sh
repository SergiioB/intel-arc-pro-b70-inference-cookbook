#!/usr/bin/env bash
# B70 vLLM MTP server launch — native INT4 target + BF16-unquantized MTP draft.
# Self-contained: uses the patches from this repo (../patches/).
#
# Usage: bash benchmarks/qwen36-35a3/launch-mtp-bf16draft.sh /path/to/model [PORT]
#
# Prereqs: Arc Pro B60/B70, Docker, B70 drivers + oneAPI runtime, the
# MTP-preserved GPTQ model from llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4
set -euo pipefail

printf '%s\n' \
  'RETIRED: this script requires the unpublished local intel/vllm:0.21.0-xpu-int4moe image.' \
  'Use: bash benchmarks/qwen36-35a3/launch-mtp4-128k-nightly.sh /path/to/model 8000' >&2
exit 2

MODEL_DIR="${1:-/mnt/models/Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4}"
PORT="${2:-8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PATCH_V4="${REPO_ROOT}/patches/patch_xpu_int4_moe_v4.py"
PATCH_MTP="${REPO_ROOT}/patches/patch_mtp_bf16_draft.py"
SPEC_FILE=/tmp/b70-spec-mtp.json
LOG="${LOG:-/tmp/vllm-mtp-serve.log}"

printf '%s\n' '{"method":"mtp","num_speculative_tokens":1}' > "$SPEC_FILE"

# optional: set power cap to MoE sweet spot (150W). Uncomment to enable:
# echo 150000000 | sudo tee /sys/class/hwmon/hwmon4/power1_cap >/dev/null

RENDER_GID=$(stat -c "%g" /dev/dri/render* | head -n1)
: > "$LOG"

CID=$(sudo docker run -d --restart unless-stopped --name b70vllm -p ${PORT}:8000 \
  --device /dev/dri --group-add "${RENDER_GID}" \
  -v /dev/dri:/dev/dri:ro \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${PATCH_V4}:/patch_v4.py:ro" \
  -v "${PATCH_MTP}:/patch_mtp.py:ro" \
  -v "${SPEC_FILE}:/spec.json:ro" \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash intel/vllm:0.21.0-xpu-int4moe \
  -lc 'set -e; python /patch_v4.py; python /patch_mtp.py; SPEC=$(cat /spec.json); echo SPEC=$SPEC; exec vllm serve /model --quantization gptq --dtype float16 --max-model-len 16384 --gpu-memory-utilization 0.92 --port 8000 --cudagraph-capture-sizes 1 2 4 8 16 32 --max-num-seqs 1 --max-num-batched-tokens 8192 --served-model-name Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4 --language-model-only --speculative-config "$SPEC"')

echo "CID=$CID"
echo "Logs: $LOG  (tail -f $LOG)"
echo "Wait for 'Application startup complete', then test:"
echo "  curl http://localhost:${PORT}/v1/models"

sudo docker logs -f b70vllm >>"$LOG" 2>&1 &
LPID=$!
for i in $(seq 1 55); do
  sleep 15
  if curl -s -m 3 http://127.0.0.1:${PORT}/v1/models 2>/dev/null | grep -q '"id"'; then
    echo "UP after $((i*15))s — MTP spec decode active. Look for '[B70] GDN XPU: spec decode active' in logs."
    exit 0
  fi
  ST=$(sudo docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' b70vllm 2>/dev/null || echo gone)
  echo "t=$((i*15))s status=$ST"
  if [[ "$ST" == exited* ]] || [[ "$ST" == gone* ]]; then
    echo "FAILED — last 30 log lines:"
    tail -30 "$LOG"
    exit 1
  fi
done
echo "TIMEOUT"
tail -30 "$LOG"
exit 1
