#!/usr/bin/env bash
# exl3xpu (EXL3 4.00bpw, Qwen3.8-27B) depth campaign on one B70, same cells as tf_depth_run.sh.
# Usage: exl3_depth_run.sh OUT_DIR ARM(serial|mtp3) [GPU_INDEX=1]
set -euo pipefail
OUT=$1; ARM=$2; GPU=${3:-1}
IMG=ghcr.io/sergiiob/exl3xpu-mixedk@sha256:933eda54bb51a2e8a6669a70f5a2d7696fabd12a5f8b7c469df02fdfecc79d3a
MODEL=${B70_MODEL:-$MODELS/b70-exl3-qwen38-4bpw-20260926}
NAME=b70-depth-exl3-$ARM; PORT=8034
HW=$(python3 host_facts.py hwmon "$GPU")
PROMPTS=$RUN/prompts-qwen38.json
mkdir -p "$OUT"
[ -z "$(docker ps -q)" ] || { echo "containers running"; docker ps; exit 1; }
! pgrep -f 'tensorfold-native serve .*--port 8036|VLLM::EngineCore' >/dev/null || { echo "inference proc running"; exit 1; }
CAP0=$(cat $HW/power1_cap)
trap 'docker logs $NAME > "$OUT/server-$ARM.log" 2>&1 || true; docker rm -f $NAME >/dev/null 2>&1 || true;
      echo "{\"utc\":\"$(date -u +%FT%TZ)\",\"cap_unchanged\":$([ "$(cat $HW/power1_cap)" = "$CAP0" ] && echo true || echo false)}" > "$OUT/cleanup-$ARM.json"' EXIT
if [ "$ARM" = serial ]; then SPEC=(--set vllm.speculative_config=null --set env.EXL3_DRAFT_VOCAB=null); else SPEC=(); fi
ARGS=(docker run -d --name $NAME --network=host --device=/dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro --shm-size=8g
  -v $MODEL:/mnt/exl3model:ro -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e UR_L0_USE_COPY_ENGINE=0 -e SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0
  $IMG models/qwen3.8-27b-exl3-4.00bpw --gpu $GPU --port $PORT --model-path /mnt/exl3model
  --set vllm.gpu_memory_utilization=0.92 --set vllm.max_model_len=139264 --set vllm.enable_prefix_caching=false
  "${SPEC[@]}" --set vllm.reasoning_parser=qwen3)
printf '%q ' "${ARGS[@]}" > "$OUT/launch-$ARM.txt"; echo >> "$OUT/launch-$ARM.txt"
echo "{\"utc\":\"$(date -u +%FT%TZ)\",\"gpu\":$GPU,\"hwmon\":\"$HW\",\"cap_uw\":$CAP0,\"temp_mc\":$(cat $HW/temp2_input),\"model\":\"$MODEL\",\"image\":\"$IMG\"}" > "$OUT/env-$ARM.json"
"${ARGS[@]}"
for i in $(seq 1 240); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break
  [ "$(docker inspect -f '{{.State.Running}}' $NAME)" = true ] || { echo "server exited"; exit 1; }; sleep 5; done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "startup timeout"; exit 1; }
MODEL_ID=$(curl -s http://127.0.0.1:$PORT/v1/models | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
python3 $BENCH/depth_bench.py run --base http://127.0.0.1:$PORT --model "$MODEL_ID" --prompts $PROMPTS \
  --out $OUT --hwmon $HW --cool ${COOL:-55} --depths ${DEPTHS:-0,8k,32k,128k} --reps ${REPS:-2} --arm exl3-$ARM
