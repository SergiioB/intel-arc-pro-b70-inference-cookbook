#!/usr/bin/env bash
# TensorFold Qwen3.8-27B MLX-4bit + DFlash2 depth campaign on one B70.
# Usage: tf_depth_run.sh OUT_DIR [GPU_INDEX=1]
# Arms on one server: drafted -> serial ("draft": false) -> drafted2. C=1, greedy, thinking off.
set -euo pipefail
OUT=$1; GPU=${2:-1}
TF=/data/sata1/work/tensorfold-20261104
BIN=${TF_BIN:-$TF/src-v1.2.4/zig-out/xpu/bin/tensorfold-native}
MODEL=${TF_MODEL:-$TF/models/Qwen3.8-27B-MLX-4bit}
DRAFT=$TF/models/Qwen3.8-27B-DFlash2
PORT=8036
HW=$(python3 /home/sergio/B70-DOCS/tools/host_facts.py hwmon "$GPU")
PROMPTS=/home/sergio/B70-DOCS/results/2026/11/b70-tonight-20261108/prompts-qwen38.json
mkdir -p "$OUT"
[ -z "$(docker ps -q)" ] || { echo "containers running"; docker ps; exit 1; }
! pgrep -f 'tensorfold-native|llama-server|VLLM::EngineCore' >/dev/null || { echo "inference proc running"; exit 1; }
CAP0=$(cat $HW/power1_cap)
{ echo "utc $(date -u +%FT%TZ)"; echo "gpu $GPU hwmon $HW"; echo "tensorfold $(git -C $(dirname $(dirname $(dirname $(dirname $BIN)))) describe --tags) + local-patch.diff";
  echo "model $MODEL"; echo "drafter $DRAFT"; echo "cap_uw $CAP0 temp_mc $(cat $HW/temp2_input)"; sha256sum "$BIN" "$PROMPTS"; } > "$OUT/env.txt"
ARGS=(serve "$MODEL" --drafter "$DRAFT" --parallel 1 --context ${TF_CTX:-139264} --host 127.0.0.1 --port $PORT
      --prompt-cache-gib 0 --no-thinking --no-update-check)
echo "ZE_AFFINITY_MASK=$GPU ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE $BIN ${ARGS[*]}" > "$OUT/launch.txt"
ZE_AFFINITY_MASK=$GPU ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE nohup "$BIN" "${ARGS[@]}" > "$OUT/server.log" 2>&1 < /dev/null &
PID=$!
trap 'kill $PID 2>/dev/null; sleep 3; kill -9 $PID 2>/dev/null || true;
      echo "{\"utc\":\"$(date -u +%FT%TZ)\",\"cap_unchanged\":$([ "$(cat $HW/power1_cap)" = "$CAP0" ] && echo true || echo false)}" > "$OUT/cleanup.json"' EXIT
for i in $(seq 1 240); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "server exited"; tail -20 "$OUT/server.log"; exit 1; }
  sleep 5
done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "startup timeout"; exit 1; }
B="python3 /home/sergio/B70-DOCS/scripts/bench/depth_bench.py run --base http://127.0.0.1:$PORT --model qwen --prompts $PROMPTS --out $OUT --hwmon $HW --cool ${COOL:-55} --depths ${DEPTHS:-0,8k,32k,128k} --reps ${REPS:-2}"
$B --arm drafted
$B --arm serial --extra-json '{"draft": false}'
$B --arm drafted2
if [ -n "${CANARY:-1}" ]; then
  python3 /home/sergio/B70-DOCS/scripts/tmp/exl3xpu_mixedk_20261012/tests/canary.py --base http://127.0.0.1:$PORT --model qwen \
    --prompts /home/sergio/B70-DOCS/scripts/tmp/exl3xpu_mixedk_20261012/tests/prompts_decoded.json > "$OUT/canary.txt" 2>&1 || true
fi
