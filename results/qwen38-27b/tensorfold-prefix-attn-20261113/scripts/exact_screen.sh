#!/usr/bin/env bash
# Usage: exact_screen.sh NAME GPU PORT DEPTHS REPS [PREF]   one-exp exact attn_decode build (src-exact); no process guard
set -euo pipefail
NAME=$1; GPU=$2; PORT=$3; DEPTHS=$4; REPS=$5; PREF=${6:-0}
R=$RUN; OUT=$R/$NAME; mkdir -p "$OUT"
TF=$TF; BIN=$TF/${SRC:-src-exact}/zig-out/xpu/bin/tensorfold-native
HW=$(python3 $LAB/tools/host_facts.py hwmon "$GPU"); CAP0=$(cat $HW/power1_cap)
PROMPTS=${PROMPTS:-$PROMPTS/prompts-qwen38.json}
[ "$PREF" = 0 ] || export XPU_ATTN_PREF=$PREF
{ echo "utc $(date -u +%FT%TZ)"; echo "gpu $GPU hwmon $HW cap_uw $CAP0 temp_mc $(cat $HW/temp2_input) XPU_ATTN_PREF=${XPU_ATTN_PREF:-unset} XPU_KV_INT8=${XPU_KV_INT8:-unset} context=${TF_CTX:-139264}"; sha256sum "$BIN" "$PROMPTS"; } > "$OUT/env.txt"
ZE_AFFINITY_MASK=$GPU ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE nohup "$BIN" serve $TF/models/Qwen3.8-27B-MLX-4bit --drafter $TF/models/Qwen3.8-27B-DFlash2 \
  --parallel 1 --context ${TF_CTX:-139264} --host 127.0.0.1 --port $PORT --prompt-cache-gib 0 --no-thinking --no-update-check > "$OUT/server.log" 2>&1 < /dev/null &
PID=$!
trap 'kill $PID 2>/dev/null; sleep 3; kill -9 $PID 2>/dev/null || true;
      echo "{\"utc\":\"$(date -u +%FT%TZ)\",\"cap_unchanged\":$([ "$(cat $HW/power1_cap)" = "$CAP0" ] && echo true || echo false)}" > "$OUT/cleanup.json"' EXIT
for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; kill -0 $PID || exit 1; sleep 5; done
B="python3 $BENCH/depth_bench.py run --base http://127.0.0.1:$PORT --model qwen --prompts $PROMPTS --out $OUT --hwmon $HW --cool ${COOL:-60} --depths $DEPTHS --reps $REPS ${KINDS:+--kinds $KINDS}"
$B --arm drafted
$B --arm serial --extra-json '{"draft": false}'
