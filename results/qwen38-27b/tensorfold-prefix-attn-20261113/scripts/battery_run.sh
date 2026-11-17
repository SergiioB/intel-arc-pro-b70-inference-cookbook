#!/usr/bin/env bash
# Usage: battery_run.sh NAME GPU PORT PREF(0|CH) ARMS   (329 items, --parallel 1, C=1; not a timed run)
set -euo pipefail
NAME=$1; GPU=$2; PORT=$3; PREF=$4; ARMS=$5
R=$RUN; OUT=$R/$NAME; mkdir -p "$OUT"
TF=$TF; BIN=$TF/${SRC:-src-prefix}/zig-out/xpu/bin/tensorfold-native
HW=$(python3 $LAB/tools/host_facts.py hwmon "$GPU"); CAP0=$(cat $HW/power1_cap)
[ "$PREF" = 0 ] || export XPU_ATTN_PREF=$PREF
{ echo "utc $(date -u +%FT%TZ)"; echo "gpu $GPU hwmon $HW cap_uw $CAP0 temp_mc $(cat $HW/temp2_input) XPU_ATTN_PREF=${XPU_ATTN_PREF:-unset}"; sha256sum "$BIN"; } > "$OUT/env.txt"
ZE_AFFINITY_MASK=$GPU ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE nohup "$BIN" serve $TF/models/Qwen3.8-27B-MLX-4bit --drafter $TF/models/Qwen3.8-27B-DFlash2 \
  --parallel 1 --context 16384 --host 127.0.0.1 --port $PORT --prompt-cache-gib 0 --no-thinking --no-update-check > "$OUT/server.log" 2>&1 < /dev/null &
PID=$!
trap 'kill $PID 2>/dev/null; sleep 3; kill -9 $PID 2>/dev/null || true;
      echo "{\"utc\":\"$(date -u +%FT%TZ)\",\"cap_unchanged\":$([ "$(cat $HW/power1_cap)" = "$CAP0" ] && echo true || echo false)}" > "$OUT/cleanup.json"' EXIT
for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; kill -0 $PID || exit 1; sleep 5; done
python3 $R/battery.py http://127.0.0.1:$PORT "$OUT" "$ARMS"
