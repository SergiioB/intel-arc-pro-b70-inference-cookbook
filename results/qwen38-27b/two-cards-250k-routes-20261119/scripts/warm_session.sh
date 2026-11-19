#!/usr/bin/env bash
# Warm follow-up latency on TensorFold with a prompt cache sized to the session.
# Usage: warm_session.sh OUT_DIR GPU CTX CACHE_GIB KV(bf16|int8) DEPTHS("32k 64k") [TURNS=3]
set -eo pipefail
OUT=$1; GPU=$2; CTX=$3; CACHE=$4; KV=$5; DEPTHS=$6; TURNS=${7:-3}
TF=$TF
BIN=${TF_BIN:-$TF/src-kv8/zig-out/xpu/bin/tensorfold-native}
PORT=$((8036 + GPU * 10))
HW=$(python3 ~/B70-DOCS/tools/host_facts.py hwmon "$GPU")
PROMPTS=${PROMPTS:-~/B70-DOCS/results/2026/11/b70-breakthrough-20261118/prompts-warm.json}
mkdir -p "$OUT"
CAP0=$(cat $HW/power1_cap)
export XPU_ATTN_PREF=${XPU_ATTN_PREF-256}
[ "$KV" = int8 ] && export XPU_KV_INT8=1
ARGS=(serve $TF/models/Qwen3.8-27B-MLX-4bit --drafter $TF/models/Qwen3.8-27B-DFlash2 --parallel 1 --context $CTX
      --host 127.0.0.1 --port $PORT --prompt-cache-gib $CACHE --no-thinking --no-update-check)
{ echo "utc $(date -u +%FT%TZ) gpu $GPU hwmon $HW cap_uw $CAP0 temp_mc $(cat $HW/temp2_input)";
  echo "XPU_ATTN_PREF=${XPU_ATTN_PREF} XPU_KV_INT8=${XPU_KV_INT8-} ZE_AFFINITY_MASK=$GPU $BIN ${ARGS[*]}"; sha256sum "$BIN"; } > "$OUT/env.txt"
ZE_AFFINITY_MASK=$GPU ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE nohup "$BIN" "${ARGS[@]}" > "$OUT/server.log" 2>&1 < /dev/null &
PID=$!
trap 'kill $PID 2>/dev/null; sleep 3; kill -9 $PID 2>/dev/null || true; echo "{\"utc\":\"$(date -u +%FT%TZ)\",\"cap_unchanged\":$([ "$(cat $HW/power1_cap)" = "$CAP0" ] && echo true || echo false)}" > "$OUT/cleanup.json"' EXIT
for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "server exited"; tail -20 "$OUT/server.log"; exit 1; }; sleep 3; done
for d in $DEPTHS; do
  echo "== depth $d"
  python3 ~/B70-DOCS/scripts/bench/session_stability.py run --base http://127.0.0.1:$PORT --model qwen \
    --prompts $PROMPTS --depth $d --kind code --turns $TURNS --max-tokens 128 --hwmon $HW --out "$OUT/session-$d" || echo "session $d failed"
done
grep -E "prompt cache|done chat" "$OUT/server.log" | sed -E 's/chatcmpl-[0-9a-f]+ //' | cut -c1-200
