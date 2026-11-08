#!/usr/bin/env bash
# A/B TensorFold binaries on one card: same model, prompts, drafted greedy. Usage: tf_kernel_ab.sh OUT GPU name=bin ...
set -uo pipefail
OUT=$1; GPU=$2; shift 2
TF=$TF; PORT=$((8040+GPU))
HW=$(python3 host_facts.py hwmon "$GPU")
PROMPTS=$RUN/prompts-qwen38.json
mkdir -p "$OUT"
for round in ${ROUNDS:-1}; do
for nb in "$@"; do
  name=${nb%%=*}; bin=${nb#*=}
  echo "$(date -u +%FT%TZ) $name start $(sha256sum $bin | cut -c1-16)" >> "$OUT/ab.log"
  ZE_AFFINITY_MASK=$GPU ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE nohup "$bin" serve $TF/models/Qwen3.8-27B-MLX-4bit --drafter $TF/models/Qwen3.8-27B-DFlash2 \
    --parallel 1 --context 139264 --host 127.0.0.1 --port $PORT --prompt-cache-gib 0 --no-thinking --no-update-check > "$OUT/server-$name.log" 2>&1 < /dev/null &
  PID=$!
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; kill -0 $PID || break; sleep 3; done
  python3 $BENCH/depth_bench.py run --base http://127.0.0.1:$PORT --model qwen --prompts $PROMPTS --out "$OUT" \
    --hwmon $HW --cool ${COOL:-55} --depths ${DEPTHS:-0,32k,128k} --kinds ${KINDS:-code} --reps ${REPS:-1} --arm $name ${EXTRA:+--extra-json "$EXTRA"}
  kill $PID; sleep 5; kill -9 $PID 2>/dev/null
done
done
