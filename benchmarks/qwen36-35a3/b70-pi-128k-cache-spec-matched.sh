#!/usr/bin/env bash
# B70 Pi 128K cache/spec matrix — eight matched real-world serving cells.
#
# WHEN TO USE: reproduce no-spec/MTP1/MTP2/MTP4 with prefix caching on/off.
# WHAT IT DOES: generates exact rendered prompts, starts eight clean servers,
#   records five cold p130944/g128 requests and five changed resident-session
#   follow-ups per cell, and keeps raw SSE, metrics, logs, and manifests.
# OUTPUT: results/vllm-pi-128k-cache-spec-matched-*/.
# SAFETY: one inference server at a time; does not change host power.
#
# Example:
#   bash benchmarks/qwen36-35a3/b70-pi-128k-cache-spec-matched.sh /path/to/model
set -euo pipefail

MODEL_DIR=${1:?usage: $0 MODEL_DIR}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_ID="vllm-pi-128k-cache-spec-matched-$(date -u +%Y%m%dT%H%M%SZ)-$$"
OUT="$ROOT/results/$RUN_ID"
MODEL=Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4
IMAGE='vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97'
LAUNCHER=$ROOT/benchmarks/qwen36-35a3/launch-vllm-128k-mode.sh
GEN=$ROOT/benchmarks/b70-generate-exact-prompts.py
COLD_HARNESS=$ROOT/benchmarks/b70-realworld-context-harness.py
SESSION_HARNESS=$ROOT/benchmarks/b70-resident-session-harness.py
MON=$ROOT/benchmarks/b70-sync-monitor.py
PI_SYSTEM=$ROOT/benchmarks/pi-system-prompt.txt
COLD_PROMPTS=$OUT/exact-pi-p130944-six.json
SESSION_PROMPTS=$OUT/exact-pi-p120000-one.json
PORT=8001
CURRENT_CONTAINER=
CURRENT_MONITOR_PID=
CURRENT_LOG_PID=
mkdir -p "$OUT"

stop_cell() {
  set +e
  if [ -n "$CURRENT_MONITOR_PID" ]; then
    kill -TERM "$CURRENT_MONITOR_PID" 2>/dev/null
    wait "$CURRENT_MONITOR_PID" 2>/dev/null
    CURRENT_MONITOR_PID=
  fi
  if [ -n "$CURRENT_CONTAINER" ]; then
    sudo docker rm -f "$CURRENT_CONTAINER" >/dev/null 2>&1
    CURRENT_CONTAINER=
  fi
  if [ -n "$CURRENT_LOG_PID" ]; then
    kill -TERM "$CURRENT_LOG_PID" 2>/dev/null
    wait "$CURRENT_LOG_PID" 2>/dev/null
    CURRENT_LOG_PID=
  fi
  set -e
}
trap stop_cell EXIT INT TERM

if [ ! -d "$MODEL_DIR" ]; then
  echo "Model directory not found: $MODEL_DIR" >&2
  exit 1
fi
if pgrep -af '[l]lama-server|[l]lama-bench|[v]llm serve|[v]llm-xpu' >/dev/null; then
  echo 'Refusing to benchmark: another inference process is active.' >&2
  exit 1
fi

printf 'milestone=tokenizer_start target=130944\n'
sudo docker run --rm \
  -v "$MODEL_DIR:/model:ro" -v "$GEN:/generate.py:ro" \
  -v "$PI_SYSTEM:/pi-system.txt:ro" -v "$OUT:/out" \
  --entrypoint python "$IMAGE" /generate.py \
  --model /model --system-prompt-file /pi-system.txt \
  --output "/out/$(basename "$COLD_PROMPTS")" --targets 130944 --per-target 6
printf 'milestone=tokenizer_start target=120000\n'
sudo docker run --rm \
  -v "$MODEL_DIR:/model:ro" -v "$GEN:/generate.py:ro" \
  -v "$PI_SYSTEM:/pi-system.txt:ro" -v "$OUT:/out" \
  --entrypoint python "$IMAGE" /generate.py \
  --model /model --system-prompt-file /pi-system.txt \
  --output "/out/$(basename "$SESSION_PROMPTS")" --targets 120000 --per-target 1
printf 'milestone=tokenizer_done\n'

{
  echo "run_id=$RUN_ID"
  echo 'status=PROVISIONAL_SELF_REPORTED_NOT_INDEPENDENTLY_REPRODUCED'
  echo 'modes=no-spec,mtp1,mtp2,mtp4'
  echo 'cache_settings=enabled,disabled'
  echo 'cold=p130944/g128/n5'
  echo 'resident=p120000 preparation plus changed p~120148/g128/n5 followups'
  echo 'scheduler_budget=8192'
  echo 'max_model_len=131072'
  echo 'gpu_memory_utilization=0.85'
  echo "image=$IMAGE"
  echo "model=$MODEL_DIR"
  echo "mtp_patch_sha256=$(sha256sum "$ROOT/patches/patch_mtp_nightly.py" | cut -d' ' -f1)"
  echo "boundary_patch_sha256=$(sha256sum "$ROOT/patches/patch_mtp_boundary.py" | cut -d' ' -f1)"
  echo "cold_prompt_sha256=$(sha256sum "$COLD_PROMPTS" | cut -d' ' -f1)"
  echo "session_prompt_sha256=$(sha256sum "$SESSION_PROMPTS" | cut -d' ' -f1)"
} > "$OUT/manifest.txt"

run_cell() {
  local mode=$1
  local cache=$2
  local cell=$OUT/cache-$cache/$mode
  local no_spec_arg=
  local session_cache_arg=
  mkdir -p "$cell"
  [ "$mode" = no-spec ] && no_spec_arg=--no-spec
  [ "$cache" = on ] && session_cache_arg=--cache-enabled

  stop_cell
  printf 'milestone=launch_start mode=%s cache=%s\n' "$mode" "$cache"
  bash "$LAUNCHER" "$MODEL_DIR" "$mode" "$cache" "$PORT" > "$cell/launch.log"
  CURRENT_CONTAINER="b70-${mode}-cache-${cache}"
  sudo docker logs -f "$CURRENT_CONTAINER" > "$cell/server.log" 2>&1 &
  CURRENT_LOG_PID=$!

  for attempt in $(seq 1 120); do
    sleep 5
    if curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      break
    fi
    sudo docker inspect -f '{{.State.Running}}' "$CURRENT_CONTAINER" 2>/dev/null |
      grep -q true || { echo "container stopped: $CURRENT_CONTAINER" >&2; return 1; }
    [ "$attempt" -lt 120 ] || { echo "readiness timeout: $CURRENT_CONTAINER" >&2; return 1; }
  done

  local capacity
  capacity=$(sed -n 's/.*GPU KV cache size: \([0-9,]*\) tokens.*/\1/p' \
    "$cell/server.log" | sed -n '$p' | tr -d ',')
  [ -n "$capacity" ] && [ "$capacity" -ge 131072 ] || {
    echo "KV capacity below exact 128K: ${capacity:-unknown}" >&2
    return 2
  }
  {
    echo "mode=$mode"
    echo "cache=$cache"
    echo "logged_kv_capacity_tokens=$capacity"
  } > "$cell/manifest.txt"
  if [ -r /sys/kernel/debug/dri/0000:0b:00.0/tile0/vram_mm ]; then
    sudo cat /sys/kernel/debug/dri/0000:0b:00.0/tile0/vram_mm > "$cell/vram-after-load.txt"
  fi
  printf 'milestone=launch_ready mode=%s cache=%s capacity=%s\n' "$mode" "$cache" "$capacity"

  python3 "$MON" "$cell/monitor.jsonl" 0.5 &
  CURRENT_MONITOR_PID=$!
  sleep 1
  python3 "$COLD_HARNESS" --mode context --prompts "$COLD_PROMPTS" \
    --target 130944 --output 128 --budget 8192 --reps 5 --model "$MODEL" \
    --root "http://127.0.0.1:$PORT" --outdir "$cell/cold" $no_spec_arg \
    2>&1 | tee "$cell/cold-harness.log"
  python3 "$SESSION_HARNESS" --request-module "$COLD_HARNESS" \
    --prompts "$SESSION_PROMPTS" --target 120000 --output 128 --reps 5 \
    --model "$MODEL" --root "http://127.0.0.1:$PORT" \
    --outdir "$cell/resident" $session_cache_arg $no_spec_arg \
    2>&1 | tee "$cell/resident-harness.log"

  kill -TERM "$CURRENT_MONITOR_PID" 2>/dev/null || true
  wait "$CURRENT_MONITOR_PID" || true
  CURRENT_MONITOR_PID=
  curl -fsS "http://127.0.0.1:$PORT/metrics" > "$cell/metrics-after.txt"
  echo 'result=completed' >> "$cell/manifest.txt"
  printf 'milestone=cell_completed mode=%s cache=%s\n' "$mode" "$cache"
  stop_cell
}

run_cell no-spec on
run_cell no-spec off
run_cell mtp1 off
run_cell mtp1 on
run_cell mtp2 on
run_cell mtp2 off
run_cell mtp4 off
run_cell mtp4 on

echo 'result=all_eight_cells_completed' >> "$OUT/manifest.txt"
echo "completed_utc=$(date -u +%FT%TZ)" >> "$OUT/manifest.txt"
printf 'milestone=campaign_completed run=%s\n%s\n' "$RUN_ID" "$OUT"
