#!/usr/bin/env bash
# B70 Pi prefill/decode matrix — separate exact prefill, decode length, and full-context cells.
#
# WHEN TO USE: publish clear vLLM prefill and g32/g128/g256/g512 measurements.
# WHAT IT DOES: starts no-spec/MTP1/MTP2/MTP4 servers with production cache enabled,
#   generates distinct entropy-first exact prompts per coordinate, and records five C1 samples.
# OUTPUT: results/vllm-pi-prefill-decode-matrix-*/ with SSE, logs, telemetry, and manifests.
# SAFETY: one server only, >=500 MiB free after load, cooldown <=55C,
#   leaves host power unchanged, and stops on the first invalid cell.
#
# Example:
#   bash benchmarks/b70-pi-prefill-decode-matrix.sh /path/to/model
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_ID="vllm-pi-prefill-decode-matrix-$(date -u +%Y%m%dT%H%M%SZ)-$$"
OUT="$ROOT/results/$RUN_ID"
MODEL_DIR=${1:?usage: $0 MODEL_DIR}
MODEL=$(basename "$MODEL_DIR")
MODEL_ID=${MODEL_ID:-$MODEL}
CONFIGURED_CAP_W=${CONFIGURED_CAP_W:-unknown}
IMG=vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97
PATCH_MTP=$ROOT/patches/patch_mtp_nightly.py
PATCH_BOUNDARY=$ROOT/patches/patch_mtp_boundary.py
GEN=$ROOT/benchmarks/b70-generate-exact-prompts.py
HARNESS=$ROOT/benchmarks/b70-realworld-context-harness.py
MON=$ROOT/benchmarks/b70-sync-monitor.py
PI_SYSTEM=$ROOT/benchmarks/pi-system-prompt.txt
STANDARD_SYSTEM=$ROOT/benchmarks/benchmark-system-prompt.txt
CONTAINER=b70phases
PORT=8001
CURRENT_MONITOR_PID=
CURRENT_LOG_PID=
mkdir -p "$OUT/prompts"

# name|prompt tokens|requested output|class
COORDINATES=(
  'prefill-p512|512|1|prefill'
  'prefill-p2048|2048|1|prefill'
  'prefill-p4096|4096|1|prefill'
  'prefill-p6144|6144|1|prefill'
  'prefill-p8192|8192|1|prefill'
  'decode-p512-g32|512|32|decode_small_prompt'
  'decode-p512-g128|512|128|decode_small_prompt'
  'decode-p512-g256|512|256|decode_small_prompt'
  'decode-p512-g512|512|512|decode_small_prompt'
  'decode-p8192-g32|8192|32|decode'
  'decode-p8192-g128|8192|128|decode'
  'decode-control-p9445-g128|9445|128|decode_historical_control'
  'decode-p8192-g256|8192|256|decode'
  'decode-p8192-g512|8192|512|decode'
  'prefill-full-p131071|131071|1|prefill_full_context'
  'decode-full-p130944-g128|130944|128|decode_full_context'
  'decode-full-p130560-g512|130560|512|decode_full_context'
)

find_hwmon() {
  python3 - <<'PY'
import glob, os
matches = [path for path in glob.glob('/sys/class/hwmon/hwmon*')
           if open(path + '/name').read().strip() == 'xe'
           and '0000:0b:00.0' in os.path.realpath(path + '/device')]
assert len(matches) == 1, matches
print(matches[0])
PY
}

HWMON=$(find_hwmon)
TEMP=$HWMON/temp2_input

visible_free() {
  sudo -n grep visible_avail /sys/kernel/debug/dri/0000:0b:00.0/tile0/vram_mm |
    grep -oP '\d+' | sed -n '1p'
}

stop_server() {
  set +e
  if [ -n "$CURRENT_MONITOR_PID" ]; then
    kill -TERM "$CURRENT_MONITOR_PID" 2>/dev/null
    wait "$CURRENT_MONITOR_PID" 2>/dev/null
    CURRENT_MONITOR_PID=
  fi
  sudo -n docker rm -f "$CONTAINER" >/dev/null 2>&1
  if [ -n "$CURRENT_LOG_PID" ]; then
    kill -TERM "$CURRENT_LOG_PID" 2>/dev/null
    wait "$CURRENT_LOG_PID" 2>/dev/null
    CURRENT_LOG_PID=
  fi
  set -e
}

cleanup() {
  stop_server
}
trap cleanup EXIT INT TERM

cooldown() {
  while [ $(( $(<"$TEMP") / 1000 )) -gt 55 ]; do sleep 2; done
}

ensure_empty_gpu() {
  if ps -eo args | grep -Eq '[l]lama-server|[l]lama-bench|[v]llm serve'; then
    echo 'isolation failure: inference process remains' >&2
    return 1
  fi
  if [ -n "$(sudo -n docker ps -q)" ]; then
    echo 'isolation failure: a container remains active' >&2
    return 1
  fi
  local free
  free=$(visible_free)
  [ "$free" -ge 31000 ] || {
    echo "isolation failure: visible_free=$free MiB" >&2
    return 1
  }
}

stop_server
ensure_empty_gpu

for coordinate in "${COORDINATES[@]}"; do
  IFS='|' read -r name prompt_tokens output class <<< "$coordinate"
  system_file=$PI_SYSTEM
  case "$class" in
    prefill|decode|decode_small_prompt) system_file=$STANDARD_SYSTEM ;;
  esac
  prompt_file="$OUT/prompts/$name.json"
  printf 'milestone=tokenizer_start coordinate=%s\n' "$name"
  sudo -n docker run --rm \
    -v "$MODEL_DIR:/model:ro" -v "$GEN:/generate.py:ro" \
    -v "$system_file:/system.txt:ro" -v "$OUT/prompts:/out" \
    --entrypoint python "$IMG" /generate.py \
    --model /model --system-prompt-file /system.txt \
    --output "/out/$name.json" --targets "$prompt_tokens" --per-target 6
  printf 'milestone=tokenizer_done coordinate=%s output=%s class=%s\n' "$name" "$output" "$class"
done
for _ in $(seq 1 30); do
  [ "$(visible_free)" -ge 31000 ] && [ -z "$(sudo -n docker ps -q)" ] && break
  sleep 1
done
ensure_empty_gpu

{
  echo "run_id=$RUN_ID"
  echo 'protocol_version=2.2'
  echo "started_utc=$(date -u +%FT%TZ)"
  echo 'status=PROVISIONAL_NOT_FOR_PUBLIC_HEADLINE'
  echo 'class=phase_separated_vllm_C1_prefill_decode_full_context'
  echo 'modes=no-spec,mtp1,mtp2,mtp4'
  echo 'prefix_cache=enabled; unique cold prefixes; zero hit delta required'
  echo 'prefill_coordinates=p512/g1,p2048/g1,p4096/g1,p6144/g1,p8192/g1,p131071/g1'
  echo 'decode_coordinates=p512/g32,g128,g256,g512; p8192/g32,g128,g256,g512; historical control p9445/g128'
  echo 'full_context_decode_coordinates=p130944/g128,p130560/g512'
  echo 'measured_repetitions_per_cell=5'
  echo 'full_output_same_shape_warmups_per_cell=1'
  echo 'scheduler_budget=8192'
  echo 'gpu_memory_utilization=0.85'
  echo 'max_model_len=131072'
  echo 'max_num_seqs=64'
  echo "configured_cap_W=$CONFIGURED_CAP_W"
  echo 'timing_source=client_monotonic_SSE'
  echo 'prefill_metric=actual_endpoint_prompt_tokens/client_TTFT; includes scheduling and first-token work'
  echo 'post_first_formula=(completion_tokens-1)/(request_end-first_generated)'
  echo "image=$IMG"
  echo "model=$MODEL_DIR"
  echo "model_id=$MODEL_ID"
  echo "pi_system_file_sha256=$(sha256sum "$PI_SYSTEM" | cut -d' ' -f1)"
  echo "standard_system_file_sha256=$(sha256sum "$STANDARD_SYSTEM" | cut -d' ' -f1)"
  echo "mtp_patch_sha256=$(sha256sum "$PATCH_MTP" | cut -d' ' -f1)"
  echo "boundary_patch_sha256=$(sha256sum "$PATCH_BOUNDARY" | cut -d' ' -f1)"
  for coordinate in "${COORDINATES[@]}"; do
    IFS='|' read -r name prompt_tokens output class <<< "$coordinate"
    echo "${name}_prompt_set_sha256=$(sha256sum "$OUT/prompts/$name.json" | cut -d' ' -f1)"
  done
} > "$OUT/manifest.txt"

run_mode() {
  local mode=$1
  local n=$2
  local mode_out=$OUT/$mode
  local server_log=$mode_out/server.log
  local spec_file=$mode_out/spec.json
  mkdir -p "$mode_out"

  stop_server
  sleep 5
  ensure_empty_gpu
  cooldown

  local spec_args=
  local no_spec_arg=
  if [ "$n" -gt 0 ]; then
    printf '{"method":"mtp","num_speculative_tokens":%s}\n' "$n" > "$spec_file"
    spec_args='--speculative-config "$SPEC"'
  else
    printf '%s\n' '{}' > "$spec_file"
    no_spec_arg=--no-spec
  fi

  printf 'milestone=launch_start mode=%s\n' "$mode"
  local gid
  gid=$(stat -c '%g' /dev/dri/render* | sort -u | sed -n '1p')
  sudo -n docker run -d --restart unless-stopped --name "$CONTAINER" -p "$PORT:8000" \
    --device /dev/dri --group-add "$gid" -v /dev/dri:/dev/dri:ro \
    -v "$MODEL_DIR:/model:ro" -v "$spec_file:/spec.json:ro" \
    -v "$PATCH_MTP:/patch_mtp.py:ro" -v "$PATCH_BOUNDARY:/patch_boundary.py:ro" \
    -e VLLM_TARGET_DEVICE=xpu -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
    -e B70_MTP_BF16_DRAFT=1 -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    --entrypoint bash "$IMG" -lc \
    "set -e; python /patch_mtp.py; python /patch_boundary.py; SPEC=\$(cat /spec.json); exec vllm serve /model --quantization gptq --dtype float16 --max-model-len 131072 --gpu-memory-utilization 0.85 --port 8000 --max-num-seqs 64 --max-num-batched-tokens 8192 --enable-prefix-caching --served-model-name $MODEL --language-model-only $spec_args" >/dev/null

  sudo -n docker logs -f "$CONTAINER" > "$server_log" 2>&1 &
  CURRENT_LOG_PID=$!
  for i in $(seq 1 120); do
    sleep 5
    if curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
    sudo -n docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true || return 1
    [ "$i" -lt 120 ] || { echo "server readiness timeout: $mode" >&2; return 1; }
  done
  sudo -n docker exec "$CONTAINER" python -c \
    "import importlib.metadata as m; print('vllm=' + m.version('vllm')); print('vllm-xpu-kernels=' + m.version('vllm-xpu-kernels'))" \
    > "$mode_out/package-versions.txt"

  local free capacity
  free=$(visible_free)
  sudo -n cat /sys/kernel/debug/dri/0000:0b:00.0/tile0/vram_mm > "$mode_out/vram-after-load.txt"
  capacity=$(sed -n 's/.*GPU KV cache size: \([0-9,]*\) tokens.*/\1/p' "$server_log" | sed -n '$p' | tr -d ',')
  grep -q "'enable_prefix_caching': True" "$server_log" || {
    echo "cache startup state mismatch: $mode" >&2
    return 2
  }
  {
    echo "mode=$mode"
    echo "num_speculative_tokens=$n"
    echo 'prefix_cache=enabled'
    echo "visible_free_after_load_MiB=$free"
    echo "logged_kv_capacity_tokens=${capacity:-unknown}"
  } > "$mode_out/manifest.txt"
  [ "$free" -ge 500 ] || { echo "free VRAM floor failed: $free" >&2; return 2; }
  [ -n "$capacity" ] && [ "$capacity" -ge 131072 ] || {
    echo "KV capacity failed: ${capacity:-unknown}" >&2
    return 2
  }
  printf 'milestone=launch_ready mode=%s free_MiB=%s capacity=%s\n' "$mode" "$free" "$capacity"

  python3 "$MON" "$mode_out/monitor.jsonl" 0.5 &
  CURRENT_MONITOR_PID=$!
  sleep 1
  for coordinate in "${COORDINATES[@]}"; do
    IFS='|' read -r name prompt_tokens output class <<< "$coordinate"
    exact_output_arg=
    [ "$output" -eq 1 ] || exact_output_arg=--ignore-eos
    cooldown
    printf 'milestone=coordinate_start mode=%s coordinate=%s\n' "$mode" "$name"
    python3 "$HARNESS" --mode context \
      --prompts "$OUT/prompts/$name.json" --target "$prompt_tokens" \
      --output "$output" --budget 8192 --reps 5 --model "$MODEL" \
      --root "http://127.0.0.1:$PORT" --outdir "$mode_out/$name" \
      --full-output-warmup $exact_output_arg $no_spec_arg 2>&1 | tee "$mode_out/$name-harness.log"
    printf 'milestone=coordinate_completed mode=%s coordinate=%s\n' "$mode" "$name"
  done

  kill -TERM "$CURRENT_MONITOR_PID" 2>/dev/null || true
  wait "$CURRENT_MONITOR_PID" || true
  CURRENT_MONITOR_PID=
  curl -fsS "http://127.0.0.1:$PORT/metrics" > "$mode_out/metrics-after.txt"
  echo 'result=completed' >> "$mode_out/manifest.txt"
  echo "completed_utc=$(date -u +%FT%TZ)" >> "$mode_out/manifest.txt"
  printf 'milestone=mode_completed mode=%s\n' "$mode"
  stop_server
  cooldown
}

run_mode no-spec 0
run_mode mtp1 1
run_mode mtp2 2
run_mode mtp4 4

echo 'result=all_four_modes_completed' >> "$OUT/manifest.txt"
echo "completed_utc=$(date -u +%FT%TZ)" >> "$OUT/manifest.txt"
printf 'milestone=campaign_completed run=%s\n%s\n' "$RUN_ID" "$OUT"
