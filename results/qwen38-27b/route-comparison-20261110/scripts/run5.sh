#!/usr/bin/env bash
# Whole-context comparison, Qwen3.8-27B, GPU 1 only, 2026-11-10.
# Operator rule: judge routes across the whole context, not only short answers.
# Per route (one slot, full context, prefix cache on, model-card sampling):
#   decode + cold TTFT at 0/8K/32K/64K/128K/240K (code corpus, thinking mode, 3 reps, 512 tokens)
#   retrieval: three codes at 10/50/90 % of a 32K/128K/240K prompt (non-thinking, 3 reps)
# Prompts regenerated from the same corpus for every route: prompts-full.json, needles.json.
set -uo pipefail
ROUTES=${1:-E3 C D E0}
W=/data/nvme/work/route-verify-20261110; R=$W/out5; mkdir -p "$R"; ST=$W/out/static
GPU=1; HW=$(python3 ~/B70-DOCS/tools/host_facts.py hwmon $GPU)
DB=$W/depth_bench_full.py; PR=$W/prompts-full.json; ND=$W/needles.json
DK="sudo -n docker -H unix:///run/b70-exl3-docker.sock"
THINK='{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"chat_template_kwargs":{"enable_thinking":true}}'
NOTHINK='{"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"chat_template_kwargs":{"enable_thinking":false}}'
NOTHINK_P0='{"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"chat_template_kwargs":{"enable_thinking":false}}'
DEPTHS=0,8k,32k,64k,128k,240k
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$R/run.log"; }
busy(){ pgrep -f 'tensorfold-native serve|llama-server|VLLM::EngineCore' >/dev/null || [ -n "$($DK ps -q 2>/dev/null)" ]; }
if busy; then log "v5 refusing: inference running"; exit 1; fi
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
CAP0=$(cat $HW/power1_cap)
sha256sum $DB $PR $ND $W/needle_check.py > "$R/inputs.sha256"
log "v5 start routes='$ROUTES' cap_uw=$CAP0 temp_mc=$(cat $HW/temp2_input)"
energy(){ cat $HW/energy1_input; }
up(){ for i in $(seq 1 ${2:-240}); do curl -sf "$1/health" >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }
measure(){ # name base model nothink
  local out="$R/$1"
  python3 $DB run --base "$2" --model "$3" --prompts $PR --out "$out" --hwmon $HW --cool 55 --depths $DEPTHS --kinds code \
    --reps 3 --max-tokens 512 --timeout 5400 --arm think --extra-json "$THINK" >> "$out/depth.log" 2>&1
  log "$1 depth done"
  python3 $W/needle_check.py run "$2" "$3" $ND "$4" "$out/needles.json" >> "$out/needles.log" 2>&1
  log "$1 needles: $(tr '\n' ' ' < $out/needles.log | tail -c 300)"
}

exl3(){ # name cmdfile
  local name=$1 out="$R/$1" cmd e0 t0; mkdir -p "$out"
  cmd=$(sed 's/^docker -H unix:\/\/\/run\/b70-exl3-docker.sock run /run --label b70.owner=claude /' "$2")
  echo "$DK $cmd" > "$out/launch.txt"; e0=$(energy); t0=$(date +%s)
  eval "nohup $DK $cmd" > "$out/server.log" 2>&1 < /dev/null &
  up http://127.0.0.1:8100 200 || { log "$name not healthy"; $DK stop b70-qwen38-27b-exl3-8100 >/dev/null 2>&1; return 1; }
  log "$name up load_s=$(( $(date +%s)-t0 ))"
  local M; M=$(curl -s http://127.0.0.1:8100/v1/models | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
  measure "$name" http://127.0.0.1:8100 "$M" "$NOTHINK"
  echo "{\"energy_uj\": $(( $(energy)-e0 )), \"wall_s\": $(( $(date +%s)-t0 ))}" > "$out/energy.json"
  $DK stop -t 30 b70-qwen38-27b-exl3-8100 >/dev/null 2>&1; sleep 10; log "$name stopped"
}

tf(){
  local out="$R/C" port=8036 TF=/data/sata1/work/tensorfold-20261104 e0 t0 pid ctx
  local BIN=$TF/src-v1.2.5/zig-out/xpu/bin/tensorfold-native; mkdir -p "$out"; e0=$(energy); t0=$(date +%s)
  for ctx in 262144 131072; do
    local sargs=(serve $TF/models/Qwen3.8-27B-MLX-4bit --drafter $TF/models/Qwen3.8-27B-DFlash2 --parallel 1 --context $ctx
                 --host 127.0.0.1 --port $port --prompt-cache-gib 2 --no-update-check)
    echo "ZE_AFFINITY_MASK=$GPU $BIN ${sargs[*]}" > "$out/launch.txt"
    ZE_AFFINITY_MASK=$GPU ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE nohup "$BIN" "${sargs[@]}" > "$out/server-$ctx.log" 2>&1 < /dev/null &
    pid=$!
    if up http://127.0.0.1:$port 60; then log "C up ctx=$ctx load_s=$(( $(date +%s)-t0 ))"; break; fi
    log "C cannot serve ctx=$ctx: $(tail -2 $out/server-$ctx.log | tr '\n' ' ' | cut -c1-200)"; kill $pid 2>/dev/null; sleep 5; kill -9 $pid 2>/dev/null; pid=
  done
  [ -n "${pid:-}" ] || return 1
  measure C http://127.0.0.1:$port qwen "$NOTHINK_P0"
  echo "{\"energy_uj\": $(( $(energy)-e0 )), \"wall_s\": $(( $(date +%s)-t0 ))}" > "$out/energy.json"
  kill $pid; sleep 5; kill -9 $pid 2>/dev/null; log "C stopped"
}

llama(){
  local out="$R/D" port=8040 e0 t0 pid; mkdir -p "$out"
  local BIN=/data/nvme/engines/llama.cpp-latest/build-sycl-f16/bin/llama-server
  local sargs=(-m /data/sata2/gguf/qwen38-27b-Q4_K_M.gguf -ngl 99 -fa on -t 8 -lm none -c 262144 -np 1 -ctk q8_0 -ctv q8_0
               -b 2048 -ub 2048 --jinja --host 127.0.0.1 --port $port)
  echo "ONEAPI_DEVICE_SELECTOR=level_zero:$GPU $BIN ${sargs[*]}" > "$out/launch.txt"; e0=$(energy); t0=$(date +%s)
  ONEAPI_DEVICE_SELECTOR=level_zero:$GPU ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE nohup "$BIN" "${sargs[@]}" > "$out/server.log" 2>&1 < /dev/null &
  pid=$!
  up http://127.0.0.1:$port 120 || { log "D not healthy: $(tail -2 $out/server.log | tr '\n' ' ')"; kill $pid; return 1; }
  log "D up load_s=$(( $(date +%s)-t0 ))"
  measure D http://127.0.0.1:$port qwen "$NOTHINK"
  echo "{\"energy_uj\": $(( $(energy)-e0 )), \"wall_s\": $(( $(date +%s)-t0 ))}" > "$out/energy.json"
  kill $pid; sleep 5; kill -9 $pid 2>/dev/null; log "D stopped"
}

for r in $ROUTES; do
  case $r in
    E3) exl3 E3 $ST/dryrun-mtp3-full.cmd ;;
    E0) exl3 E0 $ST/dryrun-nomtp-full.cmd ;;
    C)  tf ;;
    D)  llama ;;
  esac
done
log "v5 done cap_unchanged=$([ "$(cat $HW/power1_cap)" = "$CAP0" ] && echo true || echo false)"
