#!/usr/bin/env bash
# Route comparison v3, Qwen3.8-27B, GPU 1 only, 2026-11-10.
# Operator rules: recipes use the model card's recommended sampling (not greedy);
# prefix cache ON by default; pick the winner across all routes, then act.
# Card (Qwen/Qwen3.8-27B README, "Best Practices"):
#   thinking (default): temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0
#   non-thinking:       temperature 0.7, top_p 0.80, top_k 20, min_p 0, presence_penalty 1.5
# Greedy is used only for the cache-on vs cache-off corruption check (AGENTS.md §7).
#
# Routes (one server each, 32K context per request, cache on):
#   D   llama.cpp b12244 SYCL F16 build, Q4_K_M, q8_0/q8_0 KV, 4 slots
#   C   TensorFold v1.2.5 + DFlash2, MLX 4-bit, prompt cache 4 GiB, 4 slots
#   E0  EXL3 4.00bpw, --no-mtp, cache on (launcher preview command)
#   E3  EXL3 4.00bpw, MTP3, cache on (launcher preview command, recipe default)
# Extra servers for the corruption check: C0 (TF, prompt cache 0), E0off (EXL3 --no-mtp --cache off).
set -uo pipefail
ROUTES=${1:-D C E0 E3}
W=/data/nvme/work/route-verify-20261110; R=$W/out3; mkdir -p "$R"
GPU=1
HW=$(python3 ~/B70-DOCS/tools/host_facts.py hwmon $GPU)
DB=~/B70-DOCS/scripts/bench/depth_bench.py
SS=~/B70-DOCS/scripts/bench/session_stability.py
PROMPTS=~/B70-DOCS/results/2026/11/b70-tonight-20261108/prompts-qwen38.json
BATT=~/hq4-work/battery/run_battery.py
DK="sudo -n docker -H unix:///run/b70-exl3-docker.sock"
ST=$W/out/static
THINK='{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"chat_template_kwargs":{"enable_thinking":true}}'
NOTHINK='{"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"chat_template_kwargs":{"enable_thinking":false}}'
NOTHINK_P0='{"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"chat_template_kwargs":{"enable_thinking":false}}'
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$R/run.log"; }
busy(){ pgrep -f 'tensorfold-native serve|llama-server|VLLM::EngineCore' >/dev/null || [ -n "$(docker ps -q)" ] \
        || [ -n "$($DK ps -q 2>/dev/null)" ]; }
if busy; then log "v3 refusing: inference process or container running"; exit 1; fi
CAP0=$(cat $HW/power1_cap)
# oneAPI once, before any array is defined: setvars.sh clobbers a variable named "args".
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
sha256sum $DB $SS $PROMPTS $BATT > "$R/harness.sha256"
{ echo "THINK=$THINK"; echo "NOTHINK=$NOTHINK"; echo "NOTHINK_P0=$NOTHINK_P0"; } > "$R/sampling.txt"
log "v3 start routes='$ROUTES' gpu=$GPU hwmon=$HW cap_uw=$CAP0 temp_mc=$(cat $HW/temp2_input)"
energy(){ cat $HW/energy1_input; }
up(){ for i in $(seq 1 ${2:-240}); do curl -sf "$1/health" >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }
model_id(){ curl -s "$1/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])'; }
probe(){ # base model extra -> http code of one sampled request
  curl -s -o /dev/null -w '%{http_code}' "$1/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$(python3 -c "import json,sys;b={'model':sys.argv[1],'messages':[{'role':'user','content':'Say OK.'}],'max_tokens':16};b.update(json.loads(sys.argv[2]));print(json.dumps(b))" "$2" "$3")"; }
depth(){ python3 $DB run --base "$1" --model "$2" --prompts $PROMPTS --out "$3" --hwmon $HW --cool 55 --depths 0,8k --reps 3 \
           --max-tokens 512 --arm "$4" --extra-json "$5" >> "$3/depth.log" 2>&1; }
batt(){ python3 $BATT --base-url "$1" --model "$2" --out "$3" --parallel 4 --timeout 1800 --label "$4" --extra-body "$5" >> "$3.log" 2>&1; }
session(){ python3 $SS run --base "$1" --model "$2" --prompts $PROMPTS --depth 8k --kind code --turns 10 --max-tokens 384 \
             --hwmon $HW --out "$3" ${4:+--extra-json "$4"} >> "$3.log" 2>&1; }
metrics(){ curl -s "$1/metrics" | grep -E '^vllm:.*prefix_cache_(hits|queries)' > "$2" 2>/dev/null; }

measure(){ # name base model nothink_json
  local name=$1 base=$2 M=$3 NT=$4 out="$R/$1"
  local code; code=$(probe "$base" "$M" "$NOTHINK"); echo "nothink_presence1.5_http=$code" > "$out/probe.txt"
  code=$(probe "$base" "$M" "$THINK"); echo "think_http=$code" >> "$out/probe.txt"
  log "$name probe: $(tr '\n' ' ' < "$out/probe.txt")"
  depth "$base" "$M" "$out" think "$THINK"
  depth "$base" "$M" "$out" nothink "$NT"
  for i in 1 2 3; do  # independent repeats: a fixed seed would make sampled runs identical
    local NTs; NTs=$(python3 -c "import json,sys;d=json.loads(sys.argv[1]);d['seed']=1000+int(sys.argv[2]);print(json.dumps(d))" "$NT" $i)
    batt "$base" "$M" "$out/battery-nothink-$i" "$name-nothink-$i" "$NTs"; done
  metrics "$base" "$out/metrics-before-session.txt"
  session "$base" "$M" "$out/session-nothink" "$NT"
  metrics "$base" "$out/metrics-after-session.txt"
  log "$name measured"
}

llama_route(){
  local out="$R/D" port=8040; mkdir -p "$out"
  local BIN=/data/nvme/engines/llama.cpp-latest/build-sycl-f16/bin/llama-server
  local sargs=(-m /data/sata2/gguf/qwen38-27b-Q4_K_M.gguf -ngl 99 -fa on -t 8 -lm none -c 131072 -np 4 -ctk q8_0 -ctv q8_0
              -b 2048 -ub 2048 --jinja --host 127.0.0.1 --port $port)
  echo "ONEAPI_DEVICE_SELECTOR=level_zero:$GPU $BIN ${sargs[*]}" > "$out/launch.txt"
  git -C /data/nvme/engines/llama.cpp-latest describe --tags > "$out/engine-version.txt"
  local e0=$(energy) t0=$(date +%s)
  ONEAPI_DEVICE_SELECTOR=level_zero:$GPU ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE nohup "$BIN" "${sargs[@]}" > "$out/server.log" 2>&1 < /dev/null &
  local pid=$!
  up http://127.0.0.1:$port 120 || { log "D not healthy"; kill $pid; return 1; }
  log "D up load_s=$(( $(date +%s)-t0 ))"
  measure D http://127.0.0.1:$port qwen "$NOTHINK"
  echo "{\"energy_uj\": $(( $(energy)-e0 )), \"wall_s\": $(( $(date +%s)-t0 ))}" > "$out/energy.json"
  kill $pid; sleep 5; kill -9 $pid 2>/dev/null; log "D stopped"
}

tf_server(){ # out cache_gib parallel -> echoes pid
  local out=$1 gib=$2 par=$3 port=8036
  local TF=/data/sata1/work/tensorfold-20261104
  local BIN=$TF/src-v1.2.5/zig-out/xpu/bin/tensorfold-native
  local sargs=(serve $TF/models/Qwen3.8-27B-MLX-4bit --drafter $TF/models/Qwen3.8-27B-DFlash2 --parallel $par --context 32768
              --host 127.0.0.1 --port $port --prompt-cache-gib $gib --no-update-check)
  echo "ZE_AFFINITY_MASK=$GPU $BIN ${sargs[*]}" > "$out/launch.txt"; sha256sum "$BIN" > "$out/bin.sha256"
  ZE_AFFINITY_MASK=$GPU ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE nohup "$BIN" "${sargs[@]}" > "$out/server.log" 2>&1 < /dev/null &
  echo $!
}

tf_route(){
  local out="$R/C" port=8036; mkdir -p "$out"
  local e0=$(energy) t0=$(date +%s) pid
  pid=$(tf_server "$out" 4 4)
  if ! up http://127.0.0.1:$port 120; then
    log "C not healthy at parallel 4; retry parallel 2"; kill $pid; sleep 5; kill -9 $pid 2>/dev/null
    pid=$(tf_server "$out" 4 2); up http://127.0.0.1:$port 120 || { log "C not healthy"; kill $pid; return 1; }
  fi
  log "C up load_s=$(( $(date +%s)-t0 )) ($(cat $out/launch.txt | grep -oE -- '--parallel [0-9]+'))"
  local NT="$NOTHINK"; [ "$(probe http://127.0.0.1:$port qwen "$NOTHINK")" = 200 ] || { NT="$NOTHINK_P0"; log "C refuses presence_penalty 1.5; non-thinking runs use presence_penalty 0 (deviation from card)"; }
  measure C http://127.0.0.1:$port qwen "$NT"
  session http://127.0.0.1:$port qwen "$out/session-greedy-cacheon"
  echo "{\"energy_uj\": $(( $(energy)-e0 )), \"wall_s\": $(( $(date +%s)-t0 ))}" > "$out/energy.json"
  kill $pid; sleep 5; kill -9 $pid 2>/dev/null; log "C stopped"
  # corruption reference: same greedy session with the prompt cache off
  local o0="$R/C0"; mkdir -p "$o0"; pid=$(tf_server "$o0" 0 1)
  up http://127.0.0.1:$port 120 && session http://127.0.0.1:$port qwen "$o0/session-greedy-cacheoff"
  kill $pid; sleep 5; kill -9 $pid 2>/dev/null; log "C0 stopped"
}

exl3_server(){ # out cmdfile
  local out=$1 cmdf=$2 cmd
  cmd=$(sed 's/^docker -H unix:\/\/\/run\/b70-exl3-docker.sock run /run --label b70.owner=claude /' "$cmdf")
  cp "$cmdf" "$out/launcher-preview.cmd"; echo "$DK $cmd" > "$out/launch.txt"
  eval "nohup $DK $cmd" > "$out/server.log" 2>&1 < /dev/null &
}
exl3_stop(){ $DK stop -t 30 b70-qwen38-27b-exl3-8100 >/dev/null 2>&1; sleep 10; }

exl3_route(){ # name cmdfile corruption_check(0/1)
  local name=$1 cmdf=$2 cc=$3 out="$R/$1" port=8100; mkdir -p "$out"
  local e0=$(energy) t0=$(date +%s)
  exl3_server "$out" "$cmdf"
  up http://127.0.0.1:$port 180 || { log "$name not healthy"; exl3_stop; return 1; }
  log "$name up load_s=$(( $(date +%s)-t0 ))"
  local M; M=$(model_id http://127.0.0.1:$port)
  b70 test $port > "$out/b70-test.txt" 2>&1; log "$name b70 test rc=$?"
  measure "$name" http://127.0.0.1:$port "$M" "$NOTHINK"
  if [ "$cc" = 1 ]; then
    for i in 1 2; do metrics http://127.0.0.1:$port "$out/metrics-greedy$i-before.txt"
      session http://127.0.0.1:$port "$M" "$out/session-greedy-cacheon-$i"
      metrics http://127.0.0.1:$port "$out/metrics-greedy$i-after.txt"; done
  fi
  echo "{\"energy_uj\": $(( $(energy)-e0 )), \"wall_s\": $(( $(date +%s)-t0 ))}" > "$out/energy.json"
  exl3_stop; log "$name stopped"
  if [ "$cc" = 1 ]; then
    local o0="$R/${name}off"; mkdir -p "$o0"; exl3_server "$o0" $ST/dryrun-nomtp-cacheoff.cmd
    up http://127.0.0.1:$port 180 && session http://127.0.0.1:$port "$M" "$o0/session-greedy-cacheoff"
    metrics http://127.0.0.1:$port "$o0/metrics-after.txt"; exl3_stop; log "${name}off stopped"
  fi
}

for r in $ROUTES; do
  case $r in
    D)  llama_route ;;
    C)  tf_route ;;
    E0) exl3_route E0 $ST/dryrun-nomtp-cacheon.cmd 1 ;;
    E3) exl3_route E3 $ST/dryrun-mtp3-cacheon.cmd 0 ;;
  esac
done
log "v3 done cap_unchanged=$([ "$(cat $HW/power1_cap)" = "$CAP0" ] && echo true || echo false)"
