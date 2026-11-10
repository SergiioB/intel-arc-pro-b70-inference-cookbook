#!/usr/bin/env bash
# Completes run5 (died 2026-11-10 ~13:30 UTC during TensorFold's 64K cell) and the long-chat sessions.
# GPU 1, model-card sampling, prefix cache on. Same prompts as run5 (prompts-full.json, needles).
#   C : TensorFold v1.2.5, ctx 131072 (262144 refused): depth 64k,128k; needles 32k,128k; session 64k
#   D : llama.cpp b12244 Q4_K_M q8_0 KV, ctx 262144: depth 0..240k; needles 32k,128k,240k; session 64k
#   E3: EXL3 4.00bpw MTP3, ctx 262144: session 64k (depth + needles done in run5)
set -uo pipefail
W=/data/nvme/work/route-verify-20261110; R=$W/out7; mkdir -p "$R"; ST=$W/out/static
GPU=1; HW=$(python3 ~/B70-DOCS/tools/host_facts.py hwmon $GPU)
DB=$W/depth_bench_full.py; PR=$W/prompts-full.json; SS=~/B70-DOCS/scripts/bench/session_stability.py
DK="sudo -n docker -H unix:///run/b70-exl3-docker.sock"
THINK='{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"chat_template_kwargs":{"enable_thinking":true}}'
NOTHINK='{"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"chat_template_kwargs":{"enable_thinking":false}}'
NOTHINK_P0='{"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"chat_template_kwargs":{"enable_thinking":false}}'
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$R/run.log"; }
busy(){ pgrep -f 'tensorfold-native serve|llama-server|VLLM::EngineCore' >/dev/null || [ -n "$($DK ps -q 2>/dev/null)" ]; }
if busy; then log "v7 refusing"; exit 1; fi
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
CAP0=$(cat $HW/power1_cap); log "v7 start cap_uw=$CAP0 temp_mc=$(cat $HW/temp2_input)"
up(){ for i in $(seq 1 ${2:-240}); do curl -sf "$1/health" >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }
depth(){ python3 $DB run --base "$1" --model "$2" --prompts $PR --out "$3" --hwmon $HW --cool 55 --depths "$4" --kinds code \
           --reps 3 --max-tokens 512 --timeout 3600 --arm think --extra-json "$THINK" >> "$3/depth.log" 2>&1; log "$(basename $3) depth $4 done"; }
needle(){ python3 $W/needle_check.py run "$1" "$2" "$3" "$4" "$5/needles.json" >> "$5/needles.log" 2>&1; log "$(basename $5) needles: $(tr '\n' ' ' < $5/needles.log | tail -c 260)"; }
sess(){ python3 $SS run --base "$1" --model "$2" --prompts $PR --depth 64k --kind code --turns 10 --max-tokens 384 \
          --hwmon $HW --out "$3" --timeout 3600 --extra-json "$4" >> "$3.log" 2>&1; log "$(basename $(dirname $3)) session done"; }

# C: TensorFold
TF=/data/sata1/work/tensorfold-20261104; mkdir -p $R/C; P=
for gib in 4 2; do
  ZE_AFFINITY_MASK=$GPU ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE nohup $TF/src-v1.2.5/zig-out/xpu/bin/tensorfold-native serve $TF/models/Qwen3.8-27B-MLX-4bit \
    --drafter $TF/models/Qwen3.8-27B-DFlash2 --parallel 1 --context 131072 --host 127.0.0.1 --port 8036 --prompt-cache-gib $gib --no-update-check \
    > $R/C/server-cache$gib.log 2>&1 < /dev/null &
  P=$!; if up http://127.0.0.1:8036 80; then log "C up ctx=131072 prompt-cache-gib=$gib"; break; fi
  log "C refused cache $gib: $(tail -1 $R/C/server-cache$gib.log | cut -c1-160)"; kill $P; sleep 5; kill -9 $P 2>/dev/null; P=
done
if [ -n "$P" ]; then
  depth http://127.0.0.1:8036 qwen $R/C 64k,128k
  needle http://127.0.0.1:8036 qwen $W/needles-131k.json "$NOTHINK_P0" $R/C
  mkdir -p $R/C; sess http://127.0.0.1:8036 qwen $R/C/session "$NOTHINK_P0"
  kill $P; sleep 5; kill -9 $P 2>/dev/null
fi
log "C stopped"
# D: llama.cpp
mkdir -p $R/D
ONEAPI_DEVICE_SELECTOR=level_zero:$GPU ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE nohup /data/nvme/engines/llama.cpp-latest/build-sycl-f16/bin/llama-server \
  -m /data/sata2/gguf/qwen38-27b-Q4_K_M.gguf -ngl 99 -fa on -t 8 -lm none -c 262144 -np 1 -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 --jinja \
  --host 127.0.0.1 --port 8040 > $R/D/server.log 2>&1 < /dev/null &
P=$!
if up http://127.0.0.1:8040 120; then
  log "D up"
  depth http://127.0.0.1:8040 qwen $R/D 0,8k,32k,64k,128k,240k
  needle http://127.0.0.1:8040 qwen $W/needles.json "$NOTHINK" $R/D
  sess http://127.0.0.1:8040 qwen $R/D/session "$NOTHINK"
fi
kill $P; sleep 5; kill -9 $P 2>/dev/null; log "D stopped"
# E3: EXL3 MTP3 session
mkdir -p $R/E3; cmd=$(sed 's/^docker -H unix:\/\/\/run\/b70-exl3-docker.sock run /run --label b70.owner=claude /' $ST/dryrun-mtp3-full.cmd)
eval "nohup $DK $cmd" > $R/E3/server.log 2>&1 < /dev/null &
if up http://127.0.0.1:8100 200; then
  M=$(curl -s http://127.0.0.1:8100/v1/models | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
  sess http://127.0.0.1:8100 "$M" $R/E3/session "$NOTHINK"
fi
$DK stop -t 30 b70-qwen38-27b-exl3-8100 >/dev/null 2>&1; sleep 10; log "E3 stopped"
log "v7 done cap_unchanged=$([ "$(cat $HW/power1_cap)" = "$CAP0" ] && echo true || echo false)"
