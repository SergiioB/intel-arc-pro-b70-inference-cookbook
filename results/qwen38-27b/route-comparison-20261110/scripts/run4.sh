#!/usr/bin/env bash
# Follow-up to run3 (2026-11-10), GPU 1 only.
#  - EXL3 thinking-mode depth cells: run3's harness missed the exl3xpu `reasoning` delta
#    field, so those cells had no TTFT/decode. Re-run with depth_bench_reasoning.py.
#  - Prefix-cache corruption check for EXL3 --no-mtp (AGENTS.md §7): 9 exact-answer
#    questions sharing an ~8K prefix, cache on (twice: cold, then warm) vs cache off.
set -uo pipefail
W=/data/nvme/work/route-verify-20261110; R=$W/out4; mkdir -p "$R"; ST=$W/out/static
GPU=1; HW=$(python3 ~/B70-DOCS/tools/host_facts.py hwmon $GPU)
DB=$W/depth_bench_reasoning.py
PROMPTS=~/B70-DOCS/results/2026/11/b70-tonight-20261108/prompts-qwen38.json
DK="sudo -n docker -H unix:///run/b70-exl3-docker.sock"
THINK='{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"chat_template_kwargs":{"enable_thinking":true}}'
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$R/run.log"; }
busy(){ pgrep -f 'tensorfold-native serve|llama-server|VLLM::EngineCore' >/dev/null || [ -n "$($DK ps -q 2>/dev/null)" ]; }
if busy; then log "v4 refusing: inference running"; exit 1; fi
CAP0=$(cat $HW/power1_cap); sha256sum $DB $W/prefix_cache_check.py > "$R/harness.sha256"
log "v4 start cap_uw=$CAP0 temp_mc=$(cat $HW/temp2_input)"
up(){ for i in $(seq 1 ${2:-240}); do curl -sf "$1/health" >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }
srv(){ local out=$1 cmdf=$2 cmd; mkdir -p "$out"
  cmd=$(sed 's/^docker -H unix:\/\/\/run\/b70-exl3-docker.sock run /run --label b70.owner=claude /' "$cmdf")
  echo "$DK $cmd" > "$out/launch.txt"; eval "nohup $DK $cmd" > "$out/server.log" 2>&1 < /dev/null &
  up http://127.0.0.1:8100 180 || { log "$out not healthy"; return 1; }
  curl -s http://127.0.0.1:8100/v1/models | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])'; }
stop(){ $DK stop -t 30 b70-qwen38-27b-exl3-8100 >/dev/null 2>&1; sleep 10; }
hits(){ curl -s http://127.0.0.1:8100/metrics | grep -E '^vllm:.*prefix_cache_(hits|queries)' > "$1"; }

# 1. EXL3 --no-mtp, cache on
M=$(srv "$R/E0" $ST/dryrun-nomtp-cacheon.cmd) || exit 1; log "E0 up model=$M"
python3 $DB run --base http://127.0.0.1:8100 --model "$M" --prompts $PROMPTS --out "$R/E0" --hwmon $HW --cool 55 \
  --depths 0,8k --reps 3 --max-tokens 512 --arm think --extra-json "$THINK" >> "$R/E0/depth.log" 2>&1
hits "$R/E0/hits-before-check.txt"
python3 $W/prefix_cache_check.py http://127.0.0.1:8100 "$M" "$R/E0/prefix-check-cold.json" >> "$R/E0/prefix.log" 2>&1
python3 $W/prefix_cache_check.py http://127.0.0.1:8100 "$M" "$R/E0/prefix-check-warm.json" >> "$R/E0/prefix.log" 2>&1
hits "$R/E0/hits-after-check.txt"; stop; log "E0 done: $(tail -2 $R/E0/prefix.log | tr '\n' ' ')"
# 2. EXL3 --no-mtp, cache off (reference)
M=$(srv "$R/E0off" $ST/dryrun-nomtp-cacheoff.cmd) || exit 1
python3 $W/prefix_cache_check.py http://127.0.0.1:8100 "$M" "$R/E0off/prefix-check.json" >> "$R/E0off/prefix.log" 2>&1
hits "$R/E0off/hits-after-check.txt"; stop; log "E0off done: $(tail -1 $R/E0off/prefix.log)"
# 3. EXL3 MTP3, cache on: thinking-mode depth cells
M=$(srv "$R/E3" $ST/dryrun-mtp3-cacheon.cmd) || exit 1
python3 $DB run --base http://127.0.0.1:8100 --model "$M" --prompts $PROMPTS --out "$R/E3" --hwmon $HW --cool 55 \
  --depths 0,8k --reps 3 --max-tokens 512 --arm think --extra-json "$THINK" >> "$R/E3/depth.log" 2>&1
python3 $W/prefix_cache_check.py http://127.0.0.1:8100 "$M" "$R/E3/prefix-check.json" >> "$R/E3/prefix.log" 2>&1
stop; log "E3 done: $(tail -1 $R/E3/prefix.log)"
log "v4 done cap_unchanged=$([ "$(cat $HW/power1_cap)" = "$CAP0" ] && echo true || echo false)"
