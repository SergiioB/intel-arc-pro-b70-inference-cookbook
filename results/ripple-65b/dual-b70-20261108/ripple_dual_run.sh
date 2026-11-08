#!/usr/bin/env bash
# Ripple-65B (dense, 80 blocks, SWA 4/5) on two B70s with llama.cpp SYCL: -sm layer vs -sm tensor.
# Usage: ripple_dual_run.sh OUT_DIR [MODEL]
set -uo pipefail
OUT=$1; MODEL=${2:-$MODELS/Ripple-65B-Preview-AD-IQ4_XS.gguf}
BIN=$LLAMA/build-sycl/bin
mkdir -p "$OUT"
H0=$(python3 host_facts.py hwmon 0); H1=$(python3 host_facts.py hwmon 1)
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u  # setvars.sh exits a shell running with set -u
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE SYCL_CACHE_PERSISTENT=0 ONEAPI_DEVICE_SELECTOR='level_zero:0,1'
! pgrep -f 'tensorfold-native serve|llama-server|llama-bench|VLLM::EngineCore' >/dev/null || { echo "inference proc running"; exit 1; }
{ echo "utc $(date -u +%FT%TZ) caps $(cat $H0/power1_cap) $(cat $H1/power1_cap) temps $(cat $H0/temp2_input) $(cat $H1/temp2_input)";
  echo "llama.cpp $(git -C ${BIN%/build-sycl/bin} describe --tags --always)"; ls -la "$MODEL"; } > "$OUT/env.txt"
cell() { # name, extra args...
  local name=$1; shift
  while [ $(( $(cat $H1/temp2_input) / 1000 )) -gt 72 ] || [ $(( $(cat $H0/temp2_input) / 1000 )) -gt 60 ]; do sleep 2; done
  local a0=$(cat $H0/energy1_input) a1=$(cat $H1/energy1_input) t0=$(date +%s)
  timeout ${TMO:-3000} $BIN/llama-bench -m "$MODEL" -ngl 99 -fa 1 -t 8 -p 512 -n 128 -d ${DEPTHS:-0,16384,32768} -r ${REPS:-2} -o jsonl "$@" \
     > "$OUT/$name.jsonl" 2> "$OUT/$name.err"
  local rc=$?
  echo "{\"cell\":\"$name\",\"rc\":$rc,\"secs\":$(( $(date +%s) - t0 )),\"gpu0_j\":$(( ($(cat $H0/energy1_input)-a0)/1000000 )),\"gpu1_j\":$(( ($(cat $H1/energy1_input)-a1)/1000000 ))}" >> "$OUT/runs.jsonl"
  python3 -c "
import json,sys
for l in open('$OUT/$name.jsonl'):
    try: r=json.loads(l)
    except Exception: continue
    print('$name','d=%s'%r.get('n_depth'),'p=%s n=%s'%(r['n_prompt'],r['n_gen']),round(r['avg_ts'],2),flush=True)"
}
cell layer-q8kv  -sm layer  -ctk q8_0 -ctv q8_0
cell layer-f16kv -sm layer  -ctk f16 -ctv f16
cell tensor-f16kv -sm tensor -ctk f16 -ctv f16
