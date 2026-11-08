#!/usr/bin/env bash
# llama.cpp SYCL build A/B: decode (tg128) and prefill (pp512) at depth, per KV type, one B70.
# Usage: llama_kv_depth_ab.sh OUT_DIR MODEL.gguf [GPU_INDEX=0]
# Env: BUILDS="name=/path/to/bin ..." KVS="f16:f16 q8_0:q8_0 ..." DEPTHS=0,16384,65536 REPS=2
set -uo pipefail
OUT=$1; MODEL=$2; GPU=${3:-0}
mkdir -p "$OUT"
HW=$(python3 host_facts.py hwmon "$GPU")
BUILDS=${BUILDS:-"oct15=$LLAMA_OLD/build-sycl/bin nov08=$LLAMA_NEW/build-sycl/bin"}
KVS=${KVS:-"f16:f16 q8_0:q8_0 q4_0:q4_0 q5_0:q4_1"}
DEPTHS=${DEPTHS:-0,16384,65536}
REPS=${REPS:-2}
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u  # setvars.sh exits a shell running with set -u
export ONEAPI_DEVICE_SELECTOR=level_zero:$GPU ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE SYCL_CACHE_PERSISTENT=0
! pgrep -x 'llama-bench|llama-server' >/dev/null || { echo "llama proc running"; exit 1; }
{ echo "utc $(date -u +%FT%TZ) gpu $GPU $HW cap_uw $(cat $HW/power1_cap)"; sha256sum "$MODEL" 2>/dev/null | cut -c1-16;
  for b in $BUILDS; do d=${b#*=}; echo "${b%%=*} $(git -C ${d%/build-sycl/bin} describe --tags --always)"; done; } > "$OUT/env.txt"
for kv in $KVS; do
  for b in $BUILDS; do
    name=${b%%=*}; bin=${b#*=}; ctk=${kv%%:*}; ctv=${kv##*:}
    while [ $(( $(cat $HW/temp2_input) / 1000 )) -gt ${COOL:-55} ]; do sleep 2; done
    f="$OUT/$name-$ctk-$ctv.jsonl"
    e0=$(cat $HW/energy1_input); t0=$(date +%s.%N)
    # llama-bench on some builds hangs after printing; cap the run and keep what printed
    timeout ${TMO:-2400} "$bin/llama-bench" -m "$MODEL" -ngl 99 -fa 1 -ctk $ctk -ctv $ctv -t 8 \
      -p 512 -n 128 -d $DEPTHS -r $REPS -o jsonl > "$f" 2> "$OUT/$name-$ctk-$ctv.err"
    rc=$?
    echo "{\"build\":\"$name\",\"kv\":\"$kv\",\"rc\":$rc,\"secs\":$(echo "$(date +%s.%N) - $t0" | bc),\"energy_j\":$(( ($(cat $HW/energy1_input) - e0) / 1000000 ))}" >> "$OUT/runs.jsonl"
    python3 - "$f" "$name" "$kv" <<'EOF'
import json,sys
for l in open(sys.argv[1]):
    try: r=json.loads(l)
    except Exception: continue
    print(sys.argv[2], sys.argv[3], 'd=%s'%r.get('n_depth'), 'p=%s n=%s'%(r.get('n_prompt'),r.get('n_gen')), round(r.get('avg_ts',0),2), flush=True)
EOF
  done
done
