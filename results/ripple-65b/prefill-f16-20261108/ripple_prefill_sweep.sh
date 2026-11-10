#!/usr/bin/env bash
# RQ-41: Ripple-65B prefill on two B70s. Arms: build (F16 off/on) x ubatch x split x quant file.
# Usage: ripple_prefill_sweep.sh OUT_DIR
set -uo pipefail
OUT=$1; mkdir -p "$OUT"
E=$LLAMA
M=$MODELS
H0=$(python3 host_facts.py hwmon 0); H1=$(python3 host_facts.py hwmon 1)
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u  # setvars.sh exits a shell running with set -u
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE SYCL_CACHE_PERSISTENT=0 ONEAPI_DEVICE_SELECTOR='level_zero:0,1'
! pgrep -x 'llama-bench|llama-server' >/dev/null || { echo "llama proc running"; exit 1; }
echo "utc $(date -u +%FT%TZ) caps $(cat $H0/power1_cap) $(cat $H1/power1_cap) commit $(git -C $E describe --tags --always)" > "$OUT/env.txt"
cell() { # name build model extra-args...
  local name=$1 build=$2 model=$3; shift 3
  [ -f "$model" ] || { echo "$name: missing $model"; return; }
  # GPU1 idles 64-69 C after load on this host; wait for <=70/60 or a 60 s plateau
  local t=0; while { [ $(( $(cat $H1/temp2_input) / 1000 )) -gt 70 ] || [ $(( $(cat $H0/temp2_input) / 1000 )) -gt 60 ]; } && [ $t -lt 60 ]; do sleep 2; t=$((t+2)); done
  local a0=$(cat $H0/energy1_input) a1=$(cat $H1/energy1_input) t0=$(date +%s)
  timeout ${TMO:-2400} $E/$build/bin/llama-bench -m "$model" -ngl 99 -fa 1 -t 8 -r ${REPS:-2} -o jsonl "$@" > "$OUT/$name.jsonl" 2> "$OUT/$name.err"
  local rc=$?
  echo "{\"cell\":\"$name\",\"rc\":$rc,\"secs\":$(( $(date +%s) - t0 )),\"gpu0_j\":$(( ($(cat $H0/energy1_input)-a0)/1000000 )),\"gpu1_j\":$(( ($(cat $H1/energy1_input)-a1)/1000000 ))}" >> "$OUT/runs.jsonl"
  python3 - "$OUT/$name.jsonl" "$name" <<'PY'
import json,sys
for l in open(sys.argv[1]):
    try: r=json.loads(l)
    except Exception: continue
    print(sys.argv[2], 'ub=%s'%r.get('n_ubatch'), 'd=%s'%r.get('n_depth'), 'p=%s n=%s'%(r['n_prompt'],r['n_gen']), [round(x,1) for x in r['samples_ts']], flush=True)
PY
}
IQ=$M/Ripple-65B-Preview-AD-IQ4_XS.gguf; QK=$M/Ripple-65B-Preview-Q4_K_M.gguf
T=(-sm tensor -ctk f16 -ctv f16); L=(-sm layer -ctk q8_0 -ctv q8_0)
cell f16off-tensor-iq4xs build-sycl     $IQ "${T[@]}" -p 2048 -n 128 -b 2048 -ub 512,2048
cell f16on-tensor-iq4xs  build-sycl-f16 $IQ "${T[@]}" -p 2048 -n 128 -b 2048 -ub 512,2048
cell f16on-layer-iq4xs   build-sycl-f16 $IQ "${L[@]}" -p 2048 -n 128 -b 2048 -ub 512,2048
[ -n "${Q4KM:-}" ] && cell f16off-tensor-q4km  build-sycl     $QK "${T[@]}" -p 2048 -n 128 -b 2048 -ub 2048
[ -n "${Q4KM:-}" ] && cell f16on-tensor-q4km   build-sycl-f16 $QK "${T[@]}" -p 2048 -n 128 -b 2048 -ub 2048
