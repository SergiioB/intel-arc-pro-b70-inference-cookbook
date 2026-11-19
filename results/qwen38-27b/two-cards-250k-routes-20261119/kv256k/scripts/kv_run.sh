#!/usr/bin/env bash
# One server campaign for the kv256k task: speed at ~250K (code prompt) and the needle retrieval check.
# Usage: kv_run.sh OUT TP(1|2) KV(bf16|int8) CTX [CACHE_GIB=0]
#   env XPU_ATTN_PREF (default 256; set empty to turn the prefix kernel off)
#   env PLAN = space-separated steps, each "PROMPTS_KEY:DEPTHS:KINDS:ARM:MAXTOK"
#     PROMPTS_KEY deep = prompts-deep.json, ret = needles.json, warm = prompts-warm.json
#   default PLAN: "deep:256k:code:drafted:512 deep:256k:code:serial:512 ret:32k,120k,250k:ret:drafted:160"
# TP=1 runs on GPU index 1 (ZE_AFFINITY_MASK=1). TP=2 sees both cards. Samples vram_mm of both cards every 2 s.
set -eo pipefail
OUT=$1; TP=$2; KV=$3; CTX=$4; CACHE=${5:-0}
TF=$TF
BIN=$TF/src-tp3586/zig-out/xpu/bin/tensorfold-native
PORT=${PORT:-8046}
HW=$(python3 ~/B70-DOCS/tools/host_facts.py hwmon 1)
HWB=$(python3 ~/B70-DOCS/tools/host_facts.py hwmon 0)
PCI1=$(basename "$(readlink -f /sys/class/drm/card1/device)"); PCI0=$(basename "$(readlink -f /sys/class/drm/card0/device)")
declare -A P=([deep]=~/B70-DOCS/results/2026/11/tf-prefix-attn-20261113/prompts-deep.json
              [ret]=$WORK/kv256k/needles.json
              [warm]=~/B70-DOCS/results/2026/11/b70-breakthrough-20261118/prompts-warm.json)
PLAN=${PLAN:-"deep:256k:code:drafted:512 deep:256k:code:serial:512 ret:32k,120k,250k:ret:drafted:160"}
mkdir -p "$OUT"
[ -n "${SCHED:-}" ] || [ -z "$(docker ps -q)" ] || { echo "containers running"; exit 1; }
[ -n "${SCHED:-}" ] || [ -z "$(pgrep -x tensorfold-nati)" ] || { echo "a tensorfold server is running"; exit 1; }
CAP1=$(cat $HW/power1_cap); CAP0=$(cat $HWB/power1_cap)
export XPU_ATTN_PREF=${XPU_ATTN_PREF-256} ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
[ -n "$XPU_ATTN_PREF" ] || unset XPU_ATTN_PREF
unset XPU_KV_INT8; [ "$KV" = int8 ] && export XPU_KV_INT8=1
if [ "$TP" = 1 ]; then export ZE_AFFINITY_MASK=${CARD:-1}; else unset ZE_AFFINITY_MASK; fi
ARGS=(serve $TF/models/Qwen3.8-27B-MLX-4bit --drafter $TF/models/Qwen3.8-27B-DFlash2 --parallel 1 --context $CTX --tp $TP
      --host 127.0.0.1 --port $PORT --prompt-cache-gib $CACHE --no-thinking --no-update-check)
vram() { for p in $PCI0 $PCI1; do printf '%s ' "$(sudo -n cat /sys/kernel/debug/dri/$p/tile0/vram_mm | awk '/visible_avail/{print $2}')"; done; }
{ echo "utc $(date -u +%FT%TZ)"
  echo "gpu1 $PCI1 $HW cap_uw $CAP1 temp_mc $(cat $HW/temp2_input) max_freq $(cat /sys/class/drm/card1/device/tile0/gt0/freq0/max_freq); gpu0 $PCI0 $HWB cap_uw $CAP0 temp_mc $(cat $HWB/temp2_input) max_freq $(cat /sys/class/drm/card0/device/tile0/gt0/freq0/max_freq)"
  echo "XPU_ATTN_PREF=${XPU_ATTN_PREF-unset} XPU_KV_INT8=${XPU_KV_INT8-} ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK-} $BIN ${ARGS[*]}"
  echo "PLAN=$PLAN"
  git -C $TF/src-tp3586 rev-parse HEAD; git -C $TF/src-tp3586 status --short | head; sha256sum "$BIN" ${P[deep]} ${P[ret]}
  echo "vram_visible_avail_before_load gpu0 gpu1: $(vram)"; } > "$OUT/env.txt"
( while :; do echo "$(date +%s) $(vram)"; sleep 2; done ) > "$OUT/vram.tsv" 2>/dev/null &
VPID=$!
nohup "$BIN" "${ARGS[@]}" > "$OUT/server.log" 2>&1 < /dev/null &
PID=$!
trap 'kill $PID 2>/dev/null; sleep 3; kill -9 $PID 2>/dev/null || true; kill $VPID 2>/dev/null || true
      echo "{\"utc\":\"$(date -u +%FT%TZ)\",\"server_gone\":$(kill -0 $PID 2>/dev/null && echo false || echo true),\"caps_unchanged\":$([ "$(cat $HW/power1_cap)" = "$CAP1" ] && [ "$(cat $HWB/power1_cap)" = "$CAP0" ] && echo true || echo false)}" > "$OUT/cleanup.json"' EXIT
for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "server exited"; tail -20 "$OUT/server.log"; exit 3; }; sleep 3; done
echo "vram_visible_avail_after_load gpu0 gpu1: $(vram)" >> "$OUT/env.txt"
for step in $PLAN; do
  IFS=: read -r pk depths kinds arm mt <<< "$step"
  extra=(); [ "$arm" = serial ] && extra=(--extra-json '{"draft": false}')
  eb0=$(cat $HWB/energy1_input); ea0=$(cat $HW/energy1_input); t0=$(date +%s.%N)
  python3 ~/B70-DOCS/scripts/bench/depth_bench.py run --base http://127.0.0.1:$PORT --model qwen --prompts ${P[$pk]} \
    --out "$OUT" --hwmon $HW --cool ${COOL:-62} --depths $depths --reps 1 --kinds $kinds --arm $arm --max-tokens $mt "${extra[@]}" \
    || echo "{\"step\":\"$step\",\"failed\":true}" >> "$OUT/failures.jsonl"
  echo "{\"step\":\"$step\",\"gpu0_energy_j\":$(( ($(cat $HWB/energy1_input) - eb0) / 1000000 )),\"gpu1_energy_j\":$(( ($(cat $HW/energy1_input) - ea0) / 1000000 )),\"wall_s\":$(echo "$(date +%s.%N) - $t0" | bc)}" >> "$OUT/arm_energy.jsonl"
  kill -0 $PID 2>/dev/null || { echo "server died during $step"; tail -20 "$OUT/server.log"; exit 4; }
done
