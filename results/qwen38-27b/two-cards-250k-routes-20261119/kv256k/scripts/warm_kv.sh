#!/usr/bin/env bash
# Warm follow-up turns at ~250K (code-256k of prompts-deep.json; env DEPTH/PROMPTS for another), one server.
# Usage: warm_kv.sh OUT TP(1|2) KV(bf16|int8) CTX CACHE_GIB [TURNS=3]
# Run once with CACHE_GIB=0 (every turn re-reads the whole prompt: the "fresh" text) and once with a cache
# sized to one state (the "resumed" text); compare session.jsonl sha256 per turn.
set -eo pipefail
OUT=$1; TP=$2; KV=$3; CTX=$4; CACHE=$5; TURNS=${6:-3}
TF=$TF; BIN=$TF/src-tp3586/zig-out/xpu/bin/tensorfold-native; PORT=${PORT:-8046}
HW=$(python3 ~/B70-DOCS/tools/host_facts.py hwmon 1); HWB=$(python3 ~/B70-DOCS/tools/host_facts.py hwmon 0)
PCI1=$(basename "$(readlink -f /sys/class/drm/card1/device)"); PCI0=$(basename "$(readlink -f /sys/class/drm/card0/device)")
PROMPTS=${PROMPTS:-~/B70-DOCS/results/2026/11/tf-prefix-attn-20261113/prompts-deep.json}; DEPTH=${DEPTH:-256k}
mkdir -p "$OUT"
[ -n "${SCHED:-}" ] || [ -z "$(docker ps -q)" ] || { echo "containers running"; exit 1; }
[ -n "${SCHED:-}" ] || [ -z "$(pgrep -x tensorfold-nati)" ] || { echo "a tensorfold server is running"; exit 1; }
CAP1=$(cat $HW/power1_cap); CAP0=$(cat $HWB/power1_cap)
export XPU_ATTN_PREF=256 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
unset XPU_KV_INT8; [ "$KV" = int8 ] && export XPU_KV_INT8=1
if [ "$TP" = 1 ]; then export ZE_AFFINITY_MASK=${CARD:-1}; else unset ZE_AFFINITY_MASK; fi
ARGS=(serve $TF/models/Qwen3.8-27B-MLX-4bit --drafter $TF/models/Qwen3.8-27B-DFlash2 --parallel 1 --context $CTX --tp $TP
      --host 127.0.0.1 --port $PORT --prompt-cache-gib $CACHE --no-thinking --no-update-check)
vram() { for p in $PCI0 $PCI1; do printf '%s ' "$(sudo -n cat /sys/kernel/debug/dri/$p/tile0/vram_mm | awk '/visible_avail/{print $2}')"; done; }
{ echo "utc $(date -u +%FT%TZ) gpu1 cap_uw $CAP1 temp_mc $(cat $HW/temp2_input); gpu0 cap_uw $CAP0 temp_mc $(cat $HWB/temp2_input)"
  echo "XPU_ATTN_PREF=256 XPU_KV_INT8=${XPU_KV_INT8-} ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK-} $BIN ${ARGS[*]}"; sha256sum "$BIN" "$PROMPTS"
  echo "vram_visible_avail_before_load gpu0 gpu1: $(vram)"; } > "$OUT/env.txt"
( while :; do echo "$(date +%s) $(vram)"; sleep 2; done ) > "$OUT/vram.tsv" 2>/dev/null &
VPID=$!
nohup "$BIN" "${ARGS[@]}" > "$OUT/server.log" 2>&1 < /dev/null &
PID=$!
trap 'kill $VPID 2>/dev/null || true; kill $PID 2>/dev/null || true; sleep 3; kill -9 $PID 2>/dev/null || true; kill $VPID 2>/dev/null || true
      echo "{\"utc\":\"$(date -u +%FT%TZ)\",\"server_gone\":$(kill -0 $PID 2>/dev/null && echo false || echo true),\"caps_unchanged\":$([ "$(cat $HW/power1_cap)" = "$CAP1" ] && [ "$(cat $HWB/power1_cap)" = "$CAP0" ] && echo true || echo false)}" > "$OUT/cleanup.json"' EXIT
for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "server exited"; tail -20 "$OUT/server.log"; exit 3; }; sleep 3; done
echo "vram_visible_avail_after_load gpu0 gpu1: $(vram)" >> "$OUT/env.txt"
python3 ~/B70-DOCS/scripts/bench/session_stability.py run --base http://127.0.0.1:$PORT --model qwen \
  --prompts $PROMPTS --depth $DEPTH --kind code --turns $TURNS --max-tokens 128 --hwmon $HW --out "$OUT/session" || echo "session failed"
grep -E "prompt cache|done chat" "$OUT/server.log" | sed -E 's/chatcmpl-[0-9a-f]+ //' | cut -c1-240
