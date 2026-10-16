#!/bin/bash
# Chain after the export: wait for run_bolt_export.sh -> restore NVIDIA's modeling file ->
# static check -> vLLM TP2 serve (cookbook launcher) -> 6-case greedy gate -> speed.
# Stops at the first failure and leaves the server running only on success.
#
# usage:
#   SRC=/models/bolt74b-bf16 OUT=/models/bolt74b-autoround-w4a16 \
#   ORIG=/models/modeling_nemotron_h.py.orig PY=/path/to/venv/bin/python \
#   setsid nohup ./post_export_chain.sh > post_export_chain.log 2>&1 < /dev/null & disown
#
# env (defaults in brackets):
#   SRC         BF16 source checkpoint (required)
#   OUT         export directory given to run_bolt_export.sh (required)
#   ORIG        NVIDIA's unpatched modeling_nemotron_h.py, saved before the einsum patch (required)
#   EXPORT_LOG  stdout of run_bolt_export.sh [$WORK/run_bolt_export.log]
#   PY          python with torch + safetensors for check_artifact.py [python3]
#   PORT        vLLM port [8000]
#   WAIT_S      readiness timeout in seconds [2400]
#   WORK        run directories land in $WORK/run-<UTC time> [directory of this script]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
WORK=${WORK:-$HERE}
SRC=${SRC:?set SRC}
OUT=${OUT:?set OUT}
ORIG=${ORIG:?set ORIG to the saved original modeling_nemotron_h.py}
EXPORT_LOG=${EXPORT_LOG:-$WORK/run_bolt_export.log}
PY=${PY:-python3}
PORT=${PORT:-8000}
WAIT_S=${WAIT_S:-2400}
CONTAINER=b70-nemotron-bolt-tp2
R=$WORK/run-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$R"
log() { echo "[chain $(date -u +%H:%M:%S)] $*"; }

while pgrep -f run_bolt_export.sh >/dev/null; do sleep 60; done
log "export runner finished: $(tail -2 "$EXPORT_LOG" | tr '\n' ' ')"
grep -q 'status=ok' "$EXPORT_LOG" || { log "FAIL: export status not ok"; exit 1; }

CFG=$(find "$OUT" -maxdepth 2 -name config.json | head -1)
[ -n "$CFG" ] || { log "FAIL: no config.json under $OUT"; exit 1; }
ART=$(dirname "$CFG")
log "artifact: $ART ($(du -sh "$ART" | cut -f1))"

# The export copied the einsum-patched modeling file; ship NVIDIA's original.
cp "$ORIG" "$ART/modeling_nemotron_h.py"
cp "$ORIG" "$SRC/modeling_nemotron_h.py"
log "restored original modeling_nemotron_h.py (sha $(sha256sum "$ART/modeling_nemotron_h.py" | cut -c1-16))"

"$PY" "$HERE/check_artifact.py" --src "$SRC" --artifact "$ART" --json "$R/check.json" > "$R/check.log" 2>&1 \
  || { log "FAIL: static check (see $R/check.log)"; tail -20 "$R/check.log"; exit 2; }
log "static check PASS"

# Serve with the cookbook launcher (graph mode, 65,536 context), then wait for the API and
# require the int4 MoE kernel on both ranks (load gate).
bash "$REPO/benchmarks/nemotron35-bolt-74b/launch-nemotron-bolt-vllm-tp2.sh" "$ART" "$PORT" > "$R/serve.log" 2>&1 \
  || { log "FAIL: launcher (see $R/serve.log)"; tail -30 "$R/serve.log"; exit 3; }
t0=$(date +%s); ready=no
while [ $(( $(date +%s) - t0 )) -lt "$WAIT_S" ]; do
  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=yes; break; fi
  [ -n "$(docker ps --filter "name=^/$CONTAINER$" -q)" ] || break
  sleep 10
done
docker logs "$CONTAINER" > "$R/vllm-container.log" 2>&1
n=$(grep -c 'Using XPUExpertsWNA16' "$R/vllm-container.log")
if [ "$ready" != yes ] || [ "$n" -lt 2 ]; then
  log "FAIL: serve/load gate (ready=$ready, XPUExpertsWNA16 lines=$n; see $R/vllm-container.log)"
  docker rm -f "$CONTAINER" >/dev/null 2>&1; exit 3
fi
log "server up in $(( $(date +%s) - t0 )) s, load gate PASS"

python3 "$HERE/correctness6.py" "$R/correctness" --url "http://127.0.0.1:$PORT/v1" > "$R/correctness.log" 2>&1
grep -q 'CORRECTNESS: 6/6 passed' "$R/correctness.log" || { log "FAIL: correctness (see $R/correctness.log)"; cat "$R/correctness.log"; exit 4; }
log "correctness PASS: $(grep CORRECTNESS "$R/correctness.log")"

URL="http://127.0.0.1:$PORT" python3 "$HERE/speed2.py" "$R/speed" > "$R/speed.log" 2>&1
log "speed: $(tail -4 "$R/speed.log" | tr '\n' ' ')"
log "DONE run dir $R (server left running: docker rm -f $CONTAINER)"
