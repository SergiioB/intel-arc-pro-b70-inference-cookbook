#!/usr/bin/env bash
#
# xpu-wedge-watchdog.sh - detect and recover from Intel Arc GPU (Xe2 / Battlemage)
# Level-Zero wedge during sustained LLM inference.
#
# Problem:
#   Under sustained multi-GPU Level-Zero inference load, the xe kernel driver can
#   reset a compute/copy engine (ccs/bcs) and return "Fault response: Unsuccessful
#   -ENOENT/-EINVAL". After the reset the userspace Level-Zero context stays wedged
#   permanently: the serving engine hangs (vLLM: TimeoutError -> EngineDeadError)
#   until the container is restarted. Corroborating reports:
#     - intel/compute-runtime#948 (dual Arc Pro B70, vLLM TP=2)
#     - vllm-project/vllm#41663 (same hardware, GP fault + bcs engine reset)
#     - darktable#20257 (same ccs engine-reset signature via OpenCL, no vLLM)
#
# What this watchdog does:
#   1. Polls a health endpoint (OpenAI-compatible /health by default).
#   2. Watches kernel logs for engine-reset / fault-response signatures.
#   3. When health is down AND a wedge signature appeared since the last good
#      check -> run the recovery command (default: docker restart) and verify.
#   4. Emits every event as one parseable line with an ISO-8601 timestamp.
#
# Usage:
#   ./xpu-wedge-watchdog.sh               # single pass
#   ./xpu-wedge-watchdog.sh --loop        # keep scanning (used by systemd)
#   ./xpu-wedge-watchdog.sh --self-test   # offline detection test, no docker
#
# Configuration (env vars, all optional):
#   HEALTH_URL        health endpoint                 (default http://127.0.0.1:8000/health)
#   HEALTH_OK_CODES   space-separated ok status codes (default "200" "0"=unreachable-forbidden? see code)
#   CONTAINER         container name for default recovery (default vllm-serve)
#   RECOVERY_CMD      full recovery command           (default "docker restart $CONTAINER")
#   SCAN_INTERVAL_S   seconds between scans in --loop (default 10)
#   HEALTH_TIMEOUT_S  curl timeout per check           (default 5)
#   FAIL_STREAK       failed checks before acting      (default 3)
#   RECOVERY_TIMEOUT_S max seconds to wait for health  (default 180)
#   TRIGGER_MODE      both|kernel|health              (default both)
#   KMSG_SOURCE       auto|journal|dmesg|file:/path    (default auto)
#   WEBHOOK_URL       optional notification endpoint   (default empty)
#
# Exit codes: 0 ok/recovered, 2 no usable kernel-log source, 3 recovery failed.

set -u

# ---- Config with defaults -------------------------------------------------
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/health}"
HEALTH_OK_CODES="${HEALTH_OK_CODES:-200}"
CONTAINER="${CONTAINER:-vllm-serve}"
RECOVERY_CMD="${RECOVERY_CMD:-docker restart "$CONTAINER"}"
SCAN_INTERVAL_S="${SCAN_INTERVAL_S:-10}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-5}"
FAIL_STREAK="${FAIL_STREAK:-3}"
RECOVERY_TIMEOUT_S="${RECOVERY_TIMEOUT_S:-180}"
TRIGGER_MODE="${TRIGGER_MODE:-both}"          # both | kernel | health
KMSG_SOURCE="${KMSG_SOURCE:-auto}"            # auto | journal | dmesg | file:/path
STATE_FILE="${STATE_FILE:-/tmp/xpu-wedge-watchdog.state}"
WEBHOOK_URL="${WEBHOOK_URL:-}"

# Kernel signatures that correlate with the L0 wedge (see issues above).
# Multi-line pattern: newlines act as alternation for grep -E.
SIGNATURES='Engine reset: engine_class=(ccs|bcs)
|Fault response: Unsuccessful
|trying reset from guc_exec_queue_timedout_job
|TLB invalidation fence timeout
|Completion-Wait loop timed out'

SNAPSHOT_TAIL=500
SELF="$(readlink -f "$0")"

# ---- State -----------------------------------------------------------------
STREAK=0          # consecutive failed health checks

# ---- Logging ----------------------------------------------------------------
stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log()  { printf '[%s] %s\n' "$(stamp)" "$*"; }

notify() {
  [ -n "$WEBHOOK_URL" ] || return 0
  curl -sS -o /dev/null -w "" --max-time 5 \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"$(printf '%s' "$*" | sed 's/"/\\"/g')\"}" "$WEBHOOK_URL" \
    >/dev/null 2>&1 || true
}

# ---- Health check ------------------------------------------------------------
is_healthy() {
  local code
  code=$(curl -sS -o /dev/null -w "%{http_code}" --max-time "$HEALTH_TIMEOUT_S" "$HEALTH_URL" 2>/dev/null) || code=000
  for c in $HEALTH_OK_CODES; do
    [ "$code" = "$c" ] && return 0
  done
  return 1
}

# ---- Kernel log snapshot ------------------------------------------------------
# Returns the current kernel-log snapshot (tail) on stdout.
kernel_snapshot() {
  local src="$KMSG_SOURCE" file
  case "$src" in
    auto)
      if command -v journalctl >/dev/null 2>&1 && journalctl -k -n 1 >/dev/null 2>&1; then
        journalctl -k -n "$SNAPSHOT_TAIL" -o cat 2>/dev/null | tail -n "$SNAPSHOT_TAIL"
      elif command -v dmesg >/dev/null 2>&1; then
        dmesg 2>/dev/null | tail -n "$SNAPSHOT_TAIL"
      fi
      ;;
    journal) journalctl -k -n "$SNAPSHOT_TAIL" -o cat 2>/dev/null | tail -n "$SNAPSHOT_TAIL" ;;
    dmesg)   dmesg 2>/dev/null | tail -n "$SNAPSHOT_TAIL" ;;
    file:*)  file="${src#file:}"; [ -r "$file" ] && tail -n "$SNAPSHOT_TAIL" "$file" ;;
  esac
}

# Diff-style: lines in CURRENT that were not in PREV (content-based, order kept).
# An empty PREV means no baseline yet - every line is "new" w.r.t. nothing.
new_kernel_lines() {
  local cur="$1" prev="$2" line
  [ -n "$prev" ] || { printf '%s\n' "$cur"; return 0; }
  printf '%s\n' "$cur" | while IFS= read -r line; do
    case "$prev" in
      *"$line"*) : ;;
      *) printf '%s\n' "$line" ;;
    esac
  done
}

# ---- Recovery ----------------------------------------------------------------
recover() {
  log "RECOVERY run cmd='$RECOVERY_CMD'"
  notify "Xe2 watchdog: wedge detected, restarting ${CONTAINER} (${HEALTH_URL})"
  eval "$RECOVERY_CMD" >/dev/null 2>&1
  local waited=0
  while [ "$waited" -lt "$RECOVERY_TIMEOUT_S" ]; do
    if is_healthy; then
      log "RECOVERED container=${CONTAINER} after=${waited}s"
      notify "Xe2 watchdog: ${CONTAINER} recovered after ${waited}s"
      STREAK=0
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done
  log "RECOVERY_FAILED container=${CONTAINER} waited=${RECOVERY_TIMEOUT_S}s"
  notify "Xe2 watchdog: RECOVERY FAILED for ${CONTAINER} after ${RECOVERY_TIMEOUT_S}s - manual intervention"
  return 3
}

# ---- One scan pass -------------------------------------------------------------
run_once() {
  local cur new sigs ret=0 state prev

  cur="$(kernel_snapshot)" || true
  # An empty snapshot is valid for file: sources (log not yet written) but
  # signals an unavailable source for journal/dmesg.
  if [ -z "$cur" ] && [ "${KMSG_SOURCE#file:}" = "$KMSG_SOURCE" ]; then
    log "KMSG_UNAVAILABLE source=${KMSG_SOURCE} (need root or adm/systemd-journal; set KMSG_SOURCE=dmesg or file:/path)"
    return 2
  fi

  # Kernel snapshot state is persisted across invocations, so one-shot runs
  # (systemd timer / cron) build on the previous run's baseline correctly.
  state="$STATE_FILE"
  if [ ! -f "$state" ]; then
    printf '%s\n' "$cur" > "$state"
    log "BOOTSTRAP state=${state} snapshot_seeded lines=$(printf '%s\n' "$cur" | wc -l)"
    return 0
  fi
  prev="$(cat "$state" 2>/dev/null || true)"
  new="$(new_kernel_lines "$cur" "$prev")"
  printf '%s\n' "$cur" > "$state"

  if is_healthy; then
    if [ "$STREAK" -ne 0 ]; then
      log "HEALTH_OK restored"
      STREAK=0
    fi
    return 0
  fi

  STREAK=$((STREAK + 1))
  sigs="$(printf '%s\n' "$new" | grep -Ei "$SIGNATURES" || true)"
  log "HEALTH_DOWN streak=${STREAK} new_kernel_lines=$(printf '%s\n' "$new" | wc -l | tr -d ' ')"

  [ "$STREAK" -ge "$FAIL_STREAK" ] || return 0

  if [ "$TRIGGER_MODE" = "health" ]; then
    log "WEDGE trigger=health streak=${STREAK} (TRIGGER_MODE=health)"
    recover; return $?
  fi

  if [ -n "$sigs" ]; then
    log "WEDGE trigger=kernel streak=${STREAK} first_match=$(printf '%s\n' "$sigs" | head -n1)"
    recover; return $?
  fi

  if [ "$TRIGGER_MODE" = "kernel" ]; then
    log "DEGRADED no_kernel_signature - manual check required"
    notify "Xe2 watchdog: ${CONTAINER} unhealthy (${HEALTH_URL}, streak ${STREAK}) but NO GPU engine-reset signature - not restarting, please check"
    return 0
  fi

  log "DEGRADED no_kernel_signature - waiting for signature or recovery"
  return 0
}

# ---- Offline self-test -----------------------------------------------------------
self_test() {
  local tmp kmsg logf rc=0 sport srvpid
  tmp="$(mktemp -d)"
  kmsg="$tmp/kmsg.log"; : > "$kmsg"

  # Healthy-path server: any 200 works (directory listing is fine).
  sport=$(( 21000 + RANDOM % 1000 ))
  if command -v python3 >/dev/null 2>&1; then
    ( cd "$tmp" && exec python3 -m http.server "$sport" >/dev/null 2>&1 ) &
  elif command -v nc >/dev/null 2>&1; then
    ( while :; do printf 'HTTP/1.1 200 OK\r\n\r\nok' | nc -l "$sport" >/dev/null 2>&1 || break; done ) &
  else
    log "SELFTEST SKIP: need python3 or nc"; rm -rf "$tmp"; return 1
  fi
  sleep 1

  # 1) Healthy path: bootstrap pass, then pass 2 must be a silent no-op.
  HEALTH_URL="http://127.0.0.1:${sport}/" KMSG_SOURCE="file:$kmsg" STATE_FILE="$tmp/state-ok" \
    RECOVERY_CMD="echo would-restart" "$SELF" > "$tmp/out-1.log" 2>&1
  HEALTH_URL="http://127.0.0.1:${sport}/" KMSG_SOURCE="file:$kmsg" STATE_FILE="$tmp/state-ok" \
    RECOVERY_CMD="echo would-restart" "$SELF" > "$tmp/out-2.log" 2>&1
  if grep -q "BOOTSTRAP" "$tmp/out-1.log" \
     && ! grep -qE "WEDGE|DEGRADED|RECOVERY|HEALTH_DOWN" "$tmp/out-2.log"; then
    log "SELFTEST ok: healthy path no-op"
  else
    log "SELFTEST FAIL: healthy path (out-2 below)"; sed -n '1,20p' "$tmp/out-2.log"; rc=1
  fi
  pkill -f "http.server $sport" 2>/dev/null || true
  sleep 0.3

  # 2) Wedge path: bootstrap on empty log, then inject signatures + kill health.
  : > "$kmsg"
  HEALTH_URL="http://127.0.0.1:1/health" KMSG_SOURCE="file:$kmsg" STATE_FILE="$tmp/state-wedge" \
    RECOVERY_CMD="echo would-restart" "$SELF" > "$tmp/w-1.log" 2>&1
  printf '%s\n' \
    'xe 0000:c7:00.0: [drm] Tile0: GT0: Engine reset: engine_class=ccs, logical_mask: 0x1, guc_id=23, state=0x289' \
    'xe 0000:c7:00.0: [drm] Tile0: GT0: Fault response: Unsuccessful -ENOENT' >> "$kmsg"
  HEALTH_URL="http://127.0.0.1:1/health" HEALTH_TIMEOUT_S=2 FAIL_STREAK=1 \
    RECOVERY_TIMEOUT_S=10 KMSG_SOURCE="file:$kmsg" STATE_FILE="$tmp/state-wedge" \
    RECOVERY_CMD="echo would-restart" \
    "$SELF" > "$tmp/w-2.log" 2>&1
  if grep -q "WEDGE" "$tmp/w-2.log" && grep -q "RECOVERY run" "$tmp/w-2.log"; then
    log "SELFTEST ok: wedge triggers restart"
  else
    log "SELFTEST FAIL: wedge path (w-2 below)"; sed -n '1,30p' "$tmp/w-2.log"; rc=1
  fi

  rm -rf "$tmp"
  return $rc
}

# ---- Entrypoint ---------------------------------------------------------------
main() {
  case "${1:-}" in
    --loop)
      log "START mode=loop health=${HEALTH_URL} container=${CONTAINER} trigger=${TRIGGER_MODE} interval=${SCAN_INTERVAL_S}s"
      notify "Xe2 watchdog: ${CONTAINER} monitoring started (${HEALTH_URL})"
      while :; do
        run_once || { [ $? -eq 2 ] && { log "NO_KMSG_SOURCE - exiting"; return 2; }; }
        sleep "$SCAN_INTERVAL_S"
      done
      ;;
    --self-test)
      self_test
      ;;
    "")
      run_once
      ;;
    *)
      echo "usage: $0 [--loop|--self-test]" >&2
      return 2
      ;;
  esac
}

main "$@"