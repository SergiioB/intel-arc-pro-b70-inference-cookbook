#!/usr/bin/env bash
#
# install-watchdog.sh - one-command installer for the Xe2 wedge watchdog.
#
# Usage:
#   sudo bash watchdog/install-watchdog.sh --container vllm-serve
#   sudo bash watchdog/install-watchdog.sh --container qwen38 --health http://127.0.0.1:8765/health
#   sudo bash watchdog/install-watchdog.sh --container qwen38 \
#       --recovery-cmd "systemctl restart llama-profile"        # native (no docker) deploy
#   bash watchdog/install-watchdog.sh --container vllm-serve --dry-run   # preview, no changes
#   bash watchdog/install-watchdog.sh --uninstall
#
# Flags:
#   --container NAME   docker container (or alias used by --recovery-cmd)
#   --health URL       health endpoint          (default http://127.0.0.1:8000/health)
#   --recovery-cmd CMD full recovery command    (default "docker restart NAME")
#   --interval N       scan seconds             (default 10)
#   --streak N         failed checks before acting (default 3)
#   --webhook URL      optional notification endpoint
#   --dry-run          print what would happen, change nothing
#   --uninstall        remove service + binary + state file
#
# Requires: bash, curl, sudo (for install), systemd (for --uninstall/verify).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATCHDOG="$SCRIPT_DIR/xpu-wedge-watchdog.sh"
UNIT_NAME="xpu-wedge-watchdog.service"
UNIT_PATH="/etc/systemd/system/$UNIT_NAME"
BIN_PATH="/usr/local/bin/xpu-wedge-watchdog.sh"
STATE_FILE="/tmp/xpu-wedge-watchdog.state"

CONTAINER=""
HEALTH_URL="http://127.0.0.1:8000/health"
RECOVERY_CMD=""
INTERVAL="10"
STREAK="3"
WEBHOOK=""
DRY=0

usage() { grep -E "^#   " "$0" | sed 's/^#   //'; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --container) CONTAINER="${2:-}"; shift 2 ;;
    --health)    HEALTH_URL="${2:-}"; shift 2 ;;
    --recovery-cmd) RECOVERY_CMD="${2:-}"; shift 2 ;;
    --interval)  INTERVAL="${2:-}"; shift 2 ;;
    --streak)    STREAK="${2:-}"; shift 2 ;;
    --webhook)   WEBHOOK="${2:-}"; shift 2 ;;
    --dry-run)   DRY=1; shift ;;
    --uninstall) MODE_UNINSTALL=1; shift ;;
    -h|--help)   usage ;;
    *) echo "unknown flag: $1" >&2; usage ;;
  esac
done

if [ "${MODE_UNINSTALL:-0}" = "1" ]; then
  if [ "$DRY" = "1" ]; then
    echo "would: systemctl disable --now $UNIT_NAME && rm $UNIT_PATH $BIN_PATH $STATE_FILE"
    exit 0
  fi
  sudo systemctl disable --now "$UNIT_NAME" 2>/dev/null || true
  sudo rm -f "$UNIT_PATH" "$BIN_PATH" "$STATE_FILE"
  echo "watchdog uninstalled."
  exit 0
fi

if [ -z "$CONTAINER" ]; then
  echo "ERROR: --container NAME is required (the docker container or alias your serve runs as)." >&2
  usage
fi
[ -n "$RECOVERY_CMD" ] || RECOVERY_CMD="docker restart $CONTAINER"

echo "== Xe2 wedge watchdog install =="
echo "   container : $CONTAINER"
echo "   health    : $HEALTH_URL"
echo "   recovery  : $RECOVERY_CMD"
echo "   interval  : ${INTERVAL}s   fail-streak: $STREAK"
[ -n "$WEBHOOK" ] && echo "   webhook   : $WEBHOOK"

if [ "$DRY" = "1" ]; then
  echo "DRY RUN - no changes made. Would:"
  echo "  1. cp $WATCHDOG -> $BIN_PATH"
  echo "  2. write unit $UNIT_PATH (Environment lines below)"
  echo "  3. systemctl daemon-reload && enable --now $UNIT_NAME"
  echo "--- generated unit ---"
fi

GEN_UNIT() {
  echo "[Unit]"
  echo "Description=Intel Arc GPU (Xe2) Level-Zero wedge watchdog"
  echo "After=docker.service network-online.target"
  echo "Wants=network-online.target"
  echo ""
  echo "[Service]"
  echo "Type=simple"
  echo "Environment=HEALTH_URL=$HEALTH_URL"
  echo "Environment=CONTAINER=$CONTAINER"
  echo "Environment=RECOVERY_CMD=$RECOVERY_CMD"
  echo "Environment=SCAN_INTERVAL_S=$INTERVAL"
  echo "Environment=FAIL_STREAK=$STREAK"
  echo "Environment=TRIGGER_MODE=both"
  [ -n "$WEBHOOK" ] && echo "Environment=WEBHOOK_URL=$WEBHOOK"
  echo "ExecStart=$BIN_PATH --loop"
  echo "Restart=on-failure"
  echo "RestartSec=5"
  echo "NoNewPrivileges=true"
  echo "ProtectSystem=full"
  echo "ProtectHome=true"
  echo "PrivateTmp=true"
  echo "ReadWritePaths=/var/log"
  echo ""
  echo "[Install]"
  echo "WantedBy=multi-user.target"
}

if [ "$DRY" = "1" ]; then
  GEN_UNIT
  exit 0
fi

[ -f "$WATCHDOG" ] || { echo "ERROR: $WATCHDOG not found (run from the repo root)." >&2; exit 1; }
[ "$(id -u)" = "0" ] || { echo "ERROR: run with sudo." >&2; exit 1; }

cp "$WATCHDOG" "$BIN_PATH" && chmod +x "$BIN_PATH"
GEN_UNIT > "$UNIT_PATH"
systemctl daemon-reload
systemctl enable --now "$UNIT_NAME"

echo ""
echo "installed. Verify:"
echo "  systemctl status $UNIT_NAME"
echo "  journalctl -u $UNIT_NAME -f"
echo "  bash $BIN_PATH --self-test"
echo ""
echo "Tune or swap container later: edit Environment= lines in $UNIT_PATH, then:"
echo "  sudo systemctl daemon-reload && sudo systemctl restart $UNIT_NAME"