# Watchdog: Xe2 (Battlemage) Level-Zero wedge detection & recovery

Operational companion to the model family recipes. Runs next to your vLLM XPU
image and restarts the serving container when the GPU wedges instead of leaving
the box serving dead traffic.

## The problem it solves

Under sustained multi-GPU Level-Zero inference load, the `xe` kernel driver can
reset a compute/copy engine and return `Fault response: Unsuccessful
-ENOENT/-EINVAL`. After the reset, the userspace Level-Zero context is
permanently wedged: the in-flight kernel never completes, the serving engine
dies (`TimeoutError: RPC call to sample_tokens timed out` ->
`EngineDeadError` in vLLM), and only a full process/container restart recovers.

Corroborated reports:

- **intel/compute-runtime#948** - dual Arc Pro B70 + vLLM TP=2, reproduces
  every 2-6 h on every stack tried (kernel 7.0.0-27, GuC 70.58.0).
- **vllm-project/vllm#41663** - same dual-B70 hardware, GP fault + bcs engine
  reset in `intel/vllm:0.17.0-xpu`.
- **darktable/darktable#20257** - identical `ccs` engine-reset signature via
  OpenCL, no vLLM/oneCCL involved: driver/firmware-level, not an app bug.

## How detection works

Two independent signals, both required by default (`TRIGGER_MODE=both`):

1. **Health**: `curl` against `HEALTH_URL` fails `FAIL_STREAK` times in a row.
2. **Kernel signature**: since the last scan, kernel logs contain one of
   `Engine reset: engine_class=ccs|bcs`, `Fault response: Unsuccessful`,
   `trying reset from guc_exec_queue_timedout_job`, `TLB invalidation fence
   timeout`, or `Completion-Wait loop timed out`.

When both hold, the recovery command runs (default `docker restart
<CONTAINER>`) and the watchdog polls health until recovered
(`RECOVERY_TIMEOUT_S`) or fails loudly with exit 3.

A health failure **without** a kernel signature logs `DEGRADED` and notifies -
an app-level hang that a GPU restart will not fix.

## Install (one command)

```bash
sudo bash watchdog/install-watchdog.sh --container vllm-serve
```

The installer copies the script to `/usr/local/bin`, writes a tuned systemd
unit, and starts the service. Useful flags:

- `--container NAME` - the docker container your serve runs as (the cookbook
  launchers take it from `$CONTAINER`; the watchdog needs the same name).
- `--health URL` - health endpoint (default `http://127.0.0.1:8000/health`).
- `--recovery-cmd "systemctl restart llama-profile"` - for native/systemd
  deploys that do not use docker.
- `--webhook URL` - optional notification endpoint.
- `--dry-run` - print the generated unit without touching the system.
- `--uninstall` - remove service, binary and state file.

Verify: `journalctl -u xpu-wedge-watchdog -f`

## Install (manual, 5 min)

```bash
sudo cp watchdog/xpu-wedge-watchdog.sh /usr/local/bin/xpu-wedge-watchdog.sh
sudo chmod +x /usr/local/bin/xpu-wedge-watchdog.sh
sudo cp watchdog/xpu-wedge-watchdog.service /etc/systemd/system/
# edit the Environment= lines in the unit: HEALTH_URL, CONTAINER, WEBHOOK_URL
sudo systemctl daemon-reload
sudo systemctl enable --now xpu-wedge-watchdog
```

Read events: `journalctl -u xpu-wedge-watchdog -f`. Every event is one line:

```
[2026-08-31T10:11:12Z] WEDGE trigger=kernel streak=3 first_match=xe 0000:c7:00.0: [drm] Tile0: GT0: Engine reset: engine_class=ccs, ...
[2026-08-31T10:12:37Z] RECOVERED container=vllm-serve after=45s
```

## Config reference

| Env | Default | Meaning |
|---|---|---|
| `HEALTH_URL` | `http://127.0.0.1:8000/health` | OpenAI-compatible health endpoint |
| `HEALTH_OK_CODES` | `200` | space-separated status codes treated as healthy |
| `CONTAINER` | `vllm-serve` | container name used by the default recovery |
| `RECOVERY_CMD` | `docker restart $CONTAINER` | full recovery command (quote carefully) |
| `SCAN_INTERVAL_S` | `10` | scan period in `--loop` mode |
| `HEALTH_TIMEOUT_S` | `5` | curl timeout per health check |
| `FAIL_STREAK` | `3` | failed checks before acting |
| `RECOVERY_TIMEOUT_S` | `180` | max wait for health after recovery |
| `TRIGGER_MODE` | `both` | `both` (kernel sig + health), `kernel`, or `health` |
| `KMSG_SOURCE` | `auto` | `auto`, `journal`, `dmesg`, or `file:/path` |
| `STATE_FILE` | `/tmp/xpu-wedge-watchdog.state` | kernel-snapshot baseline, persisted across runs |
| `WEBHOOK_URL` | *(empty)* | optional curl-POST notification endpoint |

## Verify it works

```bash
sudo /usr/local/bin/xpu-wedge-watchdog.sh --loop        # watch one full pass
sudo /usr/local/bin/xpu-wedge-watchdog.sh --self-test   # offline: healthy + wedge paths
```

The self-test needs a listening port (uses `nc`); it does **not** touch docker.

## Caveats

- **Kernel-log access**: kernel messages need root / `adm` / `systemd-journal`.
  The systemd unit runs as root - keep it that way. Do not run as an unprivileged
  user or `KMSG_SOURCE=dmesg` will silently return nothing and every health
  outage is classified `DEGRADED`.
- **In-flight requests are lost** on `docker restart`. For production traffic,
  front the server with a proxy that retries on connection error
  (nginx `proxy_next_upstream error`, or an OpenAI-compatible router).
- **Bootstrap pass** seeds the kernel snapshot without acting, so stale
  messages from boot cannot cause a false restart on first start.
- If you restart from a systemd unit **other** than this one, disable this
  watchdog first - two supervisors fighting over the container is worse than
  the wedge.