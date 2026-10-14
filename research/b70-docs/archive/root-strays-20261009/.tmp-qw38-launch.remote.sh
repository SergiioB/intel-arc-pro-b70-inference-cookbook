#!/usr/bin/env bash
# Qwen3.8-27B official FP8 — TP1/PP2 dual-B70 serving launch WITH per-worker
# ZE_AFFINITY_MASK (host-RAM fix validation, 2026-08-27).
#
# Variant of qw38-fp8-pp2-serving-campaign.sh with exactly one functional
# change: the container is launched with --cap-add SYS_PTRACE so oneCCL can
# use its pidfd IPC path. With per-rank single-GPU masks (installed by
# scripts/tmp/patch_vllm_worker_affinity.py), oneCCL's drmfd fallback
# deadlocks; pidfd requires SYS_PTRACE under docker's default seccomp
# (verified via scripts/tmp/xccl_mask_minirepro.py, 2026-08-27).
# All watchdogs, gates, monitors, and cleanup are unchanged from the parent
# script. Every other delta is header/comment only.
#
# WHEN TO USE: internal capacity/correctness research on the dual-B70 host.
# WHAT IT DOES: launches the pinned vLLM-XPU image at TP1/PP2 with the
# per-worker affinity patch. On health + settle it LEAVES THE SERVER RUNNING
# for a later benchmark/close-out phase, writing server-live.json.
# OUTPUT: results/qw38-fp8-pp2-affval*/ with manifest, logs, 1 Hz monitor,
# rank markers, VRAM snapshots, replay records, smoke verdict, cleanup state.
# SAFETY: aborts below 3 GiB MemAvailable, on swap growth beyond allowance,
# timeout, dead container, or post-load <500 MiB on either GPU (fail-closed;
# identical gates to the parent script — none relaxed). On ANY failure or
# Ctrl-C removes the container and restores both cards to 150 W. On SUCCESS
# the close-out owner must restore 150 W and tear down.
# EXAMPLE: MAXLEN=9216 bash scripts/tmp/qw38-fp8-pp2-affinity-launch.sh
# CLOSEOUT (after benchmarks):
#   sudo docker rm -f qw38fp8-pp2-affval
#   printf 150000000 | sudo tee /sys/class/hwmon/<gpu-hwmon>/power1_cap  # both cards
set -euo pipefail

ROOT=~/B70-DOCS
UTC=$(date -u +%Y%m%dT%H%M%SZ)
OUT=${OUT:-$ROOT/results/qw38-fp8-pp2-serving-$UTC}
IMG='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
MODEL=${MODEL_DIR:-/mnt/<models-disk>/hf/qw38-fp8}
MODEL_TAG=${MODEL_TAG:-qw38-fp8}
MODEL_BYTES=${MODEL_BYTES:-30890081596}
QUANT_FLAGS=${QUANT_FLAGS:---quantization fp8 --dtype bfloat16}
KVDT_FLAGS=${KVDT_FLAGS:-}
PREFIX_FLAGS=${PREFIX_FLAGS:---no-enable-prefix-caching}
PATCH=$ROOT/scripts/tmp/patch_vllm_worker_affinity.py
CTN=qw38fp8-pp2-affval
PORT=${PORT:-8903}
MAXLEN=${MAXLEN:-16384}
MAXSEQS=${MAXSEQS:-4}
BATCHED=${BATCHED:-$MAXLEN}
GPUUTIL=${GPUUTIL:-0.90}
FLOOR_MB=${FLOOR_MB:-3072}
SWAP_GROWTH_MB=${SWAP_GROWTH_MB:-1024}
STARTUP_TIMEOUT_S=${STARTUP_TIMEOUT_S:-1200}
KEEP_ON_SUCCESS=${KEEP_ON_SUCCESS:-1}
EXTRA_VLLM_ARGS=${EXTRA_VLLM_ARGS:-}
AFFINITY=${AFFINITY:-0,1}
TPSIZE=${TPSIZE:-1}
PPSIZE=${PPSIZE:-2}
GATE_CARDS=${GATE_CARDS:-both}
KEEP_SERIAL=${KEEP_SERIAL:-0}
CAMPAIGN_CAP_UW=${CAMPAIGN_CAP_UW:-150000000}
MODEL_REPOSITORY=${MODEL_REPOSITORY:-Qwen/Qwen3.8-27B-FP8}
MODEL_REVISION=${MODEL_REVISION:-unknown}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}
JOB="qw38-fp8-pp2-affval-$UTC"
# operator overrides may redefine OUT/CTN/etc., so they must load before mkdir
if [ -f /tmp/qw38-campaign.env ]; then . /tmp/qw38-campaign.env; fi
mkdir -p "$OUT"

log() { printf '[milestone %s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/milestones.log"; }
mem_avail() { awk '/MemAvailable:/{print int($2/1024)}' /proc/meminfo; }
mem_cached() { awk '/^Cached:/{print int($2/1024)}' /proc/meminfo; }
swap_used() { awk '/SwapTotal:/{t=$2}/SwapFree:/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo; }
visible_free() { sudo -n awk '/visible_avail/{gsub(/MiB/,"");print $2}' "/sys/kernel/debug/dri/$1/tile0/vram_mm"; }
find_hwmon() {
  local bdf=$1 h
  for h in /sys/class/hwmon/hwmon*; do
    [ "$(cat "$h/name" 2>/dev/null || true)" = xe ] || continue
    case "$(readlink -f "$h/device" 2>/dev/null || true)" in *"$bdf"*) printf '%s\n' "$h"; return 0;; esac
  done
  return 1
}

H0=$(find_hwmon 0000:0b:00.0)
H1=$(find_hwmon 0000:0f:00.0)
capture_evidence() {
  set +e
  if sudo -n docker inspect "$CTN" >/dev/null 2>&1; then
    sudo -n docker logs "$CTN" > "$OUT/server.log" 2>&1 || true
    sudo -n docker cp "$CTN:/dev/shm/b70-pp-serial-materialize" "$OUT/rank-markers" >/dev/null 2>&1 || true
  elif [ -d /dev/shm/b70-pp-serial-materialize ]; then
    mkdir -p "$OUT/rank-markers"
    cp -a /dev/shm/b70-pp-serial-materialize/. "$OUT/rank-markers/" 2>/dev/null || true
  fi
  memscene
}
memscene() {
  local dir="$OUT/crash-scene"
  mkdir -p "$dir"
  cp /proc/meminfo "$dir/meminfo.txt"
  cat /proc/vmstat > "$dir/vmstat.txt"
  sudo -n cat /proc/slabinfo > "$dir/slabinfo.txt" 2>/dev/null || true
  sudo -n cat /proc/buddyinfo > "$dir/buddyinfo.txt" 2>/dev/null || true
  sudo -n dmesg -T 2>/dev/null | tail -80 > "$dir/dmesg-tail.txt" || true
  sudo -n docker top "$CTN" -eo pid,ppid,nlwp,rss,vsz,comm,args > "$dir/docker-top.txt" 2>/dev/null || true
  # per-process forensic rollup for the heaviest pids in the container cgroup
  local cg
  cg=$(sudo -n docker inspect -f '{{.Id}}' "$CTN" 2>/dev/null || true)
  if [ -n "$cg" ] && [ -f "/sys/fs/cgroup/system.slice/docker-${cg}.scope/cgroup.procs" ]; then
    : > "$dir/proc-status.txt"
    while read -r p; do
      sudo -n grep -E '^(Name|Pid|PPid|Threads|VmRSS|VmHWM|VmLck|VmPin|VmSwap|RssAnon|RssFile|RssShmem):' "/proc/$p/status" 2>/dev/null >> "$dir/proc-status.txt" && echo "---" >> "$dir/proc-status.txt"
      printf 'PID %s smaps_rollup:\n' "$p" >> "$dir/proc-status.txt"
      sudo -n cat "/proc/$p/smaps_rollup" >> "$dir/proc-status.txt" 2>/dev/null
      echo "===" >> "$dir/proc-status.txt"
    done < "/sys/fs/cgroup/system.slice/docker-${cg}.scope/cgroup.procs"
  fi
  # global scan for any process holding pinned pages
  : > "$dir/global-vmpin-nonzero.txt"
  for st in /proc/[0-9]*/status; do
    p=$(awk '/^Pid:/{print $2}' "$st" 2>/dev/null); n=$(awk '/^Name:/{print $2}' "$st" 2>/dev/null)
    v=$(awk '/^VmPin:/{print $2}' "$st" 2>/dev/null)
    l=$(awk '/^VmLck:/{print $2}' "$st" 2>/dev/null)
    if [ "${v:-0}" != 0 ] || [ "${l:-0}" != 0 ]; then
      echo "pid=$p name=$n VmPin_kB=$v VmLck_kB=$l" >> "$dir/global-vmpin-nonzero.txt"
    fi
  done
  sudo -n ls -la /sys/kernel/debug/dri/0000:0b:00.0/ > "$dir/gem-debug-gpu0.txt" 2>/dev/null || true
}
teardown_container() {
  set +e
  [ -n "${MON_PID:-}" ] && kill "$MON_PID" >/dev/null 2>&1 || true
  [ -n "${LOG_PID:-}" ] && kill "$LOG_PID" >/dev/null 2>&1 || true
  [ -n "${KPID:-}" ] && kill "$KPID" >/dev/null 2>&1 || true
  [ -n "${PPID_WATCH:-}" ] && kill "$PPID_WATCH" >/dev/null 2>&1 || true
  capture_evidence
  if sudo -n docker inspect "$CTN" >/dev/null 2>&1; then
    sudo -n docker rm -f "$CTN" >/dev/null 2>&1 || true
  fi
  [ -n "${SWAPPINESS_OLD:-}" ] && printf %s "$SWAPPINESS_OLD" | sudo -n tee /proc/sys/vm/swappiness >/dev/null || true
  printf 150000000 | sudo -n tee "$H0/power1_cap" >/dev/null
  printf 150000000 | sudo -n tee "$H1/power1_cap" >/dev/null
  {
    echo "ended_utc=$(date -u +%FT%TZ)"
    echo "container_present=$(sudo -n docker inspect "$CTN" >/dev/null 2>&1 && echo yes || echo no)"
    echo "gpu0_cap=$(cat "$H0/power1_cap")"
    echo "gpu1_cap=$(cat "$H1/power1_cap")"
    echo "gpu0_visible_avail=$(visible_free 0000:0b:00.0)"
    echo "gpu1_visible_avail=$(visible_free 0000:0f:00.0)"
  } > "$OUT/cleanup.txt"
}
on_error() { teardown_container; }
if [ "$KEEP_ON_SUCCESS" = 1 ]; then
  trap on_error EXIT INT TERM
else
  trap teardown_container EXIT INT TERM
fi

log "preflight out=$OUT strategy=per_worker_affinity len=$MAXLEN seqs=$MAXSEQS util=$GPUUTIL campaign_cap=$((CAMPAIGN_CAP_UW/1000000))W"
[ -f "$MODEL/config.json" ] && [ -f "$MODEL/model.safetensors.index.json" ]
[ -f "$PATCH" ]
[ "$(du -sb "$MODEL" | cut -f1)" -eq "$MODEL_BYTES" ]
pgrep -af 'llama-server|llama-bench|vllm|vllm-xpu' | grep -v "$0" > "$OUT/preflight-processes.txt" && { log 'abort inference process present'; exit 1; } || true
[ -z "$(sudo -n docker ps --format '{{.Names}}')" ] || { log 'abort running container present'; exit 1; }
if [ "$GATE_CARDS" = both ]; then GATE_BDFS="0000:0b:00.0 0000:0f:00.0"; else GATE_BDFS="$GATE_CARDS"; fi
for bdf in $GATE_BDFS; do
  f=$(visible_free "$bdf"); log "gate $bdf visible_avail=${f}MiB"; [ "$f" -ge 31000 ] || exit 1
done
printf %s "$CAMPAIGN_CAP_UW" | sudo -n tee "$H0/power1_cap" >/dev/null
printf %s "$CAMPAIGN_CAP_UW" | sudo -n tee "$H1/power1_cap" >/dev/null
SWAPPINESS_OLD=$(cat /proc/sys/vm/swappiness)
printf 10 | sudo -n tee /proc/sys/vm/swappiness >/dev/null || true
log "vm.swappiness $SWAPPINESS_OLD -> 10 for load window"

SWAP0=$(swap_used)
PREFIX_CACHE=true
[[ "$PREFIX_FLAGS" == *--no-enable-prefix-caching* ]] && PREFIX_CACHE=false
SPECULATION=false
[[ "$EXTRA_VLLM_ARGS" == *--speculative-config* ]] && SPECULATION=true
jq -n \
  --arg status "PROVISIONAL_NOT_FOR_PUBLIC_HEADLINE" \
  --arg classification "TP${TPSIZE}_PP${PPSIZE}_serving_per_worker_affinity_validation" \
  --arg mode_affinity "$AFFINITY (per-worker override installed)" \
  --arg mode_tp_pp "TP=$TPSIZE PP=$PPSIZE" \
  --arg serial_overlay_enabled "$KEEP_SERIAL" \
  --arg extra_vllm_args "${EXTRA_VLLM_ARGS:-none}" \
  --arg model_repository "$MODEL_REPOSITORY" \
  --arg model_revision "$MODEL_REVISION" \
  --arg model_path "$MODEL" \
  --argjson model_bytes "$MODEL_BYTES" \
  --arg image "$IMG" \
  --argjson tensor_parallel_size "$TPSIZE" \
  --argjson pipeline_parallel_size "$PPSIZE" \
  --argjson max_model_len "$MAXLEN" \
  --argjson max_num_seqs "$MAXSEQS" \
  --argjson max_num_batched_tokens "$BATCHED" \
  --argjson prefix_cache "$PREFIX_CACHE" \
  --argjson speculation "$SPECULATION" \
  --argjson graphs "${XPU_GRAPH:-0}" \
  --argjson enforce_eager "${EAGER:-1}" \
  --arg kv_cache_dtype "$KV_CACHE_DTYPE" \
  --arg load_strategy "default_streaming_per_worker_ze_affinity_mask" \
  --arg strategy_change_vs_blocked_runs "per-worker ZE_AFFINITY_MASK=rank at spawn + SYS_PTRACE for oneCCL pidfd IPC" \
  --arg overlay_sha256 "$(sha256sum "$PATCH" | cut -d' ' -f1)" \
  --argjson host_mem_floor_MiB "$FLOOR_MB" \
  --argjson max_swap_growth_MiB "$SWAP_GROWTH_MB" \
  --argjson swap_used_start_MiB "$SWAP0" \
  --argjson configured_power_cap_W "$((CAMPAIGN_CAP_UW/1000000))" \
  --arg cache_drop_before_launch "${DROP_CACHES:-0}" \
  '{status:$status,classification:$classification,mode_affinity:$mode_affinity,mode_tp_pp:$mode_tp_pp,serial_overlay_enabled:$serial_overlay_enabled,extra_vllm_args:$extra_vllm_args,model_repository:$model_repository,model_revision:$model_revision,model_path:$model_path,model_bytes:$model_bytes,image:$image,observed_vllm_version_in_server_log:null,tensor_parallel_size:$tensor_parallel_size,pipeline_parallel_size:$pipeline_parallel_size,max_model_len:$max_model_len,max_num_seqs:$max_num_seqs,max_num_batched_tokens:$max_num_batched_tokens,prefix_cache:$prefix_cache,speculation:$speculation,graphs:($graphs == 1),enforce_eager:($enforce_eager == 1),kv_cache_dtype:$kv_cache_dtype,load_strategy:$load_strategy,strategy_change_vs_blocked_runs:$strategy_change_vs_blocked_runs,overlay_sha256:$overlay_sha256,host_mem_floor_MiB:$host_mem_floor_MiB,max_swap_growth_MiB:$max_swap_growth_MiB,swap_used_start_MiB:$swap_used_start_MiB,configured_power_cap_W:$configured_power_cap_W,measured_draw:"not_measured",cache_drop_before_launch:$cache_drop_before_launch}' > "$OUT/manifest.json"

[ "${DROP_CACHES:-0}" = 1 ] && { sync; printf 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null || true; }
sync
PATCH_SEG="python /patch_pp_serial.py; "
[ "${SKIP_PATCH:-0}" = 1 ] && PATCH_SEG=""
log "launch TP=$TPSIZE PP=$PPSIZE serving maxlen=$MAXLEN mem_available=$(mem_avail) cached=$(mem_cached) swap_used=$SWAP0 campaign_cap=$((CAMPAIGN_CAP_UW/1000000))W"
GID=$(stat -c '%g' /dev/dri/render* | sort -u | head -1)
sudo -n docker run -d --name "$CTN" -p "$PORT:8000" ${CGMEM:+--memory "$CGMEM"} \
  --cap-add SYS_PTRACE \
  --device /dev/dri --group-add "$GID" --ipc=host \
  -v /dev/dri:/dev/dri:ro \
  -v "$MODEL:/models/$MODEL_TAG:ro" \
  -v "$PATCH:/patch_pp_serial.py:ro" \
  -e VLLM_TARGET_DEVICE=xpu \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=${XPU_GRAPH:-0} \
  -e ZE_AFFINITY_MASK=$AFFINITY \
  $( [ "${FLAT:-0}" = 1 ] && printf '%s' "-e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE" ) \
  -e VLLM_USE_V1=1 \
  -e B70_WORKER_AFFINITY=1 \
  -e B70_PP_SERIAL_MATERIALIZE=$KEEP_SERIAL \
  -e B70_PP_SERIAL_JOB="$JOB" \
  -e B70_PP_SERIAL_TIMEOUT_S=900 \
  -e MALLOC_ARENA_MAX=4 \
  -e OMP_NUM_THREADS=8 \
  -e B70_PP_ITERATOR_TRIM=${ITER_TRIM:-0} \
  -e B70_PP_ITERATOR_TRIM_BYTES=${ITER_TRIM_BYTES:-402653184} \
  --entrypoint bash "$IMG" -lc \
  "set -e; rm -rf /dev/shm/b70-pp-serial-materialize; ${PATCH_SEG}exec vllm serve /models/$MODEL_TAG \
    --served-model-name qw38-fp8 $QUANT_FLAGS \
    --tensor-parallel-size $TPSIZE --pipeline-parallel-size $PPSIZE \
    --max-model-len $MAXLEN --max-num-seqs $MAXSEQS --max-num-batched-tokens $BATCHED \
    --gpu-memory-utilization $GPUUTIL $( [ "${EAGER:-1}" = 1 ] && echo --enforce-eager ) --language-model-only \
    $KVDT_FLAGS $PREFIX_FLAGS $EXTRA_VLLM_ARGS \
    --port 8000 --host 0.0.0.0" > "$OUT/docker-run.out"

sudo -n docker logs -f "$CTN" >> "$OUT/server.log" 2>&1 & LOG_PID=$!
(
  printf 'utc,unevictable_mib,mlocked_mib,sreclaimable_mib,sunreclaim_mib,shmem_mib,vmallocused_mib,committed_as_gib\n' > "$OUT/kernel-mem.csv"
  while [ "$(sudo -n docker inspect -f '{{.State.Running}}' "$CTN" 2>/dev/null || echo false)" = true ]; do
    u=$(awk '/Unevictable:/{print int($2/1024)}' /proc/meminfo)
    ml=$(awk '/Mlocked:/{print int($2/1024)}' /proc/meminfo)
    sr=$(awk '/SReclaimable:/{print int($2/1024)}' /proc/meminfo)
    su=$(awk '/SUnreclaim:/{print int($2/1024)}' /proc/meminfo)
    sh=$(awk '/Shmem:/{print int($2/1024)}' /proc/meminfo)
    vu=$(awk '/VmallocUsed:/{print int($2/1024)}' /proc/meminfo)
    ca=$(awk '/Committed_AS/{printf "%.1f",$2/1048576}' /proc/meminfo)
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$(date -u +%FT%TZ)" "${u:-0}" "${ml:-0}" "${sr:-0}" "${su:-0}" "${sh:-0}" "${vu:-0}" "${ca:-0}" >> "$OUT/kernel-mem.csv"
    sleep ${RECORDER_S:-10}
  done
) & KPID=$!
(
  while [ "$(sudo -n docker inspect -f '{{.State.Running}}' "$CTN" 2>/dev/null || echo false)" = true ]; do
    {
      printf '#%s\n' "$(date -u +%FT%TZ)"
      sudo -n docker top "$CTN" -eo pid,nlwp,rss,comm 2>/dev/null | sort -k3 -n -r | head -6
      for st in /proc/[0-9]*/status; do
        v=$(awk '/^VmPin:/{print $2}' "$st" 2>/dev/null); l=$(awk '/^VmLck:/{print $2}' "$st" 2>/dev/null)
        if [ "${v:-0}" != 0 ] || [ "${l:-0}" != 0 ]; then
          echo "VMPIN pid=$(awk '/^Pid:/{print $2}' "$st") name=$(awk '/^Name:/{print $2}' "$st") pin_kB=$v lock_kB=$l"
        fi
      done
    } >> "$OUT/procs-watch.txt" 2>/dev/null
    sleep ${RECORDER_S:-10}
  done
) & PPID_WATCH=$!
(
  printf 'utc,mem_available_mib,cached_mib,swap_used_mib,gpu0_visible_mib,gpu1_visible_mib,gpu0_pkg_w,gpu1_pkg_w,container_mem_mib,container_running\n' > "$OUT/monitor.csv"
  e0=$(cat "$H0/energy1_input"); e1=$(cat "$H1/energy1_input"); t_prev=$(date +%s.%N)
  while sleep "${RECORDER_S:-10}"; do
    now=$(date +%s.%N); dt=$(awk -v a="$t_prev" -v b="$now" 'BEGIN{print b-a}')
    m=$(mem_avail); c=$(mem_cached); s=$(swap_used)
    g0=$(visible_free 0000:0b:00.0); g1=$(visible_free 0000:0f:00.0)
    en0=$(cat "$H0/energy1_input"); en1=$(cat "$H1/energy1_input")
    w0=$(awk -v d0="$en0" -v e0="$e0" -v dt="$dt" 'BEGIN{print (dt>0)?sprintf("%.1f",(d0-e0)/dt/1000000):0}')
    w1=$(awk -v d1="$en1" -v e1="$e1" -v dt="$dt" 'BEGIN{print (dt>0)?sprintf("%.1f",(d1-e1)/dt/1000000):0}')
    e0=$en0; e1=$en1; t_prev=$now
    running=$(sudo -n docker inspect -f '{{.State.Running}}' "$CTN" 2>/dev/null || echo false)
    printf '%s,%s,%s,%s,%s,%s,%s,%s,monitor_off\n' "$(date -u +%FT%TZ)" "$m" "$c" "$s" "$g0" "$g1" "$w0" "$w1" >> "$OUT/monitor.csv"
    if [ "$m" -lt "$FLOOR_MB" ]; then
      log "watchdog MemAvailable=${m}MiB below ${FLOOR_MB}MiB"
      printf 'blocked_host_ram mem_available_mib=%s\n' "$m" > "$OUT/watchdog-verdict.txt"
      capture_evidence
      sudo -n docker stop -t 3 "$CTN" >/dev/null 2>&1 || true
      break
    fi
    if [ $((s - SWAP0)) -gt "$SWAP_GROWTH_MB" ]; then
      log "watchdog swap growth=$((s-SWAP0))MiB"
      printf 'blocked_swap_thrashing growth_mib=%s\n' "$((s-SWAP0))" > "$OUT/watchdog-verdict.txt"
      capture_evidence
      sudo -n docker stop -t 3 "$CTN" >/dev/null 2>&1 || true
      break
    fi
    [ "$running" = true ] || break
  done
) & MON_PID=$!

start=$(date +%s); ready=0
while [ $(( $(date +%s) - start )) -lt "$STARTUP_TIMEOUT_S" ]; do
  if curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
  running=$(sudo -n docker inspect -f '{{.State.Running}}' "$CTN" 2>/dev/null || echo false)
  [ "$running" = true ] || break
  sleep 3
done
capture_evidence
if [ "$ready" -ne 1 ]; then
  log "server failed before health verdict=$(cat "$OUT/watchdog-verdict.txt" 2>/dev/null || echo startup_failed_or_timeout)"
  [ -f "$OUT/watchdog-verdict.txt" ] || printf 'startup_failed_or_timeout\n' > "$OUT/watchdog-verdict.txt"
  exit 2
fi
log "server healthy mem_available=$(mem_avail) cached=$(mem_cached) swap_used=$(swap_used)"
if [ "${SETTLE_CHECK:-1}" = 1 ]; then
  log "settle check begin (90s): need end avail >= $((FLOOR_MB+900))MiB, min avail >= FLOOR, swap slope <=96MiB over final 30s"
  s_start=$(swap_used); m_min=$(mem_avail); s_12=
  for i in $(seq 1 18); do
    sleep 5
    s_now=$(swap_used); m_now=$(mem_avail)
    printf '%s,settle,%s,%s\n' "$(date -u +%FT%TZ)" "$m_now" "$s_now" >> "$OUT/settle.csv"
    [ "$m_now" -lt "$m_min" ] && m_min=$m_now
    if [ "$i" -ge 12 ]; then s_12=${s_12:-$s_now}; fi
  done
  s_end=$(swap_used); m_end=$(mem_avail)
  slope=$((s_end - ${s_12:-$s_start}))
  log "settle done avail=$m_end min=$m_min swap=$s_end slope=${slope}MiB"
  if [ "$m_end" -ge $((FLOOR_MB+900)) ] && [ "$m_min" -ge "$FLOOR_MB" ] && [ "$slope" -le 96 ]; then
    log "settle PASSED"
  else
    printf 'settle_failed avail=%s min=%s swap=%s slope=%s\n' "$m_end" "$m_min" "$s_end" "$slope" > "$OUT/watchdog-verdict.txt"
    log "SETTLE FAILED"
    exit 6
  fi
fi
log "correctness replay passed"
sudo -n docker logs "$CTN" > "$OUT/server.log" 2>&1 || true
for bdf in $GATE_BDFS; do
  sudo -n cat "/sys/kernel/debug/dri/$bdf/tile0/vram_mm" > "$OUT/vram-after-request-${bdf//:/_}.txt"
done
printf 'accepted_load_path_server_left_running\n' > "$OUT/verdict.txt"
log "DONE accepted_load_path — server left running on port $PORT for benchmark phase"
kill "$LOG_PID" >/dev/null 2>&1 || true
kill "$MON_PID" >/dev/null 2>&1 || true
kill "$KPID" >/dev/null 2>&1 || true
kill "$PPID_WATCH" >/dev/null 2>&1 || true
if [ "$KEEP_ON_SUCCESS" = 1 ]; then
  trap - EXIT INT TERM
  log "exit0 keeper-mode: container=$CTN remains; recorders terminated (no orphan watchdogs); closeout owner must restore 150W and remove $CTN"
else
  teardown_container
fi
