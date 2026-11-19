# Sourced by jobs under nightsched.py.
set -euo pipefail
[[ ${SCHED:-0} == 1 ]]
repo=~/B70-DOCS
work=$WORK/tpdiag
limit=$(date -d '2026-11-19 14:00:00 CET' +%s)  # target was 04:00 CET; lead moved it to 06:00 (03:08) and 08:30 (05:50)
(( $(date +%s) < limit - 15 )) || { echo "14:00 CET cutoff"; exit 124; }
export TMPDIR=$work/tmp XDG_CACHE_HOME=$work/cache
hw=( "$(python3 "$repo/tools/host_facts.py" hwmon 0)" "$(python3 "$repo/tools/host_facts.py" hwmon 1)" )
record() {
  date -Is
  for i in 0 1; do
    printf 'card %s %s\n' "$i" "${hw[$i]}"
    cat "${hw[$i]}"/{power1_cap,temp2_input}
    cat "${hw[$i]}"/device/tile0/gt0/freq0/max_freq
  done
}
run_bounded() {
  local seconds=$1 remaining=$((limit - $(date +%s) - 15))
  shift
  (( remaining > 0 )) || return 124
  (( seconds <= remaining )) || seconds=$remaining
  timeout -k 5s "${seconds}s" "$@"
}
idle() {
  docker ps --format '{{.Image}} {{.Names}}'
  if [[ -n $(docker ps -q) ]] || ps -eo comm= | awk '/^(tensorfold-nati|xpu-qwen27|llama-server|llama-bench|vllm|VLLM::EngineCor|p2p_direct|tpdiag-ze)$/ {print; found=1} END {exit !found}'; then
    echo 'GPU preflight: unrelated workload present'
    return 1
  fi
  for pci in 0b 0f; do
    sudo -n awk '/^visible_avail:/ {print FILENAME, $0; ok=($2+0 >= 31000)} END {exit !ok}' "/sys/kernel/debug/dri/0000:$pci:00.0/tile0/vram_mm"
  done
  awk '/MemAvailable:/ {print; exit ($2 < 20971520)}' /proc/meminfo
}
cool() {
  # Lead edit 2026-11-19 03:05: card 1 drives the desktop and idles at 44 W / 66 C, so a 55 C gate on it
  # always ran into the 240 s bound (4 min lost per arm). Card 0 <= 55 C, card 1 <= 68 C, bound 120 s.
  run_bounded 120 bash -c '
    while (( $(cat "$1/temp2_input") > 55000 || $(cat "$2/temp2_input") > 68000 )); do sleep 2; done
  ' _ "${hw[@]}" || echo "cool-down bound reached: $(cat "${hw[0]}/temp2_input") / $(cat "${hw[1]}/temp2_input") mC"
}
