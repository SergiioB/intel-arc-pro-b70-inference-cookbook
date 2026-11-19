#!/usr/bin/env bash
# Queue entry, both cards. Usage: repro.sh BIN OUT MODE [PREF [KEY [CTX [REPS]]]]
{ # parsed whole before it runs: an edit while a job runs cannot change that job
set -euo pipefail
: "${SCHED:?Run this script through nightsched}"
if [[ ${B70_TPTF_BOUNDED:-} != 1 ]]; then
    remaining=$(( $(date -d '2026-11-19 14:00:00 CET' +%s) - $(date +%s) - 35 ))
    ((remaining > 0)) || exit 124
    exec timeout --kill-after=30s "${remaining}s" env B70_TPTF_BOUNDED=1 bash "$0" "$@"
fi
BIN=$1 OUT=$2 MODE=$3 PREF=${4:-256} KEY=${5:-code-8k} CTX=${6:-16384} REPS=${7:-3}
TF=$TF
DOCS=~/B70-DOCS
[[ -x $BIN ]] || { echo "No binary $BIN" >&2; exit 3; }
# A failed build leaves the previous binary behind: refuse a binary older than any source file of its tree.
stale=$(find "${BIN%%/zig-out/*}/zig" \( -name '*.zig' -o -name '*.cl' -o -name '*.h' \) -newer "$BIN" -print -quit)
if [[ -n $stale ]]; then
    echo "Stale binary $BIN: $stale is newer" >&2
    exit 3
fi
G=$DOCS/scripts/tmp/breakthrough_20261118/gates/gates.py
if [[ -z ${PROMPTS:-} ]]; then
    PROMPTS=$DOCS/results/2026/11/b70-tonight-20261108/prompts-qwen38.json
    [[ $KEY != code-256k ]] || PROMPTS=$DOCS/results/2026/11/tf-prefix-attn-20261113/prompts-deep.json
fi
BASE=http://127.0.0.1:8046
mkdir -p "$OUT"
export TMPDIR=$WORK/tptf/tmp
export XDG_CACHE_HOME=$WORK/tptf/xdg-cache
export NEO_CACHE_DIR=$WORK/tptf/neo-cache
export PYTHONDONTWRITEBYTECODE=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE XPU_ATTN_PREF="$PREF"
unset ZE_AFFINITY_MASK XPU_KV_INT8
[[ $MODE != resume && $MODE != probe && $MODE != churn || ${KV:-int8} != int8 ]] || export XPU_KV_INT8=1
TP=${TP:-2} # TP=1 serves on card TP1_CARD (default 1)
PCI=(0000:0b:00.0 0000:0f:00.0)
HW=()
CAP=()
for gpu in 0 1; do
    HW[gpu]=$(python3 "$DOCS/tools/host_facts.py" hwmon "$gpu")
    CAP[gpu]=$(cat "${HW[gpu]}/power1_cap")
done
vram() {
    local free
    for pci in "${PCI[@]}"; do
        free=$(sudo -n cat "/sys/kernel/debug/dri/$pci/tile0/vram_mm" | awk '/^visible_avail:/ {print int($2)}')
        [[ $free =~ ^[0-9]+$ && $free -ge $1 ]] || {
            echo "$pci has ${free:-unknown} MiB visible VRAM free. Need $1 MiB." >&2
            return 1
        }
        printf '%s %s MiB\n' "$pci" "$free"
    done
}
stop() {
    if [[ -n ${UNIT:-} ]]; then
        [[ ! -s ${PIDFILE:-} ]] || read -r PID < "$PIDFILE"
        if [[ ! ${PID:-} =~ ^[0-9]+$ ]]; then
            echo "No numeric server PID (${PID:-none}); stopping scope $UNIT" >&2
            PID=
            sudo -n systemctl stop "$UNIT" 2>/dev/null || true
        fi
    fi
    if [[ -n ${PID:-} ]]; then
        kill "$PID" 2>/dev/null || true
        for ((i=0; i<20; i++)); do
            kill -0 "$PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$PID" 2>/dev/null || true
        PID=
    fi
    if [[ -n ${WRAPPER:-} ]]; then
        wait "$WRAPPER" 2>/dev/null || true
        WRAPPER=
        UNIT=
    fi
}
cleanup() {
    stop
    {
        date -Is
        vram 0
        for gpu in 0 1; do
            printf 'gpu%d as-found %s, final %s µW\n' "$gpu" "${CAP[gpu]}" "$(cat "${HW[gpu]}/power1_cap")"
        done
    } > "$OUT/cleanup.txt"
}
trap cleanup EXIT
trap 'exit 143' TERM INT
start() {
    local arm=$1 parallel=$2 cache=$3
    if (( $(date +%s) >= $(date -d '2026-11-19 13:55:00 CET' +%s) )); then
        echo "No GPU startups after 13:55 CET (GPU limit 14:00, lifted from 07:00 by the lead)" >&2
        return 1
    fi
    mkdir -p "$arm"
    awk '/MemAvailable:/ {if ($2 < 20*1024*1024) exit 1}' /proc/meminfo
    if curl -s --connect-timeout 1 "$BASE/health" >/dev/null; then
        echo "Port 8046 is already serving requests" >&2
        return 1
    fi
    if [[ -n $(docker ps -q) ]] || ps -eo comm= | awk '/^(tensorfold-nati|xpu-qwen27|llama-server|llama-bench|vllm|VLLM::EngineCor)$/ {print; found=1} END {exit !found}'; then
        echo "Unrelated inference workload present" >&2
        return 1
    fi
    vram 30000 > "$arm/vram-before.txt"
    local -a args=(serve "$TF/models/Qwen3.8-27B-MLX-4bit" --drafter "$TF/models/Qwen3.8-27B-DFlash2"
        --tp "$TP" --parallel "$parallel" --context "$CTX" --host 127.0.0.1 --port 8046
        --prompt-cache-gib "$cache" --no-thinking --no-update-check)
    {
        date -Is
        printf 'XPU_ATTN_PREF=%s XPU_KV_INT8=%s XPU_ATTN_PREF_SHARED_TREE=%s ' \
            "$PREF" "${XPU_KV_INT8-}" "${XPU_ATTN_PREF_SHARED_TREE-}"
        printf '%q ' "$BIN" "${args[@]}"
        printf '\n'
        sha256sum "$BIN" "$PROMPTS"
        for gpu in 0 1; do
            printf 'gpu%d cap %s µW temp %s m°C\n' "$gpu" "${CAP[gpu]}" "$(cat "${HW[gpu]}/temp2_input")"
        done
    } > "$arm/env.txt"
    UNIT="b70-tptf-$$.scope"
    PIDFILE="$arm/server.pid"
    rm -f "$PIDFILE"
    local -a opts=(TMPDIR="$TMPDIR" NEO_CACHE_DIR="$NEO_CACHE_DIR" XDG_CACHE_HOME="$XDG_CACHE_HOME" XPU_ATTN_PREF="$PREF"
        ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE)
    [[ -z ${XPU_KV_INT8:-} ]] || opts+=(XPU_KV_INT8=1)
    [[ -z ${XPU_ATTN_PREF_SHARED_TREE:-} ]] || opts+=(XPU_ATTN_PREF_SHARED_TREE="$XPU_ATTN_PREF_SHARED_TREE")
    if [[ $MODE == bench && ${arm##*-} == candidate ]]; then
        opts+=("${EXCHANGE_SWITCH:?Set the candidate exchange environment variable}=1")
    fi
    if [[ -n ${EXTRA_ENV:-} ]]; then
        read -ra extra <<< "$EXTRA_ENV"
        for kv in "${extra[@]}"; do
            [[ $kv =~ ^[A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*$ ]] || { echo "Bad EXTRA_ENV item $kv" >&2; return 1; }
        done
        opts+=("${extra[@]}")
    fi
    printf '%q ' "${opts[@]}" >> "$arm/env.txt"
    printf '\n' >> "$arm/env.txt"
    [[ -z ${XPU_CACHE_TRACE_MIN:-} ]] || opts+=(XPU_CACHE_TRACE_MIN="$XPU_CACHE_TRACE_MIN")
    sudo -n systemd-run --scope --quiet --expand-environment=no --unit="$UNIT" \
        -p MemoryMax=28G -p MemorySwapMax=0 -- setpriv --reuid="$(id -u)" --regid="$(id -g)" --init-groups \
        env -u ZE_AFFINITY_MASK $([[ $TP != 1 ]] || echo "ZE_AFFINITY_MASK=${TP1_CARD:-1}") "${opts[@]}" \
        bash -c 'printf "%s\n" "$$" > "$1"; shift; exec "$@"' _ "$PIDFILE" "$BIN" "${args[@]}" \
        > "$arm/server.log" 2>&1 &
    WRAPPER=$!
    for ((i=0; i<90; i++)); do
        if curl -sf --connect-timeout 1 "$BASE/health" >/dev/null; then
            vram "$((parallel > 1 ? 3072 : 1024))" > "$arm/vram-after.txt"
            rg -F "tp=$TP, $parallel streams of $CTX tokens" "$arm/server.log"
            return
        fi
        kill -0 "$WRAPPER" 2>/dev/null || { tail -20 "$arm/server.log"; return 1; }
        sleep 2
    done
    echo "Server failed to become healthy" >&2
    return 1
}

case $MODE in
    resume)
        if [[ -n ${FRESH_DIR:-} ]]; then
            mkdir "$OUT/fresh"
            cp "$FRESH_DIR/"{session.jsonl,server.log} "$OUT/fresh/"
        else
            start "$OUT/fresh" 1 0
            python3 "$G" session --base "$BASE" --prompts "$PROMPTS" --key "$KEY" --turns "$REPS" \
                --max-tokens 128 --out "$OUT/fresh"
            stop
        fi
        start "$OUT/cached" 1 "${CACHE_GIB:-1.5}"
        python3 "$G" replay --base "$BASE" --session "$OUT/fresh/session.jsonl" --out "$OUT/cached" || true
        stop
        python3 "$(dirname "$0")/check.py" resume "$OUT" | tee "$OUT/check.jsonl"
        ;;
    probe)
        # cache-on server only, one per key: does every resumed turn finish? (bisection, no fresh reference)
        IFS=, read -ra keys <<< "$KEY"
        for key in "${keys[@]}"; do
            start "$OUT/$key" 1 "${CACHE_GIB:-5}"
            python3 "$G" session --base "$BASE" --prompts "$PROMPTS" --key "$key" --turns "$REPS" \
                --max-tokens 128 --out "$OUT/$key" || echo "session $key failed"
            stop
            grep -E "prompt cache:|done chatcmpl|ended chatcmpl|DEBUG-tptf" "$OUT/$key/server.log" | sed -E 's/chatcmpl-[0-9a-f]+ //' | cut -c1-200
        done
        ;;
    churn)
        # interleaved conversations (ORDER) on a cache-off server, then the same requests on a cache-on server
        C=$(dirname "$0")/churn.py
        if [[ -z ${REF_DIR:-} ]]; then
            start "$OUT/fresh" 1 0
            python3 "$C" run "$BASE" "$PROMPTS" "${ORDER:-A,B,A,C,B,A}" "$OUT/fresh" "$OUT/fresh/server.log"
            stop
            REF_DIR=$OUT/fresh
        fi
        start "$OUT/cached" 1 "${CACHE_GIB:?Set the cache budget}"
        python3 "$C" replay "$BASE" "$REF_DIR/churn.jsonl" "$OUT/cached" "$OUT/cached/server.log" || true
        stop
        vram 0 > "$OUT/vram-end.txt"
        python3 "$C" check "$REF_DIR" "$OUT/cached" | tee "$OUT/check.jsonl" || true
        ;;
    conc)
        start "$OUT" 4 0
        for arm in draft serial; do
            extra='{}'
            [[ $arm != serial ]] || extra='{"draft":false}'
            python3 "$G" conc --base "$BASE" --prompts "$PROMPTS" --keys code-0,chat-0,code-8k,chat-8k \
                --max-tokens 512 --tag "$arm" --reps "$REPS" --extra "$extra" --out "$OUT"
        done
        stop
        python3 "$(dirname "$0")/check.py" conc "$OUT" | tee "$OUT/check.jsonl"
        ;;
    bench)
        for ((pair=1; pair<=REPS; pair++)); do
            arms=(control candidate)
            ((pair % 2)) || arms=(candidate control)
            for arm in "${arms[@]}"; do
                dir="$OUT/pair$pair-$arm"
                start "$dir" 1 0
                python3 "$(dirname "$0")/bench.py" record "$BASE" "$pair" "$KEY" "$dir"
                stop
            done
        done
        python3 "$(dirname "$0")/bench.py" compare "$OUT" "$REPS" "$KEY" | tee "$OUT/pairs.json"
        ;;
    *)
        echo "Unknown mode $MODE" >&2
        exit 2
        ;;
esac
exit
}
