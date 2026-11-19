#!/usr/bin/env bash
# Crash diagnosis for the one-card llama-bench segfault (2026-11-19 07:25). Card 1 only, not timed.
# Arms: A = measured build, exact failing flags, under gdb; B = same with -lm mmap; C = lab's older build
# (llama.cpp-latest d5bbcd1f3) with the failing flags.
set -uo pipefail
[[ ${SCHED:-} == 1 ]] || exit 2
ulimit -c 0
campaign=$(dirname "$(readlink -f "$0")")
out=$campaign/evidence/diag
mkdir -p "$out"
export TMPDIR="$campaign/tmp" XDG_CACHE_HOME="$campaign/cache" SYCL_CACHE_DIR="$campaign/cache/sycl"
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null; set -u
export ONEAPI_DEVICE_SELECTOR='level_zero:*' SYCL_ENABLE_DEFAULT_CONTEXTS=1 ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export UR_LOADER_USE_LEVEL_ZERO_V2=0 SYCL_CACHE_PERSISTENT=1 GGML_SYCL_ENABLE_GRAPH=0 ZE_AFFINITY_MASK=1
unset GGML_SYCL_TP_P2P GGML_SYCL_TP_DIAG_SKIP
model=$DATA/sata2/gguf/qwen38-27b-Q4_K_M.gguf
vram=$(sudo -n cat /sys/kernel/debug/dri/0000:0f:00.0/tile0/vram_mm | awk '/^visible_avail:/ { sub(/MiB$/, "", $2); print $2 }')
echo "card1 visible_avail_MiB=$vram"; (( vram >= 24000 )) || { echo "card 1 busy"; exit 3; }
ps -eo pid,comm | awk '$2 ~ /^(llama-|tensorfold|vllm|VLLM)/'
flags=(-m "$model" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -b 512 -ub 512 -t 8 -dev SYCL0 -sm none -p 512 -n 128 -r 1 -o json)
arm() {
    local name=$1 bin=$2; shift 2
    echo "== arm $name $(date -Is)"
    timeout --kill-after=10s 300s gdb -batch -nx -ex 'set pagination off' -ex 'set print thread-events off' \
        -ex run -ex 'bt 40' -ex 'info registers rip' -ex 'thread apply all bt 6' \
        --args "$bin" "${flags[@]}" "$@" > "$out/$name.gdb.txt" 2>&1
    echo "rc=$? $(date -Is)"
    grep -E "SIGSEGV|SIGBUS|exited|^#[0-9]+ " "$out/$name.gdb.txt" | head -25
}
for a in ${@:-A B C}; do
    case "$a" in
        A) arm A-measured-lm-none "$campaign/llama.cpp/build/bin/llama-bench" -lm none ;;
        B) arm B-measured-lm-mmap "$campaign/llama.cpp/build/bin/llama-bench" -lm mmap ;;
        C) arm C-old-d5bbcd1f3-lm-none $ENGINES/llama.cpp-latest/build-sycl/bin/llama-bench -lm none ;;
        U) arm U-upstream-78e05298-lm-none "$campaign/llama-upstream/build/bin/llama-bench" -lm none ;;
        P0) SYCL_CACHE_PERSISTENT=0 arm P0-measured-cache-off "$campaign/llama.cpp/build/bin/llama-bench" -lm none ;;
        P1) rm -rf "$campaign/cache/sycl-fresh"; mkdir -p "$campaign/cache/sycl-fresh"
            SYCL_CACHE_DIR="$campaign/cache/sycl-fresh" arm P1-measured-cache-on-fresh-dir "$campaign/llama.cpp/build/bin/llama-bench" -lm none ;;
    esac
done
