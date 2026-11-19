#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
TF=$TF
SRC=$TF/src-tp3586
BIN=$SRC/zig-out/bin/xpu-qwen27
IDS=$work/../ids
OUT=$work/raw/tf
mkdir -p "$OUT"
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null
set -u
export XPU_ATTN_PREF=256 XPU_CTX=40960 XPU_DRAFTER=$TF/models/Qwen3.8-27B-DFlash2 XPU_TREE=15
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export NEO_CACHE_DIR=$work/cache
unset XPU_PROFILE XPU_TP ZE_AFFINITY_MASK
runtime_env=(LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" TMPDIR="$TMPDIR" XDG_CACHE_HOME="$XDG_CACHE_HOME"
  XPU_ATTN_PREF="$XPU_ATTN_PREF" XPU_CTX="$XPU_CTX" XPU_DRAFTER="$XPU_DRAFTER" XPU_TREE="$XPU_TREE"
  ZE_FLAT_DEVICE_HIERARCHY="$ZE_FLAT_DEVICE_HIERARCHY" NEO_CACHE_DIR="$NEO_CACHE_DIR")
git -C "$SRC" rev-parse HEAD
git -C "$SRC" diff --stat
sha256sum "$BIN" "$IDS"/code-{0,32k}.ids
trap record EXIT
# Interleaved one-card (card 1) / two-card pairs per depth, then one XPU_PROFILE run each,
# then (depth 0 only) a same-binary TP2 screen with NEO direct submission forced on and off.
declare -A plan=(
  [0]="wall-1 wall-2 wall-2 wall-1 wall-1 wall-2 profile-1 profile-2 dson-2 dsoff-2"
  [32k]="wall-1 wall-2 wall-2 wall-1 profile-1 profile-2"
)
# Override for follow-up screens: TF_DEPTHS="0" TF_PLAN_0="..." (tags get the TF_TAG prefix).
[[ -n ${TF_PLAN_0:-} ]] && plan[0]=$TF_PLAN_0
for depth in ${TF_DEPTHS:-0 32k}; do
  n=0
  for arm in ${plan[$depth]}; do
    n=$((n + 1))
    tag=${TF_TAG:-}$depth-$n-$arm
    cool && idle
    record > "$OUT/$tag.telemetry"
    modes=()
    [[ $arm == *-1 ]] && modes+=(ZE_AFFINITY_MASK=1)
    [[ $arm == profile-* ]] && modes+=(XPU_PROFILE=1)
    [[ $arm == dson-* ]] && modes+=(NEOReadDebugKeys=1 EnableDirectSubmission=1 DirectSubmissionOverrideComputeSupport=1 DirectSubmissionOverrideBlitterSupport=1)
    [[ $arm == dsoff-* ]] && modes+=(NEOReadDebugKeys=1 EnableDirectSubmission=0)
    # The dev runner runs serial first and checks the drafted result against it.
    # XPU_TP is the dev-runner equivalent of the served --tp option.
    if run_bounded 240 sudo -n systemd-run --scope --quiet -p MemoryMax=20G -p MemorySwapMax=0 --uid=sergio --gid=sergio -- \
        env "${runtime_env[@]}" "${modes[@]}" XPU_TP="${arm##*-}" "$BIN" "$TF/models/Qwen3.8-27B-MLX-4bit" \
        "$IDS/code-$depth.ids" 257 > "$OUT/$tag.log" 2>&1; then
      echo "$tag ok $(grep -E '^(decode|drafted):' "$OUT/$tag.log" | tr '\n' ' ')"
    else
      rc=$?
      echo "$tag exit $rc" | tee -a "$OUT/$tag.log"
    fi
    record >> "$OUT/$tag.telemetry"
  done
done

