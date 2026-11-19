#!/usr/bin/env bash
{ # Queue entry, both cards, exclusive. Same-binary A/B of one env flag on TP2, interleaved pairs.
  # Usage: speed_ab.sh FLAG[=VALUE] BIN OUT FIRST_PAIR LAST_PAIR (on arm: FLAG=VALUE, default 1)   (env ARMS, DEPTHS, KINDS pass through to tp_depth_run.sh)
set -euo pipefail
: "${SCHED:?Run this script through nightsched}"
FLAG=${1%%=*} BIN=$2 OUT=$3 FIRST=$4 LAST=$5
VALUE=1
[[ $1 != *=* ]] || VALUE=${1#*=}
[[ $FLAG =~ ^XPU_[A-Z0-9_]+$ && $VALUE =~ ^[A-Za-z0-9_]+$ ]] || { echo "Bad flag $1" >&2; exit 2; }
[[ -x $BIN ]] || { echo "No binary $BIN" >&2; exit 3; }
stale=$(find "${BIN%%/zig-out/*}/zig" \( -name '*.zig' -o -name '*.cl' -o -name '*.h' \) -newer "$BIN" -print -quit)
[[ -z $stale ]] || { echo "Stale binary $BIN: $stale is newer" >&2; exit 3; }
RUN=~/B70-DOCS/scripts/tmp/breakthrough_20261118/tp_depth_run.sh
export TF_BIN=$BIN COOL=${COOL:-0} DEPTHS=${DEPTHS:-0,8k,32k,128k} ARMS=${ARMS:-drafted}
unset "$FLAG"
mkdir -p "$OUT"
for ((p = FIRST; p <= LAST; p++)); do
    order=(off on)
    ((p % 2)) || order=(on off)
    for arm in "${order[@]}"; do
        dir=$OUT/pair$p-$arm
        mkdir -p "$dir"
        { echo "flag $FLAG=$([[ $arm == on ]] && echo "$VALUE" || echo unset)"; echo "tp ${TP:-2} (TP=1 serves on card 1) arms $ARMS depths $DEPTHS"; sha256sum "$BIN"; date -Is; } > "$dir/ab.txt"
        if [[ $arm == on ]]; then
            env "$FLAG=$VALUE" bash "$RUN" "$dir" "${TP:-2}" 139264
        else
            bash "$RUN" "$dir" "${TP:-2}" 139264
        fi
    done
done
exit 0
}
