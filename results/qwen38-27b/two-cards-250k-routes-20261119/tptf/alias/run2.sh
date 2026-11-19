#!/usr/bin/env bash
{ # Queue entry, both cards: Level Zero copy checks (straddling copies, TensorFold's save layout), per card.
set -euo pipefail
: "${SCHED:?Run this script through nightsched}"
D=$(cd "$(dirname "$0")" && pwd)
OUT=$D/out2-$(date +%H%M%S)
mkdir -p "$OUT"
gcc -O2 -o "$OUT/copy_alias" "$D/copy_alias.c" -lze_loader
{ sha256sum "$D/copy_alias.c" "$OUT/copy_alias"; dpkg -l libze-intel-gpu1 libze1 2>/dev/null | awk '/^ii/ {print $2, $3}'; uname -r; } > "$OUT/env.txt"
run() { echo "== $*" | tee -a "$OUT/result.txt"; timeout 300 env "$@" 2>&1 | tee -a "$OUT/result.txt" || true; }
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
for card in 0 1; do run ZE_AFFINITY_MASK=$card "$OUT/copy_alias" straddle 2700 4200; done
run "$OUT/copy_alias" layout 165056
run ZE_AFFINITY_MASK=0 "$OUT/copy_alias" layout 165056
run ZE_AFFINITY_MASK=1 "$OUT/copy_alias" layout 165056
run "$OUT/copy_alias" layout 130816
exit 0
}
