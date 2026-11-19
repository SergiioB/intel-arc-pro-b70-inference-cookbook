#!/usr/bin/env bash
{ # Queue entry, both cards: copies crossing a 4 GiB-aligned GPU address, per card, with and without the second card.
set -euo pipefail
: "${SCHED:?Run this script through nightsched}"
D=$(cd "$(dirname "$0")" && pwd)
OUT=$D/out3-$(date +%H%M%S)
mkdir -p "$OUT"
gcc -O2 -o "$OUT/copy_alias" "$D/copy_alias.c" -lze_loader
sha256sum "$D/copy_alias.c" "$OUT/copy_alias" > "$OUT/env.txt"
run() { echo "== $*" | tee -a "$OUT/result.txt"; timeout 300 env "$@" 2>&1 | tee -a "$OUT/result.txt" || true; }
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
run "$OUT/copy_alias" vacross 2700 4200 6000
for card in 0 1; do run ZE_AFFINITY_MASK=$card "$OUT/copy_alias" vacross 2700 4200 6000; done
exit 0
}
