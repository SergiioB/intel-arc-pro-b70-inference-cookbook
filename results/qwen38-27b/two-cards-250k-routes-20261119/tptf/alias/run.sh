#!/usr/bin/env bash
{ # Queue entry, both cards: build the Level Zero page-alias check, then run it on card 0 and card 1.
set -euo pipefail
: "${SCHED:?Run this script through nightsched}"
D=$(cd "$(dirname "$0")" && pwd)
OUT=$D/out-$(date +%H%M%S)
mkdir -p "$OUT"
gcc -O2 -o "$OUT/page_alias" "$D/page_alias.c" -lze_loader
sha256sum "$D/page_alias.c" "$OUT/page_alias" > "$OUT/env.txt"
dpkg -l libze-intel-gpu1 libze1 2>/dev/null | awk '/^ii/ {print $2, $3}' >> "$OUT/env.txt"
uname -r >> "$OUT/env.txt"
status=0
for card in 0 1; do
    echo "== card $card" | tee -a "$OUT/result.txt"
    ZE_AFFINITY_MASK=$card ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE timeout 300 "$OUT/page_alias" 2300 2700 4040 4200 2>&1 | tee -a "$OUT/result.txt" || status=1
done
echo "exit $status" | tee -a "$OUT/result.txt"
exit 0
}
