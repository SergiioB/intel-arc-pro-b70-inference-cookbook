#!/usr/bin/env python3
"""Summarize RESULT/ttft.jsonl: per (arm, N) median TTFT and cold-input rate over both
blocks; R(N) = median TTFT B / median TTFT A; 190K completion + min VRAM; canary.
Usage: summarize_ttft.py RESULT_DIR
"""
import json, statistics, sys
from pathlib import Path

R = Path(sys.argv[1])
rows = [json.loads(l) for l in open(R / "ttft.jsonl")]

by = {}
for r in rows:
    if r.get("warmup"):
        continue
    by.setdefault((r["arm"], r["n"]), []).append(r)

print(f"{'N':>7} {'arm':>3} {'reps':>4} {'ttft med (s)':>13} {'tok/s med':>10}   all ttft")
meds = {}
for (arm, n), rs in sorted(by.items(), key=lambda x: (x[0][1], x[0][0])):
    ttfts = [r["ttft_s"] for r in rs]
    rates = [r["cold_input_tok_s"] for r in rs]
    meds[(arm, n)] = statistics.median(ttfts)
    print(f"{n:>7} {arm:>3} {len(rs):>4} {statistics.median(ttfts):>13.3f} "
          f"{statistics.median(rates):>10.0f}   {['%.2f' % t for t in ttfts]}")

print("\nR(N) = median TTFT B / median TTFT A:")
for n in (8192, 32768, 131072):
    if (("A", n) in meds) and (("B", n) in meds):
        print(f"  R({n}) = {meds[('B', n)] / meds[('A', n)]:.4f}")

print("\n190K cells:")
for arm in ("A", "B"):
    rs = by.get((arm, 190000), [])
    done = len(rs) == 1 and rs[0]["prompt_tokens"] == 190000
    ttft = rs[0]["ttft_s"] if rs else None
    try:
        vmin = min(float(l.split("=")[1]) for l in open(R / f"vram_{arm}_b2.log"))
    except Exception:
        vmin = None
    print(f"  arm {arm}: completed={done} ttft={ttft and round(ttft,1)}s min_visible_avail={vmin}MiB")
