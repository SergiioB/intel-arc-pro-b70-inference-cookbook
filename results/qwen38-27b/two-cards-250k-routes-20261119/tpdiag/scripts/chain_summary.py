#!/usr/bin/env python3
"""Chain test summary: per arm, list mode, bytes and direct-submission mode, min/median per-step cost (rep 0 dropped).
Columns from peer.cpp chain rows: append_us = host append per step (both ranks; replay: build per step),
execute_us = replay submit time, event_wait_us = host wait after the last append/submit, wall_us = first append
to both ranks done, gpu0_us = wall/128 = per-step cost, gpu1_us = host reset of the arm's 256 ready events."""
import csv, statistics, sys
from collections import defaultdict
from pathlib import Path

raw = Path(sys.argv[1])
cells = defaultdict(list)
for path in sorted(raw.glob("chain-*.csv")) + sorted(raw.glob("smoke-*.csv")):
    mode = path.stem.split("-", 1)[1]
    src = path.stem.split("-", 1)[0]
    for row in csv.DictReader(path.open()):
        if row["test"] != "chain" or int(row["rep"]) == 0:
            continue
        cells[(src, mode, row["arm"], int(row["bytes"]))].append(row)
print("source\tds_mode\tarm\tbytes\tn\terrors\tstep_us_min\tstep_us_med\tappend_us_per_step_med\treset_us_med\tdelta_vs_local_med_us")
local = {}
for key, rows in sorted(cells.items()):
    src, mode, arm, size = key
    med = statistics.median(float(r["gpu0_us"]) for r in rows)
    if arm.startswith("local_"):
        local[(src, mode, arm.split("_", 1)[1], size)] = med
for key, rows in sorted(cells.items()):
    src, mode, arm, size = key
    step = [float(r["gpu0_us"]) for r in rows]
    med = statistics.median(step)
    kind = arm.rsplit("_", 1)[1]
    base = local.get((src, mode, kind, size))
    print(f"{src}\t{mode}\t{arm}\t{size}\t{len(rows)}\t{sum(int(r['errors']) for r in rows)}\t{min(step):.2f}\t{med:.2f}\t"
          f"{statistics.median(float(r['append_us']) for r in rows):.2f}\t"
          f"{statistics.median(float(r['gpu1_us']) for r in rows):.1f}\t"
          f"{'' if base is None else f'{med - base:.2f}'}")
