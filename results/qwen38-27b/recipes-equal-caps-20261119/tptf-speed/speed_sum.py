#!/usr/bin/env python3
"""Summarise speed_ab.sh: per depth and kind, flag off against on.

  speed_sum.py OUT [GATE_PCT=3]

Reads OUT/pair*-{off,on}/cells.jsonl (client rows) and server.log (engine 'done' lines, warm-ups skipped), joined in
order with the prompt counts checked. Prints medians of client decode tok/s, server ms per round (decode / rounds),
TTFT; the engine token hash sets; drafted == serial per flag (pairs that ran the serial arm); text changes off -> on.
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DONE = re.compile(r"done chatcmpl-\S+ prompt=(\d+) cached=(\d+).*? tokens=(\d+) sha=(\w+) finish=(\w+) rounds=(\d+).*? ttft=([\d.]+) decode=([\d.]+)")


def load(d):
    cells = [json.loads(x) for x in open(d / "cells.jsonl")]
    done = [m for m in DONE.findall((d / "server.log").read_text()) if int(m[0]) > 20]  # the warm-up ("Say ok.") is 15 tokens
    if len(done) < len(cells):
        raise SystemExit(f"{d}: {len(cells)} client rows against {len(done)} engine lines")
    done = done[: len(cells)]  # the canary prompts tp_depth_run.sh sends after the arms come last
    rows = []
    for c, m in zip(cells, done):
        p, hit, n, sha, finish, rounds, ttft, dec = m
        if int(p) != c["prompt_tokens"] or int(hit) != 0:
            raise SystemExit(f"{d}: prompt {p} cached {hit} against client {c['prompt_tokens']}")
        rows.append({**c, "token_hash": sha, "rounds": int(rounds), "ms_round": 1000 * float(dec) / max(int(rounds), 1)})
    return rows


def main(out, gate=3.0):
    out = Path(out)
    by = defaultdict(lambda: defaultdict(list))  # (arm, depth, kind) -> flag -> rows
    for d in sorted(out.glob("pair*-*")):
        flag = d.name.rsplit("-", 1)[1]
        for r in load(d):
            by[(r["arm"].rstrip("2"), r["depth"], r["kind"])][flag].append(r)
    report = {"cells": [], "drafted_eq_serial": {}, "text_changes_off_to_on": 0, "cells_compared": 0}
    for (arm, depth, kind), fl in sorted(by.items()):
        if arm != "drafted":
            continue
        cell = {"depth": depth, "kind": kind}
        for flag in ("off", "on"):
            rs = fl.get(flag, [])
            if not rs:
                continue
            cell[flag] = {
                "n": len(rs),
                "client_tok_s": round(statistics.median(r["decode_tok_s"] for r in rs), 2),
                "ms_round": round(statistics.median(r["ms_round"] for r in rs), 2),
                "ttft_s": round(statistics.median(r["ttft_s"] for r in rs), 3),
                "raw_tok_s": [r["decode_tok_s"] for r in rs],
                "hashes": sorted({r["token_hash"] for r in rs}),
            }
        if "off" in cell and "on" in cell:
            cell["gain_pct"] = round(100 * (cell["on"]["client_tok_s"] / cell["off"]["client_tok_s"] - 1), 2)
            cell["ms_round_delta"] = round(cell["on"]["ms_round"] - cell["off"]["ms_round"], 2)
            cell["passes_gate"] = cell["gain_pct"] >= gate
            report["cells_compared"] += 1
            report["text_changes_off_to_on"] += cell["off"]["hashes"] != cell["on"]["hashes"]
        report["cells"].append(cell)
        for flag in ("off", "on"):
            ser = by.get(("serial", depth, kind), {}).get(flag, [])
            dr = fl.get(flag, [])
            if ser and dr:
                report["drafted_eq_serial"].setdefault(flag, []).append(
                    {"depth": depth, "kind": kind, "equal": {r["sha256"] for r in dr} == {r["sha256"] for r in ser}})
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 3.0)
