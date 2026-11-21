#!/usr/bin/env python3
"""Compile the Inkling 2 Mini llama.cpp SYCL evidence into one summary JSON.

Usage: python3 benchmarks/inkling2-mini/summarize.py results/inkling2-mini/llamacpp-sycl-20261121

Reads only the raw rows in that folder. Every number the recipe publishes is in the output.
"""

from __future__ import annotations

import json
import pathlib
import re
import statistics
import sys
from collections import defaultdict
from typing import Any

DEPTH_RUNS = {
    "graph_off_ctl1": "ctl1-graphoff-20261121T024921Z",
    "graph_on_p0001": "cand1-graphon-20261121T032935Z",
    "graph_off_ctl2": "ctl2-graphoff-20261121T040547Z",
    "p0001_ctlA": "T1-ctlA-20261121T102722Z",
    "p0001_0006_0002on": "T1-cand-20261121T103748Z",
    "p0001_ctlB": "T1-ctlB-20261121T104708Z",
    "baseline_unpatched": "inkling-base-20261121T020357Z",
    "dspark_n3_unpatched": "inkling-dspark-20261121T020936Z",
}
GREEDY_RUNS = {
    "reference_graph_off_p0001": "ctl1-graphoff-20261121T024921Z",
    "graph_on_p0001": "cand1-graphon-20261121T032935Z",
    "exact_default_0002_masked": "T1-cand-nop2-20261121T105733Z",
    "p0001_0006_0002on": "T1-cand-20261121T103748Z",
}


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def median(values: list[float]) -> float:
    return round(statistics.median(values), 2)


def depth_cells(root: pathlib.Path, run: str) -> dict[str, Any]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    failed: list[str] = []
    for row in read_jsonl(root / run / "depth" / "cells.jsonl"):
        if row.get("prompt_tokens") is None:  # the server died before the first token
            failed.append(f"depth {row['depth']} seed {row.get('seed')}: no token, {row['wall_s']} s")
            continue
        groups[row["prompt_tokens"]].append(row)
    out: dict[str, Any] = {"failed": failed} if failed else {}
    for prompt_tokens, rows in sorted(groups.items()):
        out[str(prompt_tokens)] = {
            "n": len(rows),
            "seeds": [r.get("seed") for r in rows],
            "completion_tokens": sorted({r["completion_tokens"] for r in rows}),
            "decode_tok_s": median([r["decode_tok_s"] for r in rows]),
            "decode_tok_s_range": [min(r["decode_tok_s"] for r in rows), max(r["decode_tok_s"] for r in rows)],
            "engine_prompt_tok_s": median(
                [r["last_chunk_extra"]["timings"]["prompt_per_second"] for r in rows]
            ),
            "ttft_s": median([r["ttft_s"] for r in rows]),
            "decode_card_w": median([r["decode_avg_w"] for r in rows]),
            "decode_j_per_token": median([r["decode_j_per_tok"] for r in rows]),
            "nonempty_answers": f"{sum(1 for r in rows if r['chars'] > 0)}/{len(rows)}",
            "sampling": rows[0].get("sampling_fields"),
        }
    return out


def greedy(root: pathlib.Path, run: str) -> dict[str, Any]:
    rows = read_jsonl(root / run / "exact.jsonl")
    return {
        "sha256": [r["sha256"] for r in rows],
        "decode_tok_s": [round(r["timings"]["predicted_per_second"], 2) for r in rows],
        "completion_tokens": [r["timings"]["predicted_n"] for r in rows],
    }


def quality_file(root: pathlib.Path, run: str, name: str) -> dict[str, Any]:
    rows = read_jsonl(root / run / name)
    tokens = [r["usage"]["completion_tokens"] for r in rows]
    walls = [r["wall_s"] for r in rows]
    return {
        "pass": f"{sum(1 for r in rows if r.get('pass'))}/{len(rows)}",
        "nonempty": f"{sum(1 for r in rows if r.get('nonempty'))}/{len(rows)}",
        "completion_tokens_median": statistics.median(tokens),
        "completion_tokens_max": max(tokens),
        "wall_s_median": round(statistics.median(walls), 2),
    }


def kl(root: pathlib.Path, name: str) -> dict[str, str]:
    text = (root / "Q1-kl-20261121T120248Z" / name).read_text()
    fields = {"mean_kld": r"Mean\s+KLD:\s+(-?[\d.]+)", "max_kld": r"Maximum KLD:\s+([\d.]+)",
              "same_top1_pct": r"Same top p:\s+([\d.]+)"}
    out = {}
    for key, pattern in fields.items():
        match = re.search(pattern, text)
        out[key] = match.group(1) if match else "missing"
    return out


def ppl(root: pathlib.Path, rel: str) -> str:
    match = re.search(r"PPL = ([\d.]+ \+/- [\d.]+)", (root / rel).read_text())
    return match.group(1) if match else "missing"


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    summary: dict[str, Any] = {"depth": {}, "greedy": {}}
    for key, run in DEPTH_RUNS.items():
        summary["depth"][key] = {"run": run, "cells": depth_cells(root, run)}
    for key, run in GREEDY_RUNS.items():
        summary["greedy"][key] = {"run": run, **greedy(root, run)}
    ref = summary["greedy"]["reference_graph_off_p0001"]["sha256"]
    summary["greedy_identical_to_reference"] = {
        key: f"{sum(a == b for a, b in zip(v['sha256'], ref))}/{len(ref)}"
        for key, v in summary["greedy"].items()
    }
    qdir = "Q1-effort-canary-20261121T120543Z"
    summary["canary_9x3_effort_0.9"] = quality_file(root, qdir, "canary.jsonl")
    summary["effort_sweep_9_items"] = {
        e: quality_file(root, qdir, f"effort-{e}.jsonl") for e in ("0.1", "0.2", "0.7", "0.9")
    }
    summary["kl_vs_p0001_wikitext2_20x512"] = {
        "p0001_0006_0002on": kl(root, "kl-loc.txt"),
        "exact_default_0002_masked": kl(root, "kl-loc-nop2.txt"),
    }
    summary["ppl_wikitext2_40x512"] = {
        "p0001": ppl(root, "P1-profile/ppl-wikitext2-c512-40.txt"),
        "p0001_0006_0002on": ppl(root, "VL1-verify-local/ppl-local.txt"),
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
