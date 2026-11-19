#!/usr/bin/env python3
"""Record paired cold-prefix C1 measurements and token identity for TP2."""
import importlib.util
import json
import re
import statistics
import sys
import time
from pathlib import Path

GATES = Path("~/B70-DOCS/scripts/tmp/breakthrough_20261118/gates/gates.py")
PROMPTS = Path("~/B70-DOCS/results/2026/11/b70-tonight-20261108/prompts-qwen38.json")
INPUTS = {"code-0": 68, "code-8k": 7976, "code-32k": 32552, "code-128k": 130856}
DONE = re.compile(r"done chatcmpl-\S+ prompt=(\d+) cached=(\d+).*? tokens=(\d+) sha=(\w+) finish=(\w+).*? tok/s=([\d.]+)")


def temperatures():
    return [int(Path(gates.hwmon(i), "temp2_input").read_text()) / 1000 for i in (0, 1)]


def record(base, index, keys, out):
    prompts = json.loads(PROMPTS.read_text())
    for key in keys:
        messages = [{"role": "user", "content": prompts[key]["content"]}]
        kinds = ["warmup", "drafted"]
        if index == 1:
            kinds.append("serial")
        for kind in kinds:
            seen = len(DONE.findall((out / "server.log").read_text()))
            before = temperatures()
            row, text, reasoning = gates.chat(base, messages, 512, {"draft": False} if kind == "serial" else None)
            after = temperatures()
            for _ in range(100):  # the server logs its "done" line after the stream closes
                records = DONE.findall((out / "server.log").read_text())
                if len(records) > seen:
                    break
                time.sleep(0.1)
            else:
                raise ValueError(f"Missing completed request in {out / 'server.log'}")
            p, c, n, sha, finish, rate = records[-1]
            row.update(
                key=key, kind=kind, temp_start_c=before, temp_end_c=after, text=text, reasoning=reasoning,
                server_token_hash=sha, server_post_first_tok_s=float(rate),
            )
            gates.append(out / "bench.jsonl", row)
            if not (
                row["prompt_tokens"] == INPUTS[key] == int(p)
                and row["completion_tokens"] == int(n) == 512
                and row["finish"] == finish == "length"
                and int(c) == 0
                and row["cached_tokens"] in (None, 0)
            ):
                raise ValueError(f"Incomplete or mismatched {kind} request at {key}: {p=} {n=} {c=} {finish=}")
            print(key, kind, sha, row["decode_tok_s"], row["ttft_s"], row["wall_s"], flush=True)


def compare(out, pairs, keys):
    all_rows = {
        (pair, arm): [json.loads(ln) for ln in (out / f"pair{pair}-{arm}/bench.jsonl").read_text().splitlines()]
        for pair in range(1, pairs + 1)
        for arm in ("control", "candidate")
    }
    result = {}
    for key in keys:
        per_dir = {d: [r for r in rows if r["key"] == key] for d, rows in all_rows.items()}
        hashes = {
            (r["server_token_hash"], r["sha256"])
            for rows in per_dir.values()
            for r in rows
        }
        if len(hashes) != 1:
            raise ValueError(f"Token/text mismatch at {key}. Raw rows under {out}")
        summaries = {}
        for arm in ("control", "candidate"):
            rows = [next(r for r in per_dir[p, arm] if r["kind"] == "drafted") for p in range(1, pairs + 1)]
            measures = ("decode_tok_s", "server_post_first_tok_s", "ttft_s", "wall_s")
            summaries[arm] = {
                metric: {"median": statistics.median(r[metric] for r in rows), "raw": [r[metric] for r in rows]}
                for metric in measures
            }
        ctl, new = (summaries[arm]["decode_tok_s"]["median"] for arm in ("control", "candidate"))
        result[key] = {**summaries, "decode_ratio": new / ctl, "exact_requests": sum(map(len, per_dir.values()))}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    if mode == "record":
        spec = importlib.util.spec_from_file_location("gates", GATES)
        gates = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gates)
        base, pair, keys, out = args
        record(base, int(pair), keys.split(","), Path(out))
    else:
        out, pairs, keys = args
        compare(Path(out), int(pairs), keys.split(","))
