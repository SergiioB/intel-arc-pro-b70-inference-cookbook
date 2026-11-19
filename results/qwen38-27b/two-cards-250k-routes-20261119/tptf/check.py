#!/usr/bin/env python3
"""Judge a retained cache replay or concurrent workload from the client and server records."""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DONE = re.compile(r"done chatcmpl-\S+ prompt=(\d+) cached=(\d+).*? tokens=(\d+) sha=(\w+) finish=(\w+)")


def rows(path):
    return list(map(json.loads, path.read_text().splitlines()))


def done(path):
    return [
        (int(prompt), int(cached), int(tokens), sha, finish)
        for prompt, cached, tokens, sha, finish in DONE.findall(path.read_text())
        if int(prompt) != 15  # the gate client's fixed health warmup
    ]


def resume(out):
    fresh = rows(out / "fresh/session.jsonl")
    path = out / "cached/replay.jsonl"
    cached = rows(path) if path.exists() else []
    cached += [{}] * (len(fresh) - len(cached))  # a turn the replay never finished fails below
    reference = {p: (n, sha, finish) for p, hit, n, sha, finish in done(out / "fresh/server.log") if not hit}
    resumed = {p: (hit, n, sha, finish) for p, hit, n, sha, finish in done(out / "cached/server.log")}
    ok = True
    for a, b in zip(fresh, cached, strict=True):
        prompt = a["prompt_tokens"]
        result = resumed.get(prompt)
        exact = bool(result) and reference.get(prompt) == result[1:]
        matching = (
            a["finish"] in ("stop", "length")
            and b.get("finish") == a["finish"]
            and b.get("completion_tokens") == a["completion_tokens"]
            and b.get("prompt_tokens") == prompt
            and b.get("equal") is True
            and exact
            and (result[0] > 0) == (a["turn"] > 0)
        )
        report = {
            "turn": a["turn"], "prompt_tokens": prompt, "cached_tokens": b.get("cached_tokens"),
            "completion_tokens": [a["completion_tokens"], b.get("completion_tokens")],
            "fresh_text_sha256": a["sha256"], "resumed_text_sha256": b.get("sha256"),
            "server_token_hashes": [reference.get(prompt), result],
            "pass": matching,
        }
        print(json.dumps(report))
        ok &= matching
    return ok


def concurrent(out):
    data = rows(out / "conc.jsonl")
    solo = {}
    by_prompt = defaultdict(set)
    lengths = set()
    errors = []
    for row in data:
        key = row["key"]
        if row["phase"] == "alone":
            solo.setdefault(key, row["sha256"])
        if (
            row["finish"] != "length" or row["completion_tokens"] != 512
            or row["cached_tokens"] not in (None, 0) or row["sha256"] != solo[key]
        ):
            errors.append({k: row.get(k) for k in ("key", "phase", "tag", "rep", "sha256", "finish")})
        lengths.add(row["prompt_tokens"])
    for prompt, cache, tokens, sha, finish in done(out / "server.log"):
        if not cache and tokens == 512 and finish == "length":
            by_prompt[prompt].add(sha)
    hashes_match = set(by_prompt) == lengths and all(len(shas) == 1 for shas in by_prompt.values())
    print(json.dumps({
        "requests": len(data), "client_mismatches": errors,
        "server_token_hashes": {p: sorted(shas) for p, shas in by_prompt.items()},
        "pass": not errors and hashes_match,
    }))
    return not errors and hashes_match


if __name__ == "__main__":
    mode, output = sys.argv[1:]
    sys.exit(not {"resume": resume, "conc": concurrent}[mode](Path(output)))
