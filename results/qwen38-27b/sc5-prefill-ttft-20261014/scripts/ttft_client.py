#!/usr/bin/env python3
"""Cold-input TTFT cells for the SC5 int8-vs-fp16 prefill pricing (2026-10-14). Stdlib only.

Prompt = 32 random token ids (unique entropy, fresh per request) + wikitext token ids, exact total N.
Streams /v1/completions with max_tokens=1; TTFT = monotonic time to the first chunk carrying text,
cold-input rate = usage.prompt_tokens / TTFT. Asserts usage.prompt_tokens == N.
Usage: ttft_client.py --base URL --arm A --block B --cells 8192:3,32768:3,131072:2 --out FILE.jsonl [--warmup]
"""
import argparse, json, random, time, urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--model", default="qwen3.8-27b-exl3")
ap.add_argument("--arm", required=True)
ap.add_argument("--block", required=True)
ap.add_argument("--cells", required=True)
ap.add_argument("--corpus", required=True, help="JSON list of wikitext token ids (the prompt corpus)")
ap.add_argument("--out", required=True)
ap.add_argument("--warmup", action="store_true", help="one discarded same-shape request before each cell")
a = ap.parse_args()

ids = json.loads(Path(a.corpus).read_text())
rng = random.SystemRandom()


def prompt(n):
    start = rng.randrange(0, len(ids) - n)
    return [rng.randrange(1000, 150000) for _ in range(32)] + ids[start:start + n - 32]


def one(n):
    body = {"model": a.model, "prompt": prompt(n), "max_tokens": 1, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(a.base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first, usage = None, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if first is None and ev.get("choices") and ev["choices"][0].get("text"):
                first = time.monotonic()
            if ev.get("usage"):
                usage = ev["usage"]
    t1 = time.monotonic()
    first = first or t1
    assert usage and usage["prompt_tokens"] == n, (n, usage)
    return {"ttft_s": first - t0, "wall_s": t1 - t0, "prompt_tokens": usage["prompt_tokens"],
            "cold_input_tok_s": usage["prompt_tokens"] / (first - t0)}


with open(a.out, "a") as f:
    for cell in a.cells.split(","):
        n, reps = (int(x) for x in cell.split(":"))
        if a.warmup:
            w = one(n)
            print(json.dumps({"arm": a.arm, "block": a.block, "n": n, "warmup": True, **w}), flush=True)
        for i in range(reps):
            r = {"arm": a.arm, "block": a.block, "n": n, "rep": i, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **one(n)}
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(json.dumps(r), flush=True)
