#!/usr/bin/env python3
"""Focused prefill sweep: measure cold prefill across prompt lengths.

For a running vLLM server, measures prompt_tokens / ttft (cold prefill,
unique prompt per rep to defeat prefix caching) at several prompt lengths.
Reports steady-state prefill t/s. Decode is a fixed 8-token generation.

Usage:
  python3 b70-mtp-prefill-sweep.py <base-url> <model> <out.json> <n_spec>
Prompt lengths: 500, 1000, 2000, 4000, 8000. 3 reps each, unique suffix.
"""
import json, sys, time, urllib.request

BASE = sys.argv[1]
MODEL = sys.argv[2]
OUT = sys.argv[3]
NSPEC = sys.argv[4] if len(sys.argv) > 4 else "?"

FILLER = ("The evolution of AI hardware accelerators spans GPUs, TPUs, and specialized "
          "inference chips, each optimizing different tradeoffs between throughput, "
          "latency, memory bandwidth, and power efficiency. ")
PROMPT_CELLS = [("p500", 20), ("p1k", 40), ("p2k", 80), ("p4k", 160), ("p8k", 320)]
GEN = 8
REPS = 3


def stream_call(prompt, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        first = True
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                d = json.loads(p)
            except json.JSONDecodeError:
                continue
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and first and ch[0].get("delta", {}).get("content"):
                ttft = time.time() - t0
                first = False
    return ttft, time.time() - t0, usage


def main():
    print(f"MTP{NSPEC} prefill sweep", flush=True)
    stream_call("hi", 4)
    out = {"model": MODEL, "n_spec": NSPEC, "cells": []}
    for pname, reps in PROMPT_CELLS:
        runs = []
        for r in range(REPS):
            prompt = (FILLER * reps + f" Summarize; theme {r} of {pname}.")
            ttft, total, usage = stream_call(prompt, GEN)
            pn = usage.get("prompt_tokens", 0) if usage else 0
            runs.append({"rep": r + 1, "prompt_n": pn, "ttft_s": round(ttft, 4),
                         "prefill_tps": round(pn / ttft, 1) if ttft else None})
        steady = runs[1:]
        avg = round(sum(x["prefill_tps"] for x in steady) / len(steady), 1)
        out["cells"].append({"prompt": pname, "prompt_n": runs[0]["prompt_n"],
                             "runs": runs, "prefill_tps": avg})
        print(f"  {pname:5s} p={runs[0]['prompt_n']:>5}  prefill={avg:>7.1f} t/s (reps: "
              + ", ".join(str(x["prefill_tps"]) for x in runs) + ")", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()