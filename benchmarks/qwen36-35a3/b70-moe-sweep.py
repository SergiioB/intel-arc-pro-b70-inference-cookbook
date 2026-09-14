#!/usr/bin/env python3
"""MoE prefill × gen-length sweep for apples-to-apples engine comparison.

Grid: prompt sizes (short/512/1K/2K/4K/8K) × gen lengths (32/128/256/512).
Reports per-cell: prefill t/s (from TTFT) + steady-state decode t/s (from
include_usage true completion_tokens / (total - ttft)).

Usage:
  python3 b70-moe-sweep.py <endpoint> <model> <out.json> <label> [reps]
"""
import json, sys, time, urllib.request, urllib.error

BASE = sys.argv[1]
MODEL = sys.argv[2]
OUT = sys.argv[3]
LABEL = sys.argv[4]
REPS = int(sys.argv[5]) if len(sys.argv) > 5 else 2

FILLER = ("The evolution of AI hardware accelerators spans GPUs, TPUs, and specialized "
          "inference chips, each optimizing different tradeoffs between throughput, "
          "latency, memory bandwidth, and power efficiency. ")


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
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and ttft is None and ch[0].get("delta", {}).get("content"):
                ttft = time.time() - t0
    total = time.time() - t0
    return ttft, total, usage


def cell(reps_filler, gen):
    results = []
    for i in range(REPS):
        p = (FILLER * reps_filler) + " Summarize the key tradeoffs mentioned."
        try:
            ttft, total, usage = stream_call(p, gen)
        except urllib.error.HTTPError as e:
            print(f"  [skip] reps_f={reps_filler} g={gen}: HTTP {e.code}", flush=True)
            return None
        comp = usage.get("completion_tokens", 0) if usage else 0
        pn = usage.get("prompt_tokens", 0) if usage else 0
        decode_s = max(total - ttft, 0.001)
        results.append({
            "prompt_n": pn, "comp": comp, "ttft": round(ttft, 3),
            "prefill_tps": round(pn / ttft, 1) if ttft else None,
            "decode_tps": round(comp / decode_s, 1),
        })
    avg_prefill = round(sum(r["prefill_tps"] for r in results) / len(results), 1)
    avg_decode = round(sum(r["decode_tps"] for r in results) / len(results), 1)
    # best (steady-state) decode — drop first rep if it's an outlier
    decs = sorted(r["decode_tps"] for r in results)
    best_decode = decs[-1]
    return {"reps": results, "avg_prefill_tps": avg_prefill,
            "avg_decode_tps": avg_decode, "best_decode_tps": best_decode}


def main():
    # warmup
    stream_call("hi", 16)
    # reps_filler approx token counts: 1->~55, 14->~512, 28->~1K, 55->~2K, 110->~4K, 215->~8K (fits 8192)
    grid = [(1, "short"), (14, "p512"), (28, "p1k"), (55, "p2k"), (110, "p4k"), (215, "p8k")]
    gens = [32, 128, 256, 512]
    out = {"label": LABEL, "model": MODEL, "reps_per_cell": REPS, "grid": {}}
    for reps_f, tag in grid:
        out["grid"][tag] = {}
        for g in gens:
            c = cell(reps_f, g)
            if c is None:
                out["grid"][tag][f"g{g}"] = {"skipped": True}
                print(f"[{LABEL}] {tag} g{g}: SKIPPED (overflow)", flush=True)
                continue
            out["grid"][tag][f"g{g}"] = {
                "avg_prefill_tps": c["avg_prefill_tps"],
                "avg_decode_tps": c["avg_decode_tps"],
                "best_decode_tps": c["best_decode_tps"],
                "prompt_n": c["reps"][-1]["prompt_n"],
            }
            print(f"[{LABEL}] {tag} g{g}: prompt_n={c['reps'][-1]['prompt_n']} "
                  f"prefill={c['avg_prefill_tps']} decode(avg)={c['avg_decode_tps']} "
                  f"decode(best)={c['best_decode_tps']}", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("\n=== SUMMARY (avg decode t/s) ===", flush=True)
    print(f"{'prompt':>8} | " + " | ".join(f"g{g:>4}" for g in gens), flush=True)
    for tag in [t for _, t in grid]:
        row = out["grid"][tag]
        cells = []
        for g in gens:
            c = row.get(f"g{g}", {})
            cells.append(f"{c['avg_decode_tps']:>5}" if not c.get("skipped") else "  -- ")
        print(f"{tag:>8} | " + " | ".join(cells), flush=True)
    print("\n=== PREFILL t/s (avg) ===", flush=True)
    for tag in [t for _, t in grid]:
        row = out["grid"][tag]
        c32 = row.get("g32", {})
        if c32.get("skipped"):
            continue
        print(f"{tag:>8} prompt_n={c32['prompt_n']:>5} prefill={c32['avg_prefill_tps']}", flush=True)


if __name__ == "__main__":
    main()
