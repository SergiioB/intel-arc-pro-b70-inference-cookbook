#!/usr/bin/env python3
"""MoE prefill × gen-length sweep for llama.cpp (/completion endpoint).

Uses llama-server's timings.predicted_per_second (engine decode rate) and
timings.prompt_n / timings.prompt_ms (prefill) — the AGENTS.md §9.4 correct
method for llama.cpp. Same grid as b70-moe-sweep.py for apples-to-apples.

Usage: python3 b70-moe-sweep-llamacpp.py <endpoint> <out.json> <label> [reps]
"""
import json, sys, time, urllib.request, urllib.error

BASE = sys.argv[1]  # http://host:8765/completion
OUT = sys.argv[2]
LABEL = sys.argv[3]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 2

FILLER = ("The evolution of AI hardware accelerators spans GPUs, TPUs, and specialized "
          "inference chips, each optimizing different tradeoffs between throughput, "
          "latency, memory bandwidth, and power efficiency. ")


def completion(prompt, n_predict):
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
            "stream": False, "cache_prompt": False}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read().decode())
    wall = time.time() - t0
    ti = d.get("timings", {}) or d.get("timings", {})
    return {
        "wall_s": round(wall, 3),
        "prompt_n": ti.get("prompt_n", 0),
        "prompt_ms": ti.get("prompt_ms", 0),
        "predicted_n": ti.get("predicted_n", 0),
        "predicted_ms": ti.get("predicted_ms", 0),
        "predicted_per_second": ti.get("predicted_per_second", 0),
        "prompt_per_second": ti.get("prompt_per_second", 0),
    }


def cell(reps_filler, gen):
    results = []
    for i in range(REPS):
        p = (FILLER * reps_filler) + " Summarize the key tradeoffs mentioned."
        try:
            r = completion(p, gen)
        except urllib.error.HTTPError as e:
            print(f"  [skip] reps_f={reps_filler} g={gen}: HTTP {e.code}", flush=True)
            return None
        results.append(r)
    avg_decode = round(sum(r["predicted_per_second"] for r in results) / len(results), 1)
    avg_prefill = round(sum(r["prompt_per_second"] for r in results) / len(results), 1)
    best_decode = round(max(r["predicted_per_second"] for r in results), 1)
    return {"reps": results, "avg_prefill_tps": avg_prefill,
            "avg_decode_tps": avg_decode, "best_decode_tps": best_decode,
            "prompt_n": results[-1]["prompt_n"]}


def main():
    completion("hi", 8)  # warmup
    grid = [(1, "short"), (14, "p512"), (28, "p1k"), (55, "p2k"), (110, "p4k"), (215, "p8k")]
    gens = [32, 128, 256, 512]
    out = {"label": LABEL, "engine": "llama.cpp SYCL b10255+ (0804)", "reps_per_cell": REPS, "grid": {}}
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
                "prompt_n": c["prompt_n"],
            }
            print(f"[{LABEL}] {tag} g{g}: prompt_n={c['prompt_n']} "
                  f"prefill={c['avg_prefill_tps']} decode(avg)={c['avg_decode_tps']} "
                  f"decode(best)={c['best_decode_tps']}", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("\n=== SUMMARY (avg decode t/s, engine rate) ===", flush=True)
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
