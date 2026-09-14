#!/usr/bin/env python3
"""Context-scaling sweep: how do prefill + decode behave as context fills up?

Tests progressive prompt sizes from 4K to 128K (or max context). For each:
1. Send a large prompt + small gen (measures prefill throughput)
2. Reports decode rate of the generated tokens

This reveals whether decode degrades as KV cache fills (attention becomes O(n²)
in some paths) and how prefill throughput scales at extreme context.
"""
import json, sys, time, urllib.request

BASE = sys.argv[1]
MODEL = sys.argv[2]
OUT = sys.argv[3]
LABEL = sys.argv[4]

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
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:"): continue
                payload = line[5:].strip()
                if payload == "[DONE]": break
                try: d = json.loads(payload)
                except: continue
                if d.get("usage"): usage = d["usage"]
                ch = d.get("choices") or []
                if ch and ttft is None and ch[0].get("delta", {}).get("content"):
                    ttft = time.time() - t0
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "ttft": None}
    total = time.time() - t0
    comp = usage.get("completion_tokens", 0) if usage else 0
    pn = usage.get("prompt_tokens", 0) if usage else 0
    decode_s = max(total - (ttft or 0), 0.001)
    return {
        "prompt_n": pn, "comp": comp, "ttft": round(ttft or 0, 3),
        "total": round(total, 2),
        "prefill_tps": round(pn / ttft, 1) if ttft else 0,
        "decode_tps": round(comp / decode_s, 1) if comp else 0,
        "ttft_ms": round((ttft or 0) * 1000, 0),
    }


def main():
    import urllib.error
    # warmup
    stream_call("hi", 8)
    # reps_filler approx tokens: 110->~4K, 280->~10K, 550->~20K, 1100->~40K, 1800->~65K, 3500->~128K
    grid = [(110, "p4k"), (280, "p10k"), (550, "p20k"), (1100, "p40k"),
            (1800, "p65k"), (3500, "p128k")]
    out = {"label": LABEL, "model": MODEL, "context_length_config": 131072, "results": []}
    print(f"=== {LABEL} context-scaling sweep (gen=64 each) ===")
    print(f"{'prompt':>8} | {'tokens':>7} | {'prefill t/s':>11} | {'decode t/s':>10} | {'TTFT':>8} | {'total':>6}")
    print("-" * 70)
    for reps_f, tag in grid:
        p = (FILLER * reps_f) + " Summarize the key tradeoffs in one sentence."
        r = stream_call(p, 64)
        if "error" in r:
            print(f"{tag:>8} | ERROR: {r['error']}")
            out["results"].append({"tag": tag, "reps": reps_f, **r})
            continue
        print(f"{tag:>8} | {r['prompt_n']:>7} | {r['prefill_tps']:>11.0f} | {r['decode_tps']:>10.1f} | {r['ttft_ms']:>6.0f}ms | {r['total']:>5.1f}s", flush=True)
        out["results"].append({"tag": tag, "reps": reps_f, **r})
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
