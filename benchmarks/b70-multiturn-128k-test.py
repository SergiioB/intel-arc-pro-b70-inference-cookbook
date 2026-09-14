#!/usr/bin/env python3
"""Multi-turn conversation test at 128K context.

Proves the cached-context claim: turn 1 loads the full 128K context (slow,
one-time), then each follow-up turn only prefills the NEW tokens and continues
decode on the warm KV cache. Measures per-turn TTFT + decode rate.

Usage: python3 b70-multiturn-128k-test.py <base-url> <model> <out.json>
"""
import json, sys, time, urllib.request

BASE = sys.argv[1]
MODEL = sys.argv[2]
OUT = sys.argv[3]

FILLER = ("The evolution of AI hardware accelerators spans GPUs, TPUs, and specialized "
          "inference chips, each optimizing different tradeoffs between throughput, "
          "latency, memory bandwidth, and power efficiency. ")

# The "document" = the huge initial context (~120K tokens)
DOC_REPS = 3500


def call(messages, max_tokens):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    usage = None
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
    total = time.time() - t0
    comp = usage.get("completion_tokens", 0) if usage else 0
    pn = usage.get("prompt_tokens", 0) if usage else 0
    return {
        "turn_prompt_tokens": pn, "total_context_tokens": None,
        "ttft_s": round(ttft or 0, 3), "total_s": round(total, 2),
        "decode_tps": round(comp / max(total - (ttft or 0), 0.001), 1),
        "completion_tokens": comp,
    }


def main():
    # Turn 1: load the "document" (~120K tokens) + first question
    doc = FILLER * DOC_REPS
    messages = [
        {"role": "system", "content": "You are analyzing this document."},
        {"role": "user", "content": doc + "\n\nFirst question: What is the main topic?"},
    ]
    print("=== Turn 1: loading full ~120K context ===", flush=True)
    t1 = call(messages, 128)
    print(f"  TTFT={t1['ttft_s']}s total={t1['total_s']}s decode={t1['decode_tps']} t/s "
          f"(prompt tokens={t1['turn_prompt_tokens']})", flush=True)

    # Append the assistant reply to the conversation
    messages.append({"role": "assistant", "content": "The document discusses AI hardware accelerators."})

    # Turns 2-5: small follow-ups — only NEW tokens prefill, KV cache is warm
    turns = [
        "What tradeoffs does it mention between GPUs and TPUs?",
        "What role does memory bandwidth play in the analysis?",
        "Summarize the power efficiency discussion.",
        "What is the final conclusion of the document?",
    ]
    results = {"turn1": t1, "followups": []}
    print("=== Turns 2-5: follow-ups on warm 128K KV cache ===", flush=True)
    for i, q in enumerate(turns, start=2):
        messages.append({"role": "user", "content": q})
        r = call(messages, 64)
        # context tokens = sum of all message tokens; we report turn_prompt as the new tokens
        r["total_context_tokens"] = r["turn_prompt_tokens"]  # usage.prompt_tokens = full context each turn
        results["followups"].append({"turn": i, **r})
        print(f"  Turn {i}: TTFT={r['ttft_s']}s total={r['total_s']}s decode={r['decode_tps']} t/s "
              f"(full ctx={r['turn_prompt_tokens']} tokens)", flush=True)
        messages.append({"role": "assistant", "content": "Understood, continuing analysis."})

    json.dump(results, open(OUT, "w"), indent=1)
    print(f"\nSaved to {OUT}", flush=True)
    print("\n=== SUMMARY ===", flush=True)
    print(f"Turn 1 (cold, 120K load): TTFT={t1['ttft_s']}s", flush=True)
    avg_ttft = sum(f["ttft_s"] for f in results["followups"]) / len(results["followups"])
    avg_dec = sum(f["decode_tps"] for f in results["followups"]) / len(results["followups"])
    print(f"Turns 2-5 (warm 128K KV): avg TTFT={avg_ttft:.2f}s, avg decode={avg_dec:.1f} t/s", flush=True)
    print(f"=> warm follow-up TTFT is {t1['ttft_s']/avg_ttft:.0f}x faster than the cold load", flush=True)


if __name__ == "__main__":
    main()
