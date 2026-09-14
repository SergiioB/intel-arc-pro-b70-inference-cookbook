#!/usr/bin/env python3
"""Corrected vLLM XPU bench: capture true completion_tokens via include_usage.

The reddit-bench v1 undercounts because reasoning models emit reasoning_content
(separate from content). v2 uses stream_options.include_usage to read the real
usage.completion_tokens from the final stream chunk, then reports the true
engine decode rate = completion_tokens / (total - ttft).
"""
import json, time, urllib.request, sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001/v1/chat/completions"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen3.6-35B-A3B-GPTQ-Int4"
OUT = sys.argv[3] if len(sys.argv) > 3 else "/tmp/vllm-bench.json"
VERSION = sys.argv[4] if len(sys.argv) > 4 else "0.21.0-xpu"

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
    content_n = 0
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
            if ch:
                delta = ch[0].get("delta", {})
                c = delta.get("content")
                if c:
                    if ttft is None:
                        ttft = time.time() - t0
                    content_n += 1
    total = time.time() - t0
    return ttft, content_n, total, usage


def bench(tag, filler_reps, gen_tokens):
    stream_call("hi", 8)  # warmup
    results = []
    for i in range(3):
        p = FILLER * filler_reps + " Summarize the key tradeoffs mentioned."
        ttft, content_n, total, usage = stream_call(p, gen_tokens)
        comp = usage.get("completion_tokens", content_n) if usage else content_n
        prompt_n = usage.get("prompt_tokens", 0) if usage else 0
        decode_s = max(total - ttft, 0.001)
        r = {"rep": i + 1, "prompt_n": prompt_n, "completion_tokens": comp,
             "content_n": content_n, "ttft_s": round(ttft, 3),
             "total_s": round(total, 3),
             "prefill_tps": round(prompt_n / ttft, 1) if ttft else None,
             "decode_tps": round(comp / decode_s, 1),
             "decode_tps_content_only": round(content_n / decode_s, 1)}
        results.append(r)
        print(f"[{tag}] rep{i+1}: prompt_n={prompt_n} comp={comp} (content={content_n}) "
              f"TTFT={ttft:.3f}s prefill={r['prefill_tps']} decode={r['decode_tps']} t/s", flush=True)
    return results


def main():
    out = {"model": MODEL, "config": f"vLLM {VERSION}, single request, 230W",
           "method": "streaming include_usage, true completion_tokens decode rate"}
    out["tg32"] = bench("tg32", 1, 32)
    out["pp2048"] = bench("pp2048", 55, 32)
    out["tg128"] = bench("tg128", 1, 128)
    json.dump(out, open(OUT, "w"), indent=1)
    print("\n" + json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
