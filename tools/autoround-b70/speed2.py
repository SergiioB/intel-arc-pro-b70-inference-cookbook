#!/usr/bin/env python3
"""C1 speed on the vLLM server: exact prompt sizes via /tokenize, streamed timing.

usage: REPS=5 speed2.py <outdir> [target_tokens ...]   (default 886 8192 15000; REPS default 3)
env: URL (default http://127.0.0.1:8000), MODEL (default nemotron-bolt)
The published numbers used: REPS=5 speed2.py <outdir> 886 8192 32000 64000
Per shape: 1 discarded warmup + REPS reps, temperature 0, thinking off, 128 output tokens,
ignore_eos. Every request starts with a unique nonce so no prefix is shared (prefix cache
on or off). Reports server-counted prompt tokens, TTFT (prompt tok / TTFT = cold prefill rate
incl. client overhead) and post-first decode rate; median over the reps.
"""
import json, os, pathlib, sys, time, urllib.request, uuid

URL = os.environ.get("URL", "http://127.0.0.1:8000").rstrip("/")
MODEL = os.environ.get("MODEL", "nemotron-bolt")
GEN = 128


def post(path, body, stream=False):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=1800) if stream else json.load(urllib.request.urlopen(req, timeout=1800))


def count(prompt):
    return post("/tokenize", {"model": MODEL, "prompt": prompt})["count"]


def build(target, nonce):
    head = f"[{nonce}] Summarize the inventory below in one line.\n"
    line = lambda i: f"entry {i}: crate {i * 7 % 991} holds {i * 13 % 577} units of part {i * 31 % 4093}.\n"
    lines, lo, hi = [], 0, target
    while lo < hi:  # largest number of lines that stays <= target
        mid = (lo + hi + 1) // 2
        if count(head + "".join(line(i) for i in range(mid))) <= target:
            lo = mid
        else:
            hi = mid - 1
    text = head + "".join(line(i) for i in range(lo))
    while count(text + " x") <= target:
        text += " x"
    return text


def run(prompt):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": GEN, "temperature": 0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.monotonic(); first = None; n = 0; usage = None
    with post("/v1/completions", body, stream=True) as r:
        for raw in r:
            s = raw.decode().strip()
            if not s.startswith("data:") or s == "data: [DONE]":
                continue
            d = json.loads(s[5:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices") and d["choices"][0].get("text"):
                n += 1
                if first is None:
                    first = time.monotonic()
    t1 = time.monotonic()
    pt, ct = usage["prompt_tokens"], usage["completion_tokens"]
    return {"prompt_tokens": pt, "completion_tokens": ct, "ttft_s": round(first - t0, 4),
            "prefill_tok_s": round(pt / (first - t0), 1),
            "decode_tok_s": round((ct - 1) / (t1 - first), 2), "wall_s": round(t1 - t0, 3)}


def main():
    out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    targets = [int(x) for x in sys.argv[2:]] or [886, 8192, 15000]
    res = {}
    for tgt in targets:
        rows = []
        for rep in range(1 + int(os.environ.get("REPS", "3"))):
            r = run(build(tgt, uuid.uuid4().hex[:12])); r["rep"] = rep; r["warmup"] = rep == 0
            rows.append(r); print(tgt, json.dumps(r), flush=True)
        m = [x for x in rows if not x["warmup"]]
        med = lambda k: sorted(x[k] for x in m)[len(m) // 2]
        res[tgt] = {"samples": rows, "median_prefill_tok_s": med("prefill_tok_s"),
                    "median_decode_tok_s": med("decode_tok_s"), "median_ttft_s": med("ttft_s")}
        print(f"== {tgt}: prompt {m[0]['prompt_tokens']} tok, prefill {res[tgt]['median_prefill_tok_s']} tok/s, "
              f"TTFT {res[tgt]['median_ttft_s']} s, decode {res[tgt]['median_decode_tok_s']} tok/s", flush=True)
    (out / "speed2.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
