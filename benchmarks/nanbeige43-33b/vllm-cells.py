#!/usr/bin/env python3
"""C1 cell client for a vLLM OpenAI-compatible server: exact prompt-token counts,
streamed TTFT/decode split, and card energy from the energy1_input counter.

vLLM returns no timings block, so decode rate is measured from the stream:
ttft = first token, decode = (completion-1)/(last-first). Wall rate is reported
separately; never mix the two in one column.

Usage:
  python3 vllm-cells.py --port 8110 --out cells.json --n 5 --prompt-tokens 512
  python3 vllm-cells.py --port 8110 --out deep.json --prompt-tokens 24000 --gen 128
"""
import argparse
import glob
import json
import os
import secrets
import time
import urllib.request


def post(port, path, payload, timeout=7200):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def stream(port, payload, timeout=7200):
    """Yield (arrival_time, chunk_dict) for an SSE chat completion."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({**payload, "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                return
            yield time.monotonic(), json.loads(body)


def find_energy(pci=None):
    """Return the hwmon dirs whose PCI address matches pci (comma-separated list)."""
    cards = []
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            if open(f"{h}/name").read().strip() != "xe":
                continue
            if not glob.glob(f"{h}/energy1_input"):
                continue
        except OSError:
            continue
        cards.append((h, os.path.realpath(f"{h}/device")))
    if pci:
        want = [x.strip() for x in pci.split(",") if x.strip()]
        found = [h for w in want for h, dev in cards if w in dev]
        if len(found) != len(want):
            raise SystemExit(f"no xe hwmon for every {want}; found {cards}")
        return found
    if len(cards) == 1:
        return [cards[0][0]]
    raise SystemExit("several xe cards; pass --pci: " + ", ".join(h for h, _ in cards))


def prompt_of(tokens):
    """~4 chars per token is enough; the server reports the exact count."""
    head = f"[run {secrets.token_hex(8)}]\n"
    filler = "Note the deployment risk of holding a model resident while a second job starts. "
    body = (filler * (tokens // len(filler.split()) + 2))
    return head + body[: max(0, tokens * 4 - len(head))]


def one(port, model, hwmon, prompt_tokens, gen, temp):
    text = prompt_of(prompt_tokens)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": gen,
        "temperature": temp,
        "top_p": 1.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    e0 = sum(int(open(f"{h}/energy1_input").read()) for h in hwmon) if hwmon else None
    t0 = time.monotonic()
    first = last = None
    ntok = 0
    usage = {}
    for ts, chunk in stream(port, payload):
        if chunk.get("usage"):
            usage = chunk["usage"]
        for ch in chunk.get("choices", []):
            piece = ch.get("delta", {}).get("content")
            if piece:
                if first is None:
                    first = ts
                last = ts
                ntok += 1
    wall = time.monotonic() - t0
    e1 = sum(int(open(f"{h}/energy1_input").read()) for h in hwmon) if hwmon else None
    prompt_tokens_actual = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens", ntok)
    ttft = (first - t0) if first else None
    decode = (completion - 1) / (last - first) if first and last and last > first else None
    return {
        "prompt_tokens": prompt_tokens_actual,
        "completion_tokens": completion,
        "ttft_s": round(ttft, 3) if ttft else None,
        "wall_s": round(wall, 3),
        "decode_tps": round(decode, 2) if decode else None,
        "wall_tps": round(completion / wall, 2),
        "energy_j": round((e1 - e0) / 1e6, 2) if e0 is not None else None,
        "avg_w": round((e1 - e0) / 1e6 / wall, 1) if e0 is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8110)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--hwmon", default=None)
    ap.add_argument("--pci", default=None, help="card DBDF(s), comma-separated to sum both")
    args = ap.parse_args()

    model = args.model
    if not model:
        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=60) as r:
            model = json.load(r)["data"][0]["id"]
    hwmon = args.hwmon or find_energy(args.pci)
    out = {
        "port": args.port,
        "model": model,
        "hwmon": hwmon,
        "hwmon_pci": [os.path.realpath(f"{h}/device") for h in hwmon] if hwmon else None,
        "prompt_tokens_target": args.prompt_tokens,
        "gen_requested": args.gen,
        "temperature": args.temp,
        "warmup": None,
        "samples": [],
    }
    out["warmup"] = one(args.port, model, hwmon, args.prompt_tokens, args.gen, args.temp)
    for _ in range(args.n):
        out["samples"].append(one(args.port, model, hwmon, args.prompt_tokens, args.gen, args.temp))

    def med(xs):
        xs = sorted(x for x in xs if x is not None)
        return xs[len(xs) // 2] if xs else None

    dec = [s["decode_tps"] for s in out["samples"]]
    out["summary"] = {
        "n": len(dec),
        "prompt_tokens": sorted({s["prompt_tokens"] for s in out["samples"]}),
        "completion_tokens": sorted(s["completion_tokens"] for s in out["samples"]),
        "decode_tps_median": med(dec),
        "decode_tps_min": min(x for x in dec if x) if dec else None,
        "decode_tps_max": max(x for x in dec if x) if dec else None,
        "ttft_s_median": med([s["ttft_s"] for s in out["samples"]]),
        "wall_tps_median": med([s["wall_tps"] for s in out["samples"]]),
        "avg_w_median": med([s["avg_w"] for s in out["samples"]]),
    }
    tpj = out["summary"]["decode_tps_median"], out["summary"]["avg_w_median"]
    out["summary"]["tokens_per_joule"] = (
        round(tpj[0] / tpj[1], 4) if tpj[0] and tpj[1] else None
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
