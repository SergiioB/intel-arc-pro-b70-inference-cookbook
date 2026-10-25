#!/usr/bin/env python3
"""C1 cell client for a llama.cpp server: unique-prefix requests, endpoint-verified counts,
cache delta, and card energy from the energy1_input counter.

Usage:
  python3 client.py --port 8084 --out cells.json --n 5
  python3 client.py --port 8084 --out deep.json --prompt-chars 40000 --gen 128
"""
import argparse
import json
import secrets
import time
import urllib.request


def post(port, path, payload, timeout=3600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=60) as r:
        return json.load(r)


def find_energy(pci=None):
    """Locate the energy counter of the card in use.

    With two cards installed, a first-match search can report the wrong card's
    power, so pass --pci 0000:0f:00.0 (the card llama.cpp is pinned to) or
    --hwmon explicitly.
    """
    import glob
    import os
    cards = []
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            if open(f"{h}/name").read().strip() != "xe":
                continue
            if not glob.glob(f"{h}/energy1_input"):
                continue
        except OSError:
            continue
        dev = os.path.realpath(f"{h}/device")
        cards.append((h, dev))
    if pci:
        for h, dev in cards:
            if pci in dev:
                return h
        raise SystemExit(f"no xe hwmon under {pci}; found {cards}")
    if len(cards) == 1:
        return cards[0][0]
    raise SystemExit(
        "several xe cards found; pass --pci <DBDF> or --hwmon <path>: "
        + ", ".join(f"{h} ({dev})" for h, dev in cards)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8084)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--prompt-chars", type=int, default=2000)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--hwmon", default=None)
    ap.add_argument("--pci", default=None, help="card DBDF, e.g. 0000:0f:00.0")
    args = ap.parse_args()

    props = get(args.port, "/props")
    template = props.get("chat_template") or ""
    hwmon = args.hwmon or find_energy(args.pci)
    out = {
        "port": args.port,
        "hwmon": hwmon,
        "model_path": props.get("model_path"),
        "chat_template_chars": len(template),
        "chat_template_markers": {
            m: (m in template) for m in ("enable_thinking", "preserve_thinking", "tool_call_format")
        },
        "gen_requested": args.gen,
        "temperature": args.temp,
        "samples": [],
    }

    def one(label):
        text = f"[run {secrets.token_hex(8)}]\nSummarise the risk of running out of disk during a model load.\n"
        text += "x " * max(0, (args.prompt_chars - len(text)) // 2)
        payload = {
            "messages": [{"role": "user", "content": text}],
            "max_tokens": args.gen,
            "temperature": args.temp,
            "top_p": 1.0,
            "ignore_eos": True,
            "cache_prompt": False,
            "stream": False,
        }
        e0 = int(open(f"{hwmon}/energy1_input").read())
        t0 = time.monotonic()
        resp = post(args.port, "/v1/chat/completions", payload)
        dt = time.monotonic() - t0
        e1 = int(open(f"{hwmon}/energy1_input").read())
        usage = resp.get("usage", {})
        choice = resp.get("choices", [{}])[0]
        tm = resp.get("timings", {})
        return {
            "label": label,
            "wall_s": round(dt, 3),
            "energy_j": round((e1 - e0) / 1e6, 2),
            "avg_w": round((e1 - e0) / 1e6 / dt, 1),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": choice.get("finish_reason"),
            "cache_n": tm.get("cache_n"),
            "prompt_per_second": tm.get("prompt_per_second"),
            "predicted_per_second": tm.get("predicted_per_second"),
            "content_head": (choice.get("message", {}).get("content") or "")[:160],
        }

    out["warmup"] = one("warmup_discarded")
    for i in range(args.n):
        out["samples"].append(one(f"sample_{i + 1}"))

    tg = sorted(s["predicted_per_second"] for s in out["samples"] if s["predicted_per_second"])
    w = sorted(s["avg_w"] for s in out["samples"])
    if tg:
        med = tg[len(tg) // 2]
        out["summary"] = {
            "n": len(tg),
            "tg_median": round(med, 2),
            "tg_min": round(tg[0], 2),
            "tg_max": round(tg[-1], 2),
            "avg_w_median": w[len(w) // 2],
            "tokens_per_joule": round(med / w[len(w) // 2], 4) if w[len(w) // 2] else None,
            "all_finish_length": all(s["finish_reason"] == "length" for s in out["samples"]),
            "completion_tokens": sorted(s["completion_tokens"] for s in out["samples"]),
        }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out.get("summary", {}), indent=1))


if __name__ == "__main__":
    main()
