#!/usr/bin/env python3
"""6-case greedy correctness battery for the Bolt W4A16 vLLM-XPU server.

OpenAI chat API, temperature 0 / top_k 1 / seed 0, chat_template_kwargs
enable_thinking=false, raw `message.content` (no parser in the serve command, so a
parser mismatch cannot fake a failure). Writes <outdir>/correctness6.json and prints
one line per case.

Usage: correctness6.py <outdir> [--url http://127.0.0.1:8000/v1] [--model nemotron-bolt]
                       [--image red-circle-42.png]
--image defaults to red-circle-42.png next to this script (512x512, red circle with "42").
Exit 0 only when all 6 cases pass.
"""
import argparse
import base64
import json
import pathlib
import time
import urllib.error
import urllib.request

CASES = [
    ("arith", "What is 17 * 23? Reply with the number only.", ["391"]),
    ("capital", "What is the capital of Australia? One word.", ["canberra"]),
    ("count_r", "How many letters r are in the word strawberry? Reply with the number only.", ["3"]),
    ("code", "Write a Python function is_prime(n) and nothing else.", ["def is_prime"]),
    ("spanish", "Translate into Spanish: 'The cat sleeps'. Reply with the translation only.",
     ["gato", "duerme"]),
]
VISION = ("vision",
          "What shape and color is in this image, and what number is written? Answer briefly.",
          ["red", "circle", "42"], ["image", "video"])


def post(url, body, timeout=900):
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.load(r)
    except urllib.error.HTTPError as e:
        out = {"error": {"code": e.code, "body": e.read().decode(errors="replace")[:2000]}}
    out["_wall_s"] = round(time.monotonic() - t0, 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="nemotron-bolt")
    ap.add_argument("--image", default=str(pathlib.Path(__file__).resolve().with_name("red-circle-42.png")))
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    endpoint = args.url.rstrip("/") + "/chat/completions"
    results = []

    for name, prompt, must in CASES:
        r = post(endpoint, {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens, "temperature": 0, "top_k": 1, "seed": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        text = (r.get("choices", [{}])[0].get("message", {}).get("content") or "") if "choices" in r else ""
        ok = bool(text) and all(m in text.lower() for m in must)
        results.append({"case": name, "prompt": prompt, "must_contain": must, "pass": ok,
                        "text": text, "usage": r.get("usage"), "wall_s": r.get("_wall_s"),
                        "error": r.get("error")})
        print(f"{name:8s} pass={ok} {text[:90]!r}", flush=True)

    name, prompt, must, _kinds = VISION
    img = pathlib.Path(args.image)
    if img.is_file():
        b64 = base64.b64encode(img.read_bytes()).decode()
        r = post(endpoint, {
            "model": args.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}],
            "max_tokens": 128, "temperature": 0, "top_k": 1, "seed": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        text = (r.get("choices", [{}])[0].get("message", {}).get("content") or "") if "choices" in r else ""
        ok = bool(text) and all(m in text.lower() for m in must)
        results.append({"case": name, "image": str(img), "must_contain": must, "pass": ok,
                        "text": text, "usage": r.get("usage"), "wall_s": r.get("_wall_s"),
                        "error": r.get("error")})
        print(f"{name:8s} pass={ok} {text[:120]!r}", flush=True)
    else:
        results.append({"case": name, "image": str(img), "pass": None,
                        "error": f"image not found: {img}"})
        print(f"{name:8s} SKIPPED (image not found: {img})", flush=True)

    passed = sum(1 for r in results if r.get("pass") is True)
    summary = {"endpoint": endpoint, "model": args.model, "passed": passed,
               "total": len(results), "results": results}
    (out / "correctness6.json").write_text(json.dumps(summary, indent=1))
    print(f"CORRECTNESS: {passed}/{len(results)} passed -> {out/'correctness6.json'}", flush=True)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
