#!/usr/bin/env python3
"""8-case exact-answer battery against an OpenAI-compatible server (greedy, thinking off).

usage: exact_battery.py     env: URL (default http://127.0.0.1:8000), MODEL (default nemotron-bolt)

Includes the hybrid-model W4A16 defect case sum(i*i for i in range(4)) -> 14 (earlier
Qwen3.8-27B W4A16 exports answered 30). A failing case is re-asked with thinking on.
"""
import json
import os
import urllib.request

URL = os.environ.get("URL", "http://127.0.0.1:8000").rstrip("/")
MODEL = os.environ.get("MODEL", "nemotron-bolt")


def chat(q, think=False, mt=200):
    b = {"model": MODEL, "temperature": 0, "top_k": 1, "seed": 0, "max_tokens": mt,
         "chat_template_kwargs": {"enable_thinking": think}, "messages": [{"role": "user", "content": q}]}
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        URL + "/v1/chat/completions", json.dumps(b).encode(),
        {"Content-Type": "application/json"}), timeout=900))
    m = r["choices"][0]["message"]
    return (m.get("content") or "").strip(), r["usage"]["completion_tokens"]


CASES = [
    ("sumsq", "What does this Python print? Reply with the number only.\nprint(sum(i*i for i in range(4)))", "14"),
    ("sum3", "What does this print? Reply with the number only.\nx = 0\nfor i in range(1, 5):\n    x += i * 2\nprint(x)", "20"),
    ("mul", "What is 48 * 27? Reply with the number only.", "1296"),
    ("div", "What is 1001 divided by 7? Reply with the number only.", "143"),
    ("days", "How many days are in a leap year? Reply with the number only.", "366"),
    ("prime", "Is 91 a prime number? Reply yes or no.", "no"),
    ("letters", "How many letters are in the word 'banana'? Reply with the number only.", "6"),
    ("seq", "What is the next number: 2, 4, 8, 16? Reply with the number only.", "32"),
]


def main():
    ok = 0
    for name, q, want in CASES:
        a, _ = chat(q)
        p = bool(a) and a.lower().strip(" .`'\"\n").split()[0:1] == [want]
        ok += p
        print(f"{name:8s} want={want:5s} pass={p} {a[:60]!r}")
        if not p:
            a2, n2 = chat(q, True, 3000)
            print(f"   thinking on -> {a2[-80:]!r} ({n2} tok)")
    print(f"EXACT: {ok}/{len(CASES)} (thinking off)")


if __name__ == "__main__":
    main()
