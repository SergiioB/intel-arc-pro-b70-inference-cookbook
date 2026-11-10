#!/usr/bin/env python3
"""Prefix-cache corruption check: 9 requests share one ~8K prefix, each asks a question
with one exact answer. Greedy, thinking off. Keeps every reply's text.
Usage: prefix_cache_check.py BASE MODEL OUT_JSON
Pass = every reply contains its expected answer and none looks degenerate."""
import json, re, sys, time, urllib.request
base, model, out = sys.argv[1:4]
P = json.load(open('~/B70-DOCS/results/2026/11/b70-tonight-20261108/prompts-qwen38.json'))
prefix = P['code-8k']['content'].rsplit('\n\n', 1)[0]  # the corpus part, without its task line
QS = [("What is 17 * 23? Reply with the number only.", "391"),
      ("What is 144 / 12? Reply with the number only.", "12"),
      ("What is sum(i*i for i in range(4)) in Python? Reply with the number only.", "14"),
      ("How many letters are in the word 'kernel'? Reply with the number only.", "6"),
      ("What is 2 ** 10? Reply with the number only.", "1024"),
      ("What is the capital of France? Reply with one word.", "Paris"),
      ("What is 99 + 1? Reply with the number only.", "100"),
      ("What is 7 * 8? Reply with the number only.", "56"),
      ("Reverse the string 'abc'. Reply with the result only.", "cba")]
def ask(q):
    body = {"model": model, "messages": [{"role": "user", "content": prefix + "\n\nQuestion about nothing above: " + q}],
            "temperature": 0, "max_tokens": 32, "seed": 4242, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time(); r = json.loads(urllib.request.urlopen(req, timeout=600).read()); dt = time.time() - t
    return r["choices"][0]["message"].get("content") or "", dt, r.get("usage", {})
def degenerate(s):
    s = s.strip()
    if not s: return True
    words = re.findall(r"\w+", s)
    return len(words) > 6 and len(set(words)) / len(words) < 0.35
rows = []
for q, want in QS:
    text, dt, usage = ask(q)
    rows.append({"q": q, "want": want, "text": text, "s": round(dt, 2), "ok": want.lower() in text.lower(), "degenerate": degenerate(text), "usage": usage})
res = {"correct": sum(r["ok"] for r in rows), "degenerate": sum(r["degenerate"] for r in rows), "n": len(rows), "rows": rows}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("correct", "degenerate", "n")}), [r["s"] for r in rows])
