#!/usr/bin/env python3
"""Deterministic quality suite v2 for OpenAI-compatible B70 servers (stdlib only).

40 items, one sample each, temperature 0, checks are mechanical: exact answers,
regex constraints, JSON validation, executed code against tests, needle recall,
and tool-call parsing. Nothing is graded by a model.

The suite is a new item set: it is NOT the private 94-task v1-compare set and its
totals must never be mixed with that table.

Usage:
  python3 benchmarks/quality-suite.py --url http://127.0.0.1:8084 --model nanbeige43-33b \
      --out results/nanbeige43-33b/quality-suite-v2.json --label "Nanbeige4.3-33B Q4_K_M llama.cpp"

Areas: math 8, coding 8, structured 6, instruction 6, retrieval 6, tool 4, multilingual 2.
"""
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request

FILLER = (
    "The deployment log records a routine checkpoint restore, a cache warm-up and a "
    "health probe across two accelerator cards. "
)

RAW_ITEMS = [
    # --- math (8) ---
    ("math-01", "math", "What is 48 * 27? Reply with the number only.", "exact", "1296"),
    ("math-02", "math", "What does this Python print? Reply with the number only.\nprint(sum(i*i for i in range(4)))", "exact", "14"),
    ("math-03", "math", "A train travels 120 km at 60 km/h and returns at 40 km/h. Reply with the average speed in km/h as a number only.", "exact", "48"),
    ("math-04", "math", "What is 1001 divided by 7? Reply with the number only.", "exact", "143"),
    ("math-05", "math", "Solve for x: 3x + 7 = 31. Reply with the number only.", "exact", "8"),
    ("math-06", "math", "What is 15% of 240? Reply with the number only.", "exact", "36"),
    ("math-07", "math", "Is 91 prime? Reply yes or no.", "exact", "no"),
    ("math-08", "math", "What is the next number in 2, 4, 8, 16? Reply with the number only.", "exact", "32"),
    # --- coding (8): the block is executed against tests ---
    ("coding-01", "coding", "Write a Python function `add(a, b)` that returns the sum. Reply with one fenced code block only.", "code",
     "print(add(2, 3))", "5"),
    ("coding-02", "coding", "Write a Python function `rev(s)` that returns the string reversed. Reply with one fenced code block only.", "code",
     "print(rev('abc'))", "cba"),
    ("coding-03", "coding", "Write a Python function `evens(xs)` that returns only the even numbers, preserving order. Reply with one fenced code block only.", "code",
     "print(evens([1,2,3,4]))", "[2, 4]"),
    ("coding-04", "coding", "Write a Python function `count_words(s)` that returns a dict of word counts, splitting on whitespace. Reply with one fenced code block only.", "code",
     "print(count_words('a b a'))", "{'a': 2, 'b': 1}"),
    ("coding-05", "coding", "Write a Python function `fib(n)` that returns the nth Fibonacci number with fib(0)=0. Reply with one fenced code block only.", "code",
     "print(fib(10))", "55"),
    ("coding-06", "coding", "Write a Python function `flatten(x)` that flattens one level of nested lists. Reply with one fenced code block only.", "code",
     "print(flatten([[1,2],[3]]))", "[1, 2, 3]"),
    ("coding-07", "coding", "Write a Python function `is_pal(s)` that returns True when a string reads the same backwards. Reply with one fenced code block only.", "code",
     "print(is_pal('abba'), is_pal('ab'))", "True False"),
    ("coding-08", "coding", "Write a Python function `safe_div(a, b)` that returns a/b, or None when b is zero. Reply with one fenced code block only.", "code",
     "print(safe_div(6, 3), safe_div(1, 0))", "2.0 None"),
    # --- structured output (6) ---
    ("json-01", "structured", 'Return only this JSON object with no prose: key "answer" has integer value 42.', "json", {"answer": 42}),
    ("json-02", "structured", 'Return only a JSON array of the integers 1, 2, 3.', "json", [1, 2, 3]),
    ("json-03", "structured", 'Return only a JSON object with keys "name" (string "b70") and "cards" (integer 2).', "json", {"name": "b70", "cards": 2}),
    ("json-04", "structured", 'Return only a JSON object {"ok": true} and nothing else.', "json", {"ok": True}),
    ("json-05", "structured", 'Return only a JSON object with key "items" mapped to an array of two strings "a" and "b".', "json", {"items": ["a", "b"]}),
    ("json-06", "structured", 'Return only a JSON object with key "speed" as the number 41.0.', "json", {"speed": 41.0}),
    # --- instruction following (6) ---
    ("ifev-01", "instruction", "Reply with exactly three lines, each starting with '- ' and containing one of these words: alpha, beta, gamma.", "regex", r"^- .*alpha.*\n- .*beta.*\n- .*gamma.*$"),
    ("ifev-02", "instruction", "Reply with the single word 'ready' in lowercase and nothing else.", "regex", r"^ready[.!]?$"),
    ("ifev-03", "instruction", "Reply with the numbers 1 to 5 separated by commas and no spaces.", "regex", r"^1,2,3,4,5[.]?$"),
    ("ifev-04", "instruction", "Reply with the word 'ack' repeated exactly four times separated by single spaces.", "regex", r"^ack ack ack ack[.]?$"),
    ("ifev-05", "instruction", "Reply with a sentence of at most six words that contains the word 'cache'.", "regex",
     (r"^(?=\S*(?:\s+\S+){0,5}$)(?=.*\bcache\b).*[.!]?$", False)),
    ("ifev-06", "instruction", "Reply with UPPERCASE only: the word 'done'.", "regex", r"^DONE[.!]?$"),
    # --- retrieval at depth (6) ---
    ("ret-01", "retrieval", "Record A: key=H7, value=5821. Record B: key=J3, value=9184. Reply with the value for H7 only.", "exact", "5821"),
    ("ret-02", "retrieval", "NEEDLE_4K", "needle", "albatross-4417"),
    ("ret-03", "retrieval", "NEEDLE_8K", "needle", "pelican-8823"),
    ("ret-04", "retrieval", "NEEDLE_16K", "needle", "condor-1934"),
    ("ret-05", "retrieval", "Record C: temperature 31.5, pressure 1008. Record D: temperature 27.0, pressure 1013. Reply with the temperature in record C only.", "exact", "31.5"),
    ("ret-06", "retrieval", "NEEDLE_32K", "needle", "kestrel-6641"),
    # --- tool use (4) ---
    ("tool-01", "tool", "Call the tool get_weather for city Oslo. Emit a tool call with name get_weather and arguments {\"city\": \"Oslo\"} and nothing else.", "tool", ("get_weather", {"city": "Oslo"})),
    ("tool-02", "tool", "Call the tool read_file with path /etc/hosts. Emit a tool call with name read_file and arguments {\"path\": \"/etc/hosts\"} and nothing else.", "tool", ("read_file", {"path": "/etc/hosts"})),
    ("tool-03", "tool", "Call the tool set_power_limit with watts 180. Emit a tool call with name set_power_limit and arguments {\"watts\": 180} and nothing else.", "tool", ("set_power_limit", {"watts": 180})),
    ("tool-04", "tool", "Call the tool search_docs with query b70 kv cache. Emit a tool call with name search_docs and arguments {\"query\": \"b70 kv cache\"} and nothing else.", "tool", ("search_docs", {"query": "b70 kv cache"})),
    # --- multilingual (2) ---
    ("ml-01", "multilingual", "Translate 'the cache is full' into German. Reply with the translation only.", "keywords", ["cache", "voll"]),
    ("ml-02", "multilingual", "Translate 'the model is loaded' into Spanish. Reply with the translation only.", "keywords", ["modelo", "cargad"]),
]

# coding items carry (snippet, expected stdout) as a 6-tuple; normalise to 5
ITEMS = [
    (i[0], i[1], i[2], i[3], (i[4], i[5])) if len(i) == 6 else i
    for i in RAW_ITEMS
]

NEEDLE_SIZES = {"NEEDLE_4K": 4000, "NEEDLE_8K": 8000, "NEEDLE_16K": 16000, "NEEDLE_32K": 32000}
NEEDLES = {"NEEDLE_4K": "albatross-4417", "NEEDLE_8K": "pelican-8823", "NEEDLE_16K": "condor-1934", "NEEDLE_32K": "kestrel-6641"}


def build_prompt(item):
    """Expand synthetic retrieval items into a real prompt with a buried needle."""
    item_id, area, prompt, kind, _ = item
    if kind != "needle":
        return prompt
    target = NEEDLE_SIZES[prompt]
    needle = NEEDLES[prompt]
    body = FILLER * (target // 12)
    marker = f"The audit token for this shard is {needle}.\n"
    cut = len(body) // 2
    text = body[:cut] + marker + body[cut:]
    return (
        f"Read the log below and answer with the audit token only.\n\n{text}\n\n"
        "Which audit token appears in the log? Reply with the token only."
    )


def post(url, model, prompt, max_tokens=1536):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def visible(msg):
    """Answer text with reasoning stripped; thinking models put it elsewhere."""
    return (msg.get("content") or "").strip()


def norm(text):
    t = text.strip().strip("`'\" \n")
    t = re.sub(r"\s+", " ", t)
    return t.lower().rstrip(".!").strip()


def fenced_code(text):
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    if m:
        return m.group(1)
    return text


def check(item, answer):
    _, _, _, kind, expected = item
    a = visible({"content": answer})
    if kind == "exact":
        return norm(a).split(" ")[0] == norm(expected).split(" ")[0], a
    if kind == "regex":
        pattern, sensitive = expected if isinstance(expected, tuple) else (expected, True)
        flags = re.M if sensitive else re.M | re.I
        return bool(re.match(pattern, a.strip(), flags)), a
    if kind == "json":
        try:
            got = json.loads(a)
        except json.JSONDecodeError:
            return False, a
        return got == expected, a
    if kind == "code":
        snippet, want = expected
        program = fenced_code(a) + "\n"
        try:
            out = subprocess.run(
                [sys.executable, "-c", program + "\n" + snippet],
                capture_output=True, text=True, timeout=20,
            )
        except subprocess.TimeoutExpired:
            return False, a[:400]
        return out.returncode == 0 and out.stdout.strip() == want, (out.stdout.strip() or out.stderr.strip())[:200]
    if kind == "needle":
        return norm(expected) in norm(a), a
    if kind == "tool":
        name, args = expected
        if name not in a:
            return False, a
        try:
            obj = json.loads(a[a.index("{"): a.rindex("}") + 1])
        except (json.JSONDecodeError, ValueError):
            obj = None
        if isinstance(obj, dict):
            candidate = obj.get("arguments", obj.get("args", obj))
            if isinstance(candidate, str):
                try:
                    candidate = json.loads(candidate)
                except json.JSONDecodeError:
                    candidate = None
            if candidate == args:
                return True, a
        flat = json.dumps(a)
        return all(str(v) in flat for v in args.values()), a
    if kind == "keywords":
        low = a.lower()
        return all(k in low for k in expected), a
    return False, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8084")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--max-tokens", type=int, default=1536)
    args = ap.parse_args()

    results = []
    t0 = time.time()
    for item in ITEMS:
        item_id, area, prompt, kind, expected = item
        text = build_prompt(item)
        start = time.monotonic()
        try:
            resp = post(args.url, args.model, text, args.max_tokens)
            msg = resp["choices"][0]["message"]
            answer = visible(msg)
            ok, detail = check(item, answer)
            usage = resp.get("usage", {})
        except Exception as exc:  # noqa: BLE001 - recorded per item, never fatal
            ok, detail, usage, answer = False, f"error: {exc}", {}, ""
        elapsed = time.monotonic() - start
        results.append({
            "id": item_id, "area": area, "kind": kind, "pass": bool(ok),
            "detail": detail if isinstance(detail, str) else str(detail),
            "answer_head": answer[:160],
            "seconds": round(elapsed, 2),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        })
        print(f"{item_id:10s} {'PASS' if ok else 'FAIL'} {elapsed:6.1f}s", flush=True)

    areas = sorted({r["area"] for r in results})
    totals = {a: [sum(1 for r in results if r["area"] == a), sum(1 for r in results if r["area"] == a and r["pass"])] for a in areas}
    summary = {
        "schema": "quality-suite-v2",
        "label": args.label,
        "url": args.url,
        "model": args.model,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sampling": {"temperature": 0, "top_p": 1.0, "max_tokens": args.max_tokens, "samples_per_item": 1},
        "grading": "mechanical only: exact answers, regex, JSON equality, executed code, needle recall, tool-call args, keywords",
        "wall_seconds": round(time.time() - t0, 1),
        "total": [sum(v[0] for v in totals.values()), sum(v[1] for v in totals.values())],
        "areas": {a: {"items": v[0], "pass": v[1]} for a, v in totals.items()},
        "items": results,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({"label": args.label, "total": summary["total"], "areas": summary["areas"]}, indent=1))


if __name__ == "__main__":
    main()
