#!/usr/bin/env python3
"""Run golden canary probes against an active OpenAI-compatible inference endpoint."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any

CANARIES = [
    {
        "id": "arithmetic",
        "name": "Arithmetic & Determinism",
        "prompt": "Calculate 47 + 53. Output only the final number.",
        "validator": lambda resp: "100" in resp,
    },
    {
        "id": "json",
        "name": "JSON Schema Conformance",
        "prompt": "Output a valid JSON object with key 'status' set to 'ok' and key 'device' set to 'b70'. Output only JSON.",
        "validator": lambda resp: _validate_json_canary(resp),
    },
    {
        "id": "code",
        "name": "Code Generation Syntax",
        "prompt": "Write a python function named 'add' that takes two parameters 'a' and 'b' and returns their sum. Output code only.",
        "validator": lambda resp: bool(re.search(r"def\s+add\s*\(\s*a\s*,\s*b\s*\)\s*:", resp) and "return" in resp),
    },
    {
        "id": "prose",
        "name": "Instruction Following & English Prose",
        "prompt": "Write exactly one sentence describing why high memory bandwidth matters for LLM decoding.",
        "validator": lambda resp: len(resp.split()) >= 4 and ("bandwidth" in resp.lower() or "memory" in resp.lower()),
    },
]


def _validate_json_canary(text: str) -> bool:
    cleaned = text.strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        data = json.loads(cleaned)
        return data.get("status") == "ok" and data.get("device") == "b70"
    except Exception:
        return False


def query_endpoint(
    url: str,
    model: str,
    prompt: str,
    timeout: float = 30.0,
) -> tuple[str, float, int]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 128,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start_time = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        duration = time.perf_counter() - start_time
        res_bytes = response.read()
        data = json.loads(res_bytes.decode("utf-8"))
        choices = data.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", len(content.split()))
        return content, duration, completion_tokens


def run_canaries(
    host: str,
    port: int,
    model: str,
    mock: bool = False,
) -> dict[str, Any]:
    endpoint = f"http://{host}:{port}/v1/chat/completions"
    results: list[dict[str, Any]] = []

    if mock:
        # Dry run for CI/testing without GPU
        for canary in CANARIES:
            results.append({
                "id": canary["id"],
                "name": canary["name"],
                "passed": True,
                "duration_s": 0.05,
                "tokens": 15,
                "tok_per_sec": 300.0,
                "response": "mock response 100 {'status':'ok','device':'b70'} def add(a, b): return a + b memory bandwidth",
            })
        return {
            "endpoint": endpoint,
            "model": model,
            "all_passed": True,
            "results": results,
        }

    for canary in CANARIES:
        canary_id = canary["id"]
        try:
            content, duration, tokens = query_endpoint(endpoint, model, canary["prompt"])
            passed = canary["validator"](content)
            tok_s = round(tokens / duration, 2) if duration > 0 else 0.0
            results.append({
                "id": canary_id,
                "name": canary["name"],
                "passed": passed,
                "duration_s": round(duration, 3),
                "tokens": tokens,
                "tok_per_sec": tok_s,
                "response": content.strip()[:100],
            })
        except Exception as exc:
            results.append({
                "id": canary_id,
                "name": canary["name"],
                "passed": False,
                "error": str(exc),
            })

    all_passed = all(r.get("passed", False) for r in results)
    return {
        "endpoint": endpoint,
        "model": model,
        "all_passed": all_passed,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--model", default="qwen38", help="Served model name")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode without active server")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    data = run_canaries(args.host, args.port, args.model, mock=args.mock)
    if args.json:
        print(json.dumps(data, indent=2))
        return 0 if data["all_passed"] else 1

    print(f"=== Golden Canary Verification: {data['model']} on {data['endpoint']} ===")
    for r in data["results"]:
        status = "[PASS]" if r.get("passed") else "[FAIL]"
        speed = f"({r.get('tok_per_sec', 0)} tok/s, {r.get('duration_s', 0)}s)" if "tok_per_sec" in r else ""
        print(f"{status} {r['name']} {speed}")
        if not r.get("passed"):
            if "error" in r:
                print(f"       Error: {r['error']}")
            else:
                print(f"       Failed response: {r.get('response')}")

    if data["all_passed"]:
        print("\nResult: ALL CANARIES PASSED. Model output adheres to exact correctness requirements.")
        return 0
    else:
        print("\nResult: CANARY FAILURES DETECTED. Check model configuration or MTP cache flags.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
