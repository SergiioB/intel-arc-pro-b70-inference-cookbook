#!/usr/bin/env python3
"""Exact-length C1 code sweep. Run only in an exclusive nightsched job."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[6]
STAMP = ROOT / "results/2026/11"
TASK = (
    "\n\nImplement a complete C++17 header for a thread-safe bounded ring buffer "
    "for KV block IDs. Include a license comment, includes, a namespace, and an "
    "implementation with push, pop, peek, blocking push and pop with timeouts, "
    "resize preserving order, closure, and a concurrency-safe statistics snapshot. "
    "Document invariants and include a full example demonstrating each operation. "
    "Write the code directly, with each operation fully implemented."
)
THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
OUTPUT = 256


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def write_row(path, row):
    with path.open("a") as f:
        print(json.dumps(row, ensure_ascii=False), file=f, flush=True)


def http_request(base, endpoint, body=None):
    return Request(
        base + endpoint,
        data=json.dumps(body, ensure_ascii=False).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )


def open_request(req, timeout):
    try:
        return urlopen(req, timeout=timeout)
    except HTTPError as exc:
        raise ValueError(f"HTTP {exc.code} {req.selector}: {exc.read(3000).decode()}") from exc


def get_json(base, endpoint, body=None):
    with open_request(http_request(base, endpoint, body), timeout=45) as response:
        return json.load(response)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def variant_prefix(length, rep):
    nonce = sha(f"intelsweep-wave3-{length}-{rep}")
    return f"Reference {nonce}\n\n"


def corpus_offsets(tokenizer):
    source = STAMP / "tf-prefix-attn-20261113/prompts-deep.json"
    corpus = json.loads(source.read_text())["code-256k"]["content"]
    older = STAMP / "b70-tonight-20261108/prompts-qwen38.json"
    appendix = json.loads(older.read_text())["code-32k"]["content"]
    corpus = f"{corpus}\n\n===== Additional engine source =====\n{appendix}"
    offsets = tokenizer.encode(corpus, add_special_tokens=False).offsets
    return corpus, offsets


def exact_prompt(base, model, corpus, offsets, length, rep, prompt_dir, source_dir):
    name = f"{length}-{rep}.txt"
    saved = prompt_dir / name
    candidates = (source_dir / name, prompt_dir.parent.parent / "prompt-candidates" / name)

    def count(content):
        fields = {
            "model": model, "messages": [{"role": "user", "content": content}],
            "add_generation_prompt": True, **THINKING,
        }
        return get_json(base, "/tokenize", fields)["count"]

    if saved.exists():
        content = saved.read_text()
        if count(content) != length:
            raise ValueError(f"retained {name} has the wrong server token count")
        return content
    for potential in candidates:
        if potential.exists():
            content = potential.read_text()
            if count(content) == length:
                saved.write_text(content)
                return content
    prefix = variant_prefix(length, rep)
    # Size on local offset estimates, then use the server's actual chat renderer.
    # A standalone word after the corpus gives fine control near the boundary.
    span = length - 150
    for _ in range(12):
        candidate = prefix + corpus[:offsets[span - 1][1]] + "\n\nEnd of reference." + TASK
        remaining = length - count(candidate)
        if remaining == 0:
            saved.write_text(candidate)
            return candidate
        if 0 < remaining <= 24:
            padded = candidate.replace(
                "\n\nEnd of reference." + TASK,
                "\n\nEnd of reference." + " note" * remaining + TASK,
            )
            if count(padded) == length:
                saved.write_text(padded)
                return padded
        span += remaining - 8
    raise ValueError(f"could not size {name} to {length} server tokens")


def gpu_snapshot(paths):
    states = []
    for i, hwmon in enumerate(paths):
        freq = Path(f"/sys/class/drm/card{i}/device/tile0/gt0/freq0/max_freq")
        states.append({
            "card": i,
            "temp_c": int((hwmon / "temp2_input").read_text()) / 1000,
            "cap_w": int((hwmon / "power1_cap").read_text()) / 1_000_000,
            "clock_cap_mhz": int(freq.read_text()),
        })
    return states


def cache_hits(base, tensorfold):
    with open_request(http_request(base, "/metrics"), timeout=20) as response:
        metrics = response.read().decode()
    engine = "tensorfold" if tensorfold else "vllm"
    matches = re.findall(
        rf"^{engine}:prefix_cache_hits(?:_total)?(?:\{{[^\n]*\}})?\s+(\S+)",
        metrics, flags=re.MULTILINE,
    )
    # TensorFold hides the counter when no prompt cache is configured.
    if not matches and not tensorfold:
        raise ValueError("vLLM prefix-cache counter missing")
    return sum(map(float, matches))


def stream(base, body, path):
    # Record before decoding JSON so parsing/file I/O does not become TTFT.
    req = http_request(base, "/v1/chat/completions", body)
    first = last = None
    parts, usage, runtime, spec = [], {}, {}, {}
    finish = response_id = None
    done = False
    with path.open("w") as raw:
        t0 = time.monotonic_ns()
        with open_request(req, timeout=1200) as response:
            for line in response:
                now = time.monotonic_ns()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip().decode()
                print(json.dumps({"ns": now - t0, "data": data}), file=raw)
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise ValueError(f"SSE server error: {event['error']}")
                response_id = event.get("id", response_id)
                usage = event.get("usage") or usage
                runtime = event.get("tensorfold") or runtime
                spec = event.get("speculative") or spec
                for choice in event.get("choices", ()):
                    delta = choice.get("delta") or {}
                    if delta.get("reasoning_content") or delta.get("reasoning"):
                        raise ValueError("reasoning content with thinking disabled")
                    piece = delta.get("content")
                    if piece:
                        first = now if first is None else first
                        last = now
                        parts.append(piece)
                    finish = choice.get("finish_reason") or finish
    elapsed = (time.monotonic_ns() - t0) / 1e9
    text = "".join(parts)
    response_count = usage.get("completion_tokens")
    ttft = (first - t0) / 1e9 if first is not None else None
    decode = (response_count - 1) / ((last - first) / 1e9) if response_count and first != last else None
    return {
        "response_id": response_id, "done": done, "finish_reason": finish,
        "wall_s": elapsed, "ttft_s": ttft, "decode_tps": decode,
        "text_sha256": sha(text), "text": text, "usage": usage,
        "tensorfold": runtime, "speculative": spec, "content_events": len(parts),
    }


def attempt(args, model, length, rep, arm, content, paths, attempt_no):
    body = {
        "model": model, "messages": [{"role": "user", "content": content}],
        "temperature": 0, "top_p": 1, "seed": 4242, "max_tokens": OUTPUT,
        "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
        **THINKING,
    }
    tensorfold = args.setup.startswith("tf-")
    if tensorfold:
        body["draft"] = arm == "drafted"
    else:
        body["min_tokens"] = OUTPUT
    record = {
        "utc": utc(), "input_target": length, "rep": rep, "arm": arm,
        "attempt": attempt_no, "prompt_sha256": sha(content), "request": {
            k: v for k, v in body.items() if k != "messages"
        },
        "gpu_before": gpu_snapshot(paths),
    }
    try:
        hits_before = cache_hits(args.base, tensorfold)
        record.update(stream(args.base, body, args.out / f"{length}-{rep}-{arm}-{attempt_no}.sse.jsonl"))
        usage = record["usage"]
        runtime = record["tensorfold"]
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        record["cache_hit_delta"] = cache_hits(args.base, tensorfold) - hits_before
        prefill = runtime.get("prefill_seconds") or record["ttft_s"]
        record["prompt_tps"] = usage.get("prompt_tokens", 0) / prefill if prefill else None
        record["prompt_timing"] = "server_prefill" if runtime.get("prefill_seconds") else "client_ttft"
        if not record["done"]:
            raise ValueError("incomplete stream")
        if usage.get("prompt_tokens") != length:
            raise ValueError(f"server counted {usage.get('prompt_tokens')} input tokens")
        if record["cache_hit_delta"] or cached not in (0, None) or (tensorfold and cached != 0):
            raise ValueError(f"prefix cache hit: {record['cache_hit_delta']=} {cached=}")
        completed = usage.get("completion_tokens")
        if type(completed) is not int or completed < 0:
            raise ValueError("missing or invalid completion usage")
        if completed < OUTPUT and record["finish_reason"] in {"stop", "length"}:
            record["status"] = "early_stop"
        elif completed != OUTPUT or record["finish_reason"] != "length":
            raise ValueError(f"expected {OUTPUT} generated tokens and finish_reason=length")
        elif not record["decode_tps"] or not record["prompt_tps"]:
            raise ValueError("untimed stream")
        else:
            record["status"] = "ok"
        if tensorfold and record["status"] == "ok":
            drafts = record["speculative"].get("drafted")
            if runtime.get("enable_thinking") is not False or runtime.get("drafts") != body["draft"]:
                raise ValueError("server did not confirm the requested thinking and drafting modes")
            if arm == "drafted" and not drafts or arm == "serial" and drafts != 0:
                raise ValueError(f"unexpected number of draft tokens: {drafts}")
            if not runtime.get("token_sha256"):
                raise ValueError("TensorFold did not return full token SHA-256")
    except (OSError, ValueError, KeyError) as exc:
        record.update(status="failed", error=str(exc))
    record["gpu_after"] = gpu_snapshot(paths)
    write_row(args.out / "samples.jsonl", record)
    print({k: record[k] for k in ("input_target", "rep", "arm", "status", "utc")}, flush=True)
    if record["status"] == "failed":
        print(record["error"], flush=True)
    return record["status"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("setup")
    parser.add_argument("base")
    parser.add_argument("out", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("lengths", nargs="+", type=int)
    args = parser.parse_args()
    model = get_json(args.base, "/v1/models")["data"][0]["id"]
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    corpus, offsets = corpus_offsets(tokenizer)
    paths = [
        Path(subprocess.check_output(["python3", str(ROOT / "tools/host_facts.py"), "hwmon", str(i)], text=True).strip())
        for i in (0, 1)
    ]
    prompt_dir = args.out / "prompts"
    prompt_dir.mkdir(exist_ok=True)
    source_dir = args.out.parent / "tf-tp2-bf16" / "prompts"
    arms = ("serial", "drafted") if args.setup.startswith("tf-") else (args.setup.removeprefix("exl3-"),)
    failed = False
    for length in args.lengths:
        if args.setup == "tf-tp1-int8":
            content = exact_prompt(args.base, model, corpus, offsets, length, 0, prompt_dir, source_dir)
            body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": OUTPUT, **THINKING}
            for draft in (False, True):
                try:
                    get_json(args.base, "/v1/chat/completions", {**body, "draft": draft})
                except ValueError as exc:
                    write_row(args.out / "refusals.jsonl", {
                        "utc": utc(), "input_target": length, "output_requested": OUTPUT,
                        "draft": draft, "error": str(exc),
                    })
                    print(str(exc), flush=True)
                    continue
                raise SystemExit("server unexpectedly accepted input plus output beyond the configured window")
            return 1
        for rep in (-1, 0, 1, 2):
            try:
                content = exact_prompt(args.base, model, corpus, offsets, length, rep, prompt_dir, source_dir)
            except (OSError, KeyError, ValueError) as exc:
                write_row(args.out / "refusals.jsonl", {"utc": utc(), "input_target": length, "rep": rep, "error": str(exc)})
                failed = True
                break
            order = arms[::-1] if rep == 1 else arms
            for arm in order:
                status = attempt(args, model, length, rep, arm, content, paths, 1)
                if status == "early_stop":
                    status = attempt(args, model, length, rep, arm, content, paths, 2)
                failed |= status != "ok"
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
