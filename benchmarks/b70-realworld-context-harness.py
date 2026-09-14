#!/usr/bin/env python3
"""B70 real-world vLLM harness for exact-token scheduler and context tests.

Uses prebuilt prompts whose complete rendered chat-template token counts were
calibrated offline. Every cold request starts with unique entropy and validates
endpoint prompt tokens, prefix-cache deltas, direct MTP counters, TTFT, first
visible-content latency, completion timing, and raw SSE evidence.
"""
import argparse
import hashlib
import json
import os
import statistics
import time
import urllib.request

COUNTER_CANDIDATES = {
    "prefix_cache_hits": ("vllm:prefix_cache_hits", "vllm:prefix_cache_hits_total"),
    "prefix_cache_queries": ("vllm:prefix_cache_queries", "vllm:prefix_cache_queries_total"),
    "mtp_draft_tokens": ("vllm:spec_decode_num_draft_tokens", "vllm:spec_decode_num_draft_tokens_total"),
    "mtp_accepted_tokens": ("vllm:spec_decode_num_accepted_tokens", "vllm:spec_decode_num_accepted_tokens_total"),
    "mtp_drafts": ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_drafts_total"),
}
PER_POSITION_CANDIDATES = (
    "vllm:spec_decode_num_accepted_tokens_per_pos",
    "vllm:spec_decode_num_accepted_tokens_per_pos_total",
)


def metric_series(text, candidates):
    for candidate in candidates:
        rows = {}
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            sample = fields[0]
            if sample == candidate or sample.startswith(candidate + "{"):
                rows[sample] = float(fields[-1])
        if rows:
            return candidate, rows
    raise RuntimeError(f"none of the required metrics are exposed: {candidates}")


def snapshot(root, require_spec=True):
    text = urllib.request.urlopen(root + "/metrics", timeout=30).read().decode()
    values = {}
    resolved = {}
    for logical, candidates in COUNTER_CANDIDATES.items():
        try:
            resolved[logical], rows = metric_series(text, candidates)
            values[logical] = sum(rows.values())
        except RuntimeError:
            if require_spec or not logical.startswith("mtp_"):
                raise
            resolved[logical] = None
            values[logical] = 0.0
    if require_spec:
        position_name, positions = metric_series(text, PER_POSITION_CANDIDATES)
    else:
        position_name, positions = None, {}
    return {
        "values": values,
        "resolved_names": resolved,
        "accepted_per_position_name": position_name,
        "accepted_per_position": positions,
    }


def request(url, root, model, messages, max_tokens, raw_path, cache_expected,
            expected_prompt_tokens=None, require_spec=True, ignore_eos=False):
    before = snapshot(root, require_spec=require_spec)
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": ignore_eos,
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start_ns = time.monotonic_ns()
    first_generated_ns = None
    first_visible_ns = None
    generated_events = 0
    usage = None
    finish_reason = None
    reasoning_parts = []
    content_parts = []
    with open(raw_path, "x") as raw, urllib.request.urlopen(req, timeout=7200) as response:
        for encoded in response:
            now_ns = time.monotonic_ns()
            line = encoded.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            raw.write(json.dumps({"monotonic_ns": now_ns, "payload": payload}) + "\n")
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                reasoning = delta.get("reasoning_content")
                content = delta.get("content")
                if reasoning:
                    reasoning_parts.append(reasoning)
                if content:
                    content_parts.append(content)
                if reasoning or content:
                    generated_events += 1
                    if first_generated_ns is None:
                        first_generated_ns = now_ns
                    if content and first_visible_ns is None:
                        first_visible_ns = now_ns
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    end_ns = time.monotonic_ns()
    after = snapshot(root, require_spec=require_spec)
    if before["resolved_names"] != after["resolved_names"]:
        raise RuntimeError("Prometheus counter names changed during request")
    usage = usage or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    deltas = {
        name: after["values"][name] - before["values"][name]
        for name in COUNTER_CANDIDATES
    }
    position_keys = set(before["accepted_per_position"]) | set(after["accepted_per_position"])
    per_position_deltas = {
        name: after["accepted_per_position"].get(name, 0.0)
        - before["accepted_per_position"].get(name, 0.0)
        for name in sorted(position_keys)
    }
    if expected_prompt_tokens is not None and prompt_tokens != expected_prompt_tokens:
        raise RuntimeError(
            f"endpoint prompt-token mismatch: expected {expected_prompt_tokens}, got {prompt_tokens}")
    if cache_expected == "cold" and deltas["prefix_cache_hits"] != 0:
        raise RuntimeError(
            f"cold-prefix contamination: cache-hit delta={deltas['prefix_cache_hits']}")
    if first_generated_ns is None or completion_tokens < 1:
        raise RuntimeError(
            f"request produced no timed output token: completion_tokens={completion_tokens}")

    generation_interval_s = max((end_ns - first_generated_ns) / 1e9, 1e-9)
    post_first_tokens = completion_tokens - 1
    accepted = deltas["mtp_accepted_tokens"]
    proposed = deltas["mtp_draft_tokens"]
    post_first_tpot_s = (
        generation_interval_s / post_first_tokens if post_first_tokens else None)
    record = {
        "request_start_ns": start_ns,
        "first_generated_ns": first_generated_ns,
        "first_visible_ns": first_visible_ns,
        "request_end_ns": end_ns,
        "ttft_s": (first_generated_ns - start_ns) / 1e9,
        "ttfc_s": ((first_visible_ns - start_ns) / 1e9) if first_visible_ns else None,
        "total_s": (end_ns - start_ns) / 1e9,
        "post_first_generation_s": generation_interval_s,
        "post_first_tpot_s": post_first_tpot_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "client_post_first_tps": (
            post_first_tokens / generation_interval_s if post_first_tokens else None),
        "finish_reason": finish_reason,
        "reasoning_text": "".join(reasoning_parts),
        "content_text": "".join(content_parts),
        "generated_stream_events": generated_events,
        "counter_names": before["resolved_names"],
        "counter_before": before["values"],
        "counter_after": after["values"],
        "counter_delta": deltas,
        "prefix_cache_hits_delta": deltas["prefix_cache_hits"],
        "prefix_cache_queries_delta": deltas["prefix_cache_queries"],
        "mtp_proposed_tokens": proposed,
        "mtp_accepted_tokens": accepted,
        "mtp_acceptance_rate": accepted / proposed if proposed else None,
        "mtp_accepted_per_position_delta": per_position_deltas,
        "cache_expected": cache_expected,
        "expected_prompt_tokens": expected_prompt_tokens,
        "messages_sha256": hashlib.sha256(
            json.dumps(messages, sort_keys=True).encode()).hexdigest(),
        "raw_sse": raw_path,
    }
    record["input_tokens_per_ttft_s"] = prompt_tokens / max(record["ttft_s"], 1e-9)
    return record


def stats(values):
    if not values:
        return {
            "n": 0,
            "median": None,
            "mean": None,
            "min": None,
            "max": None,
            "pstdev": None,
        }
    return {
        "n": len(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "pstdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scheduler", "context", "pi"], required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--target", type=int)
    parser.add_argument("--output", type=int, default=128)
    parser.add_argument("--root", default="http://127.0.0.1:8001")
    parser.add_argument("--no-spec", action="store_true")
    parser.add_argument("--full-output-warmup", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=False)
    with open(args.prompts) as source:
        prompt_data = json.load(source)
    prompts = prompt_data["prompts"]
    if args.target is not None:
        prompts = [p for p in prompts if p["target_tokens"] == args.target]
    if len(prompts) < args.reps + 1:
        raise RuntimeError(
            f"need one shape-warmup prompt plus {args.reps} unique measured prompts; "
            f"found {len(prompts)}")

    print("milestone=warmup_generic_start", flush=True)
    generic_content = "RUN" + os.urandom(48).hex() + " Warm the loaded model."
    generic = request(
        args.root + "/v1/chat/completions", args.root, args.model,
        [{"role": "user", "content": generic_content}], 16,
        args.outdir + "/warmup-generic.sse.jsonl", "cold",
        require_spec=not args.no_spec)
    print("milestone=warmup_generic_done", flush=True)

    shape_prompt = prompts[0]
    print(
        f"milestone=warmup_shape_start target={shape_prompt['calibrated_tokens']}",
        flush=True)
    shape = request(
        args.root + "/v1/chat/completions", args.root, args.model,
        shape_prompt["messages"],
        args.output if args.full_output_warmup else min(args.output, 32),
        args.outdir + "/warmup-shape.sse.jsonl", "cold",
        expected_prompt_tokens=shape_prompt["calibrated_tokens"],
        require_spec=not args.no_spec,
        ignore_eos=args.ignore_eos)
    print("milestone=warmup_shape_done", flush=True)

    records = []
    measured_prompts = prompts[1:args.reps + 1]
    for rep, item in enumerate(measured_prompts, start=1):
        print(
            f"milestone=rep_start rep={rep} target={item['calibrated_tokens']}",
            flush=True)
        raw_path = args.outdir + f"/rep{rep}.sse.jsonl"
        record = request(
            args.root + "/v1/chat/completions", args.root, args.model,
            item["messages"], args.output, raw_path, "cold",
            expected_prompt_tokens=item["calibrated_tokens"],
            require_spec=not args.no_spec,
            ignore_eos=args.ignore_eos)
        record.update({
            "rep": rep,
            "mode": args.mode,
            "scheduler_budget": args.budget,
            "target_prompt_tokens": item["target_tokens"],
            "calibrated_prompt_tokens": item["calibrated_tokens"],
            "family": item["family"],
            "scenario": item.get("scenario"),
            "requested_output_tokens": args.output,
            "ignore_eos": args.ignore_eos,
            "entropy_prefix": item["entropy_prefix"],
        })
        records.append(record)
        print(json.dumps(record), flush=True)
        print(f"milestone=rep_done rep={rep}", flush=True)

    normalized_input_rates = [r["input_tokens_per_ttft_s"] for r in records]
    decode_values = [
        r["client_post_first_tps"] for r in records
        if r["client_post_first_tps"] is not None
    ]
    output = {
        "mode": args.mode,
        "budget": args.budget,
        "target": args.target,
        "requested_output_tokens": args.output,
        "ignore_eos": args.ignore_eos,
        "timing_note": (
            "input_tokens_per_ttft_s is endpoint input tokens divided by client TTFT; "
            "it is not isolated engine prefill throughput"),
        "warmups_discarded": {"generic": generic, "shape": shape},
        "records": records,
        "summary": {
            "input_tokens_per_ttft_s": stats(normalized_input_rates),
            "client_post_first_decode_tps": stats(decode_values),
            "actual_prompt_tokens": [r["prompt_tokens"] for r in records],
            "actual_completion_tokens": [r["completion_tokens"] for r in records],
            "prefix_cache_hits_delta": sum(
                r["prefix_cache_hits_delta"] for r in records),
            "mtp_proposed_tokens": sum(r["mtp_proposed_tokens"] for r in records),
            "mtp_accepted_tokens": sum(r["mtp_accepted_tokens"] for r in records),
        },
    }
    with open(args.outdir + "/results.json", "x") as destination:
        json.dump(output, destination, indent=2)


if __name__ == "__main__":
    main()
