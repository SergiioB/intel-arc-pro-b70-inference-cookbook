#!/usr/bin/env python3
"""Measure changed follow-ups over one prepared long Pi session."""
import argparse
import hashlib
import importlib.util
import json
import os
import statistics


def stats(values):
    return {
        "n": len(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "pstdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def load_request_module(path):
    spec = importlib.util.spec_from_file_location("b70_context_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-module", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--root", default="http://127.0.0.1:8001")
    parser.add_argument("--target", type=int, default=120000)
    parser.add_argument("--output", type=int, default=128)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--cache-enabled", action="store_true")
    parser.add_argument("--no-spec", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=False)
    with open(args.prompts) as source:
        prompt_data = json.load(source)
    candidates = [
        prompt for prompt in prompt_data["prompts"]
        if prompt["target_tokens"] == args.target
    ]
    if not candidates:
        raise RuntimeError(f"no calibrated prompt for target {args.target}")

    module = load_request_module(args.request_module)
    base = candidates[0]
    url = args.root + "/v1/chat/completions"
    require_spec = not args.no_spec

    print(f"milestone=session_prepare_start target={args.target}", flush=True)
    preparation = module.request(
        url, args.root, args.model, base["messages"], args.output,
        args.outdir + "/preparation.sse.jsonl", "cold",
        expected_prompt_tokens=base["calibrated_tokens"],
        require_spec=require_spec,
    )
    assistant_text = preparation["content_text"]
    assistant_source = "content_text"
    if not assistant_text:
        assistant_text = preparation["reasoning_text"]
        assistant_source = "reasoning_text_fallback"
    if not assistant_text:
        raise RuntimeError("preparation produced no reusable assistant text")
    print("milestone=session_prepare_done", flush=True)

    questions = [
        "List the three most important constraints from this session.",
        "Which configuration risk should I check first, and why?",
        "Give one concrete next action based on the retained context.",
        "Name the strongest measured fact and its exact limitation.",
        "What changed between the historical and current recipe?",
    ]
    if args.reps > len(questions):
        raise RuntimeError(f"maximum supported repetitions is {len(questions)}")

    records = []
    for index, question in enumerate(questions[:args.reps], start=1):
        messages = list(base["messages"]) + [
            {"role": "assistant", "content": assistant_text},
            {"role": "user", "content": question},
        ]
        print(f"milestone=session_rep_start rep={index}", flush=True)
        record = module.request(
            url, args.root, args.model, messages, args.output,
            args.outdir + f"/rep{index}.sse.jsonl",
            "warm" if args.cache_enabled else "disabled",
            require_spec=require_spec,
        )
        hits = record["prefix_cache_hits_delta"]
        queries = record["prefix_cache_queries_delta"]
        if args.cache_enabled and hits <= 0:
            raise RuntimeError(f"expected resident-prefix hits, got {hits}")
        if not args.cache_enabled and hits != 0:
            raise RuntimeError(f"cache-disabled request recorded hits={hits}")
        if record["completion_tokens"] != args.output:
            raise RuntimeError(
                f"rep {index}: expected {args.output} output tokens, "
                f"got {record['completion_tokens']}")
        record.update({
            "rep": index,
            "question": question,
            "cache_enabled": args.cache_enabled,
            "reused_tokens": hits,
            "queried_tokens": queries,
            "novel_or_unreused_tokens": record["prompt_tokens"] - hits,
            "assistant_source": assistant_source,
            "assistant_text_sha256": hashlib.sha256(
                assistant_text.encode()).hexdigest(),
        })
        records.append(record)
        print(json.dumps(record), flush=True)
        print(f"milestone=session_rep_done rep={index}", flush=True)

    fields = {
        "ttft_s": [record["ttft_s"] for record in records],
        "ttfc_s": [record["ttfc_s"] for record in records
                   if record["ttfc_s"] is not None],
        "total_s": [record["total_s"] for record in records],
        "post_first_tpot_s": [record["post_first_tpot_s"] for record in records],
        "client_post_first_tps": [record["client_post_first_tps"] for record in records],
        "prompt_tokens": [record["prompt_tokens"] for record in records],
        "reused_tokens": [record["reused_tokens"] for record in records],
        "novel_or_unreused_tokens": [record["novel_or_unreused_tokens"] for record in records],
    }
    summary = {name: stats(values) for name, values in fields.items() if values}
    summary.update({
        "cache_enabled": args.cache_enabled,
        "preparation_prompt_tokens": preparation["prompt_tokens"],
        "preparation_completion_tokens": preparation["completion_tokens"],
        "preparation_total_s": preparation["total_s"],
        "preparation_content_sha256": hashlib.sha256(
            assistant_text.encode()).hexdigest(),
        "prefix_cache_hits_delta": sum(
            record["prefix_cache_hits_delta"] for record in records),
        "prefix_cache_queries_delta": sum(
            record["prefix_cache_queries_delta"] for record in records),
        "mtp_proposed_tokens": sum(record["mtp_proposed_tokens"] for record in records),
        "mtp_accepted_tokens": sum(record["mtp_accepted_tokens"] for record in records),
    })
    output = {
        "scenario": "resident_pi_session_changed_followups",
        "target_preparation_prompt_tokens": args.target,
        "requested_output_tokens": args.output,
        "repetitions": args.reps,
        "cache_enabled": args.cache_enabled,
        "timing_note": "client monotonic timings; TTFC excludes reasoning-only events",
        "preparation": preparation,
        "records": records,
        "summary": summary,
    }
    with open(args.outdir + "/results.json", "x") as destination:
        json.dump(output, destination, indent=2)


if __name__ == "__main__":
    main()
