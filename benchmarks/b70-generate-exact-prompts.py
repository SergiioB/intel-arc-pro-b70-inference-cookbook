#!/usr/bin/env python3
"""Generate entropy-first prompts at exact rendered chat-template token lengths."""
import argparse
import hashlib
import json
import secrets
import string
from transformers import AutoTokenizer

FAMILIES = {
    "assistant": "You are Pi, a practical private assistant. Analyze the supplied conversation and answer accurately. ",
    "research": "You are helping with a technical research session. Preserve evidence, constraints, and uncertainty. ",
    "rag": "Use the following retrieved notes to answer a new question without inventing unsupported details. ",
    "tool": "Review the following tool results and produce a concise operational recommendation with safety checks. ",
    "document": "Analyze this long document as a resident assistant and retain its important facts for follow-up questions. ",
}
BODY = (
    "The session contains architecture notes, benchmark observations, configuration changes, user questions, "
    "tool outputs, error reports, and follow-up decisions. The assistant must distinguish measured facts from "
    "estimates, preserve exact context, identify operational risks, and provide a useful response. "
)
ALPHABET = string.ascii_letters + string.digits


def rendered(tokenizer, content, system_prompt=None):
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True)
    return len(encoded["input_ids"])


def exact_content(tokenizer, target, family, entropy, system_prompt=None):
    prefix = entropy + " " + FAMILIES[family]

    # Find the largest repeated-body prompt that remains below the target. This
    # keeps calibration logarithmic; repeatedly decoding 9K-token prefixes is
    # prohibitively slow with this tokenizer.
    low, high = 0, 1
    while rendered(tokenizer, prefix + BODY * high, system_prompt) < target:
        low, high = high, high * 2
    while low + 1 < high:
        mid = (low + high) // 2
        if rendered(tokenizer, prefix + BODY * mid, system_prompt) <= target:
            low = mid
        else:
            high = mid
    candidate = prefix + BODY * low
    count = rendered(tokenizer, candidate, system_prompt)

    # " x" is one token for this tokenizer in ordinary text. Add it in a
    # binary-sized batch, then close any small tokenizer-boundary discrepancy
    # with a bounded set of short fillers.
    remaining = target - count
    if remaining:
        trial = candidate + " x" * remaining
        trial_count = rendered(tokenizer, trial, system_prompt)
        if trial_count <= target:
            candidate, count = trial, trial_count
    fillers = [" x", " y", " z", " analysis", " evidence", "."]
    for _ in range(64):
        if count == target:
            break
        for filler in fillers:
            trial = candidate + filler
            trial_count = rendered(tokenizer, trial, system_prompt)
            if count < trial_count <= target:
                candidate, count = trial, trial_count
                break
        else:
            raise RuntimeError(f"cannot calibrate target {target}, stuck at {count}")
    if count != target:
        raise RuntimeError(
            f"failed exact calibration for {family} target {target}: {count}")
    if not candidate.startswith(entropy):
        raise RuntimeError("entropy prefix was lost during calibration")
    return candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--per-target", type=int, default=5)
    parser.add_argument("--system-prompt-file")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    system_prompt = None
    system_prompt_file_sha256 = None
    if args.system_prompt_file:
        with open(args.system_prompt_file, "rb") as source:
            system_prompt_bytes = source.read()
        system_prompt_file_sha256 = hashlib.sha256(system_prompt_bytes).hexdigest()
        system_prompt = system_prompt_bytes.decode().rstrip("\n")
    system_prompt_sha256 = (
        hashlib.sha256(system_prompt.encode()).hexdigest()
        if system_prompt is not None else None)

    families = list(FAMILIES)
    prompts = []
    for target in [int(x) for x in args.targets.split(",")]:
        for index in range(args.per_target):
            entropy = "RUN" + "".join(secrets.choice(ALPHABET) for _ in range(96))
            family = families[index % len(families)]
            content = exact_content(
                tokenizer, target, family, entropy, system_prompt)
            count = rendered(tokenizer, content, system_prompt)
            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": content})
            prompts.append({
                "target_tokens": target,
                "calibrated_tokens": count,
                "family": family,
                "scenario": "designed_realworld_pi_cold_context",
                "entropy_prefix": entropy,
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "system_prompt_sha256": system_prompt_sha256,
                "system_prompt_file_sha256": system_prompt_file_sha256,
                "messages": messages,
            })
    with open(args.output, "x") as destination:
        json.dump({
            "schema": 1,
            "tokenizer": args.model,
            "system_prompt_sha256": system_prompt_sha256,
            "system_prompt_file_sha256": system_prompt_file_sha256,
            "prompts": prompts,
        }, destination, indent=2)


if __name__ == "__main__":
    main()
