#!/usr/bin/env python3
"""Summarize route-verify v3: speed, sampled quality, chat sessions, cache-corruption check."""
import glob
import json
import os
import statistics as st
import sys

R = sys.argv[1] if len(sys.argv) > 1 else "/data/nvme/work/route-verify-20261110/out3"
NAMES = {"D": "llama.cpp b12244 Q4_K_M", "C": "TensorFold v1.2.5 + DFlash2", "E0": "EXL3 4.00bpw, MTP off", "E3": "EXL3 4.00bpw, MTP3"}


def jl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


out = {}
for r, name in NAMES.items():
    d = os.path.join(R, r)
    if not os.path.isdir(d):
        continue
    row = {"route": name}
    cells = jl(os.path.join(d, "cells.jsonl"))
    for arm in ("think", "nothink"):
        for depth in ("0", "8k"):
            v = [c["decode_tok_s"] for c in cells if c["arm"] == arm and c["depth"] == depth and c.get("decode_tok_s")]
            if v:
                row[f"decode_{arm}_{depth}"] = round(st.median(v), 1)
                row[f"decode_{arm}_{depth}_n"] = len(v)
        cold = [c["ttft_s"] for c in cells if c["arm"] == arm and c["depth"] == "8k" and c["rep"] == 0 and c.get("ttft_s") is not None]
        if cold:
            row[f"cold_ttft_8k_{arm}"] = round(st.median(cold), 2)
    scores, code, reg, canary14 = [], [], [], []
    for f in sorted(glob.glob(os.path.join(d, "battery-nothink-*/summary.json"))):
        s = json.load(open(f))
        g = s.get("graded", {})
        scores.append(g.get("total", {}).get("passed"))
        code.append(g.get("code", {}).get("passed"))
        reg.append(g.get("regression", {}).get("passed"))
        canary14.append(bool(s.get("gates", {}).get("a") == "PASS"))
    if scores:
        row["battery_scores"] = scores
        row["battery_mean"] = round(st.mean(scores), 1)
        row["battery_code"] = code
        row["battery_regression"] = reg
        row["batt_canaries_pass_runs"] = sum(canary14)
    ses = jl(os.path.join(d, "session-nothink", "session.jsonl"))
    if ses:
        follow = ses[1:]
        row["session_turn0_ttft"] = ses[0].get("ttft_s")
        row["session_followup_ttft_median"] = round(st.median(t["ttft_s"] for t in follow), 2) if follow else None
        row["session_followup_ttft_max"] = round(max(t["ttft_s"] for t in follow), 2) if follow else None
        row["session_decode_median"] = round(st.median(t["decode_tok_s"] for t in ses), 1)
        row["session_ctx_end"] = ses[-1].get("context_tokens") or ses[-1].get("prompt_tokens")
    probe = os.path.join(d, "probe.txt")
    if os.path.exists(probe):
        row["probe"] = open(probe).read().strip().replace("\n", " ")
    e = os.path.join(d, "energy.json")
    if os.path.exists(e):
        row["energy"] = json.load(open(e))
    out[r] = row


def sha_turns(path):
    return [t.get("sha256") for t in jl(path)]


cc = {}
for name, on_globs, off in (
    ("E0", ["E0/session-greedy-cacheon-1/session.jsonl", "E0/session-greedy-cacheon-2/session.jsonl"], "E0off/session-greedy-cacheoff/session.jsonl"),
    ("C", ["C/session-greedy-cacheon/session.jsonl"], "C0/session-greedy-cacheoff/session.jsonl"),
):
    ons = [sha_turns(os.path.join(R, g)) for g in on_globs]
    offs = sha_turns(os.path.join(R, off))
    if not offs or not any(ons):
        continue
    res = {"turns": len(offs)}
    res["on_vs_off_equal_turns"] = [sum(a == b for a, b in zip(o, offs)) for o in ons]
    if len(ons) == 2:
        res["on_vs_on_equal_turns"] = sum(a == b for a, b in zip(ons[0], ons[1]))
    first_diff = [next((i for i, (a, b) in enumerate(zip(o, offs)) if a != b), None) for o in ons]
    res["first_divergent_turn_vs_off"] = first_diff
    cc[name] = res
out["cache_check"] = cc
print(json.dumps(out, indent=1))
