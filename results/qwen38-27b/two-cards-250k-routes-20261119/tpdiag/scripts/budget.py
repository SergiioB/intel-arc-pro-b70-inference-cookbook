#!/usr/bin/env python3
"""Close the per-round / per-step budget from the per-rank profiles (prof_summary format). Usage: budget.py TP1_LOG TP2_LOG UNPROF_TP1_MS UNPROF_TP2_MS LABEL"""
import re, sys
from collections import defaultdict
LINE = re.compile(r"^  (\S.*?)\s+([\d.]+) ms\s+[\d.]+ %\s+(\d+) calls\s+([\d.]+) ms idle before$")
HEAD = re.compile(r"GPU time, (.+?)(?:, rank (\d+))?: ([\d.]+) ms")
def load(path, label):
    text = open(path).read()
    d = re.search(r"drafted: (\d+) tokens in (\d+) rounds", text); s = re.search(r"decode: (\d+) tokens", text)
    n = int(d[2]) if label == "drafted rounds" else int(s[1])
    out = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0])); sec = None
    for ln in text.splitlines():
        m = HEAD.match(ln)
        if m: sec = int(m[2] or 0) if m[1] == label else None; continue
        m = LINE.match(ln)
        if m and sec is not None:
            out[sec][m[1]] = [float(m[2]) / n, float(m[3]) / n, float(m[4]) / n]
    return out
tp1, tp2 = load(sys.argv[1], sys.argv[5]), load(sys.argv[2], sys.argv[5])
CLS = {"GEMM (weights)": ["dec_wide", "dec_wide64", "dec_narrow", "ngen_narrow", "ngen_wide"],
       "gdn_step": ["gdn_step"],
       "norms, attention, GDN small kernels": ["add_rmsnorm_dec", "add_rmsnorm", "gdn_pre", "gated_norm", "attn_prep", "attn_gate", "swiglu", "conv_commit", "attn_decode", "attn_merge", "kv_move", "attn_pref", "attn_tail", "attn_merge2", "embed", "tap"],
       "argmax (full vocab)": ["argmax_part", "argmax"],
       "drafter's own kernels (other)": ["(other)"],
       "exchange kernel tp_ar2_f16": ["tp_ar2_f16"]}
def cls(rk, d):
    return {c: sum(d[rk].get(k, [0, 0, 0])[0] for k in ks) for c, ks in CLS.items()}
a, r0, r1 = cls(0, tp1), cls(0, tp2), cls(1, tp2)
idle0 = sum(v[2] for v in tp2[0].values()); idle1 = sum(v[2] for v in tp2[1].values())
print(f"{sys.argv[5]}: class | one card | TP2 rank 0 | TP2 rank 1 | rank 1 - half of one card | rank 0 - rank 1")
for c in CLS:
    print(f"{c:40s} {a[c]:7.3f} {r0[c]:7.3f} {r1[c]:7.3f} {r1[c]-a[c]/2:+7.3f} {r0[c]-r1[c]:+7.3f}")
b1, b0, bb1 = sum(a.values()), sum(r0.values()), sum(r1.values())
print(f"{'GPU busy':40s} {b1:7.3f} {b0:7.3f} {bb1:7.3f}")
print(f"idle between kernels (profiled): rank 0 {idle0:.3f}, rank 1 {idle1:.3f}; idle before tp_ar2: rank 0 {tp2[0]['tp_ar2_f16'][2]:.3f}, rank 1 {tp2[1]['tp_ar2_f16'][2]:.3f}")
u1, u2 = float(sys.argv[3]), float(sys.argv[4])
print(f"unprofiled wall: one card {u1}, two cards {u2}; wall - busy: one card {u1-b1:.3f}, two cards (rank 0) {u2-b0:.3f}")
# exchange-call counts for the per-call idle
print(f"tp_ar2 calls per unit: {tp2[0]['tp_ar2_f16'][1]:.0f}; dec_wide calls one card {tp1[0]['dec_wide'][1]:.0f}/TP2 {tp2[1]['dec_wide'][1]:.0f}; dec_wide64 {tp1[0]['dec_wide64'][1]:.0f}/{tp2[1]['dec_wide64'][1]:.0f}; dec_narrow {tp1[0]['dec_narrow'][1]:.0f}/{tp2[1]['dec_narrow'][1]:.0f}/{tp2[0]['dec_narrow'][1]:.0f}")
