#!/usr/bin/env python3
"""One summary of the 2026-11-10 Qwen3.8-27B route comparison (runs 3, 4, 5, 7)."""
import json, os, statistics as st, sys
W = os.path.dirname(os.path.abspath(__file__))
def jl(p):
    p = os.path.join(W, p)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def med(v): return round(st.median(v), 1) if v else None
ROUTES = {"E3": "EXL3 4.00 bpw + MTP3 (exl3xpu, vLLM 0.26.1 fork)", "E0": "EXL3 4.00 bpw, MTP off",
          "C": "TensorFold v1.2.5 + DFlash2 (MLX 4-bit)", "D": "llama.cpp b12244 SYCL, Q4_K_M, q8_0 KV"}
out = {"conditions": "Qwen3.8-27B, one Arc Pro B70 (GPU 1), 180 W cap, C1, prefix cache on, model-card sampling "
       "(thinking T1.0/top_p0.95/top_k20; non-thinking T0.7/top_p0.8/top_k20/presence 1.5, TensorFold presence 0)", "routes": {}}
for r, name in ROUTES.items():
    o = {"route": name}
    short = jl(f"out3/{r}/cells.jsonl") + (jl(f"out4/{r}/cells.jsonl") if r in ("E0", "E3") else [])
    for arm in ("think", "nothink"):
        for d in ("0", "8k"):
            v = [c["decode_tok_s"] for c in short if c["arm"] == arm and c["depth"] == d and c.get("decode_tok_s")]
            if v: o[f"short_{arm}_{d}"] = med(v)
    deep = jl(f"out5/{r}/cells.jsonl") + jl(f"out7/{r}/cells.jsonl")
    o["depth"] = {}
    for d in ("0", "8k", "32k", "64k", "128k", "240k"):
        cells = [c for c in deep if c["depth"] == d]
        v = [c["decode_tok_s"] for c in cells if c.get("decode_tok_s")]
        t = [c["ttft_s"] for c in cells if c.get("ttft_s") is not None]
        if v: o["depth"][d] = {"decode_tok_s": med(v), "cold_ttft_s": round(min(t), 1) if t else None,
                               "prompt_tokens": cells[0].get("prompt_tokens"), "n": len(v)}
    import glob
    scores = []
    for f in sorted(glob.glob(os.path.join(W, f"out3/{r}/battery-nothink-*/summary.json"))):
        g = json.load(open(f)).get("graded", {}).get("total", {})
        scores.append(g.get("passed"))
    if scores: o["battery_40_sampled"] = scores
    s = jl(f"out3/{r}/session-nothink/session.jsonl")
    if s: o["session_8k_10turns"] = {"followup_ttft_median_s": med([t["ttft_s"] for t in s[1:]]), "decode_median": med([t["decode_tok_s"] for t in s])}
    s64 = jl(f"out7/{r}/session/session.jsonl")
    if s64: o["session_64k_10turns"] = {"turn0_ttft_s": s64[0]["ttft_s"], "followup_ttft_median_s": med([t["ttft_s"] for t in s64[1:]]),
                                        "decode_median": med([t["decode_tok_s"] for t in s64])}
    for p in (f"out5/{r}/needles.json", f"out7/{r}/needles.json"):
        if os.path.exists(os.path.join(W, p)):
            n = json.load(open(os.path.join(W, p)))
            o["needles"] = {d: [x["found"] for x in n if x["depth"] == d] for d in ("32k", "128k", "240k") if any(x["depth"] == d for x in n)}
    for p, k in ((f"out4/{r}/prefix-check-warm.json", "prefix_cache_check_warm"), (f"out4/{r}/prefix-check.json", "prefix_cache_check")):
        if os.path.exists(os.path.join(W, p)):
            j = json.load(open(os.path.join(W, p))); o[k] = f"{j['correct']}/{j['n']} correct, {j['degenerate']} degenerate"
    out["routes"][r] = o
if os.path.exists(os.path.join(W, "out4/E0off/prefix-check.json")):
    j = json.load(open(os.path.join(W, "out4/E0off/prefix-check.json"))); out["routes"]["E0"]["prefix_cache_check_cache_off"] = f"{j['correct']}/{j['n']} correct"
out["tensorfold_context_limit"] = "refuses 262144 with a 2 GiB prompt cache and 131072 with 4 GiB on one 32 GB card; runs 131072 with 2 GiB"
json.dump(out, open(os.path.join(W, "summary.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
