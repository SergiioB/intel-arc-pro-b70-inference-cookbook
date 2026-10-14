#!/usr/bin/env python3
"""Score PLAN-v2 (SC5 only, chat panel, two passes per server). Reuses analyze.py's estimators.
Usage: analyze_v2.py RAW_V2_DIR RAW_V1_DIR OUT_JSON
"""
import json, sys
from pathlib import Path

RAW2, RAW1, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
src = (Path(__file__).parent / "analyze.py").read_text().split("report = {")[0].replace(
    "RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2])", "")
ns = {}
exec(src, ns)
coarse_kl, per_position, summarize = ns["coarse_kl"], ns["per_position"], ns["summarize"]


def load(raw, arm, rep):
    ns["RAW"] = raw
    return ns["load"](arm, rep)


A = {(a, r): load(RAW2, a, r) for a in ("sc5_fp16", "sc5_int8") for r in (1, 2)}
pairs = {
    "effect r1: fp16 -> int8": summarize(per_position(A["sc5_fp16", 1], A["sc5_int8", 1])),
    "effect r2: fp16 -> int8": summarize(per_position(A["sc5_fp16", 2], A["sc5_int8", 2])),
    "noise fp16: r1 -> r2": summarize(per_position(A["sc5_fp16", 1], A["sc5_fp16", 2])),
    "noise int8: r1 -> r2": summarize(per_position(A["sc5_int8", 1], A["sc5_int8", 2])),
}
# cross-server reproducibility on rows shared with v1 (fp16 arm, different server load)
v1 = {a: load(RAW1, a, 1) for a in ("sc5_fp16", "sc5_int8")}
for a in v1:
    shared = {k: A[a, 1][k] for k in v1[a] if k in A[a, 1]}
    pairs[f"cross-load {a}: v1 -> v2"] = summarize(per_position({k: v1[a][k] for k in shared}, shared))

RL179_SC5_CHAT = 0.00141
e1, e2 = pairs["effect r1: fp16 -> int8"]["P2"], pairs["effect r2: fp16 -> int8"]["P2"]
noise = pairs["noise fp16: r1 -> r2"]["P2"]
E = (e1["kl_lb"] + e2["kl_lb"]) / 2
lo = min(e1["kl_lb_ci"][0], e2["kl_lb_ci"][0])
valid = noise["kl_lb"] <= 0.1 * E
K, K_lo = E / RL179_SC5_CHAT, lo / RL179_SC5_CHAT
verdict = ("inconclusive (fp16 noise floor too high)" if not valid else
           "material" if K_lo >= 0.25 else "negligible" if K < 0.10 else "minor")
decision = {"E_kl_lb": E, "E_ci_low": lo, "fp16_noise_kl_lb": noise["kl_lb"], "valid": valid,
            "K": K, "K_ci_low": K_lo, "dnll_r1": e1["dnll"], "dnll_r1_ci": e1["dnll_ci"],
            "dnll_r2": e2["dnll"], "dnll_r2_ci": e2["dnll_ci"], "top1_r1": e1["top1"], "verdict": verdict}
OUT.write_text(json.dumps({"pairs": pairs, "decision": decision}, indent=1))
for name, s in pairs.items():
    for panel, v in s.items():
        print(f"{name:36s} {panel} n={v['n']:6d} KLlb={v['kl_lb']:.5f} [{v['kl_lb_ci'][0]:.5f},{v['kl_lb_ci'][1]:.5f}] "
              f"tail={v['tail_p']:.4f} top1={v['top1']:.4f} dNLL={v['dnll']:+.5f} [{v['dnll_ci'][0]:+.5f},{v['dnll_ci'][1]:+.5f}]")
print(json.dumps(decision, indent=1))
