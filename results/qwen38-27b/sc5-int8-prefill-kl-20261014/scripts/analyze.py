#!/usr/bin/env python3
"""Score the as-served int8-prefill check. Stdlib only.

Coarse KL(P||Q) per position: S = P's top-k tokens whose Q logprob is known (in Q's returned set),
plus one tail bucket for everything else. Grouping tokens can only lower KL, so this is a lower
bound on the true KL, tight when the tail mass is small (reported as tail_p). NLL uses the
observed token's logprob, which the server always returns, so it is exact.
CIs: 2000 bootstrap resamples of 256-position blocks within each panel.
Usage: analyze.py RAW_DIR OUT_JSON
"""
import gzip, json, math, random, sys
from pathlib import Path

RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2])
# RL-179 reference-weight KL vs BF16 (wikitext / chat), for magnitude context only (different slices)
RL179 = {"plain": {"P1": 0.01391, "P2": 0.00876}, "sc5": {"P1": 0.00582, "P2": 0.00141}}


def load(arm, rep):
    d = {}
    with gzip.open(RAW / f"{arm}_rep{rep}.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            d[r["row"]] = r
    return d


def coarse_kl(ptop, qtop):
    q = dict(qtop)
    s = [(t, lp) for t, lp in ptop if t in q]
    ps = sum(math.exp(lp) for _, lp in s)
    qs = sum(math.exp(q[t]) for t, _ in s)
    pr, qr = max(1.0 - ps, 1e-12), max(1.0 - qs, 1e-12)
    kl = sum(math.exp(lp) * (lp - q[t]) for t, lp in s) + pr * math.log(pr / qr)
    return max(kl, 0.0), pr


def per_position(A, B):
    """A = reference arm (fp16 prefill), B = candidate. Returns {panel: list of tuples}."""
    out = {}
    for row, ra in A.items():
        rb = B[row]
        assert [x[:2] for x in ra["pos"]] == [x[:2] for x in rb["pos"]], row
        for pa, pb in zip(ra["pos"], rb["pos"]):
            kl, tail = coarse_kl(pa[3], pb[3])
            top1 = pa[3][0][0] == pb[3][0][0]
            out.setdefault(ra["panel"], []).append((kl, tail, top1, -pa[2], -pb[2]))
    return out


def boot(xs, f, n=2000, blk=256, seed=20261014):
    blocks = [xs[i:i + blk] for i in range(0, len(xs), blk)]
    rng = random.Random(seed)
    vals = sorted(f([x for _ in blocks for x in rng.choice(blocks)]) for _ in range(n))
    return [vals[int(0.025 * n)], vals[int(0.975 * n)]]


def mean(v):
    return sum(v) / len(v)


def summarize(pp):
    res = {}
    for panel, xs in sorted(pp.items()):
        res[panel] = {
            "n": len(xs),
            "kl_lb": mean([x[0] for x in xs]), "kl_lb_ci": boot(xs, lambda s: mean([x[0] for x in s])),
            "tail_p": mean([x[1] for x in xs]),
            "top1": mean([x[2] for x in xs]),
            "nll_ref": mean([x[3] for x in xs]), "nll_cand": mean([x[4] for x in xs]),
            "dnll": mean([x[4] - x[3] for x in xs]), "dnll_ci": boot(xs, lambda s: mean([x[4] - x[3] for x in s])),
        }
    return res


report = {"method": __doc__.strip().splitlines()[0], "pairs": {}, "nll": {}, "decision": {}}
arms = {}
for a in ("plain_int8", "plain_fp16", "sc5_int8", "sc5_fp16"):
    arms[a] = load(a, 1)
for art in ("plain", "sc5"):
    report["pairs"][f"{art}: fp16 -> int8 prefill"] = summarize(per_position(arms[f"{art}_fp16"], arms[f"{art}_int8"]))
# determinism floor: same arm, same server, second pass (P1 only)
for a in arms:
    p2 = RAW / f"{a}_rep2.jsonl.gz"
    if p2.exists():
        r2 = load(a, 2)
        r1 = {k: v for k, v in arms[a].items() if k in r2}
        report["pairs"][f"{a}: rep1 -> rep2"] = summarize(per_position(r1, r2))
# quant gap in NLL on identical tokens, as served, both prefill modes
qp = {m: summarize(per_position(arms[f"sc5_{m}"], arms[f"plain_{m}"])) for m in ("fp16", "int8")}
report["pairs"]["fp16: sc5 -> plain"] = qp["fp16"]
report["pairs"]["int8: sc5 -> plain"] = qp["int8"]
eng = report["pairs"]["sc5: fp16 -> int8 prefill"]
for panel in sorted(eng):
    gap = qp["fp16"][panel]["dnll"]          # NLL(plain) - NLL(SC5), fp16 prefill
    r = eng[panel]["dnll"] / gap if gap > 0 else None
    kl_ratio = eng[panel]["kl_lb"] / RL179["sc5"][panel] if panel in RL179["sc5"] else None
    material = (r is not None and r >= 0.25) or (kl_ratio is not None and kl_ratio >= 0.25)
    negligible = (r is None or r < 0.10) and (kl_ratio is None or kl_ratio < 0.10)
    report["decision"][panel] = {
        "sc5_vs_plain_nll_gap_fp16": gap, "int8_dnll_sc5": eng[panel]["dnll"],
        "R_int8_share_of_quant_gap": r, "int8_kl_lb_over_rl179_sc5_kl": kl_ratio,
        "verdict": "material" if material else ("negligible" if negligible else "minor")}
OUT.write_text(json.dumps(report, indent=1))
for name, s in report["pairs"].items():
    for panel, v in s.items():
        print(f"{name:32s} {panel} n={v['n']:6d} KLlb={v['kl_lb']:.5f} [{v['kl_lb_ci'][0]:.5f},{v['kl_lb_ci'][1]:.5f}] "
              f"tail={v['tail_p']:.4f} top1={v['top1']:.4f} dNLL={v['dnll']:+.5f} [{v['dnll_ci'][0]:+.5f},{v['dnll_ci'][1]:+.5f}]")
print(json.dumps(report["decision"], indent=1))
