#!/usr/bin/env python3
"""Request prompt logprobs for every panel row from one running exl3xpu server, C1, sequentially.

Writes one gzip JSONL per (arm, rep): per row, for every scored position t the observed token,
its logprob, and the server's top-k (token id, logprob). Verifies usage.prompt_tokens equals the
panel length and records the prefix-cache counters before and after (required delta: zero hits).
Usage: client_plogprobs.py --base URL --arm NAME --rep N --panels panels.json --out DIR [--only P1]
"""
import argparse, gzip, json, re, time, urllib.request
from pathlib import Path


def post(url, body, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def cache_counters(base):
    txt = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    out = {}
    for name in ("prefix_cache_queries", "prefix_cache_hits"):
        vals = [float(m.group(1)) for m in re.finditer(rf"^vllm:{name}(?:_total)?(?:{{[^}}]*}})?\s+([0-9.eE+-]+)$", txt, re.M)]
        out[name] = sum(vals) if vals else None
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--model", default="qwen3.8-27b-exl3")
ap.add_argument("--arm", required=True)
ap.add_argument("--rep", type=int, required=True)
ap.add_argument("--panels", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--only", default="")
a = ap.parse_args()

panels = json.loads(Path(a.panels).read_text())["panels"]
if a.only:
    panels = [p for p in panels if p["panel"] in a.only.split(",")]
out = Path(a.out) / f"{a.arm}_rep{a.rep}.jsonl.gz"
summary = {"arm": a.arm, "rep": a.rep, "cache_before": cache_counters(a.base), "rows": []}
with gzip.open(out, "wt") as f:
    for p in panels:
        t0 = time.monotonic()
        r = post(a.base + "/v1/completions", {"model": a.model, "prompt": p["ids"], "max_tokens": 1,
                                              "temperature": 0, "prompt_logprobs": p["k"]})
        dt = time.monotonic() - t0
        n_prompt = r["usage"]["prompt_tokens"]
        assert n_prompt == len(p["ids"]), (p["row"], n_prompt, len(p["ids"]))
        plp = r["choices"][0]["prompt_logprobs"]
        assert len(plp) == len(p["ids"]) and plp[0] is None, (p["row"], len(plp))
        pos = []
        for t in range(max(1, p["score_from"]), len(p["ids"])):
            d = plp[t]
            obs = p["ids"][t]
            top = sorted(((int(k), v["logprob"]) for k, v in d.items()), key=lambda x: -x[1])
            obs_lp = d[str(obs)]["logprob"]
            # top holds the server's top-k plus the observed token when it fell outside the top-k
            pos.append([t, obs, obs_lp, top])
        f.write(json.dumps({"panel": p["panel"], "row": p["row"], "n_prompt": n_prompt, "k": p["k"], "pos": pos}) + "\n")
        summary["rows"].append({"row": p["row"], "n_prompt": n_prompt, "scored": len(pos), "seconds": round(dt, 2)})
        print(json.dumps(summary["rows"][-1]), flush=True)
summary["cache_after"] = cache_counters(a.base)
hb, ha = summary["cache_before"]["prefix_cache_hits"], summary["cache_after"]["prefix_cache_hits"]
summary["cache_hit_delta"] = None if hb is None or ha is None else ha - hb
(Path(a.out) / f"{a.arm}_rep{a.rep}.summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({"out": str(out), "cache_hit_delta": summary["cache_hit_delta"]}))
