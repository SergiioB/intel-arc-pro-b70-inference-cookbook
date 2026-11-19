#!/usr/bin/env python3
"""Print every table of the 2026-11-19 pages from the raw files in this folder.

Usage: python3 summarize.py  (run in this folder; stdlib only)
"""
import collections
import glob
import json
import os
import re
import statistics as st


def jl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def h(title):
    print(f"\n## {title}\n")


def route_depth():
    h("Same session, one card: decode by depth (routecmp)")
    arms = {
        "TF drafted": ("routecmp/tf1-depth/cells.jsonl", "drafted"),
        "TF serial": ("routecmp/tf1-depth/cells.jsonl", "serial"),
        "exl3 MTP3": ("routecmp/exl3-mtp3/cells.jsonl", None),
        "exl3 MTP off": ("routecmp/exl3-serial/cells.jsonl", None),
    }
    print("| arm | depth | kind | prompt tok | n | decode median (min-max) | TTFT median s | J/token median | texts |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, (path, arm) in arms.items():
        rows = [r for r in jl(path) if arm is None or r["arm"].startswith(arm)]
        groups = collections.defaultdict(list)
        for r in rows:
            groups[(r["depth"], r["kind"])].append(r)
        order = {"0": 0, "8k": 1, "32k": 2, "128k": 3}
        for k in sorted(groups, key=lambda k: (order.get(k[0], 9), k[1])):
            g = groups[k]
            d = [r["decode_tok_s"] for r in g]
            print(f"| {name} | {k[0]} | {k[1]} | {g[0]['prompt_tokens']} | {len(g)} | {st.median(d):.2f} ({min(d):.2f}-{max(d):.2f}) "
                  f"| {st.median(r['ttft_s'] for r in g):.2f} | {st.median(r['decode_j_per_tok'] for r in g):.2f} | {len({r['sha256'] for r in g})} |")
    # exactness
    tf = jl("routecmp/tf1-depth/cells.jsonl")
    dr = {(r["depth"], r["kind"]): r["sha256"] for r in tf if r["arm"].startswith("drafted")}
    se = {(r["depth"], r["kind"]): r["sha256"] for r in tf if r["arm"].startswith("serial")}
    print(f"\nTF drafted == serial: {sum(dr[k] == se.get(k) for k in dr)} of {len(dr)}")
    m = jl("routecmp/exl3-mtp3/cells.jsonl")
    s = jl("routecmp/exl3-serial/cells.jsonl")
    sset = collections.defaultdict(set)
    for r in s:
        sset[(r["depth"], r["kind"])].add(r["sha256"])
    print(f"exl3 MTP3 texts equal to a serial text: {sum(r['sha256'] in sset[(r['depth'], r['kind'])] for r in m)} of {len(m)}")
    mset = collections.defaultdict(set)
    for r in m:
        mset[(r["depth"], r["kind"])].add(r["sha256"])
    print(f"exl3 MTP3 prompts with 2 texts over 2 repeats: {sum(len(v) > 1 for v in mset.values())} of {len(mset)}")
    reps3 = {k: v for k, v in sset.items() if sum(1 for r in s if (r['depth'], r['kind']) == k) >= 3}
    print(f"exl3 MTP off prompts with one text over 3 repeats: {sum(len(v) == 1 for v in reps3.values())} of {len(reps3)}"
          f" (prompts with 3 repeats; {len(sset)} prompts in all)")


def route_250k():
    h("250K on one card: exl3xpu (routecmp/exl3-250k) and TensorFold (kv256k)")
    print("| arm | prompt tok | decode tok/s | TTFT s | J/token | sha256 |")
    print("|---|---|---|---|---|---|")
    for r in jl("routecmp/exl3-250k/cells.jsonl"):
        print(f"| {r['arm']} | {r['prompt_tokens']} | {r['decode_tok_s']} | {r['ttft_s']} | {r['decode_j_per_tok']} | {r['sha256'][:12]} |")
    for run in ("a-tp1-int8-262k", "b-tp2-int8-262k", "c-tp2-bf16-262k"):
        for r in jl(f"kv256k/{run}/cells.jsonl"):
            print(f"| {run} {r['arm']} {r['kind']} {r['depth']} | {r['prompt_tokens']} | {r['decode_tok_s']} | {r['ttft_s']} | {r['decode_j_per_tok']} | {r['sha256'][:12]} |")
    for line in open("routecmp/exl3-250k/startup.jsonl"):
        print("exl3 startup", line.strip())


def conc():
    h("Several streams, short prompts, 512 tokens a stream (aggregate tok/s)")
    print("| arm | streams | aggregate tok/s | per-stream min / median / max | max TTFT s | J/token |")
    print("|---|---|---|---|---|---|")
    for path in ("routecmp/tf1-conc/conc.jsonl", "routecmp/exl3-mtp3/conc.jsonl", "routecmp/tf-conc/tf1/conc.jsonl", "routecmp/tf-conc/tf2/conc.jsonl"):
        for r in jl(path):
            mn, md, mx = r["per_stream_decode_min_med_max"]
            print(f"| {r['arm']} | {r['concurrency']} | {r['aggregate_tok_s']} | {mn} / {md} / {mx} | {r['ttft_max_s']} | {r['j_per_tok']} |")


def battery():
    h("329-item battery (passes by split, then total)")
    for name, path in (("exl3xpu MTP3, one card", "routecmp/exl3-mtp3-battery/grade/graded.json"),
                       ("TensorFold two cards, run 2", "gates/battery-tp2-run2/grade/graded.json"),
                       ("TensorFold one card, prefill8 flag on", "prefill8/battery-on/graded.json"),
                       ("TensorFold one card, prefill8 flag off", "prefill8/battery-off/graded.json")):
        for arm, items in json.load(open(path)).items():
            by = collections.Counter(v["split"] for v in items.values() if v["passed"])
            print(f"{name} ({arm}): {dict(sorted(by.items()))} total {sum(by.values())} of {len(items)}")
    rows = [line.rstrip("\n").split("\t") for line in open("gates/battery-tp2-identity.tsv")][1:]
    print(f"TF two cards run 1 vs run 2: same text on {sum(r[3] == 'True' for r in rows)} of {len(rows)} items")


def long_context():
    h("250K: needles, refusals, warm sessions (kv256k)")
    for line in open("kv256k/needle-grade.jsonl"):
        r = json.loads(line)
        print(f"needles {r['run']} {r['depth']}: {r['hits']} of {r['of']}")
    for run in sorted(glob.glob("kv256k/d*")) + sorted(glob.glob("kv256k/w*")):
        log = open(f"{run}/server.log").read()
        refused = re.search(r"cannot hold a stream of \d+ tokens and a [\d.]+ GiB prompt cache", log)
        cmd = open(f"{run}/env.txt").read()
        ctx = re.search(r"--context (\d+)", cmd)
        pc = re.search(r"--prompt-cache-gib ([\d.]+)", cmd)
        tp = re.search(r"--tp (\d)", cmd)
        kv = "int8" if "XPU_KV_INT8=1" in cmd else "bf16"
        print(f"{run}: tp {tp.group(1) if tp else '?'} {kv} context {ctx.group(1) if ctx else '?'} cache {pc.group(1) if pc else '?'} GiB -> "
              + (f"REFUSED: {refused.group(0)}" if refused else "served"))
        if os.path.exists(f"{run}/session/session.jsonl"):
            for r in jl(f"{run}/session/session.jsonl"):
                print(f"   turn {r['turn']}: prompt {r['prompt_tokens']} ttft {r['ttft_s']} decode {r['decode_tok_s']} tokens {r['completion_tokens']}")
        errs = re.findall(r"ended reason=error \((\w+)\)[^\n]*", log)
        if errs:
            print("   errors:", collections.Counter(errs))


def gates():
    h("Two cards, warm resume and gates (gates/)")
    for name in ("resume-32k", "resume-32k-fresh", "resume-128k"):
        rows = [json.loads(line) for line in open(f"gates/{name}.console") if line.startswith("{")]
        for r in rows:
            print(f"{name}: turn {r['turn']} prompt {r['prompt_tokens']} cached {r['cached_tokens']} ttft {r['ttft_s']} decode {r['decode_tok_s']} equal {r.get('equal', '-')}")
    for path in ("gates/conc/tp2-par4/conc_summary.jsonl", "gates/conc/two-servers/conc_summary.jsonl"):
        for r in jl(path):
            print(f"{r['tag']} rep {r['rep']}: aggregate {r['aggregate_tok_s_wall']} per-stream {r['per_stream_tok_s']} equal_alone {r['equal_alone']}")
    for r in jl("gates/conc/sampled-thinking/cells.jsonl"):
        print(f"{r['tag']} rep {r['rep']}: ttft {r['ttft_s']} decode {r['decode_tok_s']} reasoning_chars {r['reasoning_chars']} sha {r['sha256'][:12]}")


def tptf():
    h("Fork fixes (tptf/raw)")
    for run in ("b-prefix", "b-fixed", "b-slices", "b-final", "b-head"):
        r = jl(f"tptf/raw/{run}/check.jsonl")[0]
        print(f"bug B {run}: requests {r['requests']}, mismatches {len(r.get('client_mismatches', []))}, pass {r.get('pass')}")
    for run in ("a-fix-160k", "a-fix-250k", "a-head-160k", "3586-fix-160k"):
        rows = jl(f"tptf/raw/{run}/check.jsonl")
        print(f"bug A {run}: {sum(r['pass'] for r in rows)} of {len(rows)} turns resumed == cache-off, cached {[r['cached_tokens'] for r in rows]}")
    c = jl("tptf/raw/a-fix-250k/cached/replay.jsonl")
    f = jl("tptf/raw/a-fix-250k/fresh/session-turns.jsonl")
    for a, b in zip(c, f):
        print(f"  250K turn {a['turn']}: cache on ttft {a['ttft_s']} s, cache off ttft {b['ttft_s']} s, same text {a['sha256'] == b['sha256']}")


def tpdiag():
    h("Why two cards lose time (tpdiag/tf logs, dev runner, 256 steps)")
    runs = collections.defaultdict(list)
    for path in sorted(glob.glob("tpdiag/tf/*.log")):
        name = os.path.basename(path)[:-4]
        text = open(path).read()
        d = re.search(r"drafted: \d+ tokens in \d+ rounds \(([\d.]+) a round\), ([\d.]+) tok/s.*equal to serial: (\w+)", text)
        s = re.search(r"decode: \d+ tokens in [\d.]+ s \(([\d.]+) tok/s\)", text)
        if not d or not s:
            print(f"{name}: no result line (profile run hung or skipped)")
            continue
        m = re.match(r"(ds-)?(0|32k)-\d+-(\w+)-(\d)", name)
        key = (m.group(1) or "", m.group(2), m.group(3), m.group(4))
        runs[key].append((float(d.group(2)), float(d.group(1)), float(s.group(1)), d.group(3)))
    print("| job | prompt | arm | cards | n | drafted tok/s | ms a round | serial tok/s | serial ms/token | drafted == serial |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(runs):
        v = runs[key]
        dt = st.mean(x[0] for x in v)
        tpr = st.mean(x[1] for x in v)
        sv = st.mean(x[2] for x in v)
        print(f"| {'111' if key[0] else '110'} | {key[1]} | {key[2]} | {key[3]} | {len(v)} | {', '.join(str(x[0]) for x in v)} (mean {dt:.1f}) "
              f"| {1000 * tpr / dt:.1f} | {sv:.2f} | {1000 / sv:.1f} | {all(x[3] == 'true' for x in v)} |")
    h("Exchange step cost (tpdiag/chain-summary.tsv, replayed lists, default mode)")
    rows = [line.rstrip("\n").split("\t") for line in open("tpdiag/chain-summary.tsv")]
    head = rows[0]
    for r in rows[1:]:
        d = dict(zip(head, r))
        if d["ds_mode"] == "default" and d["arm"] in ("event_read_devpool_replay", "event_only_replay", "local_replay"):
            print(f"{d['arm']} {d['bytes']} B: step median {d['step_us_med']} us, delta vs local {d['delta_vs_local_med_us']} us, n {d['n']}, errors {d['errors']}")


def sweep():
    h("Decode by prompt length, two cards, bf16 K/V (sweep-tp2-bf16/samples.jsonl; rep -1 = warm-up, excluded)")
    rows = jl("sweep-tp2-bf16/samples.jsonl")
    by = collections.defaultdict(list)
    for r in rows:
        if r["rep"] >= 0:
            by[(r["input_target"], r["arm"])].append(r)
    print("| prompt tokens | drafted mean (min-max) | serial mean (min-max) | TTFT mean s | prompt tok/s mean | drafted == serial | tokens a round (drafted) |")
    print("|---|---|---|---|---|---|---|")
    for t in sorted({k[0] for k in by}):
        d = sorted(by[(t, "drafted")], key=lambda r: r["rep"])
        s = sorted(by[(t, "serial")], key=lambda r: r["rep"])
        assert all(r["status"] == "ok" for r in d + s)
        dv = [r["decode_tps"] for r in d]
        sv = [r["decode_tps"] for r in s]
        eq = sum(a["text_sha256"] == b["text_sha256"] for a, b in zip(d, s))
        tpr = [r["speculative"]["tokens_per_round"] for r in d]
        print(f"| {t:,} | {st.mean(dv):.1f} ({min(dv):.1f}-{max(dv):.1f}) | {st.mean(sv):.1f} ({min(sv):.1f}-{max(sv):.1f}) "
              f"| {st.mean(r['ttft_s'] for r in d + s):.2f} | {st.mean(r['prompt_tps'] for r in d + s):.0f} | {eq} of {len(d)} | {st.mean(tpr):.2f} |")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    route_depth()
    route_250k()
    conc()
    battery()
    long_context()
    gates()
    tptf()
    tpdiag()
    sweep()
