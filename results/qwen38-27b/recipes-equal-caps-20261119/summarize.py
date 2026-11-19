#!/usr/bin/env python3
"""Print every table of the equal-caps recipe update from the raw files in this folder.

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


def caps(env):
    t = open(env).read()
    return re.findall(r"(gpu\d) \S+ cap_uw (\d+) temp_mc \d+ max_freq (\d+)", t)


def rebaseline():
    h("Two cards, served, same binary: 180/230 W (2026-11-18) against 230/230 W (2026-11-19)")
    print("caps before:", caps("tp2-prefix-180-230/tp2/env.txt"), "| after:", caps("rebaseline-230w/tp2/env.txt"))
    old = {(r["arm"], r["depth"], r["kind"]): r for r in jl("tp2-prefix-180-230/tp2/cells.jsonl")}
    new = jl("rebaseline-230w/tp2/cells.jsonl")
    print("| arm | depth | kind | prompt tok | 180/230 W tok/s | 230/230 W tok/s | ratio | TTFT s before -> after | same text |")
    print("|---|---|---|---|---|---|---|---|---|")
    same = 0
    order = {"0": 0, "8k": 1, "32k": 2, "128k": 3}
    for r in sorted(new, key=lambda r: (r["arm"], order[r["depth"]], r["kind"])):
        o = old[(r["arm"], r["depth"], r["kind"])]
        same += r["sha256"] == o["sha256"]
        print(f"| {r['arm']} | {r['depth']} | {r['kind']} | {r['prompt_tokens']} | {o['decode_tok_s']} | {r['decode_tok_s']} | "
              f"{r['decode_tok_s'] / o['decode_tok_s']:.3f} | {o['ttft_s']} -> {r['ttft_s']} | {r['sha256'] == o['sha256']} |")
    print(f"same text {same} of {len(new)}")


def devrunner():
    h("Dev runner, 68 tokens: equal caps (eq230) and runtime 26.05 against 26.40 (n40); tpdiag/prof")
    runs = collections.defaultdict(list)
    for path in sorted(glob.glob("tpdiag/prof/*.log.txt")):
        name = os.path.basename(path)[:-8]
        m = re.match(r"(eq230|n40)-0-\d+-(wall|n40wall)-(\d)$", name)
        if not m:
            continue
        t = open(path).read()
        d = re.search(r"drafted: \d+ tokens in \d+ rounds \(([\d.]+) a round\), ([\d.]+) tok/s.*equal to serial: (\w+)", t)
        s = re.search(r"decode: \d+ tokens in [\d.]+ s \(([\d.]+) tok/s\)", t)
        tel = open(path[:-8] + ".telemetry").read().split()
        cap0 = tel[tel.index("0") + 2] if "0" in tel else "?"
        runs[(m[1], m[2], m[3])].append((float(d[2]), float(d[1]), float(s[1]), d[3], cap0))
    print("| job | arm | cards | n | drafted tok/s | ms a round | serial tok/s | drafted == serial | card 0 cap (uW) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k in sorted(runs):
        v = runs[k]
        dm = st.mean(x[0] for x in v)
        print(f"| {k[0]} | {k[1]} | {k[2]} | {len(v)} | {', '.join(str(x[0]) for x in v)} (mean {dm:.2f}) | "
              f"{1000 * st.mean(x[1] for x in v) / dm:.2f} | {st.mean(x[2] for x in v):.2f} | {all(x[3] == 'true' for x in v)} | {sorted({x[4] for x in v})} |")


def levers():
    h("TensorFold two-card speed levers, 3 interleaved pairs, equal caps (tptf-speed)")
    for lever in ("l1", "l2", "l1b-d16", "l1b-v32", "l1b-w8"):
        s = json.load(open(f"tptf-speed/{lever}/summary.json"))
        flag = open(f"tptf-speed/{lever}/pair1-on/ab.txt").readline().strip()
        print(f"{lever}: {flag}; text changes off -> on {s['text_changes_off_to_on']} of {s['cells_compared']}; "
              + (f"drafted == serial off {all(c['equal'] for c in s['drafted_eq_serial']['off'])}, on {all(c['equal'] for c in s['drafted_eq_serial']['on'])}"
                 if s["drafted_eq_serial"].get("off") else "no serial arm in this screen"))
        print("| depth | kind | off median (raw) | on median (raw) | gain % | same token hash |")
        print("|---|---|---|---|---|---|")
        order = {"0": 0, "8k": 1, "32k": 2, "128k": 3}
        for c in sorted(s["cells"], key=lambda c: (order[c["depth"]], c["kind"])):
            print(f"| {c['depth']} | {c['kind']} | {c['off']['client_tok_s']} {c['off']['raw_tok_s']} | {c['on']['client_tok_s']} {c['on']['raw_tok_s']} | "
                  f"{c['gain_pct']} | {c['off']['hashes'] == c['on']['hashes']} |")
        print("caps:", caps(f"tptf-speed/{lever}/pair1-on/env.txt"))


def llamacpp():
    h("llama.cpp Qwen3.8-27B Q4_K_M, llama-bench (llamacpp-tp/*/bench.json), q8_0 K/V")
    for d in ("base-single", "probem-off", "probem-on2", "probem-diag"):
        env = open(f"llamacpp-tp/{d}/env.txt").read()
        mode = re.search(r"GGML_SYCL_TP_P2P=(\d)", env)
        skip = "GGML_SYCL_TP_DIAG_SKIP=1" in env
        for r in json.load(open(f"llamacpp-tp/{d}/bench.json")):
            print(f"{d}: split {r['split_mode']} p2p mode {mode[1] if mode else 'unset'}{' DIAG_SKIP' if skip else ''} "
                  f"pp{r['n_prompt']} tg{r['n_gen']}: {r['avg_ts']:.2f} ± {r['stddev_ts']:.2f} tok/s, samples {r['samples_ts']}")
    for d in ("base-single", "probem-off", "probem-on2"):
        for r in jl(f"llamacpp-tp/{d}/served.jsonl"):
            if r.get("phase") == "measured":
                print(f"served {d}: depth {r['depth']} prompt {r['prompt_tokens']} decode {r['decode_tok_s']} tok/s ttft {r['ttft_s']} s text {r['sha256'][:12]}")
    h("llama.cpp allreduce microbenchmark (llamacpp-tp/check-m2)")
    for f in sorted(glob.glob("llamacpp-tp/check-m2/*.txt")):
        t = open(f).read()
        lines = re.findall(r"(?:timing|rss) [^\n]*|verified n=\d+[^\n]*|checked \d+ F32 values|zeEventCreate = \d+|zeEventDestroy = \d+[^\n]*", t)
        out = []
        for line in lines:
            m = re.search(r"rss n=(\d+) calls=(\d+) before_kib=(\d+) after_kib=(\d+)", line)
            out.append(line + (f"  -> {(int(m[4]) - int(m[3])) / int(m[2]):.2f} KiB a call" if m else ""))
        print(f"== {os.path.basename(f)}\n" + "\n".join(dict.fromkeys(out)))
    t = open("llamacpp-tp/queue-tpllama-exact-m2b.log.txt").read()
    print("mode 1 exit codes:", re.findall(r"(time p2p1 rc=\d+)", t), "| segfaults:", t.count("Segmentation fault"))
    for f in ("P0-measured-cache-off", "P1-measured-cache-on-fresh-dir"):
        t = open(f"llamacpp-tp/diag/{f}.gdb.txt").read()
        print(f, "->", "SIGSEGV in " + re.search(r"#2 .*? in ([\w:]+)", t)[1] if "SIGSEGV" in t else re.search(r"\[Inferior 1 [^\]]*\]", t)[0])


def intelinside():
    h("intelinside.ai submission (intelinside/response-tf-tp2-20261119.json)")
    objs = json.load(open("intelinside/response-tf-tp2-20261119.json"))
    print("custom runtime:", objs[0]["id"], objs[0]["url"])
    for o in objs[1:]:
        for it in o["items"]:
            print(it["id"], it["url"], it["inputTokens"], round(it["decodeTps"], 2), "draft" if it["runtimeFlags"].endswith("draft=true") else "serial",
                  "caps:", re.search(r"GPU0 [^.]*\. GPU1 [^.]*\.", it["notes"])[0])


def deep():
    h("Several streams, one card (card 1, 230 W), three passes a level (routecmp-deep)")
    groups = collections.defaultdict(list)
    for f in sorted(glob.glob("routecmp-deep/*/conc.jsonl")):
        for r in jl(f):
            groups[(r["arm"], r["concurrency"])].append(r)
    print("| arm | streams | aggregate median (min-max) | slowest TTFT s | texts equal to solo |")
    print("|---|---|---|---|---|")
    for (arm, c), rs in sorted(groups.items()):
        agg = [r["aggregate_tok_s"] for r in rs]
        eq = [tuple(map(int, r["solo_match"].split("/"))) for r in rs]
        print(f"| {arm} | {c} | {st.median(agg):.1f} ({min(agg):.1f}-{max(agg):.1f}) | {max(r['ttft_max_s'] for r in rs):.2f} | "
              f"{sum(a for a, _ in eq)}/{sum(b for _, b in eq)} |")
    for f in sorted(glob.glob("routecmp-deep/*/env.txt")):
        print(f, open(f).read().splitlines()[0])


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    rebaseline()
    devrunner()
    levers()
    llamacpp()
    intelinside()
    deep()
