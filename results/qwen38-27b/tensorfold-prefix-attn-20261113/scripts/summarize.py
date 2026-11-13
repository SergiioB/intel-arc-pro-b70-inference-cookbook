#!/usr/bin/env python3
"""Summarize the run folders next to scripts/: one row per (run, arm, depth, kind).

  python3 scripts/summarize.py > summary.md     (run from the evidence folder)
"""
import json, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORDER = {'0': 0, '8k': 1, '32k': 2, '128k': 3, '256k': 4}
print('| run | arm | depth | kind | prompt tok | n | decode tok/s median (min–max) | TTFT s | decode W | J/token | texts in cell |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
sha = defaultdict(dict)
for run in sorted(p.parent.name for p in ROOT.glob('*/cells.jsonl')):
    g = defaultdict(list)
    for line in (ROOT / run / 'cells.jsonl').read_text().splitlines():
        r = json.loads(line)
        g[(r['arm'], r['depth'], r['kind'])].append(r)
    for (arm, d, k), rs in sorted(g.items(), key=lambda x: (ORDER[x[0][1]], x[0][2], x[0][0])):
        v = [r['decode_tok_s'] for r in rs]
        shas = sorted({r['sha256'][:10] for r in rs})
        sha[(run, d, k)][arm] = shas
        w = [r['decode_avg_w'] for r in rs if r.get('decode_avg_w')]
        j = [r['decode_j_per_tok'] for r in rs if r.get('decode_j_per_tok')]
        print(f"| {run} | {arm} | {d} | {k} | {rs[0]['prompt_tokens']} | {len(rs)} | {st.median(v):.2f} ({min(v):.2f}–{max(v):.2f}) | "
              f"{st.median(r['ttft_s'] for r in rs):.2f} | {st.median(w):.1f} | {st.median(j):.2f} | {' '.join(shas)} |" if w and j else
              f"| {run} | {arm} | {d} | {k} | {rs[0]['prompt_tokens']} | {len(rs)} | {st.median(v):.2f} ({min(v):.2f}–{max(v):.2f}) | "
              f"{st.median(r['ttft_s'] for r in rs):.2f} | – | – | {' '.join(shas)} |")
print('\nDrafted against `"draft": false` inside each run (first 10 hex digits of the answer sha256):\n')
print('| run | depth | kind | drafted | serial | drafted2 | equal |')
print('|---|---|---|---|---|---|---|')
for (run, d, k), m in sorted(sha.items(), key=lambda x: (x[0][0], ORDER[x[0][1]], x[0][2])):
    allv = {s for a in m.values() for s in a}
    f = lambda a: ' '.join(m[a]) if a in m else '–'
    print(f"| {run} | {d} | {k} | {f('drafted')} | {f('serial')} | {f('drafted2')} | {'yes' if len(allv) == 1 and len(m) > 1 else ('one arm' if len(m) == 1 else 'NO')} |")
