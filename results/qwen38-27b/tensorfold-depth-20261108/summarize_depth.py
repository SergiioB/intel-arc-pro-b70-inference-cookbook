#!/usr/bin/env python3
"""Summarize depth_bench cells: medians per (arm, depth, kind) and output identity between arms.

  summarize_depth.py RUN_DIR [RUN_DIR ...] [--md OUT.md] [--json OUT.json]
"""
import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path

args = [a for a in sys.argv[1:] if not a.startswith('--')]
opt = {sys.argv[i]: sys.argv[i + 1] for i in range(1, len(sys.argv) - 1) if sys.argv[i].startswith('--')}
dirs = [a for a in args if a not in opt.values()]
cells = []
for d in dirs:
    for l in Path(d, 'cells.jsonl').read_text().splitlines():
        r = json.loads(l); r['run'] = Path(d).name; cells.append(r)
ORDER = {'0': 0, '8k': 1, '32k': 2, '128k': 3}
g = defaultdict(list)
for r in cells:
    g[(r['arm'], r['depth'], r['kind'])].append(r)
med = lambda xs: round(st.median(xs), 2) if xs else None
rows = []
for (arm, d, k), rs in sorted(g.items(), key=lambda x: (x[0][0], ORDER[x[0][1]], x[0][2])):
    rows.append({'arm': arm, 'depth': d, 'kind': k, 'n': len(rs), 'prompt_tokens': rs[0]['prompt_tokens'],
                 'decode_tok_s': med([r['decode_tok_s'] for r in rs if r['decode_tok_s']]),
                 'decode_min': min(r['decode_tok_s'] for r in rs), 'decode_max': max(r['decode_tok_s'] for r in rs),
                 'ttft_s': med([r['ttft_s'] for r in rs]),
                 'prefill_tok_s': med([r['prompt_tokens'] / r['ttft_s'] for r in rs if r['ttft_s'] and r['prompt_tokens'] and r['prompt_tokens'] > 1000]),
                 'decode_w': med([r.get('decode_avg_w') for r in rs if r.get('decode_avg_w')]),
                 'decode_j_per_tok': med([r.get('decode_j_per_tok') for r in rs if r.get('decode_j_per_tok')]),
                 'temp_end_c': max(r['temp_end_c'] or 0 for r in rs),
                 'reps_identical': len({r['sha256'] for r in rs}) == 1})
# identity between arms: compare each arm's rep-0 sha per cell
by = defaultdict(dict)
for r in cells:
    by[(r['depth'], r['kind'])].setdefault(r['arm'], set()).add(r['sha256'])
arms = sorted({r['arm'] for r in cells})
ident = []
for (d, k), m in sorted(by.items(), key=lambda x: (ORDER[x[0][0]], x[0][1])):
    ident.append({'depth': d, 'kind': k, **{a: (sorted(m[a])[0][:10] + ('+' if len(m[a]) > 1 else '')) if a in m else None for a in arms},
                  'all_equal': len(set().union(*m.values())) == 1})
out = {'rows': rows, 'identity': ident}
if '--json' in opt:
    Path(opt['--json']).write_text(json.dumps(out, indent=1))
L = ['| arm | depth | kind | prompt tok | decode tok/s (median, min–max) | TTFT s | prefill tok/s | decode W | J/token | reps identical |',
     '|---|---|---|---|---|---|---|---|---|---|']
for r in rows:
    L.append(f"| {r['arm']} | {r['depth']} | {r['kind']} | {r['prompt_tokens']} | {r['decode_tok_s']} ({r['decode_min']}–{r['decode_max']}) | {r['ttft_s']} | {r['prefill_tok_s'] or '–'} | {r['decode_w']} | {r['decode_j_per_tok']} | {r['reps_identical']} |")
L += ['', '| depth | kind | ' + ' | '.join(arms) + ' | all equal |', '|---|---|' + '---|' * (len(arms) + 1)]
for i in ident:
    L.append(f"| {i['depth']} | {i['kind']} | " + ' | '.join(str(i[a]) for a in arms) + f" | {i['all_equal']} |")
md = '\n'.join(L) + '\n'
if '--md' in opt:
    Path(opt['--md']).write_text(md)
print(md)
