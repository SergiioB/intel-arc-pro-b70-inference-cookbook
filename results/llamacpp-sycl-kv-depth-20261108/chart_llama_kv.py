#!/usr/bin/env python3
"""tg128 decode at depth, Oct-15 vs Nov-8 llama.cpp SYCL, per KV type (Qwen3.8-27B Q4_K_M, one B70)."""
import json, statistics as st, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

src = Path(sys.argv[1]); out = Path(sys.argv[2])
rows = json.loads((src / 'summary.json').read_text())
val = {}
for r in rows:
    if r['test'] != 'tg':
        continue
    for b in ('oct15', 'nov08'):
        s = r[b]
        # f16 depth 0 on nov08 has one host-event outlier rep (10.42 vs 27.84): use the max of 2 reps everywhere
        val[(r['kv'], r['depth'], b)] = max(s)
KV = [('q5_0-q4_1', 'q5_0 / q4_1'), ('f16-f16', 'f16 / f16'), ('q8_0-q8_0', 'q8_0 / q8_0'), ('q4_0-q4_0', 'q4_0 / q4_0')]
D = [0, 16384, 65536]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), dpi=150, sharey=True)
fig.patch.set_facecolor('#fcfcfb')
for ax, d in zip(axes, D):
    ax.set_facecolor('#fcfcfb')
    y = range(len(KV))
    o = [val[(k, d, 'oct15')] for k, _ in KV]; n = [val[(k, d, 'nov08')] for k, _ in KV]
    ax.barh([i + 0.2 for i in y], o, height=0.36, color='#86b6ef', label='llama.cpp Oct 15 (v0.7.0)')
    ax.barh([i - 0.2 for i in y], n, height=0.36, color='#2a78d6', label='llama.cpp Nov 8 (b12195)')
    for i in y:
        ax.text(n[i] + 0.4, i - 0.2, f'{n[i]:.1f}', va='center', fontsize=9, color='#1a1a19')
        ax.text(o[i] + 0.4, i + 0.2, f'{o[i]:.1f}', va='center', fontsize=9, color='#6b6a63')
    ax.set_yticks(list(y), [lab for _, lab in KV])
    ax.set_title(f'{d // 1024}K tokens in context' if d else 'empty context', fontsize=10.5, color='#1a1a19')
    ax.set_xlim(0, 33); ax.grid(axis='x', color='#e4e3dd', lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c3c2b7')
    ax.set_xlabel('decode tok/s (tg128)', color='#3b3b38')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc='upper left', bbox_to_anchor=(0.01, 0.93), ncol=2, fontsize=9)
fig.suptitle('Qwen3.8-27B Q4_K_M on one Arc Pro B70: update llama.cpp and use q8_0 or q4_0 KV (2.1x at 64K)',
             x=0.01, ha='left', fontsize=11.5, color='#1a1a19')
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig(str(out) + '.svg'); fig.savefig(str(out) + '.png')
print(out)
