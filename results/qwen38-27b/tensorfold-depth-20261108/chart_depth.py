#!/usr/bin/env python3
"""Decode tok/s vs context depth for Qwen3.8-27B routes on one B70 (code prompts). Writes SVG + PNG."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path('/home/sergio/B70-DOCS/results/2026/11/b70-tonight-20261108')
out = Path(sys.argv[1]) if len(sys.argv) > 1 else R / 'qwen38-27b-depth-20261108'


def rows(p):
    return {(r['arm'], r['depth'], r['kind']): r for r in json.loads(Path(p).read_text())['rows']}


g1 = rows(R / 'summary-gpu1-230w.json')
g0 = rows(R / 'tf-attn-230w-gpu0' / 'summary.json')
D = ['0', '8k', '32k', '128k']
X = [0, 1, 2, 3]
series = [  # label, color, linestyle, values
    ('TensorFold v1.2.4 + DFlash2 (lossless)', '#2a78d6', '-', [g0[('base', d, 'code')]['decode_tok_s'] for d in D]),
    ('TensorFold + our attention patch (lossless)', '#eb6834', '-', [g0[('E', d, 'code')]['decode_tok_s'] for d in D]),
    ('EXL3 4.0bpw + MTP3 (output differs from serial)', '#1baf7a', '--', [g1[('exl3-mtp3', d, 'code')]['decode_tok_s'] for d in D]),
    ('EXL3 4.0bpw, no speculation', '#eda100', ':', [g1[('exl3-serial', d, 'code')]['decode_tok_s'] for d in D]),
]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
fig.patch.set_facecolor('#fcfcfb'); ax.set_facecolor('#fcfcfb')
for lab, c, ls, v in series:
    ax.plot(X, v, color=c, lw=2, ls=ls, marker='o', ms=7, mec='#fcfcfb', mew=2, label=lab, zorder=3)
    ax.annotate(f'{v[-1]:.0f}', (X[-1], v[-1]), xytext=(8, 0), textcoords='offset points', va='center', color='#3b3b38', fontsize=10)
    ax.annotate(f'{v[0]:.0f}', (X[0], v[0]), xytext=(-8, 0), textcoords='offset points', va='center', ha='right', color='#3b3b38', fontsize=10)
ax.set_xticks(X, ['~70', '8K', '32K', '131K'])
ax.set_xlabel('Prompt length (tokens) before a 512-token answer', color='#3b3b38')
ax.set_ylabel('Decode tok/s, one stream', color='#3b3b38')
ax.set_xlim(-0.5, 3.45); ax.set_ylim(0, 140)
ax.grid(axis='y', color='#e4e3dd', lw=0.8); ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
for s in ('left', 'bottom'):
    ax.spines[s].set_color('#c3c2b7')
ax.tick_params(colors='#3b3b38')
ax.set_title('Qwen3.8-27B on one Intel Arc Pro B70 (230 W): short context favours TensorFold, long context EXL3',
             loc='left', fontsize=11.5, color='#1a1a19')
ax.legend(frameon=False, loc='upper right', fontsize=9.5)
fig.text(0.01, 0.01, 'Greedy, thinking off, code prompts, median of 2. TensorFold rows on card 0, EXL3 rows on card 1 (cards differ ~4 % at short context).',
         fontsize=8, color='#6b6a63')
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(str(out) + '.svg'); fig.savefig(str(out) + '.png')
print(out)
