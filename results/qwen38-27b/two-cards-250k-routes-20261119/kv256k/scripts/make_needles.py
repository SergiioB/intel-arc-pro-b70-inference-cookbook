#!/usr/bin/env python3
"""Build retrieval prompts: 5 labelled 8-digit codes planted at 5/25/50/75/95 % of a code corpus.

  make_needles.py OUT.json   (reads the 256K code prompt from prompts-deep.json and the model tokenizer)

Keys: ret-32k, ret-120k, ret-250k (depth_bench.py format) plus 'facts' (label -> code) and
'positions' (token index of each planted line per key). Grader: grade_needles.py.
"""
import hashlib, json, random, sys
from pathlib import Path
from tokenizers import Tokenizer

TF = '$TF'
DEEP = '~/B70-DOCS/results/2026/11/tf-prefix-attn-20261113/prompts-deep.json'
TASK_CODE = ("Above is part of a C/C++ inference engine's source tree.")
LABELS = ['amber', 'cobalt', 'juniper', 'obsidian', 'saffron']
FRACS = [0.05, 0.25, 0.50, 0.75, 0.95]
SIZES = {'32k': 32000, '120k': 120000, '250k': 249500}

rng = random.Random(20261118)
codes = {}
while len(codes) < 5:
    c = str(rng.randrange(10_000_000, 100_000_000))
    if c not in codes.values():
        codes[LABELS[len(codes)]] = c

tok = Tokenizer.from_file(f'{TF}/models/Qwen3.8-27B-MLX-4bit/tokenizer.json')
deep = json.loads(Path(DEEP).read_text())['code-256k']['content']
body = deep[: deep.rindex('\n\n' + TASK_CODE)]
enc = tok.encode(body)
question = ("\n\nThe document above contains five lines of the form `VAULT NOTE: the access code of the <name> "
            "vault is <8 digits>.` for the vaults amber, cobalt, juniper, obsidian and saffron. Answer with "
            "exactly five lines, one per vault, in the form `name: code`, using the codes written in the document. "
            "Nothing else.")
out = {'facts': codes, 'positions': {}}
for key, n in SIZES.items():
    end_char = enc.offsets[n - 1][1]
    text = body[:end_char]
    pieces, last, pos = [], 0, []
    for frac, label in zip(FRACS, LABELS):
        c = enc.offsets[int(n * frac)][0]
        c = text.find('\n', c)  # plant on a line boundary
        pieces.append(text[last:c])
        pieces.append(f"\nVAULT NOTE: the access code of the {label} vault is {codes[label]}.")
        last = c
        pos.append(len(tok.encode(''.join(pieces)).ids))
    pieces.append(text[last:])
    content = ''.join(pieces) + question
    ntok = len(tok.encode(content).ids)
    out[f'ret-{key}'] = {'kind': 'ret', 'depth': key, 'content': content, 'approx_tokens': ntok,
                         'sha256': hashlib.sha256(content.encode()).hexdigest()}
    out['positions'][key] = [round(p / ntok, 3) for p in pos]
    print(key, ntok, out['positions'][key], file=sys.stderr)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print(codes, file=sys.stderr)
