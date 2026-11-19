#!/usr/bin/env python3
"""Grade needle answers: grade_needles.py NEEDLES.json RUN_DIR... -> one line per (run, depth) with hits/5.

A hit = the answer has a line naming the vault label whose 8-digit number equals the planted code.
"""
import json, re, sys
from pathlib import Path

facts = json.loads(Path(sys.argv[1]).read_text())['facts']
for run in sys.argv[2:]:
    for line in Path(run, 'outputs.jsonl').read_text().splitlines():
        o = json.loads(line)
        if o['kind'] != 'ret':
            continue
        got = {}
        for ln in o['text'].splitlines():
            for lab in facts:
                if lab in ln.lower():
                    m = re.search(r'\d{8}', ln)
                    got[lab] = m.group(0) if m else None
        hits = [lab for lab in facts if got.get(lab) == facts[lab]]
        miss = {lab: got.get(lab) for lab in facts if lab not in hits}
        print(json.dumps({'run': run, 'depth': o['depth'], 'hits': len(hits), 'of': 5, 'missed': miss,
                          'sha256': o['sha256'][:12]}))
