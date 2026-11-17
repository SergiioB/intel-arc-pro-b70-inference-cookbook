#!/usr/bin/env python3
"""329 frozen RQ-35 items against one TensorFold server, C=1. Usage: battery.py BASE_URL OUT_DIR ARMS(drafted,serial,...)"""
import hashlib, json, sys, time, urllib.request
from pathlib import Path
BASE, OUT, ARMS = sys.argv[1], Path(sys.argv[2]), sys.argv[3].split(',')
FX = json.loads(Path('$LAB/results/2026/11/qwen38-wsp-20261104/fixtures.json').read_text())
ITEMS = [(c, 1024) for c in FX['code']] + [(r, 1536) for r in FX['regression']]
import os
ITEMS = ITEMS[int(os.environ.get('BATTERY_FROM', 0)):int(os.environ.get('BATTERY_TO', 10**9))]  # an item range
def ask(item, cap, draft):
    body = {'model': 'qwen', 'messages': [{'role': 'user', 'content': item['prompt']}], 'max_tokens': cap,
            'temperature': 0, 'top_p': 1, 'seed': 4242, 'stream': False, 'chat_template_kwargs': {'enable_thinking': False}}
    if not draft: body['draft'] = False
    t = time.monotonic()
    req = urllib.request.Request(BASE + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    resp = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    text = resp['choices'][0]['message'].get('content') or ''
    return {'id': item['item_id'], 'split': item.get('split', item.get('area')), 'draft': draft,
            'wall_s': round(time.monotonic() - t, 3), 'finish': resp['choices'][0]['finish_reason'], 'usage': resp.get('usage'),
            'sha256': hashlib.sha256(text.encode()).hexdigest(), 'output': text}
for a in ARMS:
    t = time.monotonic(); toks = 0
    with (OUT / 'generations.jsonl').open('a') as f:
        for it, cap in ITEMS:
            r = ask(it, cap, a != 'serial'); toks += (r['usage'] or {}).get('completion_tokens', 0)
            f.write(json.dumps({'tag': a, **r}) + '\n'); f.flush()
    print(json.dumps({'arm': a, 'n': len(ITEMS), 'wall_s': round(time.monotonic() - t, 1), 'completion_tokens': toks}), flush=True)
