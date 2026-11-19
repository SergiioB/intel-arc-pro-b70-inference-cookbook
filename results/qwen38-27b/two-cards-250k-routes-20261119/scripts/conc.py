#!/usr/bin/env python3
"""Concurrency screen: C streams at once against one OpenAI-compatible server, short prompts, greedy, 512 tokens.

Usage: conc.py --base URL --model NAME --out DIR --arm NAME --hwmon HW [--levels 1,4,8] [--extra-json JSON]

Each stream sends one request. Stream i gets prompt code-0 (even i) or chat-0 (odd i) from the frozen prompt set with
a unique first line ("Request <i>-<C>.") so no two streams share a prefix. Records per stream: TTFT, decode tok/s after
the first token, completion tokens; per level: aggregate tok/s = sum completion tokens / (last end - first start),
and the card's energy over the level. Stdlib only.
"""
import argparse, hashlib, json, threading, time, urllib.request
from pathlib import Path

PROMPTS = '~/B70-DOCS/results/2026/11/b70-tonight-20261108/prompts-qwen38.json'


def energy(hw):
    try:
        return int(Path(hw, 'energy1_input').read_text())
    except Exception:
        return None


def one(a, content, extra, res, i):
    body = {'model': a.model, 'messages': [{'role': 'user', 'content': content}], 'max_tokens': 512, 'temperature': 0,
            'top_p': 1, 'seed': 4242, 'stream': True, 'stream_options': {'include_usage': True},
            'chat_template_kwargs': {'enable_thinking': False}}
    body.update(extra)
    req = urllib.request.Request(a.base + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t0 = time.monotonic(); tf = tl = None; usage = None; parts = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith('data:'):
                continue
            d = line[5:].strip()
            if d == '[DONE]':
                break
            o = json.loads(d)
            if o.get('usage'):
                usage = o['usage']
            for ch in o.get('choices') or []:
                piece = (ch.get('delta') or {}).get('content') or ''
                if piece:
                    now = time.monotonic(); tf = tf or now; tl = now; parts.append(piece)
    t1 = time.monotonic()
    n = (usage or {}).get('completion_tokens') or len(parts)
    text = ''.join(parts)
    res[i] = {'stream': i, 't0': t0, 't1': t1, 'ttft_s': round(tf - t0, 3) if tf else None, 'completion_tokens': n,
              'decode_tok_s': round((n - 1) / (tl - tf), 2) if tf and tl and tl > tf else None,
              'sha256': hashlib.sha256(text.encode()).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    for k in ('base', 'model', 'out', 'arm', 'hwmon'):
        ap.add_argument('--' + k, required=True)
    ap.add_argument('--levels', default='1,4,8'); ap.add_argument('--extra-json', default='')
    a = ap.parse_args()
    extra = json.loads(a.extra_json) if a.extra_json else {}
    P = json.loads(Path(PROMPTS).read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    one(a, 'Say ok.', extra, {}, 0)  # warm-up, discarded
    for c in [int(x) for x in a.levels.split(',')]:
        res = {}
        th = [threading.Thread(target=one, args=(a, f'Request {i}-{c}.\n' + P['code-0' if i % 2 == 0 else 'chat-0']['content'], extra, res, i))
              for i in range(c)]
        e0 = energy(a.hwmon); t0 = time.monotonic()
        for t in th:
            t.start()
        for t in th:
            t.join()
        t1 = time.monotonic(); e1 = energy(a.hwmon)
        rows = [res[i] for i in sorted(res)]
        start = min(r['t0'] for r in rows); end = max(r['t1'] for r in rows)
        toks = sum(r['completion_tokens'] for r in rows)
        dec = sorted(r['decode_tok_s'] for r in rows if r['decode_tok_s'])
        lvl = {'arm': a.arm, 'concurrency': c, 'streams_ok': len(rows), 'completion_tokens': toks,
               'wall_s': round(end - start, 2), 'aggregate_tok_s': round(toks / (end - start), 1),
               'per_stream_decode_min_med_max': [dec[0], dec[len(dec) // 2], dec[-1]] if dec else None,
               'ttft_max_s': max(r['ttft_s'] or 0 for r in rows),
               'energy_j': round((e1 - e0) / 1e6, 1) if e0 is not None and e1 is not None else None,
               'j_per_tok': round((e1 - e0) / 1e6 / toks, 3) if e0 is not None and e1 is not None and toks else None,
               'utc': time.strftime('%FT%TZ', time.gmtime()), 'streams': rows}
        with (out / 'conc.jsonl').open('a') as f:
            f.write(json.dumps(lvl) + '\n')
        print(json.dumps({k: v for k, v in lvl.items() if k != 'streams'}), flush=True)
        time.sleep(20)


if __name__ == '__main__':
    main()
