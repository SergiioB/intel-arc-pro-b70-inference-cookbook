#!/usr/bin/env python3
"""Route-gate clients for the two-card TensorFold recipe (wave 2, agent `gates`). Stdlib only.

  gates.py session --base URL --prompts P.json --key code-32k --turns 4 --max-tokens 256 --out DIR
      One conversation: the seed prompt (512 tokens out), then N follow-ups; every reply appended.
      Writes DIR/session.jsonl with each turn's full messages-in, text, sha256, TTFT, decode tok/s.
  gates.py replay --base URL --session DIR/session.jsonl --out DIR2
      Sends each turn's exact messages-in again (to a server with an empty cache), same max_tokens.
  gates.py conc --base URL --prompts P.json --keys code-0,chat-0,code-8k,chat-8k --out DIR [--extra '{}'] [--tag T]
      Each prompt alone (one after another), then all at once (one thread a prompt).
      With --bases URL1,URL2 the concurrent phase spreads prompts round-robin over servers.
  gates.py cell --base URL --prompts P.json --key code-0 --out DIR --tag T --extra '{"temperature":0.7,"seed":7}'
  gates.py soak --base URL --prompts P.json --keys code-0,chat-0,code-8k,code-32k --minutes 120 --out DIR
      Rotates prompts until the time is up; a sampler logs both cards' temperature and power every 30 s.
"""
import argparse, hashlib, json, subprocess, threading, time, urllib.request
from pathlib import Path

FOLLOWUPS = [
    "In one short paragraph, summarize the document above.",
    "List the five most important files or functions mentioned above, one line each.",
    "Write a small unit test for the code you just wrote.",
    "Now refactor that into a shorter version without changing behaviour.",
]
HWMON = {0: None, 1: None}
PCI = {0: '0000:0b:00.0', 1: '0000:0f:00.0'}


def hwmon(i):
    if HWMON[i] is None:
        HWMON[i] = subprocess.check_output(['python3', '~/B70-DOCS/tools/host_facts.py', 'hwmon', str(i)], text=True).strip()
    return HWMON[i]


def rd(path):
    try:
        return int(Path(path).read_text())
    except Exception:
        return None


def vram_free_mib(i):
    try:
        t = subprocess.check_output(['sudo', '-n', 'cat', f'/sys/kernel/debug/dri/{PCI[i]}/tile0/vram_mm'], text=True)
        for line in t.splitlines():
            if line.startswith('visible_avail:'):
                return int(line.split()[1].rstrip('MiB'))
    except Exception:
        return None


def chat(base, messages, max_tokens, extra=None, timeout=3600):
    body = {'model': 'qwen', 'messages': messages, 'max_tokens': max_tokens, 'temperature': 0, 'top_p': 1,
            'seed': 4242, 'stream': True, 'stream_options': {'include_usage': True},
            'chat_template_kwargs': {'enable_thinking': False}}
    body.update(extra or {})
    req = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t0 = time.monotonic(); t_first = t_last = None
    parts, rparts, usage, nchunks, finish = [], [], None, 0, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                break
            obj = json.loads(data)
            if obj.get('usage'):
                usage = obj['usage']
            for ch in obj.get('choices') or []:
                finish = ch.get('finish_reason') or finish
                d = ch.get('delta') or {}
                c, rc = d.get('content') or '', d.get('reasoning_content') or ''
                if c or rc:
                    now = time.monotonic()
                    t_first = t_first or now
                    t_last = now; nchunks += 1
                    parts.append(c); rparts.append(rc)
    t_end = time.monotonic()
    text, reasoning = ''.join(parts), ''.join(rparts)
    n = (usage or {}).get('completion_tokens') or nchunks
    dec = (t_last - t_first) if t_first and t_last and t_last > t_first else None
    row = {'t_start_rel': t0, 'wall_s': round(t_end - t0, 3), 'ttft_s': round(t_first - t0, 3) if t_first else None,
           'completion_tokens': n, 'prompt_tokens': (usage or {}).get('prompt_tokens'),
           'cached_tokens': ((usage or {}).get('prompt_tokens_details') or {}).get('cached_tokens'),
           'decode_tok_s': round((n - 1) / dec, 2) if dec else None, 'finish': finish,
           'sha256': hashlib.sha256(text.encode()).hexdigest(), 'chars': len(text),
           'reasoning_chars': len(reasoning), 'reasoning_sha256': hashlib.sha256(reasoning.encode()).hexdigest() if reasoning else None}
    return row, text, reasoning


def append(path, obj):
    with open(path, 'a') as f:
        f.write(json.dumps(obj) + '\n')


def warm(base):
    chat(base, [{'role': 'user', 'content': 'Say ok.'}], 8)


def cmd_session(a):
    p = json.loads(Path(a.prompts).read_text())[a.key]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    warm(a.base)
    messages = [{'role': 'user', 'content': p['content']}]
    for i in range(a.turns + 1):
        if i:
            messages.append({'role': 'user', 'content': FOLLOWUPS[(i - 1) % len(FOLLOWUPS)]})
        mt = 512 if i == 0 else a.max_tokens
        row, text, _ = chat(a.base, messages, mt)
        row.update({'turn': i, 'max_tokens': mt, 'key': a.key, 'utc': time.strftime('%FT%TZ', time.gmtime())})
        append(out / 'session.jsonl', {**row, 'messages_in': messages, 'text': text})
        print(json.dumps({k: row[k] for k in ('turn', 'prompt_tokens', 'cached_tokens', 'ttft_s', 'decode_tok_s', 'sha256')}), flush=True)
        messages = messages + [{'role': 'assistant', 'content': text}]


def cmd_replay(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    warm(a.base)
    for line in open(a.session):
        s = json.loads(line)
        row, text, _ = chat(a.base, s['messages_in'], s['max_tokens'])
        row.update({'turn': s['turn'], 'max_tokens': s['max_tokens'], 'key': s['key'], 'utc': time.strftime('%FT%TZ', time.gmtime()),
                    'session_sha256': s['sha256'], 'equal': row['sha256'] == s['sha256']})
        append(out / 'replay.jsonl', {**row, 'text': text})
        print(json.dumps({k: row[k] for k in ('turn', 'prompt_tokens', 'cached_tokens', 'ttft_s', 'decode_tok_s', 'equal')}), flush=True)


def cmd_conc(a):
    P = json.loads(Path(a.prompts).read_text())
    keys = a.keys.split(','); extra = json.loads(a.extra) if a.extra else {}
    bases = a.bases.split(',') if a.bases else [a.base]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for b in bases:
        warm(b)
    alone = {}
    if not a.skip_alone:
        for i, k in enumerate(keys):
            b = bases[i % len(bases)]
            row, text, _ = chat(b, [{'role': 'user', 'content': P[k]['content']}], a.max_tokens, extra)
            alone[k] = row['sha256']
            row.update({'phase': 'alone', 'tag': a.tag, 'key': k, 'base': b})
            append(out / 'conc.jsonl', {**row, 'text': text})
            print(json.dumps({x: row[x] for x in ('phase', 'key', 'ttft_s', 'decode_tok_s', 'sha256')}), flush=True)
    for rep in range(a.reps):
        rows = [None] * len(keys)
        barrier = threading.Barrier(len(keys))

        def go(i, k):
            b = bases[i % len(bases)]
            barrier.wait()
            row, text, _ = chat(b, [{'role': 'user', 'content': P[k]['content']}], a.max_tokens, extra)
            row.update({'phase': 'concurrent', 'rep': rep, 'tag': a.tag, 'key': k, 'base': b,
                        'equal_alone': (row['sha256'] == alone[k]) if k in alone else None})
            rows[i] = (row, text)
        T0 = time.monotonic()
        th = [threading.Thread(target=go, args=(i, k)) for i, k in enumerate(keys)]
        [t.start() for t in th]; [t.join() for t in th]
        span = time.monotonic() - T0
        toks = sum(r[0]['completion_tokens'] for r in rows)
        t_first = min(r[0]['t_start_rel'] + r[0]['ttft_s'] for r in rows)
        t_end = max(r[0]['t_start_rel'] + r[0]['wall_s'] for r in rows)
        for row, text in rows:
            append(out / 'conc.jsonl', {**row, 'text': text})
            print(json.dumps({x: row[x] for x in ('phase', 'key', 'ttft_s', 'decode_tok_s', 'completion_tokens', 'equal_alone')}), flush=True)
        summ = {'tag': a.tag, 'rep': rep, 'streams': len(keys), 'completion_tokens': toks, 'span_s': round(span, 3),
                'aggregate_tok_s_wall': round(toks / span, 2),
                'aggregate_tok_s_decode_window': round(toks / (t_end - t_first), 2),
                'ttft_s': sorted(r[0]['ttft_s'] for r in rows), 'per_stream_tok_s': [r[0]['decode_tok_s'] for r in rows],
                'equal_alone': [r[0]['equal_alone'] for r in rows]}
        append(out / 'conc_summary.jsonl', summ)
        print(json.dumps(summ), flush=True)


def cmd_cell(a):
    P = json.loads(Path(a.prompts).read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    extra = json.loads(a.extra) if a.extra else {}
    warm(a.base)
    for rep in range(a.reps):
        row, text, reasoning = chat(a.base, [{'role': 'user', 'content': P[a.key]['content']}], a.max_tokens, extra)
        row.update({'tag': a.tag, 'rep': rep, 'key': a.key, 'extra': extra, 'utc': time.strftime('%FT%TZ', time.gmtime())})
        append(out / 'cells.jsonl', {**row, 'text': text, 'reasoning': reasoning})
        print(json.dumps({x: row[x] for x in ('tag', 'rep', 'ttft_s', 'decode_tok_s', 'completion_tokens', 'reasoning_chars', 'finish', 'sha256')}), flush=True)


def sampler(out, stop):
    hw = {i: hwmon(i) for i in (0, 1)}
    last = {i: (rd(f'{hw[i]}/energy1_input'), time.monotonic()) for i in (0, 1)}
    while not stop.wait(30):
        rec = {'utc': time.strftime('%FT%TZ', time.gmtime()), 'mono': round(time.monotonic(), 1)}
        for i in (0, 1):
            e, t = rd(f'{hw[i]}/energy1_input'), time.monotonic()
            e0, t0 = last[i]
            rec[f'gpu{i}_temp_c'] = (rd(f'{hw[i]}/temp2_input') or 0) / 1000
            rec[f'gpu{i}_avg_w_30s'] = round((e - e0) / 1e6 / (t - t0), 1) if e is not None and e0 is not None else None
            last[i] = (e, t)
        append(out / 'sensors.jsonl', rec)


def cmd_soak(a):
    P = json.loads(Path(a.prompts).read_text())
    keys = a.keys.split(','); extra_d = {}; extra_s = {'draft': False}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    append(out / 'vram.jsonl', {'when': 'start', 'utc': time.strftime('%FT%TZ', time.gmtime()), 'gpu0_free_mib': vram_free_mib(0), 'gpu1_free_mib': vram_free_mib(1)})
    warm(a.base)
    stop = threading.Event(); th = threading.Thread(target=sampler, args=(out, stop), daemon=True); th.start()
    T0 = time.monotonic(); i = 0; first_sha = {}
    while time.monotonic() - T0 < a.minutes * 60:
        k = keys[i % len(keys)]
        serial = a.serial_every and i % a.serial_every == a.serial_every - 1
        rec = {'i': i, 'key': k, 'arm': 'serial' if serial else 'drafted', 'utc': time.strftime('%FT%TZ', time.gmtime()), 'elapsed_s': round(time.monotonic() - T0, 1)}
        try:
            row, text, _ = chat(a.base, [{'role': 'user', 'content': P[k]['content']}], a.max_tokens, extra_s if serial else extra_d, timeout=900)
            rec.update(row); rec.pop('t_start_rel', None)
            first_sha.setdefault(k, row['sha256'])
            rec['equal_first'] = row['sha256'] == first_sha[k]
        except Exception as e:
            rec['error'] = repr(e)
        append(out / 'requests.jsonl', rec)
        print(json.dumps({x: rec.get(x) for x in ('i', 'elapsed_s', 'key', 'arm', 'ttft_s', 'decode_tok_s', 'equal_first', 'error')}), flush=True)
        i += 1
    stop.set(); th.join(5)
    append(out / 'vram.jsonl', {'when': 'end', 'utc': time.strftime('%FT%TZ', time.gmtime()), 'gpu0_free_mib': vram_free_mib(0), 'gpu1_free_mib': vram_free_mib(1)})


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    s = sp.add_parser('session'); s.add_argument('--key', required=True); s.add_argument('--turns', type=int, default=4)
    s.add_argument('--max-tokens', type=int, default=256)
    r = sp.add_parser('replay'); r.add_argument('--session', required=True)
    c = sp.add_parser('conc'); c.add_argument('--keys', required=True); c.add_argument('--extra', default='')
    c.add_argument('--tag', default=''); c.add_argument('--bases', default=''); c.add_argument('--reps', type=int, default=1)
    c.add_argument('--max-tokens', type=int, default=512); c.add_argument('--skip-alone', action='store_true')
    e = sp.add_parser('cell'); e.add_argument('--key', required=True); e.add_argument('--tag', required=True)
    e.add_argument('--extra', default=''); e.add_argument('--reps', type=int, default=1); e.add_argument('--max-tokens', type=int, default=512)
    k = sp.add_parser('soak'); k.add_argument('--keys', required=True); k.add_argument('--minutes', type=float, default=120)
    k.add_argument('--max-tokens', type=int, default=512); k.add_argument('--serial-every', type=int, default=0)
    for x in (s, r, c, e, k):
        x.add_argument('--base', default='http://127.0.0.1:8046'); x.add_argument('--out', required=True)
        x.add_argument('--prompts', default='~/B70-DOCS/results/2026/11/b70-tonight-20261108/prompts-qwen38.json')
    a = ap.parse_args()
    {'session': cmd_session, 'replay': cmd_replay, 'conc': cmd_conc, 'cell': cmd_cell, 'soak': cmd_soak}[a.cmd](a)


if __name__ == '__main__':
    main()
