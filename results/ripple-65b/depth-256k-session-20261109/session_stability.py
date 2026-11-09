#!/usr/bin/env python3
"""Decode stability across one long session: does speed hold as the conversation grows?

  session_stability.py run --base URL --model NAME --prompts PROMPTS.json --depth 128k \
      --kind code --turns 10 --max-tokens 384 --hwmon /sys/class/hwmon/hwmon8 --out DIR \
      [--extra-json '{"draft": false}'] [--no-thinking-field] [--window 64]

Builds ONE conversation: turn 0 sends the depth_bench corpus+task prompt for the requested
depth, then `turns` short follow-up tasks alternate; every assistant reply is appended, so
context grows by (prompt + each reply). Per turn it records TTFT, decode tok/s, per-window
decode tok/s inside the turn (default 64-token windows), prompt/context tokens, energy and
temperature. Writes OUT/session.jsonl (one row per turn) and OUT/meta.json.

Stdlib only. Greedy, fixed seed, thinking off.
"""
import argparse, hashlib, json, time, urllib.request
from pathlib import Path

FOLLOWUPS = [
    "In one short paragraph, summarize the document above.",
    "List the five most important files or functions mentioned above, one line each.",
    "Write a small unit test for the code you just wrote.",
    "Now refactor that into a shorter version without changing behaviour.",
    "What edge case would break it? Name two, one line each.",
    "Rewrite your answer to the original task in half the words.",
    "Add thorough error handling to the code above and show only the changed lines.",
    "Give a concrete example input and the exact expected output.",
    "Which part of the document above contradicts your design, if any? One paragraph.",
    "Write a docstring for it, then stop.",
]


def energy(hw):
    try:
        return int(Path(hw, 'energy1_input').read_text())
    except Exception:
        return None


def temp(hw):
    try:
        return int(Path(hw, 'temp2_input').read_text()) // 1000
    except Exception:
        return None


def turn(args, messages, max_tokens):
    body = {'model': args.model, 'messages': messages, 'max_tokens': max_tokens,
            'temperature': 0, 'top_p': 1, 'seed': 4242, 'stream': True,
            'stream_options': {'include_usage': True}}
    if not args.no_thinking_field:
        body['chat_template_kwargs'] = {'enable_thinking': False}
    body.update(json.loads(args.extra_json) if args.extra_json else {})
    req = urllib.request.Request(args.base + '/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    e0, t0 = energy(args.hwmon), time.monotonic()
    t_first = None
    stamps, parts, usage = [], [], None
    with urllib.request.urlopen(req, timeout=args.timeout) as r:
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
                d = ch.get('delta') or {}
                piece = (d.get('content') or '') + (d.get('reasoning_content') or '')
                if piece:
                    now = time.monotonic()
                    if t_first is None:
                        t_first = now
                    stamps.append(now)
                    parts.append(d.get('content') or '')
    t_end, e_end = time.monotonic(), energy(args.hwmon)
    text = ''.join(parts)
    n = (usage or {}).get('completion_tokens') or len(stamps)
    # per-window decode rate inside this turn (chunks as token proxy before usage arrives)
    windows = []
    w = args.window
    for i in range(w, len(stamps) + 1, w):
        dt = stamps[i - 1] - stamps[i - w]
        if dt > 0:
            windows.append({'tokens': w, 'tok_s': round(w / dt, 2),
                            'at_turn_token': i})
    dec_s = (stamps[-1] - t_first) if len(stamps) > 1 else None
    row = {'wall_s': round(t_end - t0, 3), 'ttft_s': round(t_first - t0, 3) if t_first else None,
           'completion_tokens': n, 'prompt_tokens': (usage or {}).get('prompt_tokens'),
           'decode_tok_s': round((n - 1) / dec_s, 2) if dec_s else None,
           'windows': windows, 'chars': len(text),
           'sha256': hashlib.sha256(text.encode()).hexdigest(),
           'temp_start_c': None, 'temp_end_c': temp(args.hwmon)}
    if e0 is not None and e_end is not None:
        row['energy_j'] = round((e_end - e0) / 1e6, 2)
    return row, text


def run(args):
    prompts = json.loads(Path(args.prompts).read_text())
    p = prompts[f'{args.kind}-{args.depth}']
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    messages = [{'role': 'user', 'content': p['content']}]
    # warmup on a throwaway short request, discarded
    turn(args, [{'role': 'user', 'content': 'Say ok.'}], 8)
    meta = {'model': args.model, 'base': args.base, 'kind': args.kind, 'depth': args.depth,
            'turns': args.turns, 'max_tokens_followups': args.max_tokens,
            'seed_prompt_sha256': p['sha256'], 'approx_seed_tokens': p['approx_tokens'],
            'utc_start': time.strftime('%FT%TZ', time.gmtime())}
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    for i in range(args.turns + 1):
        if i == 0:
            mt = 512
        else:
            messages.append({'role': 'user', 'content': FOLLOWUPS[(i - 1) % len(FOLLOWUPS)]})
            mt = args.max_tokens
        row, text = turn(args, messages, mt)
        row['turn'] = i
        row['temp_start_c'] = temp(args.hwmon)
        with (out / 'session.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        messages.append({'role': 'assistant', 'content': text})
        print(json.dumps({k: row.get(k) for k in ('turn', 'prompt_tokens', 'completion_tokens',
                                                  'ttft_s', 'decode_tok_s', 'temp_end_c')}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('run')
    r.add_argument('--base', required=True); r.add_argument('--model', required=True)
    r.add_argument('--prompts', required=True); r.add_argument('--depth', default='128k')
    r.add_argument('--kind', default='code'); r.add_argument('--turns', type=int, default=10)
    r.add_argument('--max-tokens', type=int, default=384)
    r.add_argument('--hwmon', required=True); r.add_argument('--out', required=True)
    r.add_argument('--extra-json', default=''); r.add_argument('--timeout', type=int, default=3600)
    r.add_argument('--no-thinking-field', action='store_true')
    r.add_argument('--window', type=int, default=64)
    a = ap.parse_args()
    run(a)


if __name__ == '__main__':
    main()
