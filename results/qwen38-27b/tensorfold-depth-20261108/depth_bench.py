#!/usr/bin/env python3
"""Engine-agnostic single-stream depth benchmark for OpenAI-compatible servers.

One harness for TensorFold, vLLM/exl3xpu and llama.cpp so the cells match.

  depth_bench.py make-prompts OUT.json --tokenizer DIR/tokenizer.json
  depth_bench.py run --base URL --model NAME --prompts OUT.json --out DIR \
      --arm drafted|serial --hwmon /sys/class/hwmon/hwmon8 [--depths 0,8k,32k,128k] [--reps 2]
      [--no-draft-field] [--extra-json '{"draft": false}']

Per request it records: TTFT (first content token), decode tok/s over the
streamed window ((n-1)/(t_last-t_first) with n = completion tokens from usage),
wall time, GPU card energy over the whole request and over the decode window,
and the sha256 of the output so drafted and serial arms can be compared
byte for byte. Greedy, thinking off, fixed max_tokens.

Stdlib only.
"""
import argparse, hashlib, json, os, sys, time, urllib.request
from pathlib import Path

CODE_ROOT = Path('$LLAMA_NEW')
PROSE_ROOT = Path('$COOKBOOK/docs')
DEPTHS = {'0': 0, '8k': 8192, '32k': 32768, '128k': 131072}

TASKS = {
    'code': ("Above is part of a C/C++ inference engine's source tree. Write a new, self-contained C++17 "
             "header `ring_kv.hpp` implementing a thread-safe fixed-capacity ring buffer for KV-cache "
             "block ids with push, pop, peek and a resize that preserves order. Include doc comments and "
             "a short usage example in a comment at the end."),
    'chat': ("Above are documentation pages about running language models on Intel Arc Pro GPUs. "
             "Write a clear, well-structured explanation for a newcomer of how to choose a model, an engine "
             "and a context length for one 32 GB card, with concrete recommendations and caveats. Use "
             "headings and bullet points."),
}
SHORT = {
    'code': ("Write a self-contained C++17 header `ring_kv.hpp` implementing a thread-safe fixed-capacity "
             "ring buffer for KV-cache block ids with push, pop, peek and a resize that preserves order. "
             "Include doc comments and a short usage example in a comment at the end."),
    'chat': ("Explain for a newcomer how to choose a language model, an inference engine and a context "
             "length for one 32 GB GPU, with concrete recommendations and caveats. Use headings and bullet points."),
}


def corpus(kind):
    if kind == 'code':
        files = sorted(list((CODE_ROOT / 'src').rglob('*.cpp')) + list((CODE_ROOT / 'ggml/src/ggml-sycl').rglob('*.cpp')))
    else:
        files = sorted(PROSE_ROOT.rglob('*.md'))
    for f in files:
        try:
            t = f.read_text(errors='replace')
        except Exception:
            continue
        yield f"\n\n===== {f.name} =====\n{t}"


def make_prompts(args):
    from tokenizers import Tokenizer  # local only, for sizing
    tok = Tokenizer.from_file(args.tokenizer)
    out = {}
    for kind in TASKS:
        text = ''.join(corpus(kind))
        ids = tok.encode(text).ids
        for name, n in DEPTHS.items():
            if n == 0:
                content = SHORT[kind]
            else:
                body_ids = ids[: n - 300]
                if len(body_ids) < n - 300:
                    raise SystemExit(f'corpus too small for {kind} {name}: {len(ids)}')
                content = tok.decode(body_ids) + '\n\n' + TASKS[kind]
            out[f'{kind}-{name}'] = {'kind': kind, 'depth': name, 'content': content,
                                     'approx_tokens': len(tok.encode(content).ids),
                                     'sha256': hashlib.sha256(content.encode()).hexdigest()}
            print(name, kind, out[f'{kind}-{name}']['approx_tokens'], file=sys.stderr)
    Path(args.out).write_text(json.dumps(out, indent=1))


def energy(hw):
    try:
        return int(Path(hw, 'energy1_input').read_text())
    except Exception:
        return None


def ask(args, content, extra):
    body = {'model': args.model, 'messages': [{'role': 'user', 'content': content}],
            'max_tokens': args.max_tokens, 'temperature': 0, 'top_p': 1, 'seed': 4242, 'stream': True,
            'stream_options': {'include_usage': True},
            'chat_template_kwargs': {'enable_thinking': False}}
    body.update(extra)
    req = urllib.request.Request(args.base + '/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    e0, t0 = energy(args.hwmon), time.monotonic()
    t_first = t_last = None
    e_first = None
    parts, usage, last_obj, nchunks = [], None, None, 0
    with urllib.request.urlopen(req, timeout=args.timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                break
            obj = json.loads(data)
            last_obj = obj
            if obj.get('usage'):
                usage = obj['usage']
            for ch in obj.get('choices') or []:
                d = ch.get('delta') or {}
                piece = (d.get('content') or '') + (d.get('reasoning_content') or '')
                if piece:
                    now = time.monotonic()
                    if t_first is None:
                        t_first, e_first = now, energy(args.hwmon)
                    t_last = now
                    nchunks += 1
                    parts.append(d.get('content') or '')
    t_end, e_end = time.monotonic(), energy(args.hwmon)
    text = ''.join(parts)
    n = (usage or {}).get('completion_tokens') or nchunks
    dec_s = (t_last - t_first) if t_first and t_last and t_last > t_first else None
    row = {'wall_s': round(t_end - t0, 3), 'ttft_s': round(t_first - t0, 3) if t_first else None,
           'completion_tokens': n, 'prompt_tokens': (usage or {}).get('prompt_tokens'),
           'chunks': nchunks,
           'decode_tok_s': round((n - 1) / dec_s, 2) if dec_s else None,
           'e2e_tok_s': round(n / (t_end - t0), 2),
           'sha256': hashlib.sha256(text.encode()).hexdigest(), 'chars': len(text)}
    if e0 is not None and e_end is not None:
        row['energy_j'] = round((e_end - e0) / 1e6, 2)
        row['avg_w'] = round((e_end - e0) / 1e6 / (t_end - t0), 1)
        if e_first is not None and dec_s:
            row['decode_avg_w'] = round((e_end - e_first) / 1e6 / (t_end - t_first), 1)
            row['decode_j_per_tok'] = round((e_end - e_first) / 1e6 / max(n - 1, 1), 3)
    row['last_chunk_extra'] = {k: v for k, v in (last_obj or {}).items() if k not in ('choices', 'id', 'object', 'created', 'model')}
    return row, text


def temp(hw):
    try:
        return int(Path(hw, 'temp2_input').read_text()) // 1000
    except Exception:
        return None


def run(args):
    prompts = json.loads(Path(args.prompts).read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    extra = json.loads(args.extra_json) if args.extra_json else {}
    depths = args.depths.split(',')
    kinds = args.kinds.split(',')
    # warmup, discarded
    ask(args, 'Say ok.', extra)
    for rep in range(args.reps):
        for d in depths:
            for k in kinds:
                p = prompts[f'{k}-{d}']
                # cool to the target, or to the card's plateau (no drop in 30 s) if the floor is above it
                best, since = temp(args.hwmon), time.monotonic()
                while args.cool and temp(args.hwmon) and temp(args.hwmon) > args.cool and time.monotonic() - since < 30:
                    time.sleep(2)
                    if temp(args.hwmon) < best:
                        best, since = temp(args.hwmon), time.monotonic()
                t_start = temp(args.hwmon)
                row, text = ask(args, p['content'], extra)
                row.update({'arm': args.arm, 'rep': rep, 'depth': d, 'kind': k, 'prompt_sha256': p['sha256'],
                            'temp_start_c': t_start, 'temp_end_c': temp(args.hwmon), 'utc': time.strftime('%FT%TZ', time.gmtime())})
                with (out / 'cells.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
                with (out / 'outputs.jsonl').open('a') as f:
                    f.write(json.dumps({'arm': args.arm, 'rep': rep, 'depth': d, 'kind': k, 'sha256': row['sha256'], 'text': text}) + '\n')
                print(json.dumps({x: row.get(x) for x in ('arm', 'rep', 'depth', 'kind', 'prompt_tokens', 'completion_tokens',
                                                           'ttft_s', 'decode_tok_s', 'decode_avg_w', 'decode_j_per_tok', 'temp_end_c')}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    m = sp.add_parser('make-prompts'); m.add_argument('out'); m.add_argument('--tokenizer', required=True)
    r = sp.add_parser('run')
    r.add_argument('--base', required=True); r.add_argument('--model', required=True)
    r.add_argument('--prompts', required=True); r.add_argument('--out', required=True)
    r.add_argument('--arm', required=True); r.add_argument('--hwmon', required=True)
    r.add_argument('--depths', default='0,8k,32k,128k'); r.add_argument('--kinds', default='code,chat')
    r.add_argument('--reps', type=int, default=2); r.add_argument('--max-tokens', type=int, default=512)
    r.add_argument('--extra-json', default=''); r.add_argument('--timeout', type=int, default=3600)
    r.add_argument('--cool', type=int, default=0, help='wait until GPU temp <= this before each cell')
    a = ap.parse_args()
    make_prompts(a) if a.cmd == 'make-prompts' else run(a)


if __name__ == '__main__':
    main()
