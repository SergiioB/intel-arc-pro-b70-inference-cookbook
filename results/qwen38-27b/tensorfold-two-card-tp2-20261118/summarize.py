#!/usr/bin/env python3
"""Tables for PROCESS.md from the raw cells. Usage: summarize.py  (run in this folder; stdlib only)"""
import json, glob, os, statistics as st
def cells(d):
    p = os.path.join(d, 'cells.jsonl')
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def tab(a_dir, b_dir, a_name, b_name):
    A, B = cells(a_dir), cells(b_dir)
    key = lambda r: (r['depth'], r['kind'])
    print(f'| prompt tokens | kind | arm | {a_name} tok/s | {b_name} tok/s | ratio | {a_name} TTFT s | {b_name} TTFT s | drafted == serial ({b_name}) |')
    print('|---|---|---|---|---|---|---|---|---|')
    order = {'0': 0, '8k': 1, '32k': 2, '128k': 3}
    for k in sorted({key(r) for r in A + B}, key=lambda k: (order.get(k[0], 9), k[1])):
        for arm in ('drafted', 'serial'):
            a = [r for r in A if key(r) == k and r['arm'].startswith(arm)]
            b = [r for r in B if key(r) == k and r['arm'].startswith(arm)]
            if not a and not b: continue
            f = lambda rs: ', '.join(f"{r['decode_tok_s']:.1f}" for r in rs) or '–'
            m = lambda rs: st.mean(r['decode_tok_s'] for r in rs) if rs else None
            ratio = f'{m(b) / m(a):.2f}' if a and b else '–'
            hb = {r['sha256'] for r in B if key(r) == k}
            pt = (a or b)[0]['prompt_tokens']
            t = lambda rs: f"{st.mean(r['ttft_s'] for r in rs):.2f}" if rs else '–'
            print(f"| {pt:,} | {k[1]} | {arm} | {f(a)} | {f(b)} | {ratio} | {t(a)} | {t(b)} | {'yes' if len(hb) == 1 else 'NO: ' + str(len(hb)) + ' texts' if b else '–'} |")
def same_text(a_dir, b_dir):
    A = {(r['depth'], r['kind']): r['sha256'] for r in cells(a_dir) if r['arm'] == 'drafted'}
    B = {(r['depth'], r['kind']): r['sha256'] for r in cells(b_dir) if r['arm'] == 'drafted'}
    ks = sorted(set(A) & set(B))
    return sum(A[k] == B[k] for k in ks), len(ks)
if __name__ == '__main__':
    print('## tp2-prefix: one card (tp1) against two (tp2), same binary\n'); tab('tp2-prefix/tp1', 'tp2-prefix/tp2', 'one card', 'two cards')
    print('\nsame text one card vs two: %d of %d cells' % same_text('tp2-prefix/tp1', 'tp2-prefix/tp2'))
    for d in ('tp2-prefix/tp1', 'tp2-prefix/tp2'):
        p = os.path.join(d, 'arm_energy.jsonl')
        if os.path.exists(p): print(d, [json.loads(l) for l in open(p)])
    pairs = sorted(glob.glob('confirm/pair*-tp1'))
    if pairs:
        print('\n## confirmation pairs (drafted, code)\n')
        print('| pair | depth | one card | two cards | ratio |'); print('|---|---|---|---|---|')
        acc = {}
        for p in pairs:
            a, b = cells(p), cells(p.replace('-tp1', '-tp2'))
            for ra in a:
                rb = [r for r in b if r['depth'] == ra['depth']]
                if not rb: continue
                acc.setdefault(ra['depth'], []).append((ra['decode_tok_s'], rb[0]['decode_tok_s']))
                print(f"| {os.path.basename(p)[4:-4]} | {ra['prompt_tokens']:,} | {ra['decode_tok_s']:.1f} | {rb[0]['decode_tok_s']:.1f} | {rb[0]['decode_tok_s'] / ra['decode_tok_s']:.3f} |")
        for d, v in acc.items():
            r = [y / x for x, y in v]
            print(f"depth {d}: n={len(v)} one card mean {st.mean(x for x, _ in v):.1f} (min {min(x for x, _ in v):.1f}, max {max(x for x, _ in v):.1f}); two cards mean {st.mean(y for _, y in v):.1f} (min {min(y for _, y in v):.1f}, max {max(y for _, y in v):.1f}); ratio min {min(r):.3f} mean {st.mean(r):.3f} max {max(r):.3f}")
