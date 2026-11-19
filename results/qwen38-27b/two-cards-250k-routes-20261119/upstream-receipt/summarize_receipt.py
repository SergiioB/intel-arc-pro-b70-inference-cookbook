#!/usr/bin/env python3
"""Tables for the #2543 combined receipt: main against main+#2595+#2799, from runs/*/cells.jsonl and server.log.
Usage: python3 summarize_receipt.py  (run in this folder; stdlib only)"""
import glob, json, os, re, statistics as st

ORDER = {'0': 0, '8k': 1, '32k': 2, '128k': 3}
KIND = {'code': 'code', 'chat': 'prose'}


def load(side):
    rows, logs = [], []
    for d in sorted(glob.glob(f'runs/*-{side}')):
        rows += [json.loads(l) for l in open(os.path.join(d, 'cells.jsonl'))]
        logs += [l for l in open(os.path.join(d, 'server.log')) if l.startswith('[tensorfold] done ') and ' tokens=512 ' in l]
    return rows, logs


def server_rows(logs):
    out = []
    for l in logs:
        f = dict(kv.split('=', 1) for kv in l.split()[3:] if '=' in kv)
        out.append({'prompt': int(f['prompt']), 'sha': f['sha'], 'accepted': f['accepted'], 'tps': float(f['tok/s'])})
    return out


def main():
    data = {s: load(s) for s in ('main', 'combo')}
    srv = {s: server_rows(data[s][1]) for s in data}
    keys = sorted({(r['depth'], r['kind'], r['prompt_tokens']) for r in data['main'][0] + data['combo'][0]},
                  key=lambda k: (ORDER[k[0]], k[1]))
    print('## Token hashes (server `sha`, first 12 hex of the token sha256; client text sha256 in brackets)\n')
    print('| Prompt tokens | Kind | main drafted | main `"draft": false` | combined drafted | combined `"draft": false` | all equal |')
    print('|---|---|---|---|---|---|---|')
    allok = True
    for d, k, n in keys:
        cells, shas = [], set()
        for s in ('main', 'combo'):
            for arm in ('drafted', 'serial'):
                rs = [r for r in data[s][0] if r['depth'] == d and r['kind'] == k and r['arm'].startswith(arm)]
                sv = [x['sha'] for x in srv[s] if x['prompt'] == n]
                tx = {r['sha256'][:12] for r in rs}
                shas |= tx
                cells.append(f"[{'/'.join(sorted(tx))}] ×{len(rs)}")
        ss = {x['sha'] for s in srv for x in srv[s] if x['prompt'] == n}
        ok = len(shas) == 1 and len(ss) == 1
        allok &= ok
        cells = [f"`{'/'.join(sorted(ss))}` " + c for c in cells]
        print(f"| {n:,} | {KIND[k]} | " + ' | '.join(cells) + f" | {'yes' if ok else 'NO'} |")
    print(f'\nAll cells equal: {allok}\n')
    print('## Decode tok/s after the first token (client, streamed), TTFT s\n')
    print('| Prompt tokens | Kind | Arm | main | combined | ratio | TTFT main / combined |')
    print('|---|---|---|---|---|---|---|')
    for d, k, n in keys:
        for arm in ('drafted', 'serial'):
            v = {}
            for s in ('main', 'combo'):
                v[s] = [r for r in data[s][0] if r['depth'] == d and r['kind'] == k and r['arm'].startswith(arm)]
            if not v['main'] or not v['combo']:
                continue
            m = {s: st.mean(r['decode_tok_s'] for r in v[s]) for s in v}
            t = {s: st.mean(r['ttft_s'] for r in v[s]) for s in v}
            f = lambda rs: ', '.join(f"{r['decode_tok_s']:.1f}" for r in rs)
            print(f"| {n:,} | {KIND[k]} | {'drafted' if arm == 'drafted' else '`\"draft\": false`'} | {f(v['main'])} | "
                  f"{f(v['combo'])} | {m['combo'] / m['main']:.3f} | {t['main']:.2f} / {t['combo']:.2f} |")


if __name__ == '__main__':
    main()
