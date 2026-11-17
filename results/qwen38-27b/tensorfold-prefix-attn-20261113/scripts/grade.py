"""Runs inside a no-network container. /fx: fixtures.json, graders.py; /run: generations.jsonl -> graded.json"""
import json, sys
sys.path.insert(0, '/fx'); import graders
FX = json.load(open('/fx/fixtures.json')); items = {i['item_id']: i for i in FX['code'] + FX['regression']}
out = {}
for line in open('/run/generations.jsonl'):
    r = json.loads(line); it = items[r['id']]
    try: score, passed, _, detail, _ = graders.grade_item(it, {'answer_text': r['output']})
    except Exception as e: passed, detail = False, f'grader error: {e}'
    if r['finish'] == 'length': passed = False
    out.setdefault(r['tag'], {})[r['id']] = {'split': r['split'], 'passed': bool(passed), 'finish': r['finish'], 'sha256': r['sha256'], 'detail': str(detail)[-120:]}
json.dump(out, open('/run/graded.json', 'w'), indent=0)
for tag, g in out.items():
    s = {}
    for v in g.values(): s.setdefault(v['split'], [0, 0]); s[v['split']][0] += v['passed']; s[v['split']][1] += 1
    print(tag, s)
