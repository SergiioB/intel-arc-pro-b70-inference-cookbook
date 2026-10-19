#!/usr/bin/env python3
"""Save matched C1 chat responses and server timings for Strata quant comparisons."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--url', default='http://127.0.0.1:8000/v1')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--label', required=True)
p.add_argument('--repeats', type=int, default=3)
p.add_argument('--suite', choices=['speed', 'quality'], default='speed')
p.add_argument('--effort', choices=['none', 'low', 'medium', 'high'], default='none')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)

def request(path, body=None):
    req = urllib.request.Request(a.url + path, data=None if body is None else json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)

models = request('/models')
(a.output / 'models.json').write_text(json.dumps(models, indent=2) + '\n')
cases = {
    'code': 'Write a complete Python implementation of an LRU cache with threading locks and benchmarks.',
    'reasoning': 'A train travels 120 km at 60 km/h and returns 120 km at 40 km/h. Explain the average speed for the whole journey, showing the calculation.',
    'prefill': 'Summarize the following technical notes accurately.\n' + 'Modern operating systems employ virtual memory management with paging and segmentation. ' * 2700,
}
expected = {}
if a.suite == 'quality':
    checks = {
        'arithmetic': ('Return only the integer result of 137 * 29.', '3973'),
        'average-speed': ('Travel 120 km at 60 km/h then 120 km at 40 km/h. Return only the average speed in km/h, as a number.', '48'),
        'logic': ('All dax are wugs. No wugs are zibs. Can a dax be a zib? Reply only yes or no.', 'no'),
        'json': ('Return exactly this JSON object with no prose: key answer has integer value 42.', '{"answer":42}'),
        'python': ('What does Python print for list(range(7))[1::2]? Return only the printed list.', '[1,3,5]'),
        'retrieval': ('Record A: key=H7, value=5821. Record B: key=J3, value=9184. Return only the value for H7.', '5821'),
    }
    cases = {name: prompt for name, (prompt, answer) in checks.items()}
    expected = {name: answer for name, (prompt, answer) in checks.items()}
for name, prompt in cases.items():
    for n in range(a.repeats + 1):
        # Distinct prefixes prevent prompt-prefix reuse; only the fixed run identifier changes.
        body = {'model': models['data'][0]['id'], 'messages': [{'role': 'user', 'content': f'Run {name}-{n}.\n' + prompt}], 'max_tokens': 512 if name != 'prefill' else 128, 'temperature': 0, 'reasoning_effort': a.effort, 'cache_prompt': False}
        start = time.monotonic()
        response = request('/chat/completions', body)
        row = {'label': a.label, 'case': name, 'warmup': n == 0, 'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'wall_s': time.monotonic() - start, 'request_sha256': hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(), 'request': body, 'response': response}
        if name in expected:
            content = response['choices'][0]['message'].get('content') or ''
            row['expected'] = expected[name]
            row['correct'] = ''.join(content.lower().split()) == expected[name].lower()
        (a.output / f'{name}-{n}.json').write_text(json.dumps(row, indent=2) + '\n')
        print(a.label, name, n, response.get('usage'), response.get('timings'), flush=True)
