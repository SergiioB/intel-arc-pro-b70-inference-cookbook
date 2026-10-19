#!/usr/bin/env python3
"""Submit matched native-context Strata code measurements after catalog sync."""
import argparse
import json
import os
from pathlib import Path
import statistics
import urllib.request

API = 'https://www.intelinside.ai/api/v1'
ROOT = Path(__file__).resolve().parents[1]
RECIPE = 'https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/docs/qwen38-flash-next/QWEN38-FLASH-NEXT-XESTRATA.md'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--submit', action='store_true', help='write to IntelInside; default only prepares the payload')
args = parser.parse_args()

def call(path, body=None, idem=None):
    headers = {'Content-Type': 'application/json'}
    if body is not None:
        key = os.environ.get('INTELINSIDE_API_KEY') or (Path.home() / '.config/intelinside/api_key').read_text().strip()
        headers['Authorization'] = 'Bearer ' + key
        headers['Idempotency-Key'] = idem
    request = urllib.request.Request(API + path, data=None if body is None else json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)

catalog = call('/catalog')
model = next(m for m in catalog['models'] if m['id'] == 'qwen3-8-flash-next')
missing = [q for q in ('iq3_xxs', 'iq3_s') if q not in model['quants']]
if not any(r['id'] == 'strata' for r in catalog['runtimes']):
    missing.append('runtime:strata')
items = []
for quant, folder in [('iq3_xxs', 'iq3_xxs-native'), ('iq3_s', 'iq3_s-native')]:
    rows = [json.loads((ROOT / 'results/qwen38-strata-quant-20261019' / folder / f'code-{n}.json').read_text()) for n in (1, 2, 3)]
    if any(r['response']['timings']['cache_n'] != 0 or r['response']['timings']['predicted_n'] != 512 for r in rows):
        raise SystemExit('Refusing a cached or truncated-coordinate code sample: ' + quant)
    items.append({'rigId': '1', 'modelId': model['id'], 'quant': quant, 'runtimeId': 'strata',
                  'runtimeVersion': 'XeStrata xe0.1.45-1 SYCL', 'runDate': '2026-10-19',
                  'componentId': 'intel-arc-pro-b70', 'componentQuantity': 2, 'batchSize': 1,
                  'contextLength': 262144, 'repoUrl': RECIPE,
                  'decodeTps': statistics.median(r['response']['timings']['predicted_per_second'] for r in rows),
                  'promptTps': statistics.median(r['response']['timings']['prompt_per_second'] for r in rows),
                  'runtimeFlags': '--layer-split 24 --pipeline-windows 2 --spec 4 --kv int8 --max-context 262144',
                  'notes': 'C1 code generation, 34 prompt / 512 output tokens, temperature 0, reasoning_effort none, zero reported cache hits; median of 3 after warm-up. Both rates come from this code workload, not the separate 32K prefill test. As-found power caps: GPU0 180 W, GPU1 230 W; not changed. Engine timings. Self-reported; see raw responses in recipe.'})
if not args.submit:
    print(json.dumps({'catalog_missing': missing, 'items': items}, indent=2))
    raise SystemExit(0)
if missing:
    raise SystemExit('Catalog sync required before submission (PR #116): ' + ', '.join(missing))
runtime = call('/custom-runtimes', {'runtimeId': 'strata', 'name': 'XeStrata xe0.1.45-1 SYCL + MTP4',
               'repoUrl': 'https://github.com/MistvVK/XeStrata',
               'summary': 'XeStrata fork with SYCL, two-GPU layer split and pipelined MTP4.',
               'notes': 'Engine SHA256 d633e321ba22179c9b1515840b922a49d0d426caea4819237fa2350f4c92e450. Reproduction and raw measurements: ' + RECIPE},
               'sergiiob-xestrata-correct-runtime-20261019-v1')
for item in items:
    item['customRuntimeId'] = runtime['id']
receipt = call('/results/batch', {'items': items}, 'sergiiob-strata-iq3-native-code-20261019-v1')
print(json.dumps(receipt, indent=2))
