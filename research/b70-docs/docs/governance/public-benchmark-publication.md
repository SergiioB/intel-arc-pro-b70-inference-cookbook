---
title: "Public Benchmark Publication Workflow"
type: governance
date: 2026-08-30
status: living
---

---
doc_id: GOV-PUBLIC-BENCHMARK-PUBLICATION
doc_type: governance_procedure
title: Public Benchmark Publication Workflow
lifecycle: active
created: 2026-08-30
updated: 2026-08-30
canonical_for:
  - private-to-public-benchmark-promotion
  - xecores-benchmark-publication
---

# Public Benchmark Publication Workflow

Use this after a B70 campaign has complete raw evidence and Sergio has authorized public publication. Benchmark completion alone does not authorize a cookbook, XeCores, leaderboard, or social update.

## Data flow

```text
private B70-DOCS campaign + results
            │ claims, privacy, and evidence audit
            ▼
public cookbook evidence + data/benchmarks.v1.json
            │ live GitHub catalog
            ├──────────────► cookbook generated catalog
            └──────────────► XeCores homepage and Cookbook page
                                  │
                                  └── vendored fallback refreshed before deploy
```

The public cookbook catalog is the only public benchmark-data authority. Do not copy benchmark values into XeCores HTML, JavaScript, or its fallback JSON by hand.

## 1. Finish the private evidence record

Keep the complete campaign under one UTC result root and follow `docs/methods/benchmark-protocol.md` plus `.agents/skills/b70-benchmark-claims-standard/SKILL.md`.

Required before promotion:

1. exact model and revision;
2. hardware count and per-card configuration;
3. engine, image/build, kernel, driver, and patches;
4. quantization, dtype, KV cache, speculation, parallelism, context, scheduler, and power cap;
5. exact prompt/output targets and actual token counts;
6. explicit C1/C2/C4/C8/C16/C32 concurrency where measured;
7. warmups, measured samples, failures, interval definition, and summary statistic;
8. raw request samples, logs, cache/speculation counters, telemetry, and cleanup state;
9. correctness/quality gate result;
10. claims audit stating what is safe to publish.

Missing fields are not guessed during publication.

## 2. Choose the public record class

Use `kind: benchmark` only when the public artifact includes:

- exact hardware, model, runtime, and workload coordinates;
- P/G/C values;
- sample count and statistic;
- named metric and unit;
- full-SHA public evidence URL.

Use `kind: capability` when a recipe works but any ranking coordinate is missing. Capability records may summarize an early observation, but they carry no ranking metric.

Examples as of 2026-08-30:

- Dual-B70 TP2/PP2: `validated` capability until a complete matched numeric card is published.
- Qwen3.8-27B FP8 W8A16: `provisional` capability because the public 48 token/s note lacks output length, sample count, and the full workload coordinate.

## 3. Create sanitized public evidence

Publish only the evidence needed to interpret the claim. Remove or generalize:

- private result-root paths;
- hostnames, LAN addresses, user names, mount topology, and machine IDs;
- API keys, bearer headers, credentials, private contact details, and environment dumps;
- unrelated logs and raw model output containing personal data.

Keep the public commit immutable enough to support a full-SHA evidence link. The private result root remains the audit authority.

## 4. Edit the benchmark once

In the public inference cookbook, edit:

```text
data/benchmarks.v1.json
```

Then run:

```bash
python3 scripts/render-benchmark-catalog.py
python3 scripts/render-benchmark-catalog.py --check
python3 -m unittest tests/test_benchmark_catalog.py -v
```

Commit the JSON, the cookbook repo generated catalog (`intel-arc-pro-b70-inference-cookbook/docs/BENCHMARK-CATALOG.md`), and any new public evidence together. Push the cookbook only after every check passes.

## 5. XeCores synchronization

XeCores reads the live cookbook file from GitHub, so a cookbook push updates the public tables without manually editing XeCores or redeploying it.

Before the next XeCores deployment, refresh its outage fallback and public-doc corpus:

```bash
npm run sync:cookbook
```

Commit the resulting fallback and provenance changes with the site. The fallback must name the cookbook commit it copied.

## 6. Community runner/API path

For a new automated run against a local vLLM, SGLang, or llama.cpp OpenAI-compatible endpoint:

```bash
curl -fsSLo xecores-runner.py https://xecores.com/xecores-runner.py
python3 xecores-runner.py \
  --engine vllm \
  --endpoint http://127.0.0.1:8000/v1 \
  --model MODEL_ID \
  --model-revision REVISION \
  --engine-version VERSION \
  --quantization FORMAT \
  --dtype DTYPE \
  --gpu-model "Intel Arc Pro B70" \
  --gpu-count 1 \
  --vram-gib 32 \
  --power-limit-w 150 \
  --prompt-tokens 512,8192 \
  --output-tokens 128 \
  --concurrency c1,c2,c4,c8,c16,c32 \
  --warmups 2 \
  --requests-per-worker 3 \
  --ignore-eos
```

Inspect `xecores-results.json` before adding `--submit`. A submission enters the private `pending` queue. API acceptance proves schema compliance and runner provenance; it does not prove the contributor machine or result is truthful.

Never place local server keys in command history when avoidable. Use `OPENAI_API_KEY`; the runner does not include it in the payload.

## 7. Trust and approval

Public trust labels stay distinct:

- `automated-community`: produced by the official runner and reviewed for schema/evidence;
- `cross-confirmed`: a materially matching independent machine result exists;
- `xecores-reproduced`: XeCores reproduced the result;
- `official-lab`: measured under the public cookbook process with commit-pinned evidence.

Only approved D1 rows are public. Promoting an API row into the canonical cookbook still requires sanitized public evidence and a cookbook claims review.

## Verification checklist

- [ ] Private campaign passes benchmark and claims standards.
- [ ] Public evidence contains no credentials or private machine identifiers.
- [ ] Every numeric row has P/G/C, n/statistic, metric semantics, and pinned evidence.
- [ ] Incomplete observations are capabilities, outside rankings.
- [ ] Cookbook renderer and unit tests pass.
- [ ] GitHub raw catalog returns the new record after push.
- [ ] XeCores renders the live record and retains a commit-pinned fallback.
- [ ] API submissions remain private until reviewed.
