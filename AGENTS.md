# AGENTS.md - Intel Arc Pro B60/B70 Inference Cookbook

> Public control plane for agents reproducing, operating, or extending this cookbook.

## Start here

1. Read `data/system-map.v1.json`.
2. Choose the closest task intent: read, setup, reproduce, connect, operate, submit, publish, or triage.
3. Follow only that intent's `start` and `then` paths until the task requires another authority.
4. Read `docs/AGENT-SYSTEM.md` before changing repository structure or public claims.

This repository is the public recipe and evidence catalog. Private lab campaigns do not belong here. Never publish credentials, private host paths, local addresses, unpublished image names, or raw evidence that exposes them.

## Hard invariants

- `data/benchmarks.v1.json` is the public numeric catalog authority. `docs/BENCHMARK-CATALOG.md` is generated.
- `docs/IMAGE-AND-PATCH-MATRIX.md` owns image and ordered patch compatibility.
- `docs/BENCHMARK-FORMAT.md` owns measurement definitions, evidence layout, and publication gates.
- Model-specific commands live under one `docs/<family>/` route and one `benchmarks/<family>/` slice.
- Never mix image, patch, model, or benchmark numbers across family routes.
- Any changed image, model revision, tokenizer, patch, runtime, workload, or timing definition starts a new result generation.
- Historical evidence stays immutable and labeled. Supersede by adding a new generation and linking the old one.
- A GPU-free patch apply test proves source compatibility only. It does not prove runtime correctness, stability, or speed.

## Common task routes

- Host setup: `docs/FULL-SETUP-COMMANDS.md`
- Reproduction or patch choice: `docs/IMAGE-AND-PATCH-MATRIX.md`
- Client configuration: `docs/CONNECTING-CLIENTS.md`
- Production operation: `docs/RELIABILITY-REPORT.md`
- Leaderboard submission: `docs/localmaxxing-submission-schema.md`
- New or changed recipe: `docs/ADDING-A-RECIPE.md`
- Benchmark or capability record: `data/benchmarks.v1.json`
- Issue or PR triage: `docs/MAINTAINER-PRECHECKS.md`

Do not add a second family guide when the existing route can be extended. Do not copy authority tables into README or family prose.

## Validation

Run before handing off any repository change:

```bash
python3 scripts/check-repo.py
```

Use `python3 scripts/check-repo.py --format json` for machine-readable handoff.

For executable changes, also run the narrowest applicable verifier or smoke test. Report the exact command, last result, GPU use, evidence tier, and untested scope.

## Definition of done

Another agent can:

1. discover the change from the system map or family hub;
2. select one unambiguous compatible route;
3. reproduce it without private context;
4. trace every public number to a stable record and commit-pinned evidence;
5. run deterministic validation and get the same result.
