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

## Ledger discipline (required)

`docs/DO-NOT-REPEAT.md` is the canonical record of closed arms; accepted routes go to the family route plus `data/benchmarks.v1.json` and are indexed in `docs/WHAT-WORKED.md`. The rules are in `docs/EVIDENCE-LEDGERS.md`; `scripts/check-ledgers.py`, run by `scripts/check-repo.py`, enforces the record shape.

Read `docs/DO-NOT-REPEAT.md` **before** starting any optimisation, engine port, image bump, or quantisation swap. If an arm is closed and your situation does not match its **Revisit when** condition, do not run it.

- **No speed number without its shape.** Model revision, quantisation, engine image/commit, patch stack, card count and placement, context depth, concurrency, power cap, timing observer, cold/warm state. A rate without a shape is not a result.
- **Parity gate before a promotion.** Compare outputs against the reference artifact on a fixed prompt set and record matched *and* unmatched positions. A canary smoke proves the engine runs; it proves nothing about quality. Faster and unverified stays unpromoted.
- **Unknown tensor formats fail closed.** Read the per-tensor type census; never infer a storage type from a filename or a friendly label (`DNR-014`).
- **One variable per arm, same-session control.** Two simultaneous changes make attribution impossible and the record must say so.
- **Record the cause, or write `unexplained`.** A guessed cause closes a door that may be open.
- **History is immutable.** Supersede by adding a new generation and linking the old one; never rewrite a prior measurement.
- **Private lab work contributes causes, not rates.** Never import a private rate into a public table, never present an external result as our reproduction.

Closing an arm from a source reading — kernel coverage tables, plugin dispatch, memory arithmetic — is legitimate and costs no GPU time. Say so in the evidence column.

## Common task routes

- Host setup: `docs/FULL-SETUP-COMMANDS.md`
- Reproduction or patch choice: `docs/IMAGE-AND-PATCH-MATRIX.md`
- Client configuration: `docs/CONNECTING-CLIENTS.md`
- Production operation: `docs/RELIABILITY-REPORT.md`
- Leaderboard submission: `docs/localmaxxing-submission-schema.md`
- New or changed recipe: `docs/ADDING-A-RECIPE.md`
- Before any experiment: `docs/DO-NOT-REPEAT.md`
- Ledger rules and record shape: `docs/EVIDENCE-LEDGERS.md`
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
