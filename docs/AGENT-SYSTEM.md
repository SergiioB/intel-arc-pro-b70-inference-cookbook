# Agent System Guide

This repository is a layered publication system. Use the narrowest authority that answers the task; do not load the whole repository.

## Control loop

1. Classify the intent using the system map: read, setup, reproduce, connect,
   operate, submit, publish, or triage.
2. Read `data/system-map.v1.json` and follow that intent's `start` and `then` paths.
3. Select one model family and one serving route before reading commands.
4. Change an authority, not a rendered view.
5. Run `python3 scripts/check-repo.py`, the stable GPU-free validation entry point.
6. Report the changed authority, generated outputs, evidence, and remaining uncertainty.

The check command validates the system map, all tracked Markdown links, generated
catalog drift, Python syntax, and unit tests.

## Tower of abstractions

| Layer | Question it answers | Authority |
|---|---|---|
| L0 Control | Where do I start and what must remain true? | `AGENTS.md`, `data/system-map.v1.json` |
| L1 Contracts | What is valid and compatible? | `BENCHMARK-FORMAT`, image/patch matrix, reliability map |
| L2 Recipes | How do I run this exact family and route? | `docs/<family>/`, `benchmarks/<family>/` |
| L3 Evidence | What happened under exact coordinates? | raw/compact `results/`, `data/benchmarks.v1.json` |
| L4 Views | What should readers see? | README, generated catalog, SVGs |

Dependencies flow downward when deciding and upward when publishing: control selects contracts; contracts constrain recipes; recipes produce evidence; renderers turn evidence into views.

## Resource discipline

- Read the system map first, then at most one contract and one family hub until the task requires more.
- Search by stable record ID, family slug, image digest prefix, or patch filename. Do not search by a copied throughput value.
- Prefer JSON authorities and deterministic renderers over prose synchronization.
- Use `python3 scripts/check-repo.py --format json` when another agent needs structured gate results.
- Reuse an existing family route unless the engine, artifact class, or compatibility boundary is genuinely different.
- Keep raw campaign evidence outside the public catalog when it contains private paths or excessive bulk. Publish a compact, commit-pinned summary.

## Change routing

| Change | Edit first | Derived or linked work |
|---|---|---|
| Benchmark value | `data/benchmarks.v1.json` | Render catalog; update recipe only if it explains the result |
| Image or patch compatibility | `docs/IMAGE-AND-PATCH-MATRIX.md` | Family recipe, launcher, tests |
| Measurement method | `docs/BENCHMARK-FORMAT.md` | Harness/compiler and generation boundary |
| New family or route | `docs/ADDING-A-RECIPE.md` | Hub, launcher, evidence, catalog record |
| Failure ownership | `docs/RELIABILITY-REPORT.md` | Recipe warning or issue precheck |
| Host setup | `docs/FULL-SETUP-COMMANDS.md` | Image matrix and selected family recipe |
| Client connection | `docs/CONNECTING-CLIENTS.md` | Selected family recipe |
| Production operation | `docs/RELIABILITY-REPORT.md` | Watchdog, power, and topology guides |
| Leaderboard submission | `docs/localmaxxing-submission-schema.md` | Benchmark contract and catalog record |
| Reader navigation | `data/system-map.v1.json` | README/AGENTS links |

## Accretion rule

A contribution is complete only when another agent can discover it from the system map, reproduce it from one family route, validate its evidence without private context, and distinguish it from every incompatible or superseded route.

Use `docs/ADDING-A-RECIPE.md` for the complete write path and acceptance gates.
