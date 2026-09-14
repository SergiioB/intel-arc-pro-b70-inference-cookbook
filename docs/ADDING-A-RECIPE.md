# Adding or Changing a Recipe

Use this workflow for a model family, serving route, patch stack, benchmark generation, or public capability record.

## 1. Declare the delta

Write down:

- family slug and route;
- user-visible capability being added or changed;
- compatibility boundary: image, engine, model artifact, topology, patches;
- evidence tier and claims explicitly out of scope;
- which existing route this supersedes, if any.

If none of those boundaries changes, extend the existing route. Do not create a parallel guide.

## 2. Place each fact once

| Fact | Authority |
|---|---|
| Image digest and ordered patch compatibility | `docs/IMAGE-AND-PATCH-MATRIX.md` |
| Measurement definitions and validity gates | `docs/BENCHMARK-FORMAT.md` |
| Commands and route-specific warnings | `docs/<family>/<recipe>.md` |
| Public numeric/capability record | `data/benchmarks.v1.json` |
| Failure layer and operational status | `docs/RELIABILITY-REPORT.md` |
| Navigation | family `README.md` and `data/system-map.v1.json` only when a new authority is added |

Link to an authority instead of copying its table.

## 3. Build the family slice

A discoverable family has:

```text
docs/<family>/README.md              route chooser and claim boundary
docs/<family>/<recipe>.md            exact reproducible route
benchmarks/<family>/<launcher>       pinned executable launch path
results/<family>/<summary>.json      compact public evidence
data/benchmarks.v1.json              public record or capability
```

A family README must answer: which route, which hardware, which artifact, which authority, and what must not be mixed.

## 4. Preserve provenance

Every result generation records the coordinates required by `BENCHMARK-FORMAT.md`. Never overwrite a prior generation because a package or method changed. Add a new record, mark the old route superseded in prose, and keep its evidence link valid.

Stable identifiers:

- family slug for navigation;
- benchmark record `id` for public claims;
- immutable image digest and model revision for reproduction;
- commit SHA for published evidence;
- UTC generation ID for raw runs.

## 5. Generate views

```bash
python3 scripts/render-benchmark-catalog.py
```

Do not hand-edit `docs/BENCHMARK-CATALOG.md`. README contains route summaries, not a second benchmark database.

## 6. Run the acceptance gates

```bash
python3 scripts/check-repo.py
```

Use `python3 scripts/check-repo.py --format json` for machine-readable handoff.

For executable recipes, also run the narrowest applicable text-patch verifier or live smoke test and state whether it used a GPU. A GPU-free apply test proves source compatibility only; it does not prove runtime correctness or performance.

## 7. Publish a bounded claim

The final change note states:

1. authority changed;
2. exact route and generation;
3. validation commands and real results;
4. evidence tier;
5. what remains untested.

The contribution is incomplete if a future agent must infer patch order, choose between duplicated numbers, depend on a private path, or read an entire campaign narrative to find the current route.
