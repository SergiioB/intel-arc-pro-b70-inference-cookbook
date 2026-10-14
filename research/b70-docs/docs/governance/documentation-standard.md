---
title: "B70 Documentation Standard"
status: living
type: governance
date: 2026-09-15
---

---
id: b70-gov-documentation-standard
title: B70 Documentation Standard
document_type: governance
status: canonical
version: "1.0"
effective_date: 2026-08-19
last_reviewed: 2026-08-19
authority: docs/methods/benchmark-protocol.md
scope: Repository documentation, evidence references, and public-facing technical claims
---

# B70 Documentation Standard

## Purpose

This standard defines how B70 inference documentation is organized, sourced, reviewed, and maintained. It applies to technical guides, benchmark reports, research notes, recommendations, reference material, visual assets, and public claim copy.

The [B70 Benchmark Protocol](../methods/benchmark-protocol.md) remains the normative authority for benchmark validity and claim publication. This document governs how that authority is represented in documentation; it does not replace the protocol or reinterpret measured results.

## Document classes

Every new canonical document must declare one `document_type` in YAML front matter.

| Type | Purpose | Expected content |
|---|---|---|
| `governance` | Defines authority, lifecycle, identifiers, and repository rules | Normative requirements and ownership boundaries |
| `method` | Defines a repeatable validation or evaluation procedure | Inputs, controls, steps, outputs, and pass/fail semantics |
| `reference` | Summarizes stable facts or current support decisions | Definitions, matrices, decision records, and source links |
| `report` | Records a bounded investigation or campaign | Scope, dated evidence, findings, limitations, and disposition |
| `guide` | Explains an operational workflow | Prerequisites, commands, safety checks, and recovery |

Research notes and dated reports may remain provisional. A file becomes canonical only when its front matter says `status: canonical` and its subject is assigned authority in the [authority map](authority-map.md).

## Required front matter

Canonical documents use these keys in this order:

```yaml
---
id: b70-<domain>-<slug>
title: Human-readable title
document_type: governance | method | reference | report | guide
status: canonical | provisional | superseded | archived
version: "1.0"
effective_date: YYYY-MM-DD
last_reviewed: YYYY-MM-DD
authority: repository-relative/path.md
scope: Short statement of what the document governs
---
```

Identifiers and field semantics are defined in the [metadata and ID schema](metadata-and-id-schema.md). Dates use ISO 8601 calendar form. Repository paths in metadata are repository-relative and must resolve at review time.

## Source discipline

1. Link to the evidence of record instead of copying large result tables into governance, method, or reference documents.
2. Use [benchmark history](../../archive/legacy-docs-20260915/benchmark-history.md) as the chronological run index and `results/` as the raw measurement trail.
3. Keep exact workload coordinates and qualifiers attached to every repeated result: engine/build, model artifact and quantization, configured context, actual prompt tokens, requested and actual output tokens, C1 or Cn, cache state, timing source, statistic, repetitions, configured power cap, measured draw when available, correctness scope, and evidence status.
4. Write `unknown` or `not measured` when a required coordinate is absent. Do not infer it from a filename, configured context, neighboring run, or leaderboard field.
5. Do not convert a configured power cap into measured draw, a Cn aggregate into C1 decode, a cold input rate into isolated prefill, or a maximum observed sample into a median or sustained result.
6. External sources support background or externally published evidence. They do not silently upgrade a local result to independent reproduction.

## Vocabulary

Use the protocol vocabulary without local synonyms:

- Result classes: `prefill`, `C1 decode`, `sustained decode`, `aggregate concurrency`, `latency`, `efficiency`, `correctness`, and `capacity`.
- Cache states: `cold_model_cold_prefix`, `warm_model_cold_prefix`, `warm_model_warm_prefix`, and `mixed_or_unknown`.
- Statistics: `maximum observed`, `median`, `mean`, `sustained`, `single observation`, and `aggregate`.
- Comparison classes: `matched_exact`, `matched_with_tokenizer_delta`, `matched_except_quant`, `best_cell_cross_result`, `directional_only`, and `not_comparable`.
- Truth verdicts: `CONFIRMED`, `PARTIAL`, `REFUTED`, and `UNCERTAIN`.
- Evidence tiers: `E0` through `E5`, as defined by the protocol.

Do not use “verified,” “production,” “safe,” “faster,” or “supports” without naming the exact scope that justifies the term.

## Canonical writing pattern

A canonical technical statement should follow this order:

1. **Decision or definition.** State the narrow conclusion.
2. **Scope.** Name the engine, artifact, workload, or repository surface covered.
3. **Evidence.** Link to the protocol, run index, raw artifact, or external source.
4. **Limit.** State what the evidence does not establish.
5. **Disposition.** Say whether the item is current, provisional, superseded, invalid, or archived.

Prefer exact tables for decisions and compact prose for interpretation. Avoid decorative benchmark tables, repeated hardware marketing specifications, and copied launch commands that can drift from their source.

## Links and paths

- Use relative links for repository documents.
- Use repository-relative code paths in prose, enclosed in backticks.
- Link to a directory only when that directory is itself the evidence root and its manifest identifies the contained artifacts.
- Link directly to the manifest, summary, raw result, or report when a claim depends on one file.
- Validate local links before review. A missing target blocks canonical status.
- Historical links remain valid when their evidence is superseded; mark the evidence status rather than deleting the link.

## Change control

Canonical documents are updated in place when the rule or current decision changes. The editor must:

1. update `version` and `last_reviewed`;
2. identify the changed rule or decision in the review description;
3. preserve historical results and corrections under the [evidence lifecycle](evidence-lifecycle.md);
4. update dependent canonical references when authority changes; and
5. confirm that public wording still passes the publication gate in the protocol.

Do not rewrite raw evidence, silently remove failed samples, or move an existing document merely to improve taxonomy. Supersession is explicit and traceable.

## Review checklist

Before a canonical document is accepted, verify:

- required front matter is present and consistent;
- the authority path resolves and the document does not conflict with a higher authority;
- all local links resolve;
- benchmark vocabulary matches the protocol;
- repeated numbers are necessary, sourced, and fully qualified;
- provisional evidence is labeled as provisional;
- corrections preserve the old evidence and point to replacements; and
- the document distinguishes current live state from historical recorded state.

## Accretion

A campaign or research close is incomplete until:

1. the UTC result root is retained;
2. `registry/research-ledger.json` has an append-only row (Markdown `research/LEDGER.md` in parity);
3. `registry/model-status.json` is updated when a route or gate changed;
4. `registry/campaigns.yaml` lists the campaign when a result root exists;
5. `python3 tools/research.py render && python3 tools/research.py check` passes (program/ledger/status views in parity).

Do not copy numeric claims into `AGENTS.md` or XeCores HTML. Public numbers enter `intel-arc-pro-b70-inference-cookbook/data/benchmarks.v1.json` via [public benchmark publication](public-benchmark-publication.md).

See [B70 Agent Operating System](agent-operating-system.md).

## Related authorities

- [B70 Benchmark Protocol](../methods/benchmark-protocol.md)
- [Metadata and ID Schema](metadata-and-id-schema.md)
- [Evidence Lifecycle](evidence-lifecycle.md)
- [Authority Map](authority-map.md)
- [B70 Agent Operating System](agent-operating-system.md)
