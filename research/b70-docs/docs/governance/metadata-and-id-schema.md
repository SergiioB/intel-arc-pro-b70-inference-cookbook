---
title: "B70 Metadata and ID Schema"
status: living
type: governance
date: 2026-09-15
---

---
id: b70-gov-metadata-and-id-schema
title: B70 Metadata and ID Schema
document_type: governance
status: canonical
version: "1.0"
effective_date: 2026-08-19
last_reviewed: 2026-08-19
authority: docs/governance/documentation-standard.md
scope: Identifiers and YAML metadata for canonical B70 documentation and evidence references
---

# B70 Metadata and ID Schema

## Purpose

This schema gives canonical B70 documents stable identities and machine-readable lifecycle metadata. It covers Markdown documents and the references they make to campaigns, observations, claims, artifacts, and decisions. It does not rename existing result directories or retroactively require front matter in historical files.

## Canonical document ID

Format:

```text
b70-<domain>-<slug>
```

Rules:

- lowercase ASCII letters, digits, and hyphens only;
- begin with `b70-`;
- use one registered domain;
- use a descriptive, stable slug rather than a date or file extension;
- do not reuse an ID after a document is superseded or archived; and
- keep the ID stable when wording, version, or path changes.

Registered domains:

| Domain | Use |
|---|---|
| `gov` | Governance and authority |
| `method` | Validation and evaluation methods |
| `ref` | Stable or current reference material |
| `report` | Bounded reports and audits |
| `guide` | Operational guides |

Examples: `b70-gov-evidence-lifecycle`, `b70-method-correctness-validation`, and `b70-ref-engine-support-matrix`.

## Required document fields

| Field | Type | Requirement |
|---|---|---|
| `id` | string | Unique canonical document ID |
| `title` | string | Human-readable title matching the H1 |
| `document_type` | enum | `governance`, `method`, `reference`, `report`, or `guide` |
| `status` | enum | `canonical`, `provisional`, `superseded`, or `archived` |
| `version` | quoted string | Document schema/content version, for example `"1.0"` |
| `effective_date` | date | Date the current version became effective |
| `last_reviewed` | date | Most recent substantive review date |
| `authority` | path | Repository-relative controlling document |
| `scope` | string | Narrow statement of covered subject matter |

A canonical document must contain all fields. Optional metadata may be added only when a repository-wide consumer exists; do not create one-off keys that duplicate prose.

## Field semantics

### `status`

- `canonical`: current approved authority for its assigned subject.
- `provisional`: useful but not approved for normative or headline use.
- `superseded`: replaced by a named document or decision; retained for history.
- `archived`: retained as historical context and not maintained.

`invalid` is an evidence disposition, not a document status. A report describing invalid evidence may itself remain canonical.

### `version`

Use semantic intent without claiming software compatibility:

- increment the major component when requirements or field meanings change;
- increment the minor component for compatible clarification or expanded coverage; and
- keep the value quoted so YAML parsers do not coerce it to a number.

### `authority`

The path identifies the document that controls conflicts. A top-level normative authority may self-reference when no higher repository document exists. For this documentation set, the [B70 Benchmark Protocol](../methods/benchmark-protocol.md) controls benchmark validity and claim publication.

### Dates

Dates use `YYYY-MM-DD`. `last_reviewed` changes only after a substantive content and link review. Formatting-only edits do not imply evidence was revalidated.

## Evidence identifiers

Existing result directories remain authoritative and are not renamed. New references should preserve the recorded run identifier exactly.

Recommended logical forms:

| Entity | Form | Example meaning |
|---|---|---|
| Campaign | `<run_id>` | Existing manifest or result-directory ID |
| Observation | `<run_id>#<cell-or-observation-id>` | One measured cell or request |
| Claim | `claim:<document-id>:<slug>` | A reviewable prose claim |
| Decision | `decision:<document-id>:<slug>` | A durable technical decision |
| Correction | `correction:<run_id-or-claim-id>:<date>` | A dated correction event |

These are reference conventions, not instructions to rewrite historical JSON. When a source already defines an observation ID, use that exact ID instead of manufacturing a second one.

## Minimum metadata by entity

### Campaign

Campaign metadata follows [benchmark protocol §6](../methods/benchmark-protocol.md). At minimum it identifies protocol version, host and stack provenance, engine/build, exact model artifact and quantization, configuration, workload, cache and concurrency state, timing source, samples, power/thermal method, raw paths, exclusions, evidence tier, and gate status.

### Observation

An observation must be addressable within its campaign and retain:

- actual prompt tokens;
- requested and actual output tokens;
- C1 or requested/achieved Cn;
- cache state and relevant counter delta;
- metric class and timing source;
- raw value and unit;
- validity or exclusion status; and
- artifact path.

### Claim

A claim record or review note identifies:

- exact proposed wording;
- linked observations;
- statistic and scope;
- comparison class when applicable;
- truth verdict;
- evidence tier;
- publication gate status; and
- correction or supersession links.

### Decision

A decision record identifies context, decision, alternatives, evidence, limits, and triggers for reconsideration. The [quantization decision record](../reference/quantization-decision-record.md) is the canonical example for this documentation set.

## Path and link rules

- Metadata paths are repository-relative and omit a leading slash.
- Markdown links are relative to the containing document.
- Absolute local filesystem paths may appear only when they are part of recorded evidence or an operational command; they are not portable provenance by themselves.
- A local-only container tag, binary path, or model path must not be presented as a public reproduction identity. Use a pullable repository plus immutable digest or an exact source revision where required by the protocol.

## Compatibility and migration

This schema is prospective. Existing documents, manifests, and result directories remain unchanged. When an old artifact is cited, map its available fields in prose or a new index; do not mutate the historical artifact merely to satisfy this schema.

If a future schema changes a field meaning, increment this document's major version and document the mapping. Consumers must accept absent optional fields and must not infer unknown benchmark coordinates.

Executable situation indexes (`registry/research-ledger.json`, `registry/model-status.json`) use the ledger/model-status JSON shapes already in-tree. Markdown views are not a second identity namespace.

## Related authorities

- [Documentation Standard](documentation-standard.md)
- [Evidence Lifecycle](evidence-lifecycle.md)
- [Authority Map](authority-map.md)
- [B70 Benchmark Protocol](../methods/benchmark-protocol.md)
- [B70 Agent Operating System](agent-operating-system.md)
