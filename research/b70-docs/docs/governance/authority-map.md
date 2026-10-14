---
title: "B70 Documentation Authority Map"
status: living
type: governance
date: 2026-08-08
---

---
id: b70-gov-authority-map
title: B70 Documentation Authority Map
document_type: governance
status: canonical
version: "1.0"
effective_date: 2026-08-19
last_reviewed: 2026-08-24
authority: docs/governance/documentation-standard.md
scope: Precedence and ownership of canonical B70 documentation subjects
---

# B70 Documentation Authority Map

## Purpose

This map resolves conflicts among B70 documents. It assigns one canonical authority to each subject and identifies supporting sources. A supporting document may provide evidence or operational detail, but it must not silently override the assigned authority.

Agent load order is defined by [B70 Agent Operating System](agent-operating-system.md). That document does not override measured evidence or the benchmark protocol; it overrides *which files an agent opens first*.

## Precedence rules

When two repository statements conflict, apply this order:

1. raw evidence for what a recorded observation contains;
2. the [B70 Benchmark Protocol](../methods/benchmark-protocol.md) for measurement validity and public claims;
3. the canonical governance, method, or reference document assigned below;
4. dated audits and campaign reports for their bounded scope;
5. current guides and recommendations;
6. research notes and plans;
7. archived material.

A newer date does not by itself create authority. A canonical decision can be changed only by updating or explicitly superseding its controlling document.

## Subject map

| Subject | Canonical authority | Supporting sources and limits |
|---|---|---|
| Benchmark validity, campaign lanes, metric definitions, statistics, comparison gate, evidence tiers, publication gate, corrections | [B70 Benchmark Protocol](../methods/benchmark-protocol.md) | [Standards audit](../../archive/legacy-docs-20260915/benchmark-standards-audit-2026-08-08.md) explains the vLLM tooling decision but does not override the protocol |
| Documentation structure, sourcing, terminology, links, and review | [Documentation Standard](documentation-standard.md) | Existing guides remain content sources, not taxonomy authorities |
| Agent context compile, accretion, constellation boundary | [B70 Agent Operating System](agent-operating-system.md) | Implementation spec: [situation compiler plan](../plans/2026-09-01-agent-situation-compiler.md). JSON ledger/status are executable; Markdown views are commentary |
| Public XeCores hub vs deferred intelmaxxing.com plans | xmxlab `AGENTS.md` + `DESIGN-AGENT-SYSTEM.md` | `docs/plans/intelmaxxing-com-architecture-plan.md` and `docs/plans/intelarcmaxxing-hub-plan.md` are tombstones |
| Document metadata and stable identifiers | [Metadata and ID Schema](metadata-and-id-schema.md) | Prospective; historical files are not renamed |
| Evidence state, retention, validation, correction, and supersession | [Evidence Lifecycle](evidence-lifecycle.md) | Protocol hard rules and publication gate remain controlling |
| Correctness test scopes and reporting | [Correctness Validation](../methods/correctness-validation.md) | Speed evidence never substitutes for correctness evidence |
| Model-weight and KV quantization evaluation procedure | [Quantization Evaluation](../methods/quantization-evaluation.md) | [KV cache reference](../reference/kv-cache-quantization.md) provides existing measured analysis |
| Power, thermal, VRAM, cache, spec-counter, and progress observability | [Observability Standard](../methods/observability-standard.md) | Protocol §§1, 6, 9, and 12 remain normative |
| Stable host hardware and required stack provenance fields | [Hardware and Software Stack](../reference/hardware-software-stack.md) | [Current Host State](../operations/current-host-state.md) is the dated volatile snapshot; live checks remain required |
| Engine capability and evidence status | [Engine Support Matrix](../reference/engine-support-matrix.md) | Dated engine reports show bounded observations, not permanent support |
| Current quantization decisions and reconsideration triggers | [Quantization Decision Record](../reference/quantization-decision-record.md) | [Current recommended configurations](../operations/recommended-configs.md) contains qualified per-model route choices |
| Chronological campaign index | [Campaign Index](../indexes/campaign-index.md) plus the [Benchmark History compatibility view](../../archive/legacy-docs-20260915/benchmark-history.md) | `registry/campaigns.yaml` contains canonical promoted campaigns; the archived full log covers history not yet promoted; `results/` remains the measurement trail |
| Current per-model operational recommendations | [Current Recommended Configurations](../operations/recommended-configs.md) and the relevant model dossier | Must remain within protocol-qualified evidence and current artifact availability |
| Launcher behavior and operational commands | [Operational Launcher Guide](../operations/launcher-guide.md) | The old `archive/legacy-docs-20260915/compat-stubs/launcher-guide.md` path is a compatibility tombstone only; verify live installed configuration before use |
| Storage safety, current mounts, and model placement | [Storage and VRAM Safety](../operations/storage-and-vram-safety.md) plus [Current Host State](../operations/current-host-state.md) and live `df`/`findmnt` | The old `archive/legacy-docs-20260915/compat-stubs/lvm-and-storage.md` path is a compatibility tombstone; remembered free-space values are never authoritative |
| Dual-B70 PCIe topology and link widths | [X570-F dual-B70 topology](../../research/2026/x570f-dual-b70-pcie-topology-20260830.md) (supersedes [B450-F topology](../../research/2026/b450f-dual-b70-pcie-topology-20260824.md) | Live `lspci` / debugfs remain required after GPU 1 power cycles |
| Dual-card serving product (independent stacks vs TP2 vs llama split) | [ADR-0008](../../decisions/adr-0008-dual-b70-independent-default.md) | [humble-b70-llm capture](../../research/2026/humble-b70-llm-dual-tp2-20260824.md) is external E0 recipe evidence, not a local campaign or public headline |

## Evidence versus interpretation

Raw artifacts are authoritative for recorded fields, timestamps, samples, and outputs. They are not self-interpreting. The protocol and methods determine whether a field supports a metric or claim.

Examples:

- A response timing field can be authentic raw evidence yet unsuitable for a vLLM TTFT claim if the required client timing source is absent.
- A successful server launch can be authentic evidence of load success yet not evidence of a context-capacity completion.
- A coherent response can support `coherent_output_smoke_tested` yet not token or logit parity.
- A leaderboard record can prove external publication yet not independent reproduction.

## Current versus historical state

The repository contains both historical measurements and live operational state. Use these rules:

- The run manifest controls what existed for that campaign.
- The dated [Current Host State](../operations/current-host-state.md) snapshot plus live checks control what is available now; `AGENTS.md` supplies the operating contract, not volatile inventory.
- A deleted model or local-only image does not invalidate its historical observation, but it cannot remain an unqualified current recipe.
- A new build does not erase an older run; it supersedes the older run only for decisions whose material coordinates changed.

## Conflict procedure

When a conflict is found:

1. identify the exact statements and their subjects;
2. apply the precedence rules;
3. inspect the linked raw evidence where a measured fact is disputed;
4. classify the affected claim with a truth verdict;
5. add a correction or supersession under the [evidence lifecycle](evidence-lifecycle.md); and
6. update the canonical authority and all dependent public surfaces.

Do not “resolve” a conflict by deleting historical evidence or averaging incompatible numbers.

## Authority maintenance

Any new canonical document must be added to this map before it can displace an existing authority. Changes to authority require a version update, a reviewed rationale, and validation of all dependent links.
