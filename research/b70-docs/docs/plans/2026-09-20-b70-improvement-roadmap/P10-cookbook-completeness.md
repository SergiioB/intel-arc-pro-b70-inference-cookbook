---
type: plan
date: 2026-09-21
phase: P10
title: "Cookbook completeness and release workstream"
status: RESEARCH
opened: 2026-09-20
updated: 2026-09-20
ledger: RL-TBD-P10   # only for genuine research findings; doc-only work uses the normal tracker
depends_on: ["P0 inventory", "engineering winners known"]
effort: incremental per phase + 2–4 final sessions
---

# P10 — Cookbook completeness and release workstream

## Status log

| Date | Status change | Note / evidence |
|---|---|---|
| 2026-09-20 | filed as PLANNED | Roadmap filed from b70-architect draft |
| 2026-09-20 | PLANNED → RESEARCH | Gap audit executed (read-only): catalog holds 26 records (vLLM XPU 14, llama.cpp SYCL 12); **OpenVINO GenAI and Cascadia have 0 records** despite on-disk evidence; 1 live stale record (`qwen38-fp8-tp2-k8-p1024-g128-c1` = 60.13, superseded) plus 2 reconcile-before-reuse; 5 engine-taxonomy violations (merged OV/Cascadia/llama series in 2 pages and 3 SVGs) that the 2026-09-15 standard forbids. Deliverable: `worklogs/P10-gap-plan-20260920.md` (ordered worklist with a verification command per edit). **No edits made to the cookbook** — this phase's own rule is that a record cannot be transcribed into existence. |
| 2026-09-21 | first edits applied (working tree only, no commit/push) | The three unambiguous power-claim-hygiene edits from `worklogs/P10-apply-ready-20260921.md` are **applied and verified** in the cookbook working tree: `cookbook:docs/qwen38-27b/OPENVINO-CASCADIA.md` ("at 230 W," → "at 230 W cap,"; "150 W / 73 °C" → "150 W cap / 73 °C") and `cookbook:docs/qwen38-27b/ENGINE-COMPARISON-CTX-20260914.md` ("at 230W decode" → "at 230W cap decode"). All three verification greps return empty and `python3 scripts/render-benchmark-catalog.py --check` still reports `Catalog valid and current: 26 records`. Diff is 3 lines in 2 files — **nothing committed or pushed**; the operator reviews before release. Deliberately NOT applied: the stale `qwen38-fp8-tp2-k8-p1024-g128-c1` (60.13) annotation (wording is an owner decision and no protocol-grade like-for-like replacement exists — a re-measure is required), and the OV/Cascadia table/chart splits (they need authored text and a committed renderer that does not exist in the cookbook). |
| 2026-09-21 | two items blocked on operator wording, one on a missing renderer | (a) **stale record `qwen38-fp8-tp2-k8-p1024-g128-c1` (60.13)** — confirmed stale at `cookbook:data/benchmarks.v1.json:317/332`, but no protocol-grade like-for-like replacement exists, so the honest fix needs an owner decision on the annotation wording (or a re-measure); left untouched. (b) **OV/Cascadia catalog records** — the best cells lack required coordinates (no measured prompt tokens, no sanitized evidence commit), so they cannot become `benchmark` records without a minimal approved re-measure, and `kind: capability` records may not carry metrics. (c) **the three merged OV/Cascadia/llama charts** — the cookbook has **no committed renderer** for them (only `render-benchmark-catalog.py`), so the split must be authored and rendered from evidence JSONs per the chart standard; hand-editing the SVGs is not an option. All three are recorded rather than forced. |

## Why this decision (the reasoning)

**The thesis: the cookbook is the public artifact, its two known gaps are OV/Cascadia
catalog coverage and power-claim hygiene, and both have a *right* way and a *wrong* way
to close — the wrong way is exactly how stale numbers have historically survived in
this ecosystem.**

1. **Why the OV/Cascadia gap is real but cannot be transcribed shut.** The engine sweep
   (2026-09-14/15) lives only in
   `cookbook:docs/qwen38-27b/ENGINE-COMPARISON-CTX-20260914.md` + raw JSONs in B70-DOCS;
   `data/benchmarks.v1.json` has no OV/Cascadia rows. But a benchmark record requires
   full coordinates (prompt/output counts, sampling, concurrency, cache state,
   n/statistic, timing source) and a full-SHA sanitized evidence URL — converting a
   prose table into authoritative JSON by transcription is precisely the failure mode
   the renderer and claims standard exist to prevent. Missing coordinates ⇒ no ranking
   record; a `kind: capability` record is allowed only under its own (stricter,
   no-metrics) rules. Some cells may need a minimal approved re-measurement — that is
   the honest cost of the gap.
2. **Why engine taxonomy is non-negotiable.** OpenVINO GenAI, Cascadia, and OVMS are
   separate engines (2026-09-15 standard; Cascadia's 4.95 tok/s bakeoff number must
   never share a table row or chart series with OV GenAI's 79 tok/s). The catalog is
   XeCores' live data source; a merged row there becomes someone else's purchase
   decision.
3. **Why SGLang stays blocked until evidence moves.** The support matrix's decision
   rules: a repository name, open PR, or successful conversion does not change support
   status. Reassess only on a pinned upstream revision/image that demonstrably changes
   the specific recorded blocker (architecture registration, quant representation, XPU
   kernel path, required ops) *and* passes a minimal correctness probe. A successful
   server startup is not a benchmark or quality result.
4. **Why the power page refresh is a claims exercise, not a numbers dump.** The
   observability standard already separates configured cap, measured average draw,
   peak interval-average draw, and energy per token. The known failure modes to
   avoid: using configured cap as measured draw in "tokens per watt" (the 150→230 W
   A/B exists precisely because cap ≠ draw), and generalizing the flat MoE-prefill
   power result to dense llama.cpp decode. Note the operator's current standing cap
   policy (230 W allowed for research) — the page must present caps as configured
   values with measured draw beside them.
5. **Why dossier obligations ride on every phase.** The display-refresh order
   (renderer → charts → dossiers → cookbook pages → release) is the anti-stale-numbers
   machine; skipping steps is how the 15.8 tok/s stale number survived until the
   2026-09-15 sweep. This phase owns the checklist; each engineering phase owns its
   own dossier update at close-out.

**Alternatives considered and rejected.**

- *Bulk-transcribe the engine-comparison doc into the catalog now.* Rejected — see 1;
  the renderer rejects capability records carrying metrics for a reason.
- *Publish engine-sweep numbers as PROVISIONAL rows.* The catalog has no
  PROVISIONAL-state row; the banner convention is for B70-DOCS evidence docs, and the
  audit gate is what upgrades them. Weakening the gate to fill the catalog is banned.
- *Leave OV/Cascadia out entirely.* Rejected: single-card long-context and the
  composite curve are the cookbook's differentiated content; the gap is worth the
  audit cost.

**What would change this decision.** XeCores/catalog schema gaining a sanctioned
provisional evidence state (re-open the classification); OV GenAI or Cascadia being
retired from the fleet (drop those rows, keep the retirement note).

## Objective

Make every validated route reproducible and every public number traceable without
collapsing distinct engines or promoting incomplete evidence. Starts with P0's
inventory; closes after engineering winners are known.

## 10.1 OV GenAI and Cascadia catalog gaps

Current representation only in:
`~/intel-arc-pro-b70-inference-cookbook/docs/qwen38-27b/ENGINE-COMPARISON-CTX-20260914.md`
(raw JSONs in B70-DOCS `results/2026/09/qw38-ov-mtp-20260914/`). Catalog target:
`~/intel-arc-pro-b70-inference-cookbook/data/benchmarks.v1.json`.

Steps:

1. Recover exact raw evidence and manifests.
2. Classify each observation: benchmark-eligible, capability-only, or insufficient.
3. Separate OpenVINO GenAI, Cascadia, and OVMS engine identities.
4. Recover exact prompt/output counts, sampling, concurrency, cache state,
   n/statistic, timing source.
5. Preserve composite context-switching rules (MTP5 ≤64K → no-MTP ≥98K).
6. Do not convert a prose table into authoritative JSON by transcription.
7. Re-measure missing coordinates only under an approved minimal campaign.
8. Create benchmark records only after claims audit and sanitized immutable public
   evidence.

**Stop rule:** missing ranking coordinates ⇒ no ranking record.

Mechanical constraints of record (from the publish-flow memory): one-edit flow —
sanitize evidence doc → commit → pin SHA in records → `render-benchmark-catalog.py`
→ `--check` → unit tests → `git diff --check` → PR; `kind: capability` records cannot
carry `metrics`/`statistics` (numbers go in summary text); release-please ignores
`chore:`/`docs:`/`charts:` commits; both checkouts on master before vendoring.

## 10.2 SGLang reassessment trigger

Reassess only when a pinned upstream revision or image demonstrably changes the
specific blocker: architecture registered; exact quant representation supported; XPU
kernel path present; required recurrent/MoE/speculative ops implemented; minimal
correctness probe passes. Then the standard card.

## 10.3 Power sweet-spot refresh

Separate configured cap / measured average draw / peak interval-average draw / energy
per successful generated token / workload and engine. Bounded screen (150/180/230 W)
where justified, matched cooldown, synchronized `energy1_input` intervals, fresh n=5
finalists. No "tokens per watt" from configured cap; no cross-workload
generalization.

## 10.4 Dossier obligations after every phase

Verified paths at filing:

- `engines/llamacpp-sycl/README.md`
- `engines/vllm-xpu/README.md`
- `engines/openvino-genai/README.md`
- `engines/cascadia/README.md`
- The affected model's canonical README and serving document.

Each update states: exact revision and artifact; what changed; quality scope;
supported workload range; rejected/blocked cases; link to immutable evidence; whether
the recipe is upstream, patched research, or public-ready.

## 10.5 Charts and release sequence

1. Generate from evidence JSON through a committed renderer.
2. Compute statistics from raw runs.
3. Fixed engine taxonomy and chart standard; log-x context axes; solid/dashed/dotted
   protocol semantics; sparse labels; legend below.
4. Validate SVG/XML; render PNG.
5. Vision-QA PASS on the rendered PNG.
6. Update engine/model dossiers.
7. Update authorized cookbook records and pages.
8. Catalog renderer + `--check` + unit tests.
9. Sanitized full-SHA evidence links.
10. Obtain separate release authorization.
11. Push/release only when authorized.
12. Refresh XeCores fallback/provenance before its next authorized deployment.

No hand-edited XeCores benchmark numbers.

## Deliverables

- `research/2026/b70-cookbook-completeness-audit-<YYYYMMDD>.md`
- `results/2026/<MM>/b70-cookbook-claims-audit-<UTC>/`

## Stop rules

Stop publication on any claims, privacy, chart-QA, schema, or reproducibility failure.
Preserve rejected records privately; never weaken requirements to fill the catalog.
