---
title: "P10 apply-ready spec (verified 2026-09-21)"
type: plan
date: 2026-09-21
status: final
---

# P10 apply-ready spec (2026-09-21)

Read-only verification of `P10-gap-plan-20260920.md` against the real
cookbook (`~/intel-arc-pro-b70-inference-cookbook`) and B70-DOCS
evidence. No GPU, no commits, no cookbook edits made.

## 1. Stale record: confirmed, no like-for-like replacement exists

`data/benchmarks.v1.json:317` (`id` line; value at `:332`) holds
`qwen38-fp8-tp2-k8-p1024-g128-c1`: kind `benchmark`, title
"Qwen3.8-27B FP8 TP2 MTP8 decode", summary "Dual B70 TP2, W8A16 reroute +
custom Xe2 small-M kernel + native MTP8. Greedy diagnostic median (n=5);
model-card sampling is in the evidence doc. p1024/g128 C1.",
`evidence_status: official-lab`, workload C1 / p1024 / g128 / ctx 9216 /
`disabled-zero-hit`, metrics `client_e2e_output_tps: 60.13` (+ max 71.06),
statistics median n=5, hardware 2x B70 `power_cap_w: 230`, engine
`vllm 0.27.2rc1.dev77+gac7509e2b`, evidence commit `8c69e5f5...`.
Superseding prose `cookbook:docs/qwen38-27b/ENGINE-COMPARISON-CTX-20260914.md:174-177`:
"Earlier banked E2 cells (53.42/60.13/33.52 hybrid, synthetic prompts,
max-model-len 9216) are superseded by this same-protocol sweep for ranking
purposes". That sweep (73.0@512 to 57.3@96K, n=3, different Pi-prompt
protocol) is NOT a like-for-like p1024/g128 correction. B70-DOCS
`research/2026/qwen38-fp8-tp2-speed-attack-and-ceilings-20260829.md`:216
(all-time FP8 48.89 p512 / 53.83 p1024) and `:275` (best 56.85 p1024 k=8)
likewise offer no same-protocol n=5 p1024/g128 value. Conclusion: record
stands as banked E2 evidence; ranking reuse needs a fresh n=5 re-measure.
Schema has NO superseded/status field (validator only checks `kind` and
`evidence_status` enums); precedent for marking is prose inside `summary`
(record `qwen38-fp8-w8a16-v1`, line 296: "Superseded by
qwen38-fp8-tp2-k8-p1024-g128-c1 ..."). Exact annotation wording is an owner
decision, so it is NOT in APPLY LIST.

## 2. Taxonomy violations: confirmed, but no ready-to-apply edit

(a) `cookbook:docs/qwen38-27b/OPENVINO-CASCADIA.md:92-98`: one table headed
"Head-to-head (same IR, 230 W, P512/G128 n=5)" with rows
"OpenVINO GenAI 2026.5 | 15.79 | 183", "Cascadia `qwen35` 2026.3.1 | 4.95
| 102", "llama.cpp SYCL Q4_K_M | 18.58 tg128 | 150 W run". Fix = split
into per-engine tables; new table text must be authored, not mechanical.
(b) `cookbook:docs/qwen38-27b/OPENVINO-CASCADIA-REPORT.md:59-63`: same merged
Backend table ("OpenVINO GenAI 2026.5 ... 15.79 ... 183" /
"Cascadia qwen35 ... 4.95 ... 102" / "llama.cpp SYCL ... 18.58 ...
150 W run"). Same split needed.
(c) `cookbook:docs/assets/b70-qwen38-ov-cascadia-headtohead.svg` (plus `-n5`,
`-hints`): one canvas, three engine panels as text nodes
"OpenVINO GenAI 2026.5" / "Cascadia qwen35 2026.3.1" /
"llama.cpp SYCL Q4_K_M" under title "Qwen3.8-27B backends". Fix = one
canvas per engine question, generated from evidence JSONs by a committed
renderer per chart-standard. No such renderer exists in the cookbook
(`scripts/` + `tools/` grep for headtohead/ov-cascadia/SVG: empty; only
`render-benchmark-catalog.py`). Chart-standard's reference renderer lives
in B70-DOCS (`scripts/tmp/ov-mtp-20260914/render-ctx-sweep-svg.py`) and
does not cover this chart. So there is NO renderer command to quote.
(d) `cookbook:docs/qwen36-35a3/MOE-INT4-THREE-ENGINE-CEILING-20260917.md:22-26`:
rows carry distinct engine+artifact labels; compliant KEEP, no action.
OVMS appears only in `archive/` + boundary prose: no action.

## 3. Validator: passes; required fields verbatim

`python3 scripts/render-benchmark-catalog.py --check` (cookbook root)
prints `Catalog valid and current: 26 records`, exit 0. Required fields
verbatim from `cookbook:scripts/render-benchmark-catalog.py:17-110`: `schema_version
== "1.0"`; `records` non-empty array; `id` matches
`^[a-z0-9][a-z0-9-]{2,100}$`, unique; `kind` in {`benchmark`,
`capability`}; `evidence_status` in {`official-lab`,
`community-reviewed`, `validated`, `provisional`}; required keys `title`,
`summary`, `hardware`, `engine`, `evidence`; `evidence.commit` full
40-hex SHA; `evidence.url` contains `/blob/<commit>/`; benchmark-only:
`workload.concurrency` in {1,2,4,8,16,32},
`workload.prompt_tokens`/`output_tokens` positive ints, non-empty
`metrics` each with finite numeric `value` + `name` + `unit`,
`statistics.statistic` in {`median`,`mean`,`single-observation`},
`statistics.samples` positive int; capability records MUST NOT carry
`metrics` (`:104-105`); catalog text free of `/home/`, `B70-DOCS`,
`192.168.x.x`, `api_key`/`bearer`.

## 4. OV/Cascadia zero-record gap (catalog: vllm 14, llama.cpp 12, OV 0, Cascadia 0)

Cascadia best cell `results/2026/09/ov-cascadia-qwen38-20260911/
n5-cascadia-int4-gdn8-230w.json`: top keys only `engine
cascadia-qwen35-ov2631`, `cap_w 230`, `runs` (p/g grid, n=5,
`median_wall_tps` 4.95@512, `completion_tokens` 128, wall timing,
telemetry, quality PASS). Missing for `benchmark`: measured
`prompt_tokens` (no prompt-token accounting in file), sanitized cookbook
evidence commit + `/blob/<commit>/` URL, upstream commit UNKNOWN.
`capability` (NO metrics) needs: `title`, `summary`, `hardware`, `engine`,
`evidence` with 40-hex commit + blob URL — evidence commit still missing.
OV best cell `results/2026/09/qw38-ov-mtp-20260914/nat5-n5.json`
(INT8 MTP nat5, 71.01 post-first): `chars` only (143/741), no prompt
tokens; cap only via `SUMMARY.md:5`; cache state unrecorded. Same verdict:
`benchmark`-blocked; `capability` still needs the evidence commit/URL.
Do not transcribe numbers lacking coordinates.

## 5. Gap-plan corrections

Gap plan §4 row 2 (106.7 vs 102.61 "same cell") is wrong: serving doc
`models/qwen/qwen3.8-27b/serving.md`:38-39 shows 106.7 = Run 47 greedy
diagnostic vs 102.61 = Run 48 recommended non-thinking sampling —
different sampling, not a conflict; label, don't reconcile by re-measure.
Gap plan §6 item 2 implies re-rendering the three SVGs via a committed
renderer; none exists in the cookbook (verified empty grep) — renderer
must be written first. The 60.13 summary annotation (§6 item 1) is
schema-legal (summary prose, precedent line 296) but exact wording is
undecided, so it stays out of APPLY LIST.

## APPLY LIST

1. file: `cookbook:docs/qwen38-27b/OPENVINO-CASCADIA.md` (cookbook) → old:
`Measured P512/G128 n=5 median at 230 W, same INT4-GDN8 1-stage shard:` →
new: `Measured P512/G128 n=5 median at 230 W cap, same INT4-GDN8 1-stage
shard:` → verify: `grep -n "median at 230 W," docs/qwen38-27b/
OPENVINO-CASCADIA.md` returns nothing; `--check` unaffected (prose-only).
2. file: `cookbook:docs/qwen38-27b/OPENVINO-CASCADIA.md` (cookbook) → old:
`14.65 tok/s at 150 W / 73 °C` → new: `14.65 tok/s at 150 W cap / 73 °C`
→ verify: `grep -rn "at 150 W /" docs/qwen38-27b/` returns nothing.
3. file: `cookbook:docs/qwen38-27b/ENGINE-COMPARISON-CTX-20260914.md` (cookbook) →
old: `llama.cpp decode is NOT power-bound: at 230W decode stays 15.2` →
new: `llama.cpp decode is NOT power-bound: at 230W cap decode stays 15.2`
→ verify: `grep -rn "at 230W decode" docs/qwen38-27b/` returns nothing.

---

## DECISIONS TAKEN 2026-09-23 (closing O6 — recorded so they are not re-litigated)

Per the standing "decide rather than defer" rule, the three items above are decided:

1. **The stale `qwen38-fp8-tp2-k8-p1024-g128-c1` = 60.13 record: STAYS LIVE, annotated as
   superseded-in-place.** Rationale: no protocol-grade like-for-like p1024/g128 replacement exists,
   and deleting a registered number would violate the correction policy (preserve old results, mark
   them). The summary text carries the supersession banner (applied in the 2026-09-22 stale-fix
   pass). A re-measure at that exact cell is now a **named task**, not an open question. Owner
   wording refinement (if any) remains Sergio's, but the record is no longer ambiguous.

2. **The OV/Cascadia catalog gap: classify as capability records only, no ranking metrics.** The
   on-disk n=5 evidence lacks the full ranking coordinates the `kind: benchmark` class requires, and
   capability records may carry no `metrics`/`statistics` field. Decision: transcribe them as
   capability records in the next cookbook edit with the numbers in prose only, and file a re-measure
   task to earn the benchmark class. This matches the engine-taxonomy rule (OV GenAI / Cascadia /
   OVMS never merged with llama.cpp or vLLM).

3. **The three merged OV/Cascadia/llama chart SVGs: no renderer exists, so the fix is NOT a
   hand-edit.** Decision: write a committed renderer first (chart-standard-compliant, generated
   from evidence JSONs, one engine per series, log-x context axis, legend panel, PROVISIONAL banner,
   vision-QA PASS) and only then regenerate. Until that renderer lands the three SVGs carry a
   **taxonomy-violation note** in the pages that embed them (the lab-separation note added in the
   2026-09-22 stale-fix pass covers the pages; the SVGs themselves are flagged in
   `worklogs/STALE-SWEEP-20260922.md` finding #36/#37). Renderer authoring is queued as P10 work.

Net P10 state after these decisions: **no open question remains that is merely waiting for a
decision** — what remains is execution (re-measure at p1024/g128, OV/Cascadia capability
transcription + re-measure for the benchmark class, chart renderer authoring).
