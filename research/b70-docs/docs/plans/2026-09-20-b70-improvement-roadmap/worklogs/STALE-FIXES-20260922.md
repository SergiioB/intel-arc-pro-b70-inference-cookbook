---
title: "Stale-number fix application — 2026-09-22"
type: worklog
date: 2026-09-22
status: living
---

# Stale-fix application 2026-09-22

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Applies the fix list from the
read-only sweep `worklogs/STALE-SWEEP-20260922.md` (46 findings). No GPU
workload, docker, or server was run. No commit or push in either repo.

## Fix table (file:line, before -> after, finding id)

| # | file:line | before -> after | finding |
|---|---|---|---|
| 1 | `AGENTS.md:334` | "PROVISIONAL best 8.87 engine tok/s" headline -> "PROVISIONAL best **9.25 clean / 9.58 patched** engine tok/s (re-measured 2026-09-21; 8.87 superseded)" | #1 WRONG-NUMBER |
| 2 | `AGENTS.md:366` | Hybrid SSM+MoE row: bottleneck "Sequential Dispatch Lag (736 launches)" + lever "Fused Mamba2 SSD" -> bottleneck "MoE+GEMM dispatch (73% device time; fusable Mamba2/SSD only 8% — P2 WITHDRAWN 2026-09-21)" + lever "Small-M/grouped-GEMM (P5 into P3)" | #31 SUPERSEDED-METHOD |
| 3 | `AGENTS.md:340-342` | Duplicated architect-range paragraph (two identical consecutive lines) -> one copy kept | #44 MISSING-QUALIFIER |
| 4 | `engines/llamacpp-sycl/README.md:22` | "best 8.87 tok/s" -> "PROVISIONAL best 9.25 clean / 9.58 patched (2026-09-21 re-measure; 8.87 superseded)" + wall re-attributed to the 5.52x efficiency gap + device-ID +3.54 % gate-failed note | #3 WRONG-NUMBER |
| 5 | `engines/llamacpp-sycl/README.md:45` | Unqualified "IQ3/IQ4 omitted from fused SYCL MUL_MAT_ID MMVQ" -> qualified as true upstream at `1692f9e50`, FALSE for the pinned tree (device-ID cases present uncommitted) + 5.52x floor note | #4 SUPERSEDED-METHOD |
| 6 | `docs/plans/2026-09-20-b70-improvement-roadmap/P3-k216-grouped-iq3-prefill.md:23` | "gate/up 28.28 ms/op (1.82 TFLOPS), down 30.02 ms/op" + "86.6 ms/layer, 3.72 s" -> "**28.87 ms/op (1.79 TFLOPS)** gate-up / **29.91 ms/op** down (recapture `perf-k216-prefill-n512.txt` re-run 2026-09-21; supersedes 28.28/30.02)" + 87.7 ms/layer, 3.77 s | #13 WRONG-NUMBER |
| 7 | `docs/plans/2026-09-20-b70-improvement-roadmap/P4-qwen38-native-mtp-subgraph.md:28` | "TTFT unchanged" (p512 +46.0 % cell) -> "TTFT **308–315 -> 281–290 ms (~ −10 %; 'unchanged' corrected per the claims audit)**" | #18 WRONG-NUMBER |
| 8 | `research/2026/dsv4-k216-request-to-output-pipeline-20260825.md:20` | Best-no-spec table "8.87 engine tok/s (n=3)" -> same + "**SUPERSEDED 2026-09-21: 9.25 clean / 9.58 patched**" | #5 WRONG-NUMBER |
| 9 | `research/2026/dsv4-k216-request-to-output-pipeline-20260825.md:173` | "~84 blocking waits/token ... the queue never keeps depth" (sync-as-dominant framing) -> same + "(Mechanism framing corrected 2026-09-21: device-ID repair +3.54 %, gate FAILED — sync real but not dominant; fused op 5.52x above floor)" | #5 WRONG-NUMBER |
| 10 | `research/2026/dsv4-k216-request-to-output-pipeline-20260825.md:209` | L5 table "8.87 (max obs 9.27)" -> same + "SUPERSEDED 2026-09-21: 9.25 clean / 9.58 patched" | #5 WRONG-NUMBER |
| 11 | `research/2026/dsv4-k216-request-to-output-pipeline-20260825.md:247` | "worse than no-spec 7.26/8.87" -> same + "(current best 9.25/9.58)" | #5 WRONG-NUMBER |
| 12 | `research/2026/dsv4-k216-request-to-output-pipeline-20260825.md:286` | "8.87 tok/s ≈ 113 ms/token" -> same + "(2026-08-25 cell — SUPERSEDED 2026-09-21: 9.25/9.58 ≈ 104–108 ms/token)" | #5 WRONG-NUMBER |
| 13 | `research/2026/dsv4-k216-request-to-output-pipeline-20260825.md:35` | "Fused SYCL MUL_MAT_ID MMVQ omits IQ3_XXS" (unqualified) -> qualified: true upstream at `1692f9e50`, pinned tree carries device-ID cases uncommitted + 2026-09-21 +3.54 % note | #5 SUPERSEDED-METHOD |
| 14 | `research/2026/dsv4-k216-request-to-output-pipeline-20260825.md:213` | "Primary: IQ3 routed GEMV + host MUL_MAT_ID waits" -> "Primary: kernel efficiency (5.52x above floor) + waits — 2026-09-21: repair +3.54 %, waits real but secondary, kernel gap dominates" | #5 SUPERSEDED-METHOD |
| 15 | `research/2026/dsv4-k216-dspark-and-public-numbers-20260825.md:22` | "median 8.87 predicted tok/s" -> same + "SUPERSEDED 2026-09-21: 9.25 clean / 9.58 patched" | #6 WRONG-NUMBER |
| 16 | `research/2026/dsv4-k216-buun-moe-cache-20260825.md:25` | "best no-spec 8.87 engine tok/s (n=3, L5)" -> same + "SUPERSEDED 2026-09-21: 9.25 clean / 9.58 patched" | #7 WRONG-NUMBER |
| 17 | `research/2026/dsv4-k216-buun-moe-cache-20260825.md:52` | "vs L5 8.87" -> same + "(SUPERSEDED: 9.25/9.58)" | #7 WRONG-NUMBER |
| 18 | `research/2026/dsv4-k216-buun-moe-cache-20260825.md:86` | "against L5 8.87" -> same + "(SUPERSEDED 2026-09-21: 9.25 clean / 9.58 patched)" | #7 WRONG-NUMBER |
| 19 | `research/2026/dsv4-k216-prefill-ubatch-20260825.md:27` | L5 row "8.87" -> same + "(SUPERSEDED 2026-09-21: 9.25 clean / 9.58 patched)" | #8 WRONG-NUMBER |
| 20 | `research/2026/megakernel-xe2-xmx-architecture-guide-20260829.md:131` | "Target: 8.87 -> 20+ tok/s" -> `recompute-required` note naming the superseded input + re-derive from 9.25/9.58 + device-ID +3.54 % note | #9 WRONG-NUMBER |
| 21 | `research/2026/megakernel-implementation-roadmap-20260829.md:97` | "throttled at 8.87 tok/s" -> "throttled at 9.25 clean / 9.58 patched (2026-09-21; 8.87 superseded)" | #10 WRONG-NUMBER |
| 22 | `research/2026/megakernel-implementation-roadmap-20260829.md:148` | "from 8.87 -> 18–25+ tok/s" -> "from 9.25–9.58 -> 18–25+ (`recompute-required`: re-derive multiplier from the re-measure; device-ID +3.54 %, gate FAILED)" | #10 WRONG-NUMBER |
| 23 | `research/2026/megakernel-implementation-roadmap-20260829.md:200` | "8.87 -> 20+ tok/s (2.3x)" -> "9.25–9.58 -> 20+ (`recompute-required`: 2.3x premise needs re-basing on the fused-IQ3 kernel, not the sync repair)" | #10 WRONG-NUMBER |
| 24 | `research/2026/b70-improvement-probe-baseline-20260920.md:28` | "K216 L5 = 8.87 engine tok/s (n=3)" -> same + "**SUPERSEDED 2026-09-21 during this same phase: 9.25 clean / 9.58 patched**" (old figure preserved, not deleted, per correction policy) | #11 WRONG-NUMBER |
| 25 | `registry/research-ledger.json` (RL-010 evidence field, exact-string Edit) | evidence "research/2026/...pipeline-20260825.md" -> same + "(§16 AGENTS): 9.25 clean / 9.58 patched tok/s best (2026-09-21 re-measure; 8.87 superseded); device-ID repair +3.54 %, gate failed, withdrawn; needs fused IQ3 MMVQ" | #12 WRONG-NUMBER |
| 26 | `research/LEDGER.md:29` (RL-010 row, manual parity — no generator script exists) | "8.87 tok/s best, needs fused IQ3 MMVQ" -> "9.25 clean / 9.58 patched tok/s best (2026-09-21 re-measure; 8.87 superseded); device-ID repair +3.54 %, gate failed, withdrawn; needs fused IQ3 MMVQ" | #12 WRONG-NUMBER |
| 27 | `registry/model-status.json` (Nemotron open_gate, exact-string Edit) | "Fused Mamba2 SSD block kernel" -> "Fused Mamba2 SSD WITHDRAWN 2026-09-21 (P2; fusable family 8% vs 73% MoE+GEMM) — reassigned to small-M/grouped-GEMM (P5 folded into P3)" | #28 SUPERSEDED-METHOD |
| 28 | `models/STATUS.md:26` (Nemotron row) | "Open gate: Fused Mamba2 SSD block kernel (GRF/SLM)" -> "Fused Mamba2 SSD **WITHDRAWN** 2026-09-21 (P2; fusable 8 % vs 73 % MoE+GEMM) — reassigned to small-M/grouped-GEMM (P5→P3)" | #29 SUPERSEDED-METHOD |
| 29 | `models/STATUS.md:23` (Qwen3.8 row) | "83.7 (Run 40); 102.61 Run 48 C1" blended -> cache-state separation added (Run 40 cache-off vs Run 48 later prefix-on stack, GPU 0 — do not blend) | #30 MISSING-QUALIFIER |
| 30 | `models/STATUS.md:40` (external cross-check row) | Internal 102.61 sharing a row with external-lab cells -> "**internal record — different lab/host/engine/quant, never merged**" lab separation | #45 TAXONOMY |
| 31 | `engines/vllm-xpu/README.md:13,47` | "102.61 tok/s p512/g128" bare -> "client post-first, Run 48 recommended sampling, later prefix-on stack, GPU 0, n=5 — not comparable to cache-off arms" | #27/#32 MISSING-QUALIFIER |
| 32 | `engines/vllm-xpu/README.md:23-25` | "57–65 tok/s" / "45.7/32.6/34.5" bare -> metric class (client post-first) + n=3 / n=1-probe labels added | #33 MISSING-QUALIFIER |
| 33 | `docs/reference/engine-support-matrix.md:44` | "79 tok/s @512, single-card champion" bare -> "79 tok/s @512 client post-first, n=3, 2026-09-15 sweep" | #34 MISSING-QUALIFIER |
| 34 | `engines/openvino-genai/README.md:43` | "79.5 tok/s @512 (single-card best)" bare -> "79.5 tok/s @512 client post-first (n=3, 2026-09-15)" | #35 MISSING-QUALIFIER |
| 35 | `docs/operations/recommended-configs.md:31` | "186.61 tok/s at p2048/g128" bare -> client post-first + restricted re-measure pointer (252.30/178.55, tag absent, GPU/cap deviations) | #46 MISSING-QUALIFIER |
| 36 | `models/qwen/qwen3.8-27b/serving.md:40-42` (draft-INT4 table) | p512 row "52.83 -> 77.14 (+46.0 %), n=3" bare -> "actual 447 endpoint tokens (word-count client, uncalibrated — n=3 screen only)"; p8192-first row -> "actual 6993 endpoint tokens (uncalibrated)" | #24/#25 MISSING-QUALIFIER |
| 37 | cookbook `cookbook/docs/qwen38-27b/ENGINE-COMPARISON-CTX-20260914.md:8` | "OpenVINO's full-protocol ceiling is 128K" -> "verified full-protocol envelope is **≤163,840** (131,072 and 163,840 PASS 2026-09-21); **≥196,608 upstream-blocked** (per-generation growth: one long generation passes at 196608, second fails at 44 MiB free; #4483)" | #38 WRONG-NUMBER |
| 38 | cookbook `cookbook/docs/qwen38-27b/ENGINE-COMPARISON-CTX-20260914.md:227` | "128K on one card: no config fits ... OV blocked >96K" -> "128K–163,840: only llama Q4 holds the verified full-protocol envelope to 163,840 ... OV verified to 163,840, upstream-blocked at ≥196,608" | #39 WRONG-NUMBER |
| 39 | cookbook `cookbook/docs/qwen38-27b/ENGINE-COMPARISON-CTX-20260914.md:271` | "run to 128K (15.9 t/s), fail at ≥196K (~33 GB in use)" -> "verified to 131,072 and 163,840 (2026-09-21), fail at ≥196,608 (per-generation growth: second 64-token generation fails at 44 MiB free / 16 MiB largest block)" | #39 WRONG-NUMBER |
| 40 | cookbook `cookbook/docs/qwen38-27b/DRAFT-INT4-S-M1.md:70-76` | p512/g128 row bare + no TTFT note -> header "all figures are **client post-first rates** (not engine throughput), capture **on** (off = inert ≤1.5 %)"; p512 row "actual 447 endpoint tokens (uncalibrated — n=3 screen only)"; p8192-first row "actual 6993 (uncalibrated)"; "TTFT is not unchanged at p512: 308–315 -> 281–290 ms" | #26 MISSING-QUALIFIER |
| 41 | cookbook `cookbook/docs/qwen38-27b/DRAFT-INT4-S-M1.md:77-81` | graph-capture enabler note present -> **#54785 disclosure added** (2026-09-01, `intel-gpu`: XPU graph capture + MTP4 can produce non-deterministic wrong logits at temperature=0, k≤3 clean; our A/B internally consistent both-arms-on; reproducers should validate greedy determinism) + one `B70-DOCS/results/...` path pointer sanitized to a campaign reference | #13 NEW DISCLOSURE |

## Deliberately NOT fixed (with reason)

- **#14 OUTCOMES.md:24, #15 P3-20260921.md worklog, #16 ARCHITECT worklog, #19/#20 P4 worklog TTFT lines, ARCHITECT-next-levers-20260921c:35** — OFF-LIMITS per brief (OUTCOMES.md, LEVERS.md, GAINS-VS-CURRENT-DATA.md, patch-provenance note, and all worklogs handled elsewhere tonight). The 28.28/30.02 figures and the "28.28 ms = 49 GB/s" arithmetic remain live in exactly those excluded files; a repo-wide Markdown grep confirms no other non-excluded carrier of 28.28/30.02 exists (remaining hits are raw `results/` JSON evidence and the sweep/worklog records themselves).
- **#17 P4:25 "TTFT unchanged" (p8192 +11.0 % cell) and "inert at short context" framing** — the brief ordered only the :28 p512 correction; :25's TTFT claim is per-cell unverified (the audit proved the p512 move, not the p8192 one) and the phase file's own §28 + §29 entries already carry the corrected attribution, so a flat same-text edit would overstate. Flagged for the worklog owner.
- **#21 serving.md:218 greedy-k8 row, #22 cookbook `data/benchmarks.v1.json`, #23 comparison-page P10 annotation, #40–42 xmxlab vendored copies** — owner/vendor decisions pending (annotation wording, re-measure, re-vendor); out of tonight's authorization.
- **#36/#37 cookbook engine-merge tables/charts** — recorded taxonomy violations with no committed renderer for the split; hand-editing is not an option per the sweep.
- **#43 cookbook uncommitted-text publication risk** — commit/push explicitly forbidden tonight; the qualifier half is fixed (#40 above), the commit itself awaits the operator.
- **Residual 8.87 carriers not in the sweep's fix list:** `P1-k216-routed-gemv.md:41,93` (budget math + historical baseline), `DONE-P0-baseline-and-probes.md:39` ("2.25x" target math), `docs/plans/2026-09-20-b70-improvement-roadmap/README.md:249` (roadmap summary "K216: ~8.87"), `scripts/tmp/k216-dspark-{cpu-moe,probe}.sh` (historical campaign scripts). These restate the superseded baseline in historical/arithmetic contexts the brief did not authorize me to rewrite; the roadmap README line is the one live summary that still teaches 8.87 and should be picked up by whoever owns the roadmap files.
- **#10-adjacent `P5-smallm-int4-specialization.md:22`** — the "n=8 8.87 TFLOPS" there is a MUL_MAT TFLOPS reading that numerically coincides with the old tok/s figure, not a restatement of it; left untouched.
- **Cookbook `B70-DOCS` raw-evidence text pointers** (DRAFT-INT4-S-M1.md lines 66, 112, 155, 177, 193, 213; ENGINE-COMPARISON line 89) — pre-existing sanitization debt outside the three ordered cookbook edits; only the one pointer inside my edited block was sanitized. Flagged, not fixed.
- **research/LEDGER.md regeneration** — the brief ordered `python3 tools/build-indexes.py` after the JSON fix, but that tool only rebuilds `docs/indexes/*`, `registry/index.json`, and `DOCUMENT-MAP.md` (295 -> 344 documents, generated_utc refresh — routine catalog growth, no content edits). `research/LEDGER.md` has no generator (kept in manual parity per `docs/governance/agent-operating-system.md:151`), so the RL-010 row was edited directly to match the JSON. Unrelated generated-file churn from the rebuild: none beyond the catalog growth above.

## Gate outputs

Cookbook (`~/intel-arc-pro-b70-inference-cookbook`), full output scanned for the literal word FAIL (0 occurrences in all six):

- `python3 scripts/render-benchmark-catalog.py --check` -> `Catalog valid and current: 26 records`
- `python3 scripts/check-markdown-links.py` -> `OK: Markdown links (67 repository files)`
- `python3 scripts/check-repo.py` -> `12 pass / 0 fail` (4 suites)
- `python3 scripts/check-file-sizes.py` -> `OK: file sizes and line counts (310 repository files)`
- `python3 scripts/check-todos.py` -> `OK: no bare work-item markers outside hash-pinned patches/`
- `python3 scripts/check-stdlib-only.py` -> `OK: scripts/, tests/, benchmarks/ import stdlib or repo-local modules only`

B70-DOCS:

- `python3 tools/build-indexes.py` -> `wrote docs/indexes/research-index.md`, `wrote docs/indexes/doc-index.md`, `wrote registry/index.json`, `wrote DOCUMENT-MAP.md`, `OK: 344 documents catalogued`
- `python3 -c json.load` on both `registry/*.json` -> parse OK (Edit-tool exact-string edits, no re-encoding)

## Verification notes

- Recapture evidence read directly: `results/2026/09/b70-improvement-probes-20260920/perf-k216-prefill-n512.txt` shows 28867.39 us (28.87 ms, 1.79 TFLOPS) gate-up and 29910.35 us (29.91 ms, 1.72 TFLOPS) down at n=512 — the replacement values check out; nothing contradicted them. Per-layer math updated accordingly (2 x 28.87 + 29.91 = 87.65 ≈ 87.7 ms; x43 = 3.77 s of the 5.06 s ubatch).
- OV envelope values verified against `P6-long-context-gates.md:24` (131,072 + 163,840 verified) and the roadmap README:43 (per-generation growth: single long generation passes at 196608, second fails at 44 MiB free).
- Nothing in the sweep's replacement values was contradicted by anything read during application.
