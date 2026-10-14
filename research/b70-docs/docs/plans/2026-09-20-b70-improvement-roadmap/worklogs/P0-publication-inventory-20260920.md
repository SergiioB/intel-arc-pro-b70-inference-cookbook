---
title: "P0 — Publication backlog inventory (read-only)"
type: worklog
status: final
date: 2026-09-20
updated: 2026-09-20
phase: P0
---

# P0 — Publication backlog inventory (2026-09-20)

Read-only audit. No GPU, no model load, no docker. Every claim below cites a file path;
values are copied from files, never recomputed. `UNKNOWN` = data not found (searches listed in §8).

Sources of authority:
- Private evidence: `~/B70-DOCS/results/2026/09/` and `research/2026/`.
- Public catalog: `~/intel-arc-pro-b70-inference-cookbook/data/benchmarks.v1.json`
  (the only public benchmark-data authority — `docs/governance/agent-operating-system.md:49,117`).
- Flow: `~/B70-DOCS/docs/governance/public-benchmark-publication.md`,
  `~/B70-DOCS/.agents/skills/publish-b70-benchmark-results/SKILL.md`.
- `registry/index.json` is a document index (86 KB), not the benchmark catalog; no B70-DOCS-side
  catalog exists. Sync tool: cookbook `cookbook:scripts/render-benchmark-catalog.py`.

## 1. Engine dossiers — state and last recorded revision

| Engine | What it is | Last recorded revision/digest | Dated evidence references |
|---|---|---|---|
| llama.cpp SYCL | One-card GGUF workhorse; only 262K-validated engine; dual-card layer split | SYCL flashnext `52d4268` (verified 2026-09-15 sweep). Other builds in catalog: `1692f9e50` (build 361, quant showdown), `d2f83055d` (Muse) | `engines/llamacpp-sycl/README.md:17-24,37-38`; `cookbook:results/2026/09/qw38-ov-mtp-20260914/ctx-sweep-llamacpp{,-mtp,-q4km-mtp}.json`; `results/2026/09/qw38-quant-format-showdown-20260913/manifest.json` |
| vLLM XPU | Serving leader (GPTQ-Int4 + MTP4); long ctx in TP2 | `vllm 0.27.2rc1 XPU`; in-container `0.27.2rc1.dev77+gac7509e2b`; pinned image digest `sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f` | `engines/vllm-xpu/README.md`:13-29; `models/qwen/qwen3.8-27b/serving.md`:106; `registry/model-status.json`:12 |
| OpenVINO GenAI | Newest engine; single-card short-ctx champion (INT4-GDN8 + grafted MTP5, 79.5 tok/s @512) | openvino `2026.5.0-22967-38a575de45f`, genai `2026.5.0.0-3412-8fa12145144` (venv `/mnt/<models-disk>/.ov-genai-venv`) | `engines/openvino-genai/README.md`:18-25; `results/2026/09/qw38-ov-mtp-20260914/SUMMARY.md`:3 |
| Cascadia | Separate Rust engine (Community Labs) + IR surgery (`qwen35`); distinct engine id | stock `0.2.3`; local rebuild vs OpenVINO `2026.3.1` (`libopenvino.so.2631`). Upstream commit: **UNKNOWN** | `engines/cascadia/README.md`:21-26; `research/2026/ov-cascadia-qwen38-bakeoff-20260911.md`:18 |
| OVMS | Referenced only in `archive/`; no current dossier | UNKNOWN | `archive/legacy-docs-20260915/b70-engine-sweep-results-2026-07-05.md`:67-160 |

## 2. Catalog status (what exists and is published/audited)

- File: `~/intel-arc-pro-b70-inference-cookbook/data/benchmarks.v1.json`
  (`schema_version 1.0`, `updated_at 2026-09-15T09:25:00Z`). Generated page: `cookbook:docs/BENCHMARK-CATALOG.md`.
- Read-only validation run 2026-09-20: `python3 -B scripts/render-benchmark-catalog.py --check`
  → `Catalog valid and current: 26 records` (exit 0).
- 26 records = **vllm 14, llama.cpp 12, OpenVINO GenAI 0, Cascadia 0**.
- Tiers: 20 `benchmark` (19 `official-lab`, 1 `community-reviewed`) + 6 `capability`
  (3 `provisional`, 1 `validated`, 2 `official-lab`).
- There is no explicit "audited" field; the published/audited proxy is
  `kind: benchmark` + `evidence_status: official-lab` (trust labels:
  `cookbook:scripts/render-benchmark-catalog.py:184-190`). Capability records are non-ranking by construction.
- All 20 benchmark records carry P/G/C, n/statistic, and a full-SHA evidence URL (machine-checked; §3).

## 3. Machine-checkable publication requirements (checklist)

From `docs/governance/public-benchmark-publication.md:44-55,59-69,77-85,95-103` and
`cookbook/scripts/render-benchmark-catalog.py:17-110`:

1. [ ] `schema_version == "1.0"`; `records` non-empty array.
2. [ ] `id` matches `^[a-z0-9][a-z0-9-]{2,100}$` and is unique.
3. [ ] `kind` ∈ {`benchmark`, `capability`}.
4. [ ] `evidence_status` ∈ {`official-lab`, `community-reviewed`, `validated`, `provisional`}.
5. [ ] Required keys: `title`, `summary`, `hardware`, `engine`, `evidence`.
6. [ ] `evidence.commit` is a full 40-hex SHA; `evidence.url` contains `/blob/<commit>/`.
7. [ ] Benchmark only: `workload.concurrency` ∈ {1,2,4,8,16,32}; `prompt_tokens`, `output_tokens`
   positive ints; non-empty `metrics` with finite numeric `value`, `name`, `unit`;
   `statistics.statistic` ∈ {`median`,`mean`,`single-observation`}; `statistics.samples` positive int.
8. [ ] Capability records must NOT carry `metrics` (numbers go in `summary`).
9. [ ] Catalog text free of private patterns (`/home/`, `B70-DOCS`, `192.168.x.x`, `api_key`/`bearer`).
10. [ ] Private record complete: exact model/revision, hardware, engine image/build/kernel/driver,
    quant/KV/speculation/parallelism/context/scheduler/power cap, prompt/output targets and actual
    token counts, concurrency, warmups/samples/failures/statistic, raw samples + telemetry +
    cleanup, correctness gate, claims audit (`public-benchmark-publication.md:44-55`).
11. [ ] n policy: n=3 is provisional screening; n=5 supports representative public claims
    (`docs/methods/benchmark-protocol.md:150-153,183-185`).
12. [ ] Power: configured cap stored separately from measured draw; `power_cap_W` is a configured
    value, never a measured draw (`benchmark-protocol.md:390,405,455`; catalog `hardware.power_cap_w`).
13. [ ] Sanitized public evidence, full-SHA pinned; no private result-root paths
    (`public-benchmark-publication.md:77-85`).
14. [ ] Renderer `--check` + catalog unit tests + `git diff --check` pass before push
    (`public-benchmark-publication.md:95-103`).
15. [ ] Human gates: claims audit + Sergio authorization; only approved rows public
    (`public-benchmark-publication.md:22,156`). **D1** is referenced at line 156 but not defined
    in `docs/governance/` or `AGENTS.md` — UNKNOWN.

## 4. Gap table — engine × model × cell with on-disk evidence

Catalog record? = record in `data/benchmarks.v1.json` matching engine+model+cell (not merely same engine).
Paths are relative to `~/B70-DOCS/` unless absolute.

| Engine | Model | Cell (pp/tg) | Evidence root path | Catalog? | n | Sampling | Power cap | Blocker to publication |
|---|---|---|---|---|---|---|---|---|
| OpenVINO GenAI | Qwen3.8-27B INT8-OV | p512/g128 MTP nat5, 71.01 post-first | `results/2026/09/qw38-ov-mtp-20260914/nat5-n5.json` | no | 5 | greedy | 230 W (`SUMMARY.md:6`; file has no cap field) | No record; campaign marked PROVISIONAL (`SUMMARY.md:98`); needs claims audit + sanitized public evidence |
| OpenVINO GenAI | Qwen3.8-27B INT8-OV | p512/g128 MTP4, 61.56 wall | `results/2026/09/qw38-ov-mtp-20260914/probe6-mtp-parity.json` | no | 1 per prompt stage | greedy | 230 W (`cap_uw=230000000`) | n=1 per cell; this is the dossier headline (`engines/openvino-genai/README.md`:21) |
| OpenVINO GenAI | Qwen3.8-27B INT4-GDN8 | p512/g128 + grafted MTP5, 79.53 | `results/2026/09/qw38-ov-mtp-20260914/ctx-sweep-ov-int4-gdn8-mtp5.json` | no | 3 (≤64K); 1 @98K (crash) | v2 single-stream greedy | 230 W per `SUMMARY.md:6`; file lacks cap field (mean_w 202-207) | No record; ≤64K speed arm only; OV long-ctx crash documented upstream |
| OpenVINO GenAI | Qwen3.8-27B INT4-GDN8 | p512→p131072 no-MTP u4 full protocol (22.4→15.9) | `results/2026/09/qw38-ov-mtp-20260914/ctx-sweep-ov-int4-gdn8-u4-full.json` | no | 3 ≤64K; 1 at 98K/131K | v2 single-stream | as above | n=1 beyond 64K; the one-config serving default is unpublished |
| OpenVINO GenAI | Qwen3.8-27B INT4-GDN8 | 196K/262K capacity probes (40 tok) | `.../ctx-sweep-ov-int4-gdn8-u4-max.json`, `.../ov-u4-maxctx.json` | no | 1 | completion probe | UNKNOWN (no cap field) | Capacity-only; must never become a ranking cell (`SUMMARY.md:80-82`) |
| OpenVINO GenAI | Qwen3.8-27B INT4-GDN8 (pre-graft) | p128/p512/p1024 n=5 (15.14/15.79/15.25) | `results/2026/09/ov-cascadia-qwen38-20260911/n5-ov-int4-gdn8-230w.json` | no | 5 | greedy | file lacks `cap_w`; 230 W stated in cookbook `cookbook:docs/qwen38-27b/OPENVINO-CASCADIA.md:77` | Superseded by MTP5; public prose already exists (`OPENVINO-CASCADIA.md:92-98`) |
| Cascadia | Qwen3.8-27B INT4-GDN8 (`qwen35`) | p512/g128, 4.95 tok/s (measured 102.1 W) | `results/2026/09/ov-cascadia-qwen38-20260911/n5-cascadia-int4-gdn8-230w.json` | no | 5 | greedy | 230 W (`cap_w` in file) | No record; must not merge with OV rows; upstream commit UNKNOWN; single-node number, multi-node untested |
| Cascadia | Qwen3.8-27B | `ov-genai` engine load fails (SIGSEGV) | `results/2026/09/ov-cascadia-qwen38-20260911/SUMMARY.json` | no | — | — | — | Negative capability only; no ranking |
| llama.cpp SYCL | Qwen3.8-27B UD-Q4_K_M + draft-MTP | p512→p262144 decode (37.81→15.95) | `results/2026/09/qw38-ov-mtp-20260914/ctx-sweep-llamacpp-q4km-mtp.json` | no | 3 ≤64K; 1 ≥98K | v2 single-stream greedy | 230 W stated in engine string; measured mean_w 152-160 | Only engine serving full 256K single-card; prefill rows cache-contaminated; long cells n=1 |
| llama.cpp SYCL | Qwen3.8-27B Q8_0 + draft-MTP | p512→p65536 decode (52.0→38.3) | `results/2026/09/qw38-ov-mtp-20260914/ctx-sweep-llamacpp-mtp.json` | no | 3 | single streaming request | 230 W stated; measured ~99-103 W | No record; prefill column INVALID (prompt-cache hits) |
| llama.cpp SYCL | Qwen3.8-27B Q8_0 no-spec | p512→p131072 | `results/2026/09/qw38-ov-mtp-20260914/ctx-sweep-llamacpp.json` | no | 5 (≤8K)/3 | greedy | `cap_uw` recorded | v1 sweep; not in catalog |
| llama.cpp SYCL | Qwen3.8-27B Q4_K_M | cold prefill (210-250 tok/s) | `results/2026/09/qw38-ov-mtp-20260914/llama-q4km-cold-prefill.json` | no | 1 | cold, no cache-prompt | UNKNOWN (no cap field) | Incomplete observation |
| vLLM XPU | Qwen3.8-27B GPTQ-Int4 MTP4 | p512→p98304 single-card v2 (75.6→51.1) | `results/2026/09/qw38-ov-mtp-20260914/ctx-sweep-vllm-v2.json` | no (catalog p512 record uses a different Pi-prompt protocol) | 3 | v2 single-stream | 230 W (`corrections`), mean_w 195-229 | No catalog record for the curve; v1 defective |
| vLLM XPU | Qwen3.8-27B GPTQ-Int4 | 98K/131K/196K ceiling probes (45.7/32.6/34.5) | `results/2026/09/qw38-ov-mtp-20260914/ctx-sweep-vllm-int4-262k-probe.json` | no | 1 | probe | UNKNOWN (no cap field) | n=1 |
| vLLM XPU | Qwen3.8-27B FP8 TP2 | p512→p96K (73.0→57.3) | `results/2026/09/qw38-ov-mtp-20260914/ctx-sweep-vllm-fp8-tp2.json` | yes `qwen38-27-fp8-tp2-ctx-sweep-512-96k` | 3 | v2 single-stream | 230 W (record) | Published; record encodes only the P98392 top point |
| vLLM XPU | Qwen3.8-27B FP8 TP2 | p1024/g128, 60.13 | cookbook `cookbook:results/qwen38-27-fp8-tp2-k8-n5/` | yes `qwen38-fp8-tp2-k8-p1024-g128-c1` | 5 | greedy diagnostic | 230 W (record) | Stale flag — see §6.1 |
| vLLM XPU | Qwen3.8-27B INT8 W8A16 TP2 | launch failed (OOM util 0.97) | `.../vllm-int8-attempt.json`, `.../ctx-sweep-vllm-int8-tp2.json` | no | 0 | — | 230 W attempt | Blocker is the open OOM; nothing to publish |
| llama.cpp SYCL | Qwen3.8-27B GGUF Q4_K_M/Q5_K_M/Q8_0 | llama-bench p512/p8192 (r=5) | `results/2026/09/qw38-quant-format-showdown-20260913/manifest.json` | yes (3 records) | 5 | mean (llama-bench) | 150 W (record) | Published |

## 5. Incomplete observations (look like results; cannot support a ranking record)

| # | Artifact | What it looks like | Missing provenance |
|---|---|---|---|
| 1 | `.../qw38-ov-mtp-20260914/ov-u4-maxctx.json` | 40-token capacity probes 131K→262K (11.28-14.72 tok/s) | n=1; no power field; capacity proof only |
| 2 | `.../ctx-sweep-ov-int4-gdn8-u4-max.json` | 196K/262K completion probes | n=1 (`warmup + n=1`); no cap field |
| 3 | `.../ov-ceiling-confirm.json` | MTP 98K 4.77 tok/s; 131K crash | step record, n=1; no cap field |
| 4 | `.../ov-nomtp-ceiling.json` | 98K/196K 40-token probes | n=1; 40-token |
| 5 | `.../llama-q4km-cold-prefill.json` | cold prefill rates | n=1; no cap field |
| 6 | `.../ctx-sweep-vllm-int4-262k-probe.json` | 98K/131K/196K decode | n=1; no cap field |
| 7 | `.../ctx-sweep-vllm-int8-tp2.json` | TP2 sweep grid | `server_error`; empty grid |
| 8 | `.../vllm-int8-attempt.json` | launch attempt | failed status (OOM); no samples |
| 9 | `.../ctx-sweep-vllm.json` | v1 ctx sweep | defective (decode incl. TTFT; idle-card watts); superseded by v2 (`ENGINE-COMPARISON-CTX-20260914.md:12-18`) |
| 10 | `cookbook:results/2026/09/ov-cascadia-qwen38-20260911/ovpy-{latency,throughput}-f16.json` | 29.8/29.6 tok/s decode | quality FAIL (garbage token id 0); superseded |
| 11 | `results/2026/09/ov-cascadia-qwen38-20260911/SUMMARY.json` entries 1-2 | 29.8 / 5.27 tok/s | `quality: false` (token id 0); superseded by n5 files |
| 12 | `.../qw38-ov-mtp-20260914/mtp-gate/int4-gdn8-mtp5.json` | graft gate run | `status: ERROR` (`sampler_num_threads`); `samples: []` |
| 13 | `.../ctx-sweep-llamacpp-mtp.json` prefill column | prefill up to 88K tok/s | INVALID — server prompt-cache hits (`ENGINE-COMPARISON-CTX-20260914.md:134-137`) |
| 14 | `.../ctx-sweep-llamacpp-q4km-mtp.json` prefill + ≥98K rows | prefill 1.9K-114K tok/s; long-ctx decode | prefill cache-contaminated; long cells n=1 |

Count: 14 artifacts — 12 fully non-ranking, 2 partially contaminated (decode usable, prefill not).

## 6. Records whose numbers could be stale relative to newer evidence

1. `qwen38-fp8-tp2-k8-p1024-g128-c1` (60.13 tok/s, n=5) — cookbook
   `cookbook:docs/qwen38-27b/ENGINE-COMPARISON-CTX-20260914.md:174-177` declares the banked E2 cells
   (53.42/60.13/33.52) "superseded by this same-protocol sweep for ranking purposes" (sweep: n=3,
   73.0@512 → 57.3@96K). The superseded benchmark row is still in the catalog.
2. `qwen38-fp8-w8a16-v1` (capability, 48 tok/s) — summary self-declares
   "Superseded by qwen38-fp8-tp2-k8-p1024-g128-c1" (`data/benchmarks.v1.json` record summary);
   stays in the catalog as provisional capability.
3. Same cell, two figures: catalog `qwen38-27-draft-int4-p512-g128-c1` = 106.7 tok/s (n=5, updated
   2026-09-15) vs `registry/model-status.json`:12 = 102.61 "Run 48 sampling" (file dated 2026-09-14).
   Catalog is newer; the discrepancy should be reconciled before either number is reused.
4. Same model, two Q4_K_M pictures: catalog `qwen38-27-gguf-q4km-v1` = 20.01 tok/s (llama-bench,
   no spec) vs newer `ctx-sweep-llamacpp-q4km-mtp.json` = 37.81@512 (draft-MTP). Different protocols;
   not stale, but easily misread.
5. OV INT8 dossier headline 61.6 (MTP4, n=1 per prompt) vs `nat5-n5.json` 71.01 post-first (n=5).
   A catalog cell would need one canonical configuration.

## 7. Engine-merge risk (OpenVINO GenAI / Cascadia / OVMS must never merge)

Rule: `docs/methods/chart-standard.md`:69-70, `AGENTS.md:180-182`,
`docs/plans/2026-09-20-b70-improvement-roadmap/P10-cookbook-completeness.md`:40-42,
`engines/cascadia/README.md`:13-17.

Live risks found (public cookbook):
- `~/intel-arc-pro-b70-inference-cookbook/docs/qwen38-27b/OPENVINO-CASCADIA.md:92-98` —
  head-to-head table places OpenVINO GenAI, Cascadia, and llama.cpp rows in one table; chart embed at line 20.
- `~/intel-arc-pro-b70-inference-cookbook/docs/qwen38-27b/OPENVINO-CASCADIA-REPORT.md:61-62` —
  same one-table OV/Cascadia rows; charts embedded at lines 14-16.
- `~/intel-arc-pro-b70-inference-cookbook/docs/assets/b70-qwen38-ov-cascadia-headtohead.svg` —
  single chart titled "Qwen3.8-27B OpenVINO vs Cascadia vs llama.cpp on one B70" with series
  "OpenVINO GenAI 2026.5" and "Cascadia qwen35 2026.3.1"; `...-n5.svg` and `...-hints.svg` likewise.
- Compliant (separate rows): cookbook `cookbook:docs/qwen38-27b/README.md:16-17`; `engines/README.md`:16-23.
- OVMS: no current evidence or catalog footprint; appears only in `archive/legacy-docs-20260915/`.
  No live merge risk found for OVMS.

## 8. UNKNOWNs and what was searched

- Cascadia upstream commit/digest — searched `engines/cascadia/README.md`, bakeoff doc, repo-wide
  `grep labscommunity|commit`; only version `0.2.3` / OV `2026.3.1` recorded.
- OVMS revision/status — no dossier; only `archive/legacy-docs-20260915/b70-engine-sweep-results-2026-07-05.md`.
- `D1` row definition (`public-benchmark-publication.md:156`) — not defined in `cookbook:docs/governance/*.md` or `AGENTS.md`.
- Power caps for capacity probes (`ov-u4-maxctx.json`, `ctx-sweep-vllm-int4-262k-probe.json`,
  `llama-q4km-cold-prefill.json`) — no cap field; UNKNOWN.
- Claims audit for the engine sweep — no `cookbook:research/2026/*claims*` / `*audit*` file found;
  `SUMMARY.md:98` states "All cells PROVISIONAL — NOT FOR PUBLIC HEADLINE until claims audit".
- Empty evidence root: `results/2026/09/qwen38-llamacpp-mtp-20260914/` contains no files.
