---
title: "Cascadia on B70 — engine dossier"
type: engine-dossier
status: living
date: 2026-09-15
updated: 2026-09-28
engine: cascadia
---

# Cascadia — living dossier

**Cascadia is a separate inference engine from OpenVINO GenAI** (Rust
server from Community Labs wrapping the C++ GenAI API plus its own IR
surgery runtimes — `labscommunity/cascadia`). It consumes the same
OpenVINO IR artifacts but is a different process model, engine set, and
measured performance class. Never fold Cascadia numbers into OpenVINO
GenAI rows or vice versa.

## Current state (as of 2026-09-28)

- Protocol-grade on this host: 2026-09-11 bakeoff — `qwen35` surgery
  runtime, greedy batch=1, **4.95 tok/s P512/G128 @ ~102 W** on
  Qwen3.8-27B INT4-GDN8 (stock 0.2.3 + bundled OV 2026.2; `--ov-*`
  ignored). Evidence: `research/2026/ov-cascadia-qwen38-bakeoff-20260911.md`
  + `results/2026/09/ov-cascadia-qwen38-20260911/` (Sept bucket by
  campaign date).
- The engine class moved ~5× on newer builds: the fleet lab's patched
  build (0.2.4-base + `--ov-*` passthrough, OV 2026.5.0.0beta1, official
  int4-ov IR) runs the same 27B at **23.78 tok/s qwen35 / 30.4 tok/s
  ov-genai** single-card (Sergio's 2026-09-21/22 fleet session on the
  Rock-orchestrated lab, portfolio-published). Binary × OV × artifact
  all differ from the bakeoff row — never mix the two classes in one
  table or claim "Cascadia got 5× faster" without naming which variable.
- The `ov-genai` engine VLM SIGSEGV (bundled OV 2026.2; 2026.3.1 rebuild
  dies in `Tokenizer::setup_tokenizer`) is **fixed by the OV 2026.5
  bump** — upstream PRs labscommunity/cascadia#160 (release/CI URL bump)
  and #161 (shim `core.add_extension`), both ours, both still open.
- Meanwhile Python `VLMPipeline` + grafted MTP5 reached 79 tok/s @512
  (`engines/openvino-genai/README.md`) and the vLLM champion is
  ~103–107 tok/s — Cascadia's best single-card remains 2.5–3.5× behind.
- Fleet lab authority: `~/projects/cascadia/` (Rock 5B; `STATUS.md` =
  master state, `KNOWLEDGE-BASE.md` = measured log, `experiments/` =
  dated evidence). Separate registry from this repo — anything feeding
  cookbook/ledger claims must be re-measured or registered here (RL-051
  pattern).

## Where it is the right tool

- Multi-node pipeline parallel across Intel boxes (`--rank/--total/--next`)
  — the only engine here with a shard-tree story.
- `cascadia doctor` for silent-CPU-fallback detection; one static binary,
  no Python on workers.
- Family surgery runtimes (`qwen35` MoE/dense split) without re-quant.

## Not a B70 speed path today

Single-card Qwen3.8 decode is ~2.6× behind OpenVINO GenAI+MTP5 (30.4 vs
79) and ~3.4× behind vLLM GPTQ-Int4+MTP4 (30.4 vs 106.7) on the lab's
best build. Revisit only if a fused INT4 GDN GEMV lands upstream (the
OV 2026.5 jump already closed most of the gap once), if `ov-dist-spec`
proves out, or if the multi-node pipeline-parallel use case becomes real.

## 0.2.4 cross-check (2026-09-26, external-but-ours data, zero GPU)

Sergio's published three-machine session ([cascadia-multi-gpu-penalty](
https://sergiiob.dev/posts/cascadia-multi-gpu-penalty/), 2026-09-22) adds two facts this
dossier lacked, both consistent with RL-051:

- **llama.cpp RPC layer-split on the same two B70s retains 54 %** (76.1 → 40.8 tok/s).
  First measured B70 number for the standing "never RPC-split the B70 pair" rule.
- **Cascadia 0.2.4 two-GPU pipeline retains 101 %** (24.02 vs 23.78 tok/s single-card,
  relay 0.249 ms/frame). RL-051 already measured ≈100 % on 0.2.3 at p1692/g128 with
  0.284–0.293 ms/frame — the 0.2.4 relay change moves nothing that matters on this host.
- Two independent single-GPU servers: 47.8 tok/s aggregate — matches ADR-0008's default.

Decision unchanged: Cascadia stays a capacity/multi-node tool here. It cannot load
`qwen4exp` (Flash-Next) and is 3–4.5× behind the vLLM single-card champion at
single-stream decode. No 0.2.4 upgrade campaign warranted.

## Rules

- Registry/engine taxonomy: `cascadia` is its own engine id; results files
  must label engine `cascadia (qwen35|ov-genai|ov-runtime)`, never
  `openvino`.
- Cookbook architecture page:
  `intel-arc-pro-b70-inference-cookbook/docs/architecture/openvino-and-cascadia.md`.
- Security: the worker probe/discovery listener binds `0.0.0.0:9101`
  plaintext even with `--listen 127.0.0.1` (RL-051) — firewall it on any
  untrusted network; no CLI flag pins it.
- Parallel track: `--elastic` memory posture (upstream #132 + our #166
  conformance suite) — certified anti-OOM for co-tenant CPU nodes; B70
  titration/conformance evidence lives in `~/projects/cascadia/
  experiments/elastic-20260922/` (does not touch GPU/USM memory).
- Upstream direction to watch: 0.2.4 Inkling expert-parallel engine
  (975B MoE, `--ep-workers`), t8's #159 pipeline micro-batches, #113
  Unix-socket activation links, `feat/tensor-parallel`.
