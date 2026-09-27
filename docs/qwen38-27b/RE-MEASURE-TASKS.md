# Qwen3.8-27B — named re-measure tasks (P10 decisions, 2026-09-23)

These are named, pending re-measurements decided in the P10 close-out
(decisions recorded 2026-09-23). They are execution items, not open questions:
each affected catalog record stays live and annotated until a like-for-like
replacement lands, because deleting a registered number would violate the
correction policy. New results become new generations with commit-pinned
evidence; the old records remain as evidence.

## 1. vLLM FP8 TP2 — like-for-like p1024/g128 re-measure

- Target record: `qwen38-fp8-tp2-k8-p1024-g128-c1` (60.13 client e2e tok/s,
  greedy diagnostic median n=5, dual B70 TP2, max-model-len 9216, 230 W cap),
  annotated superseded-in-place for ranking purposes.
- Why: the 2026-09-15 context sweep (73.0 tok/s at 512 down to 57.3 at 96K,
  n=3, Pi-prompt protocol, max-model-len 100000, record
  `qwen38-27-fp8-tp2-ctx-sweep-512-96k`) supersedes it only directionally.
  It is not a like-for-like p1024/g128 cell, and no protocol-grade
  replacement exists on disk.
- Exact cell: C1, p1024/g128, cold (disabled-zero-hit), context 9216, same
  image and patch stack, greedy diagnostic median n=5 for parity with the
  banked cell; add a model-card-sampling arm (non-thinking temp 0.7 /
  top_p 0.8 / top_k 20, presence 1.5) if the route keeps serving claims.
- Done when: a fresh n=5 benchmark record with full workload coordinates and
  commit-pinned evidence takes over the ranking reuse. The old record then
  carries a plain superseded-by pointer to the new record id.

## 2. OpenVINO GenAI INT4-GDN8 — earn the benchmark class

- Target record: `qwen38-27-openvino-genai-int4gdn8-v1` (capability).
- Missing for `kind: benchmark`: measured prompt-token accounting per cell
  (the sweep recorded configured lengths and character counts), a declared
  prefix-cache state per cell, per-cell sample counts at long context
  (98K/128K/163,840 are n=1 ladder cells), and raw evidence reachable from a
  commit-pinned URL that carries the numbers.
- Cell list: p512/g128 and p8192/g128 decode n=5 fresh samples (Lane-1
  default), plus the 98K / 131,072 / 163,840 ladder cells with
  tokenizer-accounted prompts (n=3 preferred).
- Route context: serving shape is the composite (MTP5 draft to 65,536, then
  the no-MTP u4-KV arm from 98,304). Contexts of 196,608 and beyond stay
  upstream-blocked (openvino.genai#4483, per-generation growth) — do not
  spend GPU time on them until the upstream fix lands; re-measuring them is
  out of scope for this task.

## 3. Cascadia — earn the benchmark class

- Target record: `qwen38-27-cascadia-v1` (capability).
- Missing for `kind: benchmark`: measured prompt tokens, a pullable upstream
  Cascadia build identity (the measurement used a local rebuild against
  OpenVINO 2026.3.1), a declared cache state, and per-cell coordinates
  committed with the raw samples.
- Cell list: repeat the n=5 grid (P128/G128, P512/G128, P1024/G128,
  P512/G64, P512/G256) at the 230 W cap with token accounting and the
  quality check retained.
- Taxonomy: Cascadia is a separate engine. Its rows and series never merge
  with OpenVINO GenAI (or OVMS, llama.cpp, vLLM) in tables or charts.

## Rules that bound all three

- One declared engine and one client at a time; cold-input cells keep unique
  leading entropy and a zero prefix-cache-hit delta; always declare context.
- Research work runs under the declared cap (230 W) and restores 150 W
  afterwards; cooldown to the declared package threshold between cells.
- New benchmark records need exact workload coordinates, sample counts,
  metric semantics, and commit-pinned evidence — `data/benchmarks.v1.json`
  validation enforces the schema, and `docs/BENCHMARK-CATALOG.md` is
  generated from it.
