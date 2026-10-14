---
title: "Claims audit — P4 draft-INT4 pair and P1 K216 device-ID repair (2026-09-21)"
type: plan
date: 2026-09-21
status: final
---

Independent claims audit against `.agents/skills/b70-benchmark-claims-standard/SKILL.md` and
`docs/methods/benchmark-protocol.md` v3.0. Read-only; arithmetic recomputed from raw client logs.

## Verdict table

| Claim | Status | Evidence gap | Minimum fix |
|---|---|---|---|
| P4: +37.2 % @ C1 p8192/g128 fresh n=5, champion stack, "decode throughput" | **provisional** | cell mislabeled (actual 6993 tokens); metric is client post-first, not engine decode; acceptance scalar is warmup-diluted; patches applied outside anchored generation and outside greedy validity window; no thermal/power telemetry | see fixes 1–5 below |
| P4: +46.0 % @ C1 p512/g128 n=3 | **provisional (screen-class only)** | n=3 screen cannot back a representative number (protocol §1.9 lane 2); "TTFT unchanged" contradicted; measured-window acceptance drop −0.048 breaches the patch's own 0.03 gate | fresh n=5 confirmation at calibrated token count before any public wording |
| P1: device-ID repair worth +3.54 % engine decode (9.25 → 9.577), below ≥10 % gate | **auditable-today as "patched tree vs clean export"; provisional as "the repair is worth +3.54 %"** | patched build carries 283 insertions across 9 modified + 3 untracked files, not only the mmvq device-ID cases; no retained temperature log; parity = backend-ops smoke, not token parity | single-variable rebuild (clean + mmvq.cpp cases only) rerun n≥5, or relabel the wording; retain cooldown/temps |

## Result 1 — P4 draft-side INT4 pair (champion stack)

Verified correct: same image digest `f01e24f6…`, overlay, `VLLM_XPU_ENABLE_XPU_GRAPH=1`,
`B70_MTP_BF16_DRAFT=1` in both arms (`scripts/tmp/p4-champion-p8192-ab-20260921.sh`:90-110);
fresh container per arm; ≥31,000 MiB gate per arm (`:86-88`); same-shape warmup discarded
(`:127-128`); prefix caching off and *proven* off — `enable_prefix_caching: False` in both
server.log:12 and `prefix_cache_queries/hits` 0→0 across repeated identical prompts (the
resident-prefix zero-hit test); 128/128 server tokens every sample; single serial client;
recommended sampling matches Run 48. Medians recompute: 72.68/52.99 = +37.2 %,
77.14/52.83 = +46.0 % — arithmetic matches. p8192 n=5 distributions disjoint (55.73 < 67.49).

Defects:

1. **Cell coordinate mislabeled.** Actual endpoint prompts are 6993 tokens (p8192 cell) and 447
   (p512 cell): `client.log` `prompt_tokens_server`, both arms. The client sizes prompts by word
   count (~0.75 w/t) and never calibrates (`scripts/tmp/p4-mtp-client.py`:55-61), violating
   protocol §11.1. "C1 p8192/g128" (manifest.txt:10, worklog §12, headline) is a requested, not
   actual, coordinate — arms matched, so the delta stands, but the label is wrong and blocks
   coordinate-matched comparison with the recorded champion cells.
2. **Metric naming overstates.** `post_first_tps = out_tokens/(total−ttft)` computed client-side
   (`scripts/tmp/p4-mtp-client.py`:119). The protocol requires the label "client post-first tok/s"
   for vLLM decode (benchmark-protocol.md:233-234; AGENTS §8). "Decode throughput" without that
   qualifier implies an engine-native number never captured; the worklog tables say "post-first"
   but the verdict line (`worklogs/P4-20260920.md:416-417`) and headline say "decode".
3. **Acceptance evidence is warmup-diluted and direction-unstable.** The script reads only
   metrics-after cumulative totals (`scripts/tmp/p4-champion-p8192-ab-20260921.sh`:156-161), so
   the scalar includes the discarded warmup. Recomputed measured-window (after−before):
   p8192 447/800=0.5588 → 445/792=0.5619 (**up** +0.003); p512 267/480=0.5563 → 258/508=0.5079
   (**down** −0.048). Reported: −0.007 and −0.009. So (a) "flat-to-slightly-down" is an artifact
   of warmup inclusion — in the measured window the two cells move in opposite directions, meaning
   acceptance at ±2 pp is noise at these n; (b) the p512 measured-window drop exceeds the patch
   header's own gate ("si la aceptacion baja >0.03…", `patches/patch_draft_mtp_int4.py:16-17`).
   The throughput delta is not acceptance-driven (acceptance down while tok/s up), so the speed
   claim survives, but the acceptance characterization does not.
4. **"TTFT unchanged" is false at p512.** Baseline 308.4–315.4 ms vs draftpair 281.0–289.9 ms
   (`comparison.json` ttft arrays; all three draft samples below all three baseline samples, ≈
   −10 %). The worklog compresses both arms to "~0.4 s" (`worklogs/P4-20260920.md:399-400`). At
   p8192 TTFT is genuinely flat (4.00–4.06 s both arms).
5. **Patches applied outside their anchored generation.** Both patch headers state anchors
   verified against `2c427ef` (vLLM 0.26.1.dev457)
   (`patch_draft_lmhead_int4.py:23-24`, `patch_draft_mtp_int4.py:32-33`); the champion run runs
   them on `f01e24f6` with observed vLLM `0.27.2rc1.dev77+gac7509e2b` (server.log:28). AGENTS §7:
   only patches anchored to the exact source generation.
6. **Sampling sits outside the patch's stated lossless window.** Headers: emitted sequence equals
   target "SOLO en C1/TP1 (greedy)" (`patch_draft_lmhead_int4.py:13`,
   `patch_draft_mtp_int4.py:13-14`). Runs use temp 0.7/top_p 0.8/top_k 20/penalty 1.5 — correct
   per the sampling policy for representative numbers, but the patches' losslessness argument was
   never demonstrated under stochastic verification, and no greedy parity/token check exists in
   the evidence.
7. **No measured power or thermal; uncounterbalanced order.** Configured cap 230 W only
   (manifest.txt:8; burst tier, "short bursts only", protocol §4); no energy counters or
   temperature samples in either run dir; no cooldown gate in the P4 script (teardown + `sleep 3`
   only, `:135-136`); baseline always first (`:140-141`). Thermal drift would if anything
   understate the gain, and three independent A/Bs agree in direction (local p8192 +11.0 %,
   champion n3 +24.2 %, champion n5 +37.2 %), but drift is not excluded. The 230 W pre-run cap
   also means the host was not at the contracted 150 W rest state (AGENTS §3).

## Result 2 — P1 K216 device-ID repair

Verified correct: **ABBA is real** — call order clean-r1, patched-r1, patched-r2, clean-r2
(`scripts/tmp/p1-k216-e2e-ab-20260921.sh`:151-154), each arm a fresh load with teardown, VRAM
re-gate, and cooldown to ≤55 °C package (`:104-147`); `engine_tg` is the engine's own
`timings.predicted_per_second` (`scripts/tmp/p1-k216-client.py`:84); the client rate is a separate
field, never in the delta; unique entropy-first nonce + `cache_prompt=false` + greedy in every
request (`p1-k216-client.py:59,66-69`), identical across arms; prompts 447–453 tokens both arms;
out 128/128 all samples; medians recompute 9.25 / 9.577 = +3.54 %; the ≥10 % gate quote matches
P1 plan line 47. No client/engine mixing found — Result 1 is the only naming violation.

Defects:

1. **Attribution overreach.** The patched build is the whole uncommitted tree: 8 modified + 3
   untracked files (manifest.txt:8,14), 283 insertions incl. arg/common/ggml-backend/ggml-cpu
   moe-cache machinery, of which the device-ID repair is the +38-line mmvq.cpp case block. The
   e2e run measures tree-vs-tree; "+3.54 % for the device-ID repair" is rescued only by the P0
   shape-matched operator A/B (+2.44 %/token, `operator-ab-summary.json`), which is consistent but
   is a different measurement at 180 W.
2. **"Parity holds" overstates the e2e evidence.** `parity-k216-clean-fallback.txt` is a
   test-backend-ops smoke (4/4 MUL_MAT_ID OK); no clean-vs-patched token-id parity was captured
   (greedy would have made it cheap).
3. **Probe coordinate mislabeled the same way**: "cold p512/g128" (manifest.txt:19) vs actual
   ~450 tokens (word-approximation client, `p1-k216-client.py:52-59`). Matched across arms, so
   the A/B stands; "L5 recipe verbatim" refers to the launch flags, not the request shape.
4. Minor: stray `0` line in manifest.txt:16 (tree-count echo glitch); cooldown to ≤55 °C is
   implemented but no temperature values retained (protocol §6); pre-run gpu0 cap 180 W restored
   as 180, not the contracted 150 W rest state.
5. "Prompt processing unchanged" (`P1-k216-routed-gemv.md:23`): engine_pp spread 39.5–60.2,
   medians 53.6 vs 56.7 — defensible, weak; not part of the headline.

## What may be claimed publicly today

- **Nothing from Result 1 as a public headline.** Internally, PROVISIONAL: "on the champion
  stack, quantizing the draft LM head + MTP body to INT4 raised the C1 client post-first rate by
  a median +37.2 % (fresh n=5, disjoint distributions) at an actual ~6993-token prompt, with
  acceptance noisy (measured-window ±0.03) and TTFT unchanged at this cell." The +46.0 % n=3
  stays screen-class selection evidence until a fresh n=5 exists.
- **Result 2 as a negative/withdrawal record**: "an ABBA end-to-end A/B (n=6/build, engine
  `predicted_per_second`) measured +3.54 % median engine decode for the patched tree vs the clean
  export at the L5 cell — below the ≥10 % gate; phase withdrawn, repair kept." Full attribution
  to the device-ID cases alone waits on the single-variable rebuild.

## Minimum additional evidence to publish1. Tokenizer-calibrated exact 512/8192-token prompts (or relabel to actual counts); fresh n=5 at
   both P4 cells (p512 has never had one).
2. Measured-window acceptance: bracket the measured phase with spec-decode counter snapshots
   (before/after diff), not process-cumulative after-only.
3. Champion-generation patch validation: anchor check on vLLM 0.27.2rc1.dev77 plus a greedy
   C1/TP1 token-parity spot check patched vs baseline (the headers' own validity condition).
4. Energy + named-sensor temperature telemetry over the measured windows and a real cooldown gate
   between arms, so configured cap and measured draw separate.
5. P1: clean tree + only mmvq.cpp device-ID cases, same cell, n≥5, retained temperature log, and
   a greedy token-parity dump; else publish with the tree-vs-tree wording.

## What I could not verify

- Whether the champion overlay or the 10 non-mmvq patched files are performance-inert on these
  paths (no per-file isolation runs exist).
- Thermal/power behavior during any arm (no energy or temperature samples retained; P4 cooldown
  not implemented at all).
- That MTP probabilistic acceptance keeps the output distribution equal to the target's under
  INT4 draft weights on this engine generation (no parity or KL evidence).
- Repeatability beyond three P4 A/Bs at one cap on one GPU (all GPU 1; recorded champion cells
  were GPU 0 — cross-run comparison remains correctly disclaimed in the manifests).
- The `+36.9 %` recorded-screen reference in worklog §12 (pre-existing evidence, not re-audited).
