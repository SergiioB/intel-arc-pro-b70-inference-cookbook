---
doc_id: RESEARCH-EXTERNAL-LEVER-SWEEP-20260926
title: "External lever sweep: neural.download, Reddit, forums, upstream — vs our recipes"
type: research
date: 2026-09-26
status: provisional
---

# External lever sweep 2026-09-26 — what the outside pushes that we don't have

PROVISIONAL — NOT FOR PUBLIC HEADLINE. All external numbers are B70-verified-external
(different labs/hosts) until they pass our protocol gates on X570-F. Method: 3 parallel
research agents (neural.download deep-read, Reddit/forums via Arctic Shift API, upstream
GitHub/HN/blogs), compared against our ledger + LEVERS/OUTCOMES. Today's live fetches:
reddit .json = 403 (RSS/Arctic Shift work), Arctic Shift + neural.download = 200.

## 1. Verdict table — new levers we lack, ranked by expected payoff

| # | Lever | Source & measured effect | Our status | Disposition tonight |
|---|---|---|---|---|
| E1 | **67k draft-head shortlist** (`VLLM_XPU_DRAFT_LM_HEAD_SHORTLIST`, draft lm_head rows sliced 248,320→67,248, lossless by construction — target verifies full) | neural.download: +8.1% (4B), +9.3% (9B) at depth 3, acceptance in noise; underpins their 117.53 TP2 27B record ([note](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/qwen35-9b-b70/notes/2026-09-12-the-draft-head-only-needs-a-shortlist.md)) | **ABSENT** (zero grep hits); our draft-INT4 pair uses different code (Fase-S helper), their r294 patch fails on us (`AssertionError: anchor 1`) | **TESTED TONIGHT → FINAL_LOSS**: screen +10.97/+11.45% (promoted) but fresh n=5 finalist **refuted it (−1.49/−5.53%)** with gates clean; α penalty −0.059/−0.100 on our traffic (list built from their corpora). Losslessness itself PROVEN (byte-identical both phases). Stop rule respected; only follow-up = rebuild list from our corpus. Evidence `results/2026/09/b70-improvement-probes-20260926/shortlist-ab-20260925T234630Z/RESULTS.md` |
| E2 | **FR-Spec reduced draft vocabulary** (vLLM issue [58578](https://github.com/vllm-project/vllm/issues/58578): slice lm_head to 50,521 ids) | reporter, 1×B70 Qwen3.8-27B AutoRound INT4: 59.5→74.6 tok/s (**+25%**), acceptance 2.99→2.96 | **SAME MECHANISM AS E1, same layer, same anchor** — proven by port audit (both patch `qwen3_5_mtp.py::compute_logits`; cannot coexist with E1 on our Fase-S stack) | **REDUNDANT — folded into E1**; if E1 wins, E2 is free corroboration; if E1's list misbehaves on our traffic, rebuild list from our own corpus (builders staged) |
| E3 | **TP2 allgather-as-allreduce** (`B70_ALLGATHER_ALLREDUCE=1`, ~50-line overlay on `XpuCommunicator.all_reduce`, world-size-2 only) | neural.download: +2.3% lossless (90.37 vs 88.3-88.5, FP8+MTP5+shortlist); traced ceiling ≈2% (per-call host dispatch, not bytes) | **ABSENT**; port assessment GO on our vLLM 0.27.2rc1 (`xpu_communicator.py:48` matches monkeypatch contract, plugin hook verified) | **TESTED TONIGHT → NO GAIN** (arm C, RL-075): engagement marker present, canary PASS, but C vs A = −68.3%/−69.5% and C vs B = +11% (p512) / −44% (p8192, unstable). The overlay trims host dispatch on an already-losing host-staged TP2; it cannot rescue the band. Closed unless TP2 topology changes |
| E4 | **Graph-recorded oneCCL collectives + L0-IPC P2P all-reduce** (vllm-xpu-kernels [#570](https://github.com/vllm-project/vllm-xpu-kernels/pull/570); compute-runtime ≥26.27 needed for peer IPC handles) | +5.2% (0.8B) / +1.7% (our 27B class) vs best non-default oneCCL; TP2+graphs+MTP 84.92 vs TP1 49.71 on their box; `CCL_ENABLE_SYCL_KERNELS=1` default = DEVICE_LOST/OUT_OF_RESOURCES on TP2 startup | **CONFIRMED LOCALLY, two-step failure (2026-09-26)**: attempt2 (SK default) died with their exact signature (UR 40 → UR 20, both workers) during drafter AOT compile; attempt4 (`SK=0`) passed compile (93.7 s AOT) then **segfaulted in `ze_a2a_reduce_scatter_entry`** — the peer-copy kernel, i.e. our no-remote-atomics/ADR-0008 class. Fix layered: `SK=0` + the r32-repro env that previously served TP2 on this host (4 GiB simple thresholds → host-staged path = O14's measured collectives, pidfd, ofi) | **CLOSED AS BLOCKED (2026-09-26)**: after 7 root-caused attempts, only host-staged TP2 serves here (0 segfaults/UR/narrow warnings in the final run). Peer-IPC/graph-recorded collectives — the entire +1.7% mechanism — crash before measurement on both SK settings (UR 40/20 and `ze_a2a_reduce_scatter` segfault). Reopen ONLY with remote-atomics/peer-IPC capability (compute-runtime ≥26.27 + topology change) |
| E5 | **llama.cpp SYCL XMX flash attention** (PR [#25222](https://github.com/ggml-org/llama.cpp/pull/25222), merged 2026-07-15, oneDNN graph SDPA) + [#25690](https://github.com/ggml-org/llama.cpp/pull/25690) quants row fix + [#21845](https://github.com/ggml-org/llama.cpp/pull/21845) multi-column MMVQ (spec-decode +45% claim) | PR: Qwen 27B Q8_0 pp ×1.21 @512, ×4.26 @80k; Reddit: B580 116K-ctx prefill 245→462 t/s, B60 8K PP +31% | Our SYCL secondary pinned build predates these merges [INFERENCE — check SHA] | **REBUILD queue for llama.cpp lane** (flash-next/PP ladders), not the champion |
| E6 | **GDN single-checkpoint state** (R311b overlay: 1 state block/request instead of 6) | neural.download: KV 26,178→40,140 tok (+53%) 1-card; 53.49→54.28 tok/s; 2-card speed-neutral | ABSENT; capacity lever for our Qwen3.8 long-context | Staged (overlay fetchable); adopt only if KV capacity, not speed, is the binding need |
| E7 | **V1 model-runner pin** (`VLLM_USE_V2_MODEL_RUNNER=0` on vLLM 0.29.0 or draft INT4/shortlist silently lost, 25-30%) | neural.download measured (128 vs 172 tok/s on 4B) | Our image 0.27.2 runs V1 by default (verified: qwen3_5 not in DEFAULT_V2 arch list, log proof) | **NOTED — no action on our version**; mandatory rule on any 0.29 rebase |
| E8 | **Host submission-latency probe + capture-size rule** (5-min in-container microbench; capture sizes must cover verify shape `[1,2]` for MTP1) | neural.download learn/host-tuning.html: slow host loses 1.89× without graphs; launch ≥5µs → graphs first | ABSENT | **RAN 2026-09-26T06:24Z** (pinned image, GPU0, idle): launch 5.69 µs async / 37.97 µs launch+sync, w8a16 GEMM (M=2) 63.9 µs, rms_norm (M=2) 154.0 µs. Baseline established; no cross-host gap to explain on THIS host (numbers in `results/2026/09/b70-improvement-probes-20260926/host-latency-baseline-20260926.json`) |
| E9 | **Intel llm-scaler 0.26.0-b2** as alternate base (MTP decoding perf improve, MTP prefix-cache fix, block-FP8 concurrent-decode corruption fix; ships native graph mode) | release notes 2026-09-09 | We run stock pinned vLLM XPU images | Candidate A/B base for a matched screen — zero-GPU-port first; lower priority than E1-E4 |
| E10 | **exl3xpu** (0xSero): EXL3 4bpw + MTP3 + FP8 KV + 262K ctx on 1×B70, int8-XMX prefill path ~1.9× faster linears | author-measured README: prose 91.2 C1 / 357 C8 | Our EXL3 = deferred no-go (Q3: no runtime) | **Closes the EXL3 dead-end differently** — out-of-tree plugin exists; evaluate when EXL3 interest revives |
| E11 | **Scheduled draft depth per batch size** (`num_speculative_tokens_per_batch_size [[1,8,3],[9,16,1],[17,64,0]]` + per-K graphs) | neural.download R276: 110.7 @c1 → 1150 @c64, exact vs MTP0 oracle; upstream 0.29 disables it | ABSENT; our concurrency knee (S1/S2) uses static depth | For dual independent-stack aggregate serving only; needs their R276 port — park until serving throughput becomes the objective |
| E12 | **B70 power cap >230 W** (Reddit claim: "B70 have 275W power cap") | anecdote (u/belliash 2026-09-16), unverified | Our ceiling = stock 230 W; decode byte-bound so H5 says power ≈ neutral anyway | **LOW**: read VBIOS limits once (zero-GPU); expect neutral on decode |
| E13 | DFlash2 on FP8 + PIECEWISE graphs on B70 (Reddit r/LocalLLM 1w8bj0o: 56-166 tok/s by category, 252 agg @c8) | measured (BetterBench) + counter-anecdotes (0% acceptance ×1, quality revert ×1) | Our DFlash2 = LOSS on GPTQ (D7), compile path fixed but drafter quality lost | Not adopted; re-evaluate only on an FP8 target where our GPTQ quality defect (§3) motivates a target switch |

## 2. Independent external confirmation of our closed verdicts (all today)

- **No remote atomics on these cards**: 4×B70 XPUGraph builder — "PCIe device-to-device atomics turned out unreliable… barrier hangs or corrupts" (r/LocalLLM 1u8yai1) = our RL-059.
- **P2P copy band**: AM4 x8/x8 B60 Dual owner measured ~11.5 GB/s ze_peer = inside our 11.1-11.9 (RL-059).
- **Stock TP collectives corrupt under graph capture** (same thread) = our stock-Docker-TP2 closure (RL-008).
- **oneCCL threshold lift faults both cards** (neural.download 2026-09-16: CAT errors, coredumps; pinned 4 GiB threshold is a guard) — matches our own pinned CCL_*_SIMPLE_THRESHOLD=4GiB in the Aug-31 repro.
- **MTP depth plateau** past 2-4 (Reddit u/adityazero; neural.download 9B sweep: depth 3 best, 4/5/6 degrade) = our MTP6 closure (RL-026) + D9.
- **IQ quants crawl on Battlemage** ("Q4_K outpaces sub-4bpw", r/LocalLLM 1wkpuu2) = our failed grouped-IQ3 gate (P-3/L9) and llama.cpp #29107 reject (L11).
- **Host knobs neutral**: CPU governor, iommu, NUMA pinning, Gen3-vs-Gen4 all within noise (neural.download learn/host-tuning.html); ASPM writes panicked cards twice — do not attempt.
- **Sequence-parallel for dense TP2: no-go** (neural.download: no call-count reduction, not bit-safe with M-class-sensitive oneDNN) — adopt as pre-closed.
- **Replicated drafter to skip draft collectives: −0.4 to −4%** — pre-closed.
- **vLLM ≻ llama.cpp on this model class** (independent B70 owner: 52.2 vs 29.0 tok/s) = our engine primacy.
- **B70 sustained-load CCS/BCS resets at 32-96 concurrent** (HN postmortem w/ dmesg) — corroborates our #55425-family stability signature; relevant to B>32 serving (O8).
- **Prefill promoted NOTHING** in their Sep 13/14 review ("No new runtime candidate was benchmarked or promoted") = our W4A8/ubatch/max-num-batched-tokens closures.

## 3. Conflict found: their lab quality-rejects OUR GPTQ champion — REPRODUCED TONIGHT

Their claim (b70-optimization-lab README + community/sergiiob validation doc): the exact
checkpoint `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` fails a deterministic
code-result canary: answers **30** where correct is **14** for `sum(i*i for i in range(4))`
(Q8_0/Q4_K_M arms answer 14). Scope: rejects it as a *no-quality-loss default*; speed rows stand.

**Tonight's live reproduction on our own provider (6 requests, greedy, thinking off):**
arithmetic 60✓, JSON✓, factual Au✓, logic yes✓, **code_execution 30✗ (twice, deterministic)**.
Evidence: `~/tmp/night-20260926/gptq-canary/` (FINDINGS.md, canary-results.json,
upstream battery staged). [INFERENCE] mechanism: inclusive-range off-by-one (30 = Σ₀⁴ i²).

**Gate gap (not a claim contradiction):** our 7-rung ladder has the right rung
(`correctness-validation.md` rung 6) but no exact-answer item was ever in our predeclared
set — coherence/replay/PPL all pass a fluent wrong answer. Disposition: add the 7-case
exact-answer battery as a pinned-baseline drift detector (pin champion's actual outputs,
including known-fail=30), document the defect in the registry; do NOT demote speed records
(their scope explicitly preserves them). Optional: localize checkpoint-vs-engine with
Q8_0/Q4_K_M GGUF of same source on our host (not run tonight).

## 4. Scoreboard delta (neural.download, 2026-09-13 review vs our 2026-08-31 snapshot)

New qualified rows since our snapshot: Qwen3.5-4B W4A16 **191.73** 1×B70 (TP2 245.83),
Qwen3.5-9B W4A16 **124.08** 1×B70 (TP2 177.28), AutoRound INT4 fixed-K TP2 MTP4 **117.53**
("Fastest lossless single user"), FP8 TP1 MTP5 **53.52**, Flash-Next 125B-A6B TP4 FP8 MTP1
**46.85** (+23.87%), their current FP8 TP2 headline **90.5** (MTP5+shortlist+allgather,
approved LocalMaxxing 2026-09-17). Our own champion (GPTQ+MTP4+draftpair ≈72-80 post-first
client) remains a different-lab/engine/quant class — never merged with their cells.
Withheld since baseline: Muse/Laguna/Ornith headlines after quality/identity failures;
"INT4 MTP5 101.17 experimental" demoted. Full table: `models/STATUS.md` cross-check (updated today).

## 5. What ran tonight (2026-09-26, results)

1. **E1 shortlist — FINAL_LOSS.** Screen +10.97/+11.45% promoted; fresh n=5 counterbalanced finalist refuted (−1.49/−5.53%, gates clean, lossless proven, draft-acceptance Δα flipped +0.04→−0.10: their 67k list caps drafts to their corpora's vocabulary, penalizing our traffic). Stop rule respected; only sanctioned follow-up = rebuild list from our corpus. RL-074; evidence `results/2026/09/b70-improvement-probes-20260926/shortlist-ab-20260925T234630Z/RESULTS.md`.
2. **O14 TP2 + E3 overlay — screen FAIL (0/2 cells).** B vs A −71.4%/−45.5%; C vs A −68.3%/−69.5%; E3 +11% vs B at p512 only, unstable at p8192. Only host-staged TP2 serves on this topology; peer paths crash (E4). RL-075; evidence `results/2026/09/b70-improvement-probes-20260920/tp2-night-campaign-screen-20260926T054618Z/RESULTS.md`. E4 closed as BLOCKED (topology), E3 closed as no-gain.
3. **GPTQ quality canary — reproduced FAIL** (§3): "30" vs 14, deterministic, our exact artifact; gate-gap documented, not a speed-row contradiction.
4. **E8 host-latency probe — RAN**, baseline recorded (§ E8 row + `results/2026/09/b70-improvement-probes-20260926/host-latency-baseline-20260926.json`): launch 5.69 µs / 37.97 µs sync, w8a16 M=2 GEMM 63.9 µs.
5. **Harness lessons banked** (scripts + backups): frozen-sensor `stable_pkg_temp`/continuous-poke floors, `set -e` `kill || true`, narrow-mask→oneCCL uuid mismatch, `SK=0` + r32 env as the only serving TP2 recipe.

## Sources (primary)

- https://github.com/steveseguin/b70-optimization-lab (scoreboard.md reviewed 2026-09-13; notes 2026-09-09…09-18; learn/host-tuning.html; packages/qwen38-27b-fp8-tp2-b70/README.md)
- https://github.com/vllm-project/vllm/issues/58578 (+ prototype github.com/akapug/homelab-atlas)
- https://github.com/vllm-project/vllm-xpu-kernels/pull/570 ; https://github.com/vllm-project/vllm/pull/54768
- https://github.com/ggml-org/llama.cpp/pull/25222 ; /25690 ; /21845 ; /29245 ; /28476 ; /29143
- https://github.com/intel/llm-scaler/releases (0.26.0-b2, 2026-09-09)
- Reddit (via Arctic Shift): r/LocalLLM 1u8yai1 (4×B70 XPUGraph, atomics), 1w8bj0o (DFlash2 FP8), 1wodq98 (QuixiCore-XPU pointer); r/LocalLLaMA 1ulv6g5 (XMX-FA tip), 1vulh45 (MTP plateau, vLLM-vs-SYCL 1.8×), 1wkpuu2 (IQ crawl), 1wc6fsk (expert eviction prefill), 1wiy2g4 (275W claim)
- https://news.ycombinator.com/item?id=49100410 (B70 postmortem CCS/BCS resets); /49234677 (LLM-Scaler)
- https://github.com/0xSero/exl3xpu ; https://github.com/QuixiAI/QuixiCore-XPU ; https://github.com/CySpiegel/vllm-intel
- Their quality decision: community/sergiiob-qwen38-27b-vllm-xpu/validation/2026-08-16-quality-kv-dtype-decision.md (staged locally)
