---
doc_id: DOC-RESEARCH-NEURAL-DOWNLOAD-20260831
title: External Cross-Check — neural.download (Steve Seguin b70-optimization-lab) 2026-08-31
lifecycle: active
created: 2026-08-31
updated: 2026-08-31
source_fetch_date: 2026-08-31
source_site: https://neural.download
source_repo: https://github.com/steveseguin/b70-optimization-lab
source_repo_commit: cabc9d4d4b078d0e8fd012f9b39c3dae27115699
source_repo_commit_date: 2026-08-31T12:02:55Z
source_scoreboard: https://neural.download/results/scoreboard.md
source_scoreboard_last_review: 2026-08-27
quality_label: B70-verified-external
canonical_for: [external-crosscheck-neural-download]
supersedes: none
---

# External Cross-Check — neural.download 2026-08-31

**Source:** [neural.download](https://neural.download) (Intel Arc B70 Optimization Lab) + [github.com/steveseguin/b70-optimization-lab](https://github.com/steveseguin/b70-optimization-lab) @ `cabc9d4` (2026-08-31T12:02:55Z, fetched 2026-08-31). Scoreboard last manual review 2026-08-27. Proof packets are per-model `results/` + `repro/` + `packages/` dirs in that repo; LocalMaxxing IDs listed where assigned.

**Scope:** Lab-tested B70 speeds with proof packets. This file extracts the table for cross-check, not a second benchmark truth — our `results/` remains canonical. See metric caveats §3 before comparing rows.

## 1. Lab-tested B70 speeds — extracted table

Speed = single-user median tok/s (tokens 1–100 after TTFT) over the lab's fixed 12-prompt/six-class cold suite (512-token cap, cache-zero, greedy unless noted), **except** where `†` marks a different suite/shape — not comparable across suites. `†` = 32K-input or aggregate-cN or llama-bench raw-engine or legacy 100-event metric; see `†` column tooltip on neural.download.

| Model / revision | Quant | Cards | Engine / runtime | Speed (1-user) | † 32K / aggregate | Proof link | Quality label |
|---|---|---|---|---|---|---|---|
| LFM2.5 2.6B (Liquid AI) — `Q8_0`, no MTP | Q8_0 | 1× B70 | llama.cpp/SYCL + BMG-G31 AOT | **132.14** | 89.94† (llama-bench 32K) | [details](https://neural.download/models/lfm25-26b-q8-b70.html) · [repro](https://github.com/steveseguin/b70-optimization-lab/blob/main/repro/lfm25-26b-q8-b70/README.md) | B70-verified, cache-zero, exact arrays |
| Gemma-4 26B A4B — `UD-Q8_K_XL`, MTP3 | Q8 + MTP3 | 1× B70 | llama.cpp/SYCL `c926ad098` | **122.16** (all-prompt median 123.73) | 114.85† (exact 32K HTTP) | [details](https://neural.download/models/gemma4-26b-a4b-q8-b70-125tps-20260701.html) · [packet](https://github.com/steveseguin/b70-optimization-lab/blob/main/results/gemma4-26b-a4b-q8-b70/README.md) | B70-verified, 15/15 cache-zero |
| Muse-Glimmer-30B — Q8 / WOQ, DFlash n15 | Q8/WOQ INT8 | 4× B70 TP4 | llama.cpp/SYCL `030ebb558` + oneDNN WOQ, DFlash | **100.37** (first-100 median 161.90, pooled 100.37) | — | [details](https://neural.download/models/muse-glimmer-30b-q8-woq-b70-100tps-20260813.html) · [packet](https://github.com/steveseguin/b70-optimization-lab/blob/main/results/muse-glimmer-30b-q8-woq-b70/README.md) | B70-verified closed Q8/WOQ |
| Laguna-S-2.1 INT4 — DFlash depth 11 (Poolside) | INT4 W4A16 | 4× B70 TP4+EP4 | vLLM/XPU `1a7f61fef`, kernels `99886d783` | **125.46** (conv interval; 126.73 hist) | — (32K withheld: quality fail) | [details](https://neural.download/models/laguna-s-2.1-int4-b70-125tps-20260731.html) · [packet](https://github.com/steveseguin/b70-optimization-lab/blob/main/results/laguna-s-2.1-int4-b70/README.md) | B70-verified, 13/13 exact, cache-zero |
| DeepSeek-V4-Flash-180B — FP8 dense + FP4 experts, DSpark7 (trim, experimental) | FP8+FP4 | 4× B70 TP4+EP | vLLM/XPU `264c7f2f7`, kernels `313156737` | **80.82** (3-run median 78.29, p10 71.67) | — | [packet](https://github.com/steveseguin/b70-optimization-lab/blob/main/results/deepseek-v4-flash-k160-b70/README.md) | B70-verified closed lane, experimental trim |
| MiniMax-M2.7 — AutoRound INT4 (MiniMax) | INT4 W4A16 | 4× B70 TP4 | vLLM/XPU `c51df430` | **89.31** (strict p512/n1536 mean) † | 63.91† (32K, p512/n1536) | [details](https://neural.download/models/minimax-m2-7.html) · [repro](https://github.com/steveseguin/b70-optimization-lab/blob/main/repro/minimax-m27-b70-89tps-20260520/README.md) | B70-verified strict-speed |
| Qwen3.8-27B 1× B70 — Q4_K_M, no MTP | Q4_K_M | 1× B70 | llama.cpp/SYCL (2× ASRock TP2 package variant exists) | **27.83** | 24.49† / 83.8†(c64 agg) | [details](https://neural.download/models/qwen38-27b-q4km-tp1-b70.html) | B70-verified, 12/12 exact vs MTP0 |
| Qwen3.8-27B 1× B70 — Q4_K_M + Q4_0 draft, MTP2 | Q4_K_M+MTP2 | 1× B70 | llama.cpp/SYCL fused Q6K draft-head | **42.64** (median of 2 fresh medians) | 36.51† / 68.3†(c16 agg) | [details](https://neural.download/models/qwen38-27b-q4km-mtp2-tp1-b70.html) | B70-verified, 12/12 exact vs MTP0 |
| Qwen3.8-27B 1× B70 — Q8_0, no MTP | Q8_0 | 1× B70 | llama.cpp/SYCL | **19.62** | 18.02† / 68.6† | [details](https://neural.download/models/qwen38-27b-q8-tp1-b70.html) | B70-verified |
| Qwen3.8-27B 2× ASRock B70 — Q4_K_M, no MTP (TP2) | Q4_K_M | 2× B70 TP2 | llama.cpp/SYCL mndodd `4302fb599` + fusion patch, BMG-G31 AOT | **49.72** | 44.44† / 192.3†(c96 agg) | [details](https://neural.download/models/qwen38-27b-q4km-tp2-asrock-b70.html) | B70-verified target-only, 12/12 exact |
| Qwen3.8-27B 2× B70 — Q4_K_M + Q4_0 draft, MTP2 (TP2) | Q4_K_M+MTP2 | 2× B70 TP2 | llama.cpp/SYCL | **64.24** (runs 64.18/64.29) | — | [details](https://neural.download/models/qwen38-27b-q4km-q4mtp-mtp2-tp2-b70.html#benchmark) | B70-verified, 24/24 exact vs oracle |
| Qwen3.8-27B 2× B70 — Q8_0, no MTP (TP2) | Q8_0 | 2× B70 TP2 | llama.cpp/SYCL | **36.73** | 33.85† / 163.6† | [details](https://neural.download/models/qwen38-27b-q8-tp2-asrock-b70.html) | B70-verified |
| Qwen3.8-27B 2× B70 — official FP8 W8A16, no MTP (TP2) | block-FP8 W8A16 | 2× B70 TP2 | vLLM XPU `ac7509e2b`, kernels `1e90ffa672`, oneCCL | **34.03** (baseline, 12/12 exact) | 31.49† / 1,112†(c128 agg) | [details](https://neural.download/models/qwen38-27b-fp8-vllm-tp2-asrock-b70.html) | B70-verified candidate, strict suite |
| Qwen3.8-27B 2× B70 — official FP8 W8A16, MTP1 (TP2) | FP8+MTP1 | 2× B70 TP2 | vLLM XPU, same stack | **51.92** | 46.64† / 1,091† | [details](https://neural.download/models/qwen38-27b-fp8-vllm-tp2-asrock-b70.html) | B70-verified scoped |
| Qwen3.8-27B 2× B70 — INT4, MTP5 (TP2, experimental) | INT4 MTP5 | 2× B70 TP2 | llama.cpp/SYCL, experimental | **101.17** | — | neural.download front page variant | Experimental / research lane |
| Qwen3.6-27B MTP GGUF — Q4, DFlash5 | Q4 DFlash5 | 1× B70 | llama.cpp/SYCL `e3546c794` + M6 verifier | **47.82** (p10 39.87) | — | [closure](https://github.com/steveseguin/b70-optimization-lab/blob/main/notes/2026-07-13-qwen27-dflash-sycl-closure.md) | B70-verified closed lane |
| Qwen3.6-35B-A3B — Quark W8A8 INT8 | INT8 W8A8 | 4× B70 TP4 | vLLM/XPU PIECEWISE | **93.55** (p512/n512) | — | [packet](https://github.com/steveseguin/b70-optimization-lab/blob/main/results/qwen36-35b-quark-int8-b70/README.md) | B70-verified closed ref |

Notes:
- Qwen3.8 TP2 rows on 2× ASRock B70: exact all-reduce / low-RAM patches retained; compare only at matching `ctx=2048`, greedy temp 0, 12-prompt fixed suite.
- INT4 MTP5 101.17 and FP8 MTP5 variants are experimental lanes (see §4).
- Scoreboard also lists community `B70-tested` rows (e.g. Qwen3.6-27B FP8 TP2 30.17 tok/s different prompt-length benchmark — explicitly not comparable to fixed-suite rows).

## 2. What this means for B70-DOCS hardware lanes

- **1× B70:** Our fleet is 2× B70 (X570-F x8/x8, CPU-attached, no P2P). All 1× rows above are reproducible on our GPU 0 alone — direct candidates for `ZE_AFFINITY_MASK=0` single-card lanes.
- **2× B70 TP2:** Their 2× ASRock TP2 numbers (Q4_K_M TP2 49.72, Q4_K_M+MTP2 64.24, Q8 36.73, FP8 34.03→51.92 MTP1) are the closest analog to our X570-F. Their TP2 uses exact two-card all-reduce + low-RAM/DNN-off patches — comparable to our ADR-0008 gated TP2 path (host-staged collectives, affinity patch). 4× rows are not runnable on our host.
- **4× B70:** Laguna 125.46, Muse-Glimmer 100.37, DeepSeek 80.82, MiniMax 89.31 — evidence for **future** 4-GPU expansion only. Document, do not claim on X570-F.

## 3. Metric caveats — do not rank across suites

- **This table's 1-user headline:** median tok/s for tokens 1–100 after TTFT across 12 unique cold prompts (cache-zero, deterministic, greedy unless noted). Earlier legacy rows used 100-event numerator / 99-interval denominator — multiply legacy display by 0.99 for conventional accounting; relative within-legacy comparisons unchanged.
- **† values are different suites:** Gemma 32K = 32,571-token HTTP decode; LFM2.5 89.94† = llama-bench raw-engine tg128; aggregate `192.3†` etc = HTTP concurrency aggregate (not single-user decode); MiniMax 89.31 = p512/n1536 mean (not p512/g128). Never rank `†` against non-`†`.
- **Not comparable to B70-DOCS vLLM suite:** Our cookbook bests (e.g. Qwen3.6-35B MTP4 170.91 vLLM XPU p8192/g128, Qwen3.8 draft 106.7 p512/g128, Nemotron DFlash 186.61 p2048/g128) use vLLM `bench serve` p512/g128 or p2048/g128 with different quant/engine — our numbers are vLLM XPU MTP4 / DFlash; theirs are llama.cpp SYCL Q4_K_M+draft / oneDNN WOQ / vLLM XPU block-FP8 W8A16. Same tok/s label, different numerator/denominator (suite, quant, engine). Do not claim one faster without matching suite+quant+engine+ctx.

## 4. Experimental / research lanes (do not adopt)

- DeepSeek-V4-Flash-180B (community trim K160, FP8+FP4) — research only; Muse-Glimmer Q8/WOQ custom oneDNN WOQ — requires preserved private stack; Qwen3.8 INT4 MTP5 101.17 experimental; Qwen3.6-35B AutoRound / Quark INT8 4× without host — not validated on X570-F. Keep as `research` quality, not `B70-verified-serving`.

## 5. Comparison vs B70-DOCS current bests (illustrative, not a ranking)

| Model family | B70-DOCS best (our suite) | neural.download best (its suite) | Why they differ |
|---|---|---|---|
| Qwen3.8-27B 1× | GPTQ INT4 G128 + BF16 MTP4 102.61 tok/s Run48 C1 non-thinking (vLLM XPU) | Q4_K_M+MTP2 42.64 tok/s (llama.cpp SYCL, 1×) | Different quant (GPTQ vs GGUF) + engine (vLLM XPU vs SYCL) + sampling (recommended vs greedy) + suite (p512/g128 vLLM bench serve vs 12-prompt median). |
| Qwen3.8-27B 2× | — (TP2 blocked/research, ADR-0008) | Q4_K_M+MTP2 TP2 64.24 (SYCL) / FP8 W8A16 MTP1 51.92 (vLLM) | Their TP2 is measured; ours is gated — cannot compare until we reproduce with host-staged collectives. |
| Qwen3.6-35B-A3B | MTP4 170.91 vLLM XPU (p8192/g128, greedy) | Quark INT8 93.55 p512/n512 (vLLM PIECEWISE, 4×) | Different suite (p8192/g128 vs p512/n512) and card count (1× vs 4×). |
| Nemotron 30B | DFlash 186.61 p2048/g128 (vLLM XPU, 1×) | No Nemotron 30B row in this extract (close: 4× 35B 125 type is Laguna, not Nemotron) | No overlapping suite. |
| Gemma-4-26B | Not served (no dossier) | 122.16 MTP3 (1×, SYCL) | Opportunity — lightweight single-card fast interactive. |
| LFM2.5 2.6B | Not served | 132.14 Q8 (1×, SYCL) | Opportunity — lightweight. |

> Rule: document the suite+engine+quant alongside every number. If suite or quant differs, describe the gap — never rank.

## 6. Recommendations — prioritized next researches

1. **(a) Single-B70 lightweight lane:** Probe LFM2.5 2.6B Q8 vs Gemma-4 26B Q8 MTP3 for speed/quality tradeoff on one card. Both are `B70-verified` on 1×, low VRAM, high tok/s. Gate: reproduce on `ZE_AFFINITY_MASK=0` with our `llama.cpp` SYCL stack before any quality claim. LFM2.5 = ultra-light; Gemma = general-purpose 122 tok/s.
2. **(b) Dual-B70 TP2 Qwen3.8 MTP2 vs FP8 W8A16 maturity:** Next concrete 2× target is Q4_K_M+MTP2 TP2 64.24 (SYCL) vs official FP8 W8A16 TP2 34.03→51.92 MTP1 (vLLM). Their FP8 W8A16 needs pinned vLLM `ac7509e2b` + kernels `1e90ffa672` + oneCCL + XPU Graph off; GPTQ INT4 is our proven vLLM path. Gate: pick one TP2 recipe, stage host patches, n=5 fresh medians on X570-F.
3. **(c) 4× lane is future hardware only:** Laguna 125.46 INT4 DFlash TP4, Muse-Glimmer 100.37 Q8/WOQ TP4, DeepSeek 80.82 TP4 — we have 2× X570-F, no 4× host. Record as `future if hardware expands`; do not schedule benches.
4. **(d) DFlash vs DSpark depth tuning:** DFlash n15 (Muse), DSpark7 (DeepSeek), Laguna DFlash11 show depth 7–15 viable. Depth choice is correctness-gated (exact canaries). Keep depth experiments behind `research/` + ADR, not `models/STATUS.md` promotion.
5. **(e) Reproduce neural.download packets on our 2× ASRock-equivalent X570-F before claiming:** Mandatory repro gate: fetch packet `README.md` + pinned container digest, run fixed 12-prompt suite cache-zero, verify 12/12 exact arrays vs oracle, then promote. Until then, `B70-verified-external` ≠ `B70-verified-local`.

## 7. Links

- Front page: <https://neural.download/>
- Scoreboard: <https://neural.download/results/scoreboard.md>
- GitHub repo: <https://github.com/steveseguin/b70-optimization-lab>
- Packages (per-model repro): <https://github.com/steveseguin/b70-optimization-lab/tree/main/packages>
- Repro catalog: <https://github.com/steveseguin/b70-optimization-lab/tree/main/repro>

---
*Fetched 2026-08-31. Scoreboard last manual review 2026-08-27. This file is an external cross-check; it links to external proof — it does not duplicate `results/` evidence.*
