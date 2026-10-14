---
title: "Model Status Matrix"
type: model-dossier
date: 2026-08-24
status: living
---

---
doc_id: DOC-MODEL-STATUS
title: Model Status Matrix — TP2 / PP2 Viability
lifecycle: active
created: 2026-08-31
updated: 2026-08-31
registry_mirror: registry/model-status.json
---

# Model Status Matrix

Single table of record for per-model best single-B70 and dual-card viability. **TP2/PP2 columns reference ADR-0008: independent stacks are the default; TP2 is a gated research path.**

| Family | Model (HF) | Best single-B70 (engine · quant · tok/s · P/G/C · n) | TP2 status | PP2 status | Open gate | Evidence link |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | `Qwen/Qwen3.8-27B` → turboderp **EXL3 4bpw** `@113cf7ab` (local: `/mnt/<nvme-disk>/b70-exl3-qwen38-4bpw-20260926`) | **exl3xpu** (vLLM-XPU fork, ExLlamaV3 trellis + native MTP3, fp8 KV) · 24.6 chunk-rate p512/g128 @230W sustained (+33-42% vs 150W; usage-verified ~25 post-first @150W) · **only arm at full 262,144** (fp8 pool 287,040; 131,098-tok completion proven) · clean 6/6 battery (code_execution 14/14), 8/10 exec tasks, cache 5.4x | **blocked** — 2026-09-29 decision: stock TP2 impossible at load (plugin guard + xccl-only = RL-075 class); custom TP2 conditional (repr. gate PASS 38/38 on sampled class only, RL-104; loaders + captured ≤60 µs copy-collective unproven, RL-103..RL-105) | blocked | **2026-09-27 ROUTE CHANGE (ADR-0010):** EXL3 primary; W4A16/GPTQ class CLOSED (30-defect ×3 exports, ~4.9x slower than removed champion, 256K-unreachable; RL-087); OV int4 alternate (28.5 t/s @150W); bf16 base verified correct | `models/qwen/qwen3.8-27b/README.md` · `engines/exl3xpu/README.md` · `results/2026/09/autoround-vs-exl3-20260926/` · `research/2026/b70-exl3-tp-decision-20260929.md` |
| Qwen3.8-Flash-Next | `Signal-3.8-Flash-Next` → AP-Q4_K_XL GGUF (sha256 `427a7a10…`) | llama.cpp SYCL (qwen4exp 2026.1.1 pinned build `7c2d6adc…`) · dual-card 49/51 split + PLE + 6 expert blocks CPU · KV q8_0/q4_1 · **21.9 tok/s** p512/g128 engine tg median n=3 (attempt #14, first valid — diagnostic-grade) · cold input 194 tok/s | n/a (split mode) | n/a | **RL-070:** calibration VERIFIED (never redo), coherence PASS; **NOT a speed route as placed** (3.5×/9× slower than qwen27 champion) — capacity route (94 GB / 2 cards); thermal blocker fixed (RL-069/072) | `results/2026/09/flashnext-c1-screen-20260925T215251Z/` · `registry/model-status.json` |
| Qwen3.8-27B-FP8 | `Qwen/Qwen3.8-27B-FP8` (native block-FP8) | vLLM XPU · block-FP8 W8A16 · **48.08 tok/s** p1024/g128 MTP4 W8A16 reroute (RL-003); predicted 50–55 ms step after small-M kernel | **research** — TP2 is the only viable placement (both cards occupied), requires per-worker `ZE_AFFINITY_MASK` + `SYS_PTRACE` | blocked | Conv + selector + small-M kernel build | `research/2026/qwen38-fp8-tp2-speed-attack-consolidated-20260830.md` |
| Qwen3.6-35B-A3B | `Qwen/Qwen3.6-35B-A3B` → GPTQ-INT4 G128 + BF16 MTP | vLLM XPU · GPTQ INT4 · **validated** (exact 131072-token completion, MTP4) | **provisional** — not the serving default; single-card is validated, dual-TP2 not adopted | blocked | Mixed-load `causal_conv1d` fix | `models/qwen/qwen3.6-35b-a3b/serving.md` · `campaigns/2026/qwen36-pi-128k-matched/README.md` |
| Nemotron-3.5-Lightning-30B-A3B | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B` → GPTQ-INT4 G64 + BF16 DFlash n7 | vLLM XPU · GPTQ INT4 G64 + DFlash n7 · **186.61 tok/s** p2048/g128, **157.92** p8192/g128 (n=5, 150 W, cold input 7160 tok/s) | **blocked** — independent stacks default; hybrid SSM+MoE unfused Mamba2; no TP2 campaign | blocked | Fused Mamba2 SSD **WITHDRAWN** 2026-09-21 (P2; fusable 8 % vs 73 % MoE+GEMM) — reassigned to small-M/grouped-GEMM (P5→P3) | `models/nemotron/nemotron-3.5-lightning-30b-a3b/serving.md` · `decisions/adr-0002-nemotron-dflash-serving-default.md` |
| Ornith-1.5-35B-A3B | `ornith-ai/Ornith-1.5-35B-A3B` → GPTQ-INT4 G128 + BF16 MTP1 | vLLM XPU · GPTQ INT4 G128 MTP1 · **94.3 tok/s** p512/g128 n=3 greedy (provisional) | **blocked** — route closed 2026-08-24 (quality rejected, artifacts deleted) | blocked | None (archived; evidence retained) | `models/ornith/ornith-1.5-35b-a3b/serving.md` · `campaigns/2026/ornith15-mixedcal-v2/README.md` |
| Muse-Glimmer-30B | `facebook/Muse-Glimmer-30B` → Q4_K_XL + mmproj + DFlash n2 | llama.cpp SYCL · Q4_K_XL `GGML_SYCL_F16=ON` · **26.8 tok/s** p512/g128 @128K (n=5 DFlash n2); 29.2 @64K | **blocked** — independent stacks default; no vLLM TP2 path (compressed-tensors smoke only) | blocked | Projector + reasoning token budget | `models/muse/muse-glimmer-30b/serving.md` |

**Legend:** TP2/PP2 `validated` = gated campaign already passed n=5 + correctness; `provisional` = n=3 or single-cell; `blocked` = ADR/research prohibits retry without new hypothesis; `research` = declared campaign in progress. `P/G/C` = power / greedy vs recommended sampling / context used at load.

> Before TP2/PP2 work: check this table + `decisions/adr-0008-dual-b70-independent-default.md` + `research/LEDGER.md` blocked_pattern. Never launch stock `tensor_parallel_size=2` without host-staged collectives + affinity patch.

## External cross-check — neural.download 2026-08-31

Source: [neural.download](https://neural.download) + [b70-optimization-lab@cabc9d4](https://github.com/steveseguin/b70-optimization-lab) (fetched 2026-08-31). Scoreboard 2026-08-27. **†** = different suite (32K/aggregate/legacy) — not rankable vs 1-user median.

| Family | neural.download 1-user (cards · quant · speed) | Our status | Gap / next quant·runtime to try | Priority |
|---|---|---|---|---|
| Qwen3.8-27B (SYCL lane) | 1× Q4_K_M 27.83 · 1× Q4_K_M+MTP2 42.64 · **2× Q4_K_M TP2 49.72** · **2× Q4_K_M+MTP2 TP2 64.24** · 2× Q8 TP2 36.73 | 1× GPTQ INT4+MTP4 102.61 (vLLM XPU, non-thinking, **internal record — different lab/host/engine/quant, never merged with the external-lab cells in this row**) | Our vLLM GPTQ+MTP4 is different engine/quant — not comparable. **Try:** repro their 2× Q4_K_M+MTP2 TP2 64.24 on X570-F (mndodd `4302fb599` + fusion patch, BMG-G31 AOT) vs our FP8 path; also try LFM-style Q8 vs GPTQ tradeoff | (b) next 2× TP2 target |
| Qwen3.8-27B-FP8 | 2× FP8 W8A16 TP2 **34.03** → **51.92 MTP1**; INT4 MTP5 **101.17** (experimental) | 48.08 W8A16 reroute (RL-003) | FP8 TP2 MTP1 51.92 is the closest vLLM analog to our W8A16. **Try:** pinned `ac7509e2b` + kernels `1e90ffa672` + MTP1; INT4 MTP5 is research-only (flag experimental) | (b) FP8 maturity |
| Laguna-S-2.1 | **4× INT4 DFlash11 125.46** (TP4+EP4, vLLM) | Not on host (4× lane) | **Gap:** we have 2× only — no 4× host. **Try:** document as future if hardware expands; if 2× INT4 DFlash variant emerges, evaluate | (c) future |
| Muse-Glimmer-30B | **4× Q8/WOQ DFlash15 100.37** (llama.cpp WOQ, pooled) | 1× Q4_K_XL DFlash n2 26.8 (SYCL) | Our single-card Q4 vs their 4× Q8/WOQ — different card count. **Try:** if 2× WOQ path lands, test; otherwise note 4× as future | (c) future |
| Gemma-4-26B | **1× Q8 MTP3 122.16** (SYCL, 1× fastest single-card) | Not served | **Opportunity:** single-card lightweight fast interactive — highest verified 1× tok/s on lab. **Try:** LFM2.5 Q8 vs Gemma Q8 MTP3 speed/quality tradeoff on `ZE_AFFINITY_MASK=0` | (a) lightweight |
| LFM2.5 2.6B | **1× Q8 132.14** lightweight | Not served | Ultra-light 1×, 89.94† llama-bench 32K. **Try:** bench vs Gemma for tiny-model tier | (a) lightweight |
| DeepSeek-V4-Flash-180B | **4× FP8+FP4 DSpark7 80.82** (experimental trim) | Not served | Trim is research (community K160). **Flag:** research only — do not adopt as serving | research |
| MiniMax-M2.7 | **4× INT4 89.31** strict p512/n1536 | Not served | 4× lane — future | (c) future |
| Qwen3.6-35B-A3B | 4× INT8 Quark 93.55 (TP4) | Single-card validated (131K ctx MTP4) | Their 4× INT8 vs our 1× INT4+MTP — card count mismatch. **Try:** watch for 2× INT8/W8A8 on X570-F | — |

Recommendations: (a) single-B70 LFM2.5 vs Gemma; (b) dual-B70 TP2 Qwen3.8 MTP2 vs FP8 W8A16 maturity; (c) 4× lane is future; (d) DFlash/DSpark depth tuning stays in `research/`; (e) repro neural.download packets on X570-F before claiming. Full analysis: RL-020 in `research/LEDGER.md` (source analysis was host-side, never committed).

Cross-check is **B70-verified-external** ≠ **B70-verified-local** until X570-F repro passes cache-zero + 12/12 exact gates.

## External cross-check refresh — neural.download 2026-09-26 (scoreboard reviewed 2026-09-13)

Full sweep note: `research/2026/external-lever-sweep-20260926.md`. Delta vs the 2026-08-31 table above:

| Family / lane | neural.download current (1-user unless noted) | Change vs our 08-31 snapshot | Disposition |
|---|---|---|---|
| Qwen3.8-27B AutoRound INT4 | **2× TP2 MTP4 117.53** ("Fastest lossless single user"); 1× TP1 **81.16** | NEW qualified rows (fixed-K oneDNN, CLASSPAD=0, full decode-only graphs) | Comparable class to our GPTQ lane; their TP2 number is the ceiling reference for our O14 campaign |
| Qwen3.8-27B FP8 | 2× TP2 current headline **90.5** (MTP5 + 67k shortlist + allgather; LocalMaxxing approved 09-17); 1× TP1 MTP5 **53.52**; MTP3/4/5/6 ladder 48.46/50.47/53.50/54.78 | 51.92 MTP1 era superseded by depth-5 recipe; **INT4 MTP5 101.17 experimental DEMOTED** | Their FP8 maturity path: fixed-K r309 patch + CPU embed + verify-rows overlay (all absent from us) |
| Qwen3.5-4B / 9B W4A16 | 4B **191.73** (TP2 245.83), 9B **124.08** (TP2 177.28) — 67k draft shortlist | NEW lanes | Shortlist lever proven on these (E1) — our A/B running tonight |
| Qwen3.8 Flash-Next 125B-A6B | 4× FP8 TP4 MTP1 **46.85** (+23.87%) | NEW | 4× lane — future only |
| Lightweight (LFM2.5 / Gemma-4) | 132.14 / 122.16 unchanged | no change | Priority downgraded: our focus stayed 27B-class |
| Withheld since baseline | Muse/Laguna/Ornith/Flash-Next multi-user, Ornith 35B headline: "strict headline withheld/pending" after quality/identity failures | Numbers they once showed are now withheld | Do not cite pre-09-13 external headlines |

Key adoption levers from the sweep (ranked): E1 67k draft-head shortlist, E3 allgather-as-allreduce, E4 graph-recorded oneCCL, E5 llama.cpp XMX-FA rebuild (SYCL lane). Conflict: their lab quality-rejects our GPTQ target — canary **reproduced live tonight** (30 vs 14); see sweep note §3.

**Campaign outcomes 2026-09-26 (all n=3 screen → stop rules honored):**
- **E1 shortlist → FINAL_LOSS (RL-074)**: screen +10.97/+11.45% refuted by fresh n=5 finalist −1.49/−5.53%, gates clean, byte-identical both phases; cause = draft-acceptance reversal on our corpora (their list caps drafts to their vocab). Their 117.53/90.5 TP2 rows use the SAME shortlist+allgather+graph stack we cannot run (next line).
- **O14 TP2 → screen FAIL (RL-075)**: B vs control −71.4% (p512) / −45.5% (p8192); E3 overlay C −68.3/−69.5% vs control (+11% vs B at p512 only). Only host-staged TP2 serves on our topology; per-step host dispatch breaks the +20..50% band. Their 117.53 (TP2 ceiling reference) stays EXTERNAL — not reproducible here until remote-atomics/peer-IPC gap closes (E4 blocked, both SK settings crash pre-measurement).
- **Control A tonight: 76.27 / 72.60 post_first_tps @150W floor 69C** — healthiest 1-card reading this session; champion stack unaffected.

Cross-check is **B70-verified-external** ≠ **B70-verified-local** until X570-F repro passes cache-zero + 12/12 exact gates.
