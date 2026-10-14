---
title: "B70 inference research - findings compendium (2026-08 to 2026-10-12)"
type: research
date: 2026-10-12
status: provisional
---

# B70 inference research — findings compendium

Two Intel Arc Pro B70 cards, ~2 months of work, 182 ledger rows. This is the one
write-up: what worked, what failed and why, the numbers, the realizations.

Rules used here. Numbers carry their cell and metric class and a citation
(ledger id `RL-nnn` or a file path). Claims that came from outside this host are
marked `[EXTERNAL]`; claims that are deductions are marked `[INFERENCE]`;
claims I could not verify are marked `[UNVERIFIED]`. The ledger wins over prose
notes when the two disagree; every disagreement I found is in the last section.

## 0. One-page summary

1. **The whole problem is one law:** decode tok/s = yield × bandwidth ÷ bytes per
   token, and the round is bounded by `max(bytes/bandwidth, issue/rate,
   FLOPs/peak) + serial`, never by a product (`research/2026/b70-quant-lever-hunt-20261009-summary.md` §2).
2. **Fixed-rate scalar 4-bit is capped by mathematics, not by recipe:** GPTQ/AWQ
   W4A16/NF4 at 4.0 bpw sit 2.2–2.95× above the Gaussian rate-distortion bound,
   and E8 lattice codebooks do not close it; trellis coding (EXL3/QTIP) sits at
   ~1.17–1.18× (RL-173, RL-176). That is why INT4 lost on quality (RL-170, RL-171).
3. **INT4 also lost its speed case:** stock W4A16 INT4 on vLLM XPU v0.30.0 is
   32.85 vs EXL3 32.53 tok/s no-spec (1.01×) and 71.5 vs 79.9 with MTP; the old
   1.62× anchor was a different, gone image (RL-174).
4. **Per-tensor KL-optimal bit allocation is the biggest quality lever left:**
   turboderp's self-calibrated EXL3 `SC_5.00bpw_H6` reads 0.42× plain 4.00bpw KL
   on prose and 0.16× on chat, 5/5 canaries (RL-179).
5. **SC5 serves on the B70 today** through the new image
   `b70/exl3xpu-mixedk:20261012` (mixed K=4..7): 46/46 correctness checks, 6/6
   canaries, 3/3 8K prompts; speed 0.83× plain no-spec and 0.70× with MTP3
   (RL-180, RL-182).
6. **Decode is instruction-issue-bound, not DRAM-bound:** EXL3 reaches 381 GB/s
   of a 608 GB/s pin (63 %) at M=4 and 636 GB/s logical when the trellis work is
   stubbed; the card streams ~93 % with trivial dequant `[EXTERNAL]` (RL-119,
   `docs/reference/b70-format-and-lever-guide.md` §1).
7. **Prefill is compute, and the 2× INT8 path is unused:** every quant path on
   this stack ends in FP16/BF16 DPAS. On the Flash-Next IQ3_XXS route, expert
   GEMM is 37.30 % of 8K prefill; a plain INT8 GEMM swap caps at 1.229× (below
   its 1.3× gate) while a fused dequant+INT8 design prices 1.339× (RL-131,
   `b70-quant-lever-hunt-20261009-summary.md` §3).
8. **MoE is the class advantage and dispatch is the wall:** 186.61 tok/s on a
   resident 30B-A3B vs 75.7–81.0 C1 dense; but the K216 record showed ~84+ host
   synchronizations per token and a grouped GEMM at 1.79 TFLOPS vs ~46 dense
   (~25× gap) (models/STATUS.md; RL-032; `research/2026/b70-format-route-lessons-20260930.md` §2).
9. **One card per engine is the dual-card default:** direct peer copies run
   11.1–11.9 GB/s but peer atomics are absent; stock TP2 measured −71.4 %/−45.5 %,
   PP2 −0.26 %, llama.cpp RPC split −46 % (RL-059, RL-075, RL-051, RL-083).
10. **The host, not the model, is the most common failure cause:** sealed runs
    died to a foreign llama loop, to the session being killed by systemd-oomd,
    and to two harness bugs (`GROUPS` is a read-only bash array; a missing oneAPI
    runtime), each with an exact cause and a re-entry condition (RL-137, RL-158,
    RL-165).

## 1. Hardware facts and the governing law

### 1.1 The cards

| Fact | Value | Source |
|---|---|---|
| GPU | 2× Intel Arc Pro B70 (Xe2 / BMG-G31) | `docs/reference/hardware-software-stack.md` |
| Cores / XMX | 32 Xe-cores, 256 XMX engines | B70 datasheet (product spec) |
| VRAM | 32 GB GDDR6, 32,656 MiB visible per card | `docs/reference/hardware-software-stack.md` |
| Pin bandwidth | 608 GB/s per card | B70 datasheet; `docs/reference/b70-format-and-lever-guide.md` §1 |
| Dense INT8 TOPS | 367 (datasheet) | B70 datasheet |
| Topology | X570-F, CPU-attached PCIe x8/x8 | RL-059; `research/2026/x570f-dual-b70-pcie-topology-20260830.md` |
| Card-to-card | copy-only 11.1–11.9 GB/s (vs 5.86 host-staged); **no peer atomics** | RL-059 |
| CPU / RAM | Ryzen 7 5700X3D, 8 cores/16 threads; 60 GB recorded | RL-155; `docs/reference/hardware-software-stack.md` (snapshots disagree: 30 GB and 60 GiB) |
| Driver | Linux `xe`; debugfs for VRAM, hwmon for energy/temp | `docs/reference/hardware-software-stack.md` |
| PCI ids | 0000:0b:00.0 (GPU0, display) / 0000:0f:00.0 (GPU1) | RL-123, RL-178 |

### 1.2 Datapath rates (per Xe-core per clock)

| Path | Rate | What it means here | Source |
|---|---|---|---|
| XMX FP16/BF16 | 2048 ops | Where every quant decode path ends after dequant | `docs/reference/b70-format-and-lever-guide.md` §1 |
| XMX INT8 s8×s8 | 4096 ops (367 TOPS) | The only fully exposed 2× path; unused by production quant paths | RL-126; `b70-quant-lever-hunt-20261009-summary.md` §3 |
| XMX INT4/INT2 int×int | 8192 ops | Real; measured 2.00× per instruction via ESIMD, but needs 4-bit activations | `research/2026/xmx-dpas-rate-classes-20260915.md` |
| DP4A (vector) | 1024 ops | llama.cpp MMVQ decode with Q8_1 activations, M ≤ 8 | `docs/reference/b70-format-and-lever-guide.md` §1 |
| FP8, MXFP4, NVFP4 | no DPAS | Byte formats only on Xe2, never rate formats | RL-128; `docs/reference/b70-format-and-lever-guide.md` §1 |
| SLM / GRF | 128 KB per Xe-core; 128 or 256 registers per thread | Occupancy and barrier limits | `docs/reference/b70-format-and-lever-guide.md` §1 |

### 1.3 The governing law

```
decode tok/s = yield × achievable_bandwidth / bytes_per_token
t_step >= max(bytes/bandwidth, issue/rate, FLOPs/peak) + serial
```

Source: `research/2026/b70-format-route-lessons-20260930.md` §0;
`research/2026/b70-quant-lever-hunt-20261009-summary.md` §2.

### 1.4 Measured bandwidth anchors

| Anchor | Value | Cell / class | Source |
|---|---|---|---|
| Pin | 608 GB/s | spec | datasheet |
| EXL3 M=4 verify pass | 381 GB/s (63 %) | isolated-op sum, XPUGraph replay, 180 W | RL-098; postmortem table in `research/2026/b70-architect-postmortem-20260930.md` |
| "B_FAKE" trellis-stubbed | 636 GB/s logical | isolated qkv pair, failing numerics — **not** sustainable DRAM | RL-119; `b70-quant-lever-hunt-20261009-summary.md` §2 |
| Measured-achievable read | 521 GB/s | `sum()` reference | RL-098 §4 |
| FP16 GEMV on B70 | ~570 GB/s (~93 %) | `[EXTERNAL]` Puget | `docs/reference/b70-format-and-lever-guide.md` §1 |
| Aligned Q6_K, Strata | 493 GB/s (from 160) | `[EXTERNAL]` Strata INTEL.md | `docs/reference/b70-format-and-lever-guide.md` §1 |
| EXL3 no-spec dense 27B | ~470 GB/s (77 %) | `[INFERENCE]` from load size 14.4 GB/token × 32.5 tok/s | RL-174 |

Reading rule: the same card reaches ~93 % of pin with trivial dequant and 63 %
with trellis decode. The gap is instruction issue (RL-119, RL-090).

## 2. Quantization formats

### 2.1 The rate-distortion reason INT4 lost

After a two-sided random Hadamard transform, weight entries behave like
i.i.d. Gaussians and the rate-distortion bound is `D(R) = σ²·2^(−2R)`,
`D(4) = 0.0039·σ²`. No format beats it at equal rate.

| Quantizer class (fixed rate, 4 bit) | MSE / D(R) | Bits lost vs bound | Source |
|---|---|---|---|
| Uniform scalar, optimal step | 2.95× | 0.78 | Max (1960); `research/2026/b70-xtq-format-design-20261011.md` §1 |
| Lloyd-Max scalar | 2.43× | 0.64 | same |
| Entropy-coded uniform scalar (high rate) | 1.42× | 0.25 | same |
| E8P lattice VQ (QuIP# class) | 1.42× | 0.25 | QTIP Table 1 |
| Trellis-coded (QTIP/EXL3 class) | ~1.10× | ~0.07 | QTIP Table 1 |
| EXL3 mul1 measured (Gaussian MC, K=2) | 1.178× (K=2: 1.12×) | — | RL-176 |
| E8P12 2-stage RVQ measured (Gaussian MC) | 2.37× | — | RL-173 |
| Uniform INT4 measured (Gaussian MC) | 2.955× | — | RL-173 |
| Lloyd-Max measured (Gaussian MC) | 2.432× | — | RL-173 |

The measured KL ratios track the Monte Carlo rows: one-sided Hadamard GPTQ INT4
G64 read 2.93× EXL3 final-logit KL on a 13-layer prefix, two-sided G64 2.46×,
two-sided G128 2.84× (RL-173). Full two-sided incoherence recovers only 16 % of
the G64 gap — the rest is the fixed-rate scalar grid (RL-173).

Conclusion: at 4.0 bpw on this model only a trellis-class quantizer matches
EXL3, and the reachable levers above it are (a) the objective (output KL, not
layer-local proxy error), (b) per-tensor bit allocation, (c) better incoherence
(`research/2026/b70-xtq-format-design-20261011.md` §1).

### 2.2 Every format tried on this host

| Format / artifact | Model | Quality number | Speed number | Verdict | RL ids |
|---|---|---|---|---|---|
| EXL3 4.00bpw trellis + MTP3 (exl3xpu) | Qwen3.8-27B | battery 6/6 incl. code_execution "14" twice; 9/10 code tasks; canary "14" | 63.25 tok/s p512/g128 @230 W n=5 (49.22 @150 W; 61.12 p8192); source-log rate class | **adopted** (ADR-0010) | RL-090, RL-082, RL-088, RL-092 |
| EXL3 3.00bpw (K=3) | Qwen3.8-27B | battery 6/6 ×2, tasks 9/10, replay stable | 10.25 tok/s p512/g128 @230 W (diagnostic) | quality holds; kernel program is the whole speed case | RL-094 |
| GPTQ W4A16 INT4 (auto_gptq) | Qwen3.8-27B | canary "30" instead of "14" | 1.62× EXL3 decode on the gone image (`[per-baseline]`) | **removed**; defect is in the weights | RL-087, RL-092, RL-084, RL-117 |
| AutoRound W4A16 (compressed-tensors + auto_gptq) | Qwen3.8-27B | same "30" defect, 30/30 twice; fp32 margin −2.97 | arext 18.12 GiB streamed @83.3 ms = 222 GB/s vs EXL3 15.70 GiB @61.5 ms = 255 GB/s | whole W4A16 class closed | RL-117, RL-092, RL-087 |
| Community AWQ INT4 | Qwen3.8-27B | not reached (loader not exercised) | loads as compressed-tensors; 1.079× champion bytes (20.10 vs 18.63 GB) | not iso-byte; branch open | RL-034 |
| Q4_K_M GGUF | Qwen3.8-27B | code_execution "14" twice, but arithmetic 59-vs-60 | 11.57 tok/s p0/n128/d32768 (before TILE-KV) | fails a different exact case | RL-079, RL-081 |
| Block-FP8 W8A16 | Qwen3.8-27B | diagnosed defect at W8; rerouted W8A16 | 48.08 tok/s p1024/g128 greedy diagnostic | diagnostic only, below EXL3 | models/STATUS.md; RL-002–RL-004 |
| OpenVINO INT4 | Qwen3.8-27B | clean (6/6 class) | 28.5 tok/s p512 post-first @150 W (corrected from 22.4) | alternate route; ≥196K growth bug | RL-086, RL-033 |
| R4 rotated affine INT4 (one-sided fold, GPTQ G64/G128) | Qwen3.8-27B | best arm 2.9× EXL3 KL (prefix); two-sided G64 2.46× | kernel track never needed (quality killed it) | **closed rejected** | RL-170, RL-171, RL-173 |
| E8P12 2-stage lattice RVQ 4-bit | Qwen3.8-27B | 2.37× D(4) Gaussian | none spent | **closed**, no GPU time | RL-173 |
| `SC_4.00bpw_H5` (per-tensor K=3..6) | Qwen3.8-27B | KL 0.01602 prose (1.152×, worse) / 0.00653 chat (0.746×) | not served | fails the predeclared prose gate | RL-179 |
| `SC_5.00bpw_H6` (per-tensor K=4..7) | Qwen3.8-27B | KL 0.00582 prose (0.418×) / 0.00141 chat (0.161×); top-1 96.57/98.11 %; 5/5 canary margins | 27.17 tok/s no-spec (0.83×) / 57.5 MTP3 (0.70×) | served via new image; **quality tier chosen** | RL-179, RL-180, RL-182 |
| MXFP4 MoE (llama.cpp #29809) | Qwen3.6-35B-A3B | not run | not run | **closed by architecture/bytes**: no FP4 DPAS, 17-byte blocks unaligned, ≤9 % byte saving | RL-019, RL-128 |
| NVFP4 / W4A8 / INT3 / 2:4 sparsity | — | — | — | closed on datapath or attribution | `docs/governance/standing-rules.md` §3 |
| TQ1_0 / TQ2_0 | — | — | — | no SYCL kernels (CPU fallback) | RL-126 |
| IQ3_XXS (98-byte blocks, Strata) | Flash-Next | route adopted, quality gates open | see §6 | adopted with open gates | RL-125 |
| Struct-of-arrays IQ3_XXS repack | Flash-Next | words bitwise equal | no speedup at any E (SoA 0.79–1.03× its AoS twin; gate 1.5×) | **closed rejected** | RL-172, RL-163–RL-165 |

### 2.3 Exact-answer defect, localized

The W4A16 "30-vs-14" defect reproduces across GPTQModel, AutoRound
compressed-tensors and AutoRound auto_gptq with a code-heavy corpus, and the
bf16 base answers "14"/"14" on both runs. A CPU-only fp32 faithful forward gave
bf16 margin +0.4017 and AutoRound-ext fp32 margin −2.9730: a ~−3.3-logit
inversion with exact fp32 accumulation, deepest ~−7.1 around layer 60. Weights,
not kernel or accumulation dtype (RL-084, RL-092, RL-117). Margin-requant repair
fixed the canary (+0.5584) but never transferred to the held-out near-boundary
row across four variants (n03 deltas −1.41/−1.96/−0.89/−0.82), so the terminal
window scope is dead; the missing ingredient is calibration representation of
the near-boundary class (RL-113, RL-114, RL-115).

### 2.4 What is still open in formats

- SC5 mixed-K serving speed: the M=4 verify kernels for K≠4 are being profiled;
  0.70× with MTP3 is the measured position (RL-182).
- XTQ: our own trellis + allocation + Xe2-cheap code (RQ-29), with the K4 speed
  and K5 quality gates still unrun (RL-176, RL-177, RL-181).
- SC5 vs SC4 on our own harness for the *chat* class (SC4 is better there but
  loses on prose) (RL-179).

## 3. Engines and routes

### 3.1 What each engine is good and bad for

| Engine | Good for | Bad for | Pin (2026-10-09) |
|---|---|---|---|
| **EXL3-XPU** | Dense 27B quality + speed; only route reaching native 262,144 context on one card; MTP3; prefix cache 5.4× | Trellis decode costs 37 % of bandwidth; stock TP2 impossible at load; no non-EXL3 formats | image `21412bdd` (vLLM 0.26.1); RL-103, RL-088, RL-090 |
| **Strata SYCL** | Flash-Next MoE on one card with VRAM expert cache; MTP `--spec 4`; 989 tok/s 8K prefill; layer split exists in source | SYCL port lacks int8 prompt kernels and `--peer-device`; intermittent xe engine resets on GPU0 | v0.1.40.3 = engine 0.1.41, `d5ea713`; RL-125, RL-128, RL-160 |
| **vLLM XPU** | Nemotron GPTQ-INT4 + DFlash n7 (186.61 tok/s historical); FP8 arms; MXFP4/MXFP8/GPTQ/AWQ/compressed-tensors | Qwen3.8-27B GPTQ route quality-defective; TP2 gated; no NVFP4 kernel | `0.27.2rc1` nightly; v0.30.0 used for W4A16 screens | 
| **llama.cpp SYCL** | GGUF workhorse; fleet KV `q8_0` K + `q4_1` V; quantized-KV TILE +13.0 % at 32K | local tree ~2 months behind master (misses TILE-KV, grouped-MoE XMX, MXFP4 MoE); MMQ hard-disabled | fleet tree `1692f9e50`; RL-126, RL-081 |
| **OpenVINO GenAI** | Alternate clean INT4 27B route (28.5 tok/s @150 W) | ≥196K per-generation memory growth; no thinking-off switch costs it on task batteries | openvino `2026.5.0-22967`, genai `2026.5.0.0-3412`; RL-033, RL-086, RL-088 |
| **Cascadia** | Functional PP capacity placement | PP2 −0.26 %, RPC split −46 %, 3–4.5× behind vLLM single-card single-stream | 0.2.3/0.2.4; RL-051, RL-083 |
| **SGLang XPU / torch XPU** | Exploratory; kernel-repro and profiling substrate | no validated serving route | `engines/README.md` |

### 3.2 Route of record per model (2026-10-12)

| Model | Engine · artifact | Best recorded | Open gate |
|---|---|---|---|
| Qwen3.8-27B | exl3xpu · turboderp EXL3 4bpw `113cf7ab` | 63.25 tok/s p512/g128 n=5 @230 W; 262,144 native context; C16 426 aggregate | stock TP2 blocked at load |
| Qwen3.8-27B-FP8 | vllm-xpu · native block-FP8 | 48.08 tok/s p1024/g128 | small-M kernel + DFlash2 selector |
| Qwen3.8-Flash-Next | Strata v0.1.40.3 SYCL, one B70 · GSQ-RCO IQ3_XXS + native pack | 48.03 tok/s p512/g128 n=5; 989.11 tok/s 8K prefill | RQ-17 quality closure |
| Nemotron-3.5-Lightning-30B-A3B | vllm-xpu · GPTQ-INT4 G64 + BF16 DFlash n7 | 186.61 tok/s p2048/g128 @150 W (historical metric; re-measure owed) | RQ-11 closed, zero cells on v0.30.0 |
| Qwen3.6-35B-A3B | vllm-xpu · GPTQ-INT4 G128 + BF16 MTP | capacity observation only (p130944=131072) | causal_conv1d mixed-load |
| Ornith-1.5-35B-A3B | vllm-xpu · GPTQ-INT4 G128 + MTP1 | 94.3 tok/s p512/g128 n=3 | closed quality-rejected |
| Muse-Glimmer-30B | llama.cpp-sycl · Q4_K_XL + DFlash n2 | 26.8 tok/s p512/g128 @128K | projector + reasoning budget |

Source: `models/STATUS.md` and `registry/model-status.json`.

Engine pin deltas that matter: llama.cpp master adds TILE quantized-KV, IQ3_S
multicol, grouped-MoE XMX, MXFP4/NVFP4 MMVQ; exl3xpu's vLLM 0.26.1 base predates
the zero-hit cache fix vLLM #50729 (v0.29.0+). The `sym_int4` "output fix" in
llm-scaler 0.26.0-b2 is an INT4 MoE-router residual race and does **not** cover
our dense 30-defect (RL-089, RL-126).

## 4. Speculative decoding

### 4.1 What speculation is worth

- Economics: `yield = 1 + k × α` verified within ±2.4 % on every C1 window;
  +0.05 α at k=4 = +6.25 % (RL-093 A5).
- EXL3 native MTP3: yield 2.93–3.16 of a theoretical 4.0
  (`research/2026/b70-format-route-lessons-20260930.md` §0; counter-verified
  yields 2.954–3.13 in `research/2026/b70-architect-postmortem-20260930.md`).
- Per-position survival is **not universal**: 27/76 windows cliff, 8 flat, 38
  gradual, 3 early-collapse. The served EXL3-MTP3 route is flat decay
  0.79/0.65/0.46; modal accept-length 0.46 serving (RL-093).
- Draft-side INT4 on the removed GPTQ stack: +48.6 % at p8192 and +39.6 % at
  p512, iso-power, +31.4 % measured tokens/J (RL-035); the EXL3 MTP head has no
  separate quantizable pair (`docs/governance/standing-rules.md` §2.9).

### 4.2 What was tried and what it returned

| Lever | Result | Verdict | RL |
|---|---|---|---|
| k=3 safety clamp (#54785) | costs +5.45 % throughput (k4 75.04 vs k3 71.16); corruption class not live (canary hash `1c650a7f` identical at k=3/k=4) | clamp not adopted; canary adopted as standing gate | RL-042, RL-044 |
| bf16 draft KV | α 0.5812 → 0.5394 (−0.0418) | keep fp8 KV | RL-045 |
| Probabilistic draft sampling | screen +0.0505 α, finalist −0.0037 α and −3.86 % tps | not adopted | RL-046, RL-048 |
| presence_penalty 1.5 audit | helps α by +0.025 | keep recipe | RL-047 |
| FastMTP head retrain (1,500 samples) | α 0.5023 vs 0.5537/0.5421; 71.83 vs 77.26 tok/s; positions 83.2/68.7/28.0/21.0 % | failed; position sharing harms shallow positions | RL-056, RL-049, RL-058 |
| 67k draft-head shortlist | screen +10.97 %/+11.45 % → finalist −1.49 %/−5.53 %, α flip +0.038→−0.059 | refuted; only own-corpus rebuild allowed | RL-074 |
| Own-corpus shortlist (54,274 rows) | screen +4.14 %/+2.09 % (below +5 % gate), thermal-invalid | blocked, not refuted | RL-076, RL-077 |
| Draft-vocab shortlist already live | 512 of 1940 vocab blocks = 26.4 % of lm_head | engaged on the production route; prior "absent" premise was wrong | RL-106, RL-118 |
| Draft byte share | MTP already EXL3 4-bit (8 modules, 0.2126 GB class); draft ≈1.4 GB/round ≈ 2.9–3.1 ms; draft lm_head ≈3×252 MB ≈ 0.76 GB | further draft quantization can win at most a few percent | `research/2026/b70-sub4bpw-and-margin-requant-plan-20260929.md`; postmortem §0 correction |
| MTP depth 4 vs 6 | record keeps MTP4; depth-6 arm slower (blocked-pattern) | depth 4 | RL-026 |
| DFlash2 on quantized 27B | naive attach failed; K-norm recovery confounded; compiled path 1.7× vs eager, issue #56787 filed; acceptance ~2.0 vs MTP4 ~3.5 | closed; re-entry needs a drafter trained on the quantized target's states | RL-006, RL-027, RL-028, RL-029 |
| Nemotron DFlash n7 | route of record for Nemotron | — | models/STATUS.md |
| DFlash n7 drafter → INT4 (Nemotron) | +7–11 % priced, below the +15 % floor; already a documented lever | closed as not novel | RL-129, RL-130, RQ-24 |

Rule that came out of this: acceptance work goes to retraining or draft policy,
not KV precision, and single-run α deltas ≤ 0.05 are unresolvable at n=5 given
the observed 0.48–0.70 α spread (RL-045, RL-048).

## 5. Kernels and runtime

### 5.1 EXL3 decode attribution (where the time actually is)

| Bucket | Value | Class | Source |
|---|---|---|---|
| ccs0 busy | ~88 %, no bubbles, 200 W draw vs 230 cap | PMU window | RL-090 |
| Bandwidth lost inside kernels | 57–63 % of pin floor | PMU | RL-090 |
| M=4 verify EXL3 pass | 34.46 ms / 13.126 GB / 381 GB/s | isolated-op sum, XPUGraph, 180 W | RL-098; postmortem table |
| M=1 analog pass | 29.44 ms / 446 GB/s | same | postmortem |
| Last-round composition | 87.7 % gemm (563 kernels ≈1.1 ms each), everything else ≈5 ms | Level Zero trace under graph replay | RL-072 |
| Trellis decode cost | ~336 instructions per 512 B tile-group (M=4) | disassembly | `research/2026/b70-format-route-lessons-20260930.md` §3 |
| Host control-basis prep | ≈4.8 ms (10.7 % of round) | profiler + A/B | RL-060, RL-061 |
| Prefill (p8192/g1) | oneDNN gemm_kernel 74 %, cutlass fmha 12 %, GDN 7 %, fused 6 % | torch profiler | RL-066 |
| Prefill GEMM efficiency | 109–115 TFLOPS = 64–68 % of the ~170 TFLOPS BF16 class | model-derived | RL-066 |

Consequence: single-card decode kernel hunting was declared closed on the
evidence — bytes floor, rate ceiling, and L0-visible composition all agree
(RL-062, RL-072). The open kernel target is the M=4 verify pass, not M=1.

### 5.2 Kernel work that ran

| Item | Result | Verdict | RL |
|---|---|---|---|
| IQ3_XXS grouped prefill (from scratch, tile-local) | correct (33.5M FP16 values, 0 mismatches, 0 spills) but 1.30–1.39× **slower** vs stock | failed | RL-040 |
| fg_stage_a<block_iq3_xxs> fused-gemm | 9,448 µs vs stock 33,985 µs = 3.60×; 3.03× vs baseline | gate passed | RL-050 |
| K=3 EXL3 kernel series | build pass (360 K3 symbols), then one intermittent 1-bf16-ulp boundary flip, proven deterministic and race-free | boundary confirmed; screen amended and passed 644/644 | RL-095, RL-097, RL-099, RL-100 |
| MB=4 DPAS A/B | bit-identical to MB=8; +0.41 pin pts (down_proj), +4.72 (qkv) at M=4; +5 gate missed | failed | RL-101 |
| NT strip sweep (2/8 vs 4) | family-confined: in_proj_z NT2 +10.5/+9.5 %; down_proj regresses at NT2 | partial, not adopted | RL-102 |
| C2 V2 explicit B-bank binding | numerically bit-identical; +3.17 % (down_proj), +4.43 % (qkv) slower | failed | RL-112 |
| B6 microprobe (tracer route) | established the split of the M=4 residual | diagnostic | RL-096, RL-098 |
| RQ-25 struct-of-arrays IQ3_XXS | 0.79–1.03× the current AoS kernel; not load-width-bound | closed | RL-163–RL-165, RL-172 |
| Persistence / megakernel | blocked; do not assume one work-group per core gives occupancy or safe global barriers | blocked | RL-007, ADR-0009 |
| Graph capture | always on for vLLM-family serving; unitrace launcher mode sees replayed kernels (201,795 gpu_op events) while torch profiler does not | standing rule | RL-072, `docs/governance/standing-rules.md` §1.1 |
| DPAS F32 reduction order | not bit-reproducible vs CPU oracles (max diff ~40 abs); all candidate reduction models rejected | methods rule: reduction-aware reference or tolerance class | OUTCOMES run-1 wins table |
| AOT `spir64_gen -device bmg` | miscompiles both Viterbi kernels (mul1 40× MSE; scalar 61/64 exact) | production path must be JIT | RL-181 |
| icpx fp32→fp16 host oracle | default contraction folds the 2^112/2^−110 idiom; out-of-range magnitudes return garbage | device path fine; host oracle needs `-ffp-contract=off` | RL-043 |
| INT8 prefill (exl3xpu linears) | 1.89× at +0.17 % NLL | `[EXTERNAL]` sybil fork | `research/2026/b70-untried-levers-synthesis-20261009.md` §3 |
| RQ-22 INT8 prompt GEMM for Strata | 37.30 % expert-GEMM share of the 8K prefill; plain swap caps 1.229×, fused 1.339× | deferred; fused design is the only one above gate | RL-131 |

### 5.3 Runtime rules that were learned (not guessed)

1. Capture on before reading any decode number; eager traces are blind to
   replayed kernels (RL-072).
2. Synchronize before `torch.xpu.empty_cache`; keep XPU work on one thread. An
   unsynchronized `empty_cache` plus a cross-thread XPU prefetch produced a bcs
   copy-engine page fault and reset (RL-178).
3. Allocate no more than live free VRAM − 4 GiB, and never run arena probes on
   the display card (RL-123, `docs/governance/standing-rules.md` §1.13).
4. Never name a shell variable `GROUPS`, `UID`, `EUID`, `PPID`, `RANDOM`,
   `SECONDS` or `LINENO` in a GPU driver script; assert every work-group count
   against the allocated table size before launch (RL-164, RL-165).
5. One measuring client per C1 server (RL-060; out-of-band incidents in RL-120).

## 6. MoE / Flash-Next / Strata route

### 6.1 Why MoE is the B70's class advantage

| | Dense 27B @4bpw (EXL3) | MoE 30B-A3B @4bpw |
|---|---|---|
| Weight bytes per token | ~13.5 GB (whole model) | ~1.5–2 GB (active experts + shared) |
| Theoretical ceiling @608 GB/s, yield 3 | ~111 tok/s | ~750–1000 tok/s |
| Measured here | 75.7–81.0 tok/s C1 solo @230 W; 426 aggregate C16 | 186.61 tok/s (Nemotron, historical metric) |

Source: `research/2026/b70-format-route-lessons-20260930.md` §2.

### 6.2 The measured MoE wall: dispatch

- K216: 84+ host synchronizations per token; grouped GEMM 1.79 TFLOPS vs ~46 for
  a comparable dense GEMM (~25× gap) (RL-010, RL-011, RL-012).
- K216 device-ID repair: operator level +2.44 %, end-to-end 9.25 → 9.577 engine
  tok/s = +3.54 %, below the ≥10 % gate (RL-031).
- Nemotron device time: MoE gemm 41.7 % + gemm_kernel 31.3 % = 73 %; the whole
  fusable Mamba/SSD family is 8 % with 55–357 µs medians → Mamba2 fusion
  withdrawn (RL-032).
- RQ-11 (MoE attribution) consumed 2/2 sealed attempts and produced **zero**
  measurement cells: attempt 1 lost to a foreign loop, attempt 2's container
  died at boot with empty logs. The 0/0/0/0 attribution is a schema-required
  void, not data (RL-146, RL-147, RL-148).

### 6.3 Flash-Next: placement history and the adopted route

| Placement | Decode | 8K prefill | Source |
|---|---|---|---|
| llama.cpp AP-Q4_K_XL dual-card split (49/51 + PLE + 6 expert blocks on CPU) | 21.3–21.9 tok/s | 185–195 tok/s cold input | RL-070 |
| Strata `--expert-cache-device1` two-card host-staged | 33.44 tok/s | 111.86 tok/s | RL-125 |
| **Strata one card, IQ3_XXS, `--expert-cache auto`** | **48.03 tok/s** (2.22×) | **989.11 tok/s** (5.1–5.3×) | RL-125 |
| External references | Strata B70 31/41 tok/s; sybil tier 24.0 tok/s C1 / 1056–1260 tok/s prefill | `[EXTERNAL]` | RL-121 |

Route adopted 2026-10-09 (ADR-0011) replaces the llama.cpp dual-split with the
one-card Strata route. The route is a **thinking** model: every reply carries
`reasoning_content`, so all future batteries must budget gen ≥ 256 (the canary
"failure" was a gen-64 budget truncating a correct computation: "7" was the
first digit of 703). Source: OUTCOMES run-1, RL-152.

Quality gates stay open. The last passing evidence was: canaries 6/6 in both
runs, replay 3×3 token-identical, tasks 9/10, 8K two-turn correct, blocked only
by an unsatisfiable plan conjunct (RL-156); on the prompt-class re-run,
speculation acceptance 0.789 at 3.20 tok/round, 9/10 host tasks (RL-144). The
two later attempts were lost to xe engine resets on GPU0, not to wrong answers:
`dmesg` "Timedout job … in strata", xe ccs job_timeout_ms = 5000, then
`UR_RESULT_ERROR_DEVICE_LOST` (RL-160). A clean unsealed diagnostic then ran
21/21 requests HTTP 200 on each card (42/42) with zero resets and identical
answers (RL-166).

### 6.4 The two-card residency alternatives (both still gated)

- RQ-18: Strata `--layer-split` with all 24,576 IQ3_XXS experts resident.
  Amdahl ceiling ≤1.53× decode from removing the ~33–37 % expert-miss term;
  boundary handoff ~0.06 ms/round; source-feasible in the pinned build.
  Gated on RQ-17 (RL-127, RL-128).
- RQ-23: 50 % expert-pruned Flash-Next Coder, 58.4 GB with a 29.6 GB resident
  set, one card. Costs general capability and a 58.4 GB download (RL-128).

## 7. Dual-card and multi-GPU

| Approach | Result | Verdict | RL |
|---|---|---|---|
| Independent stack per card (`ZE_AFFINITY_MASK=0/1`) | default; two independent servers 47.8 aggregate | **adopted** | RL-009, ADR-0008, RL-083 |
| Stock Docker TP2 (oneCCL/xccl) | −71.4 % (p512) / −45.5 % (p8192) vs 1-card control; +overlay −68.3 %/−69.5 % | failed | RL-075 |
| Stock TP2 for EXL3 | impossible at **load**: `NotImplementedError: exl3: tensor parallel sharding not supported yet (use DP)` on every Column/RowParallelLinear; transport is xccl-only | blocked | RL-103 |
| EXL3 128-block slice contract | 38/38 pass: N-split bitwise exact, K-split exact to fp32 reassociation (max_abs 2.9e-6/9.5e-6); a 64-wide shard is materially wrong, so the %128 assert is necessary | representation gate only | RL-104 |
| Copy-based allreduce (xccl) | bitwise int64 exact, fp32 rel 0.0; latency 4 KB 101.7 µs, 64 KB 104.5, 512 KB 129.2, 4 MB 662.9; 128×51 KB serial = 14.31 ms | correctness pass | RL-071 |
| Custom TP2 pricing | naive ~1.18×, realistic-with-overlap ~1.3–1.5×; adversarial re-derivation at the ~102 µs/op barrier gives 70.7–76.6 tok/s (1.12–1.21×) | all values conditional estimates | RL-071, RL-105 |
| PP2 (Cascadia qwen35, layers 0-31 / 32-63) | 4.267 vs 4.278 tok/s median = −0.26 % | no speed win | RL-051 |
| llama.cpp RPC split | retains 54 % (76.1 → 40.8) | failed | RL-083 |
| Cross-card megakernel | blocked; inter-card TP persistent kernel over PCIe x8/x8 with no P2P | blocked | RL-007, ADR-0009 |

The physics behind the ceiling: hidden states are ~10–40 KB per round and the
card-to-card path is 11–12 GB/s copy-only, so a decode round's collectives cost
more than the compute they save unless they are captured and overlapped
(`docs/governance/standing-rules.md` §2.11; RL-071).

## 8. Operations and safety lessons

### 8.1 The host kills the session

systemd-oomd watches `user@1000.service` and kills its largest unit when that
slice's memory pressure stays above 50 % for 20 s. It killed the terminal
hosting the agent session twice (19:39 and 00:15 local, RL-158), and on
2026-10-12 a checker capped at 8 GiB inside a `systemd-run --user` scope
thrashed at its cap and lifted the whole user slice to 82 %. A cap inside the
user slice does not protect the session: thrashing in any child raises the
parent's pressure. Rule: run any job above 1 GB RSS, or any job reading model
shards, in the system slice (container with `--memory=<cap>`, or
`sudo systemd-run --scope -p MemoryMax=<cap>`). Source:
`docs/operations/storage-and-vram-safety.md` §7; RL-155, RL-158.

### 8.2 Hardware failure modes seen

- VRAM over-allocation does not fail cleanly: ~44 GiB demand against 31.89 GiB
  free produced 40+ GT resets in a loop, screen flashing, and a required reboot
  (RL-123).
- Out-of-bounds device gathers reset the engine: 1000 work-groups read past a
  427-entry table → 6× "Engine memory CAT error [18] class=ccs" → engine reset
  (RL-164, RL-165).
- Copy-engine (bcs) page fault with `Fault response: Unsuccessful -ENOENT` after
  unsynchronized `empty_cache` + cross-thread XPU use (RL-178).
- xe staleness after a hard kill: docker-killing a vLLM at boost leaves a stale
  max `cur_freq` readout (cosmetic only) (RL-069).
- AOT bmg builds can silently miscompile (RL-181).

### 8.3 Thermal and idle behaviour

| Finding | Value | Source |
|---|---|---|
| Awake idle draw | 45–52 W, flat at 400 vs 2800 MHz | RL-069 |
| Idle temps | 56–72 °C; loaded-idle rose 58 → 69 °C overnight at ~46 W | RL-065, RL-069 |
| ≤55 °C cooldown gate | **unreachable** on this host | RL-065, RL-069 |
| Suspend blocker | Xwayland holding `/dev/dri/card1` keeps runtime_status=active | RL-069 |
| Fix that worked | operator-authorized `b70-gpu1` PCI off/on in an inference-free window → 72 → 44 °C | RL-072 |
| Aggravator | 1 Hz energy polling wakes an idle card | RL-069 |

### 8.4 Storage and capacity

- SN770 measured 4,996 MB/s on the sybil staged-NVMe pattern (19.2 % of their
  26 GB/s reference); T-FORCE SATA 495 MB/s. The sybil tier was closed on
  storage class and space (RL-124).
- Large GGUF rule: a single GGUF over ~30 GB with full GPU offload can overflow
  VRAM and hard-crash the host; use a partial GPU-layer budget
  (`docs/operations/storage-and-vram-safety.md` §6).
- Post-load VRAM gate: below 500 MiB abort; 500–1023 MiB isolated capacity
  observation only; ≥2048 MiB for C1 performance; ≥3072 MiB for mixed serving.
- Cap discipline: record as-found caps (180 W GPU0 / 230 W GPU1 in most windows)
  and restore those, never a remembered 150 W (`docs/governance/standing-rules.md` §1.12).

## 9. Process lessons

1. **Sealed runs, sealed evidence.** Plan, commands, identities, raw paired
   samples, logs and cleanup go in one UTC result root; seal before registering.
   Missing evidence is `blocked`, never `pass` (AGENTS.md §12).
2. **Two attempts per unchanged mechanism.** Most questions ended at the attempt
   limit with an exact cause and a concrete re-entry condition: RQ-11 2/2,
   RQ-17 2/2 (then 4/4 across windows), RQ-25 2/2, RQ-27 R0 2/2, RQ-26 F0 2/2
   (RL-146–RL-150, RL-163, RL-164, RL-168, RL-169).
3. **Gate amendments must be declared before the run and disclosed.** RQ-27's s1
   sanity clause was amended from "top-1 ≥ 99.9 %, KL ≤ 1e-4" to an fp32
   fold-exactness probe because the original clause measured bf16 storage noise
   (RL-167). RQ-29's K2 gate was widened by an operator amendment, made after
   the K2 result and recorded as such (RL-176, RQ-29 success_gate).
4. **Never co-schedule CPU-heavy work beside a sealed GPU run.** Four
   co-scheduled lanes beside the Strata server pushed the user slice past the
   oomd threshold and killed the run; a separate attempt was contaminated the
   same way (RL-158, RL-160).
5. **Agent coordination failures were a top failure class, and each has a
   recorded cause:** a foreign session's `arm.sh`/llama-stream loop held both
   cards all day (RL-137, RL-139); a lane stopped a non-owned container without
   fixing its `unless-stopped` restart policy (RL-132); a parser produced 0/0/0/0
   placeholders that were mistaken for measurements (RL-147); a disk reclaim
   deleted a cited path and broke `research.py check` (RL-154); a batch claims
   audit found 46 findings across the docs (17 wrong numbers, 12 superseded
   methods, 14 missing qualifiers, 3 taxonomy) (RL-039).
6. **Do not price a lever and then adopt it on arithmetic.** The k=3 clamp
   looked free on paper and cost 5.45 % (RL-044). The harness cost rule: a
   sanity control that compares a bf16-stored transformed model to bf16 logits
   measures storage noise (RL-167).

## 10. Open questions and next steps

### 10.1 RQ status (2026-10-12)

| RQ | Priority | Status | Resolution | Title (short) |
|---|---|---|---|---|
| RQ-14 | 1 | closed | passed | Flash-Next route replacement: Strata one card vs llama.cpp dual-split |
| RQ-17 | 2 | ready | — | Flash-Next quality closure (attempt limit reached; needs named reset mechanism) |
| RQ-28 | 3 | closed | rejected | E8 lattice RVQ + two-sided scalar INT4 screen |
| RQ-26 | 4 | closed | rejected | HQ4 Hadamard-incoherent affine INT4 in DPAS layout |
| RQ-25 | 5 | closed | rejected | IQ3_XXS SoA layout for Xe2 |
| RQ-18 | 6 | gated | — | Flash-Next two-card `--layer-split`, all experts resident |
| RQ-20 | 7 | deferred | — | Cheap-dequant 4-bit streaming kernel (RQ-26's kernel half) |
| RQ-11 | 8 | closed | rejected | MoE on-device router + fused grouped dequant GEMM |
| RQ-22 | 9 | deferred | — | INT8 prompt GEMM for Strata SYCL |
| RQ-23 | 10 | gated | — | Flash-Next Coder, 50 % pruned, resident on one card |
| RQ-10 | 11 | deferred | — | Retained-prefix reuse without quality regression |
| RQ-21 | 12 | gated | — | Workload-matched speculation (suffix/ngram vs MTP) |
| RQ-05 | 13 | gated | — | Aggregate serving service levels on real work |
| RQ-24 | 14 | closed | rejected | Nemotron DFlash drafter INT4 |
| RQ-19 | 15 | closed | rejected | MXFP4 MoE decode on llama.cpp SYCL |
| RQ-16 | 16 | closed | passed | Two cheap external-engine audits (xe aliasing, draft tokens) |
| RQ-01/02/03 | 17–19 | deferred | — | EXL3 kernel probe program (operator-stopped, RL-122) |
| RQ-04 | 20 | deferred | — | Break the DPAS-vs-vector confound at small M |
| RQ-06 | 21 | gated | — | Custom TP2 over copy-only P2P (flag-slot collective) |
| RQ-07 | 22 | gated | — | 3bpw/K=3 as a bytes lever |
| RQ-08 | 23 | deferred | — | Draft-head byte levers (shortlist width, head_bits 6→4) |
| RQ-09 | 24 | deferred | — | W4A16 margin-aware requant |
| RQ-12 | 25 | deferred | — | Cross-model portability of proven lever classes |
| RQ-13 | 26 | gated | — | Independent reproduction before any public headline |
| RQ-15 | 27 | closed | rejected | Flash-Next tier build (sybil) storage gate |
| RQ-29 | 29 | ready | — | XTQ: rate-distortion-first trellis with Xe2-native code |
| RQ-30 | 30 | ready | — | EXL3-SC per-tensor bit allocation, served on the B70 |

Source: `registry/research-program.json` (reviewed through 2026-10-09) plus
RL-160…RL-182.

### 10.2 Next steps, in order

1. **RQ-30 serving:** profile the K≠4 M=4 verify kernels; the 0.70× MTP3
   position must move or SC5 becomes a quality-only tier (RL-182).
2. **RQ-29 XTQ:** run the K4 speed microbench and the K5 sealed quality screen on
   the `xe_ih2x2` code (1.71× cheaper decode, +9.9 % MSE) (RL-176, RL-181).
3. **RQ-17 re-entry:** pick one named reset mechanism (post-ready settle delay,
   raised xe ccs job timeout, or page-cache churn) and test it unsealed first
   (RL-166).
4. **RQ-18:** only after RQ-17 passes; the layer-split arm needs a placement
   decision because it takes both cards (RQ-18 gate).
5. **RQ-22:** a fused dequant+INT8 DPAS prompt GEMM microbench on Flash-Next
   expert shapes; the plain swap cannot clear its own gate (RL-131).

### 10.3 Never re-enter without a named changed variable

Symmetric INT4 W4A16 (defect in weights); DFlash2/DSpark on quantized 27B
targets without a drafter trained on the target's states; the EXL3 kernel probe
program (operator-stopped); stock TP2 and PP2; the acceptance frontier on the
old GPTQ stack (MTP-head retrain, probabilistic drafting, bf16 draft KV, foreign
shortlist, verify-head INT4, depth >4); W4A8 prefill; QuaRot→INT3 and s4×s4
decode; from-scratch grouped-IQ3; Strata host-staged second-card expert cache;
the storage-tier Flash-Next build; invented Q3DT and M5X.
Source: `docs/governance/standing-rules.md` §3.

## 11. Index of evidence

| Topic | Ledger rows |
|---|---|
| Hardware, topology, P2P, telemetry | RL-059, RL-065, RL-069, RL-071, RL-072, RL-123 |
| The law, rate class, instrument caveats | RL-030, RL-031, RL-090, RL-098, RL-099, RL-100, RL-101, RL-102, RL-105, RL-119, RL-126, RL-128, RL-129, RL-131 |
| EXL3 route (dense 27B) | RL-082, RL-084, RL-085, RL-086, RL-087, RL-088, RL-090, RL-094, RL-104, RL-106, RL-118, RL-119, RL-120, RL-121, RL-122, RL-125 |
| W4A16 / INT4 quality and requant | RL-024, RL-034, RL-079, RL-084, RL-087, RL-092, RL-107, RL-113, RL-114, RL-115, RL-116, RL-117 |
| Scalar vs lattice vs trellis (rate distortion) | RL-149, RL-150, RL-151, RL-161, RL-167, RL-168, RL-169, RL-170, RL-171, RL-172, RL-173, RL-174, RL-175, RL-176, RL-177, RL-181 |
| EXL3-SC per-tensor allocation and mixed-K serving | RL-175, RL-178, RL-179, RL-180, RL-182 |
| Speculation (MTP, draft, acceptance) | RL-017, RL-018, RL-026, RL-027, RL-028, RL-029, RL-035, RL-036, RL-037, RL-041, RL-042, RL-044, RL-045, RL-046, RL-047, RL-048, RL-049, RL-056, RL-058, RL-074, RL-076, RL-077, RL-093, RL-129, RL-130 |
| Kernel work and attribution | RL-038, RL-040, RL-043, RL-050, RL-054, RL-060, RL-061, RL-062, RL-063, RL-064, RL-066, RL-067, RL-072, RL-095, RL-096, RL-097, RL-098, RL-099, RL-100, RL-101, RL-102, RL-111, RL-112, RL-145 |
| MoE, Flash-Next, Strata | RL-010, RL-011, RL-012, RL-023, RL-030, RL-031, RL-032, RL-052, RL-055, RL-057, RL-068, RL-070, RL-080, RL-081, RL-083, RL-121, RL-123, RL-124, RL-125, RL-126, RL-128, RL-131, RL-141–RL-144, RL-146–RL-148, RL-152, RL-156, RL-158, RL-159, RL-160, RL-162, RL-166 |
| Dual-card, TP2, PP2, RPC, megakernel | RL-007, RL-008, RL-009, RL-016, RL-051, RL-059, RL-071, RL-075, RL-083, RL-103, RL-104, RL-105 |
| Storage, capacity, host safety | RL-013, RL-032, RL-065, RL-068, RL-069, RL-115, RL-123, RL-124, RL-133, RL-136, RL-137, RL-138, RL-139, RL-140, RL-154, RL-158, RL-164, RL-165, RL-178 |
| External cross-checks (E0 only) | RL-020, RL-073, RL-121 |

## Contradictions found

Both sides cited. In each case the ledger entry is the authority and the prose
note or the other row is the disagreeing side.

1. **Did AutoRound-uniform beat EXL3 on KL?** RL-151 (validated) read
   AutoRound-uniform arms at 0.06–0.10× EXL3 KL on two blocks. RL-161
   (validated) corrected that: those arms were stored at 8.125 bpw because every
   tensor escaped to INT8; the only INT4-pure AutoRound arm was 12.7× *worse*
   than EXL3, and no HQ4 arm showed EXL3-class quality at ≤4.5 bpw. RL-151's
   ratios must not be cited (RL-151, RL-161).
2. **The EXL3 C1 rate of record.** RL-090 (validated) records 63.25 tok/s
   p512/g128 @230 W with yield 2.954 as the route's true token rate, and
   `models/STATUS.md` repeats it as "best recorded". `research/2026/b70-format-route-lessons-20260930.md`
   §2 records 75.7–81.0 tok/s C1 solo @230 W on the same route, and RL-174/RL-182
   measure plain 4.0 at 81.8 (MTP3) / 32.5 (no-spec) on a newer image. Both
   numbers come from the same cell and cap; the image, metric boundary and
   window conventions differ (RL-090, RL-174, RL-182, `b70-format-route-lessons-20260930.md`).
3. **Does INT4 still have a speed case?** RL-174 (validated_screen) measured
   W4A16 32.85 vs EXL3 32.53 tok/s no-spec (1.01×) and 71.5 vs 79.9 with MTP on
   vLLM XPU v0.30.0, and retired the 1.62× anchor. RL-116 (provisional, @180 W)
   measured W4A16 +6 % at p512 and −23 % at p8192, and RL-087 recorded the GPTQ
   champion at ~1.62× EXL3 decode. Different artifacts, caps and metric
   boundaries; only RL-174 is on the current image (RL-087, RL-116, RL-174).
4. **Was the draft-vocab shortlist engaged?** RL-106 (provisional) assumed the
   pruned draft head was absent from the retained incumbent and priced a
   2.86 GB/round shared-head re-read. RL-118 (validated) found 512 of 1940 vocab
   blocks = 26.4 % engaged in the champion's own startup log and set by
   `model.yaml`; the earlier byte budget was overstated ~2× (RL-106, RL-118;
   `research/2026/b70-architect-postmortem-20260930.md` §0).
5. **Did ADR-0011 reopen for quality?** RL-142 (failed) recorded the closure
   under the stop-rule marker `ADR-0011 REOPENED_QUALITY_FAILURE`. RL-143
   (provisional) ruled that RL-141's rejection was execution integrity (an
   adapter crash before any canary was scored) and the marker was **not**
   triggered. The registry also still lists RQ-17 as `ready`, while RL-160 and
   RL-166 say its attempt limit is reached and re-entry needs a named reset
   mechanism (RL-142, RL-143, RL-160, RL-166).
6. **What killed the run-2 battery?** RL-158 (blocked) attributed the lost run
   to systemd-oomd killing the launching terminal scope during replay rep2.
   RL-160 (validated) re-read the logs: the 503/400 responses at ~17:31Z
   coincide with an xe ccs engine reset at 17:30:57Z, and the oomd kill came
   9 minutes later; the 503s were a route reset, not only memory pressure
   (RL-158, RL-160).
7. **The M=4 round's pairing arithmetic.** `research/2026/b70-exl3-round-attribution-audit-20260929.md`
   quoted 44.1 ms at 63.25 tok/s with yield 2.79. RL-105 (blocked) re-derived
   from raw evidence: the correct pairings are 47.2 ms / 63.25 tok/s / yield
   2.954 (n=5 serving) and 44.10 ms / 70.1 / 3.134 (n=1 diagnostic); the 2.79
   yield exists in no window, and 34.46 ms is a 180 W isolated-op sum against a
   230 W wall (RL-090, RL-105).
8. **Is Q4_K_M exact-answer clean?** RL-117 (failed) lists "base bf16, EXL3
   trellis, OV int4 and Q4_K_M are all clean" with respect to the
   14-vs-30 code_execution defect. RL-079 (validated) measured the same Q4_K_M
   reference answering "14" twice while failing arithmetic 59-vs-60 on warmup
   and measured runs. Both can be true (different cases) but "clean" without a
   case list is not citable (RL-079, RL-117).
9. **The EXL3 3.00bpw "10.25 tok/s" classification.** RL-094 (validated)
   reports 10.25 tok/s p512/g128 @230 W as diagnostic-only because the JSON-GP
   load path was unsupported and the run used `EXL3_BACKEND=triton`. The
   standing format guide's next-candidate row for the dense regime still named
   sub-4bpw work as a bytes lever (RL-067 closed the W3 kernel half). The
   diagnostic number must not be compared to the 63.25 incumbent as a speed
   result (RL-094, RL-067).

## What I could not verify

- Resolved by the lead: the MTP head is 8 EXL3 trellis modules at K=4
  (`mtp.fc` plus q/k/v/o and gate/up/down of the one MTP layer) in both the
  plain 4.00bpw and SC_5.00bpw_H6 artifacts (index inspection 2026-10-12), and
  the serving logs print `exl3: 409 EXL3 modules (8 in MTP head)` and
  `MTP draft head uses 512 of 1940 vocab blocks (26.4% of lm_head)`
  (`results/2026/10/rq30-sc5-mixedk-serve-20261012-UNSEALED/evid/server_*.log.gz`).
- `[UNVERIFIED]` The draft share "~8–10 % of round bytes" is a ratio I derived
  from the postmortem's corrected draft ≈1.4 GB/round against the M=4 verify
  pass 13.126 GB (≈10.7 %) and the 44.1–47.2 ms wall (≈6 %). The postmortem
  itself does not state a percentage
  (`research/2026/b70-architect-postmortem-20260930.md` §0 table).
- Resolved by the lead: the "1.49–1.51× on real tensors" figure is EXL3's
  relative weight MSE on layer-2 `in_proj_qkv` (5.890e-3) and `mlp.down_proj`
  (5.833e-3), measured with the pinned `ref.py`, divided by the Gaussian D(4);
  source `research/2026/b70-run4-agent-reports-20261012/format-theory.md`
  (no ledger row). Note the Gaussian Monte Carlo figures differ by rate:
  EXL3 mul1 is 1.178× D(4) at K=4 and 1.12× D(2) at K=2 (RL-176). Real
  tensors sit higher than i.i.d. Gaussian because LDLQ shapes the error toward
  the Hessian, not toward plain MSE.
- `[EXTERNAL]`, never local: the ~570 GB/s FP16 GEMV and 493 GB/s aligned Q6_K
  streaming references, the Strata B70 31/41 tok/s release figures, the sybil
  tier build, the neural.download scoreboard, and the exl3xpu int8-prefill
  1.89× figure (RL-020, RL-073, RL-121, RL-126).
