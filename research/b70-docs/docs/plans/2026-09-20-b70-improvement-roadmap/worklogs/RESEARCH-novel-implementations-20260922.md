---
title: "RESEARCH — Novel Implementable Speedup Techniques for B70 (deep online wave, 2026-09-22)"
type: worklog
date: 2026-09-22
status: provisional
scope: llama.cpp SYCL DSV4-Flash K216 IQ3_XXS (prefill+decode) and vLLM XPU Qwen3.8-27B GPTQ-INT4 + MTP4
provenance: desk research (web + GitHub PR mining + local tree audit at ~/llama.cpp pin 1692f9e50); NOTHING here is measured on our host
---

> **PROVISIONAL — NOT FOR PUBLIC HEADLINE.** All effect sizes are desk estimates with stated
> arithmetic, anchored where possible to third-party measurements on B70/BMG-class silicon.
> Nothing in this file has been run here. Every idea carries a cheap discriminating probe that
> must pass before any engineering.

## 0. Executive summary — the single biggest finding

**Upstream landed (or is landing this week) the exact kernels we identified as "the only remaining
lever" — our pinned tree is stale relative to them.** A local audit of `~/llama.cpp`
(pin `1692f9e50`, dated 2026-08-14) shows the pin does **NOT** contain:

| Upstream change | State | What it does | Reported effect (their HW) |
|---|---|---|---|
| #29245 `sycl: add grouped MoE XMX GEMM` | **OPEN, created 2026-09-21** | One grouped `joint_matrix`/XMX kernel for all active experts; dequantizes IQ4_NL/IQ3_S to f16 **tiles in local memory** (never materializes a whole expert); VNNI-packed activations | pp **+34.4%** (274→368 tok/s), Qwen3.8-Flash-Next UD-IQ4_XS, decode unchanged |
| #29107 `sycl: IQ3 code reorder` | **OPEN, created 2026-09-18** | Persistent reordered layouts for **IQ3_XXS/IQ3_S MMVQ** + reorder-aware dequant for prefill | B70, tg256@9k ctx **12.30 → 21.59 tok/s (+75%)**; @64k +53% |
| #23142 `SYCL: improve MoE prefill throughput` (merged 2026-05-22) | merged, **NOT in pin** | Device-side contiguous expert-row mapping (no `ids` scan, no device atomic) + O(n) device counting sort | pp **+70%** (308→548 tok/s, Arc B60, Qwen3.6-35B Q4_K_S) |
| #24452 reordered Q4_K/Q5_K/Q6_K MoE MUL_MAT_ID (merged 2026-06-16) | merged, **NOT in pin** | Persistent reordered expert layout for K-quant MUL_MAT_ID | Arc Pro B70 NX2: Q5_K_M decode 81.7→85.8 t/s, Q4_K_M 85.7→91.0 t/s |
| #22147 Battlemage AOT + MMQ subgroup annotations (merged 2026-05-09) | merged, **NOT in pin** | BMG-targeted MMQ kernel tuning | n/a |
| #22119 mul_mat_id memory efficiency + BF16 fast path (merged 2026-04-22) | merged, **NOT in pin** | mul_mat_id copies/efficiency | n/a |
| #24676 MoE prefill async memcpy use-after-free fix (merged 2026-06-18) | merged, **NOT in pin** | correctness under async prefill copies | n/a |
| #27847 TOP_K row split (merged 2026-08-30) | merged, **NOT in pin** | top-k routing kernel for large vocab | n/a |

This converts our roadmap item "write a grouped-IQ3 kernel from scratch" into
**"adapt/assemble three open-source kernels that other people just wrote for this exact card."**
That is a change of implementation cost class: weeks → days.

---

## 1. Ranked idea table

| # | Idea | Mechanism | Expected effect on us (arithmetic) | Impl. cost | Discriminating probe (≤15–30 min) | Primary source |
|---|---|---|---|---|---|---|
| 1 | **Grouped MoE XMX GEMM, adapted to IQ3_XXS** (lift PR #29245; extend IQ3_S→IQ3_XXS) | One grouped `joint_matrix` kernel over active expert tiles; tile-local dequant in SLM; VNNI activation packing. Kills the 100 MB/expert FP16 materialization + per-expert GEMM inefficiency | Routed op is 86.6 ms/layer = 3.72 s of 5.06 s ubatch at 1.79 TFLOPS. Same-card dense GEMM = 46 TFLOPS. Their IQ4_XS pp gain is +34%; on us the headroom is larger (25× op gap). Band: routed segment → 0.7–1.9 s ⇒ **pp end-to-end ≈ 1.7–2.5×** (131 → 220–330 tok/s at ub1024) | 2–5 days (their source is drop-in for IQ3_S/IQ4_NL; XXS tile loader is a port of `vec_dot_iq3_xxs` grid logic already in our `vecdotq.hpp`) | Build #29245 on a clean worktree of our pin; run its own benchmark mode on **one** DSV4 ubatch (perf-k216-prefill-n512 gate). Kill if grouped IQ3_S build doesn't beat oneMKL path by ≥1.15× on pp | https://github.com/ggml-org/llama.cpp/pull/29245 |
| 2 | **IQ3 persistent reorder for decode (MMVQ)** — cherry-pick PR #29107 onto our pin | Persistent reordered weight layout makes MMVQ loads coalesced; removes the reorder-on-the-fly cost that today hits every token | Their B70 number: +75% tg256@9k, +53% @64k on IQ3. Our decode 9.58 → **13–16 tok/s** if even half transfers to the 216-expert shape (their model unconfirmed) | 1–2 days (patch touches `mmvq.cpp/vecdotq.hpp/quants.hpp` — overlaps our uncommitted device-ID cases; needs manual merge) | Worktree + cherry-pick; `test-backend-ops -o MUL_MAT_ID`; 10-min tg probe vs current at fixed 9k-equivalent context. Kill if < +10% | https://github.com/ggml-org/llama.cpp/pull/29107 |
| 3 | **Backfill the merged SYCL MoE stack** (#23142, #24452-idea extended to IQ3, #22147, #22119, #24676, #27847) | #23142 moves prefill expert-row mapping device-side (removes host counting-sort + host syncs from OUR prefill path); #24452's reorder layout is the load-time analog for MUL_MAT_ID; #22147 gives BMG-tuned MMQ subgroup sizes | Prefill non-GEMM segment = 5.06−3.72 = 1.34 s of a 5.06 s ubatch. #23142 got +70% pp on a B60 config where mapping dominated; on us expect **+10–30% pp** alone and it is the prerequisite scaffold that idea 1 composes with. Also unblocks future graph capture (fewer unsupported nodes) | 1–3 days (conflicts with our local dsv4-hc/device-ID patches likely but shallow) | Cherry-pick #23142 alone; re-run perf-k216-prefill-n512.txt gate; compare ubatch wall and the op timing (op should be unchanged, wall down) | https://github.com/ggml-org/llama.cpp/pull/23142 |
| 4 | **Spec-decode depth + draft-precision co-sweep on vLLM Qwen3.8** (we have W4A16-g128 draft at +37–46%; push depth k=5–6 and/or g32/W3 draft) | With capture on and draft W4A16, marginal draft cost per step is small; verify cost scales linearly with k but accepted tokens grow ~log — there is an optimum we have not swept on the NEW (fast-draft) operating point | If acceptance length stays ≥3.4 at k=6 and draft adds <12% step time: **+8–15% decode** on top of the +37–46%. Literature: EAGLE-3 draft quantization showed **identical acceptance rates quantized vs not** (0.34), so quality risk of going below W4-g128 is low but measurable | 0.5–1 day (config + existing campaign harness; possibly a small g32 requant of draft-side only) | Matched p512/g128 + p8192/g128 n=3 sweep of `num_speculative_tokens` ∈ {4,5,6} with current W4A16 draft, capture ON; then g32 draft variant. 30 min | https://arxiv.org/abs/2505.xxxxx (Speculative Decoding Meets Quantization), https://github.com/ggml-org/llama.cpp/pull/29107-style evidence; our own p4-champion-AB evidence |
| 5 | **TurboQuant 4-bit KV cache on vLLM XPU** (Walsh-Hadamard rotation + Lloyd-Max 4-bit KV; PR merged into vLLM #38479; Intel published the XPU deployment recipe 2026-06-12; AMD productionized) | Data-oblivious 4-bit KV with near-BF16 accuracy ⇒ ~6× KV bytes reduction ⇒ 5–6× more KV capacity per card AND proportionally faster long-context attention reads | Qwen3.8-27B GPTQ INT4 on 32 GB: KV is the capacity wall at long ctx / Cn. 6× KV ⇒ e.g. ~8k→~48k ctx per C1 card or 4–6× concurrent seqs at same ctx; long-ctx decode attention reads shrink 6× ⇒ **+15–30% tok/s at ≥32k ctx**; at short ctx ~0 | 1–2 days if the Intel XPU article's steps work with our pinned vLLM image; else blocked on image rebuild | Grep our vllm-xpu image for the TurboQuant kernels (`pip show`/import test); if present: one p8192 C1 run BF16-KV vs TQ4-KV, compare tok/s + KV peak. Kill if XPU kernels absent AND image rebuild >1 day | https://community.intel.com (Long-context high-concurrency LLM serving on Intel XPU, Jun 2026); https://github.com/rocm-blogs TurboQuant productionization; vLLM PR #38479 |
| 6 | **P/D disaggregation across our 2 independent stacks — for Qwen3.8 only** (one card = prefill specialist, other = decode specialist, KV via host) | ADR-0008 independent stacks + DistServe/Splitwise goodput result: separating P and D removes TTFT×TP interference at Cn; KV transfer on our 8.9 GB/s H2D is trivial vs model sizes (KV/token ≪ weights) | Aggregate Cn goodput **+30–80% at C4+** (paper class) IF interference today is real. **DSV4 K216 CANNOT do this (83 GB needs both cards)** — this is strictly a Qwen3.8-class lever | 3–7 days (vLLM P/D connectors exist upstream; XPU untested; host-staged KV pipe) | Measure today's interference first: C1 decode loop on card A while card B absorbs a 8k-token prefill burst; if decode tok/s drops <5%, the idea has no headroom → kill in 15 min | https://arxiv.org/abs/2401.09670 (DistServe), https://arxiv.org/abs/2309.14717 (Splitwise) |
| 7 | **GEMQ/MxMoE-style 2-bit expert artifact + oneDNN s4/s8 grouped matmul** (future-artifact bet) | oneDNN ≥3.5 supports s4/u4 **as weight type in quantized matmul on Intel GPUs** (DPAS, no materialization). A GEMQ/MxMoE artifact (2-bit experts, 4-bit attention/shared) would fit ~200B MoE in 65 GB AND run a native GEMM path | Speculative quality bet. If quality passes: **vLLM W4A16-class serving of a DSV4-class MoE becomes possible at all** on this host; pp potentially 10×+ (native DPAS vs 1.79 TFLOPS). If quality fails: dead | 1–2 weeks incl. quantization study | Offline: GEMQ-style 2-bit quantize ONE MoE layer + 4-bit attention; wiki.test chunk PPL delta vs IQ3_XXS reference. Kill if PPL blows up (>5% relative) | https://arxiv.org/abs/2501.xxxxx (GEMQ), MxMoE ICML 2025 (Duanmu et al., https://proceedings.mlr.press), oneDNN data-types doc (s4/u4 weight support) |
| 8 | **Ragged-tile expert scheduling** (skip padded/empty expert tiles inside the grouped kernel) | OpenCL PR #25433 pattern: MoE prefill wastes tiles on experts with few/zero tokens; skip them | Fold into idea 1 for free-ish extra 5–15% of the grouped-kernel time at skewed routing | included in idea 1 | Same probe as idea 1, skewed (real) prompt vs uniform prompt comparison | https://github.com/ggml-org/llama.cpp/pull/25433 |
| 9 | **SYCL graph record/replay for MoE decode** (#28725 + #25089) | Port of CUDA-graph capture to SYCL; #25089 un-blocks MUL_MAT_ID fused path in graph compat check | **Upstream already measured: negligible on B70** (significant on A770). Matches our own closed finding. Do not spend time except as a free side effect of ideas 1–3 | 0 (monitor) | None — CLOSED by third-party evidence | https://github.com/ggml-org/llama.cpp/pull/28725 |
| 10 | **FLUTE/HIGGS-style universal-LUT dequant inside idea-1's kernel** (design note, not separate build) | IQ3_XXS dequant is already grid-LUT based; FLUTE shows LUT-based dequant keeps tensor cores fed at near-Marlin rates with trivial kernels. Idea: preload 256-entry LUT into SLM once per WG instead of re-deriving bits per tile | Kernel-design guidance for idea 1; no standalone effect | 0 | n/a | https://github.com/HanGuo97/flute (EMNLP 2024 Findings), HIGGS (NAACL 2025) |

---

## 2. Idea detail sections

### 2.1 Grouped MoE XMX GEMM adapted to IQ3_XXS (rank 1) — the prefill lever, now with source to lift

**Mechanism.** PR #29245 (created 2026-09-21, author `cwriter`, AI-assisted, unmerged) replaces
"one GEMM per expert" with one grouped `joint_matrix` (XMX/DPAS) kernel over a host-side schedule
of active expert tiles, dequantizing **IQ4_NL or IQ3_S** weights into f16 tiles in **local memory**
and packing routed activations into VNNI f16 for DPAS. This is precisely the Marlin design point
(global→SLM ring → in-tile dequant → tensor core; cf. MARLIN arXiv:2408.11720, and the Red Hat
writeup of its register-dequant technique) translated to Intel's `joint_matrix` API — and it is
the "grouped-IQ3 kernel that never materializes whole experts as FP16" our prefill analysis called
the only remaining lever.

**Why it should pay more for us than their +34%:** their test (Qwen3.8-Flash-Next UD-IQ4_XS) had a
moderate op-efficiency gap; our DSV4 routed op sits at **1.79 TFLOPS vs 46 TFLOPS dense on the same
card (~25×)**, because we additionally materialize ~100 MB of FP16 expert per op (IQ3→FP16 ≈ 5.2×
byte expansion) and feed oneMKL per-expert GEMMs. The UB512 routed budget is 3.72 s of a 5.06 s
ubatch (73%). If the grouped kernel lands anywhere from 5 TFLOPS (dp4a-class vector) to 20 TFLOPS
(DPAS-class), the op drops 28.87 ms → 2.6–10.4 ms and pp end-to-end goes **1.7–2.5×**. Even the
pessimistic end clears any gate we would set.

**Gap we must fill: IQ3_XXS.** The PR ships IQ3_S and IQ4_NL tile loaders; our artifact is
IQ3_XXS. The dequant logic for XXS already exists in our tree in SYCL form
(`ggml/src/ggml-sycl/vecdotq.hpp`: `vec_dot_iq3_xxs_q8_1` with `iq3xxs_grid`), so the new tile
loader is a bounded port (grid-LUT gather in SLM; FLUTE's universal-LUT pattern is the design
reference). Our pinned tree's SYCL MMQ (`ggml/src/ggml-sycl/mmq.cpp`, 3,030 lines) has the full
tile-template pattern for 10 K-quant types to copy structure from, but **no IQ types at all** —
that is exactly the hole.

**Cost:** 2–5 days. **Probe (do this first):** clone PR #29245 into a worktree of our pin (it
touches only `common.hpp`, `fused-gemm.*`, `ggml-sycl.cpp` — minimal overlap with our dsv4-hc
patches), build, and run ONE DSV4 prefill ubatch against the existing
`results/2026/09/b70-improvement-probes-20260920/perf-k216-prefill-n512.txt` gate. If the IQ3_S
build (test artifact or XXS shim) does not beat the oneMKL path ≥1.15× on pp, stop.

**Sources:** PR https://github.com/ggml-org/llama.cpp/pull/29245 · MARLIN https://arxiv.org/abs/2408.11720 ·
Marlin design https://developers.redhat.com/articles/2024/04/17/how-marlin-pushes-boundaries-mixed-precision-llm-inference ·
FLUTE https://github.com/HanGuo97/flute

### 2.2 IQ3 persistent reorder for decode (rank 2)

**Mechanism.** PR #29107 (open, 2026-09-18) adds persistent reordered layouts for IQ3_S/IQ3_XXS
MMVQ on B70 plus reorder-aware dequant for prefill. Reorder is done once at load; per-token MMVQ
then reads coalesced instead of striding through the interleaved IQ3 block layout. Reported on B70
(oneAPI 2026.1.1): tg256 12.30→21.59 tok/s at 9k ctx (+75%), 9.76→14.92 at 64k (+53%), 7.88→10.93
at 126k (+39%). Prefill unchanged (it only accelerates the dequant/read side there).

**Caveats.** (a) Their benchmark model is unnamed ("IQ3 workload") — may not be the 216-expert
monster; treat +75% as an upper bound and +25–50% as the planning number ⇒ 9.58 → **12–16 tok/s**.
(b) The PR is AI-assisted and unmerged — gate with `test-backend-ops -o MUL_MAT_ID` and a KL-div
spot check. (c) It edits `mmvq.cpp/vecdotq.hpp/quants.hpp`, which collide with our uncommitted
device-ID cases for IQ3_XXS/IQ3_S — expect a manual merge, and the reorder layout may obviate part
of our fallback machinery.

**Cost:** 1–2 days. **Probe:** worktree + cherry-pick + 10-min tg measurement at matched context.

**Source:** https://github.com/ggml-org/llama.cpp/pull/29107

### 2.3 Backfill the merged SYCL MoE stack (rank 3)

Our pin (2026-08-14) predates/omits several merged SYCL MoE improvements (audit in §0). The single
most valuable is **#23142**: it moves the prefill expert-row mapping device-side (precomputed
contiguous per-expert row slices, no `ids` scan, no device atomic) and replaces the O(n_as·n_rows)
contraption with a device counting sort. On our host the contract describes the prefill mapping as
**host-side** ("host counting-sort grouping") — i.e., we are running the pre-#23142 shape, host
syncs included. Their +70% (B60, Q4_K_S) came from a mapping-dominated config; our mapping share is
the 1.34 s non-GEMM ubatch remainder, so expect **+10–30% pp standalone**, plus: (a) it is the
scheduling scaffold idea 1 composes with, (b) removing host syncs from prefill is the prerequisite
for any future prefill graph capture (their #28725 notes library-GEMM-in-graph is already a hard
blocker, so this is enabling, not paying, today). #24452 proves the reorder-layout approach at the
MUL_MAT_ID level for K-quants (B70: +5–6% decode); its IQ extension is superseded by #29107/#29245.
#22147 (BMG MMQ subgroup annotations) and #27847 (top-k row split) are free rides in the same
cherry-pick series.

**Cost:** 1–3 days incl. conflict resolution against our dsv4-hc/device-ID local patches.
**Probe:** cherry-pick #23142 alone; re-run the prefill gate; op time must be unchanged while
ubatch wall drops.

**Sources:** https://github.com/ggml-org/llama.cpp/pull/23142 · https://github.com/ggml-org/llama.cpp/pull/24452 ·
https://github.com/ggml-org/llama.cpp/pull/22147 · https://github.com/ggml-org/llama.cpp/pull/27847

### 2.4 Spec-decode depth + draft-precision co-sweep (rank 4)

We measured draft-side W4A16-g128 = +37–46% decode (capture ON, without which everything is inert).
The measured fact that **capture is the enabler** and the external evidence that **draft
quantization is nearly free acceptance-wise** (EAGLE-3 prototype: identical 0.34 acceptance
quantized vs not, draft cost halved; systematic study: "Speculative Decoding Meets Quantization",
arXiv May 2025) re-open the operating-point question: with the draft now ~2× cheaper, the optimal
speculation depth shifts longer (verify pays k× KV reads + logits, acceptance grows sub-linearly;
the optimum k moves as draft cost falls). Sweep k∈{5,6} with the current draft, then draft g32 (and
optionally W3 via a LUT format) as a second axis. Expected **+8–15% decode** on top of the current
route if acceptance length holds ≥3.4. Quality risk is bounded: acceptance is measured directly by
the sweep; acceptance-collapse = instant kill.

**Cost:** 0.5–1 day on the existing MTP campaign harness (matched p512/g128 + p8192/g128 n=3, fresh
n=5 finalist per protocol). **Probe:** the k∈{5,6} n=3 cells themselves — 30 min.

**Sources:** EAGLE-3 quantized-draft evidence (vLLM/AMD blog 2026-07, https://vllm.ai) ·
"Speculative Decoding Meets Quantization" (arXiv 2505, May 2025) · our
`results/2026/09/b70-improvement-probes-20260920/p4-champion-ab-*`

### 2.5 TurboQuant 4-bit KV on vLLM XPU (rank 5)

TurboQuant (vLLM PR #38479) is data-oblivious KV-cache quantization: random Walsh–Hadamard
rotation + Lloyd–Max scalar quantization, ~6× KV compression at near-BF16 accuracy (vLLM accuracy
study 2026-05: holds up on long-context/reasoning workloads with defined degradation envelopes).
Intel published a concrete "serve vLLM with TurboQuant on Intel XPU" recipe (community.intel.com,
2026-06-12) and AMD productionized it on ROCm — i.e., the kernels are upstream and someone has
walked the XPU path. For Qwen3.8-27B GPTQ-INT4 on 32 GB, KV is the capacity wall for long context
and Cn concurrency; 6× KV ⇒ proportional ctx/seq-count headroom, plus attention-read reduction
(+15–30% tok/s at ≥32k ctx; ~0 at short ctx). This is the one genuinely NEW axis for the vLLM
stack that we have not touched (our llama.cpp KV policy q8_0/q4_1 is a different engine, already
4–8 bit).

**Cost:** 1–2 days if our pinned image contains the kernels; image rebuild otherwise.
**Probe:** import/grep the vLLM install for TurboQuant modules; if present, one p8192 C1 A/B
(BF16-KV vs TQ4-KV) with KV peak and tok/s. **DSV4 note:** K216 KV is served by llama.cpp, where
our q8/q4_1 policy already applies — TurboQuant is a vLLM-stack lever only.

**Sources:** https://community.intel.com (TurboQuant on Intel XPU, Jun 12 2026) ·
https://github.com/vllm-project/vllm/pull/38479 · ROCm productionization blog (Jun 2026) ·
vLLM accuracy study (May 2026)

### 2.6 P/D disaggregation for Qwen3.8 across the two independent stacks (rank 6)

DistServe/Splitwise established that splitting prefill and decode into disjoint resources lifts
aggregate goodput ~1.3–2× at moderate concurrency by removing TTFT/TP interference. Our host is
unusual: the sanctioned default (ADR-0008) is already two independent single-card stacks, and
Qwen3.8 GPTQ-INT4 fits one card — so a "card A prefill-only, card B decode-only, KV staged through
host" topology is topologically legal for Qwen3.8 (and ILLEGAL for DSV4: 83 GB needs both cards,
full stop). vLLM has upstream P/D connectors (LMCache/KVTransfer); XPU is untested; our 8.9/7.9
GB/s H2D floor carries KV trivially (KV/token ≈ 10²–10³ KB ≪ weights). **Kill-first economics:**
if today's two-stack Cn shows <5% decode degradation under concurrent prefill load on the other
card, there is no interference to remove. **Cost:** 3–7 days if the probe passes.

**Probe (15 min):** C1 decode loop on card A; fire an 8k-token prefill burst on card B; measure
decode tok/s delta. **Sources:** https://arxiv.org/abs/2401.09670 · https://arxiv.org/abs/2309.14717

### 2.7 Sub-4-bit expert artifact + native oneDNN s4/s8 matmul (rank 7, future-artifact bet)

oneDNN ≥3.5 supports s4/u4 **as the weight type of quantized matmul** on Intel GPUs (DPAS-backed,
no materialization) — per our rate-class law this adds no TOPS vs s8, but as a
bandwidth/capacity format with native GEMM it is the legitimate endpoint for MoE expert storage.
The catch for the CURRENT artifact is capacity arithmetic: DSV4-Flash K216 at IQ3_XXS (≈3.06 bpw)
is 83 GiB ⇒ ≈200B-class params ⇒ a 4.25-bpw s4/g128 artifact would be ~106 GB — **does not fit
65 GB**. The only capacity-legal native-format route is a 2-bit-class expert format (GEMQ shows
global expert-mixed-precision with router adaptation retaining quality; MxMoE ICML 2025
auto-assigns per-expert precision and generates matching mixed-precision GroupGEMM kernels).
This is a 1–2-week study with a real quality cliff — run it only if ideas 1–3 stall.

**Probe (1 hr, offline):** 2-bit+4-bit-mixed quantize ONE layer; wiki.test chunk PPL delta vs the
IQ3_XXS reference. **Sources:** MxMoE (Duanmu et al., ICML 2025) · GEMQ (OpenReview 2026) ·
oneDNN data types (s4/u4 weight support, v3.14 doc)

### 2.8 Design notes worth keeping (rank 8–10)

- **Ragged-tile expert scheduling** (OpenCL PR #25433): skip padded/empty expert tiles; fold into
  the grouped kernel for a few % more at skewed routing.
- **FLUTE/HIGGS universal LUT** (EMNLP 2024 / NAACL 2025): LUT-in-registers dequant that keeps
  tensor cores fed; our IQ3_XXS grid-LUT dequant maps directly onto this pattern inside idea 1's
  kernel. Zero standalone cost — design guidance only.
- **IPEX/vllm-xpu-kernels W4A16 ESIMD GEMM** (WoqGemm, int4 g128, ~1.3× improved in IPEX
  2.8.10): this is the kernel already under our Qwen3.8 GPTQ route — confirms the dequant-in-tile
  design works on BMG silicon at scale; no action.
- **Marlin direct port:** unnecessary given #29245; Marlin's *design* (async global loads, SLM
  ring, register dequant) is the checklist for validating idea 1's kernel.

---

## 3. What I could not find / negative results (explicitly checked)

1. **2:4 structured sparsity on Xe2 XMX — NO native support found.** No Intel doc, ISA note, or
   oneDNN/oneAPI surface exposes sparse-DPAS on Battlemage (checked Intel oneAPI optimization
   guide, chips-and-cheese Battlemage deep dive, IGC/DPAS references; NVIDIA's 2:4 material
   dominates search results and does not transfer). Dead as a compute lever; consistent with our
   rate-class law.
2. **SYCL graph record/replay on B70 llama.cpp decode — upstream measured NEGLIGIBLE on B70**
   (PR #28725, author's own A770-vs-B70 data; graphs pay on A770/i915, not B70/xe). Independently
   confirms our closed item "SYCL graphs on dual+MUL_MAT_ID". Also documents that **oneMKL calls
   cannot be captured into a SYCL graph** ("Graph nodes cannot depend on events from outside the
   graph") — a structural reason prefill graphs are impossible until the GEMM is a custom kernel,
   which strengthens idea 1's strategic value beyond raw FLOPS.
3. **Full-model W4A16/INT4 artifact of DSV4-Flash K216 for vLLM — capacity-dead.** ~200B-class
   params ⇒ ~106 GB at 4.25 bpw > 65 GB combined. The Aug 2026 vLLM issue showing
   DeepSeek-V4-Flash running on Marlin MoE WNA16 paths is an 8-GPU-class deployment, not ours.
   The only capacity-legal vLLM route for this model would be the 2-bit expert study (§2.7).
4. **P/D disaggregation for DSV4 — capacity-dead** (needs one full model copy per role).
   Qwen3.8-only lever (§2.6).
5. **Machete (vLLM W4A16 kernel) — Hopper-WGMMA-specific; irrelevant to XMX. EXL3 — still no
   runtime (re-confirmed). AWQ loader — ecosystem ships compressed-tensors (re-confirmed; matches
   our P8 closure).**
6. **Marlin MoE WNA16 numerical-determinism caveat** (vLLM issue, Aug 2026): outputs depend on
   within-expert token ordering. If we ever reuse Marlin-derived paths, add determinism checks to
   correctness gates.
7. **IPEX weight-only-quant docs page 404s** (moved); the ESIMD WOQ GEMM source lives in
   intel-extension-for-pytorch `csrc/gpu` (WoqGemm) and is migrating into `vllm-xpu-kernels`
   (RFC #33214). Nothing actionable beyond what our GPTQ route already uses.
8. **LayerSkip/CLaSp self-speculative decoding:** not implemented in llama.cpp, and with MTP4 +
   quantized draft already beating it on this stack, no reason to chase.
9. **TurboQuant kernel availability in our pinned vLLM XPU image: UNVERIFIED.** The Intel recipe
   exists; whether our image generation contains the kernels is the first probe.

**Contradiction check against the closed list:** nothing found re-opens "device-ID repair as a
speed lever" (upstream's #28725 B70 data independently confirms launch overhead is not our
decode wall); nothing re-opens sub-8-bit-as-compute (rate-class law holds); nothing re-opens
TP2 or ubatch>1024. The one deliberate re-framing: **the grouped-IQ3 kernel was "write it
ourselves"; it is now "cherry-pick/adapt three upstream implementations" — same lever, cost
class collapsed.**

---

## 4. Top 3 to start tonight

1. **Cherry-pick PR #29107 (IQ3 reorder, decode +25–75%) onto a worktree of our pin.** Smallest
   diff, biggest single-number upside on our headline metric, self-contained files, 15-min probe
   (test-backend-ops MUL_MAT_ID + tg run). Watch the collision with our uncommitted device-ID
   cases.
2. **Clone PR #29245 (grouped MoE XMX GEMM) into a second worktree and run the prefill gate
   probe.** Do NOT integrate tonight — just get its IQ3_S grouped path executing one DSV4 ubatch
   and time it against `perf-k216-prefill-n512.txt`. This decides whether the prefill lever is a
   2-day cherry-pick+extend or a 2-week kernel project. If it clears ≥1.15×, extend the tile
   loader to IQ3_XXS using the `vec_dot_iq3_xxs` grid logic already in our `vecdotq.hpp`.
3. **Cherry-pick #23142 (device-side expert mapping + counting sort) and re-run the prefill
   gate.** Low risk, +10–30% pp standalone, and it is the scheduling scaffold idea 1 composes
   with (plus it removes the host syncs from our prefill mapping for free).

Sequencing note: probe 3 before integrating 1's prefill side; both touch the same MUL_MAT_ID
prefill plumbing from different ends (mapping vs GEMM) and are designed to compose
(mapping-schedule → grouped GEMM), which is also the shape upstream will likely converge on.

---

## 5. Appendix: audit commands and artifacts

- Pin audit: `git -C ~/llama.cpp merge-base --is-ancestor <merge> HEAD` for
  cc9e33121 (#23142), ac79caa7c (#24452), 4a4f819cb (#22147), 225088ea7 (#22119),
  ebbc1e51c (#24676), #27847 — all **NOT in tree**; pin `1692f9e50` dated 2026-08-14.
- Tree facts: `ggml/src/ggml-sycl/mmq.cpp` covers Q4_0…Q6_K only (no IQ types);
  `vecdotq.hpp` already carries SYCL `vec_dot_iq3_xxs_q8_1` / `vec_dot_iq3_s_q8_1` with grid LUTs;
  MXFP4/NVFP4 tensor types exist in this lineage (`getrows.cpp`, `convert.cpp`, `dequantize.hpp`);
  `GGML_SYCL_GRAPH` is compile-time-gated and off by default.
- Search trail: GitHub PR mining via `gh` (llama.cpp SYCL/MMQ/IQ3/MoE queries), Marlin/Machete/
  LUT-GEMM/FLUTE/QuaRot papers and repos, oneDNN s4/u4 docs, vLLM fused-MoE modular-kernel docs,
  TurboQuant (vLLM #38479, Intel XPU recipe, AMD productionization), EAGLE-3 quantized-draft
  evidence, DistServe/Splitwise, Xe2 sparse-DPAS absence checks.
