---
title: "RESEARCH — Novel implementations, deep online wave 2 (driven by tonight's upstream state)"
type: worklog
date: 2026-09-23
status: provisional
scope: vLLM XPU Qwen3.8-27B GPTQ-INT4 + MTP4 champion route; llama.cpp SYCL DSV4 K216 IQ3_XXS prefill/decode
provenance: desk research (web + GitHub PR/issue mining + local tree/build/host audit, zero GPU time spent); NOTHING here is measured
differs-from: RESEARCH-novel-implementations-20260922.md (wave 1) and RESEARCH-online-novel-levers-20260923.md — this wave only carries ideas those two do not, plus decision arithmetic on their open items
---

> **PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Desk estimates with stated arithmetic, anchored to
> third-party measurements where noted. Nothing in this file has been run on this host. Every idea
> carries a cheap discriminating probe that must pass before any engineering.

## 0. What this wave adds

Tonight's new upstream state: #29245 (grouped MoE XMX GEMM, tile-local SLM reconstruction, OPEN),
#29107 (IQ3 reorder, OPEN), #28476 + #23142 MERGED (device-ID dispatch, host counting sort), the
exact prefill waste quantification (21.74 GB/routed-layer), the measured draft-side INT4 win
(1.40–1.49x, iso-power), and **vLLM #54785** (XPU graph capture + MTP k=4 → non-deterministic wrong
greedy logits — a correctness hazard sitting on top of our champion recipe).

Wave 1 and the 2026-09-23 online wave already own: the grouped-IQ3 prefill kernel plan, #29107
cherry-pick, #23142 backfill, TurboQuant KV, P/D disagg (Qwen3.8-only), the #28725 graph scaffold,
#53997/#53989/#58082 stream fixes, vllm-xpu-kernels ≥0.1.15 bump, and the env-screen list. This wave
does NOT re-propose those. It adds: (1) the #54785 response and a guard design, (2) speculation
ECONOMICS with a quantized drafter — including an arithmetic verdict on the wave-1 k-sweep, (3) the
verify-pass head INT4 idea with its losslessness analysis, (4) where else tile-local reconstruction
applies and where it does NOT (audit results), (5) the oneMKL/graph-capture question answered with
launch-tax arithmetic, and (6) host-level ideas (page cache, pinning) sized honestly on the real
machine (60 GB RAM, oneAPI 2025.3, GGML_SYCL_F16 already ON — all verified locally today).

### Champion anatomy used by every arithmetic below (from `research/2026/qwen38-draft-side-int4-upstream-20260921.md`)

Qwen3.8-27B GPTQ-INT4 g128 target + MTP4, draft INT4 W4A16 g128, XPU graph capture ON, p512.
Target verify forward ≈ 15 GB (INT4 weights) + 2.54 GB (F16 LM head, 248,320 x 5,120) ≈ 17.5 GB
→ 28.8 ms @ 608 GB/s. Draft step ≈ 0.66 GB (INT4 head) + ~0.2 GB (W4 MTP body) + ~0.06 GB fp8 KV
≈ 0.9 GB → ~1.5 ms. Round (gamma=4): ~34.8 ms. Acceptance alpha in [0.55, 0.69] run-variable;
E[tokens/round] = (1-alpha^5)/(1-alpha) gives 2.39 at alpha=0.62 → ~14.6 ms/token ≈ the measured
72–77 tok/s (some L2/overlap). This model is self-consistent with every published cell.

---

## 1. Ranked idea table

| # | Idea | Mechanism (2 sentences) | Effect + arithmetic on OUR hardware | Cost | Kill probe (<=15 min) | Source |
|---|---|---|---|---|---|---|
| 1 | **Adopt the #54785 clamp (k=3 under capture) + always-on "spec-canary" determinism guard** | #54785 shows our exact champion stack (capture + MTP k=4) silently corrupts greedy output on captured bs=1 pieces; upstream's deployed mitigation clamps k to 3 whenever capture is active. A 2-request greedy canary with logprob compare, run at boot and before every campaign cell, detects the corruption class in ~15 s instead of after publication. | Clamp cost at alpha=0.62: tokens/round 2.39→2.243 (-6.2% ceiling) but the dropped draft step also saves 1.5 ms → net ~**-2 to -4% decode** (worst case alpha=0.69: -4.2% time-adjusted). That buys determinism on a route we intend to publish. | Config change + ~30 lines in the campaign harness | Run the issue's own repro on the champion: same 6-token greedy request x8; >=2 distinct outputs = corrupt boot (k=4 must then be clamped). ~10 min, request-only | https://github.com/vllm-project/vllm/issues/54785 |
| 2 | **Verify-pass head INT4 (target LM head -> W4A16 g128)** | The target's F16 head (2.54 GB) is re-read once per verify round; in MTP verify it only scores the k+1 positions (accept/reject + bonus token), so quantizing it is the same class of lever as the draft-head quant we already proved, but on the round-dominant read. Unlike draft-side quant, it perturbs the *verifying* distribution, so the lossless guarantee no longer holds and the change must be gated and labeled. | Round bytes 21.2→19.3 GB = **-9.0% → +10% decode ceiling, realistic +6-8%** (round 34.8→31.7 ms), plus 1.9 GB VRAM freed. Losslessness break: greedy flips require top1-top2 logit gap < ~2x(logit error of W4A16 g128 over 5,120 dims) — likely rare, but nonzero and unquantified. | 1-2 days, reusing the existing `draft_int4` runtime-patch machinery pointed at the target head (fail-closed TP gate already written) | 20 fixed greedy prompts, quantized-head arm vs F16-head arm, record first-divergence token position; median divergence > 2,000 tokens AND acceptance delta < 0.02 → promote to protocol campaign. Plus one logprobs=5 generation to histogram top1-top2 gaps | https://openreview.net/forum?id=1fZ0tGfsqb (why draft-side is lossless and verify-side is NOT), https://docs.vllm.ai (fp8 checkpoints exclude lm_head by convention — our head is F16 for exactly this reason) |
| 3 | **Zero-GPU optimal-gamma reanalysis from already-collected acceptance logs** | The Leviathan speedup formula E(gamma)/(gamma*c_d + c_v) with OUR c_d=1.5 ms, c_v=28.8 ms says the optimum is flat: gamma=4 is within ~1.5% of gamma=5-6 at alpha=0.69 and optimal at alpha<=0.62. Parsing the per-round accepted-length histograms already snapshotted in `p4-champion-ab-*` turns this from theory into our own measured curve at zero GPU cost. | Predicted: k-sweep (wave-1 item 4) buys **<=1.5%** — deprioritize it. Clamp to k=3 costs 2-4% (idea 1). The leverage is in alpha, not gamma: dE/dalpha at alpha=0.62, gamma=4 is **+9.6% throughput per +0.05 acceptance**. | 15 min of scripting, zero GPU | The script itself: if the empirical speedup(gamma) curve is flat (±2%) across gamma=3..6 on real logged rounds, the k-sweep is killed; if a real shape shows gamma>=5 winning by >5%, promote the sweep | https://arxiv.org/abs/2302.01318 (Thm 3.8), https://arxiv.org/abs/2405.19715 (SpecDec++), https://arxiv.org/abs/2411.18462 (SVIP, entropy-gated stop, +17%) |
| 4 | **Drafter op-selection audit: is every W4A16 linear actually hitting the fused kernel?** | Our runtime patch replaces the draft head's compute_logits and the MTP body's quant_method with `torch.ops._xpu_C.int4_gemm_w4a16`; any linear whose shape falls out of the fused kernel silently dequantizes to a BF16 GEMM at 4x the bytes. The measured round time (~34.8 ms) says the head is NOT falling back, but a single body-linear fallback (~0.2→0.8 GB/step) would still hide ~4 ms/round. | Confirmatory in the likely case (no 2x left); upside if a fallback is found: draft step 1.5→~2.8 ms → **+4-6% decode recovered**. Bound: even total draft fallback is 4x1.5→4x6 ms → -14%, i.e. the win we already measured bounds the damage. | 15 min, no new build | One 3-step torch-profiler capture on the running champion (existing `VLLM_TORCH_PROFILER_DIR` path), grep kernel names for `int4_gemm_w4a16` vs `aten::matmul` under the draft module; or grep the patch's load-time logs for per-module op selection | https://github.com/vllm-project/vllm/issues/33214 (op surface migrating IPEX→vllm-xpu-kernels; INT4 GEMM for GPTQ "still WIP" in the new lib — a real fallback risk during migration) |
| 5 | **#28725 SYCL graphs on MoE decode (post-#28476), with corrected expectations** | New evidence reopens graphs-for-MoE-decode: upstream reports +19.8% MoE decode with `SYCL_GRAPH_FORCE_NATIVE_RECORDING=1` + Level Zero patches, and #28476/#23142 (both merged) remove exactly the D2H memcpy+wait that made #28725 exclude `MUL_MAT_ID`. However our launch-tax arithmetic says most of their +19.8% was wait/sync tax, which we already measured at +3.54% — so our expected ceiling is much smaller. | Ceiling: ~700-900 launches/token x ~5 us = **3.5-4.3% of the 104 ms/token**; plus prefill is provably worthless (900 launches x 5 us = 4.5 ms of a 5.06 s ubatch = 0.09%). Gate at >=5% e2e, not 10%. OneAPI gap: our installed toolchain is **2025.3**, and the "library-GEMM event restriction lifted" needs oneAPI 2026 — an upgrade is a prerequisite for any non-native-recording path. | 1-2 days incl. worktree build | Worktree + cherry-pick + env vars, one 10-min tg cell vs current L5 champion; kill if <+5% | https://github.com/ggml-org/llama.cpp/pull/28725 |
| 6 | **Extend #29245's tile-local reconstruction to the dense (non-routed) IQ3 GEMMs of K216 — but only after a trace proves they materialize** | The 5.06 s ubatch is 73.5% routed ops (3.72 s) and 26.5% other (1.34 s: attention/DSA, dense projections, norms). If dense `mul_mat` on IQ3_XXS weights also runs the dequantize-whole-tensor→oneMKL pattern (as the routed ops did), the same tile-local SLM loader applies; if they already run a fused MMQ path, there is nothing to win. | Bounded: even a 1.5x on a 0.5 s GEMM share of the 1.34 s segment = -0.17 s ≈ **+3.4% pp**. Cheap to know, small to win; do not schedule before ideas in the main prefill line. | 15 min trace + (only if positive) 2-4 days | SYCL profiler trace (or `GGML_SYCL` profiling envs) of ONE n512 ubatch: if `dequantize`-kernel + MKL `gemm` pairs appear for dense `mul_mat` on IQ3 → extend the tile loader; if a fused MMQ kernel appears → dead, record it | https://github.com/ggml-org/llama.cpp/pull/29245 |
| 7 | **Page-cache residency management for the `-ncmoe` CPU-expert segment of the 83 GB GGUF** | With `-ncmoe 15`, ~31 GB of IQ3 expert bytes are served from mmap; on this host (verified: 60 GB RAM, 39 GB already in cache, 3 GB swap in use) they fit in the page cache but with little slack, and a cold cache costs a one-time ~31 GB NVMe read at first decode. `posix_fadvise(WILLNEED)`/`madvise` on just the expert byte ranges at load (plus keeping evidence writes from evicting them) removes cold-start faults and first-epoch jitter without the banned whole-file `--mlock`. | Cold start: ~31 GB @ ~5-7 GB/s ≈ **4.5-6 s saved once per load**; steady-state: guards against eviction-induced jitter, which on a 60 GB host with 83 GB mapped is a real tail-latency risk, not a throughput lever. | 0.5 day (small loader hook) or 0 (pre-warm the byte ranges with `dd`/vmtouch before the campaign) | `vmtouch` on the expert byte ranges vs not, measure cold TTFT of the first decode request twice; kill if cold-start delta < 2 s | https://manpages.debian.org/unstable/llama.cpp (mmap semantics), https://sheldylew.com (cold-page-fault TTFT case study) |
| 8 | **Remaining llama.cpp env screens on the new kernel stack (F16 already verified ON)** | #27593 reports `GGML_SYCL_ENABLE_OPT=0` costing 68% of generation (workaround possibly obsolete after the reorder PRs), `SYCL_UR_USE_LEVEL_ZERO_V2=0` +1.4% tg, and `GGML_SYCL_FA_ONEDNN` up to +49% long-ctx prefill. Each is a single-variable cell; their ceilings on us are bounded by the segment they touch (dense/routed split above). | ENABLE_OPT=0 retest: unbounded until measured (historically huge). LEVEL_ZERO_V2=0: ~+1.4% tg class. FA_ONEDNN: bounded by attention share of 1.34 s segment → <=+5-10% pp at long ctx. **Verified locally: `GGML_SYCL_F16=ON` in all three build dirs — the biggest #27593 screen is already done on our side.** | Minutes per cell (env/build flag) | Each flag is its own 10-15 min single-variable cell against the standard gate | https://github.com/ggml-org/llama.cpp/issues/27593 |
| 9 | **Host thread pinning for the (pre-#23142) counting-sort + tokenize threads** | Verified locally: single NUMA node, 8 cores/16 threads, 96 MB shared L3 — there is no NUMA placement to win, but the host counting-sort and tokenize work can still collide with the 8 inference threads on SMT siblings. Pinning server threads to 8 physical cores and giving grouping/tokenize their own physical cores removes that collision. | The host mapping share is part of the 1.34 s non-GEMM segment → **<=0.5-1% e2e**; honest verdict: only worth doing bundled with another change, or as interim hygiene until #23142 is backfilled (which removes the sort from the host entirely). | 15 min (taskset/chrt in the launch wrapper) | taskset A/B on one p7000/g1 prefill cell, 10 min; kill if delta < noise band | https://github.com/ggml-org/llama.cpp/pull/23142 |
| 10 | **Strategic: spend on alpha (drafter quality), not gamma (depth) — DFlash2/FastMTP-class head work is the highest-leverage decode axis** | The arithmetic from idea 3 inverted: at gamma=4, +0.05 acceptance is worth ~+9.6% decode, while any realistic depth change is worth <=1.5%. Drafter quality work (the registered DFlash2 route; FastMTP-style head training upstream) attacks the only term with real slope. | +0.05 alpha ≈ **+9.6% decode** at current operating point; compare: k-sweep <=1.5%, clamp cost 2-4%, verify-head +6-10%. | Existing registered campaign (qwen38-dflash2-objective) | None new — this ranks existing registered work, it does not add one | https://arxiv.org/abs/2503.01840 (EAGLE-3), Red Hat "Optimize vLLM speculative decoding with FastMTP heads" (2025) |

---

## 2. Idea sections (detail beyond the table)

### 2.1 The #54785 hazard and the guard (rank 1) — read this before publishing anything

Facts from the issue thread (verified against our stack: same GPUs, same MTP, same capture):

- Boundary is EXACT at k=4: k=1/2/3 are bit-stable and match the eager reference; k=4 cycles among
  2-3 output variants with top-1 logprobs swinging up to ~1.5 nats, degenerating after ~40 requests.
- Capture is required: `VLLM_XPU_ENABLE_XPU_GRAPH=0` + k=4 is deterministic. bs>=2 (outside the
  capture list) is clean — the corruption lives in captured bs=1 pieces. Inductor is exonerated
  (`TORCH_COMPILE_DISABLE=1` still corrupts). Suspected root: GDN speculative state buffers / the
  verify mask for 4 draft positions not reset or re-bound between piece replays.
- Upstream has deployed a **clamp: k 4→3 whenever capture is active with a speculative config**
  (boot warning), escape hatch `VLLM_XPU_ALLOW_K4_CAPTURE=1`. No fix PR merged as of 2026-09-23.
- `VLLM_XPU_GRAPH_TRACING_MODE` is not part of this issue's surface; capture pool size and capture
  mode (FULL_DECODE_ONLY only was reported) made no reported difference. TP was fixed at 2 and not
  varied — **our TP1 champion is not proven safe by their data, only unimplicated.**

What we add: (a) the clamp cost arithmetic above (2-4%, i.e. cheap insurance); (b) the always-on
guard — at server boot and before each campaign cell, fire the same 6-token greedy request twice,
require identical output bytes AND top-1 logprobs within ~0.02 nats; ~15 s per bracket, catches the
entire "captured-spec-state corruption" class including variants we have not seen. This becomes a
harness requirement, same standing as the prefix-cache snapshots. (c) Publication rule: until the
clamp or a fix lands in our image, the champion recipe is PROVISIONAL and must carry the k=4-capture
caveat in any claims audit.

### 2.2 Verify-pass head INT4 (rank 2) — the biggest remaining decode lever, and where it stops

Mechanism (why it is tempting): in vLLM MTP verify, the target model runs the k+1 positions in ONE
forward and its LM head produces the logits used to accept/reject each draft token and to emit the
bonus token. The head (F16, 2.54 GB on our artifact) is re-read once per round — 12% of round bytes.
Quantizing it W4A16 g128 (~0.64 GB incl. scales) removes 1.9 GB/round → +10% decode ceiling.

Why it is NOT the same as the draft-side win we shipped: Leviathan's lossless guarantee covers the
DRAFT ("identical output distribution for any choice of approximation model Mq") because the target
verifies with its exact distribution. The verify head IS part of the target distribution. Quantizing
it changes which tokens get accepted in near-ties and which bonus tokens get emitted → the served
model is a different distribution, full stop. Greedy byte-identity will fail by construction at the
first flip; the question is only the flip rate. Our fp8 target route keeps the head in F16 precisely
because checkpoint conventions exclude it (`exclude_modules`) for accuracy reasons — we would be
reversing a deliberate exclusion, so this needs: the 20-prompt divergence probe, a PPL/task-quality
gate, and explicit non-lossless labeling per the claims standard. If the flip rate is negligible on
measured prompts it may still be worth shipping — but it must be labeled, never sold as lossless.

Secondary payoff: 1.9 GB VRAM freed on the serving card → KV headroom at p8192.

### 2.3 Speculation economics with a quantized drafter (rank 3 + rank 10) — the joint optimization

With per-round cost `c_v (verify) + gamma*c_d (draft)` and E[gamma] = (1-alpha^(gamma+1))/(1-alpha):

- Quantizing the draft moved c_d from ~5 ms to ~1.5 ms (byte arithmetic above). The optimum gamma
  moves only weakly: at alpha=0.62, gamma=4→5 is +0.7%, and at alpha=0.69, gamma=5 is +1.4%. The
  curve is flat within ±2% for gamma=4..6 across our measured alpha band. **Decision: the wave-1
  k-sweep (k=5/6) is near-worthless as a standalone lever; the #54785 clamp to k=3 costs 2-4%.**
- The steep term is acceptance: E is near-linear in alpha at our operating point (+9.6% decode per
  +0.05 alpha at alpha=0.62, gamma=4). Drafter quality work (registered DFlash2 campaign; EAGLE-3 /
  FastMTP-class head improvements upstream) is the highest-slope axis we have.
- Adaptive-gamma literature (SpecDec++ COLM 2024: trained acceptance head; DISCO Nov 2024: trained
  lookahead classifier; SVIP EMNLP 2025: draft-entropy stop, +17% long-form) exists, but given the
  flatness measured above, none of it justifies engineering on our numbers today. Revisit only if
  the acceptance log reanalysis (idea 3) shows a real skewed shape (e.g. long low-acceptance runs
  that a stop rule would truncate).

### 2.4 Tile-local reconstruction beyond the MoE op (point A, answered)

Where the #29245 technique applies elsewhere, audited:

- **vLLM INT4 target path**: already fused. The IPEX ESIMD WOQ GEMM (WoqGemm) that serves
  GPTQ W4A16 dequantizes on-the-fly in-kernel (Intel's design goal for the kernel; source in
  `intel-extension-for-pytorch csrc/gpu`, migrating to vllm-xpu-kernels per RFC #33214). No
  whole-tensor expansion, no 2x on the table — with one caveat: during the IPEX→vllm-xpu-kernels
  migration the GPTQ/AWQ INT4 GEMM is "still WIP" in the new library, so a runtime image bump could
  silently change which kernel serves our drafter. That is what idea 4's audit guards.
- **fp8 KV reads in attention**: KV is stored fp8 and dequantized inside the attention kernel as it
  is read (the v0.1.12 fix in `concat_and_cache_mla` confirms the store-side; the read side is
  inline in the FA kernel). No whole-cache FP16 materialization exists to remove. The remaining KV
  lever is capacity-class, not byte-class (TurboQuant, wave-1 item 5).
- **Our draft-head INT4 quantizer**: it emits the W4A16 NT layout that `int4_gemm_w4a16` consumes —
  same fused kernel class as the target path. Nothing to re-do; idea 4 is the cheap confirmation.
- **Dense IQ3 in llama.cpp (K216 attention/shared projections)**: unknown → rank 6's 15-min trace
  decides; bounded at ~+3.4% pp.
- **W4A8 / INT8-activation variants**: the rate-class law says weight-only sub-byte with FP16
  activations already ties the INT8 compute class, so #29245's VNNI-F16-activation design leaves no
  compute on the table; an INT8-activation path would only add a quantization-error axis without
  adding bandwidth savings beyond W4. Nothing to chase that is not already covered by the killed
  W4A8 premise.

### 2.5 oneMKL-in-graphs (point B, answered with arithmetic, not hope)

- The blocker's status: oneMKL/oneDNN GEMM calls fail inside SYCL graph recording ("graph nodes
  cannot depend on events from outside the graph"); #28725's author found the restriction lifted on
  oneAPI 2026, but **our installed toolchain is 2025.3** (verified today), so on our host the
  restriction stands today. Shape-dependent `stream::wait()` (ubatch 1024) remains a problem in any
  case, and #29245 as written still performs the D2H expert-ID copy + wait before the grouped call —
  so even with a custom MoE GEMM, the MUL_MAT_ID node stays uncapturable until #28476/#23142-style
  on-device dispatch is stacked on top.
- Capturable fraction after the full stack (#28476 + #23142 + #29245-class kernels): ~73.5% of
  prefill ubatch time (routed ops) plus most of the remainder — call it ~95% of nodes. **And it does
  not matter**: prefill capture saves launch tax only, ~900 launches x ~5 us ≈ 4.5 ms on a 5.06 s
  ubatch = 0.09%. Prefill graph capture is dead by arithmetic; do not spend anything on it.
- Decode is the only capture that can pay, and only via launch tax: 3.5-4.3% ceiling on us (rank 5).
  The external +19.8% MoE case had different (wait-dominated) anatomy; our wait tax was already
  measured at +3.54%. Gate rank 5 at >=5% and expect to kill it.

### 2.6 Host-level ideas (point E, sized on the real machine)

- **Page cache (rank 7)**: 60 GB RAM total, 39 GB cache, 3 GB swap used (verified). The 31 GB
  `-ncmoe` expert segment fits but with little slack; cold-start costs ~5 s once; eviction jitter is
  a tail risk. WILLNEED on expert byte ranges only; never `--mlock` the whole GGUF (contract do-not
  stands — locking 31 GB of 60 GB is a different, still-risky call that needs its own gate).
- **NUMA/pinning (rank 9)**: single NUMA node, shared 96 MB L3 — no placement win exists; only the
  SMT-collision hygiene, <=0.5-1% e2e, interim until #23142.
- **UR/Level-Zero backend**: `SYCL_UR_USE_LEVEL_ZERO_V2=0` (+1.4% tg, external) is inside rank 8's
  screen family. The Vulkan-with-XMX community report stays watch-only (swings 3x between drivers).
- **INT2 XMX**: exists on Xe2 silicon (Lunar Lake Xe2 review: XMX "all the way down to INT2"), but
  INT2 throughput = INT4 throughput — no additional compute class beyond what our rate-class law
  already records. Closed item stays closed, now with silicon confirmation.
- **Block-scaled DPAS (MXFP)**: no native Xe2 instruction exists; vllm-xpu-kernels' "native Xe2
  grouped-GEMM MXFP8/block-FP8" paths are software (dequant + scale-in-accumulator on BF16 DPAS,
  v0.1.13/v0.1.15) — which is itself useful design evidence for the grouped-IQ3 kernel (fold block
  scales into the accumulator instead of materializing), already noted by the 0923 wave.
- **2:4 sparsity as capacity**: no sparse-DPAS on Xe2 (re-confirmed); a pruned-but-dense-executed
  format is strictly dominated by a lower bit-width format. Dead, again.
- **Attention kernel selection**: FA_ONEDNN screen is in rank 8; nothing new beyond it.
- **Chunked prefill+decode interleave**: already vLLM V1 default behavior on our image; llama.cpp
  side has no equivalent lever worth a campaign at our workload mix.

---

## 3. What I could not find / negative results (explicitly checked)

1. **No fix PR for #54785** exists as of 2026-09-23; the clamp is the deployed mitigation. No
   analysis of whether capture mode (FULL vs PIECEWISE) or TP=1 changes the corruption — the issue
   only reports FULL_DECODE_ONLY at TP=2. Our TP1 stack is unimplicated, not exonerated.
2. **The Intel IPEX weight-only-quant ESIMD blog 404s** (moved); kernel truth taken from the source
   tree location (`intel-extension-for-pytorch csrc/gpu` WoqGemm) and the vllm-xpu-kernels RFC.
3. **DISCO's arXiv ID**: 2411.18462 is NOT DISCO — it is SVIP ("Draft Model Knows When to Stop",
   EMNLP 2025). DISCO ("Dynamic Speculation Lookahead Accelerates Speculative Decoding", NVIDIA,
   Nov 2024) is cited by title; fetch its abs page before quoting an ID anywhere.
4. **GGML_SYCL_F16 is already ON** in all three of our build dirs (verified in CMakeCache today):
   #27593's headline prefill screen does not apply to us as a change. ENABLE_OPT / LEVEL_ZERO_V2 /
   FA_ONEDNN remain genuinely unscreened.
5. **oneAPI 2026 is not installed here** (2025.3 verified): every upstream claim of the form "works
   after oneAPI 2026" (#28725's lifted GEMM-event restriction) is out of reach without a toolchain
   upgrade, which is its own gated project.
6. **INT2/MXFP/2:4** — all three re-confirmed dead as compute or execution levers on Xe2 (see 2.6);
   the rate-class law survives external silicon review.
7. **Verify-head losslessness**: no paper claims lossless target-side head quantization; every
   speculative-decoding correctness proof routes through an exact target distribution. The only
   in-family results (QSpec, QuantSpec) quantize the DRAFT side, including draft KV — which is the
   theoretically clean family we already exploited.
8. **No mmap-warmup tooling exists in llama.cpp** for a subset of byte ranges (only whole-file
   `--mmap`/`--mlock`); the rank-7 idea needs a small custom loader hook or an external pre-warm
   step; nothing upstream to lift.
9. **Negative on our own draft stack**: the measured round time already falsifies any head-fallback
   scenario at the ~4 ms/round level; idea 4 is confirmatory and guards the library migration, not
   a discovered bug.

### CLOSED-item check (anything reopened?)

- **REOPENED (narrowly): "SYCL graph record/replay on B70"** — new upstream evidence (+19.8% MoE
  decode via native recording + L0 patches) reopens MoE DECODE graphs only, and our own arithmetic
  (rank 5) re-caps the expectation at 3.5-4.3% for our anatomy; prefill capture stays closed by
  arithmetic (0.09%). Handle via rank 5's 5% gate.
- Stays closed: device-ID repair as a speed lever (now additionally upstream-confirmed as the graph
  prerequisite, which is its recorded value); sub-8-bit DPAS compute (INT2 silicon confirmation);
  ubatch>1024; TP2; 0.25 ms/draft-step (byte budget still ~6x over); SYCL graphs for dense B70.
- Nothing found that reopens W4A8 quality, AWQ, EXL3, Machete, LayerSkip, or P/D for DSV4.

---

## 4. Top 3 to start today (all <=30 min, zero or trivial GPU)

1. **Spec-canary + clamp decision (rank 1).** Fire the 8x identical greedy request at the current
   champion; if k=4 shows >=2 distinct outputs, adopt the k=3 clamp as champion config and wire the
   2x canary into the harness before/after every cell. This protects an already-won result and any
   publication built on it.
2. **Gamma reanalysis script (rank 3).** Parse the spec-decode counter snapshots already retained
   under `results/2026/09/b70-improvement-probes-20260920/p4-champion-ab-*`; plot measured
   speedup(gamma) for gamma=3..6. Expected outcome: kill the wave-1 k-sweep and quantify the clamp
   cost with our own data instead of the formula.
3. **Drafter op audit + verify-head divergence probe (ranks 4+2, one session).** One profiler
   capture confirms every draft linear is on the fused INT4 kernel; then a throwaway runtime patch
   pointing the existing `draft_int4` quantizer at the TARGET head, 20 fixed greedy prompts,
   first-divergence histogram. That one histogram decides whether the +6-10% verify-head lever
   enters a protocol campaign or dies.

## Source index (primary URLs)

- https://github.com/vllm-project/vllm/issues/54785 — capture+MTP4 non-determinism, clamp, repro
- https://github.com/ggml-org/llama.cpp/pull/29245 — grouped MoE XMX GEMM, tile-local SLM
- https://github.com/ggml-org/llama.cpp/pull/29107 — IQ3 reorder (+75.5% tg256@9k B70, draft)
- https://github.com/ggml-org/llama.cpp/pull/28725 — SYCL graphs, MUL_MAT_ID exclusion, +19.8% MoE
- https://github.com/ggml-org/llama.cpp/issues/27593 — F16/ENABLE_OPT/LEVEL_ZERO_V2/FA_ONEDNN
- https://github.com/ggml-org/llama.cpp/pull/23142 — device-side expert mapping
- https://openreview.net/forum?id=1fZ0tGfsqb — Leviathan et al., lossless-draft guarantee, Thm 3.8
- https://arxiv.org/abs/2302.01318 / https://arxiv.org/abs/2405.19715 (SpecDec++) /
  https://arxiv.org/abs/2411.18462 (SVIP) — adaptive-gamma family
- https://arxiv.org/abs/2410.11305 (QSpec), https://arxiv.org/abs/2502.10424 (QuantSpec) — draft-side quant families
- https://arxiv.org/abs/2503.01840 (EAGLE-3); Red Hat Developers: "Optimize vLLM speculative decoding with FastMTP heads" (2025)
- https://github.com/vllm-project/vllm/issues/33214 — IPEX→vllm-xpu-kernels migration, GPTQ INT4 WIP
- https://docs.vllm.ai — ModelOpt fp8 `exclude_modules` (lm_head exclusion convention)
- https://chipsandcheese.com/p/intels-battlemage-architecture and Lunar Lake Xe2 review (Oct 2024) — XMX INT2/4/8/FP16/TF32, no block-scaled DPAS
- https://manpages.debian.org/unstable/llama.cpp (mmap/mlock semantics); https://sheldylew.com (cold-page-fault TTFT case study)

> **MEASURED CORRECTION 2026-09-23:** the #54785 k=3 clamp recommendation in this file is **REVERSED** — the clamp costs +5.45 % throughput (k4 75.04 vs k3 71.16 tok/s) and the corruption class is not live (spec-canary PASS at k=3 and k=4). Keep MTP4. Details: `worklogs/SPEC-ECONOMICS-20260923.md` §CORRECTION and `clampk3-20260923T022938Z/`.
