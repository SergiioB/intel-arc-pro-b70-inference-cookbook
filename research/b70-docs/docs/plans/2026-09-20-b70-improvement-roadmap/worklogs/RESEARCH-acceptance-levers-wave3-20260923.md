---
title: "Acceptance-rate (alpha) levers for the Qwen3.8-27B MTP drafter on XPU — wave 3 online research"
type: plan
date: 2026-09-23
status: research
---

# RESEARCH — Acceptance levers wave 3 (2026-09-23, overnight online research)

**Scope:** read-only online research. No GPU work, no model loads. Every external claim carries a URL.
**Context assumed (from 2026-09-22 zero-GPU economics, do not re-derive):** champion = Qwen3.8-27B GPTQ-INT4 G128 + BF16 MTP4, vLLM XPU 0.27.2rc1.dev77 + `vllm-xpu-kernels` 0.1.12.3, XPU graph capture ON, draft-side INT4 pair (LM head + MTP body → W4A16 g128) = +40–49 % decode, published 2026-09-22. Measured c_d ≈ 1.5 ms/draft token, c_v ≈ 28.8 ms/verify round. Depth is **flat** (k=4 optimal ±1.4 %; k=5/6 strictly worse); **α is the steep lever: +8.3…+8.8 % throughput per +0.05 α** at measured α ∈ [0.487, 0.568]; α is run-variable 0.48–0.70. Rejected tonight: #54785 k=3 clamp, vllm-xpu-kernels 0.1.15.4 fused sampler, llama.cpp IQ3 reorder.

**The framing that survived the night:** our measured aggregate α ≈ 0.49–0.57 at k=4 is exactly the signature the literature calls *vanilla-MTP step decay* — published MTP heads collapse after the first draft token (70 % → 11 % → 2 % per-step on a 7B model, FastMTP §3.4). Almost everything that raises α does so by attacking that decay. That single fact orders the whole roadmap below.

---

## 1. Ranked table of actionable α-movers

| # | Lever | Expected Δα (aggregate, k=4) | Effort | Source | Concrete implementation on this stack |
|---|---|---|---|---|---|
| 1 | **FastMTP-style recursive fine-tune of the shipped Qwen3.8 MTP head** (self-distilled data, position-shared weights, multi-step alignment) | **+0.10…+0.20** (per-step α at k≥2 is where the 5–30× headroom lives) | High (one training run; head is <3 % of backbone params; tooling exists) | FastMTP arXiv:2509.18362 (https://arxiv.org/abs/2509.18362); speculators 0.6.0 (https://developers.redhat.com — "Optimize vLLM speculative decoding with FastMTP heads", 2026-09-08); repo https://github.com/Tencent-BAC/FastMTP | Fine-tune ONLY the MTP module weights (keep target untouched → our draft-side INT4 re-quant flows unchanged). Swap the MTP safetensors in the draft checkpoint; shapes/arch identical → drop-in for `method: mtp`. Then re-run the existing draft-INT4 quant on the new head. |
| 2 | **Probabilistic draft sampling** for our non-greedy published sampling (temp 0.7) instead of greedy argmax drafting | +0.02…+0.05 (theory + vendor docs say greedy drafting undershoots α at temp > 0) | Low (flag / small patch) | vLLM spec-decode docs: "Probabilistic acceptance (temperature > 0 draft sampling) is not yet supported and will be added in a future release" (https://docs.vllm.ai/en/latest/features/speculative_decoding/); temperature-vs-α https://www.dailydoseofds.com (spec-decoding explainer); losslessness https://jaykmody.com/blog/speculative-sampling/ | Check whether 0.27.2rc1.dev77 exposes `draft_sample_method='probabilistic'` for mtp; if yes, relaunch and A/B α on 5 fixed prompts. If not present on XPU build, it is a small upstreamable patch in the draft worker's sampler call. |
| 3 | **Penalty/top-k interaction audit** (published Qwen sampling uses presence_penalty 1.5) | 0…+0.05, plus a **correctness exposure** | Very low (request-only) | Feature-tracking RFC #27691 (https://github.com/vllm-project/vllm/issues/27691); penalties-implemented-as-logits-processors incompatibility (https://discuss.vllm.ai, standalone-draft thread; behavior must be verified per release) | Request-only: identical prompt set with presence_penalty 0 vs 1.5; compare server spec-decode acceptance counters and grep logs for any "spec decode disabled/fallback" warning. If penalties silently bypass spec decode, part of our run-variable α spread is explained — and published claims must state which path ran. |
| 4 | **Confidence/entropy-gated dynamic depth** (SVIP / DSpark adaptive verification / vLLM DSL RFC) — does not raise α, raises accepted-tokens-per-draft-token | +1…+3 % throughput (α unchanged; our depth is already near-optimal so upside is the low-α tail) | Medium (wait for upstream; or gate client-side request mix) | SVIP arXiv:2411.18462 (https://arxiv.org/abs/2411.18462); vLLM DSL RFC #36657 (https://github.com/vllm-project/vllm/issues/36657, 2026-03-10); "Adaptive Verification … currently DSpark only" (https://docs.vllm.ai/en/latest/features/speculative_decoding/) | Not implementable for mtp today (adaptive verification is DSpark-only upstream). Cheap precursor probe: log per-request acceptance counters split by prompt domain (code vs prose vs chat) to see if a bimodal α distribution exists that gating would exploit. |
| 5 | **fp8 KV for the MTP layer's history view** (the planned measurement) — literature predicts **α-neutral**, throughput-positive via c_d/KV-bandwidth | Δα ≈ −0.01…0; throughput + via smaller c_d and KV traffic | Low (relaunch + measure) | QuantSpec >90 % acceptance with 4-bit draft KV (https://arxiv.org/abs/2502.10424); "draft head acceptance is bit-insensitive, K4 a sweet spot" (MTP-on-8GB-GPU note, github.com, 2026-07); counterexample: "q4 draft cache ran 29.6 tok/s; plain f16 ran 39.7 — 34 % faster" (https://ai-muninn.com, 2026-06-25); vLLM FP8-KV state-of-the-union (https://vllm-project.github.io, 2026-04-22) | Proceed as planned: relaunch with MTP-layer KV fp8, measure α AND c_d separately (α must be flat; the win, if any, is c_d/attention-bandwidth). Reject if α drops > 0.02 (would contradict the "bit-insensitive" prior and signal a kernel-quality issue, not a physics issue). |
| 6 | **Domain-matched draft training data** (fold into #1): fine-tune the head on our real serving distribution | +0.02…+0.08 on top of #1 | Low if #1 rig exists | Baseten EAGLE recipe — heads trained on target-domain data accept more (https://www.baseten.co); BentoML — "task domain mismatch" explains underperforming drafts (https://www.bentoml.com) | Use the same self-distill corpus recipe as FastMTP but seeded with our captured request distribution; retrain head only. |
| 7 | **EAGLE-3 head for Qwen3.8** (training-time test, multi-layer fusion) | Potentially ≥ #1 (EAGLE-3 τ = 4.05–7.5) but **no community head exists for Qwen3.8-27B** and EAGLE3-on-XPU is unproven on our pinned build | Very high | EAGLE-3 arXiv:2503.01840-class ("training-time test"); Intel container release notes claim EAGLE/EAGLE3 on Intel GPUs (v0.10.2-xpu); upstream XPU spec-decode tracker **closed stale with no matrix** (https://github.com/vllm-project/vllm/issues/26963); HF search found only a DFlash-vs-MTP discussion for Qwen3.8 (https://huggingface.co, 2026-08-14) | Defer behind #1. Gate: (a) trained EAGLE3 head appears for Qwen3.8, (b) EAGLE3 path validated on XPU backend in our build line. Note `parallel_drafting` is "Only compatible with EAGLE and draft-model methods" — not mtp (vLLM docs). |
| 8 | **Draft-side vocabulary compression** (FR-Spec/high-freq token map, FastMTP's "language-aware dynamic vocabulary compression") | Δα ≈ −0.005…0 (τ cost 0.026–0.068 tokens); cuts c_d → throughput + | Medium | FastMTP: compression is drafting-only, "the verification phase retains the full vocabulary space, guaranteeing lossless generation quality" (https://arxiv.org/abs/2509.18362); FR-Spec maps thunlp/Qwen2-7B-Instruct-FR-Spec (https://github.com/Tencent-BAC/FastMTP) | After #1: apply a 16–32 k token map to the draft head only; verify acceptance-length delta ≤ 0.07; combine with draft-INT4 (bandwidth lever stacks with compute lever). |
| 9 | **Suffix/n-gram auxiliary speculation for repetitive request classes** (chain, dynamic depth, model-free) | Workload-dependent (agentic/code-edit heavy traffic only) | Low | vLLM docs list `suffix` with "dynamic speculation depth" and "Maximum combined prefix-match and speculation tree depth" (https://docs.vllm.ai/en/latest/features/speculative_decoding/); suffix decoding for agentic workloads (https://jarvislabs.ai) | Only if logs show a repetitive-request majority. Cannot stack with mtp in one engine config on this build — would need per-request routing. |
| 10 | **Community retrained-head watch** (Jackrong GGUF is the *stock* head rebundled, not retrained — "trained with multiple steps" refers to pretraining) | 0 (context only) | None | https://huggingface.co/Jackrong/Qwen3.8-27B-MTP-GGUF ("MTP ~2x Faster, 77.9 % Acceptance", "Recommended setting: max draft = 2"); overbring.com measured 63.93 % draft acceptance on UD-Q4_K_XL | Set a saved-search/watch for the first genuinely FastMTP-fine-tuned Qwen3.8 head on HF; if it appears, skip our own training run and quant+validate it directly. |

**Cross-cutting community calibration points for Qwen3.8-27B MTP:** 77.9 % acceptance at max-draft 2 (Jackrong GGUF page, https://huggingface.co/Jackrong/Qwen3.8-27B-MTP-GGUF); 63.93 % draft acceptance with ~724 accepted tokens/request on a 4-bit quant (https://overbring.com, 2026-08-17); "drafting more tokens doesn't increase accepted tokens per verification pass — extra drafts are pure cost" (Towards AI, https://pub.towardsai.net, 2026-09-07); vLLM Ascend's Qwen3.8 page: "benefits depend on the acceptance rate; recommends tuning speculative token count per workload" (https://docs.vllm.ai). All are consistent with our α ∈ [0.48, 0.70] run-variable band and flat-depth economics; none contradicts them.

---

## 2. Per-point findings with citations

### 2.1 MTP training/distillation (mission item 1)

**FastMTP (arXiv:2509.18362, 2025-09; https://arxiv.org/abs/2509.18362, https://arxiv.org/html/2509.18362)** — the single most important external result for us:
- Method: "Our approach fine-tunes a single MTP head with position-shared weights on self-distilled data", aligning "MTP training with its inference pattern" (multi-step recursive drafting).
- Headline: "FastMTP achieves an average of 2.03× speedup compared to standard next token prediction with lossless output quality, outperforming vanilla MTP by 82 %."
- Acceptance collapse it fixes: "Vanilla MTP exhibits severe performance degradation beyond the first draft step"; fine-tuned "achieves substantially higher acceptance rates across all draft steps: 80 % at k=1, 56 % at k=2, and 36 % at k=3 on average"; "dramatically improving average acceptance rates from 70 % to 81 % for the first draft token, from 11 % to 56 % for the second" (and 2 % → 36 % at k=3). Math tasks: "maintaining over 50 % acceptance rate even at k=3 compared to vanilla MTP's mere 3 %."
- Cost: only the 210.8 M-param MTP head of MiMo-7B-RL was trained (<3 % of backbone); "requires only lightweight training and seamlessly integrates with existing inference frameworks."
- Losslessness preserved: vocabulary compression is drafting-only; "the verification phase retains the full vocabulary space, guaranteeing lossless generation quality" (τ cost 0.068 tokens MT-Bench / 0.026 C-Eval).

**Production tooling exists (Red Hat, 2026-09-08; https://developers.redhat.com):** "Speculators 0.6.0 implements FastMTP-style fine-tuning, a training recipe that adapts a single shipped MTP head for the recursive, multi-step" drafting; the article's notable improvement: "recursive fine-tuning on the native MTP head of Qwen3-Next-80B-A3B-Instruct improves acceptance rate" over the shipped head. Speculators docs: "MTP fine-tuning starts from the shipped head" (https://docs.vllm.ai). Upstream proposal: vLLM speculators RFC #267 (2026-01, github.com).

**Drop-in swap question (does vLLM accept a better-trained replacement head?):** yes in principle — vLLM MTP loads the MTP module from the checkpoint (`method: mtp`, "the target model includes native multi-token prediction capability… you do not need to provide a separate draft model", https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp.html). A fine-tuned head with identical shapes is a weight swap; nothing in the config path distinguishes stock vs fine-tuned MTP weights. FastMTP's own serving path demonstrates head-swap serving (SGLang flags in https://github.com/Tencent-BAC/FastMTP; vLLM equivalent via speculators).

**Other distillation variants:** DistillSpec aligns draft with target via distillation (ICLR 2024, referenced in the SpecTr line, https://arxiv.org) — applies to separate draft models, subsumed by FastMTP for the MTP case. Depth-2/dual-MTP-layer: vLLM MTP supports Qwen3-Next/Qwen3.5-class native heads, but our depth economics are flat, so a second MTP layer buys nothing (Towards AI 2026-09-07 reaches the same conclusion for Qwen3.8: https://pub.towardsai.net). EAGLE-3's "training-time test" (https://arxiv.org, 2025-03-03) is the same idea generalised to a feature-fusion head — needs target hidden states each round and an EAGLE3 path on XPU (see §2.5/§4).

### 2.2 Inference-side acceptance movers (mission item 2)

**Greedy vs sampled drafting.** vLLM docs (https://docs.vllm.ai/en/latest/features/speculative_decoding/): "use_heterogeneous_vocab currently supports greedy draft sampling only"; "Probabilistic acceptance (temperature > 0 draft sampling) is not yet supported and will be added in a future release" — i.e. `draft_sample_method='probabilistic'` exists as a concept in current vLLM but is restricted/incomplete. Theory: rejection sampling is lossless regardless of draft policy (accept min(1, p/q) + residual resample; https://jaykmody.com/blog/speculative-sampling/), and "at temperature ~0 … acceptance rates are highest; as temperature increases, acceptance rates drop" (https://www.dailydoseofds.com). Since our published representative sampling is non-greedy (Qwen model-card: temp 0.7/top_p 0.8/top_k 20/presence 1.5 — serving.md Run 48 policy), a greedy-argmax drafter maximally mismatches the target distribution exactly on the high-entropy positions where α dies; sampled drafting is the lossless fix. Baseten's EAGLE recipe says the mirror image for training/eval: "EAGLE heads perform best under greedy decoding (temperature=0); non-greedy sampling, repetition penalties, and top-p/top-k reduce acceptance rates" (https://www.baseten.co).

**Entropy/confidence gating (SVIP; https://arxiv.org/abs/2411.18462, EMNLP 2025).** "A high entropy indicates a low acceptance rate of draft tokens, and vice versa"; drafting stops when the draft's own entropy says acceptance is unlikely. Training-free, length-policy-only → standard lossless guarantee untouched. "Up to 17 % speedup on MT-Bench at 8K context … 22 % for QwQ on long-form reasoning." Same family: SmartSpec goodput scheduling (https://arxiv.org/abs/2406.14066), DSDE KLD-stability (https://arxiv.org, 2025-09-08), DISCO classifier early-stop (github.com/HF, 2024). **vLLM status:** RFC #36657 "Dynamic Speculation Length (DSL) with Confidence" (2026-03-10, https://github.com/vllm-project/vllm/issues/36657); "Adaptive Verification: Sizes verification per request from drafter confidence; currently DSpark only" (vLLM docs); DSpark blog: "DSpark's confidence head scores each drafted token's chance of surviving verification, so instead of picking a speculation length per step, the depth is tuned dynamically" (https://vllm.ai, 2026-08-14). None of this is exposed for `mtp` on our build — it is an upstream-watch item, and our flat-depth result caps its upside (§3).

**Tree shapes.** `num_speculative_tokens` on mtp is a linear chain; trees belong to EAGLE/Medusa-style drafters (NVIDIA docs: chain-like drafting "unlike EAGLE which uses tree-based attention mechanisms", https://docs.nvidia.com). vLLM today: `parallel_drafting` is "Only compatible with EAGLE and draft-model methods" (vLLM docs) — not mtp; suffix decoding offers a speculation tree with dynamic depth (vLLM docs). A May 2026 feature request for draft-tree pruning for EAGLE-3 (`--speculative-token-tree`) confirms trees exist upstream but are EAGLE-scoped (github.com). Multi-draft theory: "sampling without replacement outperforms sampling with replacement" (Towards Optimal Multi-draft Speculative Decoding, https://openreview.net). For us: no Qwen3.8 EAGLE head exists (HF search 2026-09-23), so trees are blocked on #7.

**Context/state for the draft.** Non-issue for mtp: the drafter attends to the target's own KV — state sharing is inherent. Prefix-cache sharing matters only for separate-draft setups. Nothing to do here; record as N/A.

### 2.3 KV precision for the draft's view of history (mission item 3)

- **QuantSpec (Apple, arXiv:2502.10424; https://arxiv.org/abs/2502.10424):** self-speculative decoding where the draft pass runs on "a hierarchical 4-bit quantized KV cache and 4-bit quantized weights", yet "QuantSpec maintains high acceptance rates (>90 %)" with "consistent end-to-end speedups upto ~2.5×" and ~1.3× memory reduction. Direct evidence that *aggressive* draft-side KV compression need not cost α (long-context, edge regime).
- **MTP-on-8GB-GPU note (github.com, 2026-07):** "draft head acceptance is bit-insensitive, and K4 (KIVI-style per-channel/per-token quantization) is a sweet spot — smaller than Q8 with the same acceptance rate." The closest thing to an MTP-specific KV finding in the wild.
- **Counterexample on overhead (https://ai-muninn.com, 2026-06-25):** "q4 draft cache ran 29.6 tok/s; plain f16 ran 39.7 — 34 % faster, and f16 had more free VRAM too. Quantizing the cache needs a dequant [step]" — on a llama.cpp-class runtime the dequant cost ate the bandwidth win. Different kernel regime than vLLM's fused fp8 attention, but it defines the failure mode our fp8-vs-bf16 probe must watch: α flat yet throughput *worse*.
- **Same article on weights:** "quantizing it [the MTP head] directly degrades acceptance" — consistent with our draft-side W4A16 *g128* choice being the floor; do not go below.
- **vLLM FP8 KV state-of-the-union (https://vllm-project.github.io, 2026-04-22):** per-token KV cost down to "54 % of its BF16 counterpart in the best cases", with head-dim-dependent caveats — the memory-side context for our measurement.
- **Compressed-KV speculation theory ("Strong Drafts Need Compact Memories", arXiv 2026-08-31):** long-context drafts "work well even when the draft model uses a compressed KV cache" — speedup becomes a joint function of accepted tokens and draft-step cost under compression (surfaced via https://www.catalyzex.com / https://www.aigregate.com).

**Verdict for tonight's planned fp8-vs-bf16 draft-KV measurement:** literature prior is α ≈ flat (bit-insensitive) and the decision variable is c_d + KV bandwidth, not α. If α moves materially, suspect our kernel path, not the theory.

### 2.4 Verify-pass structure and early exit (mission item 4) — see §3 for the arithmetic

- **Does vLLM support early-exit verification?** No general token-level early exit in the engine today. What exists: (a) `rejection_sample_method: standard | synthetic | block` (vLLM docs — a sampler-shape option, not an early exit); (b) DSpark-only adaptive verification (confidence-sized verification, vLLM docs); (c) RFC #36657 dynamic speculation length (proposed, 2026-03); (d) a "[Feature]: Restore length-based speculation skip" request (2026-08-13, github.com) revealing vLLM once had and lost a length-keyed skip. True rejection-suffix skipping inside the verify pass is published only as research: FASER — "verification need not process all drafted tokens equally: tokens in the rejection suffix can be identified early and skipped" (https://arxiv.org, 2026-04-22).
- **SPRT-flavoured sequential verification (SPRINTER, arXiv:2502.04557; https://arxiv.org/abs/2502.04557):** "By performing sequential approximate verification, SPRINTER does not require verification by the target LLM and is only invoked when a token is deemed unacceptable"; the paper claims only that "approximate verification can still maintain high quality generation" — i.e. **lossy**. SpecEE (ACM) similarly predicts acceptance instead of verifying exactly.
- **Theoretical round-cost saving at α ≈ 0.55 with our c_d/c_v:** §3 below — the punchline is that on a memory-bound C1 decode the lossless saving is ≈ 0, and the lossy saving is disqualified by our publication property.

### 2.5 Novel 2026 angles (mission item 5)

- **Self-speculative on XPU:** LayerSkip/early-exit self-speculation (Elhoushi et al., ACL 2024, https://aclanthology.org) and DEL (dynamic exit layer, https://openreview.net) require early-exit-trained checkpoints and custom kernels; no Qwen3.8 artifact, no XPU path. Medusa-2 = Medusa heads + draft-target joint training — heads not available for Qwen3.8, and chain-Medusa α is below MTP-class at equal compute (NVIDIA chain-vs-tree framing, https://docs.nvidia.com). EAGLE-3: strongest published ceiling (τ 4.05–7.5, https://www.e2enetworks.com; AMD-vLLM blog confirms production EAGLE3 competitiveness, https://vllm.ai, 2026-08-23) but head + XPU-path both missing for us; Intel's ai-containers release notes claim n-gram/EAGLE/EAGLE3 on Intel GPUs for newer builds than our pin (search-surfaced, unverified on our stack).
- **Qwen3.8 MTP community fine-tunes:** none found as of 2026-09-23. Jackrong/Qwen3.8-27B-MTP-GGUF is the stock head rebundled into the GGUF ("bundles the MTP head directly into the model", https://huggingface.co/Jackrong/Qwen3.8-27B-MTP-GGUF) with llama.cpp guidance "--spec-type draft-mtp --spec-draft-n-max 2" — its 77.9 % acceptance is a measurement, not a better head. DFlash2 for Qwen3.8-27B exists as a community draft (up to 3.43× claimed, https://hackernoon.com, 2026-08-20) — already a gated route in our own campaign docs, not an α mover for the MTP line. HF discussion asking DFlash-vs-MTP for Qwen3.8 (https://huggingface.co, 2026-08-14) shows demand but no retrained artifact.
- **"Draft cache"/recurrent draft state:** nothing published that applies to a single-layer MTP head (state is already the target KV). DSpark's semi-autoregressive drafter with a confidence head (https://arxiv.org, 2026-07-06; NVIDIA training guide https://docs.nvidia.com) is the closest live idea; it is a different drafter, not a bolt-on.
- **Parallel drafting:** P-EAGLE removes EAGLE-3's sequential drafting bottleneck, "up to 1.69× over vanilla EAGLE-3" (https://aws.amazon.com, 2026-03); vLLM's `parallel_drafting` kwarg excludes mtp (vLLM docs). DFlash/DSpark are the parallel-draft family vLLM's AMD blog covers (https://vllm.ai, 2026-08-23).
- **Failure registry worth knowing:** GPTOSS-20B + EAGLE3 head mismatch measured "low acceptance rates (less than 10 %)" (https://github.com/vllm-project/vllm/issues/27691) — a caution that any third-party head must be acceptance-validated on OUR target before trusting.

---

## 3. Early-exit verification arithmetic at c_d = 1.5 ms, c_v = 28.8 ms, k = 4, α = 0.55

**Baseline (linear chain, lossless):** expected accepted tokens per verify round

```
τ(k) = (1 − α^(k+1)) / (1 − α)          # standard chain result
τ(4) = (1 − 0.55^5) / 0.45 = (1 − 0.0503) / 0.45 ≈ 2.110 tokens/round
```

Round wall time `T = k·c_d + c_v = 4×1.5 + 28.8 = 34.8 ms` → ≈ 60.6 tok/s engine-side (order-matches the champion).

**What early exit can and cannot save (lossless case, FASER-style rejection-suffix skip):** when the first rejection lands at accepted-position j (prob. α^j(1−α)), the remaining k−j draft positions are wasted verify work:

```
E[skipped verify positions] = Σ_{j=0..k} α^j (1−α) (k−j)
  = 0.45·4 + 0.2475·3 + 0.1361·2 + 0.0749·1 + 0 ≈ 2.89 of 5 verify rows (~58 %)
```

But our C1 decode is **memory-bandwidth-bound**: the verify forward reads the full ~27 B INT4 weight set once whether M = 5 rows or M = 1 row flows through the GEMMs. Skipping rows saves only the M-proportional slice (activation traffic + attention over 5 vs j+1 KV rows). Even crediting an optimistic 10 % of c_v as M-proportional:

```
best-case saving ≈ 0.58 × 0.10 × 28.8 ms ≈ 1.7 ms/round ≈ +4.8 % wall (realistic +2 %)
```

And the lossless floor is harder than that: the rejecting position's target logits are **required** to resample the corrected (bonus) token, so no exact scheme can skip the weight read of the pass that produced them. Dynamic per-round row counts also fight XPU graph capture (static shapes) — the same wall our SYCL-graph and #54785 work already hit locally.

**For scale, the competing lever:** +0.05 α at α = 0.55 → τ(4) = (1−0.60^5)/0.40 = 2.306 → Δ0.196 tok/round → **+5.6 % wall**, consistent with our measured +8.3…+8.8 % per +0.05 α (which additionally prices the c_v amortisation). One FastMTP-class retrain plausibly buys four to eight of those steps.

**Lossy case (SPRINTER/SpecEE):** replacing verification with a cheap verifier saves whole c_v passes in principle — at α = 0.55, a perfect acceptance-predicting verifier would skip target calls for the ~45 % of rounds whose lead token is doomed — but the output distribution changes. That breaks the exact-output property every published claim of ours rests on. Rejected.

---

## 4. Do-not-attempt list

1. **SPRINTER/SpecEE-style approximate verification** — explicitly lossy ("approximate verification can still maintain high quality generation", https://arxiv.org/abs/2502.04557). Violates the output-preservation property underpinning our LocalMaxxing publication.
2. **Token-level early-exit verification on this stack** — ≈ 0 lossless saving in memory-bound C1 decode (§3), dynamic shapes conflict with XPU graph capture (same failure class as our rejected SYCL-graph attempts). FASER remains upstream-watch only.
3. **Raising published α by serving greedy** — greedy maximizes α (https://www.baseten.co; https://www.dailydoseofds.com) but our sampling policy fixes representative numbers at model-card sampling; greedy cells stay diagnostic-only (existing contract rule; vLLM docs confirm "greedy sampling with speculative decoding matches greedy decoding without it" — so greedy cells are exact, just not representative).
4. **Tree/EAGLE/Medusa speculation now** — no Qwen3.8 EAGLE head exists (HF, 2026-09-23); `parallel_drafting` excludes mtp (vLLM docs); tree attention + dynamic shapes vs XPU graph capture unproven; upstream XPU spec-decode tracking issue closed stale without a matrix (https://github.com/vllm-project/vllm/issues/26963). Re-open only after gate conditions in table row 7.
5. **LayerSkip/DEL self-speculation** — needs early-exit-trained checkpoints and kernels we do not have on XPU (https://aclanthology.org; https://openreview.net).
6. **Quantizing below the proven draft pair** — draft-head quantization directly degrades acceptance (https://ai-muninn.com); our W4A16 g128 draft pair already banks the bandwidth win. Likewise no q4 draft KV on llama.cpp-class runtimes: −34 % end-to-end there (https://ai-muninn.com).
7. **Depth moves** — re-litigating k (clamp to 3, push to 5/6): depth is flat/worse at our α, and the external decay curve explains it (45.4 % → 35.6 % → 28.3 % per-step acceptance at 2/3/4 drafts, https://developers.redhat.com; "extra drafts are pure cost", https://pub.towardsai.net). #54785 clamp stays rejected.
8. **vllm-xpu-kernels 0.1.15.4 fused sampler** — already measured regression (−0.15…−14.5 %); nothing in this research rehabilitates it.
9. **Trusting any third-party head without acceptance validation** — head/target mismatch has produced <10 % acceptance in-tree (https://github.com/vllm-project/vllm/issues/27691).
10. **Multi-draft / sampled-without-replacement ensembles** — theory says they raise α per round (https://openreview.net) but require multi-draft verify support absent from the mtp path; research-stage only.

---

## 5. Cheapest discriminating probe per idea (≤15 min; request-only where possible)

| Idea | Probe (≤15 min) |
|---|---|
| 1. FastMTP retrain | Request-only pre-measure: dump per-step acceptance from server spec-decode metrics at k=1..4 on 5 fixed prompts → quantify the step-decay curve of the stock head. This number IS the business case for the retrain (FastMTP predicts step-2+ α near-floor for vanilla heads). |
| 2. Probabilistic draft sampling | One server relaunch with `draft_sample_method='probabilistic'` (if the build accepts it; else check `vllm --help`/config schema first, 2 min) → same 5 prompts, same temp-0.7 sampling, compare α counters. |
| 3. Penalty audit | Request-only: same prompt, presence_penalty 0 vs 1.5 → α counters + log grep for spec-decode fallback warnings. Also settles whether published-run semantics were penalty-active. |
| 4. Confidence-gated depth | Request-only: per-request acceptance counters split by prompt domain (code/prose/chat) → is α bimodal? If unimodal, upstream gating would not pay even when it ships. |
| 5. fp8 draft KV | One relaunch, fp8 KV on the MTP layer: read α AND draft-latency counters. Decision rule: α within ±0.02 AND c_d or attention time down → adopt; α down > 0.02 → reject regardless of speed. |
| 6. Domain-matched data | Free rider on probe 1: bucket the 5 prompts by domain and note which decays fastest — that bucket seeds the fine-tune corpus. |
| 7. EAGLE-3 | No probe (blocked on artifact + XPU path). Quarterly HF/upstream watch only. |
| 8. Vocab compression | After #1 lands: apply FR-Spec-style map to draft head, 5-prompt acceptance-length delta ≤ 0.07 → keep. |
| 9. Suffix/n-gram | Log audit: fraction of requests with high prompt-echo/repetition. Below ~30 % → drop the idea. |
| 10. Community-head watch | 5 min: saved HF/upstream search for "Qwen3.8 MTP fine-tune / speculators". |

---

## Sources (primary URLs used)

- FastMTP paper: https://arxiv.org/abs/2509.18362 (HTML: https://arxiv.org/html/2509.18362); code: https://github.com/Tencent-BAC/FastMTP
- Red Hat, "Optimize vLLM speculative decoding with FastMTP heads" (2026-09-08): https://developers.redhat.com ; speculators docs: https://docs.vllm.ai
- SVIP: https://arxiv.org/abs/2411.18462
- SmartSpec: https://arxiv.org/abs/2406.14066 ; vLLM automation RFC: https://github.com/vllm-project/vllm/issues/4565
- vLLM DSL with confidence RFC #36657: https://github.com/vllm-project/vllm/issues/36657 ; DSpark adaptive verification blog: https://vllm.ai (2026-08-14); DSpark paper: https://arxiv.org (2026-07-06); RFC: https://github.com (2026-07-09)
- vLLM speculative decoding docs (incl. `draft_sample_method`, `parallel_drafting`, adaptive verification, suffix): https://docs.vllm.ai/en/latest/features/speculative_decoding/ and https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp.html
- QuantSpec: https://arxiv.org/abs/2502.10424 ; code: https://github.com/SqueezeAILab/QuantSpec
- SPRINTER: https://arxiv.org/abs/2502.04557 ; FASER: https://arxiv.org (2026-04-22) ; EESD: https://openreview.net
- vLLM spec-decode feature tracking #27691: https://github.com/vllm-project/vllm/issues/27691 ; XPU tracker #26963: https://github.com/vllm-project/vllm/issues/26963
- Community Qwen3.8-27B: https://huggingface.co/Jackrong/Qwen3.8-27B-MTP-GGUF ; https://overbring.com (2026-08-17) ; https://pub.towardsai.net (2026-09-07) ; https://hackernoon.com (2026-08-20) ; HF DFlash-vs-MTP discussion (2026-08-14): https://huggingface.co ; vLLM Ascend Qwen3.8 page: https://docs.vllm.ai
- ai-muninn, "A Counterintuitive Look at Qwen MTP" (2026-06-25): https://ai-muninn.com
- vLLM FP8 KV state-of-the-union (2026-04-22): https://vllm-project.github.io
- EAGLE-3: https://arxiv.org (2025-03-03) ; Baseten training recipe: https://www.baseten.co ; BentoML domain-mismatch: https://www.bentoml.com ; E2E τ numbers: https://www.e2enetworks.com ; AMD-vLLM blog: https://vllm.ai (2026-08-23) ; P-EAGLE: https://aws.amazon.com (2026-03)
- LayerSkip: https://aclanthology.org ; DEL: https://openreview.net ; multi-draft theory: https://openreview.net
- Losslessness refresher: https://jaykmody.com/blog/speculative-sampling/ ; temperature-vs-acceptance: https://www.dailydoseofds.com ; NVIDIA chain-vs-tree: https://docs.nvidia.com
- Red Hat acceptance-decay measurements (45.4/35.6/28.3 %): https://developers.redhat.com ("Performance improvements with speculative decoding")
- MTP-on-8GB-GPU note (K4 sweet spot): https://github.com (2026-07)
- Jarvis Labs spec-decode guide: https://jarvislabs.ai

*PROVISIONAL — research notes only; no benchmark claims. All external numbers are as-published by their sources, not reproduced here.*
