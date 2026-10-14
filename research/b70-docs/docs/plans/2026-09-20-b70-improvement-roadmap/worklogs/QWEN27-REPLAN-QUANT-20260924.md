---
title: "QWEN27 replan — Rotated-GPTQ/QuaRot premise audit (quant branch)"
type: worklog
date: 2026-09-24
status: final
opened: 2026-09-24
ledger: RL-TBD (P8 rotated-GPTQ branch evidence)
depends_on: ["P8-bounded-quant-screen.md", "DEFERRED-AND-TRACKED.md §1 (EXL3)", "OUTCOMES.md L10", "worklogs/EXL3-PRE-REPLAN-20260924.txt"]
---

# QWEN27 replan — Rotated-GPTQ / QuaRot premise audit (quant branch, 2026-09-24)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Research-only session: no GPU, no conversion,
no download, no services, no shared-doc edits. Owner: b70-quantization-engineer.
Object under test: the "Frontier 3 / P0 Rotated-GPTQ" proposal recorded in
`worklogs/EXL3-PRE-REPLAN-20260924.txt` (premises: offline Hadamard eliminates outliers;
uniform INT3/INT4 unpacks with 1-cycle shift+mask vs EXL3; +20–30% decode; 39.8 ms -> 25 ms
=> ~140–141 tok/s). Every premise was checked against primary sources (papers, repos at
pinned SHAs, local configs/trees). This worklog is the quant-branch input to the replan.

## 0. Ranked verdict (summary)

| # | Element of the proposal | Verdict | Deciding evidence (primary) |
|---|---|---|---|
| 1 | Uniform INT3 GPTQ on the champion vLLM-XPU stack | **BLOCKED — no loader, no packing, no kernel** | vLLM `auto_gptq.py` TYPE_MAP = {(4,T):uint4b8,(8,T):uint8b128}; `mixed_precision/xpu.py` supported types = (uint4, uint4b8); `vllm-xpu-kernels` exposes only `int4_gemm_w4a16`/`int4_gemm_w4a8` (oneDNN `f16_int4`/`bf16_int4`, no `num_bits` parameter). Section 3. |
| 2 | Speed: +20–30%, 39.8 -> 25 ms => 140+ tok/s | **REFUTED within the stipulated model** (E0 toy: fixed traffic composition, fixed 608/540 GB/s, no other improvements; model-not-measured): body-only 4->3 saves at most 3.8 GB of a 21.44 GB step-byte model => bound 1.215x; all-bytes-4->3 bound 1.333x; required multiple is 1.37x (from 102.61 Run 48) to 1.76x (from 80.3 patched p8192). 25 ms/step is below even the all-bytes floor (16.08 GB / 608 GB/s = 26.4 ms). The traffic model itself is uncertain (e.g. embedding residency), so this is model-scoped, not a universal impossibility. Section 5. |
| 3 | "Offline Hadamard eliminates outliers" | **PARTIAL — overclaimed**. Weight-side incoherence is real and measured (QuaRot Table 7, dense Llama-2 only); "eliminates activation outliers" is the W4A4 story and needs ONLINE transforms (QuaRot runs ~1.5 online Hadamards/layer + online post-RoPE head rotation). For weight-only, online ops are not required for the residual-space rotation, but activation-side smoothing does not exist offline (activations are runtime data). Section 4. |
| 4 | "1-cycle shift+mask" for uniform INT3 | **FALSE mechanically**: 3 does not divide 8; one hypothetical layout is 10 values per int32 (`pack_factor = 32 // 3 = 10`, floor quotient; 2 unused bits => effective ~3.2 data bits/value), but cross-word layouts also exist (e.g. 8 values per 3 bytes = 24 bits exactly, needing multi-byte load+shift). No layout is a 1-cycle nibble-style unpack; and no XPU kernel exists regardless (see #1). |
| 5 | The 13.1 GB / 24.2 ms / 141 tok/s arithmetic | **Internally inconsistent**: 13.1 GB total step bytes is only reachable by also quantizing the F16 verify head (2.54 GB) and the draft to ~3 bpw — head INT4 and FP8 both FAILED their offline gates (D8/L10: top-1 72.07%; FP8 screens 93.3–93.4% vs >=99% line). With the head retained (gate-locked), the INT3-body step-byte model is ~17.6–18.5 GB, not 13.1. |
| 6 | Rotated-GPTQ as a QUALITY lever at fixed W4 | **SURVIVES, bounded** — this is exactly parked P8 branch framing ("quality-at-iso-bytes, no speed gain assumed"). Feasibility is architecture-specific and provable offline one layer at a time vs the BF16 source. Sections 4, 6, 7. |
| 7 | LM-head INT4/FP8 reopening | **DO NOT REOPEN unchanged**; and note the >=99% top-1 line is an offline *promotion screen*, not a task-quality pass — 99% top-1 would still not establish task quality. Section 8. |

Bottom line: the P0-as-proposed (INT3 speed lever to 140 tok/s) is rejected on four
independent gates (kernel/loader, arithmetic bound, head-quant gates, acceptance risk).
The rotation idea remains open only in the parked P8 framing: weight-only, W4-at-iso-bytes
(or G64) quality/capacity, one-layer equivalence proven against BF16 first, no speed claim,
no W4A4 drift.

## 1. Identity of the model and artifact (verified, static only)

Config verified from the local BF16 source `/mnt/<sata-disk>/Qwen3.8-27B-BF16/config.json`
(no tensor reads) and cross-checked against the public card (both agree):

- `Qwen3_5ForConditionalGeneration`, `model_type qwen3_5`; text config:
  hidden_size **5120** (= 5 x 2^10, NOT a power of two), intermediate **17408** (= 17 x 1024),
  **64 layers** = **48 linear_attention (GatedDeltaNet)** + **16 full_attention** (interval 4);
  full-attn 24 Q heads / 4 KV heads, **head_dim 256**, RoPE dim 64 (card), `attn_output_gate: true`;
  GDN: 16 qk-heads x 128, 48 v-heads x 128 (v-space 6144), causal conv kernel 4;
  vocab **248320**, `tie_word_embeddings: false`; max_position 262144.
- Public primary source: [Qwen/Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
  (created 2026-08-05, Apache-2.0, BF16-only main repo, FP8 serving checkpoint referenced). The model card EXISTS
  and matches; nothing here is substituted from Qwen3.5.
- Champion artifact (existing reports, not re-read): `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`
  — contract records **400 qweight tensors, 15 BF16 MTP tensors, zero MTP leaks, 19.56 GB**
  (`models/qwen/qwen3.8-27b/artifacts.md`). Draft-side INT4 pair overlays exist separately
  (`B70_DRAFT_LMHEAD_INT4` / `B70_DRAFT_MTP_INT4`, local patches).
- Step budget provenance (LABEL: **model + cross-window inference, NOT direct measurement**):
  the 21.44 GB/step decomposition (target body INT4 15.2 GB + F16 verify head 2.543 GB +
  draft INT4 3.4 GB + FP8 KV 0.3 GB) and the 39.8–46.8 ms step time (inferred from 80.3 tok/s
  and inferred 3.2–3.76 tokens/step) come from `worklogs/ARCHITECT-next-levers-20260921c.md` §0,
  which itself states 75–89% of the 608 GB/s peak-BW floor. Composition caveat: the 15.2 GB
  "body INT4" figure appears to include the F16 embedding (~2.5 GB) that is resident but
  ~not per-step-read at decode; the truly requantizable INT4 body is ~12.5–12.7 GB. Both
  readings are carried in §5.

## 2. The papers (primary sources; one citation correction)

- **QuaRot** — the proposal cites arXiv:2403.18587. **That ID is wrong**: 2403.18587 is
  "The Impact of Uniform Inputs on Activation Sparsity and Energy-Latency Attacks in
  Computer Vision" (Müller & Quiring, DLSP 2024) — unrelated. The actual paper is
  [arXiv:2404.00456, "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs"](https://arxiv.org/abs/2404.00456)
  (Ashkboos et al.; v1 2024-03-30, v2 2024-10-29; NeurIPS 2024 per repo). Code:
  [spcl/QuaRot on GitHub](https://github.com/spcl/QuaRot) (Apache-2.0; last push 2024-11-26).
  What it actually does (paper v2 + repo, verified):
  - Residual-stream rotation Q (randomized Hadamard) is folded OFFLINE into weights
    (computational invariance; RMSNorm commutes up to norm preservation).
  - It ALSO rotates the FFN intermediate space before W_down via an **ONLINE** Hadamard
    ("1.5 Hadamard transforms per layer" online, O(d log d)) and rotates **Q/K ONLINE
    head-wise after RoPE** (RoPE prevents folding into W_q/W_k; both Q and K rotated so
    scores are unchanged; post-RoPE 4-bit KV). These online ops serve **activation/KV
    quantization (W4A4/W4A8/KV4)** — they are NOT part of a weight-only scheme.
  - Weight-only results exist (Table 7, asymmetric per-column, activations/KV FP16):
    Llama-2 7B FP16 5.47 — RTN W3 Inf / GPTQ W3 NaN / **QuaRot-GPTQ W3 6.09**;
    Llama-2 70B FP16 3.32 — GPTQ W3 5.91 / **QuaRot-GPTQ W3 3.72**; W4 7B:
    GPTQ 8.25 -> QuaRot-GPTQ 5.60. So rotation substantially improves the measured
    WikiText-2 perplexity of low-bit GPTQ on DENSE Llama-2 (an improved-quantization
    result, not losslessness) — evidence class: dense, per-column asymmetric,
    not our packing, not hybrid-GDN.
  - FP16-equivalence of the unquantized rotated model is "exact in principle (computational
    invariance theorem), conditioned on sufficient precision" — i.e. a numerical-tolerance
    claim, **not byte-exact**.
  - Non-power-of-two dims (repo `quarot/functional/hadamard.py`, `get_hadK`): supported
    composite Hadamard orders K in {172, 156, 140, 108, 60, 52, 40, 36, 28, 20, 12} with
    n/K a power of two, else `assert is_pow2(n)`.
- **SpinQuant** — [arXiv:2405.16406](https://arxiv.org/abs/2405.16406) (Liu et al.; v1 2024-05-26, v4 2025-02-20; ICLR 2025):
  LEARNED rotations (vs QuaRot's fixed randomized Hadamard — random rotations can differ by
  up to ~13 PPL-points, so learned-vs-random is a real axis); abstract scope is 4-bit
  weight+activation+KV; no 3-bit weight-only
  claim in the abstract; rotation parameterizations "identical outputs in full-precision"
  — same invariance class. Not a substitute for QuaRot evidence here; cited for the
  learned-vs-fixed distinction (fixed Hadamard is the cheap offline-foldable choice).
- **AutoRound (Intel)** — [intel/auto-round](https://github.com/intel/auto-round): rotation support exists as
  an EXPERIMENTAL algorithm composition (`--algs hadamard,awq,signround`; `quarot` listed
  as composable alg). Export bit-widths W2A16/W3A16/W4A16/W8A16 (+ G32/G64 grouped W2;
  5/6/7-bit via Humming kernel, stated CUDA-only). The README makes **no statement about
  serving rotated checkpoints in vLLM/XPU** (no online-kernel contract, no rotated-mode
  runtime requirements) — UNVERIFIED; must be verified from an actual exported config
  before any use. Regardless, its W3 exports cannot load on our stack (§3).

## 3. The INT3 hard blocker (loader -> packing -> kernel chain, from pinned trees)

Local pinned trees inspected (static, no GPU):

1. **vLLM GPTQ loader** — `~/vllm-src/vllm` @ `066ae99e24ef6aedc73c94b876c4f297d35a45e4`
   (2026-08-13): `vllm/model_executor/layers/quantization/auto_gptq.py`
   `TYPE_MAP = {(4, True): scalar_types.uint4b8, (8, True): scalar_types.uint8b128}` —
   **bits in {4, 8} only**; anything else raises "Unsupported quantization config" at parse.
   Also `pack_factor = 32 // weight_bits` (floor division): 32 // 3 = 10 is an integer
   quotient, but 32/3 is not an integer — a 10-values-per-int32 layout would carry only
   30 of 32 bits (effective ~3.2 data bits/value). That is one hypothetical layout, not a
   universal claim about GPTQ-3bit: cross-word layouts also exist (e.g. 8 values per
   3 bytes = 24 bits exactly, requiring multi-byte load+shift on unpack); none of them is
   implemented or loadable here.
2. **XPU WnA16 kernel chooser** — `vllm/model_executor/kernels/linear/mixed_precision/xpu.py`:
   `_XPUWNA16_SUPPORTED_QUANT_TYPES = (scalar_types.uint4, scalar_types.uint4b8)`;
   group_size must be a multiple of 32; input size multiple of 32. The kernel calls
   `torch.ops._xpu_C.int4_gemm_w4a16`.
3. **vllm-xpu-kernels** — `~/vllm-src/vllm-xpu-kernels` @
   `4043e76d7c64262e2096ea293124fbce1d5c6087` (2026-08-13, v0.1.13-5):
   `csrc/xpu/ops.h` + `torch_bindings.cpp` expose `fp4_gemm` (MXFP4 e2m1x2),
   `int4_gemm_w4a16`, `int4_gemm_w4a8` (+ fp8 paths, grouped GEMM). The W4A16 op
   (`onednn_matmul.cpp:199`) takes B `[k,n]` int4-packed with zp `[k/group_size, n/8]`
   (4-bit-packed zeros) into oneDNN joint dtype `f16_int4`/`bf16_int4` — **there is no
   `num_bits` parameter anywhere in the chain**; 4 is baked into the datatype.
4. Upstream release notes v0.1.12–v0.1.15 (per `worklogs/RESEARCH-online-novel-levers-20260923.md`:
   sampler, fused LayerNorm, MoE topk, MXFP8/block-FP8 grouped GEMM, DSV4 fused kernel)
   add **no W3/W2 weight-only GEMM**. One external search-level mention of "Xe2 W4A16/2-bit
   GEMM kernels (arXiv Jan 2026)" was NOT verifiable against the pinned tree or release
   notes — mark unverified-external; even if a W2 kernel existed it would not be W3.

Conclusion: a uniform INT3 artifact on this stack is not a conversion problem — it is a
new-kernel + new-loader + new-packing program (escalation class to b70-kernel-engineer /
upstream). Per the quantization decision record (Decision 3), format reputation or paper
numbers do not change this gate.

## 4. Rotation feasibility on THIS architecture (GDN + full attention)

Granularity findings (architecture-verified against §1 config):

- **Residual-stream rotation (QuaRot's Q, order 5120): offline-foldable in exact
  arithmetic for this hybrid model, provided GDN/attention INTERNAL channel spaces stay
  fixed and input/output projections absorb the rotation.** Do NOT claim GDN mathematically
  forbids residual rotation — it does not, under that constraint:
  - every reader of the residual stream is a linear (GDN in_proj, attn qkv, MLP gate/up,
    MTP-head input, LM head) whose input side absorbs H, or an RMSNorm (norm-invariant);
  - every writer's output side absorbs H (GDN out_proj, attn o_proj, MLP down_proj,
    embedding); elementwise ops (SiLU/Sigmoid gates, decay, causal conv on a/b) live in
    internal channel spaces that are left unrotated, so they are untouched;
  - the GDN state update commutes with a v-space-only rotation (S <- alpha*S(...)+v k^T
    with elementwise decay acting along the k-space columns: R_v S commutes with column
    scaling; output S q -> R_v (S q)), so a v-space rotation folded into in_proj/out_proj
    is exactly available if wanted;
  - what is NOT available: rotating q/k spaces (per-head L2 norms and data-dependent
    elementwise decay are basis-dependent) and anything crossing SiLU/gate boundaries.
- **Online transforms: the full QuaRot activation/KV online extras (post-RoPE Q/K head
  rotation, KV rotation) are unnecessary for the weight-only scheme** — the residual
  rotation folds offline. QuaRot's online FFN-intermediate Hadamard before W_down exists
  primarily to serve W4A4, but an online intermediate rotation could ALSO help weight-only
  W4 quality; it is a separate option that sits outside the selected no-runtime-overhead
  branch (recorded as a branch decision, not classified as W4A4 drift). Skipping it means
  W_down's input (intermediate, 17408) is not smoothed — the honest cost of the selected
  branch.
- **Dimension support (QuaRot's own code)**: hidden 5120 = 20 x 256 — K=20 IS in
  `get_hadK` (fast, full mixing). Intermediate 17408 = 17 x 1024 is divisible by NONE of
  the supported K values — smoothing that space needs either a new composite Hadamard
  added (order 68 = 4 x 17, Seberry-class, constructible but must be added to the code,
  as QuaRot did for Llama dims) or a block-diagonal 17 x H_1024 (orthogonal, fast, but
  only mixes within 1024-channel blocks — weaker; block-diag R does not spread outliers
  across blocks). Per-head v/qk spaces (128, 256) are powers of two — fine.
- **"FP16 exact equivalence before quantization"**: the right claim is numerical-tolerance
  equivalence against the BF16 reference (float reassociation from folded rotations), per
  QuaRot's own "assuming sufficient precision". It is provable cheaply one layer at a time
  (cosine/max-abs logit deltas on fixed positions) — but it is NOT byte-exact, and it does
  not transfer to task quality (correctness ladder: `logit_or_KL_parity_tested` is below
  `task_quality_tested`).

## 5. Byte/latency arithmetic (E0 toy — explicitly a MODEL, not measurement)

Step-byte model (§1): 21.44 GB = body-INT4-incl-embed 15.2 + F16 verify head 2.543 +
draft INT4 3.4 + FP8 KV 0.3. All "ms" figures below are model floors (GB / 608 GB/s peak,
or /540 GB/s where marked, the external-claim streaming rate), not measured times.

| Scenario | Step bytes | Floor @608 | Floor @540 | Speedup bound vs 21.44 GB |
|---|---:|---:|---:|---:|
| Champion model (4-bit body) | 21.44 GB | 35.3 ms | 39.7 ms | 1.000x |
| INT3 body, data-only 0.75x on all 15.2 GB (coordinator toy, same exclusions) | 17.64 GB | **29.0 ms** | 32.7 ms | **1.215x** |
| INT3 body, G128 scales counted (3.125/4.125 = 0.7576 on all 15.2) | 17.76 GB | 29.2 ms | 32.9 ms | 1.207x |
| INT3 body only on the truly-quantizable ~12.66 GB (embed stays F16, same exclusions) | 18.4 GB | 30.2 ms | 34.0 ms | **1.17x** |
| INT3, hypothetical int32 10-value layout (10 x 3-bit per int32 + G128 scale = 3.325 bits/elem) | ~18.5 GB | 30.4 ms | 34.2 ms | 1.16x |
| ALL bytes 4->3 (head/draft/KV too — unreachable, head gate-locked) | 16.08 GB | 26.4 ms | 29.8 ms | **1.333x (absolute max)** |
| Proposal's arithmetic (13.1 GB @ 540) | 13.1 GB | 21.5 ms | 24.2 ms | 1.64x — requires quantizing head+draft+KV below current gates |

- The proposal's own numbers (24.2 ms, 3.42 tok/step => 141 tok/s) require 13.1 GB step
  bytes — only consistent with ~2 bpw on the body PLUS a sub-F16 head, i.e. it silently
  assumes the head/draft quantizations that already FAILED their gates (§8) and that the
  MTP acceptance stays 3.42 (unproven; see §6).
- "+20–30%" exceeds the body-only bound (1.16–1.22x) at its top end and is only reachable
  if the step is purely byte-bound, packing is free, acceptance is unchanged, and rotation
  adds zero runtime — while the stack's own model says it already runs at 75–89% of the
  byte floor, so pure byte-ratio reasoning overpredicts.
- 39.8 -> 25 ms = 1.59x is above even the all-bytes bound (1.333x) — **within this
  stipulated model** (fixed traffic composition, fixed 608/540 GB/s, no other
  improvements). The traffic model itself carries uncertainty (notably whether the F16
  embedding is per-step-read), so this is a model-scoped statement, not a universal
  impossibility claim; but no 4->3-bit accounting inside this model reaches 25 ms.
- Draft+verify cost at 4->3: draft is ALREADY INT4 (champion pair); re-quantizing it to 3
  buys <=0.82 GB (~4%) at drafter-quality risk — not in the E0 toy (same exclusions).
  Rotation runtime overhead: zero for the folded weight-only scheme; nonzero the moment
  any online (activation-side) transform is added — which is W4A4 drift, out of scope.
- No tok/s ceiling is claimed from any of this: acceptance (§6) and measured overhead
  ratios are unknowns; the numbers above are bounds on a model, not predictions.

## 6. Acceptance risk (target-side 4->3 changes the target distribution)

Recorded fleet mechanism (quantization decision record addendum 2026-09-14b; DFlash2
closure): draft acceptance is conditioned on the TARGET's hidden-state/logit distribution;
quantized targets shift it. The native BF16 MTP head held up at INT4 targets, but every
0.05 alpha is worth +8.3–8.8% decode (D9 economics) — an INT3 target that costs even
0.05–0.10 alpha erases most of the (<=1.2x) byte bound. Any W3 program must gate on
measured alpha with the O2-style protocol before any speed arithmetic is quoted.

## 7. What survives, and the bounded test ladder (smallest first, explicit stop gates)

Surviving program (weight-only rotated-GPTQ, P8 framing; QUALITY/capacity axis only):

- **T0 (done in this worklog): feasibility from source.** Loader/kernel chain closed;
  rotation foldability mapped; dimension support mapped (5120 OK via K=20; 17408 needs
  new construction or block-diag).
- **T1 — one-layer equivalence proof (offline, needs BF16 source, no serving):** fold a
  randomized Hadamard H_5120 (+ optionally per-head v-side H) into ONE GDN block and ONE
  full-attention block of the BF16 source; compare per-position hidden states/logits vs
  the unrotated reference on a fixed calibration slice. PASS: pre-declared tolerance
  (e.g. max-abs / cosine at the layer output within the fp16-matmul noise floor measured
  in the D8 screen class, 99.58%-top-1-equivalent perturbation). This is numerical
  tolerance, not byte-equality. STOP if tolerance fails (rotation numerics on hybrid-GDN
  unproven — that is the new knowledge this buys).
- **T2 — one-layer quantization quality at iso-bytes:** quantize the rotated vs
  unrotated single block to W4 G128 (the only loadable width) with the AutoRound-contract
  quantizer; compare reconstruction error / next-layer logit deltas. This answers
  "does rotation buy quality at fixed INT4" for this architecture at layer scope.
  AutoRound's experimental `--algs hadamard`/`quarot` composition may generate this, but
  its rotated-export runtime contract is UNVERIFIED — T2 is offline comparison only.
- **T3 — whole-model fake-quant parity (if T1/T2 pass):** logit/KL parity vs BF16
  reference across positions (per correctness-validation scopes; the local
  llama-perplexity harness bug on hybrid-GDN models (RL-025, upstream llama.cpp#28879)
  makes that assay invalid — do not rank on it), then task quality. Only then a serving screen.
- **T4 — W3 anything:** BLOCKED upstream of all of this on kernel/loader/packing
  (§3). Escalate to b70-kernel-engineer as a new-kernel program with the QuaRot Table-7
  motivation attached; do not assume it.
- **Explicit stop gates:** stop after two failed quality gates (P8 rule); the selected
  branch carries no online transforms (a per-FFN online intermediate rotation is a
  separate branch, excluded here on runtime-cost grounds, not on quality); activation-quant
  programs (W4A4/W4A8) remain separate, previously divergent scopes — not part of this
  weight-only branch; no speed claim from T1–T3; source artifact preserved unchanged; new
  artifacts get new explicit names; no output index written through a source symlink.

## 8. The verify-LM-head gates (verified failed; do not reopen unchanged)

- **INT4 (D8/L10, measured offline 2026-09-21):** real 248320 x 5120 head, exact draft
  INT4 g128 quantizer, 4,096 hidden states: top-1 agreement **72.07%** vs the >=99%
  promotion line; fp16-matmul noise floor 99.80% (INT4 flips modal tokens ~140x more
  often); top-32 overlap 81.7% vs 99.85%; implied median first divergence 2–3 tokens vs
  >2000 gate. RETIRED. Evidence: `scripts/tmp/screen_verify_head_int4_divergence.py`;
  OUTCOMES.md L10; LEVERS.md D8.
- **FP8 (synthetic screen 2026-09-24, `EXL3-PRE-REPLAN-20260924.txt` Frontier 2):**
  block-FP8 E4M3 g128 top-1 **93.41%**, first disagreement at state 32, implied median
  divergence 10.5 steps; per-row FP8 93.33% / 10.4 steps; FP16-matmul baseline 99.58%
  top-1 / 99.82% top-32. Both REJECT under the same gate. Evidence:
  `scripts/tmp/screen_verify_head_fp8_divergence.py`.
- Scope discipline: the >=99% top-1 line is an offline **promotion screen**, not a
  quality pass — a 99% top-1 result would still not establish task quality
  (`logit_or_KL_parity_tested` != `task_quality_tested`). Reopening requires a genuinely
  changed hypothesis (e.g. T1–T3 rotation first), never a re-run of the same quantizer on
  the same head.

## 9. Source register (primary; SHAs and dates)

- Local: `/mnt/<sata-disk>/Qwen3.8-27B-BF16/config.json` (static read);
  vLLM tree `066ae99e24ef6aedc73c94b876c4f297d35a45e4` (2026-08-13);
  vllm-xpu-kernels `4043e76d7c64262e2096ea293124fbce1d5c6087` (2026-08-13, v0.1.13-5);
  deployed wheel 0.1.12.3 (image digest f01e24f6..., serving.md).
- Papers: [QuaRot, arXiv:2404.00456](https://arxiv.org/abs/2404.00456) (v1 2024-03-30, v2 2024-10-29);
  [SpinQuant, arXiv:2405.16406](https://arxiv.org/abs/2405.16406) (v1 2024-05-26, v4 2025-02-20, ICLR 2025).
  NOTE: the proposal's cited ID arXiv:2403.18587 is a different, unrelated paper.
- Code: [spcl/QuaRot](https://github.com/spcl/QuaRot) (pushed 2024-11-26; `quarot/functional/hadamard.py`
  `get_hadK`); [intel/auto-round](https://github.com/intel/auto-round) (experimental `--algs hadamard` /
  `quarot`; W2/W3/W4/W8 WOQ exports; no XPU rotated-runtime statement — unverified);
  [vllm-project/vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels) (releases v0.1.12–v0.1.15 contain no
  W3/W2 weight-only GEMM per the 2026-09-23 survey worklog).
- Model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (created 2026-08-05; public; matches
  local config). No Qwen3.5 substitution was needed or made.
- Search-level (secondary, unverified beyond listing): ggml-org/llama.cpp issue
  "Support QuaRot quantization scheme" (2024-04-02) — llama.cpp has no rotated-model
  runtime; an external "Xe2 2-bit GEMM (arXiv 2026-01)" mention — not found in pinned
  trees or release notes; treat as nonexistent until demonstrated.

## 10. Outcome

- Changed files: this worklog only. No GPU, no conversion, no download, no shared-doc
  edits, no commit.
- Checks: situation compiler ran (research intent, halt=false); 12-file local working-set
  budget respected; all kernel/loader claims read from pinned source at recorded SHAs.
- Unverified items (explicit): AutoRound rotated-export runtime contract; existence of
  any XPU W2/W3 kernel anywhere newer than v0.1.13.1 (none in evidence); hybrid-GDN
  rotation numerics (T1 is the deciding experiment); measured (not modeled) champion step
  time / traffic decomposition (Main owns that attribution with the P4-trace caveat).
- Next deciding experiment: T1 one-layer BF16-source rotation equivalence (needs an
  assigned GPU window or CPU-scale slice; no serving, no full conversion).
