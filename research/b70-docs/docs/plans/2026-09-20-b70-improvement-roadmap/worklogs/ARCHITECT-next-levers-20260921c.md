---
title: "ARCHITECT decision — ranked next levers after the champion pair (step budgets, kernel scope, upstream plan)"
type: plan
date: 2026-09-21
status: final
---

# ARCHITECT decision — next levers, 2026-09-21 (third pass)

Inputs: `worklogs/P4-20260920.md` §1–§17, `worklogs/P3-20260921.md`,
`worklogs/ARCHITECT-next-points-20260921{,b}.md`,
`models/nemotron/nemotron-3.5-lightning-30b-a3b/serving.md`,
`research/2026/dsv4-k216-prefill-ubatch-20260825.md`. Read-only session, no GPU run.
All numbers PROVISIONAL — NOT FOR PUBLIC HEADLINE.

## 0. Two step budgets that fix the ranking

**Champion decode (patched, MTP4, p8192).** Per step the engine reads: target
body INT4 ≈ 15.2 GB, the one remaining F16 head application 2.543 GB, draft side
INT4 ≈ 4×(0.636+0.212) = 3.4 GB, KV(fp8, 16 full-attn layers, 8 K ctx) ≈ 0.3 GB
→ **≈ 21.4 GB ⇒ 35.2 ms at 608 GB/s peak**. Measured 80.3 tok/s at acceptance
0.55–0.69 ⇒ tokens/step 3.2–3.76 ⇒ step 39.8–46.8 ms ⇒ **75–89 % of the peak-BW
floor, i.e. ≥ ~85 % of a realistic effective floor**. Consequences:

- Decode is now byte-bound. Total remaining decode-kernel headroom on this stack
  is ≤ 11–25 % of step time even at perfect streaming — **no decode kernel
  project clears a "breakthrough" bar**.
- The **largest removable single item is the 5th (verify) head read: 4.18 ms/step
  = 9–10.5 %**. Everything else big (body INT4 25 ms) is rate-class-floored.
- MTP depth 4→6 is arithmetically a wash: tokens/step +4–10 % (at a = 0.55–0.69)
  vs draft bytes +1.7 GB ⇒ step −6–7 %. Do not spend a session on it (RL-026
  stays a blocked cell).

**K216 routed prefill (ub512).** gate/up moves ≈ 1.39 GB of IQ3 weights in
28.28 ms = **49 GB/s effective — 8 % of 608**; BW floor 2.3 ms ⇒ **12× op
headroom on bytes alone** (and 1.82 vs ~45 TFLOPS dense = 25× on FLOPS). The FP16
materialization writes+reads back ≈ 2×10.9 GB ⇒ **≈ 36 of the 86.6 ms/layer** —
the single dominant waste, and it is removable without touching math semantics
(reconstruct to BF16 tile-locally instead of whole-expert). Measured 86.6 ms <
234 ms (the PCIe-streaming time for 2.08 GB at 8.9 GB/s), so the routed weights
are VRAM-resident — the headroom is real on-card, not host-I/O.

## 1. Ranked levers

### L1 — K216 grouped-IQ3 prefill kernel (build; biggest scoped win on the fleet)
- **Mechanism:** device-resident grouping + tile-local IQ3→BF16 reconstruction in
  SLM feeding DPAS; no whole-expert FP16 materialization, no host counting-sort,
  no per-expert host loop. Kills the ≈36 ms/layer expand traffic and lifts the
  49 GB/s effective stream toward BW floor.
- **Arithmetic:** P3's pre-declared E0 = 1.5–2× e2e prefill (130.8 → ~200–260 pp
  at ub1024). Floor case: op → ~4–6 ms/layer (weight-read + DPAS at the measured
  ~45 TFLOPS dense-class efficiency, M≈14 rows/expert keeps it BW-bound) ⇒
  3.72 s → ~0.25 s, e2e 5.06 → ~1.6 s ⇒ **~320 pp**. Quote E0 as the commit; the
  floor is the stretch.
- **Resource:** multi-session code change (llama.cpp SYCL `MUL_MAT_ID` path), no
  download; P3 already owns it.
- **Minimal version first (the gate):** standalone single-op kernel in P3's
  operator harness — one layer's gate/up, n=512, SLM tile dequant + oneMKL-class
  DPAS tile, parity-checked against the current path. **Gate: ≥2× on the isolated
  op at n=512 or the structure theory is wrong.** Kill: <2× ⇒ withdraw the E0,
  harvest ub1024 as config result, close P3 config-harvested. Second gate
  (design-selecting, not go/no-go): stage timers splitting counting-sort vs
  expand vs oneMKL at M≈14–28 rows/expert.
- **Not** a DPAS s4/IQ3-native fantasy: IQ3 is not a native format; reconstruct
  to BF16 (same numerics as today) first generation. s4×s4 second generation only
  if a W4A4 activation-quant quality gate is ever passed — separate decision.

### L2 — Cn batch-shape knee on the champion (measurement-first, ~1 h)
- **Mechanism:** under graph capture the step is a fused batch: body 15.2 GB and
  head 2.54 GB are read **once per step for the whole batch** (verify M = 5B,
  draft propose batched), so per-sequence byte cost falls ≈ 1/B. This is the
  regime the 2026-08-18 screen accidentally measured (96.6 max tok/s).
- **Arithmetic:** if step time grows sublinearly, aggregate decode at B=4 should
  approach 2–3× the C1 number (**~160–240 tok/s aggregate**) before the next wall
  (scheduler, GDN spec kernels, acceptance under load). Converts the pair from a
  C1 curiosity into a serving-capacity claim.
- **Probe:** champion stack, one declared server, `--max-num-seqs B`, B ∈
  {1,2,4,8}, p8192, n=3 intervals per B, fresh same-shape warmup per B (XPU
  graphs recapture), prefix-cache counters bracketed, successful-tokens/interval
  metric. **Kill:** aggregate at B=4 < 1.5× C1 ⇒ serialization dominates; record
  the capacity ceiling and stop.

### L3 — Draft-side INT4 transfer: Nemotron DFlash n7 (scope → convert → A/B)
- **Why plausible:** the route already has the enabler (XPU graphs + async
  scheduling on, per the serving card) and a **BF16 drafter**. Step budget:
  186.6 tok/s at ~2.3 tok/step ⇒ ~12 ms/step ⇒ **~7.4 GB/step budget**, of which
  the INT4 A3B target body is only ~2 GB — a BF16 drafter of 1.5–3 B would be
  **3–6 GB/step, i.e. the dominant item**; pair saves ~72 % of draft bytes.
- **Bound (must be computed before any GPU):** win ≤ (BF16−INT4 draft
  bytes)/608 GB/s ÷ step time. Drafter ≥ 25 % of step bytes ⇒ proceed; below ⇒
  the ceiling is < ~15 % and it does not clear the bar.
- **Zero-GPU scoping (~30 min, do first):** read the DFlash proposer loop in the
  pinned vLLM source (head applications per drafted token? shared target head?)
  + drafter safetensors headers — the exact P4 §1–§3 method, reused.
- **Resource:** one GPTQ-INT4 conversion of the drafter (quality-gated, disk) or
  the same runtime-patch route if the head is shared; then one n=3 A/B at the
  canonical cells (~1 h, 150 W as recorded). **Kill:** acceptance drop > 5 pts
  absolute costing more than the traffic saves; or scoping bound < 15 %.
- **Qwen3.8 DFlash2:** stays closed (RL-027/028/029). The transfer method above
  is what reopens *with* it if upstream fixes land — do not revive manually.

### L4 — Verify-path (5th) head quantization (quality-gated, small)
- Arithmetic: −(2.543−0.64) GB/step = −3.1 ms of 39.8–46.8 ms ⇒ **+7–9 % decode**.
  Not a breakthrough; it breaks the weight-only-on-draft safety story (touches
  target logits) so it needs the D5 logit-quality gate (perplexity/task per the
  quantization decision record). W8 intermediate step available at −1.9 ms with
  milder risk. Sequence after L2/L3, never before.

### L5 — Capture-era runtime A/Bs (each ~30–40 min, expected 0–15 %)
1. torch.compile **on top of** XPU-graph capture on the champion stack (the two
   known arms are compile-only +11 % pair effect and capture-only champion;
   never measured stacked).
2. `--async-scheduling` on the champion stack (the Nemotron recipe has it, the
   Qwen champion runs without it).
3. 150 vs 230 W on captured decode. **Prediction: flat (±2 %)** because the step
   is byte-bound at fixed 608 GB/s. Flat ⇒ permanent 150 W serving with no loss
   (an ops win, not speed); not flat ⇒ compute/clock headroom exists and the
   byte-floor model needs revision. This is a falsifier for §0, not a tuner.

### L6 — Deprioritized by arithmetic (do not spend GPU)
MTP depth 4→6 (wash, §0); champion decode kernel work (≤ 11–25 % total headroom,
byte floor); s4×s4 anywhere on decode (no A4 path, byte-bound); W4A4 dense-Qwen
prefill (dense prefill is not the pain point; K216 is); prefix-cache as a speed
lever (orthogonal, workload-dependent, defaults on in production — hygiene note
only).

## 2. Probes under two hours (exact)

| Probe | Class | Cost | Expected | Kill |
|---|---|---|---|---|
| L2 Cn knee B=1..8 | measurement | ~1 h | 2–3× aggregate at B=4 | <1.5× at B=4 |
| Acceptance-variance decomposition (§4) | measurement | ~1–1.5 h | benign/noise verdict | phase-A divergence |
| L5 trio (compile-stack, async, 150 W) | measurement | ~1.5 h all | 0–15 %; flat power | ±2 % each |
| L3 scoping | zero-GPU | 30 min | go/no-go number | draft <25 % step bytes |
| L1 single-op standalone kernel | build | first session of P3 | ≥2× isolated op | <2× ⇒ withdraw E0 |

Measurement-first: everything except the L1 kernel build.

## 3. Upstreaming the draft-side INT4 pair

- **Shape of the upstream change (vLLM):** a draft-only quantization override on
  the speculative-decode config surface, e.g. `--speculative-config
  '{"method":"mtp","num_speculative_tokens":4,"draft_quantization":"int4-g128"}'`,
  honored at draft-model weight load; when `_maybe_share_lm_head()` fires
  (`vllm/v1/spec_decode/llm_base_proposer.py`, `_sample_draft_tokens` path), the
  proposer receives a **second, quantized head module** instead of the shared F16
  reference, and the draft matmuls dispatch through the existing XPU W4A16 INT4
  GEMM (`int4_gemm_w4a16` → `cutlass_grouped_gemm_xe2`; ElementB=uint8 + group
  scales, group_size 128 is inside the supported {32,64,128,256} set — no new
  kernel is needed). Local model-side anchor: the `qwen3_5_mtp.py` forward path;
  re-derive the exact upstream file paths against the PR-time generation.
- **Validity limits to state in the filing:** greedy token-parity proven (P4 §16);
  sampled path safe by rejection-sampling construction but **not**
  distribution-proven (P4 §17 weak discriminators — do not claim it); **C1/TP1
  only** (TP>1 changes head/logits collectives; keep the cookbook warning);
  pays only when host overhead is removed (graphs on) — frame numbers with
  graphs enabled so CUDA reviewers see the same effect class, not a null.
- **Issue-with-reference-patch first, PR second.** Two blockers to a direct PR:
  the acceptance-variance attribution (§4) is outstanding, and the patches are
  not anchored to a clean source generation (claims-audit defect 5). File the
  issue with the +37…46 % A/B tables and the patch pair as reference; convert to
  PR once §4 phase A/B lands clean and the patches are regenerated against the
  current vLLM generation. If phase A shows any nondeterminism: no upstream —
  file it as a defect with the divergent hashes.

## 4. Verification job for the acceptance variance (sceptic-proof package)

The claim to establish: *the patched arm's 0.55–0.69 acceptance band is seed
draw over an INT4-flattened draft distribution, not nondeterminism.* Four
deliverables, ~1–1.5 h GPU total:

1. **Per-seed reproducibility (the core).** Recommended sampling, seed s ∈
   {1…10}, **two fresh-container relaunches per seed**, patched arm, p8192 n=3.
   Sceptic is convinced iff acceptance is identical across relaunches within a
   seed (kills kernel/graph nondeterminism, M2/M3) while the across-seed SD of
   the patched arm ≫ baseline SD (quantifies M1). Baseline arm: seeds 1…5 same
   design.
2. **Localization.** Head-only vs MTP-only arms at 3 seeds: the variance must
   follow the head (sampler-facing) arm. If it follows the MTP-body arm instead,
   the mechanism story is wrong — investigate before upstreaming.
3. **Near-tie accounting (zero-GPU).** Offline draft-forward on ~10k natural-text
   positions: per-position top-1 agreement BF16-draft vs INT4-draft ≥ ~97 % with
   disagreement mass concentrated on small top-2 margins. This is the
   quantitative mechanism: near-tie flips × autoregressive amplification ×
   presence-penalty state = sequence-level band with effective n≈5 (the §15
   observation).
4. **Honest band publication rule.** Acceptance is thereafter quoted as
   mean ± SD over ≥ 10 seeds with prompt-content class named — never a scalar.

**Kill (sceptic wins):** same seed + greedy + fixed prompt diverges across
relaunches (this is phase A of the 2026-09-21b design, still the cheapest single
falsifier, ~25 min) ⇒ nondeterministic kernel ⇒ patches stay local-only,
upstreaming blocked, defect filed with hashes.

## 5. Sequencing

1. L3 scoping + §4 decomposition (one GPU-1 session, ~2 h) — gates upstreaming
   and the transfer, both cheap, both close open threads.
2. L2 Cn knee (one GPU session, ~1 h) — the only measurement that can still
   produce a "noticed breakthrough" (capacity) with zero build.
3. L5 trio opportunistically in the same sessions.
4. L1 kernel build only after its single-op gate passes; L4 last, only on
   operator's D5 authorization.
