---
title: "B70 breakthrough architect pass — first-principles novel levers (substitute run)"
type: research
status: provisional
date: 2026-09-28
author: substitute for b70-architect, provider quota; interim findings inherited and re-verified
publication_gate: PROVISIONAL_NOT_FOR_PUBLIC_HEADLINE
---

# B70 breakthrough architect — novel levers from first principles

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Mission: find NOVEL, extreme
breakthroughs for Intel Arc Pro Bxx inference from first principles (maths,
Xe2 silicon, information travel in the PC, LLM architecture) — not another
online-lever sweep (`external-lever-sweep-20260926.md`,
`b70-engine-opportunities-20260926.md` are exhausted inputs, not repeated
here). No GPU work was run; all magnitudes are arithmetic bounds priced
against measured floors, never predictions.

Substitution note: this run substitutes for `b70-architect` (provider quota
mid-task). The six inherited corrections were re-verified against raw source
and are treated as established. Two additional raw-evidence findings were
derived fresh in this pass: the bimodal per-position acceptance structure
(§B.1) and the EXL3 effective-bandwidth unit inconsistency (§D, rows 7–8).

## Ground truth used for pricing

- Champion round (GPTQ MTP4 stack, now removed on quality grounds):
  **44.8 ms** = host ≈4.8 (mostly overlapped) + draft ≈5.5 (4×0.85 GB
  re-reads at pin rate) + head ≈4.2–4.7 (2.54 GB fp16) + body ≈28–30 +
  misc ≈2. Yield 1+4α ≈ 3.17–3.27 at α≈0.55 → ≈72–75 tok/s.
- Adopted EXL3 route (ADR-0010): artifact 15.70 GiB total; conditional
  one-read budget ≈ **16.68 GB/round** (target 12.230 + head 0.954×4-ish
  reads accounted + MTP 0.213 + embedding 2.543 conditional); recorded
  step medians ≈61.5 ms @150 W / ≈40.6 ms @230 W **if chunks = rounds**
  — unverified (correction 1). Byte floor at 608 GB/s ≈ **27.4 ms/round**
  → at yield ≈2.65 (MTP3, α≈0.55) ≈ **97 tok/s theoretical pin-rate**.
- Qwen3.8-27B geometry (from `config.json` text_config): 64 layers =
  **48 GDN linear-attention + 16 full-attention** (interval 4), hidden
  5120, dense FFN 17408, full-attn 24 q / 4 kv heads × 256 dim,
  GDN 16 k-heads × 128 / 48 v-heads × 128 (per-request recurrent state
  ≈ 48 × 128 × 128 × 4 B ≈ 3.1 MB/layer ≈ **150 MB constant**,
  context-independent), vocab 248,320, MTP head 248320×5120 = 2.37 GiB f16.
  Full-attn KV read per decode token = 16 layers × ctx × 2 KiB (fp8) —
  ≈3.2 GB/step at 100 K ctx, ≈8.4 GB/step at 262 K.
- Buses: GDDR6 ≈608 GB/s class/card (measured effective 430–495 champion);
  P2P copy verified 11.1–11.9 GB/s, **atomics absent** (RL-059);
  H2D ≈8–9 GB/s probe-limited; DDR4-2133; xccl 4 KB allreduce ≈102 µs,
  128×51 KB round ≈14.3 ms (RL-071).
- Rate-class law: ≤8-bit×8-bit ties INT8 (~363 TOPS); BF16 ~170 TFLOPS;
  pure s4×s4 = 2.00× per-instruction (measured) but no W4A4 serving path;
  no FP8 DPAS.

---

# DELIVERABLE A — THE STANDING STRUCTURE (paste-ready for AGENTS.md)

> **Superseded as an authority 2026-10-09:** the living, corrected version of
> this section is `docs/governance/standing-rules.md` (route-specific spec
> stacks, as-found caps instead of 150 W, RL-092 discriminator resolved, DFlash2
> retry condition). The text below is kept as dated evidence.

## A.1 ALWAYS-ON RULES (inviolable invariants)

1. **XPU graph capture always on for vLLM XPU serving.** Any decode number
   taken with capture off is diagnostic-only and never a champion input.
   Every new kernel/patch must be capture-safe by construction and must be
   evaluated **under graph replay** — eager torch-profiler traces are blind
   to graph-replayed kernels (RL-072 lesson).
2. **Locked spec stack:** stock MTP weights, MTP4, greedy drafting,
   draft-side INT4 pair, fp8 KV, model-card recommended sampling
   (presence_penalty 1.5 non-thinking). A change is a named variable with
   a pre-declared gate, not a tweak.
3. **Exact-answer battery + spec-canary before any speed claim.**
   Coherence, replay, and PPL all pass a fluent wrong answer (the
   30-defect). Canary runs at boot and after any stack change.
4. **Prefix-cache policy:** OFF on W4A16-class stacks until vLLM #53912
   is fixed (corruption reproduced 5/6, including zero-hit first
   requests). Elsewhere, cache state is declared and every cold cell
   carries the ≥64-char-entropy + zero-hit-delta guard.
5. **Output preservation:** every decode-path change must be lossless by
   construction or proven byte-identical (greedy) / same-modal-class
   (sampled). No approximate-verify class levers, ever.
6. **Metric naming is a claims rule:** every rate names the counted
   quantity — engine tokens, client post-first tokens, or SSE chunks.
   Chunk-rate ≠ tok/s. C1 and Cn never share an unlabeled column; Cn
   uses successful tokens ÷ declared interval.
7. **α bookkeeping:** α is always the counter form, yield = 1+4α
   (+0.05 α = +6.25 %). Per-position counters are marginal prefix
   survivals, reported separately — never silently treated as conditional
   per-position probabilities.
8. **Single-client C1 gate:** one measuring client per C1 server, with
   prefix-cache and spec counters bracketing each cell.
9. **Placement:** independent dual-card stacks (ADR-0008). TP2 only
   behind the gated matched-control campaign; never a stock Docker TP2
   launch.
10. **Quantization axes stay independent:** target weights, draft body,
    draft/MTP head, verify LM-head, activations, KV — six separate
    decisions; a verdict on one never transfers to another.
11. **Attribution honesty:** "at floor" must declare what floor was
    proven against — active-kernel share is not wall share, and a
    30–40 % tracer overhead cannot establish "no hidden waste."
12. **Thermal/power:** matched-arm floors disclosed, ≤55 °C protocol or a
    formally declared substitution, 150 W restored on exit, empty-GPU
    gate before every load.

## A.2 ALWAYS-TRY FIRST MOVES (cheap-first checklist, in order)

1. Graph capture on (or the engine's equivalent) **before** reading a
   decode number — capture-off numbers answer a different question.
2. Draft-side quantization pair on whatever drafter exists (proven lever
   class: +39.6/+48.6 % measured).
3. Byte-budget / roofline arithmetic BEFORE building: measurement adopts,
   arithmetic prices — never reverse (the k=3 clamp was adopted on
   arithmetic and cost −5.45 %).
4. Exact-answer canary first, speed second.
5. Count the draft-bytes axis: the drafter is re-read k+1 times per
   round; its floor is the k×-read traffic, not its unique bytes.
6. Acceptance-economics at the corrected +6.25 % per +0.05 α — and pull
   the **per-position survival profile** before assuming uniform α
   (the stock profile is bimodal: 0.98/0.98/0.12/0.09 on natural text).
7. Attribution before optimization: measured bucket shares vs physical
   floors; a bucket at floor is closed, but only against a named floor
   (correction 2 discipline).
8. Price the **KV-read growth term** at the campaign's context depth
   (32 KiB/token × N on this geometry — co-dominant with weights past
   ~100 K).
9. Metric audit of the harness itself: what does the counter actually
   count (chunks vs tokens vs rounds)?
10. Placement arithmetic before accepting "one card": hidden states are
    µs-class over P2P (~10 KB vs 15 GB) — ask what *could* run on the
    idle card before declaring the single-card program closed.

## A.3 CLOSED — DO NOT RETRY (compact)

Single-card champion decode kernel/host hunting (byte floor, three
levels, subject to correction-2 caveat); whole W4A16/GPTQ-class symmetric
int4 on Qwen3.8-27B (30-defect ×3 exports; kernel-vs-weights
discriminator still owed); acceptance frontier — MTP-head small-corpus
retrain (D11 −2.5 %), probabilistic drafting (D12 finalist-refuted),
bf16 draft-KV (D14 α −0.042), foreign-corpus shortlist (RL-074 finalist
−1.5/−5.5 %), verify-head INT4 (top-1 72 %), depth >4 (flat ±1.4 %);
TP2 on this topology as served by stock oneCCL (RL-075 −45…−71 %;
peer-IPC/graph collectives segfault); grouped-IQ3 from-scratch kernel
(1.39–1.41× slower); #29107 (+0.08 %); vllm-xpu-kernels 0.1.15
(regression −4.4/−14.5 %); #56096/#53989 fused QK-RoPE (wrong model
class); QuaRot→INT3 (no INT3 XPU kernel; ideal 1.215×); W4A8 prefill
(closed-by-attribution, ≤1.27×); Cascadia PP2 (−0.26 %); DFlash2 on
GPTQ; host-prep cache (+1.5 % vs +4 % gate); max-num-batched-tokens
(±0.25 %); CPU-MoE DSpark; `-sm row`; stock `--tensor-parallel-size 2`.
Re-entry only via a named changed variable.

---

# DELIVERABLE B — BREAKTHROUGH CANDIDATES

Ordered by expected value per GPU-hour within each lens. Every magnitude
is a bound, not a promise.

## Lens 1 — THE MATHS

### B1. Accept-length distribution mining + predicted shallow-round early exit
**Mechanism.** The stock MTP4 per-position survivals are **bimodal**:
0.9802/0.9752/0.1238/0.0891 (D11 baseline decay probe, 202 rounds,
natural text) — conditional P(pos2|pos1) ≈ 0.127 but P(pos3|pos2) ≈
0.72. Rounds are either "locked-in" deep-accept (~12 % run the table) or
"shallow" (~88 % die at the 3rd drafted token). A second window
(p4-champion n=2 metrics deltas) shows ~1.00/0.96/0.88/0.88 — **flat**,
no cliff ⇒ the collapse is content-dependent, not a counter artifact or
a structural drafter bug. If a cheap signal (draft logit margin at
position 2, or an accept-length classifier on the last few emitted
tokens) predicts "shallow round" with ≥85 % precision, halting the
draft at k=2 saves ~2.75 ms of the 5.5 ms draft bucket on ~88 % of
rounds → ~+5.4 % of the 44.8 ms round at ~zero yield loss (the skipped
positions were going to die anyway). With a sharper variant — **draft
only until draft confidence crosses a threshold** — the same math bounds
+4–6 %.
**Novelty.** Nobody mined the accept-length *distribution* here; D9/D10
priced static depth with uniform α; the "dynamic depth" lever was
listed but never attached to a measured mechanism or a predictor. The
bimodality itself is new evidence (this pass).
**Magnitude.** +4–6 % C1 if the predictor works; ~0 if it doesn't.
**Kill-probe.** Zero-GPU first: replay recorded draft/verify traffic (or
one instrumented run) and correlate draft pos-2 logit margin with actual
pos-3 acceptance. If AUC < 0.7, close permanently.
**Bxx transfer.** Mechanism is engine-side sampling logic; identical on
B580/B60; tells future silicon nothing (good — it's free).

### B2. Fork drafting at the collapse position (cheap tree of width 2)
**Mechanism.** Verify rows at M=5 are nearly free — the verify pass is
weight-read-bound (25–28 ms floor) and M=5→M=7 costs ~nothing extra in
bytes. Draft a second candidate token at the collapse position
(pos 3) instead of a longer chain: cost one extra draft pass (~1.4 ms)
for a chance to raise conditional pos-3 acceptance from 0.127 toward
~0.3–0.5 (two independent candidates cover more of the modal set).
Expected yield 3.17 → ~3.4–3.55 → +7–12 % tokens/round at +3 % round
cost → net +4–9 %.
**Novelty.** vLLM tree speculation is "no XPU kernels" today; nobody
priced a *width-2 fork at the empirically-identified collapse position*
vs a chain. This is drafting geometry derived from the measured
distribution, not an upstream feature port.
**Magnitude.** +4–9 % best case; requires a proposer that emits forked
drafts (engine work, medium).
**Kill-probe.** Zero-GPU oracle bound: on recorded verify logits, ask
"would the second-ranked draft token at pos-3 have been accepted?"
(computable from draft top-2 + target argmax logs if logits are
retained). If oracle-lift < +2 % yield, close.
**Bxx transfer.** Pure scheduler/proposer logic — transfers everywhere.

### B3. Margin-aware quantization — the 30-defect as a theory, then a fix
**Mechanism.** All three symmetric-INT4 exports (GPTQModel,
AutoRound→compressed-tensors, AutoRound→auto_gptq on a code corpus)
deterministically flip a fragile in-head decision: `sum(i*i for i in
range(4))` → 30 instead of 14 (an inclusive-range off-by-one; Σ0..4=30).
BF16 answers 14; EXL3 trellis, OV int4, Q4_K_M answer 14. RL-088 shows
the defect is NARROW (real code authoring fine). Hypothesis: symmetric
INT4 introduces a **correlated error direction** — not noise magnitude —
that flips low-margin logit orderings at decision tokens. T2 (rotation
improves W4 quality at iso-bytes, KL 0.40–0.58×) already proved error
*direction* is the operative variable on this stack. The discriminator:
offline INT4 matmul (fp32 accumulate) on real hidden states for the
failing prompt vs the in-kernel output. Kernel output ≠ offline INT4 ⇒
**kernel bug** (accumulation-dtype overflow has fleet precedent) and a
72–80 tok/s quality-correct route resurrects. Equal ⇒ weights ⇒
margin-targeted requant (protect low-margin directions; rotation as the
lever).
**Novelty.** The kernel-vs-weights discriminator was *never run*
(recorded in program state); the margin-aware requant objective
(optimize for decision-boundary preservation, not MSE) is a genuinely
new candidate formulation vs the ledger's format screens.
**Magnitude.** Resurrects the ~3×-faster route class (72–80 vs ~24.6
chunk-rate) if kernel; if weights, a 4 bpw-class quality route with
rotation — 1.22× bytes over EXL3 body-only (correction-5 ratio).
**Kill-probe.** Deliverable C, probe 2.
**Bxx transfer.** Discriminator is engine-agnostic; margin objective
transfers to any future export.

### B4. Exact-argmax verify head via bounded MIPS (priced-negative)
**Mechanism.** The fp16 head (2.54 GB ≈ 10 % of round) only needs
argmax + drafted-token probabilities (greedy verify) — but speculative
verify with recommended sampling needs the target distribution at
drafted positions plus the bonus argmax. Exact maximum-inner-product
search (norm-bounded pruning) reads <100 % of rows only if pruning
fires; on 248 K vocab × 5120 with a single query the realistic prune is
≤50 % rows → ≤+5 % round, and INT4/FP8 pre-screens are already
quality-dead (72 %/93 % top-1). A clustered-row permutation is lossless
but changes softmax denominator bookkeeping — doable, small gain.
**Verdict.** Priced ≤+3–5 %, complexity high — keep as documented
low-EV; do not build without a measured >60 % prune rate on real
queries.
**Kill-probe.** Offline: measure bound-tightness on logged query
vectors; prune rate <40 % ⇒ close forever.

### B5. DPAS-native entropy-constrained codes (research direction)
**Mechanism.** EXL3's trellis (bits=4, head_bits=6) is already a
vector-ish code that survives where symmetric INT4 fails — evidence
that **code shape, not bit-width, carries quality**. The frontier
version: train the codebook so its reconstruction *is* a DPAS-consumed
integer operand (codebook entries {zero + u4×scale} directly legal in
the INT8 rate class), i.e. entropy-constrained VQ co-designed with the
XMX layout — quantization-as-code-layout, not quantization-as-rounding.
P-3b (tile-local reconstruction, 3.60×) proves the consume-side
technique transfers.
**Novelty.** Ledger treats "format" and "kernel" separately; this fuses
them at training time.
**Magnitude.** Unknown; bounded today by the EXL3 attribution (B6) —
if reconstruction is the wall, co-design is the fix.
**Kill-probe.** B6 first; then a one-tensor reconstruction-cost
microbench comparing trellis-decode vs GPTQ-dequant vs a dpas-legal
code.
**Bxx transfer.** This is the thing future Bxx silicon should make
native: a fused "codebook → operand" instruction. Realistic flag for
Xe3+.

## Lens 2 — THE SILICON

### B6. EXL3 decode-step attribution (first-ever, on the ADOPTED route)
**Mechanism.** EXL3 streams ≈16.68 GB/round whose pin-rate floor is
≈27.4 ms; recorded step medians are ~61.5 ms @150 W and ~40.6 ms @230 W
**if chunks = rounds** — i.e. 255–415 GB/s effective vs the champion's
430–495. RL-086 proved it is *power-limited* (+38–42 % from 150→230 W).
A route at 42–68 % of its own byte floor has recoverable slack by
definition — the question is where: trellis reconstruction on Vector
Engines (the IQ3 lesson: non-native formats force scalar unpack),
ESIMD kernel occupancy/launch bounds, or clocks. This is the **only
route where a kernel-level fix plausibly buys 1.3–1.8×** — all other
buckets on the fleet are measured at floor.
**Novelty.** Zero attribution exists for the adopted route — every
floor-closure verdict was measured on the (removed) GPTQ stack. The
"single-card lever hunt closed" verdict does NOT cover EXL3.
**Magnitude.** If VE-bound: up to ~1.5–2× step-time (24.6→~40+ tok/s
class) — the largest single plausible C1 win left on this host.
**Kill-probe.** Deliverable C, probe 1 (unitrace or in-container kernel
timing on one decode round + a clock/utilization read under load).
**Bxx transfer.** Direct — B580/B60 run the same ESIMD path; the answer
also fixes the "ESIMD decode is power-limited" mystery class.

### B7. Persistent decode megakernel — priced honestly
**Mechanism check.** The two megakernel walls are launch gaps
(~560 µs/token) and activation round-trips (8 KB × ~280 transitions ≈
2.3 MB/step ≈ 4 µs of bandwidth). Graph capture already removes launch
gaps on the vLLM stack, and activation traffic is negligible vs 15 GB
weights ⇒ on the champion-class stack a persistent kernel buys ~1–3 %
at best. **Exception:** if B6 shows EXL3's step is gap/dispatch-bound
(not stream-bound), a persistent EXL3 round-kernel is the correct fix —
the megakernel question reduces to the B6 measurement.
**Verdict.** Contingent on B6; do not fund independently. If EXL3 is
stream-bound at 415 GB/s (correction-8 arithmetic), megakernel is dead
here too — record-close, don't build.
**Kill-probe.** B6's composition table answers it at zero extra cost.
**Bxx transfer.** On any card where a serving route is dispatch-bound,
single-card 32-WG persistent grids are the answer; never cross-card.

### B8. Verify-M=5 w4a16 small-M kernel (bounded residue)
**Mechanism.** The old M-regime probe measured n=8 at 8.87 vs ~17.3
TFLOPS-floor ⇒ ≈2× headroom *on that op*. The 44.8 ms attribution puts
the body bucket at 28–30 ms vs a 25–28 ms floor ⇒ at most ~2–5 ms
recoverable ⇒ +4–10 % e2e ceiling — and only if the gap is real
(correction 2: RL-072's "87.7 % GEMM" is active-kernel share; ~18 ms of
the 58.4 ms round was uncovered, i.e. the gap might be *larger* than the
floor-model says — or the floor model's bytes are wrong). A specialized
M=5 dpas tile schedule is the only kernel candidate if the gap
survives re-measurement.
**Kill-probe.** One instrumented round that names the M=5 dispatch
(repeat of RL-072 with corrected accounting); gap <1.5 ms ⇒ close.
**Bxx transfer.** The M=4–16 regime matters on every Bxx card for any
speculative stack — worth one kernel family, once, if the gap is real.

### B9. s4×s4 (W4A4-class) decode — priced negative, recorded
**Mechanism check.** s4×s4 doubles MACs/instruction (measured 2.00×)
but decode is **bandwidth**-bound: activations are ~10 KB, weights are
already 4-bit — there are no A4 bytes to save on the draft side and no
compute wall to halve. Draft-only W4A4 additionally degrades draft
accuracy → α penalty −0.05 = −6.25 % against a compute win that doesn't
exist in the cost model. Prefill A4 is a quality question, not a
throughput one.
**Verdict.** Do not build on this host for decode. Recorded so nobody
re-prices it: the rate-class fact is real, the application is absent.
**Bxx transfer.** Worth re-asking only if a future Bxx part is
compute-limited at decode or gains an A4-native quality path.

## Lens 3 — INFORMATION TRAVEL

### B10. Cross-card meta-speculation: draft-behind-verify on the idle GPU
**Mechanism.** The dependency chain draft(i+1)←verify(i) serializes
5.5 ms of draft before every 30 ms verify. GPU1 is idle for ~30 ms of
every round. Move the draft stack (0.85 GB INT4, its own KV) to GPU1;
the only crossing is hidden state + token IDs (~10 KB → ~1–3 µs at
11.5 GB/s). During the verify window, GPU1 **pre-drafts the next
round conditioned on hypothesized verify outcomes** — the outcome
distribution is peaked (B1: ~88 % of rounds are shallow-accept ending
in bonus-token), so hedge the top-1 accept-length × top-2 bonus-token
candidates (~3 chains ≈ 16.5 ms < 30 ms window). On a hit (est.
P≈0.5–0.7), round i+1's draft is free.
**Expected value.** Removes ~5.5 ms × hit-rate of 44.8 → +4–8 % C1;
second-order: GPU1 owns draft KV → more context headroom on GPU0.
This is a placement that is neither TP (no collectives, no split
weights) nor PP (no layer split — the drafter is a *different network*
running concurrently on a different outcome hypothesis).
**Novelty.** No "speculate on the verification outcome" mechanism
exists in the ledger or the sweeps; ADR-0008's independent-stacks
default left this concurrency unpriced because TP/PP were the only
topologies considered.
**Kill-probe.** Zero-GPU statistics first: from recorded verify logs,
measure P(bonus ∈ top-2) and P(accept-length=modal) — if joint hit
rate <0.35, dead. Then a two-process async-serving prototype (GPU1
draft server + GPU0 verify server, ids over loopback) before any
kernel work.
**Bxx transfer.** Any dual-Bxx box; the primitive is engine-level, not
silicon.

### B11. Flag-slot (sequence-number) collectives over copy-only P2P
**Mechanism.** No remote atomics — but P2P *copies* are verified. A
low-latency protocol needs only ordered writes + local polling: writer
pushes data then a monotonically-increasing seqno into the peer's VRAM;
the reader's persistent spin-kernel polls its own memory for the seqno
(no atomic RMW required — the LL/LL128 pattern). Host-staged xccl costs
~102 µs/4 KB; a device-resident flag-slot round-trip is bounded by copy
latency ~5–15 µs ⇒ **5–10× cheaper small collectives**. Re-priced TP2:
weights halve (15.2→7.6 GB ≈ 13 ms) + collectives ~2–3 ms + head/draft
≈8 ms + host ~4 ms ⇒ ~28 ms vs 44.8 ⇒ **~1.5–1.6×** — inside the
+20–50 % gate band for the first time (stock-oneCCL pricing had
collectives at 14.3 ms).
**Novelty.** RL-075 tested *stock* oneCCL transports only; no
copy-only-tolerant collective protocol was ever priced or built here.
The segfault class (`ze_a2a_reduce_scatter`) is avoided by
construction — no peer-IPC, no remote atomics, no graph-recorded
collectives needed.
**Honest catch.** Graph-capture safety of device-side spinning
primitives is unproven; TP2 also re-opens the graphs-off penalty that
previously cost the run. This is a *research primitive*, not a serving
commitment — the same primitive also powers B10's sync cheaply.
**Kill-probe.** Zero-GPU design review + a standalone spin-copy
latency microbench (outside vLLM): sustained <25 µs/4 KB round-trip ⇒
proceed to the collective-correctness gate; >60 µs ⇒ dead (host-staged
wins on simplicity).
**Bxx transfer.** If Intel never ships remote atomics on consumer Bxx,
this is THE enabling primitive for every multi-card topology.

### B12. Host residual paths (248 K sampling, detokenize, pinned buffers)
**Mechanism.** Host bucket ≈4.8 ms mostly overlapped; halving it
measured +1.51 %. Remaining named items: per-token work over the
248 K-vocab distribution (sampling + presence-penalty scan),
detokenize, SSE encode — all CPU, all overlappable. GPU-side fused
sampling exists in vllm-xpu-kernels but 0.1.15 was a measured
regression; a targeted presence-penalty/top-k path on the pinned
0.1.12.3 generation is a small surgical candidate.
**Magnitude.** +1–3 % class. Cheap only if profiled inside the same
attribution window as B6.
**Kill-probe.** Included in probe 1's bucket table; skip if not
visible.

### B13. GPU1-VRAM expert bank for flash-next (capacity, not speed)
**Mechanism.** flash-next's CPU-expert layers run on DDR4-2133 mmap
(the 21.6 tok/s placement). If spare GPU1 VRAM exists, holding the cold
expert bank in *peer VRAM* and staging over P2P (11.9 GB/s) beats
DDR4 page-cache streaming (~real <34 GB/s with dequant+sync overheads
per the K216 analysis). Capacity/cleanliness win; needs the expert
selection traffic priced (selected experts ≈55 MiB/token-class ⇒
~4.8 ms P2P vs measured CPU path — likely a wash; the *real* version
is B14).
**Verdict.** Recorded as a placement note for the flash-next program,
not a speed lever.

## Lens 4 — LLM ARCHITECTURE RESHAPED FOR THIS MACHINE

### B14. Expert-parallel (EP) token routing — the third topology
**Mechanism.** TP2 lost to collectives; PP2 measured −0.26 %. **EP is
neither**: each card owns a disjoint set of experts; a token routes to
whichever card holds its selected experts — per-layer crossing cost is
activations ~10–20 KB (µs-class), *no collectives anywhere*. For a
MoE like flash-next (125B-A6B) or MiMo, expert weight traffic never
crosses the bus at all — each card's GDDR6 streams only its own
resident experts, and the 6-expert-per-layer selection means ~half the
layers need a remote pass (~10 crossings × ~20–50 µs ≈ 0.2–0.5 ms).
The K216 lesson is fully respected: the router→dispatch must stay
device-resident (async copy + flag-slot sync from B11), never a host
round-trip.
**Novelty.** ADR-0008 closed TP; Cascadia priced PP; **EP was never
priced on this host** — it's exactly the topology copy-only P2P wants:
traffic scales with activations (tiny), not weights (fatal).
**Magnitude.** For a 2-card MoE serving, EP is the difference between
"capacity placement" (flash-next's 21.6 tok/s CPU-expert hybrid) and a
both-GPU-resident expert compute — plausibly the *defining* architecture
for >64 GB MoE models on dual Bxx. For dense qwen3.8 it does not apply.
**Kill-probe.** Static first: does vLLM-XPU/llama.cpp SYCL have any EP
dispatch primitive to anchor (deepep/xpu-ep)? If nothing exists, the
build cost is engine-program-class; the arithmetic still stands as the
placement verdict for the NEXT MoE adoption (MiMo is storage-blocked —
price EP into its plan before benchmarking placement).
**Bxx transfer.** THE dual-Bxx MoE answer; also the strongest argument
for remote atomics or a flag-slot primitive in future silicon.

### B15. GDN constant-state serving exploits
**Mechanism.** 48/64 layers are GDN: per-request state ≈150 MB
*constant*, independent of context. Only the 16 full-attn layers grow
KV (32 KiB/token fp8). Three derivatives:
(a) **Checkpoint/resume at constant cost** — a request's entire
    attention-history state is 150 MB + KV; pausing/migrating a request
    is a fixed-size copy, not a KV replay — enables preemption-class
    scheduling other models can't afford (external E6's 1-state-block
    variant measured +53 % KV tokens).
(b) **State tiering**: cold requests' GDN state to GPU1/host at
    ~150 MB — frees the scarce VRAM term at high B.
(c) **The natural experiment**: GDN is already a "lossy fixed-size
    summary" the model trained end-to-end — i.e. the architecture
    answers "what's the load-bearing fraction of context" for free. A
    principled KV-eviction policy for the 16 full-attn layers (evict to
    nothing once GDN state covers it) is the research-grade version of
    sparse attention and costs no accuracy the model didn't already
    pay.
**Magnitude.** (a)/(b) are capacity/scheduling levers at B≥8;
(c) is a research hypothesis — honest EV moderate, novelty high.
**Kill-probe.** (a) zero-GPU: enumerate vLLM-XPU's GDN state handling
(does it even checkpoint?); (c) a recall-task eval with windowed
full-attn KV at varying depths.
**Bxx transfer.** Hybrid linear-attention models are the fleet's future
(a3b/qwen3_5/qwen4exp all hybrid); this is model-shape-native.

### B16. Full-attn KV read-cap at depth (QSA-class, qwen3_5 side)
**Mechanism.** At ctx=N, the 16 full-attn layers read ≈32 KiB × N of
fp8 KV per decode step: ~3.2 GB at 100 K, ~8.4 GB at 262 K — vs ~16.7
GB of weights+head. Past ~100 K, KV reads are co-dominant and the
262 K-native EXL3 route pays it on every token. A top-2 048-row
gather (#28796-class) caps it at ~66 MiB/step ⇒ at 262 K saves ~8.3 GB
≈ 13.7 ms/step — potentially the largest long-context lever that
exists for the *adopted* route.
**Novelty.** #28796 is sanctioned-open for flash-next/llama.cpp —
different stack. For the vLLM-XPU qwen3_5 GDN path the mechanism is
unproven (QSA is a qwen4exp attention variant; qwen3_5 needs its own
sparse dispatch) and the external +1.6 % @7.1 K cell honestly bounds
it: this is a ≥100 K lever only.
**Kill-probe.** One decode-rate-vs-context-depth curve on EXL3
(p512+g128 at ~4 K/32 K/100 K/200 K occupied): if decode degrades
super-linearly or tracks the 32 KiB×N model, the KV term is confirmed
and sparse-gather is worth a kernel; if decode is flat, the KV term is
already masked and the lever closes.
**Bxx transfer.** Any hybrid model at depth on any Bxx.

---

# DELIVERABLE C — TOP-3 EXECUTION ORDER + FULL PROBE DESIGNS

Predecessor's order kept, with reasons: all three are
cheapest-first/attribution-first and each one's result reprices the
rest of the board. B1 (accept-length mining) is folded into probe 3's
zero-GPU phase where it shares the same logged evidence.

## Execution order

1. **Metric-correct EXL3 attribution** (probe 1) — fixes the adopted
   route's *denominator* (real tok/s) AND names its real bottleneck in
   one campaign. Everything else on the EXL3 route (B6–B9, B16) is
   unpriced until this lands. It is also the only candidate whose
   outcome could *immediately* change the fleet headline — if chunks
   carry ~2.6 tokens, the "24.6 tok/s" route is already ~65 tok/s.
2. **30-defect kernel-vs-weights discriminator** (probe 2) — binary,
   cheap, high-upside: kernel ⇒ the 72–80 tok/s route class resurrects
   quality-correct; weights ⇒ margin-aware requant becomes the named
   fix and T3's rotation math gets a target.
3. **Certified own-corpus draft vocabulary + accept-length forensics**
   (probe 3) — RL-077 failed only its thermal gate (mechanism
   unrefuted); the own-corpus 54,274-row list is staged. Same window
   doubles as the B1 evidence run (per-position survival logging).

## Probe 1 — EXL3 true rate + step attribution

**Question.** What does the adopted EXL3 route actually emit per second
(counted quantity: *tokens*), and where does one decode round spend
its time?

**Metric definitions (named counted quantities).**
- `completion_tokens` from `/v1/chat/completions` usage (non-streaming)
  ÷ (t_end − t_first-token): **token-rate**.
- SSE `gen_chunks` retained but demoted to a diagnostic field.
- Server-side `vllm:spec_decode_num_accepted_tokens_total` /
  `num_draft_tokens_total` deltas bracketing each cell → real
  tokens/round and rounds/s independently of client streaming.
- Step composition via unitrace (launcher mode — LD_PRELOAD breaks
  engine init) or, failing that, in-container torch.profiler on the
  engine process: buckets = trellis reconstruction/dequant, GEMM/GEMV,
  attention/KV, sampler, host/graph residual, idle gaps.

**Cells.** EXL3 4bpw + MTP3, fp8 KV, one card (GPU0), 150 W and 230 W
arms: p512/g128 and p8192/g128, n=5 per cell after one discarded
same-shape warmup. Prompts: entropy-first unique ≥64-char prefix,
tokenized to ±96 of target.

**Gates (pre-declared).**
- Token-rate metric must reconcile with counter-derived rate within
  ±5 %, else the harness is still broken — stop and fix.
- Usage `completion_tokens` must equal g128 (or `finish_reason`
  recorded and the cell re-scoped to tokens-emitted).
- If ≥2 cells show gen_chunks ≪ completion_tokens, the old campaign's
  sustained numbers are formally re-derived before any new claim.
- Zero prefix-cache-hit delta; spec counters bracketed; per-block temp
  logged; matched-arm thermal floors disclosed.

**Stop rules.** Attribution unresolved after 60 min of instrumented
capture ⇒ record "attribution unresolved," not a bottleneck guess.
Any thermal/driver fault ⇒ stop, restore 150 W.

**Success output.** (a) corrected `best_tok_s` basis for
`registry/model-status.json`; (b) a named bottleneck bucket
(reconstruction / GEMM / host / clock) with its share; (c) a go/no-go
for B6 kernel work and B16's depth-curve annex.

## Probe 2 — 30-defect discriminator (kernel vs weights)

**Question.** On the failing prompt family, is the "30" produced by the
serving kernel or by the quantized weights?

**Design.** Offline, no serving stack needed for the discriminator
itself:
1. Run the failing prompt through the BF16 base on CPU/host (or the
   existing bf16 TP2 setup if still cheap) and capture hidden states at
   the layer where the defective decision is computed — localize the
   first divergent layer by comparing bf16 vs INT4-export hidden-state
   deltas layer by layer on the same prompt.
2. At the first divergent layer: compute `h @ W_int4` in fp32 offline
   (the exact same packed weights) and compare against the kernel's
   recorded output for the same input.
   - mismatch ⇒ **kernel defect** (accumulation dtype is the live
     suspect — fleet precedent: icpx fp32→fp16 overflow corruption);
     scope = kernel fix.
   - match ⇒ **weights defect** ⇒ the margin-aware requant program
     (B3) gets the named target layer(s).
3. Counted quantities: argmax agreement, per-layer cosine/RMS deltas,
   and the exact logit margin on the two candidate tokens at the
   branch point — NOT "does the answer look right."

**Gates.** Exact same prompt tokens/template as the failing canary;
fp32 reference matmul is the oracle; result must be deterministic
across two independent computations. Zero-GPU for steps 1–2 if hidden
states can be captured in-process; one bounded capture run otherwise.

**Stop rules.** If the divergence is not localizable to ≤3 layers in
the capture budget, record partial localization and stop — do not
expand into a full-layer sweep program.

**Success output.** A one-word verdict (kernel|weights) + the layer
list + the margin evidence. Kernel ⇒ a real upstream bug report with
repro; weights ⇒ B3's requant spec with measured margins.

## Probe 3 — Own-corpus draft shortlist + per-position forensics

**Question A (RL-077 retry under valid thermals).** Does the
54,274-row own-corpus draft-head shortlist (RL-076 staged) beat the
full draft head on the champion stack — measured as token-rate and
counter-α in matched arms?

**Question B (B1 forensics, same window).** Log per-position
acceptance + draft pos-2 logit margin per round; fit the
"shallow-round" predictor offline.

**Cells.** p512/g128 + p8192/g128, ABBA/BAAB n=3 screen → fresh n=5
counterbalanced finalist only on screen pass (pre-declared ≥+5 %
one-cell gate, ≥−3 % other cell, Δα ≥ −0.03).

**Metric.** Client post-first **token** rate (usage-verified) +
counter-α + per-position survival table (4 counters per window) —
three named quantities, never merged.

**Guards.** Valid thermal floor (arm-mean Δ ≤3 °C — the RL-077 failure
was 13.5 °C, disqualifying); greedy byte-identity for losslessness;
spec-canary per boot; cache off; unique entropy prefixes.

**Stop rules.** Screen fails the pre-declared gate ⇒ no finalist, list
retired (matches blocked_pattern). If the forensics show pos-3
acceptance not predictable from logged draft confidence (AUC <0.7),
B1 closes at zero additional cost.

## Full-list EV ranking (per GPU-hour)

| Rank | Candidate | Expected value | Cost | Why |
|---:|---|---|---|---|
| 1 | Probe 1: EXL3 rate+attribution | possibly changes the fleet headline + prices B6–B9/B16 | 1 window | denominator fix + first-ever adopted-route attribution |
| 2 | Probe 2: 30-defect discriminator | resurrects 3×-faster route class or names the fix | ~zero GPU (capture) + 1 run | binary, cheap, huge |
| 3 | Probe 3: shortlist + forensics | +5–10 % if promoted; B1 evidence free in same window | 1 window | staged artifact, thermal fix is the only prior blocker |
| 4 | B16 KV depth curve | confirms/denies the largest long-ctx lever for EXL3 | 1 server, 4 cells | annex to probe 1 |
| 5 | B10 meta-speculation | +4–8 % novel topology | stats first (free), then prototype | needs probe 1's round timing |
| 6 | B11 flag-slot collectives | unblocks TP2 repricing + B10 sync | microbench (small) | primitive, not a serving claim |
| 7 | B1 early-exit drafting | +4–6 % | needs probe 3 forensics | content-predictor dependent |
| 8 | B2 fork-at-collapse | +4–9 % | engine work | oracle bound first (free) |
| 9 | B8 M=5 kernel | ≤+10 % ceiling | kernel | only if probe 1 shows a real gap |
| 10 | B14 EP for MoE | placement verdict for next MoE | static + future program | no engine support today |
| 11 | B15 GDN state exploits | capacity/scheduling | audit first | high-B serving only |
| 12 | B3 margin-requant | quality route | conversion program | after probe 2 verdict |
| 13 | B5 DPAS-codesign | research | long | after B6 |
| 14 | B12 host residuals | +1–3 % | folded into probe 1 | incremental |
| 15 | B7 megakernel / B9 s4×s4 / B4 MIPS / B13 | recorded bounds | — | priced-negative unless an attribution reopens them |

---

# DELIVERABLE D — CORRECTIONS REGISTER

Wrong text as recorded → corrected statement → evidence → file to fix.
(Operator applies; this agent does not edit authorities.)

| # | Recorded (wrong) | Corrected | Evidence | File |
|---|---|---|---|---|
| 1 | "best_tok_s: 24.6" for EXL3 (sustained) | Harness counts SSE content chunks −1 per elapsed — a chunk rate, not tok/s; cells show 4–8 chunks per requested g128 and no completion counts; RL-086/087 inherit it | `results/2026/09/autoround-vs-exl3-20260926/measure.py` (ngen=len(stamps), `gen_chunks`); `exl3-cappair.jsonl` | `registry/model-status.json`; `results/2026/09/autoround-vs-exl3-20260926/RESULTS.md`; `engines/exl3xpu/README.md` (label rows as chunk-rate until probe 1) |
| 2 | "87.7 % GEMM … no hidden waste; single-card decode record closed at three levels" | The trace arithmetic is internally inconsistent: 58.35–58.59 ms round, 553 GEMM events totaling 35.2 ms (~63.6 µs each), GPU-union ~40.2 ms ⇒ ~18 ms unaccounted; with 30–40 % tracer overhead this is NOT recoverable-slack evidence; "87.7 %" is active-kernel share, not wall share | `results/2026/09/b70-improvement-probes-20260920/p4-unitrace-decode-20260925T222227Z/` raw `python3.220.json` (recomputed) | `docs/plans/2026-09-20-b70-improvement-roadmap/OUTCOMES.md` (W20/F3), `LEVERS.md` (F3), `NIGHT-REPORT2-20260926.md` |
| 3 | D9 "acceptance +8.3…8.8 % per +0.05 α" | Counter-form yield 1+4α ⇒ +6.25 % per +0.05 α at α=0.55; conditional per-position form ⇒ +9.25 %; the recorded figure matches neither | `d11-baseline-decay-probe-20260924T142540Z/decay_curve.json` (marginal survivals, not geometric) | `LEVERS.md` D9 + the §4.3 acceptance-frontier preamble; `SPEC-ECONOMICS-20260923.md` |
| 4 | Draft byte budget "3.39 GB/step" and "streams at pin rate" | Head 0.6555648 + body 0.21897216 = 0.87453696 GB/pass ⇒ **3.49814784 GB/round incl. fp16 g128 scales** (3.39 was payload-only); 3.498/5.5 ms implies ~636 GB/s ⇒ "exactly pin rate" was model arithmetic, not measured DRAM traffic | P4 tensor shapes + conversion logs | `LEVERS.md` D16/L-B; `NIGHT-REPORT-20260925.md` Phase-2 table + correction block; `qwen38-draft-side-int4-upstream-20260921.md` (≈12.7 GB/7.6 GB figures re-derive) |
| 5 | Sub-4bpw ideal ratio stated as 4/3 = 1.333× | Mixed precision: body-only 4→3 bits = **1.224×** (body 12.230 GB of the 16.68 GB round); must name which axes move | safetensors headers, artifact `/mnt/<nvme-disk>/b70-exl3-qwen38-4bpw-20260926` | `exl3-analysis-and-next-inference-levers-20260924.md` §4.1 (its 1.215× was the GPTQ variant — keep both, labeled) |
| 6 | llama.cpp SYCL graphs "unblocked" for multi-token MoE by #28476/`304665fe7` | That merge added IQ MMVQ dispatch only; local tree still rejects `ids->ne[1] != 1` and multi-token path still does memcpy+wait | `~/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp` ~4794, ~4923–4926 | `AGENTS.md` §16; `LEVERS.md` X4 row |
| 7 | "(Pure 4×4 geometry is unmeasured and irrelevant for LLM serving.)" | s4×s4 was measured 2026-09-21: 33.91 vs 16.96 TOPS = 2.00× (with the recorded issue-limited caveats) | `research/2026/xmx-dpas-rate-classes-20260915.md`; `results/2026/09/b70-improvement-probes-20260920/dpas-rate-probe/` | `docs/reference/megakernel-comprehensive-reference.md` §5 |
| 8 | "B450-F … Gen3 x8 + Gen3 x4, no P2P"; GDDR6 "~512–576 GB/s" | Host is X570-F (since 2026-08-27), CPU-attached x8/x8, P2P copy verified 11.1–11.9 GB/s (atomics absent); B70 GDDR6 class is ~608 GB/s (256-bit × 19 Gbps); the 512–576 figure is B580-class | `AGENTS.md` §3; `research/2026/x570f-dual-b70-pcie-topology-20260830.md`; RL-059 | `research/2026/megakernel-xe2-xmx-architecture-guide-20260829.md` §2 table + §6 Pillar VI |
| 9 | §16 "no P2P (both GPUs on CPU-attached x8 slots…)" | Contradicts §3/§18 of the same file: P2P copy verified; only ATOMICS absent | RL-059 | `AGENTS.md` §16 (one clause) |
| 10 | flash-next "best_tok_s": 21.9 | Median of the n=3 first-valid screen is 21.6 (21.3–21.9 range); 21.9 is max observed, mislabeled | `results/2026/09/flashnext-c1-screen-20260925T215251Z/` | `registry/model-status.json` |
| 11 | EXL3 "255 GB/s @61.5 ms" and "~240/355 GB/s" | Unit-inconsistent: 15.70 GiB = 16.86 GB ⇒ 274 GB/s @61.5 ms; at the ~40.6 ms 230 W step it would be ~415 GB/s (near champion class, not clearly clock-bound). All effective-bandwidth claims on EXL3 must be recomputed after probe 1 defines the real round | `results/2026/09/autoround-vs-exl3-20260926/RESULTS.md`; `engines/exl3xpu/README.md` | both files (re-derive once chunk-vs-round is resolved) |
| 12 | Per-position acceptance treated as a stable head property | The survival profile is **window/content-dependent and bimodal**: 0.98/0.98/0.12/0.09 on the D11 natural-text window vs ~1.00/0.96/0.88/0.88 on a p4-champion p512 window — not a constant, and the pos-3 conditional (0.127) vs pos-4 (0.72) inversion is a distribution feature, not decay | `d11-baseline-decay-probe-…/decay_curve.json`; `p4-champion-ab-p512-n2-…/v5_baseline/metrics-{before,after}.txt` deltas | `LEVERS.md` acceptance-frontier preamble; `models/qwen/qwen3.8-27b/serving.md` (acceptance-is-content section) |
| 13 | "L-B: re-reads stream at pin rate and cannot be cached" framing implies 4×-read floor measured | The draft bucket timing (5.5 ms) and 3.498 GB traffic are model-consistent, but no DRAM-counter measurement exists; the claim is a model bound that happens to self-agree — label it model, not measured | this pass, correction 4 | same rows as #4 |

---

## Standing caveat

Everything above is PROVISIONAL — NOT FOR PUBLIC HEADLINE. No GPU work,
no authority edits, no commits, no publications occurred in this pass.
All magnitudes are arithmetic bounds against measured floors; probes are
handed to b70-bench-engineer / b70-correctness-engineer /
b70-quantization-engineer as-is.

---

# ADVERSARIAL-REVIEW OUTCOME (2026-09-29, adopted by the orchestrator)

Full review: [`b70-breakthrough-review-20260929.md`](b70-breakthrough-review-20260929.md) (b70-architect-2). Board changes adopted:

- **Probe 1 amended mid-flight** (relayed 2026-09-29): force g128-length generations (min_tokens/ignore_eos) + finish_reason per request + cache-hit deltas per cell; attribution by unitrace-under-capture ONLY (the torch.profiler fallback violated rule 1). Driver: all 13 cappair cells are 3–7-token early-EOS bursts (windows 0.125–0.423 s) and the 230 W p8192 TTFTs are cache hits (register row 14) — the old campaign never measured a full-length sustained rate.
- **B1 demoted, re-gated at precision ≥0.96** (break-even 0.887; at the ≥85 % gate the optimum is flag-nothing; the only serving-shaped survival window on record is FLAT ~1.00/0.96/0.90/0.90 — the bimodal profile came from a natural-text eval probe). Pending probe-3 forensics on the serving window.
- **B2 SOUND as a bound**: recomputed net +5.9…+9.5 % (q2 0.30–0.37); contingencies (flat profile, XPU tree-mask kernel) must be stated before any build.
- **B10 demoted to ~+0.6 % net**: MTP drafts condition on target hidden state, so hedge-hit rounds pay an α penalty (−0.10 class) that collapses the +7.4 % gross; the 3-chain window also fails at 150 W-class bandwidth.
- **B11 re-based**: against the quality-dead GPTQ 44.8 ms round the 1.5–1.6× was mispriced; against live EXL3 at matched power it is ~1.33× @230 W / ~1.49× @150 W, and the 2–3 ms collectives rest on the unbuilt flag-slot primitive.
- **B16 rescoped**: at the deployed `max_model_len=65536` the KV term is ~2.15 GB/step ≈ a 12.9 % addendum — a real but modest lever; 262 K is capability, not deployment.
- **§A structure fixed in AGENTS.md §20**: per-route spec-stack lock (MTP4-GPTQ vs MTP3-exl3xpu) and the prefix-cache rule rescoped (EXL3's measured 5.4× cache win stands; cache×MTP is a shared-risk axis with a one-check rule).
- **New candidates adopted**: **N1** GDN-state bandwidth knee fit (explains the ALREADY-MEASURED S2/O8 knee — commissioning against `conc-B*.jsonl` now), **N2** EXL3 prefill attribution (37–60 % of BF16 peak claimed by the fork's int8-XMX path), **N3** split draft/verify head on EXL3 (~0.95 GB/round ⇒ +4–6 %).
- **N1 CLOSED — REFUTED (2026-09-29 N1 addendum, same file below)**: the state arithmetic holds (152.6 MB/req fp32 state → constant 0.50–0.68 ms/request marginal) but the data contradicts it where its signature must appear — measured marginal is 10–14× too large at low B and exactly zero at high B (B32 rate 0.3 % ABOVE B16 same-boot, where the state model requires ~8 % below); low-B rise is mostly prefill-window pollution (earliest-quartile 43.5 vs latest-quartile 54.8 tok/s at B=8), clean residual ≈2.2 ms/req attributable to graph-bucket padding/scheduler. **Fallout (2 register rows applied to W10/S2)**: the recorded knee aggregates are span-diluted — steady-state implied ~**1,050 tok/s** at B=32; and S2's evidence citation corrected to `…T151528Z`. Successor open question: why clean T_step doubles B=1→32 (≈40→~89 ms) under full weight amortization — decider = staggered-arrival B-sweep with per-level counters (queued for a GPU window).
- **Probe 2A note**: no loader wall exists (transformers 5.15.0 has native qwen3_5); verdict-scoping caveats relayed (a CPU-"14" localizes to serving-path undifferentiated; fp16-arm is CPU emulation, not XMX measurement).
