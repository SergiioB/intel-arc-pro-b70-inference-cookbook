---
type: research-review
date: 2026-09-29
status: provisional
reviewer: b70-adversarial-architect (independent second opinion, CPU-only)
object: research/2026/b70-breakthrough-architect-20260928.md
publication_gate: PROVISIONAL_NOT_FOR_PUBLIC_HEADLINE
---

# Adversarial review — b70-breakthrough-architect-20260928

Independent second-opinion pass, CPU-only (reads, grep, python in /tmp; script
`/tmp/advreview_arith.py`). No GPU work, no server launches, no authority
edits. Probe 1 and probe 2A were treated as live-but-untouchable; their
designs were reviewed, not run.

## Method and sources

Object read in full (679 lines). Cross-checked against: AGENTS.md §3/§16/§17/§20
(actual file), `docs/plans/2026-09-20-b70-improvement-roadmap/{OUTCOMES.md,LEVERS.md}`,
`engines/exl3xpu/README.md`, `results/2026/09/autoround-vs-exl3-20260926/{RESULTS.md,measure.py,exl3-cappair.jsonl}`,
`results/2026/09/b70-novelty-20260928/RESULTS.md`,
`results/2026/09/b70-improvement-probes-20260920/d11-baseline-decay-probe-20260924T142540Z/decay_curve.json`,
p4-champion `v5_baseline/metrics-{before,after}.txt`, the running probe at
`results/2026/09/b70-probe2-30defect-20260929/`, and the EXL3 artifact's
config + safetensors headers at
`/mnt/<nvme-disk>/b70-exl3-qwen38-4bpw-20260926` (read-only).
Every derived number recomputed.

## Verdict summary

| # | Attack surface | Verdict | One-line reason |
|---|---|---|---|
| 1 | EXL3 byte budget 16.68 GB / 27.4 ms | **CONFIRMED number, NEEDS-FIX prose** | Number reproduces exactly (12.230 + 4×0.954 + 3×0.213 = 16.683 GB) but the stated decomposition sums to 15.94 and hides the load-bearing shared-head ×4 inference |
| 2 | Probe 1 H1/H2 framing | **NEEDS-FIX** | A third live hypothesis is confirmed by arithmetic: all 13 recorded cells are 0.13–0.42 s bursts that cannot contain g128 — cells must force full-length generations or probe 1 inherits the flaw |
| 3 | B10 meta-speculation | **NEEDS-FIX** | Gross, not net: the MTP drafter conditions on the target hidden state (`mtp.pre_fc_norm_hidden` tensor); a hidden-state-free pre-draft at α −0.10 collapses +7.4 % to +0.6 % |
| 4 | B11 TP2 repricing 1.5–1.6× | **NEEDS-FIX** | Priced against the quality-dead GPTQ 44.8 ms base; against the live EXL3 route at matched power it is ~1.33× @230 W; the 15 µs flag-slot is spec-tier |
| 5 | B1 early-exit +4–6 % | **NEEDS-FIX (near KILL as premised)** | Break-even precision is 0.887 and +4 % needs ≥0.96 — the doc's ≥0.85 gate yields exactly 0 % net; and the only serving-window profile on record is FLAT, inverting the premise |
| 6 | B2 fork drafting +4–9 % | **SOUND as bound** | Arithmetic reproduces (q2=0.30–0.37 → +9.2…+12.9 % tokens → net +5.9…+9.5 %); contingency on the bimodal profile and on an XPU tree-mask kernel must be stated |
| 7 | B16 KV-read model | **SOUND mechanism, NEEDS-FIX scope** | 2 KiB/token/layer confirmed; deployed max_model_len=65536 → 2.15 GB/step (12.9 % addendum), not co-dominant; 262 K is capability, not the deployed config |
| 8 | Probe 2A loader risk (URGENT) | **SOUND — wall already dodged** | transformers 5.15.0 has native qwen3_5; the running probe found no compressed-tensors/auto-gptq and built validated manual dequant; BF16 CPU control already answers 14 |
| 9 | Register rows 1/3/5/11 | **CONFIRMED (all four)** | Each independently reproduced from raw source; row 1 should be strengthened (burst, not sustained) and a new row 14 is owed (cache-hit TTFTs) |
| 10 | §A standing structure / cache scope | **NEEDS-FIX (2 clauses)** | A.4 cache rule under-scopes a shared-machinery risk and over-scopes EXL3; §20 rule 2 locks "MTP4" but the deployed primary route is EXL3 MTP3 |

---

## Findings per attack surface

### 1. The EXL3 byte budget (§Ground truth) — CONFIRMED number, NEEDS-FIX decomposition

Recomputed directly from the artifact's safetensors headers (index metadata
`total_size` 16,860,508,228 B = 15.70 GiB ✓):

| component | bytes | note |
|---|---:|---|
| language-layer tensors | 12.230 GB | 2048 tensors (`.trellis` + scales) |
| `lm_head.trellis` | 0.9535 GB | 6-bit (`head_bits: 6`, shape [320,15520,96] I16) |
| `mtp.*` module | 0.213 GB | 39 tensors — **contains no head of its own** |
| `embed_tokens` | 2.543 GB | BF16 [248320, 5120] |
| `model.visual.*` | 0.921 GB | vision tower, not read in text decode |

One-round one-read model that reproduces the doc's number exactly:

```
12.230 (body) + 4 × 0.9535 (head: 1 verify + 3 draft passes) + 3 × 0.213
(mtp body × 3 draft passes) = 16.683 GB  →  27.44 ms @ 608 GB/s   ✓ 16.68/27.4
```

- **The embedding attack dissolves — for the number.** 16.68 already excludes
  the 2.543 GB embedding; decode gathers only ~5–7 rows (~60 KB) per round.
  But the doc's component list ("+ embedding 2.543 conditional") is
  arithmetically incoherent as written: its stated parts sum to 15.94, and
  neither "×4-ish reads accounted" nor "MTP 0.213" (needed ×3) carries the
  algebra. The number is right; the shown work is wrong.
- **The load-bearing unstated inference:** head×4 requires the draft to reuse
  the target `lm_head` on every draft pass. I verified this structurally —
  the mtp group has 39 tensors and no head, so the MTP module must project
  through the shared lm_head (EAGLE-family convention; `mtp.pre_fc_norm_hidden`
  confirms hidden-state input). This is inference from tensor inventory, not
  measurement; probe 1's counters should confirm 3 draft head passes/round.
- Derived claims: 271 GB/s effective @61.5 ms and 411 @40.6 ms on the
  round-traffic basis (register row 11's 274/415 uses the artifact-total basis
  — both defensible, but the basis should be named). Fraction of floor:
  44.6 % / 67.6 % — the doc's "42–68 %" band is right. 96.6 tok/s at yield
  2.65 ✓ ("97").
- The 608 GB/s pin-rate floor uses the correct ceiling for a floor
  (256-bit × 19 Gbps).

**Verdict: CONFIRMED (16.68 GB / 27.4 ms / 42–68 %), NEEDS-FIX prose** —
restate as `body + 4×head + 3×MTP`, embedding explicitly excluded as a
row-gather, and mark shared-head as an inference probe 1 must confirm.

### 2. Probe 1's H1/H2 framing — NEEDS-FIX (a third hypothesis is live and confirmed by arithmetic)

`measure.py` verified: `ngen = len(stamps)` counts SSE content deltas
(line 148) and `post_first_rate = (ngen-1)/(t_end - t_first)` (line 154) —
register row 1 is exactly right, and the jsonl carries no
usage/finish_reason field at all.

**New finding (the third hypothesis, with teeth):** recomputing every cell's
post-first window from the jsonl:

```
(gen_chunks − 1) / post_first_rate  →  0.125–0.423 s across all 13 cells
```

Emitting 128 tokens in those windows would require **303–1025 tok/s** —
physically impossible (byte floor ~97 tok/s at pin rate). Therefore the
recorded generations **ended early** (EOS/stop) after ~4–8 chunks. The
inter-chunk gaps do track the power cap (56–61 ms @150 W → 40–43 ms @230 W),
so chunks arrive at *round* cadence — i.e. **H3: chunk = engine step AND
completion ≪ g128**. Consequences:

- The "sustained ~17.8/24.6" EXL3 numbers (RL-086 n=3) are 3–7-gap **burst
  cadences**, not sustained anything. Register row 1 understates the damage
  and should be strengthened.
- Probe 1's gates (usage == g128; counter reconciliation; finish_reason
  re-scoping) *will* catch this — the design is sound — but its cells reuse
  the same entropy-prefix synthetic log-continuation prompt class, which is
  exactly the class that early-stops. **Fix to relay: force full-length
  generations (`min_tokens` = g128 or `ignore_eos`, or a prompt class
  guaranteed to emit ≥128 tokens), and record finish_reason per cell.**
- Second design inconsistency: the "in-container torch.profiler" fallback
  contradicts the architect's own A.1/§20 rule 1 (eager traces are blind to
  graph-replayed kernels, RL-072). Attribution must be unitrace-launcher
  under capture; any torch.profiler run must be labeled graphs-off
  diagnostic and cannot support the B6 go/no-go.
- Third: per-cell prefix-cache-hit deltas are already gated — good, because
  the predecessor's 230 W p8192 TTFTs are cache-hit contaminated (surface 9,
  new row 14 below).

### 3. B10 cross-card meta-speculation — NEEDS-FIX (gross priced as net)

- **Window arithmetic:** an EXL3-MTP3 hedge chain reads 3 × (0.213 + 0.9535)
  = 3.50 GB (mtp body + shared head per pass — note the doc's 5.5 ms
  champion-draft figure is the GPTQ MTP4 stack; the EXL3 chain is this
  route's own traffic). At GPU1 effective bandwidths actually observed on
  this engine: 17.3 ms @608 (pin), 25.3 ms @415, **38.7 ms @271 — does not
  fit the ~30 ms verify window at 150 W-class effective rates**. "3 chains
  ≈ 16.5 ms < 30 ms" silently assumes champion-class (~636 GB/s model)
  efficiency on the very stack whose efficiency is pending probe 1.
- **P(hit) definition:** "bonus ∈ top-2 of WHAT" is undefined. The only
  distribution available to GPU1 before verify completes is *its own draft
  distribution* at the bonus position. Joint hit ≈ P(accept-length modal)
  × P(target argmax ∈ GPU1-draft top-2) ≈ 0.876 × 0.7–0.8 ≈ 0.61–0.70 — the
  doc's 0.5–0.7 is plausible **as a token-guess probability**.
- **The missing term (the failure mode the author did not price):** the MTP
  drafter conditions on the **target hidden state** — the artifact carries
  `mtp.pre_fc_norm_hidden` for exactly this input. A pre-draft conditioned
  only on a hypothesized token is a *different, weaker drafter*. Pricing it:
  if α on hit-rounds drops 0.10, the round-level net goes from +7.4 % (gross,
  hit 0.6) to **+0.6 %**; at α −0.20 it is **−6.7 %**. The +4–8 % claim is
  gross and must be re-labeled; the kill-probe must add a zero-GPU/CPU
  measurement of embedding-only chain acceptance (feasible with the probe-2A
  machinery: run the MTP module from embedding-only inputs on recorded
  rounds). Miss cost is genuinely ~zero (GPU0 drafts anyway; GPU1 KV
  rollback is trivial) — that part of the doc is right.

### 4. B11 TP2 repricing — NEEDS-FIX (dead base, spec-tier primitive)

- **Base:** "~28 ms vs 44.8 ⇒ 1.5–1.6×" prices TP2 against the *removed,
  quality-dead GPTQ champion's* round. Whose round does TP2 halve? On the
  adopted EXL3 route at matched power: ~1.49× @150 W (41.3 vs 61.5 ms),
  **~1.33× @230 W (30.6 vs 40.6 ms)** — still inside the +20–50 % gate band,
  but the board should carry the live-base number, pending probe 1.
- **Collectives 2–3 ms:** 128 × 51.2 KB allreduce; wire time 4.5 µs at the
  verified 11.5 GB/s ⇒ ~1.9 ms at 15 µs/collective — arithmetically
  consistent **iff** the flag-slot primitive sustains ~15 µs. That primitive
  is unmeasured (RL-071's host-staged xccl for the identical pattern: 14.3 ms
  — a 5–10× claim resting on an unbuilt spin-copy protocol, with graph-capture
  safety unproven, both admitted). Label spec-tier until its own microbench.
- Champion body at effective rate: 15.2 GB in 28–30 ms = 507–543 GB/s ⇒
  halved body ≈ 14–15 ms (the doc's "≈13 ms" is pin-rate, 1–2 ms generous).
- Unmodeled favorable term worth noting: 4 kv heads → 2/card, per-card KV
  read halves. Unmodeled cost: the 415 GB/s arm needs *both* cards at 230 W.
- ADR-0008's collective-correctness gate still applies — the doc says so;
  keep it.

### 5. B1 predicted shallow-round early exit — NEEDS-FIX, near KILL as premised

Recomputed with the D11 profile (marginal survivals 0.9802/0.9752/0.1238/0.0891,
yield 3.1683, verified from `decay_curve.json`), T=44.8 ms, draft bucket
5.5 ms/4 uniform byte-bound passes, halt-at-2 saves 2.75 ms, deep-round loss
= 1 + P(pos4|pos3)=0.72 → 1.72 tokens:

- **Break-even plane:** marginal break-even precision
  `p* = 1 − (R0·save)/loss = 1 − (0.07072×2.75)/1.720 = 0.887`, independent
  of flag rate. At the doc's **≥85 % precision gate the optimal action is
  f=0 — exactly 0 % net.** Honest gains (flag rate capped at
  shallow-share/precision): perfect predictor **+5.7 %**; p=0.97 → +4.3 %;
  p=0.95 → +3.4 %. **"+4–6 %" requires precision ≥ ~0.96, not 0.85**, and
  the kill-probe threshold (AUC 0.7) is far below the operating point.
- **The premise inverts under the serving evidence.** The doc's own
  correction 12 says the profile is window/content-dependent. I recomputed
  the second window from the p4-champion counter deltas (52 drafts:
  52/50/47/47 accepted per position): **1.000/0.962/0.904/0.904 — flat**
  (the doc's "0.88/0.88" is slightly off; substance right). Critically, the
  flat window is the one that brackets actual g128 *serving* cells; the
  bimodal window (D11) is a natural-text eval probe. On a flat profile only
  ~9.6 % of rounds are shallow: a *perfect* predictor gains ~+0.6 %. B1's
  "~88 % shallow" is drawn from the non-serving window.
- Pass-cost uniformity is fine (each pass reads the same 0.875 GB;
  byte-bound). Graph capture needs two pre-captured variants (k=2/k=4) —
  implementable, say so.
- Evidence caveat both windows inherit: the p4 window's implied yield
  (248 tokens / 52 rounds = 4.77/round) is inconsistent with 73.4 tok/s at
  44.8 ms (3.29) — the bracket includes non-cell traffic; use shape, not
  absolute yield, from that file.

**Verdict: NEEDS-FIX.** Re-gate at precision ≥0.96, and demote EV until
probe-3 forensics shows the *serving* window is bimodal. If probe 3 shows
the serving profile flat, B1 closes at ~zero and B2 collapses with it.

### 6. B2 fork drafting at the collapse position — SOUND as a bound

Recomputed (same profile; fork lifts conditional pos-3 acceptance from
0.127 to q2; pos-4 conditional 0.72 preserved; +1 draft pass = +3.07 % round):

| q2 | yield | tokens/round | net |
|---|---|---|---|
| 0.30 | 3.459 | +9.2 % | +5.9 % |
| 0.37 | 3.576 | +12.9 % | +9.5 % |
| 0.50 | 3.794 | +19.7 % | +16.2 % |

The doc's "+7–12 % tokens → net +4–9 %" corresponds to q2 ≈ 0.30–0.37 —
**conservative** against its own stated 0.3–0.5 lift. Contingencies to state:
(a) same profile dependence as B1 — on the flat serving window pos-3 already
accepts 0.904 and the fork buys ~nothing; (b) the +1-pass cost assumes
branch-serial drafting (5 passes vs 4) — right; (c) verify M=5→M=7 is
byte-free for weights but requires a **tree-mask verify kernel that does not
exist on XPU** (admitted) — the build is a kernel, not proposer-only.
Kill-probe (oracle on logged draft top-2 vs target argmax) is correct and
zero-GPU.

### 7. B16 full-attn KV read-cap — SOUND mechanism, NEEDS-FIX scope

- Config arithmetic **confirmed from the artifact config**: 2 × 4 kv-heads ×
  256 dim × 1 B = 2 KiB/token/layer; × 16 full-attn layers = 32 KiB/token.
  262,144 ctx → 8.59 GB/step (doc's "8.4" is 2 % low); 100 K → 3.28 GB.
- **Deployed reality:** the EXL3 route serves `--set vllm.max_model_len=65536`
  (engines/exl3xpu/README.md launch line) → 2.15 GB/step = a **12.9 %
  addendum** to the 16.68 GB round — not co-dominant. 262 K is a *proven
  capability* (ctx_max_probe), not the deployed config. B16's own "≥100 K
  lever only" honesty mostly covers this, but "the 262 K-native EXL3 route
  pays it on every token" misleads about the deployed stack. Include the
  64 K point in the depth-curve annex.
- The 66 MiB/step gather cap assumes zero-overhead row gather and a sparse
  dispatch that does not exist for qwen3_5 on XPU (admitted). Kill-probe
  design (decode-rate vs depth) is right and cheap.

### 8. Probe 2A (running) — SOUND; the loader wall was already dodged; two caveats to relay

- **The feared wall does not apply.** Verified live: transformers 5.15.0
  ships a native `models/qwen3_5/modeling_qwen3_5.py` (imported cleanly on
  this host, python 3.14). The running probe discovered that
  `compressed_tensors` / `auto_gptq` / `gptqmodel` are **not installed** and
  accordingly built manual faithful-dequant modules inside the native
  skeleton, with per-tensor dequant validation against the BF16 base
  (`packing-audit.md`: AutoGPTQ +1 zero-point storage resolved to effective
  zp=8; exports measurably better than RTN; `lm_head`/embed/norms/
  `in_proj_a/b` unquantized — the defect cannot be an lm_head-quant
  artifact). The BF16 CPU control already completed and answers **14**
  (`run-bf16.log`) — the native implementation runs on CPU and reproduces
  the correct control. **No fallback spec is needed.**
- **Caveat 1 (verdict semantics):** a "CPU-W4A16 → 14" outcome localizes to
  the *serving path*, not specifically "kernel": the serving canaries ran
  with MTP4 while the CPU forward is dense, so vLLM modeling code and
  spec-decode×quant interaction remain in the residual class. If the verdict
  lands "kernel", add one MTP-off serving cell (or scope the verdict
  "serving-path") before writing an upstream kernel bug report.
- **Caveat 2 (evidence tier):** the fp16-pipeline "accumulation-dtype" arm
  is a CPU emulation — a hypothesis filter about XMX accumulate behavior,
  not a measurement of it. Label inference-tier.
- Environment flakiness noted by the probe itself (Seagate SATA drop) —
  artifact integrity is anchored by the dequant-vs-bf16 gates, which is the
  right design.

### 9. Corrections register rows 1/3/5/11 — all four CONFIRMED; one new row owed

- **Row 1 CONFIRMED and understated.** `measure.py` verified line-by-line
  (chunk counting; no usage/finish fields; jsonl gen_chunks 4–8,
  post_first_rate 17–25, median_step_ms 40.6–60.7 all match). Add: every
  cell is a 0.13–0.42 s burst (surface 2) — "sustained" labels on
  RL-086/087 are burst labels.
- **Row 3 CONFIRMED.** Counter form recomputed: yield 1+4α at α=0.55 →
  +6.25 % per +0.05 α; geometric/relative form 0.05/0.542 = +9.22 ≈ the
  +9.25 % cited; recorded "+8.3–8.8 %" matches neither. LEVERS.md already
  carries the corrected text.
- **Row 5 CONFIRMED.** Headers read directly: body 12.230 GB of the 16.68 GB
  round; body-only 4→3 bpw ideal ratio recomputed = **1.2245×** exactly.
- **Row 11 CONFIRMED.** 15.70 GiB = 16.86 GB artifact total → 274 GB/s @
  61.5 ms / 415 @40.6; my round-traffic basis (16.683 GB) gives 271/411 —
  same conclusion; the basis should be named in the fixed text.
- **New row 14 (proposed):** the cappair **230 W p8192 TTFTs are prefix-cache
  hits**: 0.785 s for ~8,220 tokens = ~10.5 K tok/s — ~5.6× the cold 150 W
  rate in the same file (4.37–6.98 s) and far beyond compute (would be
  ~566 TFLOPS); prompts repeat across arms because `build_prompt` seeds on
  (ARM, idx, target) and the ARM string is shared. The 150↔230 W TTFT
  comparison in that file is void; cold EXL3 prefill on record is
  ~1,177–1,882 tok/s @150 W (37–60 % of the BF16 DPAS class — see N2).

### 10. §A standing structure vs AGENTS.md §20 — NEEDS-FIX (two clauses)

- §20 exists (2026-09-29) and mirrors §A faithfully (rules 1–9 map onto
  A.1; the compressed always-try list maps onto A.2; closed classes onto
  A.3). Two defects:
- **Cache-rule scope (A.1.4 / §20 rule 4).** The corruption evidence is
  W4A16+MTP4+cache-ON *in the exl3xpu fork* (novelty RESULTS: 5/6
  degenerated, zero-hit first requests); EXL3+cache-ON has no observed
  corruption (clean 5.4× cache probe; viz rounds ON) but was never run
  through a matched corruption gate — and the zero-hit-first-request
  signature implicates cache-manager×spec-decode machinery the EXL3 route
  *shares*. The rule as written both over-scopes (a reader could disable
  EXL3's measured 5.4× cache win) and under-scopes (it does not say EXL3 is
  uncleared). Fix: "OFF on the W4A16/GPTQ-family path (reproduced 5/6);
  EXL3 remains ON behind the per-cell zero-hit-delta guard until a matched
  EXL3+MTP3 corruption gate runs."
- **§20 rule 2 locks "MTP4"** — but the deployed primary route is EXL3 with
  MTP3 (`qwen3_5_mtp n=3`, engines/exl3xpu/README.md). The locked stack
  describes the removed GPTQ champion. Re-scope per route. Same family of
  nit: A.1.7's "+0.05 α = +6.25 %" is the MTP4 counter form; on MTP3 it is
  +4.3 % (1+3α).
- A.2 #2's "+39.6/+48.6 %" matches LEVERS D1 ✓; A.3's closed list is
  consistent with AGENTS §16/§18 entries I checked (IQ3 kernel, #29107,
  stock TP2, PP2, depth>4) ✓.

---

## New candidates (not in B1–B16, not on the closed lists)

### N1. GDN recurrent-state bandwidth — the unpriced term behind the concurrency knee
**Mechanism.** Each decode step reads *and writes* every request's GDN state:
48 layers × 3.1 MB fp32 ≈ 150 MB → ~300 MB/request/step of state traffic on
top of amortized weights. At B=32 that is ~9.6 GB/step against ~21 GB of
(amortized) weights — a ~46 % byte addendum landing exactly where the
measured knee flattens (LEVERS S2: B=16 → 162.74, B=32 → 172.48 tok/s — only
+6 % for 2× the requests; O8 "diminishing, not saturated", mechanism never
attributed). B15 prices GDN state as *capacity*; nobody prices it as a
per-step *bandwidth* term.
**Honest bound.** If the fit confirms, bf16 state (halves the term; quality
gate needed) or a state-layout/fused-state kernel buys the B=16→64 aggregate
headroom; ceiling arithmetic ~1.3–1.4× aggregate at B=32. At B=1 the term is
1.4 % — invisible to every C1 floor analysis on the board, which is why it
was missed.
**Kill-probe (zero-GPU).** Fit T(B) on the existing `conc-B*.jsonl`
(B=1/4/8/16/32): the model predicts a per-request marginal of ~0.5 ms/step
(300 MB ÷ 608 GB/s) plus the KV term. Slope matches ⇒ mechanism confirmed;
slope ≈ 0 ⇒ dead.
**Why §B missed it.** The board is C1-centric; the only B>1 datum (S2/O8)
was filed as a throughput footnote without attribution.

### N2. EXL3 prefill attribution — unpriced, and the 230 W TTFT cells are contaminated
**Mechanism.** Cold p8192 prefill on the adopted route measures 1,177–1,882
tok/s @150 W = 63–101 TFLOPS = **37–60 % of the BF16 DPAS class** — never
attributed (probe 1 prices decode only; LEVERS' prefill rows are K216-only).
The 230 W p8192 TTFTs in the same jsonl are cache hits (new row 14), so the
file's prefill story is partly invalid.
**Honest bound.** If composition shows non-DPAS or tile-misconfigured
dispatch, ≤1.6× TTFT-class theoretical; precedent (W4A8 prefill,
closed-by-attribution on the GPTQ stack) says realistically ≤1.3×.
**Kill-probe.** One unitrace window on a cold p8192 prefill, annexed to
probe 1's server at zero extra launch cost; ≥60 % of BF16 peak in GEMM
dispatch ⇒ close.
**Why §B missed it.** §B treats prefill as solved-by-context (the 262 K
probe) and never rooflined the adopted route's own numbers.

### N3. Complete the draft-side pair on EXL3 — split draft/verify head precision
**Mechanism.** The EXL3 round reads the shared 6-bit lm_head **four times**
(3 draft + 1 verify) = 3.81 GB of the 16.68 GB round — 23 % of the floor is
head traffic, and 3 of the 4 reads are *draft-side*. A second 4-bit trellis
copy of the head for the draft passes removes ~0.95 GB/round (5.7 % of
bytes ≈ 3.5 ms at 271 GB/s effective) for +0.64 GB VRAM. The lever class is
the fleet's biggest proven WIN (draft-side INT4 pair +39.6/+48.6 %, LEVERS
D1 — which quantized the *draft head* too); on EXL3 the pair is half-done:
`mtp_bits: 4` covers the body, the head is shared.
**Honest bound.** +4–6 % round time at current effective bandwidths, gated
on Δα ≥ −0.01 (draft-only change; the verify head is untouched — this is not
the closed verify-head-INT4 axis).
**Kill-probe.** Offline first (CPU, rides the probe-2A machinery): trellis-4
bit requant of `lm_head.trellis`, draft-chain agreement on recorded rounds;
then one matched A/B window.
**Why §B missed it.** §B priced the head as a verify-side object (B4 MIPS)
and never noticed the 4× read is draft-dominated — the decomposition in
surface 1 was never written down.

---

## What changes on the breakthrough board (ranked)

1. **Probe 1 amendments (relay now — it gates everything):** force
   full-length generations (min_tokens/ignore_eos or a long-output prompt
   class) + record finish_reason; attribution must be unitrace-launcher
   under graph capture (torch.profiler fallback = graphs-off diagnostic
   only); keep the per-cell cache-hit delta gate (the predecessor's 230 W
   TTFTs are cache-hit contaminated). Re-scope all prior EXL3 rate numbers
   as burst cadences (strengthen register row 1; add row 14).
2. **B1 re-gate:** precision requirement is ≥0.96 (break-even 0.887; the
   ≥0.85 gate yields exactly zero), and the only serving-window profile on
   record is FLAT — B1 (and B2's magnitude) must wait for probe-3 forensics
   on the serving window; demote B1 from rank 7 accordingly.
3. **B10 re-price gross→net:** add the hidden-state-conditioning penalty
   (α −0.10 collapses the EV to ~0) and the window fit at measured effective
   bandwidth (3 chains do not fit at 150 W-class rates); extend its
   zero-GPU kill-probe with the CPU embedding-only-chain acceptance
   measurement.
4. **B11 re-price against the live base:** ~1.33× @230 W EXL3 (not 1.5–1.6×
   vs the dead GPTQ round); flag-slot ~15 µs labeled spec-tier pending its
   microbench; keep behind the ADR-0008 gate.
5. **Byte-budget prose fix (surface 1):** state `12.230 + 4×0.954 + 3×0.213
   = 16.68 GB`, embedding excluded as a row-gather, visual tower excluded;
   have probe 1 confirm the 3 draft head passes.
6. **§A.4/§20 cache-rule re-scope + MTP4→per-route lock fix** (surface 10).
7. **Adopt N1's zero-GPU knee fit** (existing data, explains a measured
   fleet datum) and **N3's offline head-split check** (rides probe-2A's
   machinery; completes the fleet's biggest proven lever class on the
   adopted route).

## The single highest-value next action the first pass did not say

Run **N1's zero-GPU knee fit on the existing `conc-B*.jsonl`** — it is the
only item on the amended board that *explains an already-measured fleet
datum* (the S2/O8 knee) rather than pricing a hypothetical, costs no GPU
time, and decides whether the aggregate-serving ceiling is a wall (GDN state
bandwidth) or headroom (scheduler). It should be scheduled before any B10/B11
prototype, because if the knee is state-boundwidth, the dual-card program's
correct first move changes (B14-style placement and state precision, not
collectives).

---

## Addendum 2026-09-29 (evening) — N1 knee fit executed: REFUTED as the knee explanation

Commissioned follow-up (orchestrator). Zero-GPU fit of the GDN-state
bandwidth-knee model against the existing concurrency evidence. Script:
`/tmp/n1_knee_fit2.py` (+ per-row discriminator inline). Data:
`conc-B{1,2,4,8}.jsonl` in
`p4-champion-ab-…-20260921T143215Z/v5_baseline` (α=0.515 → y=3.06) and
`…T143819Z/v5_draftpair` (α=0.4815 → y=2.926); B=16/32 actually live in
**`…T151528Z/v5_draftpair`** (n=32 rows each; its own counters give
y=2.907) — note S2/LEVERS cites `…T151358Z`, which is a C1 n=1 arm with
**no conc files**: wrong citation.

### (a) Measured step time and marginal cost

T_step(B) = y ÷ per-request post-first rate (per-request rate is the
batched-step observable; fields in the jsonl are sufficient — no missing
field; per-round counters exist only at arm level and are used as the y
anchor):

| arm | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|---|
| baseline T_step (ms) | 54.7 | 63.5 | 76.8 | 82.0 | — | — |
| draftpair T_step (ms) | 39.9 | 47.5 | 60.7 | 66.7 | — | — |
| deepknee T_step (ms) | — | — | — | — | 89.0 | 88.7 |

Marginal ms/request (raw means): B1→2 ≈ 7.7–8.8, B2→4 ≈ 6.6, B4→8 ≈
1.3–1.5, **B16→32 = −0.02 (exactly flat)**. Linear fits T = T0 + kB give
k ≈ 3.7 ms/req but R² ≈ 0.85–0.87 — the data is not linear; the marginal
*collapses*.

### (b) State-model prediction

From the config (48 GDN layers × [48 v-heads × 128 × 128 × 4 B = 3.146 MB
+ 33 KB conv]) = 152.6 MB/request; read+write = **305 MB/req/step** →
**k_state = 0.50 / 0.59 / 0.68 ms/req at 608 / 520 / 450 GB/s** — a
*constant* marginal at every B. KV at this depth (~736 tok) adds only
0.04 ms/req.

### (c) Discrimination

- **Low B:** observed marginal ≈ 7 ms/req (B1→4) = **10–14× the state
  prediction** — state traffic cannot produce the doubling from T=39.9 to
  60.7 ms.
- **High B:** the state model *requires* +8 ms step time from B16→32
  (16 × 0.5 ms) ⇒ per-request rate at B32 ≈ 8 % below B16. Measured: B32
  rate is **0.3 % above** B16 (32.77 vs 32.66; T_step 88.7 vs 89.0). The
  signature is absent precisely where it must appear. This comparison is
  within one boot (T151528Z) and prefill contamination affects both cells
  in the same direction — correcting for it would only sharpen the
  contradiction.
- **Amortization-vs-growth:** the raw curve *saturates* (marginal → 0),
  which is the amortization/bucket-padded shape, not the linear-growth
  shape. Per-row early-vs-late windows show the low-B rise is largely
  **prefill-window contamination**: at B=8, earliest-quartile requests
  decode at 43.5 tok/s while latest-quartile (post-prefill-train, clean
  windows) run at 54.8 — clean-window T_step(B8) ≈ 53 ms vs raw 66.7.
  Clean residual B1→B8 ≈ 2.2 ms/req; B32 clean-window T ≈ 76 ms — still
  1.5–4× the state ceiling, and cross-boot/noisy (the B16 late window
  inverts, 92 ms — not usable).

### (d) Verdict

**N1 is REFUTED as the explanation of the measured knee** — stronger than
"not contradicted": the model's mandatory high-B signature (+0.5 ms/req
forever, −8 % per-request rate at B32 vs B16) is contradicted by the data
(+0.3 %), and the low-B marginal it would need to own is 10–14× oversized.
State traffic survives only as a possible ≤1 ms/req minor term the data
cannot attribute. **N1 closes** (kill-probe executed, killed).

### Residues worth registering (two)

1. **S2/O8's aggregates are span-diluted.** The recorded knee (61.86 →
   162.74 → 172.48 tok/s, "2.79×, diminishing") is total-tokens ÷
   cell-wall-span with static simultaneous launches — spans dominated by
   serial prefill trains (mean TTFT 2.8 s at B=8; 5.6–11.2 s trains at
   B=16/32) and, at B=16/32, two 16-request waves (n=32 rows). The
   steady-state aggregate at B=32 implied by per-request rates is
   **B × mean_pr ≈ 32 × 32.8 ≈ 1,050 tok/s** (T_step 88.7 ms ≈ 237 GB/s
   effective over the ~21 GB weight stream), ~6× the recorded 172.48. The
   "ceiling is above B=32" conclusion survives; its magnitude was an
   artifact. Proposed register row 15: S2 citation should be T151528Z, and
   the S2/W10/O8 numbers must be labeled span-average, not steady-state.
2. **The real open question N1's death leaves behind:** why does the clean
   decode step roughly double from B=1 to B=32 (≈40 → 76–89 ms) when weight
   bytes fully amortize? Candidates: XPU-graph batch-bucket padding
   (B16 and B32 measuring identical T is the bucket signature),
   scheduler/async-output overhead, or GDN-state kernels at ~2× my byte
   estimate (bf16 state or double-buffered r+w). Discriminating cell
   (GPU work, proposed only — not run): one B-sweep with *staggered*
   arrivals or a post-TTFT measurement window, per-level spec counters.

### Board impact

Strike N1 from the new-candidates list (kill-probe executed with a negative
result — the cheap outcome the probe was designed for). The aggregate
serving line inherits item 2 above as its new open question, and the
register gains rows 15 (S2 citation + span-average labels). No other
verdict in this review changes.

## Closing

Outcome: 10 surfaces reviewed; 4 register rows independently confirmed; 2 new
evidence findings (burst cells; cache-hit TTFTs) and 3 new candidates
proposed. Evidence: all claims above cite files/lines/numbers in this
document. Checks: `/tmp/advreview_arith.py` (arithmetic battery), direct
safetensors-header reads, counter-delta recomputations. Failures: none
(CPU-only constraint held; no GPU touched). Unverified items: shared-head ×4/round and chunk=round both remain counter-inferences until probe 1;
N1's fit has since been EXECUTED with a negative result (see Addendum — refuted as knee explanation); N3's offline gate is proposed, not run. Next deciding
experiment: probe 1 with the amended cells (it arbitrates surfaces 1, 2, 4,
7, N2 simultaneously).
