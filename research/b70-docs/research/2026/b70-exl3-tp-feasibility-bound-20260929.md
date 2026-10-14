---
title: "EXL3 TP2 feasibility bound — adversarial review"
type: research
date: 2026-09-29
status: provisional
author: "b70-adversarial"
scope: "CPU-only adversarial re-derivation from retained raw evidence (JSONL/headers/source); no GPU, no container, no build, no pull."
publication_gate: PROVISIONAL_NOT_FOR_PUBLIC_HEADLINE
---

# EXL3 TP2 feasibility bound — adversarial review (2026-09-29)

> PROVISIONAL — NOT FOR PUBLIC HEADLINE. Private review note (b70-adversarial).
> CPU-only: no GPU, no container, no build, no pull. All numbers below are
> recomputed from retained raw evidence (JSONL/headers/source), not taken from
> the reviewed note. No shared authority was edited.
>
> Target of review: `research/2026/b70-tp2-decode-feasibility-20260929.md`
> (handoff note; claims: 44.1 ms round @63.25 tok/s; TP2 ~85–110 tok/s;
> stacked with M=4 kernel ~110–140; TP2 + sub-4bpw ~150+).
> Companion claims: `research/2026/b70-breakthrough-architect-20260928.md` §B11
> (1.5–1.6×, flag-slot 5–15 µs), corrected by
> `research/2026/b70-breakthrough-review-20260929.md` §4 (1.33× @230 W,
> flag-slot spec-tier).
> Evidence base: RL-090, RL-098, RL-102, RL-071, RL-075, RL-059, RL-091,
> RL-100; artifact headers of the 4bpw and 3bpw checkpoints; the pinned
> vLLM-XPU source tree.

## 0. Verdict summary

| # | Claim | Verdict | One-line reason |
|---|---|---|---|
| 1 | Round = 44.1 ms at 63.25 tok/s (note premise) | **WEAK** | Cross-window mix. 44.10 ms is the single-request tight window (yield 3.13 → 70.1 tok/s); the n=5 p512 cell is **47.2 ms** (yield 2.95 → 63.25). |
| 2 | M4 verify = 34.46 ms / 13.12 GB / 381 GB/s = 62.6 % of 608 pin (RL-098) | **CONFIRMED** | Independent artifact census reproduces 13.126 GB and 34.46 ms (381 GB/s). |
| 3 | "Remainder 9.6 ms (attention, norms, MTP partials, gaps)" | **WEAK (composition)** | ~74 % of the remainder is **draft-side weight traffic** (3× shared 6-bit head re-read = 2.86 GB + MTP module 0.64 GB); attention/norms/sampler/host ≈ 2.5 ms. The label matters: the draft part is shardable, the 2.5 ms is not. |
| 4 | Native MTP3 | **CONFIRMED** | `mtp_num_hidden_layers=1`, `mtp_use_dedicated_embeddings=false`; 3 sequential draft forwards/round; draft logits use the **shared** target `lm_head` (source-verified). |
| 5 | Yield 2.79 (`63.25 × 0.0441`) | **REFUTED** | Measured yields: 2.954 (p512@230), 2.935 (p8192@230), 2.895/3.004 (@150), 3.134 (tight window). 2.79 exists in no window; it is the product of one window's rate and another's round. |
| 6 | TP2 ≈ 85–110 tok/s (90–110) | **REFUTED as base; conditional corner only** | Requires a 20–35 µs collective (3–5× better than the only measurement, 102 µs) **and** zero per-card shrink penalty **and** capture-safe cross-card waits. At the measured barrier: **70.7–76.6 tok/s (×1.12–1.21)**. |
| 7 | 3 bpw 75–80 tok/s | **REFUTED against the measured record** | Ideal-traffic projection is 82 tok/s (ratio 1.224, re-derived); the only measured 3 bpw serving result is **42.44 tok/s (0.671×, RL-100, adoption NO-GO)**. Re-entry needs a named changed variable (K3 kernel per-byte rate/coverage). |
| 8 | Stacked 150–190 tok/s | **REFUTED (hard)** | At the note's own yield 2.79, 190 tok/s needs a **negative** collective budget; at the measured yield it needs 4.3 µs/op (below the wire+launch floor). Additive corner: 98 (measured barrier) to 122 (60 µs barrier) tok/s. |
| 9 | Collectives = 2/block × 64 = 128 | **WEAK** | Exact count: **138 allreduces** (2×64 + 2×3 draft + 4 embedding) + 1 full logits allgather (~2.0 MB) + 3 local-argmax pair gathers. |
| 10 | Barrier cost 20–50 µs (B11: 5–15 µs) | **INSUFFICIENT EVIDENCE** | Measured xccl on this host: 101.7–112 µs/op (RL-071). The flag-slot primitive is greenfield (RL-091: no spin/poll precedent; 60 µs kill threshold); the 5–15 µs is a copy-latency estimate, not a measurement. |
| 11 | "The 9.6 ms term is card-independent" | **WEAK (errs against TP2)** | 7.1 of the 9.6 ms is shardable draft traffic; TP2's true non-collective remainder ≈ 5.7 ms. The note is conservative here. |
| 12 | TP2 + XPU graph capture | **INSUFFICIENT EVIDENCE (process blocker)** | §20 rule 1: capture always on. Cross-card waits inside XPU graph capture are unverified; RL-075's TP2 arms ran graphs-off (and lost 45–71 %). |
| 13 | The pinned `exl3xpu` fork supports TP2 | **INSUFFICIENT EVIDENCE** | Not determinable from the local mainline tree; requires static inspection of the pinned image (note's blocker 1). |

Two claims the note does **not** make that a reader will supply anyway:
the **RL-075 measured TP2 serving outcome** (−71.4 % @p512, −45.5 % @p8192
vs matched 1-card control) is not cited in the note at all, and the
**energy** consequence (2 cards ≈ 400 W for one stream) is not priced.

## 1. Ground truth recomputed from raw evidence

### 1.1 Yield / round / rate reconciliation (the 2.79 check)

Source: `results/2026/09/b70-probe1-exl3-rate-20260929/cells-{150,230}W.jsonl`
(counter deltas) and `RESULTS.md` §M4/M5 + the tight-window attribution.

| arm/cell | rounds | accepted | yield (1+A/R) | round (1/chunk-rate) | tok/s (median) |
|---|---:|---:|---:|---:|---:|
| 230 W p512/g128 | 260 | 508 | **2.954** | **47.2 ms** | 63.25 |
| 230 W p8192/g128 | 263 | 509 | **2.935** | 48.7 ms | 61.12 |
| 150 W p512/g128 | 267 | 506 | 2.895 | 63.0 ms | 49.22 |
| 150 W p8192/g128 | 256 | 513 | 3.004 | 63.2 ms | 49.00 |
| 230 W tight window (p~6064/g256, 1 request) | 82 | 175 | **3.134** | **44.10 ms** | 70.1 (diagnostic) |

Check: 2.954 × 21.17 rounds/s = 62.5 ≈ 63.25 (medians are different
samples); 3.134 / 0.0441 = 71.1 ≈ 70.1. The note's 2.79 is neither —
it is `63.25 tok/s × 0.0441 s`, i.e. the p512 **rate** against the tight
window's **round**. Correct pairing is (47.2 ms, 2.95) or (44.1 ms, 3.13).
**All tok/s projections below are stated on the 44.10 ms basis (≈ +7 %
optimistic vs the n=5 serving cell) and use yield 2.954.**

### 1.2 Round budget closure at TP1 (4 bpw, MTP3, 230 W)

Per-family bytes are the artifact's real trellis bytes (safetensors
headers, 4 bpw checkpoint), counts from RL-098 §6:

| family | shape | bytes/call | count | measured M=4 rate | time |
|---|---|---:|---:|---:|---:|
| in_proj_qkvz fused | 5120→16384 | 41.9 MB | 48 | 383 GB/s | 5.25 ms |
| out_proj (GDN) | 6144→5120 | 15.7 MB | 48 | 280 | 2.69 |
| qkv fused | 5120→14336 | 36.7 MB | 16 | 349 | 1.68 |
| o_proj (full attn) | 6144→5120 | 15.7 MB | 16 | 297 | 0.85 |
| gate_up fused | 5120→34816 | 89.3 MB | 64 | 419 | 13.64 |
| down_proj | 17408→5120 | 44.6 MB | 64 | 350 | 8.16 |
| lm_head K6 | 5120→248320 | 953.2 MB | 1 | 435 | 2.19 |
| **main pass** | | **13.126 GB** | | **381 GB/s** | **34.46** |
| draft: 3 × (MTP module 0.213 + shared head 0.954) | | 3.499 GB | | ~490 GB/s | 7.14 |
| attention/KV/GDN/norms/sampler/host/gaps | | — | | — | 2.50 |
| **round** | | **16.63 GB read/round** | | **377 GB/s** | **44.10** |

Two corrections to the note's framing:
1. The round reads **16.63 GB**, not 13.12 GB. The 3.50 GB difference is
   the draft phase, of which **2.86 GB is the shared 6-bit lm_head
   re-read three times** (source: `qwen3_5_mtp.py` `compute_logits` +
   `llm_base_proposer._draft_logits` → shared `lm_head`; the checkpoint
   has no MTP-private head tensor).
2. RL-090's "one-read budget ≈ 16.86 GB" is the **artifact size**
   (15.7 GiB, including 2.54 GB embedding and 0.92 GB vision tower that
   are never read per round). The op-level traffic 16.63 GB happens to be
   within 1.4 % of it; the *derivation* is not the same. Effective rate on
   the correct numerator: 377 GB/s (not 382).

### 1.3 Floors

- 608 GB/s = GDDR6 spec (256-bit × 19 Gbps), the "pin" floor.
- 521 GB/s = best measured pure-read kernel (`sum()` 1 GiB, RL-098 §4)
  = 85.7 % of pin. RL-098's "85 % of pin" target is therefore ≈ 99 % of
  what any kernel on this stack demonstrated. All TP2 projections below
  are capped at 495–521 GB/s, never at 608.

## 2. TP2 transport pricing — copy-only P2P, ACCESS, no atomics

### 2.1 Exact collective census per round (source-audited)

| source | count | payload |
|---|---:|---|
| Row-parallel layer outputs (2/layer × 64: `out_proj`/`o_proj` + `down_proj`) | 128 | 5120 × 2 B = 10 KiB |
| Draft MTP layer (2 × 3 sequential draft forwards) | 6 | 10 KiB |
| `VocabParallelEmbedding` allreduce (1 verify + 3 draft; `vocab_parallel_embedding.py:506`) | 4 | 10 KiB |
| **allreduces total** | **138** | **2.76 MB ring traffic** |
| Target logits allgather (verify, M=4 rows × 248320 × 2 B) | 1 | ~2.0 MB |
| Draft local-argmax pair gathers (`get_top_tokens`: [batch, 2·tp]) | 3 | ~10 B each |

### 2.2 Wire time vs fixed cost

At the verified 11.5 GB/s peer copy rate, one allreduce = 2 × 10 KiB =
20 KiB on the wire = **1.74 µs**. The measured cost is **~102 µs**
(RL-071/O14 ladder: 101.7 µs @4 KB, 104.5 @64 KB, 129.2 @512 KB — flat
in size ⇒ pure fixed cost: launch + handshake + poll + host staging).
**The 11–12 GB/s link is irrelevant to the TP2 verdict** (wire = 0.25 ms
of a 14 ms bill, 1.8 %); the whole question is the fixed per-op latency.
Without atomics the barrier must be copy + flag + local poll; that
protocol has **no precedent in any fleet kernel** (RL-091 audit C) and
must also survive XPU graph capture (§20 rule 1).

### 2.3 What each barrier band buys (allreduces + ~0.5 ms logits)

| barrier | source | collective bill | TP2 round (base) | tok/s |
|---:|---|---:|---:|---:|
| 102 µs | **measured** (xccl, RL-071) | 14.6 ms | 39.5 ms | 74.8 |
| 60 µs | RL-091 kill threshold (target) | 8.8 ms | 33.7 | 87.7 |
| 35 µs | note's mid aspiration | 5.3 | 30.3 | 97.7 |
| 20 µs | note's best aspiration | 3.3 | 28.2 | 104.8 |
| 15 µs | B11 text ("copy latency") | 2.6 | 27.4 | 107.8 |

The note's headline band lives entirely in the last two rows, i.e. in
**unmeasured** territory 3–5× better than the only measurement on this
host. The B11 text's "~102 µs/4 KB is host-staged; a device-resident
flag-slot round-trip is bounded by copy latency ~5–15 µs" is a *design
hypothesis*; RL-091's audit already calls reader-side visibility of peer
writes the core unknown and sets the 60 µs kill threshold.

### 2.4 Graph-capture requirement

§20 rule 1 makes capture-on mandatory for any champion number. RL-075's
TP2 arms ran collectives graphs-off (per O14's own recommendation) and
measured −71.4 %/−45.5 %. A TP2 result taken graphs-off is diagnostic-only
by contract; a TP2 result with collectives inside the captured region has
never been demonstrated on this stack (RL-091 audit C: warning-gated,
default-off envs). This is a **process gate**, not an optimization.

## 3. Sharding census and legality (source + config)

- 64 layers = 48 linear-attention (GDN) + 16 full-attention
  (`full_attention_interval=4`, config.json).
- Per layer, allreduce-bearing ops: `out_proj`/`o_proj` (RowParallelLinear)
  + `down_proj` (RowParallelLinear). Column-parallel (no comm):
  `in_proj_qkvz`, `qkv_proj`, `gate_up_proj`, `conv1d`.
- GDN state under TP2: temporal state `(num_v_heads/tp, 128, 128)` →
  48→24 heads/card (1.57 MB/layer, f32); conv state 10240→5120 rows/card.
  **No cross-rank collective for the recurrence itself** (heads are
  independent); only the out_proj allreduce. Total GDN state 151 MB → 75.5
  MB/card, read+written each round.
- Attention under TP2: q heads 24→12, kv heads 4→2, head_dim 256 → q+gate
  12288→6144, kv 1024→512, qkv 14336→7168 per card; KV read halves.
- lm_head / embeddings: vocab-parallel 248320 → 124160/card; logits are
  allgathered (verify) or local-argmax-reduced (draft).
- MTP module: 1 full-attention layer + `mtp.fc` (10240→5120) + shared head;
  all shardable; **no MTP-private head**.
- Replicated (not sharded): norms, `in_proj_ba` (tp-aware but small),
  `A_log`/`dt_bias`, vision tower 0.92 GB (never run for text decode).
- Divisibility: 48/2, 24/2, 16/2, 4/2 all integer; GDN `divide()` clean.
  **TP2 sharding is legal** — no divisibility blocker.
- **Open (blocker 1):** whether the pinned `exl3xpu` fork implements any of
  this for the EXL3 quant path is unverified (the local mainline tree has
  no exl3 registration; the note itself says "likely absent").

## 4. Local dequant layout and per-card shrink efficiency

**Layout (from the checkpoint headers).** Every EXL3 linear is a single
int16 trellis tensor of shape `(K/16, N/16, 64)` plus per-K-channel
`suh` and per-N-channel `svh` f16 scales and a scalar `mul1`. Dequant is
fused into the GEMM (`had_in` → `GemvKernel` M≤2 / `DpasKernel` M≥3 →
`had_out`; `SMALL_M_MAX=128`). Consequences for TP2:

- Column-parallel shards slice the **N/16 axis**; row-parallel shards
  slice the **K/16 axis**. Both are clean contiguous slices — **no
  repacking, no re-quantization, scales shard with their axis**. This is
  the one TP2 cost that is genuinely zero.
- Strip geometry per card: `n_strips = (N/16)/NT`, `P = min(tk,
  ceil(1408/n_strips))`. Work-item census at NT=4: qkvz 8192→128 strips ×
  P=11 = 1408; qkv 7168→1456; gate_up 17408→1632; down 8704→1440; o/out
  1440; lm_head 1940 — **all ≥ the 1408 target**, so the catastrophic
  small-N class (v_proj 1024 → 856 units → 11–13 % of pin) should not
  fire. The mid-N slope still applies.

**Per-card shapes and rate proxies (measured M=4 classes):**

| family | TP2 per-card shape | per-card bytes | rate proxy | time |
|---|---|---:|---:|---:|
| in_proj_qkvz | 5120→8192 (N/2) | 1.01 GB | ~345 GB/s | 2.91 ms |
| out_proj | 3072→5120 (K/2) | 0.38 | ~270 | 1.40 |
| qkv | 5120→7168 (N/2) | 0.29 | ~355 | 0.83 |
| o_proj | 3072→5120 (K/2) | 0.13 | ~288 | 0.44 |
| gate_up | 5120→17408 (N/2) | 2.86 | ~382 | 7.48 |
| down_proj | 8704→5120 (K/2) | 1.43 | ~350 | 4.08 |
| lm_head | 5120→124160 (V/2) | 0.48 | ~430 | 1.11 |
| **per-card main pass** | | **6.563 GB** | **360 GB/s** | **18.24 ms** |

Note the sharding asymmetry (this corrects an easy mis-model): only the
four column-parallel families halve N; the row-parallel families halve K
and keep N=5120, so their strip geometry is unchanged. The measured
N-slope therefore costs ~6 % vs the naive "halve bytes, keep 381 GB/s"
estimate (17.23 ms), not the 20–25 % a uniform-N-halving model predicts.
Band used below: **17.4 / 18.2 / 21.5 ms** (no penalty / shape-aware /
shape-aware + 10 % per-unit overhead haircut).

## 5. Additive round budget — conditional ceilings (NOT measurements)

Terms: main = per-card EXL3 linears; draft = per-card MTP+head; fixed =
non-shardable attention/norms/sampler/host (~2.0–2.5 ms); coll =
collectives (barrier × 138 + logits). Yield 2.954.

| scenario | main | draft | fixed | coll | round | tok/s | ×63.25 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **S0** TP1 4 bpw, tight window (**measured**) | 34.46 | 7.14 | 2.50 | — | 44.10 | 67.0 | 1.06 |
| **S1** TP1 + B6 kernel (all families to 85 % pin; RL-098 floor) | 25.38 | 7.14 | 2.50 | — | 35.02 | 84.4 | 1.33 |
| **S2** TP1 + 3 bpw ideal traffic (1.224×, K4-class rate) | 36.03 | — | — | — | 36.03 | 82.0 | 1.30 |
| **S3** TP2 4 bpw, **102 µs** barrier (high/base/low) | 17.4–21.5 | 3.52 | 2.20 | 14.58 | **38.5–41.8** | **70.7–76.6** | **1.12–1.21** |
| **S3b** TP2, 60 µs barrier (RL-091 target) | 19.20 | 3.52 | 2.20 | 8.78 | 33.70 | 87.7 | 1.39 |
| **S3c** TP2, 35 µs barrier (note mid) | 19.20 | 3.52 | 2.20 | 5.33 | 30.25 | 97.7 | 1.54 |
| **S4** stack TP2+3 bpw+kernel @495, 102 µs | 10.18 | 3.35 | 2.00 | 14.58 | 30.11 | 98.1 | 1.55 |
| **S4b** same, 60 µs barrier | 10.18 | 3.35 | 2.00 | 8.78 | 24.31 | 121.5 | 1.92 |

Hard ceiling (byte floor × yield cap, best-case 3 bpw + TP2 + kernel,
non-collective floor 14.95 ms):

| yield | max @60 µs barrier | max @102 µs barrier |
|---:|---:|---:|
| 2.954 (measured) | **127 tok/s** | **100 tok/s** |
| 3.13 (tight window) | 134.7 | 106.0 |
| 4.0 (perfect MTP3, never observed) | 172.2 | 135.4 |

S1/S2 are the *active* TP1 track's arithmetic ceilings, both measured-adjacent
(RL-102 priced the reduction patch at 1.4–2.1 ms/pass → 65.3–66.2 tok/s;
the M=4 rewrite's ≥6.5 ms residual is the ×1.26 step to ~79–84 tok/s).
S3/S4 are TP2; note that the TP2 base case (S3, measured barrier) is
**at or below the TP1 kernel track's floor** and requires new hardware
behaviour to beat it.

## 6. Attack: the three claimed bands and the double-count

### 6.1 "TP2 90–110"

Reproduced: 17.23 (halved verify) + 9.6 (unchanged remainder) +
2.8–6.9 (20–50 µs × 128–138) = 29.6–33.7 ms → 88–100 tok/s. The band is
therefore **entirely a function of the barrier assumption**; every other
term is already favourable (no shrink penalty, no draft/embedding
collectives, remainder not credited for sharding, graphs assumed free).
With the measured barrier and the same favourable terms: 41.4 ms →
**71.3 tok/s (×1.13)**. Additionally the note's own band is internally
loose: "30–33 ms → 85–110 tok/s" spans yields 2.55–3.63.

### 6.2 "3 bpw 75–80"

The 1.224× ratio is **correct** and I re-derived it from headers: 3 bpw
body trellis 9.1226 GB (4 bpw 12.1635, exactly ×0.75), head stays 6-bit
(0.9535 GB), MTP module stays 4-bit (0.2126 GB) → round traffic
13.589/16.625 = 0.817 → **1.224×**, round 44.10/1.224 = 36.0 ms →
82.0 tok/s ideal. But this is a *byte* projection at K4-class per-byte
rate, and the measured 3 bpw serving result is **42.44 tok/s (0.671×,
RL-100)**: round 68.3 ms at essentially unchanged acceptance — the loss
is kernel/round speed (K3-ESIMD vs K4-ESIMD, coverage gaps to the Triton
fallback). Quality passed 6/6, so the blocker is speed only. **75–80
requires a named changed variable** (K3 kernel per-byte rate or coverage);
the RL-102 NT sweep was K4-only and does not transfer.

### 6.3 "Stacked 150–190" and the multiplicative double-count

Multiplicative product: 63.25 × 1.26 (kernel) × 1.224 (3 bpw) × 1.33–1.75
(TP2) = **130–171 tok/s**. The additive composition of the *same* three
levers is **98 (measured barrier) / 122 (60 µs barrier)**. Three separate
errors make up the gap:

1. **Same term, one set of bytes.** All three levers act on the per-card
   EXL3-linear term. Their savings are absolute ms on one byte set:
   kernel ≤9.07 ms at TP1, ≤4.5 ms under TP2; 3 bpw ≤8.2 ms at TP1, less
   under TP2. Multiplying ratios assumes each lever re-prices the full
   round.
2. **A collective floor no lever removes.** 138 × barrier is added, not
   multiplied; at the measured barrier it is 14.6 ms — larger than the
   entire per-card main pass under 3 bpw+kernel (10.2 ms).
3. **The yield cap.** At MTP3 the round cannot emit more than 4 tokens.
   At the note's own yield 2.79, **190 tok/s requires a negative
   collective budget** (−0.27 ms) even with zero-cost barriers and a
   perfect kernel; at the measured yield 2.954, 190 tok/s needs 4.30 µs/op
   (below the 1.74 µs wire time + launch floor) and 150 tok/s needs
   34.4 µs/op (below the 60 µs kill threshold). At the *perfect* MTP3
   yield (4.0) the absolute maximum is 172 tok/s — and that already
   assumes a 60 µs barrier that does not exist.

## 7. Blocking obstacles, ordered by how little is established

1. **The barrier primitive** (least established). 20–50 µs is 2–5× better
   than the only measured number; the protocol is greenfield (no spin/poll
   precedent, RL-091), must be copy+flag+poll (no atomics), and must be
   graph-capturable (§20 rule 1). If it lands at the measured 102 µs, TP2's
   base case is ≤×1.21 and the stacked corner ≤×1.55.
2. **Engine support** for TP2 in the pinned `exl3xpu` fork (note's blocker
   1, "likely absent"). If absent, TP2 means a custom runner — a different
   project than "a probe".
3. **Graph capture of cross-card waits** (RL-091 audit C: reader-side
   visibility unknown; default-off envs). Without it, no TP2 number can be
   a champion input, and RL-075's −71 % class is the reference.
4. **Per-card shrink efficiency** on the halved geometries (no direct
   measurement; my band 17.4–21.5 ms is a proxy from nearest-N cells).
5. **Two-card power/thermal budget** (2 × ~200 W sustained for one stream;
   caps are not uniform across cards and swap across reboots, §3).
6. **Energy per token**: at the base case TP2 draws ~2× the power for
   ×1.18 rate ⇒ ~1.7× worse J/token than the incumbent. The note does not
   price this; it is decisive for a serving product even when tok/s rises.

## 8. Cheapest falsifier, thresholds, deadline, go/no-go, M4 pricing

**Falsifier ladder (cheapest first):**

- **F0 (zero GPU, ~30 min, static):** inspect the pinned image's source
  for TP2 support in the exl3 quant path (`tensor_parallel_size` handling,
  sharded loading of trellis/suh/svh). If absent → TP2 is a
  custom-runner project; the note's blocker 1 is the whole ball game.
  *Owner: b70-upstream-researcher.*
- **F1 (~minutes GPU, decisive):** re-run the O14 harness at the **actual
  payload (10 KiB)** with **XPU graph capture ON**, both pass orders, 2
  ranks, ≤5 min. Threshold: **>60 µs/op ⇒ TP2 base case dead** (RL-091's
  own kill rule); ≤35 µs required for the note's 90–110 band; the wire
  floor is 1.74 µs/op so anything ≥25 µs is 90 % fixed overhead.
  *Owner: b70-profiler.* This single number decides the campaign.
- **F2 (~1 session):** matched-control 2-card screen (graphs ON, 230 W both
  cards, p512/g128 n=3, spec-canary) only if F0/F1 pass. Pre-declared
  band +20–50 % per ADR-0008; abort on corruption/reset/OOM/graph-capture
  failure. *Owner: b70-bench-engineer / b70-executor.*

**Go/no-go:** GO to a TP2 campaign only with (a) engine TP support
confirmed statically, (b) a **captured** collective ≤60 µs at 10 KiB,
(c) per-card shrink penalty measured <10 %, (d) the matched-control screen
≥1.2× with graphs on and a clean canary. NO-GO if the barrier ≥60 µs or if
capture cannot span remote waits — in that case TP2 stays shelved exactly
as the note concludes, and the TP1 track continues.

**M4 work pricing (the TP1 track, for comparison):** reduction patch
1.4–2.1 ms/pass → 42.0–42.7 ms → **65.3–66.2 tok/s** (+3.3–4.7 %), cheap
and low-risk; per-family NT=2 ≤0.5 ms (screen only); M=4 tile rewrite
≥6.5 ms residual → round ~35 ms → **~79–84 tok/s** — the same ceiling
class as TP2's base case, with no new failure modes and no second card.
The two tracks are **not additive either**: the M=4 rewrite's benefit
halves under TP2 (same bytes argument).

## 9. Failure modes the design did not consider

1. **The measured TP2 outcome is absent from the note.** RL-075 measured
   TP2 at −71.4 %/−45.5 % against a matched 1-card control with clean
   gates; the note cites only the *correctness* result (RL-071) and the
   *primitive* hope (B11). A reader of the note alone would not know TP2
   has already been tried on this host and lost badly.
2. **Energy per token** (2 cards for 1 stream; base case ≈1.7× worse
   J/token) — invisible in a tok/s framing.
3. **Opportunity cost of the second card**: an independent stack serves a
   second stream at full speed (ADR-0008 default) instead of half of one
   round. TP2 converts 2 × 63 tok/s of capacity into 1 × ~75 tok/s.
4. **Draft-side head traffic is the largest single non-main term**
   (2.86 GB/round, 5.6 ms at M=1) and is untouched by every lever in the
   stack except a draft-head quantization (head_bits 6→4 would save ~1/3
   of it). It is also the term that makes the 9.6 ms remainder *not*
   card-independent.
5. **Graph-capture re-entry risk**: TP2 with graphs off re-opens the exact
   penalty class RL-075 measured; the note treats capture as a "blocker"
   rather than as the process gate it is under §20 rule 1.

## 10. What I did not verify (handoffs)

- Engine-side TP support in the pinned `exl3xpu` image (F0; requires
  container/source inspection — not doable CPU-only from this tree).
- Any live timing: I did not run the F1 probe (CPU-only mandate). All
  barrier numbers are the retained RL-071 measurements or the reviewed
  documents' estimates, labelled as such.
- The exact acceptance of the 3 bpw arm vs the 4 bpw arm (RL-100 states
  "essentially unchanged"; I used its medians as recorded).
- One number I could not close: RL-090's counter-rate column implies
  ~2.85–2.90 tok/round at the 230 W cells while the per-position
  survivals give 2.90–2.95; the 0.5–2 % spread is inside the recorded
  edge-effect band and does not change any verdict.

**Coordination:** conclusions to go back to b70-architect (design owner)
and b70-lead via the lead's result; F0 to b70-upstream-researcher, F1 to
b70-profiler, F2 to b70-bench-engineer/executor. No shared authority was
edited; this file is private review output.
