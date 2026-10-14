---
title: "Sub-4bpw EXL3 trellis + margin-aware requant contingency — zero-GPU plan (2026-09-29)"
type: research
date: 2026-09-29
status: provisional
---

# Sub-4bpw EXL3 trellis + margin-aware requant contingency — zero-GPU plan (2026-09-29)

- Owner: b70-quantization-engineer
- Status: PROVISIONAL — NOT FOR PUBLIC HEADLINE
- Mode: zero-GPU planning. No model loads, no conversions, no downloads, no writes to
  any artifact or authority. This file is the only output.
- Situation-compiler note: `situation.py compile --intent research` returned `halt: true`
  (blocked: RL-087, campaign in flight, no halt_reason field). This plan complies: read-only
  inspection of configs/tensor headers/mounts plus arithmetic; the assigned GPU campaign
  (probe 1, GPU0) and CPU probe (2A) are untouched.
- Inputs: `research/2026/b70-breakthrough-architect-20260928.md` §B2/B3 + §D rows 4–5;
  artifact `/mnt/<nvme-disk>/b70-exl3-qwen38-4bpw-20260926` (read-only).

Sections:

- [0. Evidence base](#0-evidence-base) — everything below is traced to a config, tensor header, mount, or API read on 2026-09-29
- [Task A — sub-4bpw EXL3 trellis plan](#task-a--sub-4bpw-exl3-trellis-plan)
- [Task B — margin-aware requant contingency](#task-b--margin-aware-requant-contingency)
- [Appendix: evidence log](#appendix-evidence-log)

---

## 0. Evidence base

All numbers in this plan were read live on 2026-09-29 (~12:40–13:20 UTC). Full log in the
Appendix. Headline corrections to the tasking premise:

1. **A complete, symlink-free BF16 source exists on host.**
   `/mnt/<nvme-disk>/b70-bf16-qwen38-20260926/` — 18/18 real shards, 0
   symlinks, 55,563,006,776 B (55.563 GB / 51.75 GiB), downloaded 2026-09-26 22:44–22:50
   (arm-C source; HF cache tree `1d4bf0f2ff60…`). The 2026-09-25 "dead-symlink farm"
   finding applies only to the **T-FORCE** copy
   (`/mnt/<sata-disk>/Qwen3.8-27B-BF16/` → dangling links to the deleted
   `/mnt/<models-disk>/qw38-bf16/`, verified dead now). **A ~55 GB canonical re-fetch is NOT
   required** for any re-export route; a sha256 verification pass against the embedded
   `.cache/huggingface/trees/1d4bf0f2….json` blob IDs is required before any conversion
   treats it as canonical.
2. **Upstream prebuilt sub-4bpw variants exist.** `turboderp/Qwen3.8-27B-exl3` publishes
   branch variants; our adopted artifact is branch `4.00bpw` @ `113cf7ab9580…` (exact
   match to our HF cache tree). Candidate branches (quantization_config fetched and
   parsed, format version 1.4.2 — same as ours):

   | Branch | bits / head_bits / mtp_bits | per-module bpw mix | size (HF tree API, safetensors only) |
   |---|---|---|---|
   | `4.00bpw` (ours) | 4.0 / 6 / 4 | {4: 400, 6: 1} | 16.861 GB (2 shards) |
   | `3.50bpw` | 3.5 / 6 / 4 | {4: 262, 3: 137, 5: 1, 6: 1} | 15.338 GB (2 shards) |
   | `3.00bpw` | 3.0 / 6 / 4 | {3: 400, 6: 1} | 13.820 GB (2 shards) |
   | `SC_3.00bpw_H4` | 3.0 / 4 / 3 | {3: 297, 4: 66, 2: 34, 5: 4} | 13.452 GB (3 shards) |
   | `SC_4.00bpw_H5` | 4.0 / 5 / 4 | (mixed ~4) | 16.705 GB (3 shards) |

   `V`-suffixed branches (quantized vision tower) require **ExLlamaV3 ≥ v1.4.4** per the
   upstream README; our stack is format v1.4.2 → V-variants are out of scope.
   Upstream also ships `kld_table.json` + `measurement.json` (their per-tensor KLD-vs-bits
   table and the KLD-per-bit allocation used to build the SC variants) — reference
   material only, never a substitute for our gates.
3. **Situation compiler** returned `halt: true` (RL-087 campaign in flight). This plan is
   zero-GPU and read-only by construction; no interference with probe 1 (GPU0) or the CPU
   probe 2A.

### Artifact decomposition (re-derived from safetensors headers, not remembered)

Parsed both shard headers of the adopted artifact (2,426 tensors). Axis buckets:

| Axis | Payload | of which trellis | Modules / notes |
|---|---:|---:|---|
| Body (`model.language_model.layers.*`) | 12.2295 GB | 12.1635 GB | 400 trellis modules @4 bpw (48 GDN layers × 6 + 16 full-attn × 7) |
| Embedding (`embed_tokens`) | 2.5428 GB | — | 1 tensor, **fp16, unquantized** (248,320 × 5120 × 2 B exactly) |
| LM head | 0.9541 GB | 0.9535 GB | 1 module @6 bpw (tile [320, 15520, 96]) |
| MTP (`mtp.*` top-level ns) | 0.2126 GB | 0.2123 GB | 8 trellis modules @4 bpw (1 layer + `mtp.fc`) |
| Vision | 0.9215 GB | — | 333 tensors, unquantized |
| **Total** | **16.8605 GB = 15.7026 GiB** | | matches file bytes 16,860,809,795 B |

Trellis payload scales exactly linearly with bpw (verified: up_proj 4.0 bits/weight;
lm_head 6.0 bits/weight) — trellis bytes = n_weights × bpw / 8, companions
(`suh`/`svh`/`mul1`) are noise (<0.07 GB total).

**Round-budget accounting.** The architect's recorded 16.68 GB/round reproduces exactly
as body 12.2295 + head ×2 + embedding 2.543 (MTP folded out): 16.6805 GB. This plan uses
a slightly more complete model — body + head × r_h + MTP + embedding — with r_h = 2
(architect-consistent) and r_h = 4 (brief's "head read up to 4×/round in MTP3") as a
sensitivity band. **All per-axis deltas are identical under either accounting**; only the
ratio denominators move a few points. KV (fp8, 16 full-attn layers) is a separate
context-proportional term: ctx × 32 KiB/token → 1.074 GB/step @32K, 3.277 @100K,
8.590 @262K (config: 4 kv heads × 256 dim × 2 (K+V) × 1 B × 16 layers). GDN recurrent
state ≈150 MB/req constant, context-independent.

---

## Task A — sub-4bpw EXL3 trellis plan

### A1. Which axes can move, and what each buys

Per-axis byte deltas (trellis bytes only; companions unchanged). "Per round" uses
r_h ∈ {2,4} sensitivity for head reads; MTP counted once (draft-loop multiplicity
unknown until probe 1 attribution — if the MTP layer runs per draft step, the mtp delta
grows up to 3×, i.e. 0.053 → 0.16 GB/round; still minor).

| Axis move | Δ VRAM/file | Δ bytes/round (r_h=2 / r_h=4) | Quality risk | Verdict |
|---|---:|---:|---|---|
| body 4.0 → 3.0 | −3.041 GB | −3.041 / −3.041 | Main risk: 400 modules drop 25% code precision; upstream per-tensor KLD table implies ~3.2× per-tensor KLD vs 4-bit | **The lever** — 92% of the available saving |
| body 4.0 → 3.5 | −1.520 GB | −1.520 / −1.520 | Half the error growth of body-3 (mixed 4/3 allocation upstream) | Fallback if body-3 fails quality |
| head 6 → 4 | −0.318 GB | −0.636 / −1.272 | Head feeds every sampled/drafted token; 6→4 on a 248K vocab is where token-identity flips begin | Only as part of SC_3.00bpw_H4; **not** standalone |
| head 6 → 5 | −0.159 GB | −0.318 / −0.636 | Small | Marginal; skip (SC_4.00bpw_H5 saves only 0.156 GB total — not worth a gate cycle) |
| mtp 4 → 3 | −0.053 GB | −0.053 (×draft steps) | Draft acceptance is conditioned on quantized draft weights (DFlash lesson); risk to α, not correctness | Accept only bundled in SC_3.00bpw_H4; gate α ≥ −0.03 |
| embedding | 0/round | ~0 (row-gather at decode) | Format keeps `embed_tokens` unquantized; quantizing it degrades the input distribution globally | **Fixed** — no round-byte benefit under gather accounting; VRAM-only lever with outsized risk |
| vision | 0/round | 0 (never read in text decode) | V-variants need format ≥1.4.4 (our stack is 1.4.2) | **Fixed** — VRAM-only, incompatible format bump |

**Recomputed round budgets and ideal ratios** (budget = body + head×r_h + mtp + embedding;
baseline r_h=2 → 16.893 GB, r_h=4 → 18.801 GB):

| Configuration (body/head/mtp) | Round @r_h=2 | Ideal @r_h=2 | Round @r_h=4 | Ideal @r_h=4 | Candidate branch |
|---|---:|---:|---:|---:|---|
| 4/6/4 (incumbent) | 16.893 GB | 1.000× | 18.801 GB | 1.000× | `4.00bpw` |
| **3/6/4** | 13.852 GB | **1.219×** | 15.760 GB | 1.192× | `3.00bpw` |
| **3/4/3** | 13.163 GB | **1.283×** | 14.436 GB | 1.302× | `SC_3.00bpw_H4` |
| 3.5/6/4 | 15.373 GB | 1.099× | 17.281 GB | 1.088× | `3.50bpw` |
| 3.5/4/4 | 14.737 GB | 1.146× | 16.009 GB | 1.174× | (no prebuilt; re-export only) |

For calibration: naive 4/3 = 1.333×; body-only 4→3 with everything else fixed = **1.22×**
(1.219–1.224 depending on head-read model) — corrections-register row 5 confirmed
against live headers. Long-context erosion: at ctx 262K the fp8 KV read (8.590 GB/step)
is co-dominant, and the 3/4/3 ideal falls to (16.893+8.590)/(13.163+8.590) = **1.192×**;
at 32K it stays ~1.27×. Sub-4bpw is a short/medium-context speed lever and an
every-context capacity lever, never a 262K speed lever.

**Honesty clause on "ideal":** these are traffic ratios. The dossier's own measurement
class (240–355 GB/s effective, power-limited ESIMD decode, corrected register row 11)
means the realized gain could be well below ideal if probe 1 attribution lands on
clock/power-bound rather than byte-bound. The quality gates below are independent of
that; the *promotion* to a serving route is not.

### A2. Artifact route decision

Ranked per Decision Record §Decisions 2–4 and AGENTS.md §11 (identify from config and
tensors; distinguish reconstruction / repack / requantization; preserve sources):

1. **Prebuilt turboderp variant — SELECTED for the first material step.**
   `3.00bpw` (13.820 GB) first, `SC_3.00bpw_H4` (13.452 GB) second. Rationale: exact same
   format version (1.4.2), same converter lineage, same `tensor_storage` schema, produced
   from turboderp's BF16 source with a KLD-guided allocation we get for free
   (`measurement.json`). It is a **new artifact of the same representation family** —
   no transformation performed on this host, no new lineage obligations beyond our own
   quality gates. New explicit output name; incumbent preserved unchanged.
2. **Re-export from the on-host BF16 source** — held in reserve. Triggers: (a) we want a
   bpw mix upstream does not ship (e.g. 3.5/4/4, or an own-corpus calibration à la E1);
   (b) upstream quality tables suggest a better allocation than the plain 3.00. Cost:
   ExLlamaV3 conversion must run inside the exl3xpu/champion container on **XPU**
   (assigned GPU window; no CUDA on host, and host python has no exllamav3). Source
   verified live (55.563 GB, symlink-free) — no 55 GB re-fetch; sha256-verify against the
   embedded trees metadata first. Storage: output ~14–17 GB + staging; see A5.
3. **Repack** — not applicable. There is nothing to repack: trellis payloads at a new bpw
   are new codes, not a re-containerization. (Shard merging would be a repack and is
   pointless here.)
4. **Requantize-from-4bpw — REJECTED on principle.** Dequantizing our 4 bpw trellis
   artifact and re-encoding at 3 bpw is a **requantization of an already-quantized
   representation**: the result approximates the 4 bpw artifact's approximation, not the
   model. It compounds error (each trellis stage adds correlated quantization noise),
   breaks the calibration contract (EXL3's KLD-guided bit allocation in
   `measurement.json` was computed against BF16 activations — calibrating against 4 bpw
   outputs optimizes the wrong target), and violates Decision 3's lineage rule that a
   lossy transform requires a declared true source and new correctness evidence. The
   artifact would be strictly worse than the upstream 3.00bpw at similar cost. Do not
   spend a GPU window on it.

### A3. Kernel/runtime support gate (must pass before any download)

What is **parameterized** (evidence): `quantization_config.tensor_storage` carries
per-module `bits_per_weight`; the production artifact already loads and serves two
different bpw (4 body + 6 head) in one model — the XPU runtime dispatches trellis
geometry per tensor, it is not hardcoded to one bpw. Top-level `bits`/`head_bits`/
`mtp_bits` are descriptive summaries.

What is **unverified**: whether the exl3xpu trellis kernels implement the code
geometries a sub-4 artifact presents. Decisive static evidence gathered zero-GPU —
trellis tile last-dim = 16 weights × bpw bits (verified on our artifact: 64 ↔ 4 bpw
body, 96 ↔ 6 bpw head; and on the 3.00bpw branch header fetched by HTTP range: 48 ↔
3 bpw):

| Branch | Tile widths presented | Proven on XPU? |
|---|---|---|
| 4.00bpw (incumbent) | {64, 96} | **Yes** — production since 2026-09-27 |
| 3.00bpw | {48} ∪ {64 and/or 96 for mtp/head, both proven} | only 48 **unproven** — exactly one new width |
| 3.50bpw | {48, 64, 80, 96} | 48, 80 unproven |
| SC_3.00bpw_H4 | {32, 48, 64, 80} | 32, 48, 80 unproven; head-4 uses 64 (proven) |

**Gate A3 (pre-download, zero-GPU):** confirm from the exl3xpu fork source that the XPU
trellis dispatch table covers tile width 48 (and 32/80 for the SC branch). Source is not
on host (`~/vllm-src` is stock vLLM; the fork lives only in image
`ghcr.io/0xsero/exl3xpu@sha256:21412bdd…`). Resolution paths, cheapest first:
(a) the parallel upstream-researcher agent (already tasked with the variant list — ask
for the fork's dispatch table too); (b) `docker create` + `docker cp` of the kernel
source from the image on the isolated daemon — **no container start, no GPU touch**,
but coordinate even that with the running campaign before touching its daemon;
(c) upstream query to 0xsero. If width 48 is absent, Task A is blocked on
b70-kernel-engineer (port the 48-wide code path) — that is an escalate, not a workaround.
MTP3 note: SC_3.00bpw_H4 carries `mtp_bits: 3` — if the speculative drafter asserts a
fixed 4-bit MTP path anywhere (loader or graph), that is a second thing A3 must confirm;
the plain `3.00bpw` keeps `mtp_bits: 4` and avoids the question entirely — one more
reason it is the first candidate.

**Gate A3b (first bounded load, one GPU window):** loader logs must show trellis
initialization for the new geometry with zero CPU/dequant fallback, plus coherent smoke.
A load is not parity (§11) — it only clears the dispatch gate.

### A4. Quality ladder and promotion gates (increasing cost order, per §11)

Ladder (correctness scopes named per `docs/methods/correctness-validation.md`):

1. **Static coverage** (zero-GPU): index↔file coverage both shards; `tensor_storage` ↔
   safetensors name agreement; trellis tile-width census vs the A3 verdict; config
   parses; VRAM budget arithmetic (weights + fp8 KV pool at target ctx ≤ gmu 0.90 with
   ≥2 GiB post-load free).
2. **Offline NLL parity** (one bounded GPU window or CPU if tractable): per-position
   teacher-forced logprob comparison vs the **BF16 source** and vs the **4 bpw
   incumbent** on a fixed corpus (E1 corpus + battery prompts), teacher-forced, declared
   positions. **PPL ranking is prohibited on this model class** (qwen3_5 hybrid-GDN,
   Decision Record addendum 2026-09-14); use mean Δlogprob / top-1 agreement / top-20-union
   KL à la the Ornith parity protocol (2026-08-22, 315 positions). Pre-declared budget:
   candidate mean Δlogprob vs BF16 ≤ **3.0× the incumbent 4 bpw's** measured on the same
   positions in the same session (3.0× derived from upstream `kld_table.json` per-tensor
   4→3-bit KLD ratio ≈ 3.2), and top-1 agreement ≥ incumbent − 2.0 pts.
3. **Exact-answer battery — KILL GATE** (greedy, thinking off, `measure.py` battery):
   **6/6 including `code_execution` 14 ×2**. This is the gate the entire W4A16 class
   died on (ADR-0010): coherence smoke passes a fluent wrong answer; only exact answers
   catch the correlated-error flip. Any 30-answer ⇒ candidate dead at this rung, no
   appeal to NLL.
4. **Host-executed code tasks** (`task_battery.py`): ≥ **8/10** (incumbent 8/10;
   json_flatten is spec-ambiguous and lfu_cache was a real miss — treat as ≥ incumbent,
   not ≥ 9).
5. **Speculation control**: speculation-off main-model control + MTP3-on acceptance
   diagnostics; counter-α ≥ −0.03 vs incumbent (SC branch's mtp_bits 3 makes this
   load-bearing; plain 3.00bpw should be α-neutral).
6. **A/B speed cells** (only after 1–5 pass; one GPU window): matched p512/g128 and
   p8192/g128, n=5 fresh after one discarded same-shape warmup, graph capture as
   deployed (`FULL_DECODE_ONLY`), **cache OFF** for the A/B, unique ≥64-char entropy
   prefixes, zero prefix-cache-hit delta, spec counters bracketed. Metric: probe-1's
   corrected token-rate (usage `completion_tokens` / post-first interval) — chunk-rate
   is retired as a comparison basis (register row 1). Promotion gate: **≥ +10%
   token-rate sustained on p512/g128** at matched cap (150 W and 230 W arms, since EXL3
   is the most cap-sensitive arm measured). Below that: record as capacity-only win
   (+~2.8 GiB → larger KV pool at same gmu) and keep the incumbent as the serving route.

Pre-declared kill/stop rules: any non-finite output, silent fallback marker, or
deterministic-replay instability (3/3 required) fails the rung and stops the ladder.
Quality verdict handoff: b70-correctness-engineer owns rungs 2–5 sign-off; I do not
declare quality.

### A5. Storage budget (live-discovered 2026-09-29T12:41Z)

- Root `/`: 33,632,423,936 B free (31.3 GiB) — **not used** for any artifact work.
- NVMe `/mnt/<nvme-disk>`: 228,823,838,720 B free (213.2 GiB).

| Route | Components simultaneously retained | Budget | Headroom check (2× rule) |
|---|---|---:|---|
| Cheapest first step (static gates, A3 via upstream-researcher) | nothing new | **0 B** | trivially safe |
| Variant trial (one branch) | incumbent 16.861 GB (kept, unchanged) + new 13.820 GB (`3.00bpw`) or 13.452 GB (SC) + partial-download reserve ≤ 1× | peak ≈ 44.5 GB (incumbent + 2× the new artifact) | 44.5 ≪ 228.8 GB → **PASS**, ~5.1× margin |
| Re-export (reserve route) | BF16 source 55.563 GB (already resident, verify in place) + output ~14–17 GB + staging/intermediates ~10 GB + 2× on output | ≈ 116 GB peak | 116 ≪ 228.8 → PASS, but record before every step (live discovery, never remembered) |

Naming: `b70-exl3-qwen38-3bpw-<date>` / `b70-exl3-qwen38-sc3-h4-<date>` on NVMe; never
write through a symlink; incumbent directory untouched (rollback = keep both).

**Task A decision block**

- **Decision:** pursue prebuilt `turboderp/Qwen3.8-27B-exl3@3.00bpw` as the first
  material candidate (one new tile width, MTP untouched, 1.219–1.224× ideal); hold
  `SC_3.00bpw_H4` as the stretch config (1.28–1.30× ideal) behind the same gates plus
  the α gate; reject requantize-from-4bpw permanently; re-export from on-host BF16 only
  if a custom mix is justified by the 3.00bpw outcome.
- **Cost:** 0 B / 0 GPU for gates A3-static; then 13.820 GB NVMe + one load window +
  one quality window + one A/B window per candidate.
- **Gates:** A3 (kernel dispatch, pre-download) → A4 ladder (static → Δlogprob ≤3×
  incumbent → battery 6/6 KILL → tasks ≥8/10 → α ≥ −0.03 → speed ≥+10% p512 sustained).
- **Cheapest falsifying probe:** the A3 source check (zero download, zero GPU): if the
  exl3xpu trellis dispatch lacks tile width 48, no variant in the table loads and Task A
  reduces to a kernel-engineer escalation. Second-cheapest: probe 1's attribution (already
  running) — a clock/power-bound verdict demotes Task A to capacity-only before any
  bytes are spent.
- **Blocked on:** upstream-researcher (variant list cross-check + A3 fork-source
  confirmation); probe 1 attribution (realized-gain pricing + true token-rate A/B
  baseline + head-read count for the round model); probe 2A (only via Task B interaction
  below); GPU windows via orchestrator for A3b/A4/A6.

---

## Task B — margin-aware requant contingency

**Trigger (external):** probe 2A returns **weights** — i.e. the offline fp32 INT4 matmul
on real hidden states reproduces the kernel output at the first divergent layer, so the
30-defect lives in the quantized weights, not the kernels. If probe 2A returns
**kernel**, Task B is void and the finding routes to b70-kernel-engineer as an upstream
bug report with repro (probe 2's success output spec).

Theory being tested (B3 of the architect note): symmetric-INT4 introduces a *correlated
error direction* that flips low-margin logit orderings at decision tokens; magnitude
(MSE) is not the operative variable — direction is (T2 rotation evidence: KL 0.40–0.58×
at iso-bytes on real activations, 2026-09-25).

### B1. Objective — protect low-margin orderings, not MSE

Baseline (what arm C ran): AutoRound signSGD minimizes block-output MSE on calibration
activations. Replacement objective:

```
L = L_block_MSE + λ · Σ_{t ∈ D} w(m_t) · Φ(x_t, ŷ_t)
```

- `D` = decision tokens: positions in the calibration set where the **BF16 reference**
  top-1/top-2 logit margin `m_t` is small (below a pre-declared percentile, e.g. p20 of
  margins, or an absolute threshold set from probe 2's measured margin at the branch
  point).
- `Φ` = order-preservation term: hinge on the reference ranking — penalize
  `logit_ŵ(ref_top2) − logit_ŵ(ref_top1)` going positive, and/or top-1
  cross-entropy at those positions. Weight `w(m_t) ∝ 1/m_t` (fragile first).
- **Cheap margin computation:** one teacher-forced BF16 pass over the calibration corpus
  (the on-host BF16 source; CPU mmap feasible per arm-C precedent, or one bounded GPU
  window) caches per-token `(top1_id, top2_id, margin)` — a few hundred MB of side data.
  During AutoRound's block tuning the activations are already cached per block
  (`data_driven` caching, arm-C log); the margin set only *reweights* the existing
  per-position losses — no extra forward cost per iteration.
- Implementation reality: auto_round 0.14.2 has no plug-in loss hook — this is a
  **local-only patch anchored to the exact container version**, recorded as such
  (AGENTS.md §7 patch discipline). The arm-C script
  (`campaigns/2026/autoround-vs-exl3/ar-work/convert_armc.py`) already carries two
  anchored local patches (regex revert fix, VL-config restore); the objective patch is a
  third in the same file.

### B2. Knobs (in descending expected value)

1. **Rotation fold (T2/QuaRot-class, the proven lever).** Whiten the error direction
   before quantization: fold an orthogonal transform R into the source (R⁻¹ absorbed by
   adjacent norm/linear), quantize R-transformed weights. hidden = 5120 = 1024 × 5 →
   block-diagonal Hadamard (1024) × 5 blocks, or a random orthogonal in bf16. Exact fold
   preserves the function up to bf16 rounding of R·W — treat as a **lossy-ish source
   transform**: it needs its own parity rung (Δlogprob vs unrotated BF16 ≈ 0 budget,
   e.g. ≤ 0.05× of the 4-bit artifact's Δlogprob) before the quantization gates even
   start. Bounded options, one at a time: (a) rotate hidden dim only; (b) rotate FFN
   intermediate (17,408 = 1024 × 17) on up/gate projections only; (c) no rotation on
   GDN in_proj_qkv/z (state-accumulation sensitivity — the 2026-09-14 addendum showed
   recurrent-state paths are where eval artifacts concentrate).
2. **Per-layer bit allocation.** Measure per-layer sensitivity (block Δlogprob at W4 vs
   BF16 during calibration), then allocate 5-bit to the worst K layers / 4-bit to the
   rest under a bytes budget ≈ the incumbent's (iso-bytes comparison — T2's rule).
   Upstream's `measurement.json` is the template for the allocation method (dkld/dbits).
3. **Outlier / low-margin-channel protection.** Already free: GDN `in_proj_a`/`in_proj_b`
   auto-skip quantization (shape not divisible by 32 — arm-C log); keep. Explicitly pin:
   all norms, `mtp.*`, `lm_head`, embed, vision (arm-C ignore list) — no change vs arm C.
4. **Group size.** G128 → G64 on the K worst layers only (+0.5 bit/weight on those
   tensors) — cheaper than a global bit bump; must stay inside the iso-bytes budget or
   be declared as a capacity cost.

### B3. Tooling on host (verified 2026-09-29)

- Host python: **no** `auto_round`, **no** `gptqmodel`, **no** `exllamav3` (import
  checks failed). All quant tooling lives in containers.
- Proven recipe: **auto-round 0.14.2 in the champion container** — arm C ran
  `device_map="cpu"`, `low_cpu_mem_usage=True`, per-block tuning ~70–140 s/block,
  peak_ram 25.8 GB / peak_vram 8.65 GB, ~80 min wall, zero RTN fallback. Reusable script:
  `campaigns/2026/autoround-vs-exl3/ar-work/convert_armc.py` (with its two local patches
  + known export caveats: flattened-config restore, ~+12.7 GB duplicated-tensor shard
  waste). `task_battery.py` / `measure.py` in `results/2026/09/autoround-vs-exl3-20260926/`
  are the gate harnesses.
- The margin pass needs only transformers + torch (present in the same container) on CPU
  against the on-host BF16 source.

### B4. Calibration corpus

- Primary: **E1 own-corpus, verified present**:
  `results/2026/09/b70-improvement-probes-20260926/shortlist-rebuild-ours-20260926/corpus/`
  (4 × `.txt.gz` — b70docs / b70lab / projects / sessions — 19 MB total + MANIFEST.tsv).
- **Additions this time (the arm-C lesson):** the corpus alone did not prevent the
  30-defect — arm C was calibrated on E1 and still answered 30. The margin objective must
  see *decision-token-dense* text: pin the canary family (`sum(i*i for i in range(4))`
  inclusive-range class) plus new same-class canaries (slice/range off-by-one traps,
  integer-sum edge cases) into the calibration set explicitly, and let the margin
  reweighting concentrate capacity there. Corpus identity (files, seed 20260609 shuffle,
  sample count, seqlen 1536) recorded in the artifact manifest; any addition = new
  manifest entry, never a silent edit.

### B5. Gates, acceptance, and the ADR-0010 interaction

- **Kill gate (same as Task A rung 3):** exact-answer battery **6/6 incl.
  code_execution 14 ×2**, greedy, thinking off. This is the exact defect class that
  closed the family; nothing below this rung matters.
- **Greedy byte-identity on passing prompts is NOT expected** (weights change) — do not
  gate on it. Instead: deterministic replay 3/3 stable; ΔNLL budget vs BF16 on the fixed
  corpus (same protocol as A4-rung-2; pre-declare ≤ the EXL3-4bpw incumbent's multiple
  measured in the same session, target ≤ 1.0× incumbent at iso-bytes given the rotation
  head start); task battery ≥ 9/10 (arm-C level); **discriminating prompts pinned before
  eval** — the 6-case battery verbatim from `measure.py` + the canary family extensions,
  fixed in the manifest, frozen before the first candidate generation.
- **Acceptance statement (pre-declared):** the artifact is promoted to "quality-correct
  W4A16 candidate" only on 6/6 + replay 3/3 + Δlogprob budget + ≥9/10 tasks, all
  retained in a registered results root.
- **ADR-0010 interaction — what reopens the W4A16 class:** ADR-0010 closed symmetric
  int4 on Qwen3.8-27B on the grounds that *every* measured W4A16 export fails the
  battery. Two reopeners exist. (1) probe 2A = **kernel** ⇒ the closure basis was
  confounded by a kernel defect; the class reopens at the kernel level (fix routes to
  b70-kernel-engineer + upstream; Task B is void). (2) probe 2A = **weights** and a
  margin-aware + rotated artifact passes the full acceptance above ⇒ new task-quality
  evidence contradicts the closure basis; per the evidence lifecycle this requires a
  **dated correction/reconsideration of ADR-0010 before any promotion**, with the
  retained battery evidence linked. Note the honest value proposition: arm C measured
  *no speed advantage* for this family (15.0/14.4 chunk-rate class) — Task B's payoff is
  a quality-correct, vLLM-native (non-fork) alternative and the margin theory verdict,
  not throughput.

**Task B decision block**

- **Decision:** spec now, execute only on probe-2A = weights. First build =
  rotation-fold + margin-weighted AutoRound, iso-bytes vs incumbent, E1+canary corpus,
  via the anchored arm-C script chain in the champion container (CPU tuning; bounded GPU
  window only for the BF16 margin pass if CPU is too slow).
- **Cost:** one conversion run (~80 min CPU-class, peak_ram ~26 GB — verify free RAM at
  execution time) + staging on NVMe: BF16 55.563 GB (resident) + output ~19 GB (incl.
  known duplication waste) + intermediates ~10 GB + 2× output headroom ≈ 122 GB peak —
  fits the 228.8 GB live NVMe free; no re-fetch.
- **Gates:** pinned-prompt battery 6/6 (kill) → replay 3/3 → Δlogprob ≤ incumbent
  (iso-bytes) → tasks ≥9/10 → then, and only then, the ADR-0010 reconsideration note.
- **Cheapest falsifying probe:** a zero-conversion margin census on the arm-C artifact
  already on NVMe (`b70-probe2-armc-copy`, 29 GB — in use by probe 2A now, do not
  touch until it finishes): at the flip token of the failing canary, measure the BF16
  top-2 margin. If the margin is *large* (the flip is not a low-margin event), the
  correlated-error theory is wrong as stated and Task B's objective degenerates to
  plain MSE — stop and re-theorize before writing any converter patch.
- **Blocked on:** probe 2A verdict (hard gate — owns the arm-C copy and the kernel/
  weights split); probe 1 only for any speed claims (none claimed here); no dependency
  on upstream-researcher.

---

## Appendix: evidence log

All reads performed 2026-09-29 ~12:40–13:20 UTC, host `b70-host`, read-only.

1. **Situation compiler:** `tools/situation.py compile --intent research` → `halt: true`,
   blocked RL-087 (campaign in flight); working set: `registry/research-ledger.json`,
   `results/2026/09/autoround-vs-exl3-20260926/autoround-local.jsonl`. Plan complies
   (zero-GPU, read-only, one plan file).
2. **Authorities loaded:** `docs/reference/quantization-decision-record.md` (v1.3,
   decisions 1–7 + 2026-09-14 addenda: PPL-ranking prohibition on qwen3_5-class;
   DFlash drafter-vs-quantized-target lesson), `docs/reference/kv-cache-reference.md`,
   `docs/reference/engine-support-matrix.md` (EXL3-XPU row: validated_research, fp8-only
   KV, ADR-0010 primary route), `docs/operations/storage-and-vram-safety.md` (§1 live
   discovery, §2 fail-closed, §7 conversion offload lesson, §9 mount map + broken
   T-FORCE symlink farm note), `docs/methods/correctness-validation.md` (scopes + ladder),
   `research/2026/b70-breakthrough-architect-20260928.md` (§Ground truth, B2/B3, probes
   1–3, §D rows 1/4/5/11), `engines/exl3xpu/README.md` (dossier + 2026-09-29 corrections).
3. **Storage (live, `df -B1`):** root `/dev/sdb4` avail 33,632,423,936 B; NVMe
   `/dev/nvme0n1p1` avail 228,823,838,720 B.
4. **Adopted artifact** `/mnt/<nvme-disk>/b70-exl3-qwen38-4bpw-20260926/`:
   `config.json` (qwen3_5, Qwen3_5ForConditionalGeneration, 64 layers = 48 linear + 16
   full attn interval 4, hidden 5120, FFN 17408, vocab 248,320, MTP 1 layer,
   `mtp_use_dedicated_embeddings: false`, max pos 262,144); `quantization_config.json`
   (exl3 v1.4.2, bits 4.0, head_bits 6, mtp_bits 4, calib 250×2048, out_scales always,
   codebook mul1, tensor_storage 707 entries, per-module `bits_per_weight`,
   mul1_multiplier 2212286765); both safetensors headers parsed (2,426 tensors;
   axis buckets and trellis-only splits as tabulated in §0; tile geometries {64-wide ×7
   families} + head [320,15520,96]); file bytes 8,471,336,006 + 8,389,473,789 =
   16,860,809,795 B = 15.7026 GiB; HF cache tree revision `113cf7ab958054860e43fb7f3063b1af19171095`.
5. **Upstream repo** `turboderp/Qwen3.8-27B-exl3` (HF API, read-only): 26 branches;
   ours = `4.00bpw` @113cf7ab; fetched `quantization_config.json` for `3.00bpw`, `3.50bpw`,
   `SC_3.00bpw_H4` (per-module mixes recorded in §0 table); tree-API sizes 13.820 /
   15.338 / 13.452 / 16.705 GB; `recipe_*.yaml`, `kld_table.json` (per-tensor KLD 1–8
   bits; 4→3-bit ratio ≈ 3.2×), `measurement.json` (dkld/dbits allocation) on main;
   README variant table + "V-variants need ExLlamaV3 ≥1.4.4"; `3.00bpw` shard-1 header
   fetched via HTTP range → trellis tiles 48-wide (7 geometry families, matching ours
   modulo width).
6. **BF16 source (live):** `/mnt/<nvme-disk>/b70-bf16-qwen38-20260926/` —
   18/18 shards, 0 symlinks, 55,563,006,776 B; HF tree `1d4bf0f2ff60…` (sha256s
   available for verification). T-FORCE `Qwen3.8-27B-BF16/` confirmed dead-symlink farm
   (targets `/mnt/<models-disk>/qw38-bf16/…`, gone). Seagate arm-C disk dead (campaign README);
   arm-C artifact copy lives on NVMe: `b70-probe2-armc-copy` (29 GB, in use by probe 2A);
   `b70-autoround-qwen38-w4a16-20260926` (19 GB) also resident.
7. **Tooling:** host python has no auto_round / gptqmodel / exllamav3 (import checks).
   AutoRound 0.14.2 lives in the champion container; arm-C recipe + patches:
   `campaigns/2026/autoround-vs-exl3/ar-work/convert_armc.py`
   (`corpus` mount per `campaigns/2026/autoround-vs-exl3/convert_autoround.sh`, line 14);
   conversion log `results/2026/09/autoround-vs-exl3-20260926/armc-convert-attempt2.log`
   (calib 128 samples, ignore-list embed/lm_head/mtp/visual, in_proj_a/b auto-skip).
8. **E1 corpus (live):**
   `results/2026/09/b70-improvement-probes-20260926/shortlist-rebuild-ours-20260926/corpus/`
   — `corpus-b70docs|b70lab|projects|sessions.txt.gz` + MANIFEST.tsv, 19 MB.
9. **Battery harnesses:** `measure.py` (6-case exact battery: arithmetic/json_schema/
   factual/logic/code_execution r1/r2 + replay) and `task_battery.py` (host-executed
   bugfix/codegen) in `results/2026/09/autoround-vs-exl3-20260926/`; incumbent evidence
   `exl3.jsonl` (6/6, code_execution 14 ×2; 24 rows) and `task2.jsonl` (8/10).
10. **exl3xpu load path (from campaign log `exl3-std-65536-g90.log`):** vLLM
    0.26.1.dev0+g568afb3a1, quantization=exl3, speculative qwen3_5_mtp n=3, fp8 KV,
    cudagraph FULL_DECODE_ONLY, capture sizes to 64. Fork source not on host
    (`~/vllm-src` is stock vLLM; no exl3/trellis files); image
    `ghcr.io/0xsero/exl3xpu@sha256:21412bdd…` is the only source location.

**Derived numbers used in the plan** (all from the above): round-budget model reproduces
the architect's 16.68 GB as body 12.2295 + 2×0.9541 head + 2.5428 embedding; deltas
−3.041 (body3), −1.520 (body3.5), −0.318/head-read (head 6→4), −0.053 (mtp 4→3); KV
ctx×32 KiB; ideal ratios 1.219×/1.283× (r_h=2) and 1.193×/1.302× (r_h=4); 262K-ctx
erosion 1.283→1.192×.

PROVISIONAL — NOT FOR PUBLIC HEADLINE. No GPU touched, no artifact modified, no
authority edited, nothing downloaded except public metadata/header bytes into /tmp.

---

## Addendum 2026-09-29 (post-RL-089) — A3 answered: width 48 UNSUPPORTED; Task A route re-decided

Evidence incorporated: orchestrator relay + `research/2026/upstream-symint4-30defect-20260929.md`
§4 (ledger RL-089). Three-level proof, read and accepted: (1) source
`csrc/exl3_ops.sycl:760` — `exl3_supported()` true only for (K=4,cb=2)/(K=6,cb=2);
K∈{2,3,5} template bodies exist but sit behind `#ifdef EXL3_ALL_CODEBOOKS`, and
`docker/Dockerfile` sets no `EXL3_FLAGS` → default build; (2) binary census of `_C.so`
extracted from the pinned image `ghcr.io/0xsero/exl3xpu@sha256:21412bdd…` (docker
create+cp, never started): only `ILi4ELi2E` (470 sites) + `ILi6ELi2E` (478); zero
`ILi3ELi2E`/`ILi2ELi2E`/`ILi5ELi2E`; (3) artifact demand matches my §A3 census exactly
(`3.00bpw` needs width 48 in 400/401 trellis modules; `SC_3.00bpw_H4` needs K∈{2,3,5}).
Consequence per the dig: a 3bpw artifact **loads today** via the Triton fallback
(`exl3xpu/ops.py:78-84` → `tk.gemm_small` + Hadamard wraps) — functional, but decode
never engages the bit-exact fused ESIMD path. My A3b "no silent fallback" rule
(correctness-validation §2.2) treats that as undeclared-path serving: **the 13.820 GB
download stays blocked as a speed route.**

Additional constraint the dig adds to the price: **int8-prefill is K∈{4,6}-only**
(`exl3_ops.sycl:658`) — a 3bpw artifact's prefill falls back to reconstruct + oneDNN
fp16 GEMM, giving up the recently-won prefill gains (128K 1234→1415 tok/s class).
The sub-4 prize is therefore **decode-only**, and any A/B must measure prefill as a
possible regression axis, not assume neutrality.

### A-routes under the real constraint

| Route | Verdict | Why |
|---|---|---|
| Prebuilt `3.00bpw` on pinned image (Triton decode) | **NO-GO as speed route** | Not the bit-exact ESIMD path; silent fallback; author's own perf notes say unlikely to beat 4bpw; plus the int8-prefill loss. Retained below only as a *quality probe vehicle*, explicitly labeled Triton-path |
| Prebuilt `SC_3.00bpw_H4` | **DEPRIORITIZED** | Needs K∈{2,3,5} — strictly more kernel work for 0.372 GB less (13.452 vs 13.820 GB measured); revisit only after K=3 alone is proven |
| Requantize-from-4bpw | **REJECTED** (unchanged) | §A2 item 4 stands |
| Re-export from on-host BF16 at 4/6/4 mix | unchanged | Nothing to do with the kernel gate; still the reserve route for custom mixes |
| **K=3 extension rebuild** (`EXL3_FLAGS=-DEXL3_ALL_CODEBOOKS` + K=3 `nt_of` entry + owned bit-exactness test) | **CONDITIONAL GO** — gated on probe 1 + quality-first, below | The only route that preserves the ESIMD decode path at 3 bpw |

### Priced honestly: the K=3 extension program (kernel-engineer scope)

Deliverables, cost, and risk — no sugar-coating:

1. **K=3 `nt_of` strip entry** (`exl3_ops.sycl:181-183`): currently special-cased K=4/K=6
   from B70 sweeps; the `K<=4` branch would give K=3 M=1→8 / else 4 — plausible but
   **unswept**. Needs a small GPU microbench sweep (M=1..8 GEMV cells per candidate
   strip config) to pick the strip honestly; one bounded window.
2. **Image rebuild** with `EXL3_FLAGS="-DEXL3_ALL_CODEBOOKS"`: instantiations exist in
   source, so this is a build-flag + table-entry patch, not new kernel authoring — but
   compile time and `_C.so` size grow (today 2,936,272 B with 2 codebooks; K∈{2,3,5}
   roughly doubles+ the instantiation count). Build runs on the isolated daemon
   (data-root NVMe) — **no GPU touch** — but it is CPU-parallel-heavy: schedule at low
   parallelism with orchestrator approval or in a gap between probe windows; probe 2A
   owns most cores today.
3. **K=3 bit-exactness test — WE OWN IT** (none upstream): compare ESIMD K=3 GEMV/GEMM
   against the in-repo slow-path reference `exl3_reconstruct` (`:545`, covers K=3) and
   the Triton fallback, on random codes + the real 3bpw tensors, deterministic across
   two independent runs. Mandatory before any serving claim — the ifdef-guarded K=3
   bodies are **unverified upstream** (no in-repo test exercises K=3; compile success ≠
   correctness). This test is upstreamable to `0xSero/exl3xpu` (same contribution
   pattern as the draft-side INT4 work).
4. **int8-prefill gap accepted** (`:658`): 3bpw prefill = reconstruct + oneDNN fp16
   GEMM. Measured as a regression axis in the A/B; extending int8 to K=3 is explicitly
   out of the first program (more kernel work, prefill-side payoff only).
5. **Prize, honestly:** decode-only 1.219–1.224× ideal (~1.19× at 262K ctx), zero
   prefill benefit with a plausible prefill cost. On the adopted route's uncorrected
   chunk-rate class that is the largest decode lever left after MTP — worth exactly one
   bounded kernel program **if and only if** the two gates below pass.

### Interaction with probe 1 (the decider) — pre-committed matrix

Probe 1's attribution buckets decide whether bytes are the wall. Pre-committing now so
the verdict converts to a decision without re-litigation:

| Probe 1 verdict | Sub-4 disposition |
|---|---|
| **Byte/near-floor** (weights streaming dominant; effective BW in the 430–495+ GB/s measured-champion class) | **GO**: byte reduction is exactly the lever → quality-first probe, then the K=3 program |
| **Reconstruction-bound** (trellis dequant bucket dominant) | **HOLD**: sub-4 buys ~nothing until reconstruction is fixed; re-sequence behind the B6-class reconstruction work, then re-open with this plan (arithmetic unchanged) |
| **Clock/power-bound** | **HOLD**: capacity-only value; 262K already fits; no kernel program |
| **Host/graph-bound** | **HOLD**: fix the host first; the 1.22× is not reachable through that wall |

### Ordering (exact)

**Start now — zero GPU, zero daemon, zero download:**
- Kernel-engineer prep as a reviewable patch series against upstream `main`
  (`15ded2f3ad`): (a) Dockerfile `EXL3_FLAGS` line; (b) draft K=3 `nt_of` entry with
  the sweep-cell spec attached; (c) the bit-exactness test (K=3 ESIMD vs `:545`
  reconstruct vs Triton) authored to run on the real 3bpw tensors later; (d) A/B cell
  spec with prefill-regression axis. Keep the series small and rebaseable by
  construction (see rebase note).
- No 13.820 GB download, no docker build, while probe 1 is in flight.

**Waits for probe 1 verdict:**
- Download decision (13.820 GB, NVMe — peak with incumbent + 2× headroom ≈ 44.5 GB vs
  228.8 GB live-free, re-discovered at execution) — only at GO.
- **Quality-first probe** (the re-sequencing win): if GO, serve `3.00bpw` on the
  *pinned* image via the Triton fallback in one bounded window and run the A4 ladder
  rungs 2–4 (Δlogprob ≤3× incumbent; battery **6/6 incl. code_execution 14×2**; tasks
  ≥8/10), every record labeled **Triton-path, weights-quality verdict only — not a
  serving-path claim**. The weights are identical to what the ESIMD path would serve,
  so a quality failure here kills the program for 13.8 GB + one window instead of after
  a kernel program.
- Docker build (isolated daemon, low parallelism or gap) — only at GO and only after
  quality passes.

**Waits for both:**
- The full K=3 program: nt_of sweep window → build + smoke → bit-exactness test window
  → A4 ladder on the rebuilt image → same-base A/B (promotion gate unchanged: ≥+10%
  sustained p512/g128 on probe-1's corrected token-rate metric; plus no unmasked
  prefill regression >5%).

### Rebase onto a newer vLLM base (#50729): EXPLICITLY OUT OF SCOPE for this program

Fact incorporated from the dig: the exl3xpu image is `FROM
intel/llm-scaler-vllm:0.26.0-b2` (in-container vLLM `568afb3a1`, 2026-07-26, + llm-scaler
patch set) — it carries b2's GDN fixes but **predates vLLM #50729** (mamba conv-state
overlapping-copy race, merged 2026-08-17; ships in v0.28.1rc+/0.29.0/0.30.0; no XPU
image we have carries it), the best already-merged match for the zero-hit MTP4
prefix-cache corruption. Coupling a base rebase into the K=3 rebuild would change two
variables at once (kernel coverage + engine base) and confound both the speed A/B and
any correctness delta — prohibited by the one-variable rule this fleet runs on. The
sub-4 A/B must run artifacts on the **same image lineage**. The #50729 question is a
separate decision with its own already-spec'd discriminator (RL-089 decision table:
novelty-cell rerun on a #50729-containing build, zero-hit 5/6 → 0/6 expected) and its
own build program (no XPU image carries the fix; llm-scaler has not rebased either).
Practical implication recorded: keep the K=3 patch series minimal and rebaseable so a
later base-jump can carry it without rework.

### Updated Task A decision block (supersedes the original block above)

- **Decision:** NO-GO on prebuilt sub-4 variants as speed routes on the pinned image
  (Triton fallback = silent-path serving; SC needs strictly more kernel work for
  0.372 GB less). **Conditional GO** on the K=3 extension program behind two gates:
  probe 1 attribution = byte/near-floor, and 3bpw weights passing the quality ladder
  on the labeled Triton path. Quality-first ordering bounds the worst case at
  13.820 GB + one GPU window.
- **Cost:** prep 0 B/0 GPU now; at GO: 13.820 GB download + 1 quality window + kernel
  program (1 sweep window + 1 build on the daemon at low parallelism + 1 test window)
  + ladder/A-B windows. New image ≈ +32–35 GB + build cache on the NVMe data-root —
  budget with live discovery; fits the 228.8 GB live figure with >2× margin.
- **Gates:** probe-1 matrix above → Triton-path quality (rungs 2–4, labeled) → ESIMD
  bit-exactness vs `:545`/Triton (deterministic ×2) → full A4 ladder on the rebuilt
  image → same-base A/B ≥+10% sustained decode with prefill regression ≤5%.
- **Cheapest falsifying probe:** unchanged winner — probe 1's attribution bucket
  (already running; zero marginal cost; converts directly via the pre-committed matrix).
  Second: the Triton-path quality battery.
- **Blocked on:** probe 1 (hard gate); probe 2A conclusion (CPU contention clears the
  daemon-build slot); kernel-engineer assignment for items 1–3; upstream-researcher
  only for upstreaming the test afterward.

(Task B is unaffected by this addendum: its gate remains probe 2A's kernel-vs-weights
verdict, and none of the kernel findings above bear on the margin-objective spec.)
