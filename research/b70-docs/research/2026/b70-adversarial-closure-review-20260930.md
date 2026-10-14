# Adversarial closure review — EXL3 decode record (RL-098/100/101/102 + C2 repair)

> **PROVISIONAL — NOT FOR PUBLIC HEADLINE. PRIVATE WORKING FILE.**
> Reviewer: b70-adversarial-architect (independent; not author of any record under review).
> Date: 2026-09-30. Method: static + arithmetic only (CPU-only; no GPU, no builds, no loads).
> Scope: attack premature-closure risk in the EXL3 Qwen3.8-27B decode record.
> No edits to shared authorities; no commits.

Evidence read in full: `results/2026/09/b70-b6-microprobe-20260929/RESULTS.md` (RL-098),
`results/2026/09/b70-b6-mb4-ab-20260929/RESULTS.md` (RL-101),
`results/2026/09/b70-b6-sweep-20260929/RESULTS.md` (RL-102),
`results/2026/09/b70-k3-completion-20260929/RESULTS.md` (RL-100),
`results/2026/09/b70-probe1-exl3-rate-20260929/` (RL-090),
`results/2026/09/b70-c2-repair-20260930T085200Z/` (C2, all JSONs),
retained source `/mnt/<nvme-disk>/b70-k3-build-20260929/src/csrc/*` and
`/tmp/b70-k3-exl3-source-20260929/` (upstream `0xSero/exl3xpu`), ledger rows RL-098…RL-104,
and the 4bpw/3bpw artifacts' safetensors headers.

---

## 0. Verdict table (the six attacks)

| # | Attack | Verdict |
|---|---|---|
| 1 | "63 % of pin is in-kernel slack, geometry-insensitive" | **Measurement ESTABLISHED; "geometry-insensitive" WRONG; "in-kernel" PLAUSIBLE-UNPROVEN; the 63 %/57–63 % band-matching narrative is WRONG in construction** |
| 2 | "NT family-confined" / "MB=4 not the mechanism" | **MB=4 ESTABLISHED (as a negative); NT WRONG as scoped** — the production path's NT, the split-K thread target, and the GRF mode were never varied |
| 3 | C2 A/B power for "3–4 % slower" | **"no measurable win" ESTABLISHED; the 3–4 % magnitude PLAUSIBLE-UNPROVEN (n=2 candidate arms, one excluded sample below the incumbent median, three mutually inconsistent headline verdicts)** |
| 4 | K=3 closed at 0.671× | **Adoption NO-GO ESTABLISHED; the named lever ("P2 strip sweep") is arithmetically incapable of closing the gap — needs +51 %/+67 %, the legal sweep space is bounded to {4,8} and its sibling measurement is ~±10 %** |
| 5 | Any single-card >2× mechanism | **No >2× latency mechanism exists in the record's arithmetic (max ~1.3–1.4× combined). The demonstrated >2× single-card paths are aggregate batching (upstream C16 = 5.8× C1; zero Cn cells in our record) and dual independent stacks (2× aggregate, no kernel work). TP2 is load-blocked (RL-103).** |
| 6 | Cheapest overturning experiment | **R1 below (target-threads sweep, ~10 min GPU, existing harness, runtime env var) — see ranked list §7** |

Unaccounted-for arithmetic found while checking: **the record's round byte budget over-counts by ~2.4 GB
because it uses the artifact size as a traffic model; the true per-round trellis traffic is ~14.5 GB, not
16.86 GB** (details §1.3), and **the round contains an unmeasured 6–10 ms bucket whose composition decides
whether the "79 tok/s ceiling" is real** (§5).

---

## 1. Attack 1 — is "63 % of pin floor" a measurement or a floor artifact?

### 1.1 What is actually measured (ESTABLISHED)

RL-098 §3 measures, per family, `trellis.numel() × 2 B / XPU-event time` under graph replay
(`b70-b6-microprobe-20260929/RESULTS.md:77–102`). The pass sum (13.126 GB, counts qkvz×48, out×48,
qkv×16, o×16, gate_up×64, down×64, head×1) over 34.46 ms is a genuine kernel-level measurement with two
independent methodologies agreeing ±5–8 % (`RESULTS.md:106`). Recomputing that sum from the §2 table
reproduces 13.126 GB (±0.7 %, rounding in the 2.63/2.62 MB and 89.3/44.6 MB entries). That part stands.

### 1.2 The floor is a spec number, not a measurement (PLAUSIBLE-UNPROVEN as "the floor")

608 GB/s is `256-bit × 19 Gbps` — a class/spec figure, correctly labelled as such
(`research/2026/b70-breakthrough-architect-20260928.md:664`, cited by RL-098 §1 `RESULTS.md:33`).
RL-098 then recalibrates to a *measured* achievable read reference of 521 GB/s (`sum()` of 1 GiB fp16,
85.7 % of spec, `RESULTS.md:119`) and reports "% of achievable" everywhere. That is good practice.

Two residual flaws in the floor choice:

1. **The 521 GB/s anchor is one trivially-parallel kernel (`sum`)**, and oneDNN's own GEMV only reaches
   421–430 GB/s (`RESULTS.md:121–122`). So the "achievable" ceiling used to compute %-of-achievable is
   *above* every real GEMM/GEMV-family rate measured on this stack. Percentages of 521 are therefore
   optimistic; percentages of the ~430–480 GB/s that the best *linear* kernels actually reach are not
   reported. The stated 73.1 % of achievable for the M=4 pass is closer to ~89–92 % of the best
   *linear-class* rate already demonstrated (in_proj_qkvz 478, lm_head 508 at M=1).
2. **"85 % of pin" ≈ "99 % of measured achievable"** (`RESULTS.md:130`) — the GO threshold is at the edge
   of physics for this stack. The consequence is correctly hedged in §158, but the *headline* number
   requested by the mission ("63 % of the 608 GB/s pin floor") is the least honest of the three framings
   and should not travel alone.

### 1.3 The denominator is weights-only, and the round's byte premise is wrong by construction (WRONG)

The metric definition is explicit: achieved GB/s = **trellis weight bytes only** (`RESULTS.md:9–10`), timed
over a region that also contains `had_in`, the GEMV/DPAS split-K, and `had_out`
(`csrc/exl3_ops.sycl:306–323`). For the most-split families the un-counted traffic is not negligible:
down_proj writes and re-reads P×M×N×4 B = 1.47 MB per call at P=18 (`b70-b6-sweep-20260929/RESULTS.md:211`),
so its true traffic is ~47.5 MB vs 44.56 MB counted → the published per-family GB/s understates the true
achieved bandwidth by ~5–6 % on that family, and the pass total (13.126 GB) understates the true moved
bytes by ~0.5–0.6 GB. The pass is therefore **~76 % of measured-achievable, not 73 %** — a correction that
*weakens* the "slack" framing slightly and should be stated.

Far larger is the **round** byte premise. Probe 1 derived "one-read budget ≈ 16.86 GB = 15.7 GiB artifact"
(`b70-probe1-exl3-rate-20260929/RESULTS.md:117–121`). Enumerating the artifact's safetensors headers shows
this is a coincidence, not a model:

| class | bytes (4bpw artifact) | read per round? |
|---|---:|---|
| main-model trellis (64 layers) | 12.164 GB | yes, ×1 (M=4 verify) |
| `lm_head.trellis` (6 bpw) | 0.954 GB | yes, ×1 verify + ×3 drafts (pruned: ×0.264) |
| `mtp.layers.0` + `mtp.fc` trellis | 0.212 GB | ×3 (one per draft step) |
| `embed_tokens` BF16 [248320,5120] | **2.543 GB** | **no** — 1-token row gather |
| vision tower BF16 (`model.visual.*`) | **0.922 GB** | **no** — unused in text decode |
| small unquantized (in_proj_a/b, conv1d, norms) | ~0.09 GB | yes (~0.5 %) |

Two independent checks confirm the round model. First, the pruned draft head **is** active in every measured
cell — the pinned image's startup logs say so: `exl3: MTP draft head uses 512 of 1940 vocab blocks (26.4 %
of lm_head)` (`b70-probe1-exl3-rate-20260929/server-startup.log:58,70`; identically
`b70-k3-completion-20260929/p4/server-b-startup.log:59,71`), matching
`exl3xpu/vllm_plugin.py:237–238,299`. Second, the residual arithmetic only closes with the drafts included:

```
round (tight window)                       44.10 ms
- M=4 verify pass (RL-098, measured)       34.46 ms
= residual                                  9.64 ms
  Model A (drafts read MTP ×3 + pruned head ×3):
    3 × 0.212 GB @ ~446 GB/s = 1.43 ms
    3 × 0.252 GB @ ~508 GB/s = 1.49 ms
    -> draft phase                         2.92 ms  -> 6.7 ms unexplained
  Model B (draft head NOT re-read):        1.43 ms  -> 8.2 ms unexplained
```

So the honest round byte budget is **≈14.6 GB**, and the effective round rate is **≈330 GB/s = 54 % of
spec / 63 % of the 521 anchor — i.e. *worse* than the M=4 pass it is supposed to corroborate**. The claim
"the M=4 pass reproduces probe 1's whole-round 57–63 % band, therefore the loss is in-kernel"
(`RESULTS.md:139`) is therefore **an artifact of two different denominators happening to land on 381 vs
382 GB/s**. It is not evidence. (Both numbers are reproducible; the *inference* built on their agreement is
not.)

### 1.4 "In-kernel" — supports and the missing split (PLAUSIBLE-UNPROVEN)

Supports: PMU shows ccs0 88–100 % busy with no idle bubble and gt-c6 residency 0
(`b70-probe1-exl3-rate-20260929/RESULTS.md:108–114`) — the time is spent *on the compute engine*, not in
host syncs or engine idle. That is established.

What is **not** established is *which* on-device work owns it. Probe 1 recorded the kernel-name split as
unresolved (`RESULTS.md:84–95`), and RL-098 §7 skipped the tracer as "not worth the remaining timebox"
(`RESULTS.md:162`). The M=4 pass itself is a bare-op measurement, so "the 34.46 ms is EXL3-linears" is
sound for *that* term; but the round's 6.7–8.2 ms residual is unattributed, and it is squarely inside the
same ccs-busy window. Calling the whole loss "in-kernel" is a two-term argument where only one term was
measured.

---

## 2. Attack 2 — was "geometry-insensitive" / "MB=4 not the mechanism" closed properly?

**Short answer: no. The sweep varied one column-strip axis of a kernel the production route does not use,
while three axes of the kernel it does use were never varied at all.**

### 2.1 What was swept (ESTABLISHED)

RL-102 varied `EXL3_DPAS4_NT ∈ {2,8}` vs 4 on the **MB=4** DPAS instantiation, 4 families, both pass
orderings (`b70-b6-sweep-20260929/RESULTS.md:145–172`), and concluded FAMILY-CONFINED G: one family
(in_proj_z, NT=2) +10.5/+9.5 % on the primary metric, no other family ≥+8 %, and the guards assigned the
residual to "M=4 4-row intensity" (`RESULTS.md:230–252`).

### 2.2 Four holes in the closure

1. **The production dispatch was never swept.** The adopted (pinned) route sends M∈{3,4} to
   **MB=8** — the MB=4 case exists only under `#ifdef EXL3_DPAS_MB4`, which is *off* in the pinned
   generation (`/tmp/b70-k3-exl3-source-20260929/csrc/exl3_ops.sycl:138–145`); P6 made MB=4 default
   *afterwards* (`b70-k3-build-20260929/src/csrc/exl3_ops.sycl:298–300`). The MB=8 instantiation's own
   column-strip constant is a **separate define**, `EXL3_DPAS8_NT` (default 4), and it was never varied:
   the sweep's NT2/NT8 builds only set `EXL3_DPAS4_NT`, and the census confirms `DpasKernel<_,_,8,4>`
   is byte-identical across all three arms (`RESULTS.md:126–136`). The MB8 anchor measured MB8 **at NT=4
   only**. So "NT is geometry-insensitive" is a statement about an experimental instantiation.
2. **The split-K thread target was never swept — and the pinned generation's own source measures it as
   worth 8–20 %.** `exl3_ops.sycl:177–179` (present verbatim in the *pinned-generation* tree; visible as
   unchanged context in the P6 diff):

   ```
   static int g_target_threads = 1024;          // ... Measured on B70 (all linears, graph-captured):
                                                // 1024 beats 4096 by 10-20% for M=1..32 ...
   static int g_target_threads_mb16 = 1408;     // DPAS MB=8/16: plateau 1280-1536
                                                // (M=4 35.1->32.2 ms, M=16 38.7->36.0), cliff at 1664
   ```

   Our M=4 cells ran at 1408 (mid-plateau, `RESULTS.md:1` of the microprobe; P-sizing at
   `exl3_ops.sycl:292–297`), and the 100-replay median was 34.46 ms — inside the author's own bracketed
   35.1→32.2 ms range. The axis is **live at runtime with no rebuild**: `EXL3_TARGET_THREADS_MB16`
   (and `EXL3_TARGET_THREADS`) are read from the environment in the plugin
   (`b70-k3-build-20260929/src/exl3xpu/ops.py:36–41`) and `exl3_set_target_threads_mb16()` is a registered
   op (`exl3_ops.sycl:803`). RL-098/101/102 used this setter exactly once — as the P=1 marginal control
   (`b70-b6-sweep-20260929/nt_sweep.py:26,232–233,345`) — and never as an A/B axis.
   Priced prize: −2.26 ms on the M=4 pass → round 44.1 → 41.8 ms → **+5.4 % decode (63.25 → ~66.7 tok/s)**
   — comparable to the entire "first bet" (reduction patch, 1.4–2.1 ms) that the sweep recommended, at
   zero code risk.
3. **GRF mode was never considered.** The shipped configuration runs the MB=4/8/16 DPAS kernels in
   128-GRF mode; 256-GRF mode (`EXL3_DPAS_LARGE_GRF`, and the `EXL3_DPAS64_LARGE_GRF` that covers MB≥40)
   is the only path whose comment names the accumulator's register demand:
   `// 256-register mode: room for the MB x 16 x NT accumulator without spills`
   (`exl3_ops.sycl:111–123`). The C2 record's own bounds file says the register question is unanswered:
   `"grf_mode": "must be emitted native128 no spills; not proven by simulator"`
   (`b70-c2-repair-20260930T085200Z/v1/c2-bounds.json`). Under AGENTS §17.6 this is a named lever class;
   the sweep never touched it, and no ISA report was obtained for the *shipped* kernel.
4. **The guard that "assigned the majority" is logically invalid.** Guard (i) (`RESULTS.md:184–200`) reads
   "vector-at-M=4 is NOT near the M=1 vector rate ⇒ the M=4 shape owns it, not the DPAS op class". The
   observation only shows that *both* op classes are slow at M=4 — it cannot exclude "the DPAS kernel is
   uniformly ~15 % behind the vector kernel per byte at *any* M". The discriminating cell (DPAS at
   M=1/M=2, reachable in one line via the already-registered `exl3_set_vec_max_m(0|1)`,
   `exl3_ops.sycl:172,772`) was **never run**. Note also that the guard's forced-vector MR=4 arm is an
   instantiation upstream caps at MR≤2 by design (`nt_of`: `MR==8?2:K<=4?(MR==1?8:4):4`,
   `/tmp/.../exl3_ops.sycl:183`), so its collapse measures an untuned code path, not shape economics.

### 2.3 MB=4 (RL-101) — ESTABLISHED as a negative, WEAK as a positive

Down_proj +0.41 pin points (order-consistent, n=2 passes) is a decisive negative for "padding is *the*
mechanism", and no family can be more decisive about it than the largest mid-N family. That stands.
The qkv "partial" (+7.33/+2.12 per ordering → +4.72 median) is inside the *documented* pass-to-pass
spread of the MB8 arm itself (p50 wandered 85.9–93.8 µs, cv 17–21 %, `RESULTS.md:135–140`), so it is not a
certified effect; and MB=4 was A/B'd on only 2 of the 7 routed families. Family confinement is thus
asserted from a 2-cell positive/negative pair, while the record's own text says this class drifts ±5–9 %
between sessions.

---

## 3. Attack 3 — C2 timing: did the A/B have the power to call 3–4 %?

**Verdict: "no measurable win" ESTABLISHED; "3–4 % slower" PLAUSIBLE-UNPROVEN.**

Facts recovered from the record itself (all in `results/2026/09/b70-c2-repair-20260930T085200Z/`):

- Candidate arms: **n=2** per cell (down_proj 111.65/112.05 µs; qkv 76.05/76.40), incumbent **n=3**
  (down_proj 107.70/108.41/108.68; qkv 72.95/73.19/73.00) — `v2/RESULTS-V2.json`. Within-arm spread is
  0.3–0.9 %, so a 3–4 % gap is ~5–10× the within-arm dispersion: **statistically real**, and the
  temperature bias runs the *other* way (candidate arms 68–72 °C vs incumbent 70–75 °C,
  `v2/GATE-REPORT.md:20–30`), so the regression is not thermal.
- But the same A/B is reported three different ways: `+3.17 %/+4.43 %` (RESULTS-V2.json),
  `+3.84 %/+4.43 %, verdict REGRESSION` (`v2/timing-summary.json`), and
  `"timing WASH (candidate ~= incumbent, incumbent ~0-4 % faster)"` (`MANIFEST-v2.json:197`), with the
  milestone close quoting **candidate 106.5 µs** (`v2/C2-MILESTONE-CLOSE.json:34,209`) — a value that
  exists **only in the excluded o0 pair** (candidate 106.5 / incumbent 354.5,
  `v2/timing-arms/down_proj-incumbent-o0.log`). 106.5 µs is *below every retained incumbent sample*.
  If that candidate observation is valid, the candidate is not uniformly slower and the "regression"
  evaporates. The exclusion rationale (`v2/RESULTS-V2.md`) excludes the *pair* for concurrent-driver
  contamination — the contaminated member is the incumbent (354 µs = cold-clock artifact), so the
  candidate member's validity is undecided by the record.
- Mechanism: the authors themselves label the interpretation "not verified"
  (`v2/GATE-REPORT.md:26–28`), and the ISA-level claims that motivated C2 ("sync.allrd covers WAR",
  "falsifier broken") came from review, not measurement (mission brief; not contradicted anywhere in the
  record). `c2-bounds.json` records the contract as *unproven* re: GRF/spills.

**So:** the *decision* (do not integrate a patch with no measurable win, on a diagnostic-grade A/B) is
sound and needs no reopening. What is **not** established is the mechanism closure "register-binding
repair costs 3–4 %, therefore the serialization hypothesis is dead". Two distinct worlds fit the data:
(i) no bank conflict ever existed; (ii) a conflict existed and the forced placement added an offsetting
cost. Only a GRF/spill/stall measurement on the *shipped* kernel separates them.

---

## 4. Attack 4 — is the K=3 route really closed? Cost of the named lever.

Adoption NO-GO is correct on its own terms (quality PASS, performance FAIL by a factor 1.64 on the
adoption gate, `b70-k3-completion-20260929/RESULTS.md:216–227,266`). The question is whether the *reason*
given ("the P2 strip sweep remains the named next lever", `RESULTS.md:240–241,277–279`) survives arithmetic
and source inspection. **It does not.**

### 4.1 Byte arithmetic, corrected (the record understates the prize)

From the safetensors headers, 3bpw vs 4bpw: main trellis 12.164 → 9.123 GB (**exactly 0.750**), while
`lm_head` (953.5 MB), `mtp.*` (212.3 MB) and the 3.47 GB BF16 block are **identical** in both artifacts.
Artifact-level ratio = 13.820/16.861 = **0.820** (the record's "13.6 GiB" is a mis-transcription — the true
3bpw safetensor size is 12.87 GiB). The **round-relevant** ratio, with the pruned draft head verified ON,
is (9.123+0.954+3×0.212+3×0.252)/(12.164+0.954+3×0.212+3×0.252) = 11.47/14.51 = **0.790** — i.e. the
round byte advantage is **~21 %**, *larger* than the artifact ratio, because the round never streams the
3.47 GB of unchanged BF16 (embed + vision). The record's "~13 % fewer artifact bytes"
(`RESULTS.md:238`) understates the prize; the mission brief's "17 %" is the artifact ratio, not the
round-relevant one. Convention point for any reopening: quote the round-relevant 21 %.

### 4.2 Required improvement vs achievable sweep range

Using round = B/BW + T_other (T_other = non-linear/sampler/gap time). B_4 = 14.51 GB, B_3 = 11.47 GB
(pruned-head model, §1.3); round_4 = 44.1 ms; round_3 = 44.1 × 63.25/42.44 = 65.7 ms (acceptance is
measured as essentially unchanged, so the rate ratio is the round-time ratio):

| T_other | BW_4bpw | BW_3bpw | parity needs | 1.10-gate needs |
|---:|---:|---:|---:|---:|
| 1.5 ms | 341 GB/s | 179 GB/s | 269 GB/s (**+51 %**) | 297 GB/s (**+66 %**) |
| 2.0 ms | 345 | 181 | 272 (**+51 %**) | 301 (**+67 %**) |
| 2.6 ms | 350 | 183 | 277 (**+51 %**) | 306 (**+67 %**) |
| 3.0 ms | 354 | 185 | 280 (**+52 %**) | 310 (**+68 %**) |

**The requirement is byte-model-independent**: parity needs BW_3/BW_3(now) = (round_3−T)/(round_4−T)
= 63.7/42.1 = **+51 %** whatever the byte counts are, because the byte advantage cancels in the ratio.
So the refutation below does not depend on my byte model being right — only on the measured round-time
ratio 1.49. **The K=3 kernel runs at 0.52× the K=4 kernel's effective byte rate and must improve by
50–70 % to matter.**

What is available in the named sweep space?

- The K=3 geometry is `Geo<3>::G=8, TP=4, V=32, D=3`, with `static_assert(NT % TP == 0)`
  (`csrc/exl3_esimd.h:28–35,464–467`) and the patch's own asserts limiting `EXL3_K3_NT_MR*` to {4,8}
  (`b70-k3-build-20260929/src/csrc/exl3_ops.sycl:185–206`). **NT=2 — the one value that produced the
  sweep's only ≥+8 % wall-break (in_proj_z, +10.5/−9.5 %, `b70-b6-sweep-20260929/RESULTS.md:165`) — is
  structurally illegal for K=3.** The legal axis {4→8} measures, in the K=4 sibling: down ±0.4 %,
  gate_up +2.7/+3.6 %, in_proj_z −8.3/−6.8 %, lm_head −2.9/−3.9 %. Best case on one family ≈ +3.6 %.
- The patch states the defaults are **not measured for K=3 at all**: `// K3 starts from the repaired
  pattern, not a claimed measured optimum` (`exl3_ops.sycl:190–193`). So the sweep's upside is unknown —
  but it is *bounded* by the sibling measurement of the same axis on the same kernel family, and that
  bound is 5–6× too small.
- The K=3 deficit is not confined to small-M geometry: the **prefill** axis regressed 1.63× at
  identical flags (`RESULTS.md:221–224`), i.e. ~0.44× per byte through whatever path prefill uses.
  Two independent M regimes showing the same ~0.4–0.55× per-byte deficit points at the **codebook-decode
  structure** (D=3 cross-word windows, V=32, G=8, TP=4 lane packing, `planarize<NP,3,GV>` on every
  tile-row — `exl3_esimd.h:120–135,198,470–500`), not at strip width.
- **No isolated-op measurement of the K=3 kernels on K=3 tensors exists anywhere in the record.** The K=3
  numbers are engine-level only (42.44 tok/s); RL-098's k3-image cross-check ran the *K=4* cells
  (`b70-b6-microprobe-20260929/RESULTS.md:164`). The claim that geometry is the lever is therefore not
  even supported at the level of evidence the K=4 conclusions rest on.

**Answer to "is 0.671× explainable by a single mis-set compile-time constant in the P2 sweep space?" — No.**
No single constant in the swept space can supply +50–70 %: the axis is measured to move ≤±10 % (and its
one winner is illegal for K=3); the two constants that *could* move tens of percent were never in the
sweep (`EXL3_DPAS8_NT` — irrelevant for K=3's MB=4 verify but relevant for the 4bpw route; and
`EXL3_TARGET_THREADS_MB16`, worth ≤~8 % per the source measurement). K=3's 3bpw route stays closed for the
right *reason* — kernel-generation per-byte efficiency — but the *named next lever* is refuted and should
be replaced by the decode-structure question.

---

## 5. Attack 5 — is there a single-card mechanism for >2×?

Ranked, with arithmetic. Round = 44.1 ms, 2.895 tokens/round, 63.25 tok/s (n=5, 230 W).

| mechanism | evidence in record | priced effect | >2×? |
|---|---|---|---|
| Kernel slack to 85 % of pin | RL-098 §6 measured pass + floor math | +26 % max (63→79) — and that ceiling ignores the unmeasured residual bucket | no |
| Split-K target / GRF / production NT | **not measured**; source comments give 8–20 % on B70 | ≤+8 % (target-threads) at best | no |
| Depth k=4–5 (verify flat to M=8 per source comment `exl3_ops.sycl:172`) | recorded closure "flat ±1.4 %" is from the **GPTQ route** (RL-042) and does not transfer: EXL3's M-cost curve is documented *flat*, the GPTQ kernel's is not | my extrapolation of probe-1 survivals (0.79/0.65/0.46 + 0.33/0.24) gives ~+5 % at k=4–5; **unproven**, and depth is policy-locked (MTP3, §20.2) | no |
| Draft-side head bytes | pruned head (512/1940) is **already ON** in every measured cell (startup logs) | residual ~0.75 GB/round; further pruning trades coverage/quality | no |
| Head re-read elimination | head = 23 % of round bytes *only if unpruned*; with pruning the draft head is 0.75 GB | ≤2 % | no |
| Attention/KV at campaign depth | p16384 cell flat (64.30 tok/s) | ~0 | no |
| Op-class change (W4A8/INT4 decode) | closed (rate-class law, no native FP8) | 0 | no |
| **Aggregate batching** | **zero Cn cells in our EXL3 record**; upstream C16 = **365.2 tok/s** = 5.77× their C1 on the same engine+weights (`README.md:19–33`) | 5.8× aggregate, latency-sacrificing; KV-bound at ~10/16 long streams | **yes (aggregate)** |
| **Dual independent stacks** | ADR-0008 default; 2× by construction, no kernel work | 2 × 63.25 = 126.5 tok/s aggregate | **yes (aggregate)** |
| Dual-card single model (TP2) | **load-blocked**: `create_weights` raises `NotImplementedError('exl3: tensor parallel sharding not supported yet (use DP)')` (RL-103) | n/a | not attemptable |

Combining the *unmeasured* single-card items optimistically: 1.26 (kernel) × 1.05 (target-threads) ×
1.05 (depth) ≈ **1.39×**, not 2×. **Honest answer: no single-card >2× latency mechanism exists in the
record's arithmetic; the >2× paths are batching (unmeasured here, 5.8× documented upstream) and dual
independent stacks (2×, available now).**

### 5.1 The 6.7–8.2 ms unattributed bucket (and why the 79 tok/s ceiling may be an artifact)

The record's ceiling ("perfect kernels → ~35 ms → ~79 tok/s", `RL-098 RESULTS.md:156–158`) assumes the
residual bucket stays at 9.6 ms and that it is irreducible. Under the verified pruned-head configuration
the residual after the measured M=4 pass and the measured-class draft work is **6.7 ms (Model A) to
8.2 ms (Model B)** of *on-device, ccs-busy* work that has **no attribution at all** — candidates include
per-node XPU-graph replay overhead across ~257 linear calls × 3 kernels (`had_in`/GEMV/`had_out`,
`exl3_ops.sycl:307–323`), the GDN/attention kernels of a 48/64-GDN-layer model, the 248k-wide
sampler/argmax (×4 per round), and the small unquantized layer weights. If a large part of it is
*fixed per-round overhead*, the ceiling arithmetic must be rebuilt (the 9.07 ms kernel prize would face a
smaller wall and a larger fixed floor: `13.13/0.521 + 6.7 = 31.9 ms` → ~91 tok/s; whereas if it is
attention/GDN work, kernel work on the linears is worth *less* than claimed). **Neither branch is
measured, and the two branches have opposite implications for the next bet.**

---

## 6. Attack 1–5 auxiliary: what the record got right, and the one comparison that must not be made

- **The artifact/revision framing is safer than the mission brief assumes.** On the only matched class
  available (synthetic greedy, thinking-OFF, prose, cold-unique, C1), upstream's own rows are
  **47.2 → 53.85 tok/s** with accept 2.372 (`README.md`/`bench/results/2026-09-23.jsonl`,
  label `full-v1-think-off`, `full-v2-think-off`) while our cell is **63.25 with accept 2.895** — we are
  ahead. Upstream's headline C1 91.2 is prose at temperature 0.7 **with thinking on**, accept 3.42, over
  60–90 s windows with cold prefill interleaved (`README.md:19–33`) — a different content class, a
  different acceptance regime, and a window convention that cannot be converted into a round time.
  **Do not import 91.2/76.6 as a round-time or per-byte comparison against our 44.1 ms.** The one thing
  that *is* transferable: their *bracket* for the matched class (76.6) is above our ceiling projection,
  which is exactly why the ceiling must be labelled **"ceiling of the adopted pinned revision, at this
  cell's acceptance"**, never "the honest ceiling" (RL-098 `RESULTS.md:158`) unqualified.
- **Power/clock claims are not comparable across the two records.** Upstream notes "PL2-throttled during
  decode: act 2600 of 2800 MHz (230 W cap)" and "~47 W card power" in the same breath
  (`docs/PROGRESS.md:102,136`); we measured 200.3 W card / 129.3 W pkg at 230 W with no idle bubble
  (`b70-probe1-exl3-rate-20260929/RESULTS.md:114`). One of those two power statements is mis-scoped
  (47 W cannot be a sustained decode load at 2600 MHz on this card). **Cannot decide from evidence** —
  flag as an unverified cross-record claim; do not cite either as a mechanism.
- **The rejection of "the M=4 pass is bandwidth-starved by 4-row intensity" deserves one more
  observation**: upstream's own kernel series shows M=4 is *fragile in a way M=1 is not* — the
  software-pipelined-load step improved M=1 (33.2 → 32.2 ms) and **destroyed M=4 (41.0 → 70.0 ms, spills)**
  (`docs/PROGRESS.md:44–49`). A pure DRAM-intensity explanation cannot produce a 70 % M=4 regression from a
  load-scheduling change while M=1 improves. Register pressure / occupancy (memory-level parallelism) fits
  both that observation *and* our own observations (forced-vector MR=4 collapse; MB padding irrelevant;
  NT flat; cv 13–21 % with p90 ≈ 1.6–1.7 × p50, `b70-b6-mb4-ab-20260929/RESULTS.md:135–140`). The record
  labels the class "4-row intensity at weight-bandwidth-bound tiles" — a label, not a mechanism. This is the
  single most consequential unresolved mechanism question in the record, because the fix differs:
  tiling/occupancy vs DRAM scheduling.

---

## 7. Ranked reopen candidates with cheapest falsifiers

**R1 — Split-K thread target on the production M=4 pass (highest value / lowest cost).**
*Why*: live runtime env var `EXL3_TARGET_THREADS_MB16`; pinned-generation source measures the axis at
"M=4 35.1→32.2 ms" on B70 across the 1280–1536 plateau, and "1024 beats 4096 by 10–20 %" for M=1..32;
our cells ran at 1408 and never swept it. Expected ≤+5.4 % decode.
*Cost*: one isolated-op container run on the existing `nt_sweep.py` harness (setter already wired),
families {down_proj, gate_up, in_qkvz-fused, lm_head}, values {1024, 1280, 1408, 1536, 1664}, both orders
≈ 10 min GPU + 15 min CPU. Then, only if ≥+8 % holds, one matched server cell (p512/g128 n=5 @230 W,
graph replay) with the exact-answer battery first.
*Gate*: pre-declared wall-break ≥+8 % GB/s per family in both orderings (RL-102's own threshold), then
≥+3 % end-to-end on the round with `alpha`/acceptance unchanged.

**R2 — Break the DPAS-vs-vector confound at small M (decides attribution).**
*Why*: RL-102's "majority owner" conclusion rests on a guard whose logic cannot exclude "DPAS is uniformly
slower per byte than the vector kernel". One registered setter (`exl3_set_vec_max_m(0|1)`) gives DPAS at
M=1/M=2 with no rebuild.
*Cost*: one container, 2 families × M∈{1,2,4} × {vector, DPAS}, ~5 min GPU.
*Gate*: if DPAS@M≤2 ≥ 15 % behind vector@M≤2 per byte, RL-102 §6.2's "not by the DPAS op class choice" is
**WRONG** and the residual is a kernel-class problem (which changes which engineer owns the fix).

**R3 — Replace the K=3 "next lever" (cheapest decisive test of the K=3 story).**
*Why*: the K=3 kernels have never been timed on K=3 tensors in isolation; the geometry lever is
arithmetically 5× too small (needs +51 %), and NT=2 (the only winner) is illegal for K=3.
*Cost*: run RL-098's graph harness against the 3bpw artifact's trellis tensors + `k3-allcodebooks` image
(the same image already retained), M∈{1,2,4}, with `EXL3_TARGET_THREADS_MB16` swept; ~15 min GPU.
*Gate*: any family ≥0.85× its K=4 sibling's GB/s ⇒ geometry/knob story alive (resume R1-style sweep on the
K=3 builds); all families ≤0.60× ⇒ the deficit is codebook-decode structure; drop the P2 strip sweep as the
named lever and record "3bpw requires a decode-structure change, not a knob".

**R4 — Price the unattributed 6.7–8.2 ms round bucket (protects or kills the 79 tok/s ceiling).**
*Why*: the ceiling's residual term is a plug; ccs0-busy says it is on-device but nothing says whose work.
*Cost*: one launch, two cells — (a) MTP depth 1 (`num_speculative_tokens=1`) and (b) no-speculation
(`--speculative-config None` if the fork allows), both p512/g128 @230 W, graph replay, n=5. Predictions to
test: a no-spec round should land within ~1–2 ms of the M=1 isolated budget (29.44 ms + 0.954 GB head ≈
30.4 ms) if the bucket does not exist; a 37–40 ms result proves a ~7–10 ms fixed term and forces the
Amdahl shares to be re-derived.
*Gate*: pre-declare the bucket size; if ≥4 ms, re-derive the kernel-slack prize and the ceiling before any
kernel program is funded.

**R5 — C2 mechanism question (low cost, do not re-litigate the decision).**
*Why*: the decision stands; the mechanism does not. The excluded candidate sample (106.5 µs) sits below
every retained incumbent sample, and the record's own merged verdicts disagree (3.84 % REGRESSION vs
"WASH" vs "incumbent 4–5 % faster, non-overlapping").
*Cost*: rerun both arms ≥4× per cell under the per-arm fresh-process protocol with preheat, report
distributions (p05/p50/p95) not medians; plus a **demand** for the b70-isa-analyst's GRF/spill/stall report
on the *shipped* `DpasKernel<4,2,8,4>` / `<4,2,4,4>` (the record's own `c2-bounds.json` says
native128/no-spills is "not proven by simulator"). ~20 min GPU + ISA work.
*Gate*: an equivalence band (±2 %) declared before the run; a GRF/spill verdict, not a review, to close the
register/serialization class.

**R6 — Batched (aggregate) cell, or explicitly declare the objective is latency-only.**
*Why*: the only demonstrated single-card >2× on this engine/weights is aggregate (upstream C16 = 365.2 =
5.77× C1); our EXL3 record has zero Cn cells, and the "B16" label in probe 1 is a cell-series tag, not
concurrency 16 (TTFTs 0.37–7.21 s are single-request). AGENTS §18 already names the B=32 knee as a lever.
*Cost*: one server (production config), C1/C4/C8/C16 at p512/g128, 60 s windows, ~30 min GPU.
*Gate*: none needed to *record*; adoption requires the C1 gate to be explicitly waived for throughput
objectives (policy decision, not measurement).

---

## 8. Blocking-obstacle list (what must be true for the record's closures to hold)

| # | Must be true | Established? |
|---|---|---|
| 1 | The M=4 pass (34.46 ms / 13.13 GB) is the round's dominant term and is DRAM-class limited | **Partly** — the pass is measured; the *limiter* is not identified (DRAM class vs GRF/occupancy/MLP). Least established item in the record. |
| 2 | The round's non-linear + overhead floor is ~9.6 ms and irreducible by kernel work | **No** — 6.7–8.2 ms of it is unattributed on-device work (§5.1). |
| 3 | Geometry is not a lever on the production M=4 kernel | **No** — three axes of the production kernel (MB=8 NT, target threads, GRF mode) were never varied. |
| 4 | Kernel slack ≤ +26 %, so no >2× single-card latency path | **Plausible** — but built on a ceiling whose floor term is unverified (item 2). |
| 5 | K=3's deficit is sweep-tunable | **Refuted** — needs +51 %; the legal axis moves ≤~10 %. |
| 6 | The 63.25 tok/s cell is a valid reference for the adopted route | **Yes** — but only for its own class/config; it is the pinned revision, pruned draft head ON, thinking off, greedy, C1, 230 W, accept 2.895. Do not generalise it and do not compare it with upstream's 91.2. |
| 7 | Round byte traffic ≈ artifact bytes (16.86 GB) | **No** — true round traffic ≈ 14.5 GB; 3.47 GB of the artifact (BF16 embed + vision) is never streamed. |

**Least established, ranked: (1) the M=4 limiter's identity → (2) the residual bucket → (3) the
production-path geometry closure.** All three are cheap to attack and all three are load-bearing for the
"the only remaining lever is a kernel rewrite" verdict.

---

## 9. Method/process notes (would not survive an external audit as written)

1. **Sums of medians used as round-time terms.** RL-098 §6 sums per-family medians and subtracts from a
   round time; RL-101/102 document the underlying distributions as cv 13–21 % with p90 ≈ 1.6–1.7 × p50.
   Sums of medians are not means; the residual absorbs the error. Report means (or medians with an explicit
   caveat and a tail term) in any follow-up.
2. **Two different round times in the same evidence set** — 44.10 ms (tight PMU window, single warm
   request) vs 45.8–47.2 ms (n=5 cell, `b70-probe1-exl3-rate-20260929/RESULTS.md:279`). Every % saving in
   RL-098 §6 and RL-102 §6 is computed against the *faster* number. Effect is small (~3 % relative) but the
   convention must be fixed.
3. **"Geometry-insensitive" is stated more broadly than the sweep supports** (see §2.2); the word should
   have been "column-strip NT is insensitive on the MB=4 instantiation for 4 families".
4. **The K=3 artifact size is mis-transcribed** (13.6 GiB vs actual 12.87 GiB;
   `b70-k3-completion-20260929/RESULTS.md:238`) — the direction of the error overstates the K=3 penalty's
   severity as a byte story, and the correct round-relevant byte advantage (~9 %) is what any reopen math
   must use.
5. **No ISA artefact anywhere for the shipped DPAS kernels** (spills/GRF mode) while the C2 work shows a
   static GRF readout is achievable in this repo (`v1/budget.json` lineage). Under AGENTS §17.6 and this
   reviewer's checklist, any claim that depends on "no spills" is INSUFFICIENT EVIDENCE until that report
   exists.

---

## 10. One-paragraph answer to the mission's framing

The record's *measurements* are unusually good: the 34.46 ms / 13.13 GB M=4 pass, the family spread
(10.6–83.5 % of pin), the MB=4 bit-identity and negative, the K=3 quality/performance split, and the
launch/gate hygiene all hold up under recomputation. What does **not** hold up is the layer of inference
built on them: the "63 % of pin / 57–63 % band" story compares two different denominators and a round byte
budget that is wrong by 2.4 GB; "geometry-insensitive" was concluded without ever varying the production
kernel's own geometry or its two documented scheduling knobs (one of which the kernel author measured at
8–20 % on exactly this pass, and which is a runtime env var in our image); the "M=4 4-row intensity" label
is a label competing with a GRF/occupancy explanation that our own numbers and upstream's spill cliff both
favour; the K=3 "next lever" cannot deliver the +51 % that break-even needs and its only winning value is
illegal at K=3; and there is a 6.7–8.2 ms on-device per-round bucket that nobody has attributed and that
gates the 79 tok/s ceiling. None of this overturns the adoption decisions — 4bpw stays the route, C2 stays
out, 3bpw stays rejected — but it does overturn the *shape* of the remaining problem: the biggest
unpriced item is not "the DPAS M=4 kernel geometry" but **what the M=4 pass and the round's residual are
actually limited by**, and both are answerable for ~30 minutes of GPU time with harnesses that already
exist.

---

### Evidence index (files touched for this review)

- `results/2026/09/b70-b6-microprobe-20260929/` — RESULTS.md, harness_microprobe.py, harness_graphprobe.py, JSONs
- `results/2026/09/b70-b6-mb4-ab-20260929/` — RESULTS.md, mb4_ab.py, symbol-census.txt
- `results/2026/09/b70-b6-sweep-20260929/` — RESULTS.md, nt_sweep.py, census-all.txt, sweep/guard JSONs
- `results/2026/09/b70-k3-completion-20260929/` — RESULTS.md, p4/server-b-startup.log
- `results/2026/09/b70-probe1-exl3-rate-20260929/` — RESULTS.md, server-startup.log, requests/cells JSONL
- `results/2026/09/b70-c2-repair-20260930T085200Z/` — MANIFEST-v2.json, v2/RESULTS-V2.{json,md},
  v2/GATE-REPORT.md, v2/C2-MILESTONE-CLOSE.json, v2/timing-*.json, v1/c2-bounds.json
- `/mnt/<nvme-disk>/b70-k3-build-20260929/src/{csrc,exl3xpu}` — exl3_ops.sycl, exl3_esimd.h, ops.py
- `/tmp/b70-k3-exl3-source-20260929/` — README.md, docs/PROGRESS.md, bench/results/2026-09-2{2,3}.jsonl,
  models/qwen3.8-27b-exl3-4.00bpw/model.yaml, csrc/exl3_ops.sycl (pinned-generation dispatch)
- Artifacts (read-only header enumeration): `b70-exl3-qwen38-4bpw-20260926`, `b70-exl3-qwen38-3bpw-20260929`
- Ledger rows RL-098, RL-099, RL-100, RL-101, RL-102, RL-103
