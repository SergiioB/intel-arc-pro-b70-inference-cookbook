---
title: "ARCHITECT decision — P4 acceptance defect mechanism, next five points, withdrawals, disk discipline"
type: plan
date: 2026-09-21
status: final
---

# ARCHITECT decision — 2026-09-21 (second pass, post-probe)

Inputs: `worklogs/P4-20260920.md` §1–§14, `worklogs/CLAIMS-AUDIT-20260921.md`,
`worklogs/ARCHITECT-next-points-20260921.md`, phase files P2/P3/P5/P6/P7/P8/P9,
`scripts/tmp/p4-champion-p8192-ab-20260921.sh`, `scripts/tmp/p4-mtp-client.py`,
run dirs `p4-champion-ab-p8192-n5-…-{113616Z,114444Z,121436Z}`. Read-only session;
no GPU work run. All numbers PROVISIONAL — NOT FOR PUBLIC HEADLINE.

## 0. Two evidence-path corrections to make while writing the next worklog

1. **§14 cites the wrong run dir.** The §14 table (baseline 55.61/0.5369, pair
   80.32/0.6949, +44.4 %) is the content of `…-20260921T114444Z/comparison.json`.
   The cited `…-20260921T113616Z` is an aborted attempt whose comparison.json has
   **n=0 in both arms** (11-line client logs). Fix the path in P4-20260920.md §14.
2. The running repeatability control is `…-20260921T121436Z` (`BOTH=1`, both arms
   patched). When it lands, its acceptance numbers belong in §14's table as the
   nondeterminism bound.

## 1. The P4 acceptance defect — mechanism ranking

The facts that constrain any mechanism (verified in the harness and run dirs):

- **No request seed is ever set** (`p4-mtp-client.py` builds the request body with
  temperature 0.7 / top_p 0.8 / top_k 20 / presence_penalty 1.5 and no `seed`
  field), so every request's target sampling *and* draft sampling draw fresh RNG.
- **Prompt content is constant across runs**: a fixed repetitive filler sentence
   tiled to length, differing only in a 16-hex-char leading nonce. Prompt-content
   variance is therefore nearly nil; the prompt class is "repetitive synthetic
   filler" (which itself inflates acceptance stability — one more reason the
   baseline band is tight).
- **Containers are ephemeral** (`docker run` mounts only the model dir and three
   patch files; no cache volume): inductor/XPU-graph compile state cannot persist
   across arms or runs. Stale-compile-artifact transfer between runs is
   structurally excluded; only *in-process* reuse effects remain possible.
- **The variance is arm-asymmetric**: baseline acceptance stayed in 0.53–0.58
   across four independent runs; the patched arm read 0.5529 / 0.5547 / 0.6949.
   Under iid binomial at n≈790 draft tokens the 0.553→0.695 move is ~8σ, so the
   variance is *not* per-event noise — it is sequence-level (autocorrelated)
   variance with effective n≈5.
- All three patched runs delivered +37…+46 % throughput, so the patch *applied*
   in all of them (a silently failed patch would reproduce the local-image nulls,
   ≈0 %).

### Ranking

**M1 — stochastic sampling over an INT4-noised draft distribution (most likely).**
The pair quantizes the draft's LM head; its logits acquire quantization jitter and
a flatter top-k set. Draft-token choices near sampling boundaries flip with the
RNG draw; one early flip re-routes the whole autoregressive continuation
(amplified by the stateful presence penalty 1.5), so acceptance moves
sequence-wide. The baseline's sharper BF16 draft is insensitive to the same seed
variation — which is exactly the observed arm asymmetry. Prediction: with fixed
seeds the per-seed acceptance is reproducible; variance across seeds is large for
the patched arm and small for baseline.

**M2 — nondeterministic INT4 kernel in the new draft path (the correctness-shaped
one).** `xe_gemm_4bits`/grouped-GEMM persistent scheduling uses atomic work
stealing; any split-K/atomic accumulation in the draft head path would jitter
logits for identical inputs. Prediction: fixed seed + greedy + fixed prompt still
yields different emitted tokens across repeats. Note the champion's *target* body
is already INT4 and shows tight acceptance, so if this exists it is specific to
the newly quantized draft head at M=1/small-M shapes. This is the mechanism that
would block upstreaming outright.

**M3 — XPU-graph/compile replay variance in-process (lower).** Graph capture
freezes launch order; if capture happens against a warmup-shaped allocation and
replays against different addresses, numerics can shift run to run. Excluded as a
*cross-run* stale artifact (no persistent cache), but a per-launch capture
lottery is possible; the deciding experiment below bounds it for free.

**M4 — patch-application variance (lowest, but nonzero hygiene debt).** The
patches ran on a source generation they were not anchored to (audit defect 5).
Throughput consistency says they applied each time; but a partially applied pair
(e.g. head applied, MTP-body silently skipped) would masquerade as an acceptance
shift with unchanged traffic gain. The deciding experiment's head-only/MTP-only
arms resolve this simultaneously.

### The single deciding experiment (beyond the running BOTH control)

The running control answers "is the patched config's acceptance stable
arm-vs-arm when nothing differs?" It cannot separate seed from kernel. The
decider is a **seeded determinism decomposition**, one client change
(`--seed N` → `body["seed"]=N`; `--nonce HEX` → fixed prompt bytes), four phases,
all on the champion stack, patched arm unless stated (~1.5–2 h GPU 1):

- **A — determinism core:** greedy (temp 0), fixed nonce prompt, 3 in-process
  repeats + 1 fresh-container repeat. Hash emitted token IDs; bracket acceptance
  counters. *Identical everywhere* → engine deterministic at fixed input; M1
  confirmed as sampling-path variance (benign, quantifiable). *Any repeat
  differs* → M2/M3 nondeterminism: a correctness defect; patches stay local-only,
  upstreaming blocked, defect filed with the divergent hashes as evidence.
- **B — variance decomposition:** recommended sampling, seeds 1…5 (one per
  sample), two container relaunches, both arms. Per-seed acceptance reproducible
  across relaunches but different across seeds → variance = seed draw (M1
  quantified): publish acceptance as mean ± SD over seeds. Same seed, different
  acceptance across relaunches → nondeterminism (back to M2/M3).
- **C — mechanism localization:** head-only and MTP-only arms at p8192 with the
  same seeds (n=3 each). If the variance follows the *head* (logits feed the
  sampler) and the MTP-only arm is tight, the mechanism is localized and the
  one-variable head-vs-body split the phase already owes is earned in the same
  session.
- **D — honest band:** one natural-text acceptance set (fixed texts, seeded),
  because the filler class is the friendliest possible content for acceptance
  and the sampling policy requires the prompt-content class on every acceptance
  number.

Kill criteria for the P4 thread: A diverges → no publication of +37…46 % beyond
"local lever, nondeterministic draft path documented"; B/SD > ~0.05 even seeded →
the speed number must carry acceptance as an unresolved covariate (mechanism
split traffic-vs-drafting stays open) and the E0 attribution sentence stays
narrowed as §14 already wrote it.

### What P4 must show before publishing or upstreaming

1. Mechanism attributed (A–D above); acceptance quoted as a band with seed count
   and prompt-content class, never a single scalar.
2. One-variable split (head / MTP / pair) at champion p8192 — the pair has never
   been split on the champion stack (§10 and §12 both moved it together).
3. Fresh calibrated n=5 at p512 (never had one) and p8192, measured-window
   counters, energy + named-sensor temperature telemetry, real cooldown, ABBA or
   reversed arm order; TTFT reported honestly — at p512 the pair *improved* TTFT
   ~10 % (all patched samples below all baseline), which is a finding, not
   "unchanged".
4. Greedy C1/TP1 token parity patched-vs-baseline (the patch headers' own
   validity condition) and the patches re-anchored/regenerated against the
   champion generation (vLLM 0.27.2rc1.dev77) — the AGENTS §7 violation
   regularized.
5. Wording: "client post-first tok/s" everywhere; the stack-conditionality stated
   (inert without host-overhead removal: local image, graphs off); GPU-1-only
   disclosure.
6. Only then: ledger entry, cookbook candidate, and the operator's upstream
   decision with the TP>1 caveat preserved in the patch headers.

## 2. Ordered next points (session budget ≈ 4–6 h GPU, clean host)

Ordered to *close* phases, not open new ones. Points 4–5 are zero-GPU and can
proceed while any GPU run is in flight.

1. **P4 seeded determinism decomposition + split + finalist cells** (~2.5–3 h,
   GPU 1). Phases A–D above, then (if A is clean) the fresh calibrated n=5 p512
   and p8192 with telemetry and ABBA. *Smallest falsifying probe:* phase A alone
   (~25 min). *Kill criterion:* A diverges → local-only verdict recorded, phase
   closes as documented local lever; otherwise publication path opens. *Evidence:*
   per-seed acceptance table, token-stream hashes, head/MTP/pair split, finalist
   cells with energy/temp.
2. **P3 attribution trace — size the kernel project before building it** (~1.5–2 h,
   dual GPU, K216 L5 placement). Per-stage prefill breakdown (host counting-sort
   vs IQ3→FP16 expansion vs launch vs oneMKL GEMM) at ub 512 and 1024, plus the
   per-expert-M histogram the phase file demands. *Smallest falsifying probe:* one
   instrumented ub1024 prefill with stage timers. *Kill criterion (the phase's
   own, pre-declared):* grouping+expansion+launch < ~30 % of prefill time →
   withdraw the 1.5–2.0× E0, harvest the 1.292× ub1024 config result into the
   cookbook, close P3 as config-harvested. If ≥30 % → the multi-session kernel
   project has a measured budget and a tile policy derived from the M histogram.
   *Evidence:* stage-ms table at two ubatches, expert-M distribution, go/no-go.
3. **P6 attribution — what grew at 196K** (~1–1.5 h, GPU 1, 150–180 W cap so
   cooldown discipline is achievable). Host-side `vram_mm` debugfs polling every
   2 s (needs no OV log level), timestamped against phase markers, three runs:
   ctx196608/g40 (passes), ctx196608/g128 (fails), ctx163840/g128 (control).
   Compute d(VRAM)/d(generated token) from the timeline and compare to analytic
   KV-per-token (Qwen3.8 hybrid: KV on the full-attention layers only, tens of
   KB/token) and to a prefill-workspace step function. *Decision criterion:*
   growth linear in generated tokens at ≫ KV analytic (~tens of MB/token) →
   leak-shaped, file upstream with the timeline; growth concentrated in prefill →
   workspace at long context → declare the supported boundary 163840 and close
   track 1; growth ≈ KV analytic → plain capacity, same closure. Track 2's
   deterministic draft-drop rule is derived from the same reserve-vs-context
   curve in the same session. *Evidence:* VRAM timeline CSVs + derivative table +
   boundary declaration or upstream issue draft.
4. **P2 withdrawal — zero GPU, evidence already on disk.** The stop rule has
   fired: the fusable Mamba/SSD family is 8 % of device time ≈ 0.3 ms/output
   against the 1.01 ms/output budget — a 2.4× shortfall no fusion increment can
   recover, and under XPU graphs the fragmentation it budgeted against is already
   paid down. Set WITHDRAWN, preserve content, reassign the *other* trace finding
   — MoE gemm + gemm_kernel = 73 % of device time — to **P5 as its entry
   attribution** (grouped-GEMM per-expert-M efficiency is exactly P5's object).
   Update board, ledger alias → allocated ID.
5. **P8 AWQ closure + P10 backlog — zero GPU.** P8: record the AWQ-branch
   non-recommendation (all five community "AWQ" artifacts declare
   compressed-tensors; 1.079× bytes; no iso-byte artifact exists; building one
   costs a conversion project for expected ~1.00× speed and speculative quality —
   fails the priority formula), park rotated-GPTQ and W4A8 in DEFERRED with their
   existing triggers. P10: commit the three power-claim edits, author the OV and
   Cascadia catalog records from existing evidence, retire the stale 60.13
   record, fix the five taxonomy violations. This is also the publication hygiene
   anything from point 1 will flow through.

Conditional, only if GPU hours remain after 1–3: P5's entry probe (per-expert-M
histogram + dispatch-tile inspection on the graph-captured Nemotron stack,
reusing the P2 harness, ~1 h) and P6 track 3's mixed-load reproducer (one short
request injected during a long prefill). Neither starts a build; both only arm
or kill a PLANNED phase.

## 3. Withdrawals

- **P2: withdraw now** (point 4). Its premise is measured and refuted by its own
  pre-declared rule; running more GPU on it would be re-confirming a 2.4×
  shortfall.
- **P8's AWQ branch: close as ecosystem-overtaken** (point 5). The phase's
  remaining object (iso-byte quality) has no artifact in the wild and no speed
  payoff by the rate-class law; authoring an artifact inverts the cost formula.
  Keep rotated-GPTQ/W4A8 deferred, not withdrawn — their triggers (capacity
  crisis, changed hypothesis) are legitimate reopen conditions.
- **Weakest remaining evidence base: P5.** Its 5–15 % is precedent (the FP8
  ladder), not attribution; both of its declared controls ("P2/P4 stable
  winners") are now gone — P2 withdrawn, P4 unstable; and its largest inherited
  object (K216 decode GEMV, 5.52× off floor) was already ruled below the phase
  bar (ceiling 1.119× e2e). What would justify dropping it outright: the entry
  probe (per-expert-M histogram + current dispatch inspection on the
  graph-captured stack) showing no M-regime waste — its own filed exit
  condition — or P4 closing without a stable champion-stack control to measure
  against. Until then it stays PLANNED, armed by P2's 73 % trace, and must not
  consume a session of its own.

## 4. Disk discipline — standing rules for the executing agent

Context: root hit 8.3 GB free before a 140 GB reclaim (95 GB staging + 45 GB
caches/images); root is now 32 GB, nvme 141 GB. The failure class is recurring.

1. **Nothing growable on root.** Docker data-root, HF/pip/torch caches, staging,
   and conversion outputs live on the big mounts. Before any pull, download, or
   conversion: `df` the destination, size the operation at ×2 (download +
   extract/convert), confirm the relevant env/config actually points at the big
   mount (docker data-root, `HF_HOME`, inductor cache), and keep root above a
   10 GB floor. Artifacts larger than a few GB (the 83 GiB GGUF class) never
   transit root even temporarily — root cannot hold them and the failure is
   guaranteed, not risky.
2. **Evidence is immutable; cache is reclaimable by record.** Everything under a
   `results/` root — manifests, raw logs, counters, failure records — is the
   citation base: never deleted, by anyone, outside an operator-authorized
   retention action (it is small; it is not the problem). Re-derivable bytes are
   cache — docker images pinned by digest (re-pullable; the digest is recorded in
   the registry), model artifacts that exist upstream or on another mount, pip
   caches, staging shards whose product is validated. Cache may be reclaimed, but
   only as a logged action: what, where, how many bytes, and the re-derivation
   path. Sole-copy conversion products and their source trees under `models/` are
   evidence-grade regardless of size — check provenance before deleting anything
   there.
3. **Declare the byte budget before spending it.** Every download/conversion
   records its expected bytes and the pre/post free-space check in its manifest;
   campaign cleanup (§13) includes staging teardown as a required step, not an
   afterthought; and the host-state snapshot (`docs/operations/current-host-state.md`)
   carries the current free-space picture so the next session inherits numbers,
   not memories.
