---
type: plan
date: 2026-09-21
phase: P4
title: "Correct native-MTP subgraph optimization (supersedes quarantined RL-021 concept)"
status: RESEARCH
opened: 2026-09-20
updated: 2026-09-20
ledger: RL-030 (P0 umbrella); RL-021 quarantine stays in force
depends_on: [P0]
effort: 4–8 sessions
---

# P4 — Correct native-MTP subgraph optimization

## Status log

| Date | Status change | Note / evidence |
|---|---|---|
| 2026-09-20 | filed as PLANNED | Supersedes the RL-021 quarantined prototype concept; quarantine guards stay in force |
| 2026-09-20 | PLANNED → RESEARCH | Entry inventory executed **without a GPU** from the checkpoint headers + config + pinned vLLM source. Ground truth: `mtp.*` is excluded from GPTQ by `quantization_config.dynamic`, so the MTP module is BF16 (849.40 MB); the MTP module has **no head and no embeddings** of its own — `_maybe_share_lm_head()` shares the target's `lm_head.weight` (**2.5428 GB F16**), and `propose()` applies it **once per draft token** (step 0 + `range(num_speculative_tokens-1)`) ⇒ **4 head reads of 2.54 GB per MTP4 step**. Byte budget: 0.25 ms/draft allows ≈152 MB; the artifact requires **3.39 GB per pass (22.3× over)**, and even a fully INT4 draft side is 5.6× over. **Stop rule applied — 0.25 ms withdrawn**; the quarantined 1.5 ms/draft figure is independently refuted by the same arithmetic. **Pivot recorded per the phase's own rule**: the LM head is 75 % of per-pass draft traffic and is the retargeted object (draft-side head quantization already exists as a local patch in the champion stack: `B70_DRAFT_LMHEAD_INT4=1`). Measured attribution (§6) then shows the step is **GEMM-device-bound (79.7 % of device time)** with ~300 k tiny elementwise kernels worth only ≈10 % — so subgraph *capture* is the wrong tool and elementwise fusion has a ~10 % ceiling. Worklog: `worklogs/P4-20260920.md`. |
| 2026-09-21 | prior art found — scope collapses to validation | The retargeted lever is **already implemented and measured on disk**: `results/2026/08/qwen38-pr-ab-20260818T205619Z/` is a clean three-arm A/B (same image, model, MTP4, 230 W, cache off, graphs on, probabilistic draft) whose arms differ **only** in `extra_env`; adding `B70_DRAFT_LMHEAD_INT4=1,B70_DRAFT_MTP_INT4=1` moves median agentic t2+ from 44.01 → **60.23 tok/s (+36.9 %)**. The patch docstring independently states the §3 byte model (2.54 GB head × 5 reads/step = 12.7 GB ≈ 21.3 ms ≈ 41 % of GEMM time). **So no new kernel is owed.** What remains: (a) the phase's own canonical validation clause — recommended-sampling p512/g128 and p8192/g128, p2048/g1 + p8192/g1 regressions, n=3 then fresh n=5 — because the existing number is an **agentic mixed-load screen, not a protocol cell**; (b) split the two variables (draft LM-head INT4 vs draft-MTP INT4) since the phase mandates one variable per gate; (c) an upstream/cookbook decision, since the patch is local-only and its own docstring restricts it to C1/TP1 greedy. This is the phase's named "known mechanism → collapse to validation + benchmark" branch. Worklog §7. |
| 2026-09-21 | corrective measurement at a protocol cell | Two-arm one-variable A/B on the **local image** at C1 p512/g128 (n=3, recommended sampling, server-reported 128 output tokens in every sample, warmup discarded, GPU 1, 230 W): adding `B70_DRAFT_LMHEAD_INT4=1` alone gives **+1.5 % eager (25.40→25.79)** and **−0.5 % compiled (42.76→42.55) — no material effect**; acceptance identical across arms (0.5429 / 0.5193). Compilation itself is worth ~1.68× (25.40→42.76). **So the recorded +36.9 % does not transfer to this cell** and must not be quoted as a general effect; it stands only for the 2026-08-18 agentic mixed-load screen, which enabled *both* draft patches on the champion image. Next single-variable steps: `B70_DRAFT_MTP_INT4` alone, then the pair, at this same cell. Worklog §8; evidence `results/2026/09/b70-improvement-probes-20260920`/…/. |
| 2026-09-21 | single-variable series closed at C1 (both null) | `B70_DRAFT_MTP_INT4=1` alone: **+0.7 %** (42.73→43.04 post-first, acceptance 0.5208→0.5429). With §8's head-only null (**−0.5 %** compiled / +1.5 % eager), **neither draft-side INT4 patch moves C1 p512/g128 materially, and by two independent nulls neither does the pair**. The recorded +36.9 % is therefore a **regime effect** (long-context agentic mixed load), not a patch effect — and the phase's E0 has **no C1 basis**. Stop rule applied to the short-context interpretation; the one remaining defensible step is a **long-context protocol cell** (the regime where the effect was observed), not another short-context arm. Worklog §9; evidence `results/2026/09/b70-improvement-probes-20260920`/…/p4-draftint4-ab-*. |
| 2026-09-21 | **first positive measurement — regime confirmed** | At **C1 p8192/g128** (local image, compiled, n=3, 128/128 server-reported tokens) the draft-side INT4 **pair** gives **38.71 → 42.96 tok/s post-first = +11.0 %** (out-throughput +4.4 %), TTFT unchanged. With §8/§9's p512 nulls (≤1.5 %), the lever is **regime-dependent**: inert at short context, worth +11 % at 8192 and presumably more at the agentic regime that produced +36.9 %. Caveats recorded: acceptance moved with the arm (0.5335→0.5578) so the split between traffic and drafting behaviour needs the head-only arm at p8192; still the local image, not the champion. Next: head-only at p8192, then a protocol-grade long-context cell (n=3 screen → fresh n=5). Worklog §10. |
| 2026-09-21 | **+24.2 % decode on the champion stack at a protocol cell** | Two-arm, one-variable A/B on the **champion** digest + overlay + XPU graphs at C1 p8192/g128 (n=3, 128/128 server-reported tokens): champion **55.68** → champion+draft-pair **69.14 tok/s post-first = +24.2 %** (out-throughput +8.5 %), TTFT flat, **acceptance down** (0.5753→0.5547) so the gain is traffic, not drafting. Same cell on the local image was +11.0 % — so the stack matters as much as the regime: removing host overhead (graphs/v5) is what makes the weight traffic visible. The recorded +36.9 % agentic figure is the top of this same curve, not an outlier. Still owed by the phase's validation clause: fresh **n=5** finalist at p8192, a **p512 regression** on the same stack, and the upstream/cookbook decision for the two local patches. Worklog §11; evidence `results/2026/09/b70-improvement-probes-20260920`/…/p4-champion-ab-p8192-…. |
| 2026-09-21 | **fresh n=5 finalist: +37.2 % — thread closed** | Same champion harness, fresh samples, 128/128 server-reported tokens in every cell: champion **52.99** → +draft-pair **72.68 tok/s post-first (+37.2 %)**, out-throughput +11.8 %, TTFT identical, **distributions disjoint** (baseline max 55.73 < patched min 67.49), **acceptance flat** (0.5594→0.5529) ⇒ the win is weight traffic, not drafting. This **reproduces the recorded +36.9 %** (2026-08-18 agentic screen) at a protocol cell and **exceeds the phase's E0 (1.10–1.20×) at 1.37×**. Owed before publication: p512 regression on the same stack, head-vs-body one-variable split, claims/quality audit; the two patches are local-only (operator decides upstream/cookbook). Worklog §12. |
| 2026-09-21 | **p512 regression: +46.0 % — attribution corrected (stack, not context)** | Same champion harness at C1 p512/g128 (n=3 fresh, 128/128 server-reported tokens): champion **52.83** → +pair **77.14 tok/s post-first (+46.0 %)**, out-throughput +40.7 %, acceptance flat (0.5429→0.5335), TTFT **308–315 → 281–290 ms (≈ −10 %; "unchanged" corrected per the claims audit)**. Combined with §12: **+46.0 % @p512 and +37.2 % @p8192 on the champion stack**, versus ~0–1.5 % / +11 % on the local image. The gate is therefore **host-overhead removal (XPU graph capture)**, not context length — every earlier null ran with `VLLM_XPU_ENABLE_XPU_GRAPH=0`, which is precisely the condition that hides the lever. E0 (1.10–1.20×) exceeded at **1.37–1.46×**. Owed: head-vs-body split, claims/quality audit (draft-side weight-only change with target verification preserved), operator decision on upstreaming the two local patches. |
| 2026-09-21 | independent claims audit — headlines downgraded to provisional | `worklogs/CLAIMS-AUDIT-20260921.md` (independent reviewer, 140 lines with file:line citations) examined the §11/§12/§13 numbers. **Defects it found, now recorded:** the cell is **p≈6993, not p8192** (the client sizes prompts by word count, no tokenizer calibration); the reported figure is a **client post-first rate**, so calling it "decode throughput" overstates it; the acceptance scalar in my summaries is **warmup-diluted**, and on the *measured window* it moves **−0.048 at p512**, which breaches the patch pair's own 0.03 acceptance gate (`cookbook:patches/patch_draft_mtp_int4.py`); "TTFT unchanged" is false at p512 (308–315 → 281–290 ms); and the patches ran outside the generation one of their headers names. **Minimum fix list:** tokenizer-calibrated prompts, fresh n=5, measured-window acceptance, greedy token-parity on the champion image, energy/temp telemetry. Nothing in P4 is publishable today. |
| 2026-09-21 | audit-compliant re-run: +44.4 %, and a new open defect | Calibrated against the server tokenizer (**8,189 of 8,192** tokens) with measured-window acceptance: champion **55.61** → +pair **80.32 post-first (+44.4 %)**, out-throughput +11.7 %. **But the patched arm's acceptance was 0.6949 here versus 0.5529/0.5547 in the earlier runs of the same configuration** (baseline stable at 0.53–0.58) ⇒ part of this run's gain is drafting behaviour, not weight traffic, and a run-variable acceptance in the patch's own draft path is a **correctness-shaped signal** that must be attributed before upstreaming/publishing. Magnitude is now reproduced three times (**+37…+46 %**), so the E0 question is answered; **the phase cannot close** on the audit's remaining fix list (greedy token-parity, energy/temp telemetry, patch-generation provenance). Worklog §14. |
| 2026-09-21 | **repeatability control: throughput robust, acceptance noisy** | With **both arms patched identically** (`BOTH=1`), throughput reproduced at **80.61 / 78.27 tok/s** — the same band as baseline-vs-patched — while acceptance still differed between the two identical arms (0.5906/0.6577 vs 0.6194/0.5934). So the **+37…46 % effect is a property of the configuration, not of a lucky acceptance run**, and the acceptance is simply noisy in the patched config (0.55–0.69 across five runs vs the baseline's tight 0.53–0.58). Mechanism ranking from the architect: INT4 noising of the draft logits makes near-ties flip and one flip re-routes the continuation under sampler penalty state (sampling variance), with a nondeterministic kernel as the correctness-blocking alternative still open. Decider: **seeded determinism decomposition** (greedy fixed-nonce token-ID hashes → seeds 1–5 × two relaunches → head-only vs MTP-only arms). Worklog §15. |
| 2026-09-21 | **greedy determinism: the pair is output-preserving** | Fixed-prompt greedy A/B (`top_k=1`, identical nonce, 3 tasks × 2 repeats): both arms are deterministic within themselves and produce **byte-identical output to each other** (same sha256 on all three tasks). So the run-variable acceptance is **drafting-efficiency variance, not a change in what is emitted** — the §14 correctness concern is resolved for the greedy path. Remaining limit: sampled-path distribution-preservation is by sampler construction, not locally measured. Worklog §16. |
| 2026-09-21 | seeded sampled path: consistent, weak discriminator | With recommended sampling + fixed seed, both arms produced identical hashes on all three prompts — but those hashes equal the greedy run's, i.e. the prompts denied the sampler any freedom, so this is supportive and **not** a distribution-level equivalence result. The claim-earning version is recorded as the next step: open-ended prompts, ≥30 seeds per arm, compare emitted distributions. Worklog §17. |
| 2026-09-21 | distribution-level check: consistent on 6/6 modal classes | Six open-ended prompts x 30 seeded draws per arm (one prompt produced 30 distinct outputs, so sampling genuinely diverged): **identical modal emitted class on every prompt**, analogous support sizes, set overlaps 0.56-1.00 (what 30 draws over ~7 classes gives even for identical distributions) — consistent with distribution preservation, explicitly not proof. |
| 2026-09-21 | **concurrency knee measured: 2.3-2.4x aggregate, stackable** | 8 requests per level at C1 p512/g128 on the champion stack: baseline config 48.45 -> **115.70 tok/s** (B=1 -> B=4, **+139 %**), patched config 61.86 -> **140.30** (B=4, **+127 %**) and 149.79 at B=8; all 128 output tokens server-reported at every level. Mechanism as predicted — fused batches read the weights once, so per-sequence traffic falls ~1/B. **It stacks with the draft-side INT4 lever** (patched B=4 = 140.3 vs baseline B=4 = 115.7). Not a single-stream number and never comparable to single-stream records (including this route's own 102.61 champion); per-request latency degrades with B and must travel with the claim. Worklog §19. |
| 2026-09-21 | deep knee: **172.48 tok/s aggregate at B=32** | Extended the sweep to B=16 and B=32 (32 requests per level, `--max-num-seqs 32`, patched config, every request server-reported at 128 tokens): **162.74** tok/s at B=16 and **172.48** at B=32 against 61.86 at B=1 = **2.79x**, with per-request rate falling 72 -> ~29 tok/s and first TTFT ~2.0-2.6 s. Diminishing but not saturated, so the knee's ceiling is above B=32 — the operator has deprioritised further concurrency work in favour of single-stream levers, so this is recorded rather than extended. Evidence: `p4-champion-ab-p512-n1-…T151358Z/conc-B*.jsonl`. |

## Why this decision (the reasoning)

**The thesis: the champion route's remaining overhead is the MTP draft loop, the only
prior attempt at it was invalidated for measuring the wrong computation, and the old
target number does not survive byte arithmetic — so this phase starts from an inventory,
not from the old sketch.**

1. **Why the draft loop at all.** Qwen3.8-27B GPTQ-INT4 + BF16 MTP4 sits at 102.61 tok/s
   (p512/g128). The base model is 98.4% bandwidth-bound (~31 ms/token floor at 608 GB/s
   for ~14–15 GB of weights), so the *base* GEMV has <1.5% to give — that anti-pattern is
   on record. The speculative machinery around the base is where the non-bandwidth
   milliseconds live: the claimed 1.5 ms per draft token × 4 iterations ≈ 6 ms per step.
   Attack what is not bandwidth-pinned.
2. **Why "claimed."** The 1.5 ms figure and the 0.25 ms target both come from the
   quarantined prototype's lineage (RL-021). The quarantine record shows the prototype
   omitted the actual MTP computation: constant-token placeholder, no attention /
   positions / KV, no MLP/residual, no embeddings or hidden-state fusion, argmax instead
   of the sampler. Anything timed on that shape measured a fiction. This phase's first
   deliverable is therefore a *real* per-step inventory: every op, its bytes, its state
   dependencies — then fusion/capture boundaries follow from the inventory.
3. **The byte-budget check that re-derives the target.** At 608 GB/s, a 0.25 ms draft
   step permits ~152 MB of ideal GDDR6 traffic. A large-vocabulary BF16 LM head alone
   can exceed that without reuse. If the real inventory shows more bytes, the target
   moves — or the design must demonstrate reuse (weights resident across the 4 draft
   iterations). Also, the step math must use *emitted outputs per step* `a`, not
   "accepted draft length": saving 5 ms/step predicts ~120 tok/s at a=3.5 but only ~116
   at a=4.5 — the two are not interchangeable.
4. **Why capture and compilation are verified separately.** `torch.compile` success does
   not prove Level Zero graph **replay** (P0 probe). Graph replay removes submission
   overhead only; it does not remove device kernels or intermediate traffic. If the
   inventory shows the draft loop is device-bound, capture is the wrong tool and local
   subgraph fusion is the right one.
5. **Why sampling is a hard constraint.** The champion number is defined under
   recommended sampling (Run 48). A fused path that changes penalties/RNG behavior can
   "win" throughput by breaking the definition of the thing being measured. Any result
   that wins by changing sampling is rejected outright.

**Quarantine lessons converted to requirements (from RL-021).**

- No constant-token placeholder; no omitted attention/MLP/state/embedding path.
- No argmax substitution for required sampling.
- No throughput claim before full functional equivalence.
- No reuse of disabled prototypes as production code.
- New implementation begins from pinned native source with executable tests.

**Alternatives considered and rejected.**

- *Base-model GEMV megakernel.* <1.5% available — recorded anti-pattern.
- *Deeper speculation instead (MTP6+).* The depth screen is closed (RL-026: MTP4
  optimal); re-running it without a changed implementation is a banned retune.
- *FP8 target switch for faster verify.* Dominated by INT4 on both axes per the
  rate-class law; off-mission for this phase.
- *Reviving the quarantined kernel and "fixing" it.* The invalid shape taints its
  timings; starting from pinned source with tests is cheaper than forensics.

**What would change this decision.** P0 subgraph trace showing <~1 ms/step of removable
non-bandwidth time (withdraw); upstream vLLM landing an optimized native-MTP draft path
(collapse to validation + benchmark); the byte inventory proving the LM head dominates
irreducibly (retarget to head reuse/caching work and record the pivot).

## Objective

Optimize the real native MTP computation — embeddings, hidden-state fusion, norm,
attention/positions/KV, MLP/residuals, output head, sampler, bookkeeping — via local
subgraph fusion or capture of supported static sequences. Supersedes the older design
concept **without bypassing or deleting RL-021 quarantine guards**.

## Baseline and E0 target

Baseline: 102.61 tok/s, p512/g128, GPTQ-INT4 G128 + BF16 MTP4
(`registry/model-status.json`). The historical 1.5→0.25 ms/draft-token estimate is
**unvalidated**. If four iterations each save 1.25 ms (5 ms/step total):

Δt_output = 5/a ms → ~120 tok/s at a=3.5, ~116 at a=4.5.

E0 range: ~1.10–1.20×; 120–125 tok/s is an aspirational boundary, not a commitment.

## Engineering work

1. Inventory embeddings, hidden-state fusion, normalization, attention/positions/KV,
   MLP/residuals, output head, sampler, bookkeeping.
2. Record actual per-step byte traffic and state dependencies.
3. Identify legal fusion boundaries.
4. Optimize local subgraphs or capture supported static sequences.
5. Preserve recommended sampling, including penalties and RNG behavior.
6. Keep the native BF16 head unless a separate quality campaign authorizes change.
7. Validate graph replay independently from compilation.
8. Reassess the target after accounting for LM-head traffic (152 MB @ 0.25 ms budget).

## Validation and benchmark plan

- Per-subgraph numerical comparisons; token/logit replay under fixed RNG where
  applicable; statistical sampler checks where exact identity is not appropriate.
- Accept/reject-state correctness.
- Recommended-sampling p512/g128 and p8192/g128; p2048/g1 and p8192/g1 regression
  checks.
- n=3 screen, fresh n=5 finals. Fixed MTP4; no MTP6 retune without a changed
  hypothesis.

## Risks and stop criteria

- Stop after two semantic/parity failures.
- Reject any result that wins by changing sampling.
- Withdraw 0.25 ms if the byte budget disproves it.
- Stop when exposed overhead cannot support the predeclared useful gain.
- Do not claim the base 27B GEMV megakernel as the solution.

## Deliverables

- `research/2026/qwen38-native-mtp-subgraph-optimization-<YYYYMMDD>.md`
- `results/2026/<MM>/qwen38-native-mtp-subgraphs-<UTC>/` (path truncated in transcript)
- Ledger entry explicitly referencing RL-021.

## Cookbook integration

Update the Qwen3.8 native-MTP recipe only with a complete implementation, immutable
build prerequisites, and audited sampling-preserving records. Preserve the old champion
as historical evidence.
