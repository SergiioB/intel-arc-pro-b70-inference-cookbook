# B70 overnight quant-format campaign plan — 2026-10-09 (operator launch pending)

**Status:** E0 PLAN, precommitted gates, execution authorized only by operator at night launch.
**Objective:** find the quant that holds quality AND delivers a large measured speedup on one B70, per
the fleet's 2026-10-09 research pass (roofline, kernel spec, lever audit, census, availability matrix,
invented ABIs). Everything below respects RL-122 (EXL3 kernel probes stopped), RL-087/092/117 (W4A16
dense-27B closed), ADR-0010/0011, and the §4/§5/§8/§9/§11 contract.

## 0. Day-time verdicts already in (small confirmation/denial tests, done)

| Candidate | Verdict | Evidence |
|---|---|---|
| Q3DT cold-tier compression (invented ABI) | **KILLED** — codec-simulated c=1.1193 mean (30 samples, L0/8/24/40/47 x gate/up/down, both +8 MiB steps); with the ABI's raw-mode fallback the realized c≈1.0000; own kill rule c>0.90 fires either way | `scripts/tmp/q3dt-census-result.json`, run in 5.8 s CPU, zero GPU. H0 per-tensor 7.29–7.96 bits/byte (down tensors 7.29–7.31 → H0/8≈0.911): the floor defeats general-purpose lossless coders on these panels; the earlier "H0/8≈0.975 defeats ANY coder" claim is withdrawn (corr. 2026-10-09 post-audit). Hot-transpose-only branch not worth a tonight slot. |
| M5X signed-5 MXFP4 recode | **KILLED for tonight** — near-pin arithmetic (5.25 bpw needs ≥547 GB/s vs 4.125@430 to match; 10% gate = 602 GB/s) | `b70-invented-quant-abis-20261009.md` §2 kill math |
| DSM dual-scale bitplanes | **DEFERRED** — no canonical BF16 source on host (only removed GPTQ repo + `Qwen3.8-27B-UD-Q4_K_M.gguf`); conversion = new ~54 GB acquisition decision | `models/qwen/qwen3.8-27b/artifacts.md` |
| TQ1_0 byte-count dispute | Resolved: 54 B/256 = 1.6875 bpw confirmed at pinned header | roofline addendum |
| Integer-DPAS vacuum | CONFIRMED at ISA level for the three audited paths (EXL3 M=4 disassembly: all DPAS `:hf`; llama.cpp SYCL: MMQ accuracy-disabled, MMVQ `__dp4a` integer but M≤8; Strata: dequant→FP16→oneMKL `:hf`). vLLM-XPU's CUTLASS int4 grouped-MoE GEMM (#553) was NOT disassembled — no claim made there. (Scoping corr. 2026-10-09 post-audit.) | census addenda |

**As-found host state (18:55-19:05):** both B70s bound to `xe`; card0 0b:00.0 / card1 0f:00.0;
renderD128/129 present; **active `llama-perplexity` PPL run PID 3911473 on BOTH render nodes
(Qwen3.6-35B-A3B-UD-Q4_K_XL Hellaswag 10042, started ~19:00 via omp) — not owned by this campaign;
tonight's GPU work must wait for it to finish or the operator must say otherwise.** Power caps: to be
re-read at launch (xpu-smi daemon returned nothing from this shell; record at start, restore at
cleanup — as-found this morning was card0 180 W / card1 230 W). My shell needs the elevated mode for
device access; tonight's runner must use the same execution path or hand commands to the operator shell.

## 1. Tonight's queue (ordered; each item precommitted, gates never lowered)

**P0 — Incumbent IQ3_XXS quality closure battery (GPU0, Strata, ~30-45 min). GATES THE NIGHT.**
Protocol: `docs/methods/correctness-validation.md` + today's precommitted gate ladder
(b70-correctness-engineer spec: 6 canaries x2 greedy gen>=64; 3x3 replay-drift gen=128; 10 host tasks
gen=256 >=8/10; 8K two-turn KV-edge check). One discarded same-shape warmup (cold-start exits 2).
Token-id files from `strata/prompts/`; detokenize via `pack/iq3xxs/tokenizer/`.
PASS → ADR-0011 open gates closed, P1 unlocked. FAIL (any canary, esp. 14-vs-30, or replay drift) →
ADR-0011 REOPENED_QUALITY_FAILURE; tonight pivots to the **UD-Q4_K_XL Strata fallback screen** and P1 dies.

**P1 — IQ2_XXS/IQ2_S experts for Flash-Next (GPU0, conditional).**
Precondition: existence sweep (running) confirms a Strata-compatible IQ2-class artifact.
Gate 0 static (dispatch, mirror <5 GiB, slots >20k) → Gate 1 canary battery (zero-defect hard stop,
NO speed numbers before pass) → Gate 2 replay/MTP → Gate 3 tasks ≥8/10 → Gate 4 matched n=5 pairs,
success = decode >53.0 tok/s (+10% over 48.03). Conversion is storage-blocked (~250 GB staging vs
~120 GB free) — artifact existence decides; no downloads over budget.

**Post-stop audit correction (2026-10-09, pre-launch):** Gate 0 and the >53.0 tok/s
bar were derived for **IQ2_XXS**, which does not exist. The existence sweep resolved:
ISTA-DASLab GSQ-RCO carries only IQ2_XS / IQ3_S / IQ3_XXS / Q2_0 (HF API verified
2026-10-09 evening). On the audit's own scaling, IQ2_XS (~1.32 MB/expert) gives
~19.2k resident slots (<20k) and a ~7.1 GiB mirror (>5 GiB) — **Gate 0 fails as
precommitted**, and the projected gain (~+10.0%) sits exactly at the success bar with
zero margin ([INFERENCE]-scaled from the audit's model). Per §2 rule 6 (never lower a
gate after seeing results), **P1 is DEAD tonight** unless an IQ2_XXS-class artifact
appears. Do not run IQ2_XS under these gates.

> **Operator change 2026-10-09 evening:** no gpt-oss models (outdated). P2 now
> runs on `noctrex/Qwen3.6-35B-A3B-MTP-MXFP4_MOE-GGUF` @ `de693c83b994`
> (21.2 GB, SN770), with the on-disk UD-Q4_K_XL GGUF as a same-model format arm.
> Authority: RQ-19 in `registry/research-program.json`.

**P2 — llama.cpp PR #29809 MXFP4 MoE decode on gpt-oss-20b (GPU1-first, or GPU0 when free).**
Build branch first (CPU-side), then download gpt-oss-20b (~13 GB, T-FORCE), then matched cells:
p512/g128 decode + 4K prefill, greedy, n=5 fresh pairs after 1 discarded warmup, stock-MXFP4-path
control on same build. Target: reproduce the author's 55.84→106.64 tok/s class ON THIS HOST.
Quality: exact-answer canary battery on gpt-oss-20b before speed claims (new model on host).


**Post-stop audit addition (2026-10-09):** the pinned #29809 capture also contains a
**Linux one-B70** data point from iki007's in-PR commits: gpt-oss-20b **tg128
53.72 → 137.7 t/s (2.56x)**, pp512 1671 → 1690 — closer to this host than the
author's Windows 11 numbers (55.84→106.64, n=5, oneAPI 2026.0, FA on). A third-party
Arc B570 run reports +40.8% tg128. In-thread caveat: early prefill numbers were
order-confounded before warmup discipline was fixed — treat prefill deltas under ~2%
as noise. Host reproduction target: the **Linux tg128 class**; the Windows pair stays
as the recorded author headline.

**P3 — INT4-G128 dual-path falsifier microbenchmark (GPU0 after P0; decisive for the
INT4-G128 streaming hypothesis only — the integer-DPAS vacuum itself is already
statically proven, and a PASS would still leave A8 parity, the RL-087/092 dense-W4A16
rejection, and the removable-wall fraction f unresolved).**
Per `b70-small-m-quant-layout-static-spec-20261009.md`: exact source pin, BMG AOT build, ISA gate
(no spills, DPAS/dp4a emission proof) BEFORE any GPU run; weight ring > L2; M=1..4; metadata-inclusive
GB/s; graph-replay-equivalent (llama.cpp: no allocation in step); 3 interleaved screen pairs then 5
fresh confirmation pairs. Hypothesis: ≥500 GB/s metadata-inclusive vs the 381 anchor. Falsified if
dominant families miss; a sub-threshold result is recorded, not passed. Single-format tonight (DSM out;
kernel-engineer confirmation pending — if he says extend, that's a second build, not tonight).

**P4 — UD-Q4_K_M 27B dense screen (GPU1 or GPU0; cheap, artifact already on disk 16.4 GB).**
llama.cpp SYCL (fleet fork @1692f9e50): exact-answer canary battery + p512/g128 decode, n=3 screen.
Question: does the unsloth-UD mixed recipe beat the record's dense GGUF arms on quality-per-byte?
(NOT a speed lever vs EXL3 at ~4.9 bpw — a quality/robustness data point; informs whether UD-class
mixed recipes deserve a P1-style Strata arm later.) Consistent with the operator's live UD-Q4_K_XL
35B-A3B PPL run — same family question.

**P5 — NVFP4 QUASAR-QAT 27B (only if sweep confirms + storage fits + time remains).**
Community-QAT lineage: exact battery FIRST (RL-087/092 method, in-head defect class), speed only after.
llama.cpp NVFP4 MMVQ exists at master; our base is 2 months stale → requires the P2 build anyway.

## 2. Rules for the night (non-negotiable)

1. One owned campaign per card; the PPL run (PID 3911473) finishes first or operator displaces it.
2. One discarded same-shape warmup per engine start; 3 interleaved screen pairs; 5 fresh confirm pairs.
3. Greedy for all quality batteries; engine-decode wall medians; actual token counts declared; single client.
4. Exact-answer canary battery before ANY speed claim on any new artifact/model.
5. Thermal: cooldown gate is unreachable on this host (RL-065/069 idle-hot wall) — declare sustained-regime
   cells instead; record caps + clocks per campaign (§5 always-on rule 4).
6. Stop rules: per-item as precommitted above; global: two attempts max per unchanged mechanism; a failed
   gate is retained with reason + re-entry condition; never lower a gate after seeing results.
7. Cleanup: stop only owned servers (Strata via dedicated dockerd socket; llama.cpp runs), restore as-found
   caps (re-read at launch; morning as-found = 180/230 W), prove cleanup, retain failures.
8. New scripts stay in `scripts/tmp/` until a promoted, sealed run justifies otherwise (§14).

## 3. Expected-value ranking if only part of the night survives

1. P0 (decides everything; already-designed, cheap)
2. P2 (highest verified-upside-per-hour: author-measured 1.91x on this exact card class)
3. P3 (the structural lever test: can an integer path actually stream ≥500 GB/s; the vacuum is already static-proven and a pass alone adopts nothing — corr. 2026-10-09 post-audit)
4. P4 (cheap, artifact on disk)
5. P1 (only if artifact exists), P5 (only if sweep + storage + time)

## 4. Inputs still pending at press time

- Existence sweep (upstream researcher): IQ2-class Flash-Next, NVFP4 QUASAR-QAT, gpt-oss revisions, Strata ≥0.1.41 rows.
- Kernel-engineer ABI pricing (DSM/M5X/Q3DT vs anchors; P3 single-format confirmation).
- Adversarial pass #2 (deepseek): KILL/CONDITIONAL/ADVANCE on the three ABIs + tonight queue.

**Post-stop resolutions (2026-10-09 evening, pre-launch):**

- Existence sweep (item 1) resolved for the queue-deciding parts: GSQ-RCO has **no
  IQ2_XXS/IQ2_S** (only IQ2_XS/IQ3_S/IQ3_XXS/Q2_0, HF API verified) → P1 DEAD per its
  addendum. NVFP4: QUASAR-QAT artifact exists; convert **#28636 merged, #27161 closed
  unmerged** (RL-126 + availability doc corrected, GitHub API verified). gpt-oss / Strata
  0.1.41 rows still unverified at press time.
- **P0 prep DONE:** the canary encoder was rewritten (two latent defects: bad
  `Tokenizer` constructor call; empty "specials" prefix that would have stripped the
  chat template) and run against the real pack tokenizer — `batt-sumsq.txt` (28 ids),
  `batt-logic.txt` (31), `batt-json.txt` (26) are created in `/work/prompts` with
  header/footer ids token-exact-identical to `batt-arith.txt`. Remaining P0 pre-flight:
  record power caps, then run the battery.
- Kernel-engineer ABI pricing (item 2) and adversarial pass #2 (item 3): still not
  returned at this correction's timestamp.

These finalize the queue at launch; the plan above is the precommitted skeleton.
