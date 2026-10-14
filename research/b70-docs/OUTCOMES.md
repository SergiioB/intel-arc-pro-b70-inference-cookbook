---
title: "OUTCOMES — B70 speed track day one (2026-10-10/11): what was detected, what died, what it taught"
type: outcomes
status: living
date: 2026-10-11
---

# OUTCOMES — speed track day one (2026-10-10/11)

PROVISIONAL — NOT FOR PUBLIC HEADLINE. Every row traces to a ledger id
(RL-131…RL-151) or a repo path. Authority for current question states:
`registry/research-program.json`; full narrative:
`research/2026/b70-speed-track-workflow-report-20261010.md` (runs 1–4).

## Detected — wins (registered or evidence-complete)

| What | Evidence | So what |
|---|---|---|
| **RQ-22 priced, not closed**: measured 37.30 % expert-GEMM + dequant share of the 8K prefill; fused s8 dequant+GEMM → **+33.9 % whole-prefill potential**; a plain GEMM swap caps at 1.229× and can never clear the ≥1.3× gate | RL-131; `research/2026/rq22-int8-prompt-gemm-price-20261010.md`; both price inputs validated by `research.py price` | The only prefill lever priced above its gate (RQ-25 and RQ-18 are decode levers, also priced above theirs). Redirects INT8 work to the fused design and away from a plain swap that cannot pass |
| **F1 kernel half closed, not a format verdict**: a first-pass microbench reached **24.4 GB/s vs the 500 GB/s gate** after one rewrite (1.54×). The package failed its own ISA check: no round-2 disassembly exists, so the cause of the 20× gap is undiagnosed | RL-145 (caveats in-row); `research/2026/b70-hq4-f1-evidence-20261010/ROUND2_FINDINGS.md` | Shows only that this kernel was untuned. It does **not** show the trellis is near-optimal: RL-119 measured trellis decode as the majority owner of EXL3's 37 % gap to pin, and stubbing it raised the M=4 cells 18–21 %. "Replace the trellis" stays open at kernel level; re-entry needs a disassembled kernel |
| **F0 raw trend — WITHDRAWN 2026-10-11 (RL-161)**: the AutoRound-uniform arms that read 0.06–0.10× EXL3 KL on att_l63 and gdn_l62 were stored at **8.125 bpw** (every tensor escaped to INT8), not 4-bit. The only INT4-pure AutoRound arm was 12.7× worse than EXL3; GPTQ arms were 1.55–4.31× worse. No HQ4 arm showed EXL3-class quality at ≤ 4.5 bpw | `research/2026/b70-run3-agent-reports-20261011/hq4-forensics.md`; `results/2026/10/RQ-26-20261010T231243348252Z-1e1f43f8d1/arm_stats.json` | The earlier "HQ4 was not losing on quality" reading was wrong. RQ-26 closed rejected; successor RQ-27 (R4) |
| **Flash-Next route quality evidence, despite the battery failing its gate**: 9/10 host tasks (gen 256), 4/6 canaries ×2, 8K two-turn turn-1 correct, replay reps token-identical, speculation acceptance **0.789 at 3.20 tok/round** | RL-144 run root `results/2026/10/RQ-17-20261010T171539823866Z-f31436becd/` | The adopted route looks healthy. ADR-0011 **not** reopened (registry: RQ-17 closed `rejected`, resolution semantics recorded) |
| **The route of record is a THINKING model** — every reply carries `reasoning_content`; the canary "failures" were the gen-64 budget truncating correct computations ("7" = first digit of 703, reasoning reached 740→703 correctly) | Same run root; `raw/serve/canary-*-batt-arith.json` | New fact about the adopted route: all future measurements must budget for (or separate) thinking tokens; battery re-author needs gen ≥ 256 |
| **Staging complete**: BF16 (18 shards) + EXL3 4 bpw (2 shards), 72.5 GB sha256-verified on SN770 with `ARTIFACTS.sha256` manifests | RL-140; manifests on SN770 | RQ-26 F2/F3 and any re-run are unblocked on materials |
| **Per-card VRAM telemetry discovered**: `sudo cat /sys/kernel/debug/dri/{0,1}/tile0/vram_mm` → `visible_avail` | Run-2 cleanup evidence | Fixes the tooling gap that blocked runs 1–3 preflights |
| **v0.30.0 Nemotron GPTQ-INT4-G64 + DFlash-n7 serve flags pinned from primary source** (incl. `--profiler-config config/profiler.py:39`, `xpu_fused_moe is_int4=True`) | `research/2026/v0300-nemotron-gptq64-dflash-n7-serve-source-20261010.md` | RQ-11 re-entry needs a boot fix, not research |
| **DPAS F32 reduction order is not bit-reproducible against CPU oracles** (max diff ~40 abs; all candidate reduction models rejected empirically) | F1 round-2 correctness evidence | Methods rule for all future kernels: DPAS-path "bitwise equal vs CPU" claims need a reduction-aware reference or tolerance class |

## Detected — losses (sealed, each with cause + re-entry)

| Question | How it died | Ledger | Re-entry |
|---|---|---|---|
| RQ-17 (Flash-Next quality gate) | 2/2 attempts: attempt 1 adapter crash (`raw/serve/` never created); attempt 2 canary gate 4/6 by **gen-64 truncation of the thinking channel** + replay segfault from foreign-loop contention | RL-141/144/142 | Operator re-author: canary gen ≥ 256, foreign-client guard, kv8k-turn2 budget. Route likely passes |
| RQ-11 (MoE attribution) | 2/2 attempts: attempt 1 foreign contention; attempt 2 serve container died at boot with empty logs (retained argv shows suspect `--speculative-config` quoting + log-capture race) | RL-146/147/148 | Interactive boot repro outside sealed runs → fix → re-author. Attribution lane fully staged |
| RQ-26 F0 (HQ4 quality screen) | 2/2 attempts: deadline overrun; then a clean run whose KLs are ~0 (1e-29 to 1e-12, some negative) on 4/6 blocks **for every arm and the EXL3 reference alike**. The metric degenerates on those blocks; it is not an EXL3-only reconstruction collapse | RL-149/150, RL-151 | Record the metric defect as a new mechanism (block-delta or logit-level KL), then re-enter. The 2 usable blocks argue for fixing, not closing |
| HQ4 kernel half | F1 gate missed ~20× (see wins) | RL-145 | Closed per stop rule |

## Never reached (budgets intact)

- **RQ-25** (SoA IQ3_XXS layout, priced 1.24×, Amdahl ceiling 1.61×): still `ready`, **2 sealed attempts unused**; needs one clean GPU window for the ~1 h isolated bench.
- **RQ-18** (two-card layer split, ≤1.53×): `gated` on RQ-17 passing a fixed battery.
- RQ-26 F2/F3 (whole-model convert + vLLM pairs): moot unless HQ4 re-enters.

## Operational facts

- The other session's llama-stream loop (arm.sh harness, Qwen3.8-27B-UD-Q4_K_S arms, later with MTP-draft speculation) episodically held both cards all day; it caused or compounded every GPU-lane failure. It exited on its own by 00:23Z. Never killed (contract).
- 4 workflow runs, 18 ledger rows (RL-131…148 machine + manual, plus 149/150), all blockers recorded with exact causes; caps 180/230 W verified as-found after every run; foreign stacks untouched in every cleanup proof.
- Registry survived every escalation: no gate lowered after data, ADR-0011 protected from a false reopen, vocabulary collisions (research_queue's passed/rejected) caught by the pen before they broke the tooling.

## Open operator decisions (queue halts on these)

1. ~~RQ-26: fix or close~~ — closed rejected 2026-10-11 (RL-161); successor RQ-27.
2. RQ-17 battery re-author authorization (gen ≥ 256) — unlocks RQ-18's dependency.
3. RQ-11 container-boot diagnosis + re-author authorization.

---

# OUTCOMES — run-2 window append (2026-10-11, second window; RL-152…RL-157)

PROVISIONAL — NOT FOR PUBLIC HEADLINE. Rows below belong to the 2026-10-11
run-2 window; the pre-window content above (ending at line 54) was not touched.
Full narrative: `research/2026/b70-run2-workflow-report-20261011.md`. Authority:
`registry/research-program.json`. **Auditability limit:** this file is untracked
in git (`git status --porcelain` → `??`), so "nothing above was rewritten"
cannot be diff-verified from the repo — it rests on the writer's session-anchored
edit (pre-window content ended at line 54). This window's own rows were
corrected in place after the claims audit; see the correction note at the end of
this section.

**Registry correction owed (material):** RQ-26's registry `next_action` sentence
"no AutoRound; the campaign was never dispatched" is FALSE for run
`results/2026/10/RQ-26-20261011T122026925511Z-0d5b5ff8b7` — 24 unscored jobs ran
after the failed sanity wave: 8 AutoRound (gdn_l01/gdn_l30 × {uniform,e2m1} ×
{h0,h1}, iters=100, all-6-tensor INT8 escape, bpw 8.125), 12 GPTQ
(gdn_l01/gdn_l30/gdn_l62 × {uniform,e2m1} × {h0,h1}; gdn_l62's 4 AutoRound jobs
never ran), 3 completed EXL3 reference cells (gdn_l01/l30/l62, 271.2/251.3/255.7 s,
kl_mean 1.55e-3/1.05e-2/5.53e-3 — the baseline side of the aborted comparison)
and 1 sanity-fix recheck; job mtimes 12:41–13:19:39Z, blocked state written
13:23:40Z (`blocked_incomplete.json` `utc`). Dispatch cause is the lane's
post-seal attribution to a scheduler indentation bug — no retained artifact (the
sealed driver was overwritten; see the hygiene row below). Verdict `blocked` is unaffected: no
scoring ran, no `measurements.json` exists. The false sentence mirrors
`sanity_control.json`'s `rule` field and needs an operator/registry correction.

## Detected — run-2 (wins and evidence)

| What | Evidence | So what |
|---|---|---|
| **RQ-17 battery: every quality element passed sealed; blocked only by an unsatisfiable plan conjunct** — canaries 6/6 in runs A+B (12/12 at finish stop, incl. batt-arith "703", the exact value RL-144's gen-64 budget truncated), replay 3×3 token-identical, tasks 9/10 (gate ≥8), 8K two-turn correct; zero truncated cells; guard log 70/70 rows rc=0 with zero foreign inference PIDs (sealed texts say 69 — off by one, cleanliness unaffected) | RL-156; run root `results/2026/10/RQ-17-20261011T114846278867Z-71a59de867` | The adopted Strata IQ3_XXS route looks healthy. The single failing conjunct (`canary_prompt_token_exactness` vs bare batt-file id counts) is a plan-contract defect: the serve surface renders a uniform +44-token template overhead, identical A==B per canary, and the same +44 exists in RL-144's sealed raw. One amended attempt (2 of 2) remains; ADR-0011 stays closed (stop rule never fired). No speed or energy number claimed |
| **RQ-26 F0 blocked at the predeclared sanity floor, zero arms scored** — att_l03 block-delta KL ≈ 0 (fp32 −2.03e-20, fp64 −2.03e-20 vs the 1e-6 floor) while all 7 linears were genuinely perturbed (rel_fro 0.2398–0.2570); the other five blocks cleared (4.7e-05 … 0.38) | RL-157; run root `results/2026/10/RQ-26-20261011T122026925511Z-0d5b5ff8b7` + `sanity_control.json`; NVMe 40.97 GB checkpoints via `nvme_pointer.json` (387/387 verified) | The bf16 block forward is structurally blind on att-class blocks — the F0 metric is unanswerable as declared. Re-entry needs an fp32 screen forward or att-block replacement; RL-155 bars attempt 2 without a new mechanism or fresh authorization. Unscored mechanism signal retained: at iters=100 the AutoRound arms diverge on real calibration data — all 48 per-tensor loss traces across the 8 AR jobs end above their start (rises ~6.5×–160.8×; spans from 2.07→46.21 up to 7817→367258; 1 of 48 non-monotone: gdn_l30 ar-uniform-h1 mlp.up_proj) and all tensors escape to INT8 (bpw 8.125 > 4.5 cap); any gate-passing F0 arm must be GPTQ. (The first pass of this close-out quoted "loss 2.6e3→5.8e4 monotone" — that is one tensor of one cell, gdn_l01 ar-uniform-h0 linear_attn.in_proj_qkv 2600.32→57935.97 — and is withdrawn as a recipe-level quote) |
| **RQ-25 ISA round 2 (compile-only, NO GPU touched)**: 16-byte q-plane loads achieved in the SoA gu kernel (`load.ugm.d32x4.a64`, rd:8) at dump lines **1058/2251/3087** — corrected; the lane's cited 157/1350/2186 resolve nowhere in the retained dumps. Spills reduced 12→8 (ocloc "around 8"); 0-spill gate still unmet; barriers 0 in all four kernels | `scripts/tmp/rq25_soa_20261011/isa/r2/` (SHA256SUMS OK) + NVMe `rq25/isa-devimg-r2/`; fresh hashes differ from round 1, round-1 files retained | Round 2 has **no registry/ledger row** (defect) and the registry `next_action` is stale ("run it on GPU0 when the card is free"). Corrected discriminating grep `\band\b.*\s2:w\b`: current 10 / SoA 0 — the lane's quoted grep returned 0 on both variants (vacuous, now withdrawn) |
| **RQ-11 boot repair passed — UNSEALED, diagnostic only**: att1 silent SIGSEGV (rc=139, RL-148 phenotype) root-caused to `ZE_AFFINITY_MASK=level_zero:0` plus `--speculative-config` quoting; T1/T4 fail vs T2/T3 pass in the probe matrix; att2 boots clean (`working_argv.sh`:73 `ZE_AFFINITY_MASK=0`), greedy "Canberra." correct, clean rc=0 shutdown | `/mnt/<nvme-disk>/b70-run2-20261011/rq11/boot-repair/` (SHA256SUMS 9/9); `results/2026/10/rq11-boot-repair-20261011-UNSEALED/POINTER.md` | "Passed" covers the boot repair ONLY — the question itself stays closed/rejected (empty run_root/ledger_id, "NOT a sealed run"); the 20% task-energy gate was never measured. Lane-reported sn770_bound_bytes=0 is wrong: ~55 KB was written to `/mnt/<disk>/rq11/` (`du -sb` = 55,495 B; the close-out's first pass said ~100 KB — overstated ~2×, corrected). The daemon left up for a follow-up sealed run was stopped at window close (follow-up never ran); bridge `<exl3-bridge>` kept per standing config (DOWN, daemonless) |
| **Window cleanup proven, re-verified twice after the proof** — caps equal as-found (180 W/230 W, PCI→hwmon5/hwmon6), unit `<exl3-dockerd>d8` inactive/not-found, socket gone, no listeners on 8015/8000/8080/8001/11434/8095, no inference processes, `/` ≥ 9.0 GB free, SN770 manifests re-verified (RQ-26 387/387 OK) | `cleanup_proof.md` sha256 `749c0181…c92aba` (re-hashed this close-out); repo pointer `results/2026/10/run2-20261011-cleanup/POINTER.md` | No lane processes remain; deliberate leftovers documented in proof §6 (bridge kept; failed `<exl3-dockerd>d6` unit record retained as failure evidence) |
| **Handoff hygiene defect found by red team, fixed by the lead**: `research.py check` FAILED at window handoff (stale `docs/indexes/research-coverage.md` — the cleanup lane's run root post-dated the last render); lead ran `research.py render` → `check` OK | Lead's commands 2026-10-11 ~17:00Z: `check` FAIL → `render` (4 generated views) → `check` "OK: research registries valid, generated views current" | Generated views only; `registry/model-status.json` untouched (mtime 2026-10-09). Two evidence-retention defects stand uncorrected: the sealed RQ-26 driver (sha256 `48187f5b…`) was overwritten in place post-seal (now `80b68d9c…`, mtime 15:48Z > run end 13:24Z) and is unrecoverable; RL-156 carries no amendment text (the VRAM-bar amendment 31000→28500 MiB is disclosed only in `measurements.json` + `SN770_POINTER.md`) |

## Run-2 losses (sealed, exact causes)

- **RQ-17** (RL-156, ledger outcome `failed` / assessment decision `rejected` — classification split disclosed: lane and registry narrative both class it a plan-contract defect, not a quality failure): `measurement_validity` failed on `canary_prompt_token_exactness` only — unsatisfiable as declared on the chat surface; the property it guards (identical prompts across arms) is proven by A==B.
- **RQ-26 F0** (RL-157, `blocked`): harness broken per the predeclared rule — att_l03 sanity KL ≈ 0 in fp32 and fp64; no scoring.
- **Never reached**: RQ-18 (gated on RQ-17 passing), RQ-25 route pairs (isolated gate not passed).

## Open operator decisions carried out of run-2

1. RQ-26: correct the registry's false "never dispatched" sentence; decide the F0 re-entry mechanism (fp32 screen forward or att-block replacement) — RL-155 requires fresh authorization either way.
2. RQ-17: authorize attempt 2 with the token-exactness declaration amended to the rendered surface (file ids + measured 44-token overhead, or A==B per canary).
3. RQ-25: record the round-2 ISA findings in the registry and refresh `next_action`; decide whether the 0-spill gate stands (12→8) or the isolated bench proceeds anyway.
4. RQ-11: authorize the sealed run with `working_argv.sh` (the isolated daemon must be restarted — it was stopped at window close).

**Correction note (2026-10-11, post-claims-audit, same window; corrections applied
in place to this window's own rows above):** (1) the RQ-26 dispatch-in-error
inventory said 20 arm jobs — it is 24 jobs: the 3 completed EXL3 reference cells
(gdn_l01/l30/l62, kl_mean 1.55e-3/1.05e-2/5.53e-3) and the sanity-fix recheck
were omitted, and the 12 GPTQ jobs' block coverage (gdn_l01/l30/l62, with
gdn_l62 having no AutoRound jobs) was never stated; (2) the AutoRound loss quote
"2.6e3→5.8e4 monotone" was one tensor of one cell and is re-scoped to the
measured aggregate (48/48 traces rise, 6.5×–160.8×, 1 non-monotone); (3) the
job window "12:41–13:23 UTC" conflated the 13:23:40Z blocked-state write with
job mtimes (last job 13:19:39Z); (4) the rq11 NVMe write is 55,495 B, not
~100 KB; (5) "nothing above was rewritten" is not repo-auditable (file
untracked) and is now stated as a session-anchored claim. All corrected values
were re-measured from the sealed run root during the audit response.

# OUTCOMES — run-3 window append (2026-10-11 evening; RL-158…RL-170)

PROVISIONAL — NOT FOR PUBLIC HEADLINE. Earlier sections stay as written.

## Detected — run-3

| What | Evidence | So what |
|---|---|---|
| **Strata engine resets are intermittent, not card-bound**: RQ-17 attempts 2–4 lost the server to xe "Timedout job … in strata" resets on GPU0; the same 10 frozen host tasks then ran 2× on each card with 42/42 HTTP 200, zero resets and identical answers | RL-160, RL-162, RL-166; `results/2026/10/strata-reset-diag-20261011-UNSEALED/` | The route answers correctly when it stays up. RQ-17 is at its attempt limit; re-entry needs a named mechanism tested unsealed first (post-ready settle delay, xe ccs job timeout, page-cache churn) |
| **RQ-25 never measured the layout**: sealed attempt 1 lacked the oneAPI runtime (`libsycl.so.9`); attempt 2 crashed GPU1 because `GROUPS` is a read-only bash array (expands to gid 1000), so 1000 work-groups read past a 427-entry table | RL-163, RL-164, RL-165; `scripts/tmp/rq25_soa_20261011/run_bench.sh:50-55` (fixed: `NGROUPS` + ring bound check) | Both failures are harness defects. Unsealed single-cell lead at groups 128: SoA 214.4 vs AoS 212.5 GB/s at 256 GRF (+0.9 %), far below the 1.5× gate |
| **R4 fold is exact; the declared s1 sanity measured bf16 storage noise** | RL-167; fold probe rel L2 1.3–4.9e-6 per layer vs bf16 compute noise 6.8e-3 | s1 amended before any sealed run (registry + plan, success gate unchanged). Harness sped up 175 → 38 s/layer |
| **R4 2-layer smoke trend is unfavourable** (not a verdict): KL vs bf16 EXL3 0.0174, r4_g64 0.0278, r4_g128 0.0332, gptq_plain 0.0356 | RL-167 (Smoke B, layers 1 and 3, full calibration) | Rotation recovers 7 % (G128) and G64 another 16 %; R4 at 1.6× EXL3's KL would fail the R0 gate |
| **R4 13-layer prefix (unsealed) confirms the trend**: KL vs bf16 EXL3 0.00122, r4_g64 0.00353, r4_g128 0.00411, gptq_plain 0.00438 (1664 positions) | RL-170; `results/2026/10/rq27-r0-prefix-smoke-20261011-UNSEALED/` | Best R4 arm at 2.9× EXL3's KL. Rotation + affine INT4 does not reach the trellis on this model |

## Run-3 losses (sealed, exact causes)

- **RQ-17 attempts 3–4** (RL-159, RL-162): xe engine resets on GPU0 mid-battery. Attempt limit reached.
- **RQ-25 attempts 1–2** (RL-163, RL-164): harness defects above. Attempt limit reached.
- **RQ-27 R0 attempts 1–2** (RL-168, RL-169): attempt 1 lost positive-definiteness in the blocked fp32 Cholesky at layer 11; attempt 2's fp64 CPU fallback (layer 2 down_proj) was OOM-killed by the 14 GiB container cap. No arm scored. Attempt limit reached.

## Open operator decisions after run-3

1. RQ-25: authorize one harness-fixed attempt 3 (≈15 min GPU1), or close on the unsealed +0.9 % lead.
2. RQ-17: choose the re-entry mechanism for the intermittent resets, or keep ADR-0011 on its RL-156 evidence.
3. RQ-27: close as rejected on RL-170 (recommended: 2.9× EXL3 KL in the unsealed prefix), or authorize attempt 3 with fp64 hinv and a ≥ 24 GiB container cap.

# OUTCOMES — run-4 window append (2026-10-11 night → 2026-10-12; RL-171…RL-174)

PROVISIONAL - NOT FOR PUBLIC HEADLINE. Earlier sections stay as written.

## Detected — run-4

| What | Evidence | So what |
|---|---|---|
| **Scalar 4-bit is capped by mathematics, not by recipe** | RL-173: two-sided Hadamard INT4 G64 = 2.46x EXL3 KL, G128 2.84x (13-layer prefix); Gaussian Monte Carlo: uniform INT4 2.95x the rate-distortion bound D(4), Lloyd-Max 2.43x, NF4 G64 2.17x; EXL3 measures 1.49-1.51x on real tensors | No rotation, group size or GPTQ variant makes fixed-rate scalar INT4 match EXL3 at 4.0 bpw. Only trellis-class quantizers do |
| **E8 lattice codebooks do not close the gap at 4 bits** | RL-173: E8P12 two-stage RVQ = 2.37x D(4) Gaussian, 1.65x EXL3 weight MSE on real tensors | Killed before any GPU time |
| **Stock INT4 has no speed edge on today's image** | RL-174: v0.30.0 no-spec W4A16 32.85 vs EXL3 32.53 tok/s (1.01x); with MTP EXL3 79.9 vs W4A16 71.5 | The old 1.62x anchor is gone. EXL3 wins on both quality and speed |
| **The speed headroom is the M=4 verify pass** | RL-119 + RL-174: EXL3 runs ~77 % of pin at M=1 but ~380 GB/s during MTP verify [INFERENCE from load size] | A new format must decode cheaply at M=4. That is the XTQ kernel target |

## Run-4 losses

- **RQ-27 R4** closed (RL-171): rotation was exact; the scalar grid lost quality.
- **RQ-25 SoA layout** closed (RL-172): no layout gave a speedup in the end-to-end dry run.
- **RQ-28 scalar/E8** closed (RL-173).

## Active after run-4

1. **RQ-29 XTQ** (`research/2026/b70-xtq-format-design-20261011.md`): trellis quality with an Xe2-cheap code decoded into the DPAS operand layout. Kill ladder K2 (CPU MSE) → K3 (ISA price) → K4 (GPU1 microbench ≥ 550 GB/s at M=4) → K5 (quality vs EXL3).
2. Host rule learned: the session died twice to systemd-oomd (user slice > 50 % memory pressure). Heavy jobs now run in capped containers, one at a time, with ≥ 20 GiB MemAvailable before start.

## Run-4 continuation (2026-10-12; RL-175…RL-180)

| What | Evidence | So what |
|---|---|---|
| **Per-tensor bit allocation is the biggest quality lever left** | RL-179 (our harness, 64 layers): turboderp SC_5.00bpw_H6 KL 0.42x plain EXL3 4.0 on prose, 0.16x on chat; 5/5 canaries. SC_4.00bpw_H5: 0.75x on chat but 1.15x (worse) on prose | SC5 is the quality tier the operator chose (RL-180). SC4 fails the predeclared prose gate |
| **Published SC numbers reproduce** | RL-179 held-out chat KL within 7-12 % of turboderp's chart (RL-175) | His in-domain KLD chart is usable as a prior for this model |
| **Trellis codes: no free lunch on Xe2** | RL-176: cheapest Xe2 code (xe_ih2x2) decodes 1.71x cheaper than EXL3 mul1 at +9.9 % MSE; codes within 5 % MSE are not cheaper | XTQ continues by operator amendment; K4 speed and K5 quality decide |
| **Own trellis quantizer exists** | RL-177: bit-exact EXL3 mul1 writer; torch-XPU ~64 h for 27B, SYCL Viterbi in progress | Needed for any XTQ artifact |
| **Copy-engine reset was a harness bug** | RL-178: bcs page fault after unsynchronized `empty_cache` + cross-thread XPU use; rerun clean after the fix | Rule: synchronize before `empty_cache`; XPU work on one thread |

Next: new image `b70/exl3xpu-mixedk` to serve SC5 (K=4..7, mixed bits in fused modules), then canaries and speed pairs against plain 4.0.

## Run-4 continuation 2 (2026-10-12 afternoon; RL-181…RL-189)

| What | Evidence | So what |
|---|---|---|
| **SC5 serves on the B70** | RL-182: `b70/exl3xpu-mixedk` image, K=2..7 mixed bits in fused modules, 46/46 kernel checks, canaries 6/6, long 3/3 | Best-quality 27B artifact now runs |
| **Dispatch fix: +23 % for SC5 with MTP** | RL-185: odd-K and tiny-N K6 calls at M≤4 routed to the GEMV kernel (tag `20261012c`); SC5 MTP3 greedy 57.0 → 70.3 tok/s; plain 4.0 unchanged | No new kernel needed; verify-round attribution measured under graph replay |
| **SC5 is still slower than the recipe** | RL-186: recipe protocol (T0.7, 5 ABBA pairs, 230 W): plain 4.0 63.56 / 64.34 tok/s vs SC5 56.62 / 56.82 (0.88-0.89x); 8-13 % more J/token | SC5 is a quality choice (KL 0.42x prose / 0.16x chat) at about 11 % speed |
| **Xe2 trellis code is load-bound** | RL-183: xe_ih2x2 runs at 0.69-0.84x of EXL3 speed in the K4 bench | RQ-29 closed; fewer decode instructions do not help |
| **Draft graph capture does not pay** | RL-187: piecewise draft capture 0.96x / 0.91x; RL-188: V2 runner FULL draft capture 0.99x / 0.87x, outputs byte-identical | Draft-capture mechanism closed after two attempts; keep FULL_DECODE_ONLY |
| **Our GPU quantizer is bit-exact** | RL-181 (Viterbi), RL-189 (ESIMD + LDLQ): 0 mismatches on 2 real tensors; 2.45-2.69 s per 1M weights | Full 27B ≈ 18-19 h on one card; 13-layer quality test next |

Next: 13-layer quality test of our quantizer (EXL3 code, same bits) against turboderp's plain 4.0. If its KL is ≤ 0.90x on both slices, quantize the full model.

## Run-4 continuation 3 (2026-10-13; RL-190…RL-194)

| What | Evidence | So what |
|---|---|---|
| **Own quantizer: not better than turboderp yet** | RL-191: 13-layer prefix with 32K-token calibration, KL 1.16x wiki / 1.41x chat vs plain 4.0; Q2 with his calibration mix: per-tensor error 1.027x (in-sample, L0-49) | The 0.2-0.8x in-sample wins were overfitting; quality gate not met |
| **Q2 paused after layer 51** | RL-194: graceful stop, checkpoint sha 058f730f…, rental kit with CUDA driver + known-answer gate (CPU-checked only) | Resume on B70 (~2.2 h) or a CUDA rental; `research/2026/xtq-q2-pause-and-resume-20261013.md` |
| **Published (operator order)** | github.com/SergiioB/exllamav3 branch `b70` @244f57e (serve patches + ESIMD quantizer); `ghcr.io/sergiiob/exl3xpu-mixedk@sha256:7bb0c5cf…` (public, rebuild from the fork gives byte-identical `_C.so`); b70-launcher v0.5.2 with recipe `qwen38-27b-sc5`; cookbook records `qwen38-27-exl3xpu-sc5-*` @b4464f7; xecores.com deployed | SC5 is a one-click quality pick on the launcher; the launcher end-to-end launch itself was not run (GPU stop) |

Next: one launcher smoke of `qwen38-27b-sc5` when a GPU is free; then resume Q2 (B70 or rental) and run full-64 KL.
