---
title: "Night/breakfast campaign closeout — Stage A baseline, C2 V2 kernel, margin-requant rounds 1-3, W4A16 diagnostic lane (2026-09-30)"
type: worklog
date: 2026-09-30
status: final
---

# NIGHT REPORT 2026-09-30 — overnight/breakfast campaign closeout

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** No champion claimed, no promotion, no publication. Ledger rows filed: RL-108…RL-116 (RL-112..116 by this close). Campaign registrations: CAM-2026-C2-V2-KERNEL-REPAIR, CAM-2026-MARGIN-REQUANT-PILOT1/2/3, CAM-2026-W4A16-IMAGE-PULL/LAUNCHVERIFY/DIAGSPEED-20260930. Production route UNCHANGED: EXL3 4bpw on exl3xpu (ADR-0010 stands).

## Scope

Continues the overnight breakthrough campaign (`results/2026/09/b70-overnight-breakthrough-20260930T020047Z`, RL-108 — baseline acceptance FAILED, partial) into the revised-resume campaign (`results/2026/09/b70-resume-breakthrough-20260930T063200Z`) plus three follow-on workstreams closed 2026-09-30 morning: the C2 V2 kernel window, the margin-requant pilot program (rounds 1-3), and the W4A16 serving-lane prep + diagnostic.

## What was attempted and measured

### 1. Stage A — fresh incumbent baseline (RL-109, already filed)

Root `results/2026/09/b70-resume-breakthrough-20260930T063200Z/A`. GPU0 as-found **180 W**, image `ghcr.io/0xsero/exl3xpu@sha256:21412bdd…`, artifact `b70-exl3-qwen38-4bpw-20260926` (read-only mount), MTP3, fp8 KV, FULL_DECODE_ONLY graph capture, cache ON, ctx 65536, gmu 0.9, C1, recommended non-thinking sampling.

Fresh n=5 medians, **endpoint-token-count-normalized content-chunk-window rate** `(usage.completion_tokens-1)/(last-first content chunk time)` — NOT engine tg, NOT exact token arrivals:

| cell | median | range |
|---|---|---|
| p512/g128 | 60.60 | 57.93–68.61 |
| p8192/g128 | 59.83 | 57.36–64.26 |
| real workload g128 | 68.72 | 68.24–71.95 (prompt tokens ~100-106) |

Quality: 36/36 greedy fixtures, 3 heldout exact answers, 3 cache/spec canaries passed BEFORE timing; every sample exact-length, zero cache-hit delta. Power: GPU0 ~181.6–186 W request-window means at 180 W cap; GPU1 idle ~46 W. Stage B (RL-110, P0 CPU suffix control: BF16 +0.4022 / arext −2.9729) and Stage C (RL-111, C1 staged-reduction + C2 index-simulator falsifiers both FAIL) were filed by the earlier close.

### 2. C2 V2 kernel — parity PASS, timing NEGATIVE (RL-112)

Root `results/2026/09/b70-c2-repair-20260930T085200Z`. GPU1 quiescent lease, as-found **230 W**, image `exl3xpu:k3-p6-mb4-20260929`.

- **Variant under test**: ESIMD_REGISTER(2048)/(3072) explicit B-bank binding on the M=4 partial-stage K-pair path vs stock `exl3::DpasKernel<4,2,4,4>`.
- **Independent ISA pre-review** approved a parity gate; scope-equality verified (same fixture npz per cell, identical 1440-subgroup × 8 grid).
- **Numerical gate PASS**: 5 synthetic + 2 real cells all bit-identical to incumbent (max_abs ≤4.7e-5 synthetic / ≤8.4e-7 real, rel_l2 ≤1.5e-7), deterministic-200, XPUGraph capture/mutation/allocator PASS; decode-parity fixture == `exl3_reconstruct` bit-exact.
- **Paired isolated timing NEGATIVE** (XPUGraph kcap=32, 100 timed replays, median + bulk200 eager cross-check, per-arm fresh containers, ABBA):
  - down_proj (K=17408 N=5120 P=18 rps=61): candidate 111.85 µs vs incumbent 108.41 µs graph (**+3.17 %**; bulk +3.55 %)
  - qkv (K=5120 N=12288 P=8 rps=40): candidate 76.23 µs vs incumbent 73.00 µs graph (**+4.43 %**; bulk +5.94 %)
- **Verdict**: validated-but-not-adopted. No variant spent — every failed run was a harness/fixture bug (gate.py in-process cooldown wait unobservable under live XPU context ~60-75 °C pkg), kernel sources untouched post-review (hash matches). A ~10:25Z concurrent-driver contamination event was identified and its samples excluded (the earlier "106 vs 354" gap was that artifact, not real).
- C1 (staged reduction) was closed earlier in the same campaign as a miss (RL-111).

### 3. Margin-requant program — rounds 1-3 (RL-113/114/115)

All CPU-only fp32 teacher-forced sequence-margin work on the arext W4A16 export (`b70-autoround-qwen38-w4a16-20260926`, source read-only and hash-verified unchanged). Frozen heldout manifest sha256 `de54921f…` throughout; nothing from heldout ever entered a gradient.

**Round 1** — `results/2026/09/b70-margin-requant-pilot1-20260930T0855Z` (RL-113):
- Scale-only ISO-4bit repair: 18 `weight_scale` tensors × mult∈[0.25,4.0] in shard model-00004; codes/zp8/g128/packing/lm_head/mtp bitwise untouched.
- Canary r01 **REPAIRED**: −2.9745 → **+0.5584** seq margin under the serialized artifact (greedy flips '30'→'14' correct; in-mem v2 +7.36).
- Serialization machinery proven: 463/481 tensors bitwise-unchanged, packing equations re-verified from derived bytes, derived == in-memory ≤1e-4.
- Heldout gate **NOT MET**: n03 (bf16 +0.62 → arext −1.09) worsened to −2.50 (v1) / −3.05 (v2); h03/n01/n02 stayed correct. Attribution: n03 is upstream-dominated (terminal window contributes +0.12 only) — two defect classes exist: window-weight-dominant (repairable) vs upstream-dominated (terminal-window scope dead).
- `INDEPENDENT-PREREVIEW.json`: CONFIRMED protocol/serialization integrity (freeze ordering, calibration/heldout disjointness), minor flags only.

**Round 2** — `results/2026/09/b70-margin-requant-pilot2-20260930T1045Z` (RL-114):
- Window widened to L55-63; v3 (cal r01,c1,c2) and v4 (full classified pool) both repair r01 to **+13.2** seq margin.
- Transfer **FAILS 4/4**: n03 delta vs arext source −1.41 (R1-v1), −1.96 (R1-v2), **−0.887** (R2-v3), **−0.824** (R2-v4). Extended window halves the collateral but never reaches non-worsened. Repair magnitude does not predict collateral.
- Class law: scale-only repair retargets REPRESENTED trajectories; unrepresented near-boundary trajectories drift negative. Missing ingredient = calibration representation of the near-boundary class.

**Round 3** — `results/2026/09/b70-margin-requant-pilot3-20260930T1145Z` (RL-115):
- **BUDGET FAIL on execution, not mechanism.** The cheap half completed with a strong positive signal:
  - 270-row disjoint pool generated (12 forms, seed 20260930, string+form disjoint from heldout, arithmetic invariants machine-verified on all 270).
  - Near-boundary structure is **FORM-CONCENTRATED**: genexp/step forms (sum(range), sum(2·i..), sum(i..,step), sum(i·i..,step)) median |margin| 0.33–1.06 with 30 rows <0.12; easy two-arg forms sit at 7.7–10.8 → the fragile class is cheap to harvest by form.
  - Two-pass classification at ~4.9 s/row with X1 window==full sanity (≤0.001/0.0003); selection S1-S4 → 97 in-band rows + 3 documented fallback fillers; **calibration v5 frozen** (102 tuning rows incl. anchors; near-boundary weight w=10); scripts p12–p15 staged and hash-recorded.
  - Measured tuner cost law: all-57-block grad steps **75–240 s/step** (memory-churn-bound at B=8 on 4 CPU threads); selected-block main loop ~25–35 s/step at B=6.
- **Failed half**: three tuner launches (28/10/6 steps) killed or died before any candidate froze — one traced to a self-matching pkill in the operator shell; the third died silently ~12:36–12:41Z, no traceback, no OOM record — the RL-092 exec-session kill class. **No v5 candidate, no serialization, heldout never re-evaluated.** Rounds-1/2 record stands.

### 4. W4A16 serving lane — de-risked, class price measured (RL-116)

- **Image pull** `results/2026/09/b70-w4a16-image-pull-20260930T1105Z`: `vllm/vllm-openai-xpu@sha256:f01e24f6…` pulled into the isolated exl3 dockerd (`unix:///run/<exl3-docker>.sock`, NVMe data-root) — 18 layers, 463 s, digest verified, ~12.86 GiB to NVMe; root disk −5.6 MB noise only; main dockerd untouched.
- **Launch verify** `b70-w4a16-launchverify-20260930T111244Z`: READY. Container on isolated dockerd, model mounted ro; Qwen3_5MTP arch resolved, compressed-tensors W4A16 g128 sym, fp8 KV (144,955 tokens), MTP4 speculative, graphs ON, prefix-cache OFF (counters confirmed 0), max-num-seqs 1; three patches applied (mtp_nightly, mtp_boundary, champion_stack_overlay); endpoint ready in 290 s; controlled stop.
- **Diagnostic speed** `b70-w4a16-diagspeed-20260930T112958Z`: GPU0 as-found **180 W**, n=3 per cell, same chunk-window metric and sampling as Stage A, zero cache-hit delta:
  - p512/g128 median **64.49** (64.49/57.39/64.50) vs EXL3 60.60 → **+6 %**
  - p8192/g128 median **46.34** (46.34/46.02/48.35) vs EXL3 59.83 → **−23 %**
  - spec α collapses at p8192 (accepted/draft ≈0.51–0.56 vs 0.64–0.77 at p512).
  - **DIAGNOSTIC ONLY** — the source artifact fails the exact-answer canary (30-defect class); NOT a serving recommendation. The historical "83.7" figure was a different artifact, different statistic, at 230 W — not comparable, not a regression claim.

### Harness / infrastructure fixes recorded

- Stage A: oneCCL by-path mount fixed from prior container inspection; calibration-exactness repair loop (bounded ≤3, used) established the frozen exact prompt pools; original partial-real samples excluded and re-measured post-repair.
- C2: gate.py cooldown-wait defect documented (in-process wait can never observe ≤55-58 °C while a live XPU context pins pkg ~60-75 °C); concurrent timing drivers killed and respawns disabled; canonical gate evidence preserved as `gate-raw-run1/run2/PASS-*` copies after a `GATE_OUT` overwrite incident.
- Pilot3: self-matching pkill/kill patterns killed the operator wrapper shell three times — bracket-protected greps adopted; kill-by-pgrep-head-1 flagged unsafe. Tuner relaunch requirement: setsid-detached (RL-092 salvage pattern).

## Wins / losses

**Wins**: canary mechanism repair real and serialized (−2.97 → +0.56, greedy '14'); serialization + freeze discipline independently confirmed; C2 V2 kernel parity-PASS (a correct kernel exists, just slower); W4A16 lane fully de-risked end-to-end (pull→launch→measured diagnostic in ~40 min); near-boundary harvest machinery works and proved form-concentration.

**Losses**: heldout transfer gate failed 4/4 (upstream-class rows unrepairable by calibration-legal scale-only edits tried so far); C2 V2 timing negative ~3-4 % both cells; round-3 tuning never executed (budget + exec-session kills); no speed breakthrough anywhere — no quality-passing candidate, no ≥20 % matched comparison.

## Uncertainties

- n03-class repair needs the disjoint near-boundary calibration pass to complete (v5 is frozen and ready); if it still fails, the two-class world stands and the upstream class needs full-model margin-aware requant or routing to EXL3.
- Whether the W4A16 class would price differently at 230 W is unmeasured this window (screen ran at as-found 180 W).
- C2 variant-3 mechanism undefined; register binding alone is closed.
- Third tuner death cause undetermined (best evidence: wrapper-shell self-kill + silent background-child death).

## Exact next moves

1. **Round-3b (cheapest, machinery ready)**: relaunch `p12_tune_v5.py` unchanged on the SAME frozen calibration v5 (captures+anchors on disk, sha-hashed) — **setsid-detached, 60+ steps, 45-60 min CPU window** — then `p13` scoring + `p15` frozen gate. If PASS → `p14` serialize + correctness review + GPU quality + matched speed window. If FAIL → two-class world stands; near-boundary class needs full-model margin-aware requant.
2. **C2 variant 3**: requires a NEW mechanism (different scheduling structure), not more register binding — gate remains the same isolated-op parity + paired timing protocol.
3. **W4A16 implication**: at 180 W the class price is +6 % short / −23 % long vs EXL3 — the requant path's value is **quality restoration of the 4-bit class, not speed**. Any future W4A16 adoption still needs the exact-answer canary to pass.

## Live state at close (STATE-VERIFICATION.json, 12:52Z)

Both dockerd instances empty; no inference processes; GPU0 31,276/32,656 MiB free (~1.35 GiB = desktop/display on card0, normal), GPU1 32,615 MiB free; caps/PM as-found (GPU0 180 W/auto, GPU1 230 W/on); ports 8000/8200 free. **The operator's production vLLM provider (PID 1210253, `vllm serve qwen38` :8000, MTP4/fp8KV, up during pilot3 ~11:35Z) has since exited and was NOT restarted** — no authorization. `llama-profile.service` inactive; `b70-codex-proxy` running with dead upstream.

Cleanup verdict: CLEAN. No caps written, no containers leaked, evidence roots sealed.
