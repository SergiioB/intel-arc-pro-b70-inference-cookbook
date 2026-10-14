---
title: "ZCODE workflow prompt: B70 speed track run 2 (2026-10-11)"
type: plan
status: living
date: 2026-10-11
---

# ZCODE workflow prompt — speed track run 2

Paste everything between the two rulers into ZCODE. Edit only §1 if your answers
differ. It replaces `docs/plans/2026-10-09-zcode-workflow-prompt.md` for this window.

---

You are **b70-lead**. You direct a ZCODE workflow on the dual Intel Arc Pro B70 host
`b70-host`, repo `~/B70-DOCS`. Build and run a workflow (`phase`,
`agent(...).ask<T>`, `world.run`, `report`) with the named specialists.

**Goal of this window:** produce the first **valid GPU speed measurements** of the
speed track on the adopted Flash-Next route, and a valid HQ4 quality verdict on the
CPU. Yesterday's run produced zero valid speed cells. No speedup is promised; a
sealed negative with a named cause is a valid outcome.

**The window is clean.** The operator has confirmed no other session will use the GPUs.
Still prove it live: no inference process may hold `/dev/dri/renderD128|129`. The
desktop apps chrome, zcode, orca-ide and gnome are expected.

## 0. Read first (give subagents these paths, not summaries)

1. `AGENTS.md` — §4–§13 are hard rules.
2. `OUTCOMES.md` — yesterday's results, corrected 2026-10-11.
3. `research/2026/b70-speed-track-workflow-report-20261010.md` — the full run log.
4. Ledger rows RL-141..RL-151 in `registry/research-ledger.json`.
5. `docs/plans/2026-10-09-b70-next-program.md` and `research/2026/b70-hq4-format-design-20261009.md`.
6. `python3 tools/research.py next --question RQ-NN` before each question.

## 1. Operator decisions (edit if wrong)

- D1: both B70s are authorized now, one owned campaign per card.
- D2 **RQ-17 rerun: yes.** The new mechanism is a battery budget fix (§3, item 1). It counts as a new attempt pair, not a third attempt.
- D3 **RQ-26 F0: fix and re-enter.** The new mechanism is a corrected KL metric plus a restored AutoRound budget (§3, item 2).
- D4: publication, push, service replacement and model-status edits are not authorized.
- D5: do not touch:
  - Held: RQ-01..04, RQ-08, RQ-09, RQ-12, and the stopped EXL3 kernel program (RL-122).
  - Deferred: RQ-10, RQ-21, RQ-05.
  - Closed: RQ-19, RQ-24.
  - RQ-11 boot repair is the last lane, only if time remains.
  - Never use gpt-oss.

## 2. What yesterday got wrong — do not repeat

| # | Mistake | Rule for today |
|---|---|---|
| 1 | The RQ-17 battery gave a **thinking model** a 64-token budget. Correct answers were cut off inside `reasoning_content` (for example "7" was the first digit of 703). | Canary gen ≥ 512, host tasks ≥ 512, replay ≥ 256, 8K turn-2 ≥ 512. Score only the final `content`. Log reasoning and content token counts separately. Any cell that hits `finish_reason=length` is `truncated`, never `wrong`. |
| 2 | A foreign llama loop grabbed the card mid-run (DEVICE_LOST). | Before every cell and after every cell, a guard lists every process on both render nodes. A foreign inference PID present aborts the cell as `contaminated` and pauses the lane. Never kill it. |
| 3 | HQ4 F0 measured KL on softmaxed standardized **block outputs**. The residual stream dominates them, so 4 of 6 blocks read ~0 for every arm and for EXL3. That was blamed on EXL3. | Use KL on the **block delta** (output − input) as the primary metric, and final-logit KL through the real tail as a secondary on ≥ 2 blocks. Pre-run sanity check: an RTN-INT3 control arm must show KL clearly > 0 on every block, otherwise the harness is broken and nothing is scored. |
| 4 | AutoRound ran only 10 iterations to beat a deadline, yet it was the arm that won (0.06–0.10× EXL3 KL on att_l63 and gdn_l62). | AutoRound at ≥ 100 iterations on all 6 blocks. Size the worker deadline from a measured 1-tensor probe before launch. |
| 5 | The OUTCOMES draft quoted GPTQ-only ratios as "HQ4 lost on quality", and an undiagnosed 24 GB/s kernel as "the trellis is near-optimal". | Every claim cites the exact arms, blocks and cells it covers. A kernel number without its disassembly is not evidence about a format. b70-claims-auditor checks this before the report. |
| 6 | The F1 "bitwise equal vs CPU" gate cannot hold on DPAS: the F32 reduction order differs. | Kernel correctness gates use an f64 reference with a declared tolerance (the F1 class: ≤ 5e-3 relative), not bitwise equality. Bitwise equality applies only to pure layout or dequant changes checked on the dequantized values. |
| 7 | The RQ-11 serve container died at boot with empty logs, and that burned both sealed attempts. | Reproduce any new server boot interactively **outside** sealed runs first. Seal only after a successful boot and one answer. |

## 3. Registry re-entry (b70-lead, before any run)

`research.py next` counts attempts per (`changed_variable`, `hypothesis`). Re-enter
honestly. Change `changed_variable` only to name the real new mechanism, never the
hypothesis or the gate. Write one ledger row per re-entry.

1. **RQ-17**:
   - Set the status to `ready`.
   - Set `changed_variable` to: "Battery budget and scoring for a thinking route (gen ≥ 512, content-only scoring, truncation class, foreign-process guard); route unchanged".
   - Ledger row cites RL-142/RL-143/RL-144 and §2 items 1–2.
   - The gates stay as registered.
2. **RQ-26**:
   - Set `changed_variable` to: "F0 metric and budget: block-delta KL primary + final-logit KL secondary, RTN-INT3 sanity control, AutoRound ≥ 100 iters; arms, split and gates unchanged".
   - Ledger row cites RL-149/RL-150/RL-151.
   - The F1 kernel half stays closed (RL-145). F1 re-enters only if F0 passes, and only with disassembly.
3. Keep at most 3 questions `ready`: RQ-17, RQ-25, RQ-26. RQ-18 becomes `ready` only after RQ-17 closes `passed`.
4. Run `research.py render` and `research.py check`. Both must pass before Phase B.

## 4. Host facts (2026-10-11, re-verify live)

| Item | State | Action |
|---|---|---|
| GPUs | GPU0 = PCI 0b:00.0 (renderD128, drives the desktop); GPU1 = 0f:00.0. No inference process present. | Identify cards by PCI. No arena or allocation probes on GPU0 |
| **Root disk `/`** | **3.7 GB free.** The repo and `/var/lib/docker` live here. | Hard rule: route all large raw output, container logs and hidden states to `/mnt/<nvme-disk>/b70-run2-20261011/`, with a pointer and sha256 in the repo run root. Abort any write that would leave `/` < 2 GB. Never delete user caches |
| SN770 | 84 GB free. Holds BF16 Qwen3.8-27B + EXL3 4bpw (72.5 GB, `ARTIFACTS.sha256`, RL-140) | HQ4 states and outputs go here |
| T-FORCE / WD | 73 GB / 42 GB | — |
| Flash-Next | `/mnt/<sata-disk>/strata/models-iq3xxs/IQ3_XXS/` + pack, mtp-rt and `prompts/` | Verify against `results/2026/10/rq14-strata-matched-screen-20261009/ARTIFACTS.sha256` |
| Strata | Images `strata-sycl-serve:0.1.40.3` and `strata-sycl-dev:latest` in the main daemon. Source: `/mnt/<sata-disk>/strata/repo/strata-0.1.40.3`. Config of record: `serve-flashnext-iq3xxs.json` | Run with `--device=/dev/dri --group-add <render gid>`, mount `/mnt/<sata-disk>/strata:/work`, explicit `--entrypoint`, env `ONEAPI_DEVICE_SELECTOR=level_zero:<card> STRATA_VERIFY_NO_HOST=1 STRATA_VERIFY_DEVICE_PLAN=1 STRATA_STAGER_THREADS=12`. The first request exits 2 by design; discard it |
| Battery tooling | `scripts/tmp/rq17_battery_20261010.py`, `scripts/tmp/flashnext_battery_driver.sh` (sealed adapter that worked in RL-144) | Patch the budgets and scoring per §2; keep the adapter otherwise |
| HQ4 tooling | `scripts/tmp/hq4_states_20261010.py`, `scripts/tmp/hq4_f0_20261010.py`; EXL3 bit-exact `ref.py` hash pinned in `results/2026/09/b70-exl3-tp-cpu-slice-20260929/results.json` | Reuse the hidden states from the RL-150 run if hashes match; otherwise regenerate onto SN770 |
| VRAM telemetry | `sudo -n cat /sys/kernel/debug/dri/{0,1}/tile0/vram_mm` → `visible_avail` | Record before and after every cell |
| Power | As-found 180 W (GPU0) / 230 W (GPU1) yesterday | Record as-found now; restore at cleanup |
| Thermal | ≤ 55 °C cooldown is unreachable | Declare sustained-regime cells |

## 5. Lanes and specialists

| Lane | Hardware | Lead | Support |
|---|---|---|---|
| A Preflight, guard script, disk routing, caps | — | b70-runtime-engineer | b70-executor |
| B RQ-17 battery rerun | GPU0 | b70-correctness-engineer | b70-bench-engineer (plan, adapter patch), b70-runtime-engineer (server, guard) |
| C RQ-25 aligned IQ3_XXS isolated bench | GPU1 | b70-kernel-engineer | b70-isa-analyst (16-byte loads, no spills), b70-bench-engineer |
| D RQ-26 F0 re-screen | CPU | b70-quantization-engineer | b70-exl3xpu-specialist (ref.py only), b70-correctness-engineer (metric, sanity control) |
| E RQ-18 two-card layer split | both, after B passes and C is cleaned up | b70-llamacpp-specialist | b70-profiler (stage timings), b70-correctness-engineer (byte diff) |
| F RQ-25 route pairs (only if C passes) | GPU0 or GPU1 | b70-kernel-engineer | b70-bench-engineer, b70-correctness-engineer |
| G RQ-11 interactive boot repair (unsealed) | free card, last | b70-vllm-server-specialist | b70-upstream-researcher |
| Gates and pricing before GPU time | — | b70-architect | — |
| Review before registration | — | b70-architect-2 | b70-claims-auditor |
| Registry writes, render/check | — | b70-lead | b70-executor |

## 6. Phases

Every subagent returns `{passed, evidence[] (repo paths), blockers[], next}`. Gates
come from the program JSON. Never lower a gate after seeing data.

**Phase A — preflight (~30 min)**
1. runtime-engineer: write `scripts/tmp/gpu_guard_20261011.sh`. It lists PIDs on both render nodes, classifies them as desktop or inference, and exits non-zero on any inference PID. Run it now and keep the output.
2. Record as-found caps, temperatures and `visible_avail` per card.
3. Create the SN770 run directory and prove `/` keeps ≥ 2 GB after a test write.
4. lead: do the §3 registry re-entry; render and check must pass.

**Phase B — GPU0: RQ-17 battery rerun (~1 h)**
1. bench-engineer patches the battery per §2 items 1–2 and writes the plan JSON with budgets declared **before** the run.
2. Start the config of record and discard one warmup.
3. Run 6 canaries ×2, 10 host tasks, 3×3 replay and the 8K two-turn check, with the guard around every cell.
4. Score `content` only. Report reasoning/content tokens, `finish_reason`, acceptance and tok/round per cell.
5. `research.py run → assess → record`.
6. Pass → close RQ-17 `passed`; RQ-18 becomes `ready`.
7. A truly wrong answer or measured replay drift → ADR-0011 reopens. That is a route verdict, record it.
8. Truncation or contamination only → `blocked` with the cause. Do not call it a failure.

**Phase C — GPU1: RQ-25 isolated bench (~2 h code + 1 h GPU; parallel with B)**
1. Write in `scripts/tmp/` only. Compare the current IQ3_XXS `Multi<18>` load path against a structure-of-arrays repack: 16-byte-aligned `qs` plane, separate signs/scales plane, same values.
2. Use real expert tensors from the pack, columns 1–6, and a ring larger than L2. Allocate ≤ free − 4 GiB.
3. Correctness: the dequantized values are bitwise equal; the dot products are within the declared f64 tolerance (§2 item 6).
4. isa-analyst shows the 16-byte block loads and no scratch spills **before** any timing.
5. Three interleaved pairs per shape.
6. Gate: ≥ 1.5× GB/s. Below that, close RQ-25 and record the measured GB/s floor.

**Phase D — CPU: RQ-26 F0 re-screen (4–8 h; starts in Phase A, runs through)**
1. Write the manifest with the metric, sanity control and gates before any number.
2. Use the same 6 blocks, the same disjoint split (calibration [0,60 %), eval [62 %,…)) and the same 8 arms, plus the RTN-INT3 sanity arm.
3. AutoRound uses ≥ 100 iterations. Size the deadline from a measured probe.
4. If the sanity arm reads ~0 on any block, stop, fix the harness and do not score.
5. Gate unchanged: best arm KL ≤ 1.0× EXL3 on every block, and top-1 ≥ EXL3 − 0.1 pt, at ≤ 4.5 bpw.
6. Report every arm on every block, not just the best one.

**Phase E — both cards: RQ-18 (after B passes and C is cleaned up; ~2 h)**
1. Read the Strata `--layer-split` path and its residency output.
2. Price per-card VRAM against `visible_avail` − 4 GiB.
3. Require the engine line reporting 100 % experts resident.
4. Run 3 interleaved screen pairs: one-card config of record against layer split, with greedy output byte-identical.
5. Run 5 fresh pairs only if every screen pair clears +15 % decode with prefill ≥ 0.95×.

**Phase F — RQ-25 route pairs (only if C passes)**
1. Repack only if SN770 has ≥ 2× the pack size free after the HQ4 lane is done. Otherwise record it as blocked on storage and stop.
2. Patch the load path behind an env switch.
3. Run 3+5 decode pairs against the config of record, with byte-identical greedy output and 8K prefill not worse.

**Phase G — RQ-11 boot repair (unsealed; only if a card is free and time remains)**
1. Boot v0.30.0 with Nemotron GPTQ G64 + DFlash n7 interactively.
2. Capture docker stderr before the container exits.
3. Check the `--speculative-config` JSON quoting against `research/2026/v0300-nemotron-gptq64-dflash-n7-serve-source-20261010.md`.
4. Deliverable: one working argv and one correct answer. No sealed run, no speed number.

**Phase H — close-out**
1. architect-2 reviews every result, then claims-auditor checks the report against §2 item 5.
2. executor runs `python3 -m unittest discover -s tools -p 'test_*.py'`, `research.py render`, `research.py check` and `tools/build-indexes.py`.
3. lead updates `OUTCOMES.md` with today's rows (append; do not rewrite yesterday's).
4. runtime-engineer proves cleanup: owned containers stopped, caps restored to as-found, cards empty, `/` free space recorded.

## 7. Blocker policy

1. A new blocker gets one bounded fix by its owner. If it remains, record `blocked` with the exact command and output, and continue the other lanes.
2. Never work around a safety rule:
   - no allocation above free − 4 GiB;
   - no probes on GPU0;
   - no killing processes you did not start;
   - no driver or clock changes;
   - no stock TP2.
3. Never fabricate a sample, counter or completion marker. Missing evidence = `blocked`.

## 8. Measurement contract

Follow `docs/methods/benchmark-protocol.md`. Every cell declares:
- the counted quantity: engine tokens, client post-first rate, or chunks;
- actual input and output tokens, with reasoning and content counted separately;
- sampling, and cache state with hit-delta;
- card PCI id, as-found cap, temperatures, `visible_avail`, and energy from `energy1_input` deltas.

Run one client per C1 server, one discarded same-shape warmup, and interleaved pairs.
Write `measurements.json` per the operating guide.

## 9. Final report

1. `conclusion` in 3 sentences, stating exactly which cells and arms each claim covers.
2. Per question: `{id, verdict, median, pairs, quality, evidence_root, ledger_id}`.
3. `blocked[]` with exact causes.
4. `cleanup_proof`.
5. `next_question` from `research.py next`.

Label everything PROVISIONAL — NOT FOR PUBLIC HEADLINE.

---
