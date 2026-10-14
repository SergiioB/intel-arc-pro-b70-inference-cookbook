---
title: "B70 run-2 workflow report 2026-10-11 — RQ-17 battery blocked on an unsatisfiable plan conjunct; RQ-26 F0 blocked at the sanity floor; RQ-25 ISA round 2 compile-only; RQ-11 boot repair passed unsealed"
type: research
date: 2026-10-11
status: provisional
---

# B70 run-2 workflow report — 2026-10-11 (window 2, b70-host)

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Close-out over the four lane results
of the 2026-10-11 run-2 window, the adversarial red-team sweep delivered with
them, and this writer's own re-checks. Where the red team refuted a lane claim,
the corrected statement is carried below and the refuted wording is withdrawn —
not defended. No public benchmark, adoption, push, deployment or service change
is implied (AGENTS.md §12); `model-status.json` was not edited.

**Verification legend.** `[lead]` = command run by this writer during this
close-out (command and output stated). `[red-team]` = finding supplied in the
close-out ask, with the red team's own runs as its basis. `[lane]` = claim whose
basis is the sealed run root or lane digest supplied in the ask; where this
writer did not re-run it, it says so.

---

## 1. Conclusion (3 sentences)

The RQ-17 quality battery — six canaries in both interleaved runs A and B, three
replay prompts × three reps at 256 output tokens, ten host-executed tasks and the
8K two-turn KV edge on the adopted Strata IQ3_XXS one-card route — sealed with
every quality element passed (canaries 12/12 at finish stop including
batt-arith "703", replays token-identical, tasks 9/10, 8K turn-2 correct, zero
truncated cells, zero foreign inference PIDs across all 70 guard-log rows) and is
blocked solely by the predeclared `canary_prompt_token_exactness` conjunct, which
is unsatisfiable as declared because the chat surface renders a uniform
44-token template overhead with A==B per canary (66/64/70/75/72/66 — the same
+44 as RL-144's sealed raw), so no speed, energy or adoption number is claimed
from any cell. RQ-26 F0 — a six-block screen (gdn_l01/l30/l62, att_l03/l31/l63)
whose declared arms were AutoRound and GPTQ variants at two Hadamard settings,
the EXL3 reference and an RTN-INT3 sanity arm — is blocked with zero arms scored
because the sanity control read ~0 block-delta KL on att_l03 (fp32 −2.03e-20,
fp64 −2.03e-20 against the 1e-6 floor) while all 7 of that block's linears were
genuinely perturbed (rel_fro 0.2398–0.2570), and the sealed run additionally
contains 24 unscored jobs that ran after the failed sanity wave — 8 AutoRound
(gdn_l01/gdn_l30 × {uniform,e2m1} × {h0,h1}, iters=100, all-tensor INT8 escape
at bpw 8.125), 12 GPTQ (gdn_l01/gdn_l30/gdn_l62 × {uniform,e2m1} × {h0,h1};
gdn_l62's four AutoRound slots never ran), 3 completed EXL3 reference cells
(gdn_l01/l30/l62, kl_mean 1.55e-3/1.05e-2/5.53e-3 — the baseline side of the
aborted comparison) and 1 sanity-fix recheck — dispatched by a defect the lane
attributed post-seal to a scheduler indentation bug (an attribution with no
retained artifact, §4.3) — so the registry's recorded sentence "the campaign
was never dispatched, no AutoRound" is false as written, though the `blocked`
verdict is unaffected (no measurements.json, no scoring_table.json, no gate
evaluation). RQ-25's round-2 evidence is compile-only ISA fact about its two
microbenchmark variants (current AoS vs SoA; fresh AOT dumps, no GPU touched):
the SoA gu kernel achieves 16-byte q-plane loads (`load.ugm.d32x4.a64`, rd:8, at
dump lines 1058/2251/3087 `[lead]` — corrected from the lane's non-resolving
157/1350/2186) but still spills (ocloc "around 8" against the 0-spill gate);
RQ-18 was correctly never unlocked; RQ-11's "passed" covers only the unsealed
boot repair (att1 rc=139 SIGSEGV root-caused to `ZE_AFFINITY_MASK=level_zero:0`
plus `--speculative-config` quoting; att2 clean boot with greedy "Canberra."
correct) with the question's 20% task-energy gate never measured; and window
cleanup is proven and re-verified by this writer (caps 180/230 W byte-equal to
as-found by PCI-resolved hwmon, isolated unit inactive, no listeners, `/` at
9.0 GB free).

## 2. Per-question results

| id | verdict | median | pairs | quality | evidence_root | ledger_id |
|----|---------|--------|-------|---------|---------------|-----------|
| RQ-17 (battery) | blocked — single failing conjunct `canary_prompt_token_exactness`, a plan-contract defect; ledger outcome `failed` / assessment decision `rejected` (classification split disclosed, §4.1) | none — no speed or energy number sealed; the battery claims none | 1 interleaved A/B pair per canary (6 canaries × runs A+B; prompt tokens A==B at 66/64/70/75/72/66 `[lead]`); replay 3 prompts × 3 reps = 9 launches × 256 tok; 1 discarded same-shape warmup | canaries 6/6 in both runs at finish stop; replays token-identical ×9; host tasks 9/10 (gate ≥8; t5's frozen expectation [12,39] is itself wrong Python — true stdout 54/39 — not re-scored); 8K two-turn correct at 8161/8200 prompt tokens `[lane]` | `results/2026/10/RQ-17-20261011T114846278867Z-71a59de867` (SN770 lane logs via `SN770_POINTER.md`) | RL-156 |
| RQ-26 F0 | blocked — predeclared sanity rule fired on att_l03; no scoring | none — no scoring ran (no measurements.json, no scoring_table.json, no gate evaluation) | none — the 24 post-sanity-wave jobs that ran are unscored diagnostics dispatched in error, not pairs `[lead]`: 8 AutoRound (gdn_l01/gdn_l30 × {uniform,e2m1} × {h0,h1}, iters=100, bpw 8.125), 12 GPTQ (gdn_l01/gdn_l30/gdn_l62 × {uniform,e2m1} × {h0,h1} — gdn_l62's four AutoRound slots never ran), 3 EXL3 reference cells (gdn_l01/l30/l62) and 1 sanity-fix recheck | not measured — zero arms scored; sanity per-block KL read by this writer from `sanity_control.json`: gdn_l01 4.67e-05, gdn_l30 0.2300, gdn_l62 0.1519, att_l31 0.3803, att_l63 0.2553 pass; att_l03 fail | `results/2026/10/RQ-26-20261011T122026925511Z-0d5b5ff8b7` → NVMe 40.97 GB via `nvme_pointer.json` (387/387 verified `[red-team]`) | RL-157 |
| RQ-25 isolated (ISA round 2) | not_run as a benchmark — compile-only ISA evidence complete (AOT compile in `strata-sycl-dev:latest`, no GPU mounted) | none — no GPU execution | none | n/a — ISA level only: 16-byte loads met; spills 8 vs 0-spill gate unmet; 0 barriers in all four kernels; round 2 has no registry/ledger row (defect, §4.6) | `scripts/tmp/rq25_soa_20261011/isa/r2/` + `/mnt/<nvme-disk>/b70-run2-20261011/rq25/isa-devimg-r2/` | none — unrecorded (defect) |
| RQ-25 route pairs | not_run — isolated gate not passed | – | – | – | – | none owed |
| RQ-18 | not_run — gated on RQ-17 (registry excluded list: `gated`) | – | – | – | – | none owed |
| RQ-11 | passed — **unsealed boot repair only**; the question itself stays closed/rejected (`run_root`/`ledger_id` empty by design; POINTER.md: "NOT a sealed run, no registry record") | none — the 20% task-energy gate was never measured | none | diagnostic only: one greedy factual response "Canberra." correct; att2 clean rc=0 shutdown `[lane]`, SHA256SUMS 9/9 `[red-team]` | `/mnt/<nvme-disk>/b70-run2-20261011/rq11/boot-repair/` + `results/2026/10/rq11-boot-repair-20261011-UNSEALED/POINTER.md` | none (unsealed by design) |

Ledger rows added this window `[lead]`: RL-152 (RQ-17, provisional re-entry
decision), RL-153 (RQ-26, provisional), RL-154 (RQ-26, provisional — RL-145
pointer repair), RL-155 (RQ-26, provisional — screen timeout 3600→7200 s from
probe evidence), RL-156 (RQ-17, failed), RL-157 (RQ-26, blocked).

## 3. blocked[] — exact causes

1. **RQ-17** — `measurement_validity` fails on exactly one conjunct,
   `canary_prompt_token_exactness`, which compared `/v1/chat/completions`
   rendered `prompt_tokens` to the bare batt-file id counts (22/20/26/28/31/26
   expected). The serve surface renders a uniform +44-token chat-template
   overhead, identical across arms: observed A==B per canary (66/64/70/75/72/66,
   `[lead]` read of `raw/serve/*.json`), and RL-144's sealed raw shows the same
   +44 (canary-A-batt-arith `usage.prompt_tokens` 66 vs file 22, `[lead]`).
   The conjunct is unsatisfiable as declared on the chat surface; the property
   it guards (identical prompts across arms) is proven by A==B. Attempt 1 of 2
   consumed (RL-156). Re-entry requires the declaration amended to the rendered
   surface. Not a quality failure and not a route failure; ADR-0011 stays
   closed (stop rule never fired).
2. **RQ-26 F0** — att_l03 block-delta KL = −2.0342040164188262e-20 (fp32) and
   −2.0342038410203182e-20 (fp64, after the one bounded fix round) against the
   1e-6 floor `[lead]` read of `sanity_control.json`, with per-tensor rel_fro
   0.2398–0.2570 on all 7 linears: the bf16 block forward cannot represent even
   a 3-bit weight perturbation on that block, so any metric through it is
   structurally blind. F0 as declared (gate on 6 blocks incl. att_l03) is
   unanswerable by this metric. RL-155 bars attempt 2 without a new mechanism
   or fresh authorization.
3. **RQ-25** — isolated bench or ISA gate not approved; substance: spills
   reduced 12→8 but the 0-spill gate stands unmet; additionally the round-2
   finding is unrecorded and the registry `next_action` is stale (§4.6).
4. **RQ-18** — gated on RQ-17 passing; never unlocked (correct).
5. **Window handoff hygiene (found failing, FIXED by this writer)** —
   `python3 tools/research.py check` returned "stale generated views:
   docs/indexes/research-coverage.md" / "FAIL: research registry check"
   `[lead]`, because the cleanup lane added `results/2026/10/run2-20261011-cleanup/`
   after the last render. Fixed: `python3 tools/research.py render` (rewrote 4
   generated views) → `check` = "OK: research registries valid, generated views
   current" `[lead]`. RQ-17's sealed "views re-rendered, research.py check OK"
   claim was true only at its record time (~15:48Z); it is not re-asserted here
   as a property of the final tree — the final tree is the post-fix state.
6. **Registry false provenance, RQ-26 (material, correction outstanding)** —
   see §4.2. Requires operator/registry action; this writer did not edit the
   recorded registry row.

## 4. Red-team findings — incorporated, corrected, not defended

1. **Classification split (RQ-17), disclosed.** Ledger RL-156 `outcome: failed`
   and `assessment.json` decision `rejected` `[lead]` read), while lane and
   registry narrative both class the run a plan-contract defect ("BLOCKED … Not
   a quality failure"; one attempt reserved). Nothing is hidden; the durable
   field downstream consumers filter on reads `failed`. The unsatisfiability
   argument is corroborated by primary evidence this writer re-read: all six
   deltas exactly +44 with A==B, and RL-144 raw shows 66 vs 22.
2. **Registry false provenance (RQ-26, material).** The RQ-26 `next_action`
   recorded with RL-157 states "no AutoRound; the campaign was never dispatched"
   `[lead]` read of `registry/research-program.json`. The sealed run root
   contradicts it: 8 AutoRound job files (`jobs_arm_gdn_l{01,30}_ar-{uniform,e2m1}-h{0,1}.json`,
   each `ar_iters: 100`, `block_bpw: 8.125`), 12 GPTQ job files
   (`jobs_arm_gdn_l{01,30,62}_gptq-{uniform,e2m1}-h{0,1}.json` — gdn_l62's
   four AutoRound slots are `false` in `blocked_incomplete.json`
   `inventory.arms` and never ran `[lead]`), 3 completed EXL3 reference cells
   (`jobs_exl3_gdn_l{01,30,62}.json`, 271.2/251.3/255.7 s, kl_mean
   1.553e-3/1.046e-2/5.531e-3 `[lead]`; the three att-class EXL3 inventory
   entries are null), the sanity-fix recheck (`jobs_sanityfix_att_l03.json`),
   and 155 `ar-*` entries in `nvme_pointer.json` `[lead]`. Post-wave job
   mtimes span 12:41 UTC (first GPTQ) to 13:19:39Z (sanity-fix recheck; last
   arm job 13:19:36Z), after the failed sanity wave (12:36:45–12:37:21Z
   `[lead]`); the blocked state itself was written 13:23:40Z
   (`blocked_incomplete.json` `utc` field `[lead]`), not by a job mtime. The
   dispatch cause is the lane's post-seal attribution to a scheduler
   indentation bug; because the sealed driver was overwritten in place (§4.3),
   that attribution has **no retained artifact** — it is disclosure, not
   evidence. The sentence mirrors `sanity_control.json`'s `rule`
   field ("harness broken: blocked, no scoring, no AutoRound; the campaign was
   never dispatched" `[lead]`), which the lane's own result already admitted
   was false for this run. **Corrected statement carried everywhere in this
   close-out: the campaign WAS dispatched in error — 24 jobs ran unscored
   (20 arm jobs, 3 EXL3 reference cells, 1 sanity-fix recheck); the
   verdict is unaffected because no scoring, no measurements.json and no gate
   evaluation exist.** The registry row itself still needs an operator-level
   correction (this writer is not authorized to rewrite recorded registry
   rows).
3. **Sealed adapter not retained (RQ-26, §12 gap).** `identity.json` records
   `adapter_sha256 48187f5b…` for `scripts/tmp/rq26_f0_driver_20261011.py`; the
   file now hashes `80b68d9c…` `[lead]`, mtime 15:48:20Z (+0200) — after run end
   (complete.json mtime 13:24:16Z `[lead]`) — overwritten in place by the
   post-seal fix. Red team's `find` over repo run root and the entire NVMe rq26
   tree returns no copy `[red-team]`. The exact sealed driver is unrecoverable
   except by digest. Recorded as an evidence-retention defect; not repairable
   by this writer.
4. **Guard count off-by-one (RQ-17, minor).** Sealed `measurements.json`
   (`"guard_calls": 69`) and `checks/measurement_validity.json`
   (`"calls": 69`) vs `raw/guard/guard_log.jsonl` = **70** rows, all `rc=0`,
   zero foreign inference PIDs, tags spanning preflight → battery-start →
   warmup/canary/task/kv8k/replay before/after `[lead]`. The cleanliness claim
   holds; the count is wrong by one. Corrected count used in §1.
5. **Amendment disclosure unmet at ledger level (RQ-17, minor).**
   `scripts/tmp/rq17_plan_amendment_20261011.md` is genuinely pre-seal (mtime
   11:42:53Z `[lead]` < manifest `created_at` 11:48:46.280379+00:00 `[lead]`)
   and promised the amendment would be carried into the ledger row, but RL-156
   contains no amendment text `[lead]` read of the full row. The amendment IS
   disclosed in sealed `measurements.json` (`controlled.config.vram_budget`)
   and `SN770_POINTER.md`; the lead authorization itself is asserted in the
   note, not evidenced by any artifact this writer can read.
6. **RQ-25 citation defects (substance stands).** The lane's cited dump lines
   157/1350/2186 (SoA d32x4) and 171/1445/2341 (current d32x2) resolve nowhere:
   line 157 of the SoA dump is a `mov` `[lead]`; the actual d32x4 sites are
   **1058/2251/3087** and the actual current d32x2 sites are
   **72/515/1095/2369/3265** (5 in the whole dump) `[lead]` grep. The lane's
   quoted check `grep -cE 'and .*, 2:w'` = 0 is vacuous (0 on both variants
   `[red-team]`) and is withdrawn; the corrected
   `grep -cE '\band\b.*\s2:w\b'` gives **current 10 / SoA 0** `[lead]`, which
   does prove the misalignment-path claim. Everything decisive verifies
   independently `[red-team]`: SHA256SUMS 8/8; SoA 6 spill stores + 10 fills
   with the exact type census; current 4 stores + 9 fills; ocloc "around 8"
   (r2) vs "around 12" (r1) vs "around 4" (current); ze_info grf_count 128 /
   simd_size 32; 0 barriers; the two scratch fills feeding q-row bases before
   the d32x4 loads. Additionally: the round-2 ISA finding has **no registry or
   ledger record** and RQ-25's `next_action` is stale ("run it on GPU0 when the
   card is free", dated 2026-10-10) `[lead]` read.
7. **RQ-11 label scope + minor number.** "Passed" covers only the unsealed
   boot repair; the question stays closed/rejected, nothing was measured
   against the 20% task-energy gate, and a headline read of "RQ-11: passed"
   must not be mistaken for the question's success gate. The lane's
   `sn770_bound_bytes: 0` is wrong: ~55 KB was written to
   `/mnt/<nvme-disk>/b70-run2-20261011/rq11/` (draft_argv.sh,
   working_argv.sh, boot-repair/; `du -sb` = 55,495 B `[lead]`; the close-out's
   first pass said ~100 KB — overstated ~2×, corrected).
8. **Cleanup vs RQ-11 'next' tension (disclosed, defensible).** RQ-11's next
   wanted the isolated daemon left up for a follow-up sealed run; that run
   never happened before window close, so cleanup stopped the owned unit —
   the condition lapsed. Verified live by this writer (§5).

## 5. cleanup_proof

- **Proof artifact:** `/mnt/<nvme-disk>/b70-run2-20261011/cleanup_proof.md`,
  sha256 `749c018133b8a1789d1b3b5c0a70a1f1beb64f407e6d85506e1941b01ac92aba`
  — re-hashed by this writer during this close-out: exact match `[lead]`.
  Repo pointer: `results/2026/10/run2-20261011-cleanup/POINTER.md`.
- **Live end state re-verified by this writer (~17:00Z)** `[lead]`:
  `systemctl is-active/is-enabled <exl3-dockerd>d8.service` →
  `inactive` / `not-found`; `/run/<exl3-docker>.sock` absent; no
  `dockerd …b70-exl3` process (only the pre-window system root dockerd, PID
  5794, `-H fd://`); bridge `<exl3-bridge>` present, state DOWN (deliberate
  leftover per proof §6); no listeners on 8015/8000/8080/8001/11434/8095; no
  vllm/sglang/llama/strata processes; `/` free 9,042,714,624 B (≥ 2 GB floor;
  the ~28 MB drop from the red team's 16:36Z reading is this writer's render
  plus the close-out writes).
- **Caps equal as-found** `[lead]`: hwmon resolved by PCI device symlink —
  0000:0b:00.0 → hwmon5 `power1_cap=180000000` uW; 0000:0f:00.0 → hwmon6
  `power1_cap=230000000` uW — both byte-equal to
  `preflight/asfound.json` (180000000 / 230000000). No clocks, driver or PCI
  changes were made by any lane or by this writer.
- **SN770 integrity** `[red-team]`: RQ-26 nvme manifest 387/387 files present,
  content sha256 verification 387/387 OK (40,965,581,463 B); rq17/rq11/preflight
  manifests re-verified OK; aborted RQ-26 launch root retained verbatim
  (MOVED.txt). Nothing was deleted; the SN770 free-space delta vs preflight is
  retained artifacts (~41 GB rq26 + ~1.1 GB rq25), i.e. growth.
- **Tension disclosed (§4.8):** the daemon stop contradicts RQ-11's stale
  "leave it up" only because the follow-up sealed run never occurred; the
  bridge remains as documented standing config, with the optional
  `sudo ip link del <exl3-bridge>` left to operator preference.

## 6. next_question (from `python3 tools/research.py next`, run by this writer)

**RQ-17** — status `ready`, priority 2, lane `correctness`
("Flash-Next quality closure: representative tasks, long/multi-turn and replay
drift on the adopted Strata IQ3_XXS route"). The registry's own next_action
records the RL-156 block and states: "One attempt of the new mechanism remains
(max_attempts 2, attempt 1 consumed by RL-156)." `execution_authorized` was
returned `false`, so the next run starts through the loop/operator, not
autonomously. The lane-consistent re-entry (matching the registry's battery
specification) is: amend the token-exactness declaration to the rendered
surface — file id counts plus the measured 44-token template overhead, or A==B
prompt-token equality per canary — as an explicitly amended mechanism, and
spend attempt 2 of 2 on the unchanged route with gates unchanged; ADR-0011
stays closed in the meantime (no wrong answer, no replay drift measured).
RQ-26 is also `ready` (one attempt remains, RL-155-gated) and RQ-25 `ready`
(stale next_action, §4.6); `next` ranks RQ-17 first.

## 7. Window hygiene actions by this writer (this close-out)

- `python3 tools/research.py check` → FAIL (stale views) `[lead]`; then
  `python3 tools/research.py render` → wrote `research/PROGRAM.md`,
  `research/LEDGER.md`, `models/STATUS.md`, `docs/indexes/research-coverage.md`;
  then `check` → "OK: research registries valid, generated views current"
  `[lead]`. These are generated views; **`registry/model-status.json` was not
  touched** (mtime 2026-10-09 17:19 +0200 `[lead]`; its `M` flag in git
  predates this window).
- `OUTCOMES.md`: window-2 section appended below the pre-window content
  (which ended at line 54, "3. RQ-11 container-boot diagnosis + re-author
  authorization."); this writer's only further OUTCOMES.md operations were the
  in-place corrections, listed in the section's correction note, to this
  window's own rows. **Verification gap, stated plainly:** the file is
  untracked (`git status --porcelain OUTCOMES.md` → `?? OUTCOMES.md`
  `[lead]`), so "nothing above was rewritten" cannot be diff-audited from the
  repo; it rests on this writer's session-anchored edit, not on a committed
  baseline.
- This report created. No GPU cell was run by this writer (no guard cell
  therefore applies); no caps, clocks, driver or PCI state changed; nothing
  deleted; no other file edited.
- Claims-audit corrections applied post-write (same close-out; corrected
  values re-measured by this writer from the sealed run root): the RQ-26
  dispatch-in-error inventory was expanded from 20 to 24 jobs (the 3 completed
  EXL3 reference cells and the sanity-fix recheck had been omitted; the 12
  GPTQ jobs' gdn_l01/l30/l62 coverage and gdn_l62's missing AutoRound jobs
  made explicit); the AutoRound loss-divergence wording was re-scoped from a
  single-cell quote to the measured aggregate (all 48 per-tensor traces rise,
  6.5×–160.8×, 1 of 48 non-monotone); the job-time bound "12:41–13:23 UTC"
  was corrected to job mtimes 12:41–13:19:39Z with the 13:23:40Z blocked-state
  write named as such; the rq11 NVMe write was corrected from ~100 KB to
  55,495 B (`du -sb`); and the OUTCOMES.md append-only claim was downgraded to
  a session-anchored statement with the untracked-file verification gap
  disclosed.
- Not done / not possible, stated plainly: the RQ-26 registry row's false
  sentence was NOT rewritten (recorded registry rows are outside this writer's
  authorization — flagged as correction owed, §4.2); the sealed RQ-26 driver
  could NOT be recovered (§4.3); RL-144's remaining five canary files and the
  RQ-26 NVMe content verification were taken on red-team runs, not re-run
  byte-for-byte by this writer (spot re-reads are labeled above).

---

*Every number above is PROVISIONAL — NOT FOR PUBLIC HEADLINE. No gate was
lowered after seeing data; no sample, counter or completion marker was
fabricated; missing evidence is reported as blocked, not passed.*
