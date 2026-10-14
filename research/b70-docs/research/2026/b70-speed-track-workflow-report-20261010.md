---
title: "B70 speed-track workflow report 2026-10-10 — four GPU lanes blocked pre-measurement; RQ-22 priced at zero GPU"
type: research
date: 2026-10-10
status: provisional
---

# B70 speed-track workflow report — 2026-10-10

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Internal workflow report over verified
lane inputs, an adversarial sweep verdict and this writer's own re-checks (each is
labeled). No public benchmark, adoption, push or service change is implied
(AGENTS.md §12). This session produced no paired sample, no median and no quality
number; the only quantitative deliverable is a zero-GPU price.

## 1. Conclusion

All four GPU lanes the speed track opened on 2026-10-10 ended before any
measurement — RQ-17 and RQ-26 F1 blocked at preflight (RL-132, RL-134), RQ-26 F0
blocked at the staging script's cross-device guard and then overtaken by a live
re-run that its registered row does not describe (RL-133; superseding row still
outstanding at write time), and RQ-11 blocked at an undiagnosed preflight failure
(RL-135) — so RQ-18 and RQ-25 were correctly skipped with no ledger rows owed,
because RQ-17 was blocked at preflight and never failed (ADR-0011 not reopened;
registry: RQ-18 status `gated`, dependencies `['RQ-17']`; the plan flowchart's
fail→reopen-ADR-0011 branch did not fire). The one sanctioned zero-GPU
deliverable, the RQ-22 INT8-prompt-GEMM price (RL-131), keeps RQ-22 open and
`deferred`: at the measured 37.30 % expert-GEMM share of the 8K prefill
(970 + 2464 = 3434 ms of the 9207 ms GPU timeline, `base.log:45`), a 2× local s8
speedup prices at 1.229241× whole-prefill — 2.05× above the 10 % close threshold
(18.18 % share needed) but permanently below the registered ≥ 1.3× gate (needs
46.15 %), so future GPU hours are justified only for the fused s8 dequant+GEMM
design (50.59 % share → 1.338616×) as an isolated parity probe once RQ-17/RQ-18
free a kernel budget. Host state at write time: cleanup is proven and re-verified
by this writer (isolated daemon inactive, socket gone, caps 180 W/230 W as-found,
both engine images retained on the data-root); the F0 staging re-run is LIVE
(5/18 bf16 shards on disk, shard 6 at 27.0 % at 08:59:23Z, 11 MB/s); and the only
standing preflight blocker is the per-card live-VRAM tooling gap, since the
nemotron-hermes restart-policy cause is resolved (`"RestartPolicy":{"Name":"no"`
in the retained `hostconfig.json`, re-verified by this writer after the sweep's
live inspect).

## 2. Per-question results

| id | verdict | median | pairs | quality | evidence_root | ledger_id |
|----|---------|--------|-------|---------|---------------|-----------|
| RQ-17 | blocked (preflight) | – | – | – | – (preflight causes recorded inline in RL-132 only; no command-output file retained) | RL-132 |
| RQ-26 F0 | staging in progress at write time (raw outcome said "blocked (staging)" — corrected per adversarial sweep; RL-133 is stale) | – | – | – | `scripts/tmp/stage_b70_speed_20261010.log` | RL-133 (superseding row required — not yet registered) |
| RQ-26 F1 | blocked (preflight) | – | – | – | – (inline in RL-134) | RL-134 |
| RQ-11 | blocked (preflight failed; cause undiagnosed) | – | – | – | – (inline in RL-135) | RL-135 |
| RQ-22 | price delivered (PROVISIONAL, zero-GPU; stays `deferred`) | – | zero-GPU | n/a (quality objective deferred to the future kernel probe) | `research/2026/rq22-int8-prompt-gemm-price-20261010.md` | RL-131 |
| RQ-18 | skipped — RQ-17 blocked at preflight, gate not passed (never failed) | – | – | – | – | none owed |
| RQ-25 | skipped — sequenced behind RQ-17 (registry next_action: "run it on GPU0 after RQ-17 frees the card"); isolation-first half backed by its own smallest_probe | – | – | – | – | none owed |

Ledger state at write time (verified by this writer): `registry/research-ledger.json`
holds 135 rows; the session added RL-131–RL-135 (outcomes: validated ×1, blocked
×4). No lane was recorded as failed anywhere this session. The raw outcomes array
showed empty ledger fields for RQ-17/RQ-26 F0/F1/RQ-11; the real row IDs above
supersede those fields.

## 3. Blocked list — exact causes

### 3.1 RQ-17 and RQ-26 F1 (RL-132, RL-134)

1. **nemotron-hermes restart policy — RESOLVED.** Both rows record
   `restart=unless-stopped` on the container of the isolated daemon
   (`vllm/vllm-openai-xpu:v0.30.0`), so any `<exl3-dockerd>d5` restart would
   auto-revive it onto `/dev/dri/renderD129`. The adversarial sweep's live
   `docker -H unix:///run/<exl3-docker>.sock inspect nemotron-hermes` returned
   `RestartPolicy.Name='no'`, State `exited` — the named owner action was
   evidently performed (who/when: unknown, no timestamped evidence retained —
   stated as not verifiable). This writer re-verified after cleanup took the
   daemon down (no live inspect possible without undoing cleanup): the retained
   data-root's `containers/305e62c0…/config.v2.json` names the container
   `/nemotron-hermes` and its `hostconfig.json` contains
   `"RestartPolicy":{"Name":"no",`. Combined with the durable `Exited (0)` stop,
   the container will not auto-revive.
2. **Per-card live VRAM unreadable — STANDING.** Re-verified by this writer at
   write time: `ls /sys/class/drm/card1/device/mem_info` → `No such file or
   directory` (no mem_info sysfs on this xe build) and `xpu-smi discovery` →
   `No device discovered` (no Battlemage support). Every pre-load gate that needs
   live free VRAM needs another mechanism (xe debugfs client memory or an
   engine-side report) and should budget conservatively. This, not the restart
   policy, is the standing preflight limitation to fix before re-entry.

### 3.2 RQ-26 F0 (RL-133 — contradicted by its own cited evidence)

- Registered cause, verbatim (log `scripts/tmp/stage_b70_speed_20261010.log`,
  2026-10-10T08:01:08Z): `FATAL: / is on a different device than
  /mnt/<nvme-disk>` — with SN770 free at start 171,553,914,880 B, so
  space was never the blocker.
- Reality: the retained log carries FOUR sessions — the 08:01:08Z FATAL refusal;
  an 08:32:58Z dry run refused on a stricter ("ask") gate; an 08:34:22Z dry run
  PASSING under the program-doc gate (workflow prompt Phase A step 2,
  `docs/plans/2026-10-09-zcode-workflow-prompt.md`: free ≥ 2× download,
  171.5 GB ≥ 144.9 GB — faithful to the written plan, not a silent gate
  lowering); and the 08:34:34Z real run (PID 1627057) downloading with per-shard
  sha256 verification.
- Live at write time (this writer): 5/18 bf16 shards complete on
  `/mnt/<nvme-disk>/b70-bf16-qwen38-20260926/`, shard 6/18 in flight
  at 27.0 % (11 MB/s, log line 08:59:23Z), 15,606,526,489 B staged (du, 08:57Z);
  totals to fetch: 55.6 GB BF16 + 16.9 GB EXL3 4bpw @113cf7ab (registry RQ-26
  next_action).
- Therefore RL-133's "Nothing was downloaded" and its re-entry condition
  ("satisfy the cross-device guard") are stale — the guard no longer fires. The
  ledger is append-only: a SUPERSEDING row is required (never edit RL-133).
  Verified outstanding: no row other than RL-133 itself mentions RL-133 in the
  135-row ledger at write time.

### 3.3 RQ-11 (RL-135)

Exact recorded cause: "preflight failed" — nothing further retained: no
mechanism, no numbers, no diagnosis (honestly declared; the thinnest of the four
rows, per blocker policy step 2). Re-entry requires a diagnosed preflight cause
on the v0.30.0/Nemotron stack before any GPU work; source-side feasibility is on
file (`research/2026/v0300-nemotron-gptq64-dflash-n7-serve-source-20261010.md`:
"supported, provisional").

### 3.4 Evidence-currency and wording defects found by the sweep (fix before quoting this report onward)

- Raw-outcome verdicts "RQ-26 F0 blocked (staging)" and "RQ-18 skipped (RQ-17
  failed)" were materially false and are corrected in §2/§3 (the gate was never
  evaluated). RQ-25's skip is likewise worded "RQ-17 not passed (blocked)".
- Dangling citation, still unfixed at write time (re-verified by this writer):
  `research/2026/rq22-int8-prompt-gemm-price-20261010.md:62` cites
  `RESULTS.md:241-249`, but
  `results/2026/10/rq14-strata-matched-screen-20261009/RESULTS.md` has exactly
  130 lines (`wc -l`) and no cap-sensitivity section, and no external copy exists
  in the root. RL-131 does not propagate it (cites `base230.log:45`, which this
  writer confirmed carries the phase table at line 45). Correct the note before
  its evidence root is treated as sealed prose.
- Numeric nits, none verdict-changing: the price JSONs round the fraction
  slightly up (0.37298 vs exact 3434/9207 = 0.372936 → 1.229241 vs 1.229239;
  fused 0.50592 vs 4658/9207 = 0.505874 → 1.338616 vs 1.338581) — immaterial
  against the 2.05× margin; the note's "within 0.2 %" should read "within
  0.21 %" (RL-131's phrasing; max per-phase delta base230 vs base is 0.206 %);
  the outcomes array's "gemm down 2434/2464 ms" is a transcription slip — every
  artifact (base.log:45, both price JSONs, RL-131) reads 2464; the staging log
  double-logs every line (cosmetic).
- Not verifiable, stated as such: who/when set RestartPolicy to `no`; that
  `registry/research-program.json` was unmodified during the session (no
  in-session baseline retained; git shows only the pre-session 2026-10-09
  revision vs HEAD — nothing contradicts the lane's sha256-unchanged claim);
  sibling steps' in-flight outputs (workflow introspection unreachable from this
  subagent).

## 4. Cleanup proof

Two layers: (a) this writer's write-time re-verification (commands run here);
(b) the workflow cleanup step's own Phase-H proof (attributed input, not re-run).

Re-verified by this writer at write time:

- `systemctl is-active <exl3-dockerd>d5` → `inactive`; `ls
  /run/<exl3-docker>.sock` → `No such file or directory`.
- Caps with live PCI identity (readlink of hwmon device): hwmon5 → PCI
  `0000:0b:00.0` (GPU0), `power1_cap` = 180000000 (180 W); hwmon6 → PCI
  `0000:0f:00.0` (GPU1), `power1_cap` = 230000000 (230 W) — equal to the as-found
  targets; no power-cap write was performed anywhere in the workflow.
- Data-root retained on NVMe:
  `/mnt/<nvme-disk>/<exl3-dockerd>/` holds both container configs
  (`305e62c0…` = `/nemotron-hermes`, vllm/vllm-openai-xpu:v0.30.0; `742a803e…` =
  ghcr.io/0xsero/exl3xpu@sha256:21412bdd…) and, per the cleanup step, both
  images (vllm/vllm-openai-xpu:v0.30.0 sha256:fc0e112a…,
  ghcr.io/0xsero/exl3xpu:0.1.0 sha256:21412bdd…) — kept for future questions.
- F0 staging artifacts on the same NVMe retained and growing (not deleted).

From the cleanup step (Phase H step 4, attributed): BEFORE/AFTER `docker ps -a`
on both sockets — main daemon `/run/docker.sock`: 10 containers, all foreign
ods-pixel/openclaw (8 Up ~10 h, 2 Exited ~26 h), deliberately untouched (never
kill someone else's workload); isolated daemon: nemotron-hermes only, `Exited
(0)`, and it stayed Exited after daemon shutdown. Owned-process sweep found only
the workflow daemon itself (no vllm/strata/exl3/llama compute clients;
`fuser -v` over renderD128/129, card0/1 shows desktop-session clients only).
Torn down: the daemon process (clean shutdown removed pidfile and the transient
unit — `systemctl status` → "could not be found"), the socket, bridge
`<exl3-bridge>` (created by this workflow; as-found absent), and the
`/run/<exl3-dockerd>` exec-root including a boot-time `netns/default` mount
remnant of this workflow-owned path. Temps unchanged before/after (GPU0
54/56 °C, GPU1 51/54 °C).

Flags for the next campaign: renderD129 is not strictly client-free — the
desktop Electron app "ZCode" PID 1440463 (child of gnome-shell, uid 1000) holds
it open; not workflow-owned, not killed per the never-kill rule, no compute
activity in debugfs (benign enumeration), but GPU1 campaigns should expect the
render node not to be empty. The user's desktop session recycled between
preflight and cleanup (Xorg/zcode PIDs swapped) — user-managed, untouched.

## 5. Next question

`python3 tools/research.py next` (run by this writer, read-only) selects **RQ-17**
(status `ready`, no dependencies): "Flash-Next quality closure: representative
tasks, long/multi-turn and replay drift on the adopted Strata IQ3_XXS route."
Registry next_action, verbatim: "At the next authorized GPU window: record caps,
confirm the Strata server is owned or stopped by the operator, run the P0
battery, register the result and close RQ-17 passed or rejected." Its uncertainty
field's "leftover unhealthy container" concern is resolved (durable stop +
restart=no) but the re-check-at-launch instruction stands until the row is
updated.

Critical path behind it (registry + sweep):

1. F0 staging completes (≈ 60 GB remaining at the observed 11 MB/s ≈ 1.5 h — an
   estimate, not a measurement) → register the superseding RL-133 row.
2. RQ-17 P0 battery at the next authorized GPU window; passing it unblocks
   RQ-18 (`gated`, dependencies `['RQ-17']`) and frees the card for RQ-25's
   isolated SoA microbenchmark.
3. RQ-26 F1 and every pre-load VRAM gate need the per-card live-VRAM mechanism
   resolved first (§3.1 cause 2).
4. RQ-22 (deferred) next GPU step, only when a kernel budget opens after
   RQ-17/RQ-18: the isolated fused s8 dequant+GEMM parity probe on real shapes,
   carrying the AGENTS.md §11 quality objective (s8 activations change outputs).
   A GEMM-swap-only change is closed off by RL-131's blocked_pattern (caps at
   1.229× < the registered ≥ 1.3× gate). The 2× is a measured device rate class
   (s8 363 TOPS vs bf16 170 TFLOPS, `research/2026/xmx-dpas-rate-classes-20260915.md:35-36`),
   not a measured application speedup — the 0.22× fused-kernel precedent on
   another card (`research/2026/flashnext-strata-source-math-20261009.md:40-44`)
   is why the probe is isolated before any route claim.

## 6. Provenance and gates

- Inputs (script-provided, verified upstream; quoted, not re-derived): the lane
  outcomes array; the adversarial sweep verdict; the cleanup step's Phase-H
  output. Everything this writer ran is quoted with its command inline; anything
  not re-run is attributed, never claimed as own.
- Repo gates round 1 (workflow run, given input; commands per
  `docs/plans/2026-10-09-zcode-workflow-prompt.md:143`): unittest=0, render=0,
  check=0, indexes=0.
- Round 2, run by this writer immediately after this report was written (exact
  commands, exit codes): `python3 -m unittest discover -s tools -p 'test_*.py'`
  → exit 0 (74 tests, OK); `python3 tools/research.py render` → exit 0 (rewrote
  PROGRAM.md, LEDGER.md, models/STATUS.md, research-coverage.md);
  `python3 tools/research.py check` → exit 0 ("OK: research registries valid,
  generated views current"); `python3 tools/build-indexes.py` → exit 0 (550
  documents catalogued). `python3 tools/check-frontmatter.py` → exit 1 with 20
  violations, all pre-existing paths (none created this session; grep for this
  report's filename in the violation list: 0 hits — the file carries
  title/date/type/status). `python3 tools/check-links.py` → exit 1 with exactly
  one broken reference: `campaigns/2026/rq17-flashnext-quality/plan.json` in
  `docs/plans/2026-10-09-zcode-workflow-prompt.md` — the pre-existing Phase-B
  forward reference named by the sweep, whose target was never created; not a
  session defect, but it must exist before the next Phase-B run claims its plan.

## 7. Run 2 (follow-up) — 2026-10-10 same-day update

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Appended by the report writer from the
run-2 verified inputs (lead outcomes array, adversarial sweep verdict, Phase-H
cleanup proof, repo gates round 1); the run-1 sections above are kept verbatim and
are superseded only where this section says so explicitly. No paired sample, median
or quality number exists for any run-2 lane.

### 7.1 Conclusion

Run 2 again produced no measurement: RQ-17, RQ-26 F1 and RQ-11 were blocked at
preflight with no run (RL-132, RL-134, RL-135 — no lane failed a gate), while RQ-18
and RQ-25 were skipped with no ledger rows owed, RQ-18 because RQ-17 was blocked at
preflight and its gate was never evaluated (ADR-0011 still `final`), RQ-25 purely by
plan sequencing (registry `ready`, no dependencies, its isolated ≥ 1.5× gate never
evaluated). The one forward step is RQ-26 F0 staging, now complete and verified:
after RL-133 (cross-device-guard refusal) and RL-136 (fail-closed crc32
finalization, no manifests), session 5 (`scripts/tmp/stage_b70_speed_20261010.log`
lines 464–629, 10:27:42–10:29:19Z) ended `exit=0 problems=0 caveats=6` and wrote
both `ARTIFACTS.sha256` manifests (32 bf16 + 17 exl3 entries), whose on-disk
presence, entry counts and self-hashes this writer re-verified — the raw outcome's
"skipped (staging not confirmed)" is stale and is corrected here, while the F0
quality screen itself has NOT been run. Cleanup is re-proven after an unexplained
12:00:16 CEST daemon restart (actor not determinable from retained evidence;
restart=no behaviorally confirmed — nemotron-hermes stayed `Exited (0)` across it):
the owned daemon is down again as of this writer's 12:48 CEST re-check, caps
as-found 180/230 W, no compute clients, and the registry's top selection is
unchanged (RQ-17, `ready`).

### 7.2 Per-question results (run 2)

| id | verdict | median | pairs | quality | evidence_root | ledger_id |
|----|---------|--------|-------|---------|---------------|-----------|
| RQ-17 | blocked (preflight) — no run | – | – | – | – (no Oct-10 results root exists; exact causes quoted inline in RL-132) | RL-132 |
| RQ-26 F1 | blocked (preflight) — no run | – | – | – | – (inline in RL-134) | RL-134 |
| RQ-11 | blocked (preflight) — no run, cause undiagnosed | – | – | – | – (nothing retained beyond "preflight failed"; §5.2 gap, §7.3) | RL-135 |
| RQ-25 | skipped — plan sequencing only; no gate evaluated, no registry hold | – | – | – | – | none owed |
| RQ-18 | skipped — RQ-17 blocked at preflight, gate never evaluated | – | – | – | – | none owed |
| RQ-26 F0 | staging complete & verified (6 caveats) — F0 quality screen pending; not a measurement | – | – | – | `scripts/tmp/stage_b70_speed_20261010.log:464-629` + both `ARTIFACTS.sha256` (§7.3) | RL-133 (superseded by RL-136); completion follow-up note owed (§7.5) |

Every median/pairs/quality cell is "–" because run 2 took zero measurements.
Throughout this section: **blocked = no run; skipped = gate or sequencing never
evaluated; the only "failed" outcome in the F0 chain is RL-136's fail-closed
staging finalization — a process failure, not a measurement gate.** Ledger size at
run-2 write time: 136 entries, ending at RL-136 (run-1 §2 recorded 135).

### 7.3 Blocked list — exact causes (run 2)

1. **RQ-17 and RQ-26 F1 (RL-132, RL-134)** — same two causes as run-1 §3.1, both
   rows quoting exact tool outputs inline (which mitigates their missing evidence
   roots): (1) nemotron-hermes restart policy — since RESOLVED (live inspect read
   `RestartPolicy.Name='no'`; behaviorally confirmed across the 12:00:16 CEST
   daemon restart, §7.4); (2) per-card live-VRAM tooling gap — STANDING (no
   mem_info sysfs on this xe build; `xpu-smi discovery` → "No device discovered"
   on Battlemage) and still the fix required before any pre-load VRAM gate.
   Evidence-root check run by this writer: no Oct-10 results root under
   `results/2026/10/` and no preflight command-output file under `scripts/tmp/`
   (its only new Oct-10 files are the staging log, script, pycache and lock), so
   "–" stands for the three preflight lanes.
2. **RQ-11 (RL-135)** — recorded cause is only "preflight failed": no exact
   command, no output, no diagnosis. That misses blocker-policy §5.2
   (`docs/plans/2026-10-09-zcode-workflow-prompt.md:149`: "record `blocked` with
   the exact command and output"). The row is sealed and the ledger append-only,
   so the remedy is a diagnostic follow-up row; re-entry correctly requires a
   diagnosed cause on the v0.30.0/Nemotron stack. Registry: RQ-11 `gated`
   (verified: `python3 tools/research.py next --question RQ-11` → `halt=true`,
   reason "gated", exit 2).
3. **Skips, exact wording (both raw-array wordings were wrong and are corrected).**
   RQ-18 = "skipped — RQ-17 blocked at preflight, gate never evaluated"; the raw
   array's "(RQ-17 failed)" is materially false — blocked-at-preflight was never a
   failure, and this is the second appearance of that exact false wording after
   run-1 §3.4 corrected it once. Verified by this writer:
   `decisions/adr-0011-flashnext-strata-one-card-route.md:3` → `status: final` (so
   the plan flowchart's fail branch `Q -->|fail| X[Reopen ADR-0011]`,
   `docs/plans/2026-10-09-b70-next-program.md:51`, never fired); registry RQ-18
   `gated`, dependencies `['RQ-17']`; `research.py next --question RQ-18` →
   `halt=true` reason "gated". RQ-25: the raw array's "isolated gate or RQ-17" is
   garbled — registry truth is status `ready`, dependencies `[]`,
   `research.py next --question RQ-25` does NOT halt (exit 0), and the isolated
   ≥ 1.5× gate was never evaluated; the actual skip basis is plan sequencing only
   (next_action: "run it on GPU0 after RQ-17 frees the card"; flowchart
   `Q -->|pass| K[GPU0 · RQ-25 isolated microbench]`,
   `docs/plans/2026-10-09-b70-next-program.md:49`).
4. **RQ-26 F0 — corrected entry: staging complete & verified; quality screen
   pending.** Chain, all in `scripts/tmp/stage_b70_speed_20261010.log`: RL-133
   blocked (08:01:08Z cross-device-guard FATAL) → RL-136 failed (fail-closed
   10:23:06Z, `session end exit=5 problems=6 caveats=0`, both artifacts "INCOMPLETE
   — ARTIFACTS.sha256 NOT written", log lines 440/446/460) → session 5 completed
   (lines 464–629, PID 1951216, `exit=0 problems=0 caveats=6`). The 6 caveats are
   the same 3 aux text files in BOTH repos — chat_template.jinja,
   generation_config.json, tokenizer_config.json (log lines 616–626) — where the
   stale repo `crc32.txt` disagrees with the downloaded bytes while sha256 against
   the authoritative HF copies passes. This writer verified on disk: both
   manifests exist with 32 and 17 entries and self-sha256 `c9a350dba4f1ce9…` /
   `891525064bf25ff6…` equal to the values the log says it wrote (log lines 608,
   614), mtimes 12:29:09/12:29:19 CEST; the full `sha256sum -c` (32/32 + 17/17 OK,
   zero mismatches) was run by the adversarial sweep and is attributed, not
   re-run here (~72.5 GB). RL-136 transcription slip (sealed row, note-only): it
   says "bf16 31 files … exl3 16 files", but the log's own scan lines read 32/17
   (lines 68/76, re-logged at 480/488 in session 5; this writer's `grep -c`:
   98 double-logged verification lines = 49 unique files) — no effect on the
   fail-closed outcome or on the verified completion.
5. **Attempt-budget note** (stated explicitly because
   `execution_budget.max_attempts=2`): the F0 staging lane used three sessions
   against RQ-26 — RL-133 guard refusal, RL-136 crc32 fail-close, session-5
   completion — without violating AGENTS.md §20's two-attempts-per-unchanged-
   mechanism rule, because the mechanism changed each time (device-guard script
   fix, then the lead-ordered crc32→caveat reclassification), and the plan-A2 disk
   gate passed on its original terms (log lines 84–88: "free now 171553914880 B vs
   2 x remaining 144941577094.0 B" → "GATE: PASS — proceeding", i.e. free ≥ 2 ×
   remaining 72,470,788,547 B, matching
   `docs/plans/2026-10-09-zcode-workflow-prompt.md:92`).

### 7.4 Cleanup proof (run-2 refresh)

- Run-1 §4's "daemon inactive" was true at run-1 write time and went stale the same
  morning: the adversarial sweep found `<exl3-dockerd>d5` ACTIVE again, up since
  12:00:16 CEST, MainPID 1872528, `/run/<exl3-docker>.sock` present, container
  nemotron-hermes `Exited (0)` (attributed sweep live checks; no longer
  observable). Who restarted the daemon at 12:00:16 CEST, and who set
  nemotron-hermes `restart=no` (RL-132 had recorded `unless-stopped` at preflight),
  have no timestamped evidence — not determinable, stated as such. restart=no is
  now behaviorally confirmed: the daemon restart did not revive the container
  (still `Exited (0)`, `FinishedAt 2026-10-10T07:08:32Z` unchanged, at the
  cleanup's 12:41:46 CEST snapshot).
- **Phase-H cleanup of run 2** (attributed input, 12:41:46–12:43:24 CEST =
  10:41:46–10:43:24Z; quoted, not re-run): owned daemon stopped (`sudo -n systemctl
  stop <exl3-dockerd>d5` → inactive, transient unit gone, socket and pidfile
  gone, remote dial fails); `containersStopped=[]` honest empty — no
  campaign-owned container existed to stop (isolated socket: nemotron-hermes only,
  Exited (0); main socket: the same 10 foreign ods-pixel/openclaw containers,
  byte-identical IDs, monotonic uptimes, no device requests); bridge
  `<exl3-bridge>` deleted (workflow-created at preflight, as-found absent);
  exec-root `/var/run/<exl3-dockerd>` removed; caps as-found with ZERO writes
  (before = after = 180000000 / 230000000); temps identical before/after (GPU0
  54/56 °C, GPU1 52/54 °C); VRAM idle via debugfs `tile0/vram_mm` (GPU0
  visible_avail 32174 MiB, GPU1 32615 MiB — both over the 31000 MiB bar); card
  clients desktop-only (`dri/1`: gnome-shell master + the ZCode PID 1440463
  render-node holder — not compute, not killed); post-cleanup pgrep: no
  vllm/strata/exl3/llama process.
- **This writer's live re-check, 12:47:59 CEST** (commands run here — current
  state): `systemctl show <exl3-dockerd>d5` → `ActiveState=inactive`,
  `SubState=dead`, `MainPID=0`; `ls /run/<exl3-docker>.sock` → No such file;
  `docker -H unix:///run/<exl3-docker>.sock ps -a` → dial fails (socket gone);
  caps by live PCI identity: hwmon5 → `0000:0b:00.0` = 180000000 (180 W), hwmon6 →
  `0000:0f:00.0` = 230000000 (230 W); `pgrep -af
  'vllm|strata|exl3|llama-server|llama-bench'` → no matches (exit 1); `sudo cat
  /sys/kernel/debug/dri/{0,1}/clients` → desktop sessions only (dri/1: gnome-shell
  (master), ZCode 1440463, tern, brave — no compute client). The run-1 §4 cleanup
  verdict holds again as of write time, after one intervening unexplained
  restart-and-stop cycle.

### 7.5 Corrections carried by this update; outstanding obligations

- **rq22 note citation FIXED here** (the run-1 §3.4 "correct the note" order):
  `research/2026/rq22-int8-prompt-gemm-price-20261010.md:62-63` cited
  `RESULTS.md:241-249` "external copy in the same directory" — verified dangling
  by this writer (`wc -l` → exactly 130 lines; `ls` → no external copy). The note
  now carries a dated in-place correction marker and a §7 correction record;
  `base.log:45`/`base230.log:45` re-read (phase tables present) and
  `sha256sum base.log` = `ef9aa512c82231b76c679175916b9dfb8643491eee6c8a0008d86070a2368d0a`,
  equal to RL-131's recorded hash — RL-131 and the price are unaffected.
- **OUTSTANDING — cannot be done by the report writer** (single-writer
  serialization of `research.py run/assess/record`): RL-136's own re-entry order is
  unmet — "both ARTIFACTS.sha256 manifests must be produced; the manifests are to
  be cited in a follow-up ledger note, not by upgrading this row". The ledger ends
  at RL-136 (136 entries), and this writer's `grep -rn ARTIFACTS registry/ research/`
  finds no citation of the produced manifests anywhere — only RL-136's own
  now-stale "NO ARTIFACTS.sha256 exists" text. The experiment writer must register
  that follow-up note citing
  `/mnt/<nvme-disk>/b70-bf16-qwen38-20260926/ARTIFACTS.sha256`
  (32 entries, self-sha256 `c9a350dba4f1ce9…`) and
  `/mnt/<nvme-disk>/b70-exl3-qwen38-4bpw-20260926/ARTIFACTS.sha256`
  (17 entries, self-sha256 `891525064bf25ff6…`). Until then, F0 staging completion
  is evidenced only by the log and this report, and the registry's RQ-26
  next_action still reads "Check SN770 free space, then download …" (stale).
- Still owed from run 1: the RQ-11 preflight diagnosis (re-entry requires the
  diagnosed cause) and the per-card live-VRAM mechanism (§3.1 cause 2, standing).

### 7.6 Not run / not verifiable (stated plainly)

- This writer did NOT re-run `sha256sum -c` over the two model farms (~72.5 GB);
  the 32/32 + 17/17 OK result is the sweep's, attributed. On disk this writer
  verified manifest existence, entry counts, self-hashes and mtimes only.
- During the run-2 preflight-lane checks, `/sys/kernel/debug/dri/*/clients` was
  unreadable (Permission denied), so GPU client enumeration relied on pgrep (which
  found no compute processes). It IS readable via sudo at write time — this writer
  read both cards (§7.4).
- Who restarted the daemon at 12:00:16 CEST; who set nemotron-hermes restart=no —
  no timestamped evidence for either.
- `registry/research-program.json` in-session immutability rests on its mtime
  (07:07Z) predating the first staged session (08:01:08Z); no hash baseline was
  retained, so that is correlative, not proof.
- Sibling steps' in-flight outputs remain unreachable (workflow introspection), as
  in run 1.

### 7.7 Next question

`python3 tools/research.py next` (re-run by this writer at 12:48 CEST, read-only)
still selects **RQ-17** (status `ready`, no dependencies): the P0 quality battery
at the next authorized GPU window — passing it unblocks RQ-18 and frees GPU0 for
RQ-25's isolated microbench. Around it, in dependency order: (1) register the F0
manifest follow-up ledger note (§7.5), after which RQ-26 F0's CPU quality screen is
the ready CPU-side lane; (2) diagnose the RQ-11 preflight failure into a follow-up
row; (3) build the per-card live-VRAM read (xe debugfs client memory or an
engine-side report) before any lane with a pre-load VRAM gate.

### 7.8 Provenance (run 2)

- Inputs quoted from the run-2 ask (verified upstream, attributed): outcomes
  array; adversarial sweep verdict (including its `sha256sum -c` over both
  manifests and its live daemon/docker checks); Phase-H cleanup output; repo gates
  round 1: unittest=0, render=0, check=0, indexes=0 (workflow-run).
- Run by this writer this update (all quoted inline above): `python3 tools/
  research.py next` and `next --question RQ-17|RQ-18|RQ-25|RQ-26|RQ-11` (read-only;
  exits 0/2/0/0/2); ledger row reads RL-131–RL-136 and the 136-entry count;
  manifest `stat`/`wc -l`/`sha256sum`; `sha256sum base.log` plus re-read of
  `base.log:45`/`base230.log:45`; `wc -l RESULTS.md` and `ls` of the rq14 root;
  `sudo cat /sys/kernel/debug/dri/{0,1}/clients`; `systemctl show`, caps,
  pgrep and docker-socket checks; `grep`/`sed` over the staging log with line
  numbers; ADR-0011 status read; RQ-18/RQ-11 registry status read; plan-doc reads
  (`:92`, `:149`, `b70-next-program.md:49,51`).
- NOT touched by the report writer: `research.py run/assess/record`
  (single-writer serialization), any ledger or registry write, any process, any
  GPU state, and the run-1 sections above.

## 8. Run 3 (follow-up) — 2026-10-10 same-day update

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Appended by the report writer from the
run-3 verified inputs (lead outcomes array, adversarial sweep verdict, Phase-H
cleanup proof, repo gates round 1); the run-1 and run-2 sections above are kept
verbatim and are superseded only where this section says so explicitly. No paired
sample, median or quality number exists for any run-3 lane. The raw outcomes
array is NOT reproduced verbatim: three of its entries were false or incomplete
and are corrected below (§8.2, §8.3, §8.4), per the sweep.

### 8.1 Conclusion

Run 3 again produced no measurement: RQ-17, RQ-26 F1 and RQ-11 were blocked at
preflight with no run (RL-137, RL-138, RL-139, all `validation_scope`
`preflight_blocked_no_run` — no lane reached, let alone failed, a measurement
gate, and no results root exists for 2026-10-10), all three on one exact cause —
a foreign inference engine (another session's arm.sh/llama-stream llama-server,
uid 1000, port 19881, qwen1.5-moe-a2.7b-chat-Q3_K_M, -ngl 99) holding BOTH
renderD128 and renderD129, which also depresses GPU0 below the protocol's
~31,000 MiB empty-card bar (visible_avail 30166–30197 MiB at preflight,
`docs/methods/benchmark-protocol.md:36` §1.1). RQ-25 and RQ-18 were skipped with
no ledger rows owed — RQ-25 purely by plan sequencing (registry `ready`, no
dependencies; its isolated ≥ 1.5× gate was never evaluated), RQ-18 because
RQ-17 was blocked at preflight and its gate was never evaluated (ADR-0011 still
`final`, `decisions/adr-0011-flashnext-strata-one-card-route.md:3`) — while
RQ-26 F0 advanced to staging complete & verified (RL-140 `validated`,
superseding RL-136 `failed`: both `ARTIFACTS.sha256` manifests written, 32 + 17
entries, self-hashes/counts/mtimes re-verified on disk by this writer), so the
raw array's "skipped (staging not confirmed)" for F0 is corrected — the F0
quality screen itself has NOT run and is pending, not skipped. Run-3 cleanup is
proven (the only owned artifact, the isolated daemon, was stopped again:
inactive, socket/bridge/exec-root gone, caps as-found 180/230 W, zero power-cap
writes, `containersStopped=[]` honest empty), but the standing foreign-loop
blocker persists at this writer's write-time re-check (13:49 CEST: a fresh
arm.sh/llama-server pair live under new PIDs, GPU0 visible_avail down to
24071 MiB), so the RL-137/138/139 re-entry condition (owner stops the loop,
cards re-verified client-free) is unmet, and RQ-17 — still the registry's top
selection — has now consumed both of its `max_attempts=2` preflight attempts, so
a third attempt is admissible only with a changed cause.

### 8.2 Per-question results (run 3)

| id | verdict | median | pairs | quality | evidence_root | ledger_id |
|----|---------|--------|-------|---------|---------------|-----------|
| RQ-17 | blocked (preflight) — no run | – | – | – | – (no Oct-10 results root; exact causes quoted inline in RL-137) | RL-132, RL-137 |
| RQ-26 F1 | blocked (preflight) — no run | – | – | – | – (inline in RL-138; F1's GPU blocker is independent of F0 staging) | RL-134, RL-138 |
| RQ-11 | blocked (preflight) — no run; cause now diagnosed | – | – | – | – (inline in RL-139, which resolves the RL-135 diagnosis gap) | RL-135, RL-139 |
| RQ-25 | skipped — plan sequencing only; the isolated ≥ 1.5× gate was never evaluated | – | – | – | – | none owed |
| RQ-18 | skipped — RQ-17 blocked at preflight, gate never evaluated | – | – | – | – | none owed |
| RQ-26 F0 | staging complete & verified (RL-140 `validated`) — F0 quality screen NOT run: pending, not skipped | – | – | – | `scripts/tmp/stage_b70_speed_20261010.log:604-629` + both `ARTIFACTS.sha256` (§8.4) | RL-133 (superseded by RL-136); RL-140 |

Every median/pairs/quality cell is "–" because run 3 took zero measurements.
Throughout this section: **blocked = no run (a preflight check failed; no
measurement gate was ever evaluated); skipped = gate or sequencing never
evaluated. The campaign's only "failed" row remains RL-136 — the staging
finalization, a process failure — already superseded by RL-140 `validated`; no
run-3 lane is or should be labeled failed.** Ledger size at run-3 write time:
140 entries, ending RL-140 (run-2 §7.2 recorded 136) — verified by this writer
by loading `registry/research-ledger.json`.

Corrections applied to the raw outcomes array (none re-labels a lane):

1. **RQ-26 F0 "skipped (staging not confirmed)" — FALSE, replaced.** Contradicted
   by RL-140 (`validated`: "RL-136's re-entry condition is SATISFIED", both
   manifests written) and by this writer's own disk re-verification (§8.4).
   Correct entry: "staging complete & verified (RL-140); F0 quality screen NOT
   run — pending, not skipped."
2. **RQ-18 "(blocked or failed)" — materially false in the failed half.** RQ-17
   never ran and no gate was ever evaluated; ADR-0011 is `final` (verified this
   ask at `decisions/adr-0011-flashnext-strata-one-card-route.md:3`), so the
   plan flowchart's fail→reopen branch (`docs/plans/2026-10-09-b70-next-program.md:51`)
   never fired. Verified: `research.py next --question RQ-18` → `halt=true`,
   reason "gated", exit 2; registry RQ-18 `gated`, dependencies `['RQ-17']`.
   Dependency half correct; wording must read "blocked at preflight, gate never
   evaluated." This is the THIRD appearance of this false wording (corrected in
   run-1 §3.4 and run-2 §7.3.3 before this).
3. **RQ-25 "or the isolated gate was not cleared" — FALSE.** The ≥ 1.5× isolated
   gate was never evaluated (no microbench, no row, no results root). Verified:
   `research.py next --question RQ-25` → `halt=false`, exit 0, status `ready`,
   dependencies `[]`, next_action "run it on GPU0 after RQ-17 frees the card" —
   the skip basis is plan sequencing only ("RQ-17 did not pass" is acceptable
   only as shorthand for blocked-with-gate-never-evaluated).
4. **Empty ledger fields filled from the registry.** RQ-17 → RL-132 AND RL-137;
   RQ-26 F1 → RL-134 AND RL-138; RQ-11 → RL-135 AND RL-139; RQ-26 F0 → RL-133
   (`superseded_by` RL-136) plus RL-140. RQ-25 and RQ-18 correctly owe no rows
   (verified: `grep -c '"RQ-18'` and `'"RQ-25'` over
   `registry/research-ledger.json` → 0 and 0).

### 8.3 Blocked list — exact causes (run 3)

1. **RQ-17, RQ-26 F1, RQ-11 (RL-137, RL-138, RL-139) — one shared cause pair,
   quoted inline in all three rows.** (1) Foreign inference engine holds BOTH
   cards — the preflight's foreign-compute check fails: llama-server PID 2133430
   (uid 1000, port 19881, qwen1.5-moe-a2.7b-chat-Q3_K_M, -ngl 99) a client of
   renderD128 AND renderD129 per `/sys/kernel/debug/dri/{0,1}/clients`, launched
   by another session's experiment loop (`bash ./arm.sh new_s0_c` PID 2132670;
   an earlier arm PID 2123188 exited mid-preflight and re-armed under new PIDs;
   `build-llama-stream.sh` PID 2105274 compiling llama-stream-bsverify-r3). Not
   started by this workflow → not killed (AGENTS.md §4/§13 never-kill rule).
   Required change: the owning session/operator must stop the arm.sh/llama-stream
   loop (including pending builds) and the cards must be re-verified client-free
   (pgrep + dri clients) before Phase A can pass; expect re-check at launch per
   RQ-17's standing instruction. (2) GPU0 below the empty-card VRAM bar as a
   CONSEQUENCE of the foreign client: visible_avail 30166–30197 MiB < ~31,000 MiB
   (`docs/methods/benchmark-protocol.md:36` §1.1); GPU1 at 32429 MiB is above the
   bar but its render node is not client-free; clears automatically once the
   foreign llama-server stops, after which tile0/vram_mm is re-read. Registrar
   re-check quoted in all three rows (2026-10-10T11:25:56Z): the loop had already
   re-armed under new PIDs — two llama-servers clients of BOTH cards (2167554,
   llama-stream-bsverify-r3, Qwen3.6-35B-A3B-UD-Q4_K_XL, port 19778; 2169335,
   port 19881) under `arm.sh new_s0_par4_1` PID 2168729, with a fresh cmake
   llama-server build (2170873) — blocked state persists in kind.
   **This writer's live re-check, 13:49 CEST** (commands run here): the dri
   client tables listed llama-server 2254185 on BOTH render nodes, which exited
   mid-check and was replaced within the same minute by `bash ./arm.sh old_ref`
   PID 2254776 with llama-server 2255435 (llama-stream-old tree,
   qwen1.5-moe-a2.7b-chat-Q3_K_M, -ngl 99 -c 1024, port 19881) — the
   re-arming-under-new-PIDs behavior of RL-137 observed first-hand; GPU0
   visible_avail 24071 MiB, GPU1 32429 MiB (`tile0/vram_mm`). **The re-entry
   condition remains unmet as of write time.** Note the per-card live-VRAM gate
   itself is operable: RL-137/138/139 read it via the debugfs node the protocol
   names, so the run-1 §3.1 "cause 2" tooling gap is functionally covered by
   debugfs (no mem_info sysfs / xpu-smi support exists; budget conservatively).
2. **RQ-11 diagnosis debt — RESOLVED by RL-139.** Run-2 §7.5 listed the RQ-11
   preflight diagnosis as owed; RL-139 records the exact cause (the foreign loop
   above, with dri client tables). Any later quotation of run-1 §3.3's "cause
   undiagnosed" wording must be marked resolved-by-RL-139 (blocker policy §5.2,
   `docs/plans/2026-10-09-zcode-workflow-prompt.md:149`).
3. **Attempt budget — RQ-17 is exhausted.** Registry
   `execution_budget.max_attempts=2` (verified via `research.py next --question
   RQ-17` this ask) and two preflight-block attempts are consumed (RL-132,
   RL-137). A third attempt is admissible only with a changed cause — i.e. the
   foreign loop cleared and cards re-verified client-free — else AGENTS.md §20's
   two-attempts-per-unchanged-mechanism rule is breached.
4. **Bookkeeping at write time.** The sweep found `python3 tools/research.py
   check` FAILING on stale generated views (research/LEDGER.md,
   docs/indexes/research-coverage.md) after RL-137..140 were registered; the
   workflow's round-1 gates then ran clean (§8.6) and this writer re-ran `python3
   tools/research.py check` → exit 0, "OK: research registries valid, generated
   views current" — views are current, no render is owed from here. Still stale:
   the registry's RQ-26 `next_action` ("Check SN770 free space, then download
   …", verified verbatim in this ask's `next --question RQ-26` output) although
   RL-140 validated the download complete — a lead/registry-write action, not
   available to the report writer.

### 8.4 RQ-26 F0 — staging complete & verified (RL-140); screen pending

- Chain: RL-133 blocked (cross-device guard) → RL-136 failed (fail-closed crc32
  finalization, no manifests) → RL-140 validated (amended re-run, session 5,
  `scripts/tmp/stage_b70_speed_20261010.log:604-629`: both manifests written,
  six crc32 mismatches logged as caveats, `session end exit=0 problems=0
  caveats=6`).
- Re-verified on disk by this writer THIS ask: `wc -l` both manifests → 32 and
  17 lines; `sha256sum` of each equals RL-140's recorded self-hashes
  (`c9a350dba4f1ce9cf140882fa7ccb9f6c5145660193789f95ffb451b919dc4e4` bf16,
  `891525064bf25ff6771ab403183c8fa555679b9144b0ea93467469afa5c60f77` exl3);
  mtimes 12:29:09 / 12:29:19 CEST.
- Attributed, NOT re-run here: the full payload `sha256sum -c` over both farms
  (32/32 + 17/17 OK, zero mismatches, ~72.5 GB) — the adversarial sweep's check.
- Consequence: the F0 artifact prerequisite (BF16 source + EXL3 4bpw reference
  on SN770) is complete; the F0 CPU quality screen itself (hidden-state
  regeneration, block KL / top-1 vs EXL3) has NOT run and no KL or top-1 number
  exists. RL-140 is `validated` in its scope (staging artifact complete) — it is
  not an adoption and not a quality result.

### 8.5 Cleanup proof (run-3 refresh)

Two layers: (a) the Phase-H cleanup step's own proof (attributed input, quoted,
not re-run); (b) this writer's live re-check at 13:49 CEST (commands run here).

(a) **Phase H of run 3 (attributed).** Owned inventory before cleanup: ZERO
containers — the Phase A preflight failed at the foreign-compute gate, so no
Strata battery/route container or Nemotron server was ever started;
`containersStopped=[]` is an honest empty list (run-2 §7.4 precedent). The only
owned artifacts were the isolated daemon `<exl3-dockerd>d5` (started by the
preflight, MainPID 2140765), its bridge `<exl3-bridge>` (created 172.31.77.1/24;
as-found absent) and exec-root `/var/run/<exl3-dockerd>`. `sudo -n systemctl
stop <exl3-dockerd>d5` → exit 0; after: `ActiveState=inactive SubState=dead
MainPID=0`, socket and pidfile gone, remote dial fails, no dockerd process.
Caps with live PCI identity, BEFORE == AFTER with ZERO writes anywhere in the
campaign: hwmon5 → `0000:0b:00.0` (GPU0) 180000000 (180 W); hwmon6 →
`0000:0f:00.0` (GPU1) 230000000 (230 W). Bridge deleted, exec-root removed (it
held only daemon-created libnetwork/ and metrics.sock), no mounts remain;
data-root `/mnt/<nvme-disk>/<exl3-dockerd>/` RETAINED with both
images (vllm/vllm-openai-xpu:v0.30.0 sha256:fc0e112a…,
ghcr.io/0xsero/exl3xpu:0.1.0 sha256:21412bdd…) and both container configs —
keep-for-future disposition as in runs 1–2. nemotron-hermes EXITED and
untouched: live inspect before the stop → restart=no, state=exited, exit=0,
FinishedAt 2026-10-10T07:08:32.974914826Z; after the stop, durable state parsed
from the retained `hostconfig.json`/`config.v2.json` → RestartPolicy
`{"Name":"no"}`, Running=False, ExitCode=0 — byte-identical. Foreign
ods-pixel/openclaw stack untouched: BEFORE/AFTER `docker ps -a` on
`/run/docker.sock` byte-identical 10 rows (8 Up ~13 h healthy, 2 Exited ~29 h).
Temps before/after (temp2/3/4/5): GPU0 63/64/48/66 → 64/66/48/68 °C (+1–2 °C
tracks the foreign llama-server load, not the cleanup — the daemon stop is
host/CPU-side); GPU1 57/58/44/59 → 56/58/43/59 °C. VRAM after via
`tile0/vram_mm`: GPU0 29973 MiB, GPU1 31964 MiB — both depressed by the foreign
resident model; this campaign never loaded a model. No owned compute client on
either card (dri clients + `fuser` show only the as-found uid-1000 desktop set
and the foreign loop); the foreign stack was left as-found: long-lived
llama-server PID 2219620 (Qwen3.6-35B-A3B-UD-Q4_K_XL, port 19778, both cards)
with the r3 arm cycling 2232480→2236790 and harness `arm.sh new_opt0_ref_3`
PID 2238094 — not owned, not stopped.

(b) **This writer's live re-check, 13:49 CEST** (current state): `systemctl show
<exl3-dockerd>d5 -p ActiveState -p SubState -p MainPID` →
`inactive/dead/0`; `ls /run/<exl3-docker>.sock` → No such file;
`ip addr show <exl3-bridge>` → does not exist; `ls -d /var/run/<exl3-dockerd>`
→ No such file; caps by live PCI identity: hwmon5 → `0000:0b:00.0` =
180000000 (180 W), hwmon6 → `0000:0f:00.0` = 230000000 (230 W); `sudo cat
/sys/kernel/debug/dri/{0,1}/clients` → desktop sessions only PLUS the foreign
loop (see §8.3 item 1 for the live PIDs); `tile0/vram_mm` → GPU0 visible_avail
24071 MiB, GPU1 32429 MiB. The run-1/run-2 cleanup verdict holds again; the
standing foreign-loop blocker is the only live GPU-side fact and it is deeper
on GPU0 than at the run-3 preflight (24071 vs 30166–30197 MiB).

### 8.6 Bookkeeping — discharged and open

Discharged since the run-2 section (drop from any outstanding list):

1. RL-136's manifest follow-up note — SATISFIED by RL-140 (`validated`, cites
   both manifests and self-hashes; this writer re-verified the hashes on disk,
   §8.4).
2. RQ-11 preflight diagnosis — SATISFIED by RL-139 (exact cause, dri client
   tables).
3. Stale generated views — found by the sweep (`research.py check` FAILING after
   RL-137..140), then cleared: round-1 gates `unittest=0, render=0, check=0,
   indexes=0` (workflow-run, attributed input; commands per
   `docs/plans/2026-10-09-zcode-workflow-prompt.md:143`), and this writer's own
   `python3 tools/research.py check` → exit 0 at write time.

Still open (owners named; none available to the report writer):

1. **Foreign arm.sh/llama-stream loop** — owner action: stop the loop including
   pending builds; cards re-verified client-free (pgrep + dri clients +
   tile0/vram_mm) at the next preflight. Unmet at write time (§8.3, §8.5b).
2. **Registry RQ-26 `next_action` stale** against RL-140 (still says "Check
   SN770 free space, then download …") — lead must update the question's next
   condition; until then the F0 screen's entry point is RL-140 + the plan, not
   the registry text.
3. **RQ-17 attempt budget** — both attempts consumed (RL-132, RL-137); a third
   requires a changed cause (§8.3 item 3).

### 8.7 Next question

`python3 tools/research.py next` remains on **RQ-17** (verified this ask:
`next --question RQ-17` → `halt=false`, exit 0, status `ready`, dependencies
`[]`): the P0 quality battery at the next authorized GPU window — but only
after the owner clears the foreign loop and the preflight re-check (pgrep + dri
clients + tile0/vram_mm) passes, because the two-attempt budget is spent
(§8.3 item 3). Passing it unblocks RQ-18 (`gated` on RQ-17) and frees GPU0 for
RQ-25's isolated microbench. Independent CPU-side lane: RQ-26 F0 quality screen,
whose artifact prerequisite is complete (RL-140); its registry next_action is
stale (§8.3 item 4) and should be corrected before launch. The per-card
live-VRAM read is operable via the protocol's debugfs node (§8.3 item 1) — the
run-1 standing tooling blocker is downgraded accordingly.

### 8.8 Provenance (run 3)

- Inputs quoted from the run-3 ask (verified upstream, attributed): the lane
  outcomes array (three entries corrected here per the sweep); the adversarial
  sweep verdict (including its payload `sha256sum -c` over both farms and its
  live daemon/docker checks); the Phase-H cleanup proof; repo gates round 1:
  unittest=0, render=0, check=0, indexes=0 (workflow-run).
- Run by this writer this ask (all quoted inline above): `python3 tools/
  research.py next --question RQ-17|RQ-11|RQ-25|RQ-18|RQ-26` (read-only; exits
  0/2/0/2/0); ledger reads RL-132–RL-140 and the 140-entry count; `ls
  results/2026/10/` (no Oct-10 root; newest root `rq14-strata-matched-screen-
  20261009`, Oct 9 17:11); `wc -l`/`sha256sum`/`stat` over both
  `ARTIFACTS.sha256`; `python3 tools/research.py check` → exit 0; live host
  checks (`systemctl show`, socket/bridge/exec-root absence, caps via hwmon
  readlink, `sudo cat /sys/kernel/debug/dri/{0,1}/clients`,
  `tile0/vram_mm`, `ps`/`pgrep` over the foreign loop); ADR-0011 status read
  (`:3` = `final`); `grep -c` for RQ-18/RQ-25 rows (0/0); doc reads
  (`benchmark-protocol.md:36` §1.1, §1.8 at `:168`; workflow-prompt Phase H
  `:140`, §5 `:146`, §6 `:158`; `b70-next-program.md:44-57`; standing-rules.md;
  operating guide §5–7).
- NOT run by the report writer: `research.py run/assess/record` and `render`
  (single-writer serialization), any ledger or registry write, any process or
  GPU state change, and any edit to the run-1/run-2 sections above. The ~72.5 GB
  payload re-hash, the registrar's 11:25:56Z live re-check and the cleanup
  step's BEFORE/AFTER docker/temperature snapshots are attributed, not re-run.

## 9. Run 4 (follow-up) — 2026-10-10/11 update

**PROVISIONAL — NOT FOR PUBLIC HEADLINE.** Appended by the report writer from the
run-4 verified inputs (lead outcomes array, adversarial sweep verdict, Phase-H
cleanup proof, repo gates round 1); the run-1 through run-3 sections above are kept
verbatim and are superseded only where this section says so explicitly. Run 4 took
the campaign's first sealed runs; no speed median and no passed quality gate exist
for any run-4 lane. The raw outcomes array is NOT reproduced verbatim: RQ-26 F1
and RQ-18 stand as given (F1 minus one stale program-status sentence, §9.3.2),
and four entries are corrected before publication — RQ-17 materially wrong,
RQ-11 incomplete and overstated, RQ-25 stale premise, RQ-26 F0 re-based off a
void headline ratio — per the adversarial sweep and this writer's own registry
reads.

### 9.1 Conclusion

Run 4 produced the campaign's first sealed measurement attempts — RQ-17 twice (RL-141, RL-144), RQ-11 twice (RL-146, RL-148), RQ-26 F0 twice (RL-149, RL-150), plus the read-only F1 kernel review RL-145 — and none yielded a usable comparison: RQ-17 and RQ-11 are closed rejected on execution integrity with 2/2 attempts consumed each, F0's attempt 1 overran its frozen deadline and its attempt 2 is blocked on a degenerate baseline, and the F1 kernel half is closed failed with 24.36 GB/s at M=4 explicitly not registrable (bitwise equality failed on all 39 round-2 RESULT lines; the ISA proof's only disassemblies predate every timed binary). The raw outcomes array is corrected before publication: RQ-17 did NOT reopen ADR-0011 (the registry rules the stop rule not fired — the canary misses are gen-64 length truncations of a thinking route with correct or unobservable answers, and the route's quality closure is UNVALIDATED, neither passed nor failed), RQ-11 produced no baseline at all (zero measurement cells — "new baseline, not comparable to 186.6" is unbacked), RQ-25's sequencing skip repeats the premise RL-143 retracted (the question is ready with no dependencies and was skipped for scope), and F0's closure is stated on the recorded KL gap (GPTQ arms 1.55–4.31× EXL3 block-output KL at like-for-like 4.125 bpw on the informative blocks) rather than the void 1.3299 headline ratio, with the RQ-26 status-field/closure-narrative inconsistency named (status `ready` vs "F0 CLOSED THE FORMAT"; `research.py next` halts on the exhausted attempt budget). Cleanup is proven — attributed Phase-H with honest-empty `containersStopped=[]`, zero power-cap writes, daemon/socket/bridge/exec-root down, foreign stacks byte-identical before/after, re-verified live by this writer (caps 180/230 W by live PCI identity, both cards desktop-only) — and the queue now blocks on lead/operator decisions rather than lanes: the RQ-26 close/resolution and status reconciliation, the RQ-17 battery re-author and the RQ-11 container-boot diagnosis (both budgets exhausted), with RQ-25 the only ready no-dependency question left.

### 9.2 Per-question results (run 4)

| id | verdict | median | pairs | quality | evidence_root | ledger_id |
|----|---------|--------|-------|---------|---------------|-----------|
| RQ-17 | closed rejected — execution integrity; ADR-0011 NOT reopened, quality closure UNVALIDATED; re-entry via operator decision | – (no speed median; gate not passed) | attempt 2 executed: 6 canaries ×2 runs (4/6 each; misses = gen-64 length truncations), 3×3 replay (rep1/rep2 token-identical, rep3 device-lost), 10 host tasks (9/10 pass), kv8k turn-1 | partial battery retained; the rejection is execution integrity — not a quality validation of the route | `results/2026/10/RQ-17-20261010T131952270494Z-1a0043228c` (attempt 1, zero canaries scored) + `results/2026/10/RQ-17-20261010T171539823866Z-f31436becd` (attempt 2) | RL-141, RL-142, RL-143, RL-144 |
| RQ-26 F1 | kernel half closed failed — gate missed on all three legs (speed, bitwise, ISA); 24.36 GB/s not registrable (review verdict) | 24.36 GB/s @ M=4 — an RC=1 row-pass rate, NOT registrable (unique-byte rate 6.08 GB/s; the RC>1 dpas path was never measured) | 9 round-2 ring cells (qkv/gate_up/down × M=1/4/8) + Hadamard-wrap cells, bitwise-gated | bitwise equality FAILED (all 39 round-2 RESULT lines gemm=0; wraps bitwise-correct); the named ISA check FAILED on provenance | `research/2026/b70-hq4-f1-evidence-20261010/` | RL-145 |
| RQ-11 | closed rejected — execution integrity ×2, ZERO measurement cells; NO baseline produced; attribution staged but void as measured | – (zero measurement cells for the v0.30.0 stack) | – (n=5 × 2 prompt shapes declared, none executed) | battery/canary declared, never measured; the 0/0/0/0 attribution is a schema-required void, never to be quoted as shares (RL-147) | `results/2026/10/RQ-11-20261010T183453712825Z-8b32e529b5` + `results/2026/10/RQ-11-20261010T212620351889Z-fa2d9769c9` | RL-146, RL-147, RL-148 |
| RQ-25 | ready — skipped for scope this run; the sequencing premise is retracted (RL-143); no gate evaluated, no row owed | – | – | – | – | none owed |
| RQ-18 | skipped — stays gated: dependency RQ-17 closed rejected (a closed dependency unlocks only at resolution passed) | – | – | – | – | none owed |
| RQ-26 F0 | attempt 2 BLOCKED (baseline invalid → gate unsatisfiable as declared); the registry narrative closes F0 per the pre-declared stop rule while the status field still reads `ready` | – (no valid ratio gate; recorded KL gap 1.55–4.31× EXL3 @ 4.125 bpw on the informative blocks; the 1.3299 headline ratio is void) | 8 arms × ≥4 blocks declared; 2 runs — attempt 1 deadline overrun (all 6 blocks incomplete), attempt 2 blocked before any comparison | assessment guard: "baseline.gdn_l30.value must be finite and positive" (gdn_l30 −1.315e-14, att_l31 −1.001e-13); AR-uniform arms went <1.0× only via full INT8 escape at 8.125 bpw, over the 4.5 cap | `results/2026/10/RQ-26-20261010T214816140328Z-daee8d535f` (RL-149) + `results/2026/10/RQ-26-20261010T231243348252Z-1e1f43f8d1` (RL-150) | RL-149, RL-150 |

Ledger state at run-4 write time (verified by this writer): `registry/
research-ledger.json` holds 150 rows, ending RL-150 (run-3 §8.2 recorded 140).
Row allocation this run: RQ-17 → RL-141–RL-144 (4 rows), RQ-11 → RL-146–RL-148
(3), RQ-26 → RL-145 (F1 kernel half) + RL-149/RL-150 (F0 attempts);
RQ-25/RQ-18 owe none (verified by `question_id` count over the ledger: 0 and 0).
Throughout this section: **failed = a sealed run hit a gate or
execution-integrity failure; blocked = no usable comparison could be produced;
skipped = no run and no gate ever evaluated.**

### 9.3 Failed vs blocked — exact causes (run 4)

1. **RQ-17 — FAILED ×2 (execution integrity), closed rejected; the raw array's
   "ADR-0011 reopened" is FALSE and is corrected here.** Attempt 1 (RL-141,
   13:19–13:27Z): preflight passed, the battery exited 1 on the adapter's
   evidence-retention bug (FileNotFoundError writing
   `raw/serve/canary-A-batt-arith.json` because `raw/serve/` was never created,
   `scripts/tmp/rq17_battery_20261010.py:622`, per RL-142/RL-143) — zero
   canaries scored, no quality number. Attempt 2 (RL-144, 17:15Z): all serve
   cells completed — canary gate 4/6 in both runs, and the misses are
   `finish_reason=length` truncations at the plan's gen-64 floor on a thinking
   route while computing CORRECTLY (registry next_action, read this ask:
   batt-arith's reasoning reaches 740 en route to the correct 703; the sweep's
   cell-by-cell read of all 12 canary cells: no wrong content — batt-sumsq's
   answer is unobserved, so the named 14-vs-30 trigger was never evaluated),
   so the pre-declared stop rule — wrong canary or measured replay drift — was
   NOT fired; replay rep1/rep2 output lines token-identical, rep3 died on
   device loss (verified this ask: `grep -c UR_RESULT_ERROR_DEVICE_LOST` → 2
   in the attempt-2 root's `raw/replay/batt-arith-rep3.log`; the registry
   attributes the seizure to the foreign llama-server re-arm); 9/10 host tasks
   pass at gen 256 (the sweep verified the one miss, t5-code-output, is a real
   task miss that does not fire the canary stop rule), kv8k-turn1 correct,
   speculation acceptance 0.789 at 3.20 tok/round; kv8k-turn2 exists unscored.
   Registry closure verbatim: "ADR-0011 REOPENED_QUALITY_FAILURE is NOT fired"
   and "ADR-0011's quality closure is UNVALIDATED - neither passed nor failed";
   re-entry is an operator decision (budget 2/2): re-author the battery with
   canary gen ≥ 256 for the thinking route, a foreign-client guard or a
   stopped loop, and a kv8k-turn2 budget check. ADR-0011
   (`decisions/adr-0011-flashnext-strata-one-card-route.md`) is unchanged.
2. **RQ-26 F1 — kernel half FAILED (all three gate legs; RL-145,
   `validation_scope` `source_and_arithmetic_no_gpu`, no GPU re-measurement).**
   Gate: "≥ 500 GB/s at M=4, bitwise equal to a CPU reference, clean ISA".
   Missed: (a) speed — 20.57–24.39 GB/s across all nine round-2 ring cells,
   ~20× under, and the headline 24.36 GB/s "at M=4" is an RC=1 ROW-PASS rate
   (the same M=1 kernel executed 4× per round; unique-byte rate 6.08 GB/s) —
   the RC=4/8 dpas-reuse path was never measured (A-operand interleave
   unresolved); (b) bitwise — all 39 round-2 RESULT lines carry gemm=0
   (embedded bitref mismatch) while the Hadamard wraps pass (wrap_x=1
   wrap_S=1); (c) ISA — the named check FAILED on provenance: the only
   disassembly dumps are of the round-1 kernel (mtimes 2026-10-10 13:56–14:08Z)
   predating the placement probes (15:55Z+), the rewrites and every timed
   binary (16:31–17:22Z); no round-2 dumps exist and the gate marker carries
   no source/binary hash. Beyond the three legs (all per the F1 row input and
   RL-145): timed-binary identity is unrecorded and round-2 invocations
   unreconstructible (no runner script retained; the retained runner launches a
   binary overwritten at 17:22:51Z, and the console log ends at the 16:57:38Z
   runner-bug failure); per-cell contention evidence is missing for every timed
   cell (the foreign arm.sh loop demonstrably re-arms under new PIDs,
   RL-137/RL-138); the round-2 values_relmax summary is unretained (the
   review's recomputation from raw .out passes the 5e-3 gate, worst 1.684e-3,
   but the doc's "all other rows ≤ 1.2e-4" claim is false — five rows exceed
   it); `round1/ring_plan_snapshot.json` is absent; the 32 MiB L2 premise is
   unverifiable from the root and `ENV.txt` energy fields are empty. Review
   verdict: the arithmetic verifies but the package is NOT registrable as a
   gated measurement; the repair list is in RL-145 (sha256-bound disassembly
   of the timed binary via the ocloc route, gate greps against it, per-cell
   contention re-checks, retained values_relmax summary, fix-or-revert the
   claimed runner fix, per-cell ring-plan snapshots). One citation corrected:
   the review's "bench.py:126-129" is `scripts/tmp/rq26_f1_bench.py:125-131`
   (same content). Id deviation recorded inside RL-145: the manual row became
   RL-145 because the experiment writer concurrently appended RL-144 (the
   RQ-17 attempt-2 machine row); the first write attempt failed safely on the
   modified-file guard. Stale sentence in the raw array: "RQ-26 remains ready
   with F0 the live stage" was true at RL-145's registration but the registry
   next_action has since been overwritten by the F0 closure narrative while
   the status field still reads `ready` — the same status/value mismatch class
   RL-143 fixed (§9.3.4). The sealed download attempts stay with the F0 screen
   (RL-136/RL-140), per RL-145's scoping note.
3. **RQ-11 — FAILED ×2 (execution integrity), closed rejected, ZERO measurement
   cells; the raw array's "new baseline" is UNBACKED.** Attempt 1 (RL-146,
   18:34Z): preflight refusal — the foreign llama-stream loop held GPU1 at
   preflight; RL-147 records that the attribution lane consumed zero sealed
   attempts under the lead's directive and that the 0/0/0/0 shares are
   schema-required void placeholders (its parser passed the registrar's
   CPU-only self-test reproduction that session). Attempt 2 (RL-148, 21:26Z):
   the serve container exited during startup with EMPTY captured logs; the
   retained launch argv (`raw/serve_container_launch.txt`) must be re-audited
   for quoting/completeness before any retry, and the log capture raced the
   container removal (registry next_action, read this ask). The hypothesis
   (on-device router + fused grouped dequant GEMM) remains untested, not
   refuted; no baseline was produced, and any future baseline must not be
   compared to the 2026-08-13 186.6 row. Semantics note, named once: RL-146
   registered `failed` for a preflight refusal — closed rows admit only
   passed/rejected (`tools/research_queue.py:43`, the constraint RL-143
   discovered when a `failed` resolution broke `research.py next`), so
   closed-rejected with recorded causes is the workable form, but a refusal is
   closer to blocked-as-found; the attribution lane's own row (RL-147)
   correctly records `blocked` with no trace captured. Re-entry: operator
   decision (budget 2/2) — diagnose the boot against the retained argv (docker
   run the same argv interactively outside a sealed run to capture real
   stderr), fix the boot, then re-author the anchor attempt design.
4. **RQ-26 F0 — attempt 1 FAILED (deadline, mislabeled against its own plan);
   attempt 2 BLOCKED (invalid baseline); the registry closes F0 on a re-based
   KL gap while the status field stays `ready`.** Attempt 1 (RL-149): workers
   exceeded the 3450 s budget and the run ended "FATAL: incomplete evidence"
   missing all six blocks (verified this ask: `command.stderr` tail —
   "DEADLINE: worker att_l31 exceeded the 3450.0s budget … FATAL: incomplete
   evidence (missing=[gdn_l01, gdn_l30, gdn_l62, att_l03, att_l31, att_l63],
   timed_out=True)") — while the frozen plan predeclared "deadline overrun or
   any incomplete block fails evidence honestly (blocked, not pass)"
   (`campaigns/2026/hq4-f0-quality/plan.json:70`): the failed-vs-blocked
   mismatch the sweep flagged is verified. Attempt 2 (RL-150, outcome
   `blocked`): the assessment guard "baseline.gdn_l30.value must be finite and
   positive" fired — the gdn_l30 baseline KL is −1.315e-14 and att_l31
   −1.001e-13 (registry next_action), so two of six block baselines are
   degenerate, the pre-declared gate (best arm KL ≤ 1.0× EXL3 on every block
   AND top-1 ≥ EXL3 − 0.001 on every block AND bpw ≤ 4.5) is unsatisfiable as
   declared, and no KL ratio on those blocks is valid. Closure basis, re-based
   per the sweep: the screen's verdict line "VERDICT: FAIL
   best_arm=ar-uniform-h1 (worst-block KL ratio 1.3299, bpw_max 8.1250)"
   (verified verbatim in the attempt-2 `command.stdout` this ask) divides by
   the invalid negative gdn_l30 baseline and is VOID as a ratio — the registry
   narrative itself says so; the defensible, registry-recorded basis is:
   GPTQ arms at like-for-like 4.125 bpw measure 1.55–4.31× EXL3's block-output
   KL on the only informative blocks (gdn_l62 3.226/3.982/4.258/4.310; att_l63
   1.552/1.662/2.043/2.113) — a real quality fail of the no-trellis format at
   equal nominal bitrate; the only arms under 1.0× (ar-uniform-h0/h1, valid-
   block ratios 0.056–0.104) got there solely via full INT8 escape at
   bpw_mean 8.125, over the 4.5 cap (not format-eligible); ar-e2m1-h0/h1
   measure 10.60–14.96× worse at ~5.8 bpw. Sweep caveat carried: the
   pre-declared "one rounding-recipe retry" was never exercised as such —
   attempt 2 changed only AR budget constants ("all other recipes are
   unchanged", its plan note) — and `max_attempts=2` is exhausted, so the
   closure is a lead/operator decision under spent budget rather than an
   automatic stop-rule firing on the quoted number; the registry leaves
   "whether RQ-26 itself closes, and with what resolution" to lead direction.
   Program inconsistency recorded: RQ-26 `status: ready` vs next_action "F0
   CLOSED THE FORMAT" — `research.py next` halts on it (exit 2, "Attempt limit
   reached. Close the question or record new mechanism evidence.", verified
   this ask); reconciliation is a lead action, not available to this writer.
5. **RQ-25 and RQ-18 — SKIPPED (no run, no gate, no rows owed).** Verified
   this ask: zero ledger rows carry either `question_id`. RQ-25: registry
   `ready`, dependencies `[]`, `research.py next --question RQ-25` →
   `halt=false`, exit 0; the raw array's skip reason ("RQ-17 did not pass or
   the isolated gate was not cleared") repeats the exact premise RL-143
   retracted (RQ-17's rejection was execution-integrity, not a route failure);
   the honest wording is skipped-for-scope this run — the isolated ≥ 1.5× SoA
   gate was never evaluated. RQ-18: registry `gated`, dependencies
   `['RQ-17']`; a closed dependency unlocks work only at resolution `passed`
   (`docs/governance/agent-operating-system.md` §7.1) and RQ-17 closed
   `rejected`, so the dependency is genuinely unsatisfied — `research.py next
   --question RQ-18` → `halt=true`, reason "gated", exit 2.

### 9.4 Cleanup proof (run-4 refresh)

Two layers: (a) the Phase-H cleanup step's own proof (attributed input, quoted,
not re-run); (b) this writer's live re-check at write time (commands run here).

(a) **Phase H of run 4 (attributed, 2026-10-11 00:23–00:25 UTC).**
`containersStopped=[]` is an HONEST empty list: the BEFORE inventory (docker
ps -a on the main socket, 00:23:46Z) shows exactly the 10 foreign
ods-pixel/openclaw containers (8 Up 25 h, 2 Exited (0) 42 h) and no
campaign-owned container ever appeared on either socket; the isolated socket
shows only `nemotron-hermes` `Exited (0)` 17 h, untouched. Caps BEFORE ==
AFTER with ZERO writes anywhere in the campaign: 180000000 (180 W) on
`0000:0b:00.0` and 230000000 (230 W) on `0000:0f:00.0`, read by live PCI
identity. Daemon stopped: `sudo -n systemctl stop <exl3-dockerd>d5` → exit 0,
then `ActiveState=inactive SubState=dead MainPID=0`, socket and pidfile gone,
remote dial fails. Bridge `<exl3-bridge>` was already torn down by the daemon's
clean shutdown (as-found absent, both before and after the delete attempt);
exec-root `/var/run/<exl3-dockerd>` removed after a checked umount; data-root
RETAINED with both images and container configs (keep-for-future, runs 1–3
precedent). Cards idle: dri clients and `fuser` over both render nodes show
only the as-found uid-1000 desktop set; `tile0/vram_mm` GPU0 visible_avail
30902 MiB, GPU1 32615 MiB (GPU0 sits just under the protocol's ~31,000 MiB
empty-card bar from desktop-session usage — as-found, not campaign-owned).
Temps unchanged within ±1 °C (host/CPU-side daemon stop). Found-state change
recorded, not an action of this campaign: the foreign llama-stream loop was
already GONE at the 00:23:46Z BEFORE snapshot — it exited on its own between
bring-up and cleanup, consistent with its observed re-arm/exit cadence; never
touched. (This writer did not reconcile how the RQ-17 attempt-2 serve phase
was hosted against the empty owned-container list; the serve evidence is
retained in the RQ-17 attempt-2 run root and the BEFORE/AFTER `docker ps -a`
on both sockets is the cleanup's authoritative container inventory.)

(b) **This writer's live re-check** (commands run here): `systemctl show
<exl3-dockerd>d5 -p ActiveState -p SubState -p MainPID` → `inactive/dead/0`;
`ls /run/<exl3-docker>.sock` → No such file; caps by live PCI identity:
hwmon5 → `0000:0b:00.0` = 180000000 (180 W), hwmon6 → `0000:0f:00.0` =
230000000 (230 W); `sudo cat /sys/kernel/debug/dri/{0,1}/clients` → both cards
desktop-only (card0: gnome-shell, Xwayland, mutter, ibus, gpaste, codexbar,
zcode, tern, brave, chrome, systemd-logind; card1: gnome-shell, ZCode, tern,
brave, chrome) — no engine-class client on either card. My `pgrep` invocation
self-matched its own shell wrapper and is inconclusive as run; the
no-compute-client conclusion rests on the dri client tables (mine) and the
cleanup step's pgrep/fuser sweep (attributed).

### 9.5 Next question

`python3 tools/research.py next` (run by this writer, read-only) HALTS:
`halt=true, question_id RQ-26, reason "Attempt limit reached. Close the
question or record new mechanism evidence."` (exit 2) — the queue's blocking
item is the lead's RQ-26 close/resolution decision plus the status-field
reconciliation (§9.3.4), not a runnable lane. Verified by direct query: RQ-25
is the only ready, no-dependency question (`next --question RQ-25` →
`halt=false`, exit 0; its next_action: read the IQ3_XXS Split/Multi load path
and `tools/iq_pack.py`, write the isolated SoA microbenchmark in
`scripts/tmp/`, run it on GPU0 when the card is free — at this writer's
write-time check no engine-class client sits on either card, but the full
preflight still gates at launch and the VRAM bar was not re-read this ask).
RQ-17 and RQ-11 are closed rejected and excluded by `next`; their re-entry is
operator work (§9.3.1, §9.3.3). RQ-18 stays gated on RQ-17.

### 9.6 Provenance and gates (run 4)

- Inputs quoted from the run-4 ask (verified upstream, attributed): the lane
  outcomes array; the adversarial sweep verdict (including its cell-by-cell
  canary/task read of the RQ-17 attempt-2 root, its rep3 log read, its F0 arm
  review and its registry walk); the Phase-H cleanup proof; repo gates round 1:
  unittest=0, render=0, check=0, indexes=0 (workflow-run, commands per
  `docs/plans/2026-10-09-zcode-workflow-prompt.md:143`).
- Run by this writer this ask (all quoted inline above): `python3 tools/
  research.py next` and `next --question RQ-17|RQ-26|RQ-11|RQ-25|RQ-18`
  (read-only; exits 2/2/2/0/2); ledger rows RL-141–RL-150 read and the 150-row
  count; registry status/resolution/dependencies/next_action reads for the
  five questions; existence checks over all seven cited evidence roots (7/7
  exist); `assessment.json` outcome/decision/reason reads for the five sealed
  roots; RL-149 `command.stderr` tail (DEADLINE/FATAL lines); the F0 attempt-2
  VERDICT line in `command.stdout` (verbatim); `grep -c
  UR_RESULT_ERROR_DEVICE_LOST` → 2 in the RQ-17 attempt-2 root's
  `raw/replay/batt-arith-rep3.log`; `campaigns/2026/hq4-f0-quality/plan.json:70`;
  `tools/research_queue.py:43`; per-`question_id` ledger row counts
  (RQ-25/RQ-18 → 0/0); `python3 tools/research.py check` → exit 0 ("OK:
  research registries valid, generated views current"); the live cleanup
  re-check (§9.4b).
- NOT run by the report writer: `research.py run/assess/record/render`
  (single-writer serialization), any ledger or registry write, any process or
  GPU state change, and any edit to the run-1–3 sections above. The sweep's
  cell-by-cell canary/task verification is attributed, not re-read cell by
  cell here; the F0 arm_stats KL-gap numbers are quoted from the registry
  next_action (read this ask), not recomputed from `arm_stats.json`; RL-145's
  raw-CSV/byte recomputations were verified at that row's drafting per its own
  text and were NOT re-run here; `tile0/vram_mm` and the unittest/render/
  indexes gates were not re-run this ask.
