# Response to review rounds 1 and 2 (`tp/review-tptf.md`, `tp/review-tptf-round2.md`)

## Round 1

New head: `15adb58cf` on `xpu-tp2-fixes-20261119` (on top of `78f42f240`, `df3baa3ea`). One file changed in
TensorFold: `zig/src/xpu/slot_states.zig`. Documents changed: `REPORT.md`, `pr-body-snapshot-layers.md`,
`pr-body-shared-tree-prefix.md`. Door: two-way (revert `15adb58cf`, then `78f42f240`).

| # | Finding | What changed |
|---|---|---|
| Blocker 1 | `slot_states.zig:18` stated driver page aliasing past 2 GiB from ~140K tokens; the mitigation applies everywhere | `slot_states.zig:18-23` (the `Snap` comment) now says what was measured: one buffer per rank failed from about 165K tokens (131K resumed); per-layer buffers were free of that failure in the two replays tested (165K, 250K); cause unknown; it links the report. Scope decision: **all configurations keep the per-layer layout.** With the trigger unknown, no configuration can be shown to be outside it, and gating on TP degree, K/V format or size would rest on no evidence and split save and restore into two code paths. `REPORT.md` section 1 and the PR body call it a tested mitigation, not a root-cause fix |
| Should-fix 1 | Bug B claim generalised from one workload; the paths differ in DPAS and bf16 weight rounding versus scalar float reduction, not only order | `REPORT.md` section 1 (Bug B paragraph) and the fix-table row B name both differences (`decode_prefix.cl:34`, `:42`; `decode.cl:125`, `:137`) and say the control isolates the path choice for this workload, not which numerical difference or a near tie (no logits kept). `pr-body-shared-tree-prefix.md` Problem and Measured sections changed the same way, and the one-card remark now says one differing and one matching repeat, scheduling not recorded. The commit message of `df3baa3ea` (already pushed) still says "different order"; I did not rewrite pushed history |
| Should-fix 2 | Reproduction omits `XPU_ATTN_PREF=256`, the fixture and the replay; revert names the #3586 port | `pr-body-snapshot-layers.md` Problem, Measured and Reproduce sections now set `XPU_ATTN_PREF=256 XPU_KV_INT8=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`, name the fixtures (`code-160k` from `prompts-bisect.json`, `code-256k` from `prompts-deep.json`), the follow-ups and the cache-off-then-replay procedure. Revert is `git revert 15adb58cf 78f42f240` |
| Should-fix 3 | Old 165K row is completion-only; #3586 is a different build and configuration | PR body table row 1 says completion only, no matched cache-off run, second follow-up 165,674 versus 165,802. The #3586 paragraph (PR body) and `REPORT.md` section 1 and section 7 item 7 say it is a different build (no prefix kernel, later base than `480167c75`, no same-build cache-off baseline) and that it does not attribute the trigger to our fork |
| Should-fix 4 | Up to 64 drains per rank per drop; budget accounting unstated | `slot_states.zig` `dropSnap` (now `:128-137`) waits once per rank, then frees that rank's buffers; a rank whose wait fails keeps its buffers until close, as before. `snapBytes` comment (`:48-49`) and the PR body's "Cache budget" paragraph state that `--prompt-cache-gib` charges payload plus the rank-0 draft bytes and does not count per-allocation rounding or bookkeeping (not measured; 114,688 B per rank at 165,056 tokens under a 4 KiB model). Eviction, partial-save failure injection and live driver memory are listed under "Not tested" |
| Should-fix 5 | Use "failure absent in the two tested replays, cause unknown"; say which interventions left the trace unchanged; correct the push arithmetic | `REPORT.md` section 1 Bug A heading and bullets, fix-table row A, section 7 items 2, 4, 5; the PR body's opening and Reproduce sentence. Interventions listed with "left the trace unchanged" and the caveat that host flags do not prove the driver path; the two-address stamp excludes only that pair; the layer bisect's implicit layer-26 zero and swallowed readback errors are stated. Push: 19.87 µs incremental median at 163,840 B (25 repeats, 0 errors), about 17 µs derived, 0.367 ms, 1.1 % of 33.2 ms (`REPORT.md` section 1) |

## Runs (review round 1)

| Job | What | Result |
|---|---|---|
| 1048 `tptf-a-old-pref0` | Old fork binary `2286abc0c`, failing 165K fixture, only `XPU_ATTN_PREF=0` | **Pass (completion):** both follow-ups done, cache hits 165,056 / 165,632, hashes `2cac6f7e6317` / `24ec0db0fe76`. The prefix path (execution or its scratch allocations) is the trigger. Recorded in `REPORT.md` section 1 and section 7, and in the PR body's #3586 paragraph |
| 1048b `tptf-build-head` | Build of `15adb58cf` (cards none, exclusive) | Built, 158 s; binary sha256 `fb240d8c…` |
| 1048c `tptf-a-head-160k` | `15adb58cf`, 165K, resumed == cache-off | **3 of 3 pass**, cached 0 / 165,056 / 165,632; engine hashes `c0c67ffd13e5`, `b2c355d78f54`, `24ec0db0fe76`, identical to the `78f42f240` run; caps restored (`raw/a-head-160k/`) |
| 1048d `tptf-b-head` | `15adb58cf`, bug B 32-request check | **32 of 32**; one engine hash per prompt, `chat-8k` `5b5869d7083a` (`raw/b-head/check.jsonl`) |
| push | `fork/xpu-tp2-fixes-20261119` | `78f42f240..15adb58cf` |
| 1048e/1048f | Old fork binary, prefix off, prefix scratch still allocated (`XPU_ATTN_PREF_SCRATCH=256`), 165K with a matched cache-off run | 1048f was refused at 06:59 by `repro.sh`'s stale 07:00 deadline (fixed); result under round 2 (1048f2) |

## Not changed

- The notes on code correctness (shared-tree bound, scratch sizes, snapshot ownership) needed no change.
- The review's root-cause probes (per-append records with independent hashing, reserved scratch with prefix execution
  bypassed, full-region stamps) were not run: the lead stopped GPU work on bug A after one discriminator. They are
  carried into `REPORT.md` section 7 as re-entry tests.

## Round 2

New head: `ac346900b` (pushed; `fork/xpu-tp2-fixes-20261119` = `15adb58cf..ac346900b`). One file changed:
`zig/src/xpu/slot_states.zig`. Door: two-way (`git revert ac346900b 15adb58cf 78f42f240`).

| # | Finding | What changed |
|---|---|---|
| Should-fix 1 | Prefix-off finding stated as "the trigger" | Bounded to: with the same old binary and the same first follow-up, prefix off completed where prefix 256 failed. The confounds are named: dispatch, scratch size 203,097,600 B against 198,144 B per rank, placement and timing. Changed in `REPORT.md` section 1 (discriminator bullet) and section 7 (item 7 and the re-entry text), and in `pr-body-snapshot-layers.md` (opening and #3586 paragraph). `slot_states.zig:19` now says "failed in the 165K and 250K tests (a 131K run completed; the first failing depth is unknown)". The scratch test result is added, bounded to its run (below) |
| Should-fix 2 | Lab replay command omitted `CACHE_GIB=5` | `pr-body-snapshot-layers.md` Reproduce section gives `CACHE_GIB=5 PROMPTS=…/prompts-bisect.json bash tp/tptf/repro.sh BIN OUT resume 256 code-160k 262144 2`. It says why 1.5 GiB never resumes this fixture, that `SCHED` must be set (line 4), and that the wall-clock deadline (lines 5–8) must be edited before any later run |
| Should-fix 3 | No cost or failure behaviour for the all-configurations default | Job 1048g on `ac346900b` (results below). `REPORT.md` section 1 and the PR body's "Churn and memory" paragraph carry it. Remaining gaps are listed under "Not tested": repeated hits under churn, failure injection, separate save/restore/drop timings, more than two states |
| Note 1 | `dropSnap` waited on an empty rank after a partial save; comment described the wrong failure | `slot_states.zig:129-141`: ranks with no buffers are skipped. The comment now says `Dev.sync` drains first, and only a later error (`XPU_PROFILE` timing read) keeps buffers until close. The `snapBytes` comment (`:48-50`) names the separate DFlash2 charge |
| Note 2 | Pin the build inputs; distinct labels for `78f42f240` and `15adb58cf` | `REPORT.md` section 6 "Build inputs": a digest per commit, plus `patches/lab-ocloc-build.diff` (the one uncommitted `zig/build/xpu.zig` line; it affects only the Gemma 4 `rows_*` images). Rows are labelled "Mitigation `78f42f240`", "Review head `15adb58cf`" and "mitigation commit". The PR body also names the TP2 int8 520 B row size, and TP1 int8 at 1,040 B |

### Round 2 runs

| Job | What | Result |
|---|---|---|
| 1048f2 `tptf-a-scratch-160k-2` | Old binary, `XPU_ATTN_PREF=0`, `XPU_ATTN_PREF_SCRATCH=256`, 165K, cache-off then cache-on | **Resumed turn fails** (`InvalidQwenToken`, resumed at 165,056). Turn 0 equal (`c0c67ffd13e5`); cache-off follow-up hash `2cac6f7e6317`. In this run the scratch allocation alone was enough; prefix execution was not needed. Mechanism not named (`raw/a-scratch-160k-2/`) |
| 1048g1 `tptf-build-head2` | Build of `ac346900b` | Built, 158 s; binary sha256 `4dcf2719efedca67…` |
| 1048g2 `tptf-churn` | Order A, B, A, C, B, A (three ~32K conversations), cache for two states; arms: two cards int8 head, two cards int8 old `2286abc0c`, one card bf16 head | **All 18 requests equal** to cache-off text and engine hash. 5 evictions and 0 failed saves per arm. Only 1 resumed request per arm (each first turn kept 2 states). Memory: two-card card 0 at 19,549–19,598 MiB free while holding 1,227–1,267 MiB, through the evictions; 31,705 MiB after stop (31,706 before). Resumed first token: head 0.305 s, old 0.270 s, one sample each (`raw/churn-*/check.jsonl`) |

### Not done in round 2

- Failure injection during a partial save.
- Separate save, restore and drop timings: the server log prints none.
- A churn order with several hits.
- More than two states at once.

None of these was run without approval.
