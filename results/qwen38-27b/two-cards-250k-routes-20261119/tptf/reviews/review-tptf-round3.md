# TensorFold TP2 review, round 3 of 3

**Approve `ac346900b` for the fork’s working branch. No merge blockers.** Two-way. The snapshot lives in process memory. Revert the snapshot stack with `git revert ac346900b 15adb58cf 78f42f240`. Revert the shared-tree change separately with `git revert df3baa3ea`. The per-layer layout remains a defensible mitigation for the two reproduced long-context failures. The new evidence covers ordinary eviction at ~32K and reproduces the 165K error with prefix execution disabled. It leaves the cause open.

| Commit | Final verdict | Scope |
|---|---|---|
| `df3baa3ea` | **Approve** | Round 2 verdict carried forward. No shared-tree code changed in this round. |
| `78f42f240` alone | **Reject** | Its driver-aliasing claim still lacks evidence in isolation. Merge with the corrections below. |
| `15adb58cf` alone | **Approve with changes** | Its empty-rank wait and cleanup wording needed correction. `ac346900b` supplies it. |
| `ac346900b` | **Approve** | Ownership and wait ordering are sound. The comment refinements below are follow-up. |
| Branch at `ac346900b` | **Approve** | Fork integration as a tested mitigation. Wider default-cost validation and the cause investigation remain follow-up. |

Source paths refer to `$WORK/tptf/src` unless named otherwise. `raw/` is next to that worktree. Queue logs are in `$WORK/queue/logs/`. Document paths are relative to [tptf](../REPORT.md).

## Code delta

`git diff 15adb58cf ac346900b` touches only `zig/src/xpu/slot_states.zig`. `git diff --check 15adb58cf ac346900b` and Zig 0.17.0 `ast-check zig/src/xpu/slot_states.zig` passed. No build or GPU execution was required for this review.

- **Empty rank.** `slot_states.zig:24` starts every pointer null. On partial save, the error defer at `:67` sees each pointer recorded at `:74` before its first copy. The all-null scan at `:134–136` therefore skips only a rank with no allocation to free and no copy referencing this snapshot.
- **Populated rank.** Every rank with at least one pointer passes through `:138` before freeing all its non-null entries at `:139`. Save and restore use the device selected for that rank (`:69–88`, `:102–120`). The copy helper appends only to that device’s list (`zig/src/xpu/dev.zig:214–215`). The skip adds neither a leak nor a missed wait for a rank that owns buffers.
- **Error path.** `slot_states.zig:129–131` now gives the right order. `Dev.sync` drains first. Failed host synchronization normally exits the process (`dev.zig:65–70`, `:140–144`). A returned profiling error makes `dropSnap` retain the rank’s buffers until close, as before. Small wording gap at `slot_states.zig:130`: the profiling error can come from the timing read **or the event reset** (`dev.zig:188–201)).
- **Charge and depth.** `slot_states.zig:48–50` correctly names the charge. `zig/src/native/xpu_lanes.zig:248–250` adds DFlash2 context bytes only when a drafter is present, and `:260–264` allocates that context on rank 0. The accounting uses one rank’s target payload and excludes per-allocation driver overhead. `slot_states.zig:19–21` now bounds the failure to the 165K and 250K tests and notes the 131K completion. Tie the first-2-MiB observation explicitly to the 165K trace (`raw/a-trace-160k/code-160k/server.log:31–32`). The retained 250K evidence shows the resume error and its absence with per-layer buffers.

## Round 2 disposition

| Item | Status | Evidence and remaining work |
|---|---|---|
| Should-fix 1. Bound the prefix-off claim | **Partly resolved** | `REPORT.md:47–52`, `:274–278`, and `pr-body-snapshot-layers.md:80–83` correctly bound the old-binary comparison. The source depth wording is fixed. `REPORT.md:30–33`, `:199`, and the PR body at `:17–26` still imply a depth threshold or a general failure rate. State which fixtures failed and which 131K run completed. The new scratch inference also needs the qualification below. Follow-up. |
| Should-fix 2. Exact replay | **Partly resolved** | `pr-body-snapshot-layers.md:141–148` now supplies `CACHE_GIB=5`, the fixture path, queue requirement and 14:00 CET deadline. The queue guard is at `repro.sh:5`, and the outer deadline is at `:6–10`. The recipe also needs the 13:55 CET startup cutoff at `:89–94`. Adjusting only the outer deadline leaves the archived wrapper unable to start a server. Follow-up. |
| Should-fix 3. Default cost and failure behavior | **Partly resolved** | `REPORT.md:59–66` and the PR body at `:92–99`, `:156–171` give ordinary churn, memory, 64 allocations per rank and the remaining gaps. Allocation-failure behavior is supported by source inspection. The run exercised zero save failures and timed no separate save, restore or drop. Follow-up. |
| Note 1. Empty-rank cleanup, error wording and draft charge | **Resolved** | `slot_states.zig:129–139` skips empty ranks and describes the drain before the error. `:48–50` names the separate draft charge. Include the event-reset detail on the next comment edit. Follow-up nit. |
| Note 2. Build pins and separate labels | **Resolved** | `REPORT.md:210–223` records the lab build diff and digest receipts. `pr-body-shared-tree-prefix.md:37–38` labels the two builds separately. I matched the current head binary to all head churn `env.txt:3` files and the tracked patch to the retained `zig/build/xpu.zig` diff. `tptf-build-head2.log:2–4` records the dirty build input. |

## Churn replay

Job 1048g2 ran 07:11:31–07:21:41 CET on 2026-11-19 (`tptf-churn.log:1,58–64`). Qwen3.8-27B MLX 4-bit with DFlash2, greedy, thinking off, one request at a time, context 40,960 and prefix 256. The B70 cap receipts give 180 W and 230 W settings. Requests were A0, B0, A1, C0, B1, A2 at 32,552 / 32,540 / 32,704 / 32,316 / 32,692 / 32,848 prompt tokens. The first scored request in each arm kept two states, and every subsequent request kept one and evicted one.

I compared the actual UTF-8 strings from each `churn.jsonl`, recomputed SHA-256, and joined the six scored rows in order to each arm’s server completion lines after the warmup. Prompt and output counts, finish reasons and engine token hashes also matched each arm’s reference. The reference logs have zero cached tokens. The old arm replays the **head TP2 cache-off reference**, as `churn_job.sh:12–13` specifies. The two head arms each have their own cache-off reference.

| Arm under `raw/` | Text and engine hash | Evictions / failed saves | Resumed requests | Client first token, resumed |
|---|---:|---:|---:|---:|
| `churn-head-tp2-int8`, `ac346900b`, 1.3 GiB | 6/6 | 5 / 0 | 1/6 | 0.305 s |
| `churn-old-tp2-int8`, `2286abc0c`, 1.3 GiB | 6/6 | 5 / 0 | 1/6 | 0.270 s |
| `churn-head-tp1-bf16`, `ac346900b`, 4.4 GiB, card 1 | 6/6 | 5 / 0 | 1/6 | 0.306 s |

All three hits were A1 at 32,512 cached tokens, with 116 output tokens and engine hash `ebb2feea161f`. See each `cached/churn.jsonl:3`, the two TP2 `cached/server.log:34–35` and `:43–44`, and TP1 `cached/server.log:31–32` and `:40–41`. Engine-reported first-token times were 0.304, 0.269 and 0.305 s respectively. The table uses client times.

On head TP2 card 0 (`0000:0b:00.0`), free visible VRAM was 31,706 MiB before the cached server, 20,830 MiB after warmup, 19,549–19,598 MiB across the six post-request samples, and 31,705 MiB after stop. Cache accounting held 1,227–1,267 MiB. Card 1 returned to 32,482 MiB from 32,482 MiB. Old TP2 card 0 went from 31,705 to 31,702 MiB. The active TP1 card went from 32,482 to 32,482 MiB. See each `cached/vram-before.txt`, `cached/vram-start.json`, `cached/churn.jsonl:1–6`, and `vram-end.txt`.

**18/18 matches includes three actual restores.** This checks two-state turnover, a successful restore and five ordinary evictions for each tested configuration. The live readings show no accumulating large loss in this short run, and process teardown returned free memory near baseline. They do not measure reclamation for individual frees, transient peaks, fragmentation over a soak, or cleanup under allocation failure. The client’s 35 ms gap has one sample per build, and the builds differ beyond snapshot layout. It cannot estimate a speed change or isolate save, restore or drop cost.

## Scratch allocation with prefix off

The first attempt ran 06:54:36–06:59:14 CET and stopped before the cached arm because of the 06:55 startup cutoff (`tptf-a-scratch-160k.log:1–6`). Job 1048f2 ran both arms at 06:59:14–07:06:23 and exited 1 on the replay check (`tptf-a-scratch-160k-2.log:1–10`). Both arms used the diagnostic build based on `2286abc0c`, SHA-256 `48ca305188718e7a492e451544885dd80a883944a45c690540e446e7caa2f3b0`, with `XPU_ATTN_PREF=0 XPU_ATTN_PREF_SCRATCH=256` (each arm’s `env.txt:3,7`). The scratch-only patch sizes the three allocations from 1,025 chunks instead of one at context 262,144, increasing their requested total from 198,144 to 203,097,600 B per rank. Dispatch still reads `q.attn_pref` (`src-scratch/zig/src/xpu/forward.zig:77–78,103–105,421–425` and `patches/scratch-diagnostic.diff`).

- Turn 0 at 165,109 tokens matched the reference text byte for byte. Both ran to 512 tokens, engine hash `c0c67ffd13e5` (`raw/a-scratch-160k-2/fresh/session.jsonl:1`, `cached/replay.jsonl:1`, `fresh/server.log:26`, `cached/server.log:29`).
- At 165,645 tokens, the cached arm resumed at 165,056 and the server ended with `InvalidQwenToken` after one engine token. The client received zero completion tokens. Cache off completed 128 tokens, hash `2cac6f7e6317` (`fresh/server.log:28`, `cached/server.log:31–33`, `check.jsonl:2`).
- The allocation configuration was sufficient to reproduce the failure **in that run with prefix dispatch disabled**. The earlier small-scratch completion used the unmodified binary with a different digest (`raw/a-old-pref0/code-160k/env.txt:3,7`, `tptf-a-old-pref0.log:8–13`). This supports investigating size, placement or allocator state. It does not isolate any one of them, establish that every later allocation moved, or show that the scratch run damaged the same bytes as the earlier trace.
- `REPORT.md:53–58`, `:288–291`, and the PR body at `:85–90` bound the observed failure to one run. Replace “which also moves every later allocation” and “the scratch allocation alone” with the exact allocation configuration and say that placement could have changed. Name the diagnostic build so readers can distinguish it from the unmodified prefix-off control.

Per-layer snapshots remain defensible on the fork. They change allocation size and placement together and preserved the matched output in the two previously reviewed long-context replays. Ordinary churn now also has a receipt. These observations support adoption as a reversible mitigation without assigning the cause to any specific size or address.

## Follow-ups in priority order

1. **Run one instrumented 165K scratch comparison when authorized.** Use a single diagnostic binary with the old single-buffer snapshot and prefix execution off. Alternate small and prefix-256 scratch sizing with the same fixture and fresh processes. Record actual scratch and snapshot addresses and allocation bounds. Independently byte-compare the first 2 MiB after each of layer 27’s four copies and verify the resumed engine hash against cache off. This directly checks whether the allocation intervention reproducibly introduces the previously traced corruption. If both sizes behave alike, the placement or runtime state needs tighter control before drawing a mechanism claim.
2. **Correct the evidence and replay wording.** Bound the two original failure fixtures and the scratch observation, name the old arm’s shared reference, and document both deadline guards. Use “first scored request” in the churn explanation (`REPORT.md:62–63`, PR body `:95`).
3. **Exercise partial-save failure.** Inject failures before the first allocation, after a successful copy append, and on the next rank’s first allocation. Verify per-rank cleanup, an uncorrupted subsequent request, and live allocator state. Include the inherited profile-error retention case.
4. **Measure default-layout cost under sustained churn.** Use same-build layout controls, multiple hits and memory-feasible mixed depths. Time save, restore and drop separately and sample memory during eviction. This also gives the existing risk table a measured cost.
