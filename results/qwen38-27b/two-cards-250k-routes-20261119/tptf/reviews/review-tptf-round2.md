# TensorFold TP2 review, round 2 of 3

**Branch head `15adb58cf` — approve with changes for the fork.** The driver-aliasing assertion is gone. The drop change is safe on the supported serving path by inspection. The per-layer layout is a tested mitigation in two long-context fixtures, with the exact-head replay covering 165K. Approval of the all-configurations default still needs allocation and eviction evidence. **Door — two-way.** The snapshot format lives in process memory. Revert the mitigation with `git revert 15adb58cf 78f42f240` on this branch. Revert the shared-tree change separately with `git revert df3baa3ea`.

| Commit | Verdict | Basis |
|---|---|---|
| `df3baa3ea` | **Approve** | The corrected PR body bounds the numerical claim. The 32-request result also holds at the new head. |
| `78f42f240` in isolation | **Reject** | Its source comment asserts driver aliasing without supporting evidence. The follow-up removes that assertion. |
| `15adb58cf` as a follow-up | **Approve with changes** | One wait suffices for each populated rank. Tighten the comment's description of depth and error handling. |
| Branch `xpu-tp2-fixes-20261119` at `15adb58cf` | **Approve with changes** | Accept the correction to the diagnosis. Close the scope, evidence wording, and reproduction findings below before approving the default. |

Source paths beginning with `zig/` refer to the TensorFold worktree at `$WORK/tptf/src`. `raw/` and `repro.sh` refer to the adjacent `tptf/` run directory. Queue logs are under `$WORK/queue/logs`. Document references use the filenames in `tptf/`. [Report](../REPORT.md), [snapshot PR body](../pr-body-snapshot-layers.md), [shared-tree PR body](../pr-body-shared-tree-prefix.md).

## Remaining findings

### Blockers

None in the round 2 delta. The original unsupported driver diagnosis is resolved in the branch source.

### Should fix before approval

1. **Bound the prefix-off finding to this run.** Use “with the same old binary and the same first follow-up, prefix off completed where prefix 256 failed.” `REPORT.md:46`, `:240`, `:252` and `pr-body-snapshot-layers.md:79` go further by naming the prefix path as *the* trigger or treating two proposed mechanisms as exhaustive. The setting changes dispatch, scratch sizes, and potentially allocation placement and command timing. The run only establishes completion and cache hits here. In `zig/src/xpu/slot_states.zig:19`, use “failed in the 165K and 250K tests” in place of “failed from about 165K.” There is also a passing 131K completion observation. The first failing depth is unknown.
2. **Give the exact cached replay command.** The manual server command now includes the right prefix setting, cache size, and fixture. The added lab command at `pr-body-snapshot-layers.md:120` omits `CACHE_GIB=5`. `repro.sh:169` defaults to 1.5 GiB in resume mode, below this fixture's roughly 2.67 GiB charged state. That recipe can fail to exercise a resume. Use the command at `REPORT.md:145` with the fixture path. State the queue requirement and the historical deadline at `repro.sh:5` and `:7` if linking the archived wrapper as a runnable command.
3. **Show the cost and failure behavior of the default layout.** `zig/src/xpu/slot_states.zig:22` gives a sound reason to avoid an arbitrary TP, format, or size cutoff. That leaves a cost applied to every cache-enabled configuration. `pr-body-snapshot-layers.md:130` and `:136` acknowledge the coverage gap. The head run confirms two ordinary evictions with one retained state. It gives no multi-state churn, allocation failure, or live memory-after-eviction check. Validate those cases and record save, restore, and drop time separately. See the allocation accounting below.

### Notes

- **New cleanup behavior on an empty rank.** `zig/src/xpu/slot_states.zig:66` can enter `dropSnap` after a partial save, before a later rank owns any buffer. `:131` now waits even on that empty rank. The old loop skipped null entries before calling the wait. Skip empty rows to avoid draining an unrelated list on this error path. For a populated rank, `Dev.sync` drains first and then can return a profiling error. A failed Level Zero host wait normally exits the process. Only the profiling-error branch of this call retains memory until close, so the new comment at `:128` describes the wrong failure. The profiler behavior for populated ranks is inherited. See `zig/src/xpu/dev.zig:65`, `:140`, and `:188`.
- **Pin the tested build inputs.** The head binary digest matches all three head run receipts. `queue/logs/tptf-build-head.log:2` also records a dirty `zig/build/xpu.zig`. The retained worktree diff at `zig/build/xpu.zig:20` changes the compiler flag for the Gemma 4 `rows_g4s` and `rows_g4g` images. The Qwen images use the separate `opts` branch at `:40`. This is a lab build based on `15adb58cf`. Record the local build diff beside the digest and give the `78f42f240` and `15adb58cf` rows distinct labels in `REPORT.md:21`, `:189`, and `pr-body-shared-tree-prefix.md:37`.

## Drop safety and budget

1. **A rank's list is the only list that receives its snapshot pointers.** Save allocates under `devAt(rk)` and stores the pointer before appending any copy. Restore uses that same rank's buffer, live state, and device list. Both call `dev.copy`, which passes only `d.list` to Level Zero. The TP peer buffers are separate and receive no snapshot pointer. Thus rank 0 can free its snapshot buffers once its own list has drained, even if a peer list still has commands. TP1 runs the same logic once on `r.d`. See `zig/src/xpu/slot_states.zig:68`, `:101`, `:125`, and `:130`, `zig/src/xpu/dev.zig:214`, and `zig/src/xpu/qwen27.zig:213` and `:328`.
2. **Multiple cached states preserve that ownership.** Each save gets distinct allocations. The engine thread owns the store, and eviction skips borrowed entries. A drop touches only the evicted entry's non-null pointers. If an append fails after earlier copies were queued, the save error defer sees each allocated pointer and the list drains before any free. A partial restore can leave copies queued on the same list, so the drop still drains the list that holds those references. The separate rank-0 DFlash2 context uses the rank-0 list and still gets its own wait in `keepDrop`. Thus the new count is one wait per rank **for the target snapshot**, plus one for the separate draft allocation when present. See `zig/src/native/xpu.zig:78` and `:80`, `zig/src/core/prompt_cache_policy.zig:28`, `zig/src/core/prompt_cache.zig:159` and `:170`, `zig/src/core/prompt_cache_lifetime.zig:14`, `zig/src/native/xpu_lanes.zig:253` and `:269`, and `zig/src/xpu/drafter.zig:345`.
3. **The byte charge has the same meaning.** The store evicts against one rank's target payload plus 21,299,200 B for DFlash2 when present. The draft context is physically on rank 0. This is neither an aggregate across TP ranks nor a limit on driver memory. Startup admission uses rank 0's memory. Each allocation requests 4 KiB alignment. Rounding each extent to 4 KiB would add 114,688 B per rank at 165,056 and 98,304 B per rank at 262,144, relative to the old single extent. Those figures model rounding only. Actual residency and bookkeeping were unmeasured. The PR body correctly explains this. The source comment describes the target component and could name the separate draft charge. See `zig/src/xpu/slot_states.zig:43` and `:48`, `zig/src/native/xpu_lanes.zig:248`, `zig/src/xpu/drafter.zig:341`, `zig/src/native/xpu.zig:118` and `:125`, `zig/src/core/prompt_cache_keep.zig:36` and `:52`, and `zig/src/xpu/ze.zig:430`.

### Where per-layer allocation costs more

The runner has 48 recurrent layers and 16 attention layers. For a rank, the requested snapshot bytes are `48 × (recBytes + convBytes) + 16 × 2 × at × (kvRow + int8ScalesRow)`, with the scale term zero for bf16. Save and restore still issue 160 copies per rank in int8 and 128 in bf16, in the same order. Allocating per layer increases target allocation and free calls from 1 to 64 per rank per snapshot, plus driver and ownership-map records. It also reduces the largest individual allocation. `zeDeviceProperties` exposes a per-allocation size field, but the path has no allocation-count check and the receipts establish no driver count limit. See `zig/src/xpu/qwen27_config.zig:8` and `:12`, `zig/src/xpu/qwen27.zig:76`, `zig/src/xpu/slot_states.zig:26`, `:71`, and `:104`, and `zig/src/xpu/ze.zig:125`, `:234`, and `:430`.

| TP and K/V at 262,144 tokens | Target payload per rank | Largest new request |
|---|---:|---:|
| TP1 bf16 | 17,333,813,248 B (16.14 GiB) | 1,073,741,824 B (1 GiB) |
| TP1 int8 | 8,878,096,384 B (8.27 GiB) | 545,259,520 B (520 MiB) |
| TP2 bf16 | 8,666,906,624 B (8.07 GiB) | 536,870,912 B (512 MiB) |
| TP2 int8 | 4,439,048,192 B (4.13 GiB) | 272,629,760 B (260 MiB) |
| TP4 bf16 | 4,333,453,312 B (4.04 GiB) | 268,435,456 B (256 MiB) |
| TP4 int8 | 2,219,524,096 B (2.07 GiB) | 136,314,880 B (130 MiB) |

`pr-body-snapshot-layers.md:35` should label the 520 B-per-row maximum as **TP2 int8**. TP1 int8 uses 1,040 B per K or V row including scales. At TP2 int8 with DFlash2, two near-limit states need roughly 8.31 GiB of charged budget. They create 128 target allocations per rank, versus two previously. Three need roughly 12.46 GiB and 192 allocations per rank. The tested 5 GiB budget keeps only one near-limit state. These are budget calculations, with fit also depending on live K/V, weights, and available memory. The new largest extent is much smaller, which could improve allocation success. More differently sized extents can add allocator overhead when states churn. The retained data establish neither direction for fragmentation. `zig/src/xpu/limits.zig:5`, `zig/src/xpu/slot_states.zig:34` and `:43`.

A concrete cost case is TP1 bf16 at an 8,192-token cache mark with prefix off. A 16,384-token server can run this shape, subject to the usual load admission. Each snapshot requests about 659 MiB. At the default 2 GiB budget, three such entries fit in budget even with the draft charge. That is 192 live target allocations instead of three. This is clearly more allocation work and more allocation failure opportunities on save. Restore allocates no snapshot buffers and keeps the same copy count, payload and synchronization boundary. A restore-time regression is possible from placement and driver bookkeeping, but source arithmetic cannot establish one. Cache-off runs allocate no snapshots. The simplicity argument supports one layout as a design choice. It does not supply the missing measurements for a default across all cache-enabled shapes.

## Round 1 disposition

| Round 1 item | Status | Round 2 evidence |
|---|---|---|
| Blocker | **Resolved** for the unsupported diagnosis. **Partly resolved** for the scope concern. | `slot_states.zig:18` removes driver attribution and the 2 GiB threshold. `:22` explains the single layout. The broad default still needs the cost and failure evidence above. |
| Should-fix 1 | **Resolved** | `REPORT.md:12` and `:24` and `pr-body-shared-tree-prefix.md:15` and `:40` distinguish the dispatch change from which arithmetic difference might flip a token. They give the mixed one-card repeats and the lack of logits. Keeping pushed `df3baa3ea` is acceptable. Its statement about order is incomplete as a causal account. Preserve the corrected wording in the PR or squash message if the change is integrated. |
| Should-fix 2 | **Partly resolved** | `pr-body-snapshot-layers.md:50`, `:54`, `:58`, and `:125` fix the server configuration, fixture, replay design, and revert. `:120` still omits `CACHE_GIB=5` for the lab replay. |
| Should-fix 3 | **Resolved** | `pr-body-snapshot-layers.md:65` and `:70` label completion only and give the different second request lengths. `:75` and `REPORT.md:42` separate #3586 by base, settings, and lack of a same-build cache-off run. The added same-build discriminator has its own wording finding above. |
| Should-fix 4 | **Partly resolved** | `slot_states.zig:48` and `:130` specify payload accounting and reduce waits. `pr-body-snapshot-layers.md:44` gives the draft charge and rounding model. Head `cached/server.log:31` and `:34` show two single-state evictions. `pr-body-snapshot-layers.md:141` leaves repeated multi-state churn, injected failure, and driver memory checks open. |
| Should-fix 5 | **Resolved** for the original wording and probe caveats. | `REPORT.md:29`, `:53`, `:61`, and `:226` give the tested scope, exact interventions, limited stamp, and implicit layer-26 check. The push arithmetic at `:68` is a derived estimate. The probe's explicit clean result and error propagation remain for a future diagnostic. |

| Round 1 note | Status after the delta |
|---|---|
| 1. Shared-tree prefix bound | **Resolved** as a correctness check. The reasoning still applies. `zig/src/xpu/forward.zig:421` is outside the delta. The new B replay preserves all four engine hashes. |
| 2. Scratch bounds and prefix-off dispatch | **Resolved** as a source check. `zig/src/xpu/forward.zig:423` and `:425` are outside the delta. The exact-head B replay covers TP2 bf16. |
| 3. Snapshot copy and ownership | **Resolved** by source review for normal use. `zig/src/xpu/slot_states.zig:71`, `:104`, and `:130` keep the rank-local copy and free ordering. The empty-rank wait is new. Failure and driver reclamation coverage remains partial. `zig/src/xpu/ze.zig:442` already ignored the free result and removed the allocation from tracking. |

## Recomputed evidence

**Head at 165K.** On two B70s at logged 180 W and 230 W caps, Qwen3.8-27B MLX 4-bit plus DFlash2, greedy and thinking off, TP2 with int8 K/V, `XPU_ATTN_PREF=256`, `--parallel 1`, 262,144 configured context. One conversation, identical replay fixture. Compared retained UTF-8 text directly and recomputed SHA-256 for both sides. Joined prompt length, finish reason, output count, cache hits, and logged engine token hash to the server receipts. The cache-off arm has zero hits in each row.

| Turn | Actual prompt tokens | Cache-on hit tokens | Output tokens per arm | Engine token hash on both arms |
|---|---:|---:|---:|---|
| 0 | 165,109 | 0 | 512 | `c0c67ffd13e5` |
| 1 | 165,645 | 165,056 | 128 | `b2c355d78f54` |
| 2 | 165,802 | 165,632 | 128 | `24ec0db0fe76` |

**3/3 match, including 2/2 resumed.** All six completions ended at the requested length. Each engine hash also matches the earlier 165K candidate on both arms. See `raw/a-head-160k/fresh/session.jsonl:1` and `cached/replay.jsonl:1` through line 3, `raw/a-head-160k/fresh/server.log:26`, `:28`, `:30` and `cached/server.log:29`, `:32`, `:35`. The previous logs have the same line numbers under `raw/a-fix-160k/`. See `queue/logs/tptf-a-head-160k.log:3` and `:13` for counts and clean exit. Startup VRAM snapshots and closing receipts give whole-run observations. They provide no measurement immediately after an eviction or failed save.

**Head concurrency check.** Same model, drafter, cards, caps and prefix setting. TP2 bf16, `--parallel 4`, 16,384 configured context, cache off. Rehashed all 32 texts from `raw/b-head/conc.jsonl:1` through line 32 and compared each to its drafted solo text. All 32 match, including 24 concurrent requests across three drafted and three serial four-request batches. Eight requests were solo. The 32 non-health server completions each have 512 output tokens, zero hits, and finish by length. The counts from `raw/b-head/server.log:26` through line 90 are eight each for 68 tokens (`1e55952eff61`), 52 (`596a68a0fce5`), 7,976 (`46ebf41df2f3`), and 7,963 (`5b5869d7083a`). See `queue/logs/tptf-b-head.log:41` and `:42`.

**Old binary discriminator.** The `XPU_ATTN_PREF=256` and `=0` runs have the same executable SHA-256 (`e6e59bfd…`, retained worktree `2286abc0c`), fixture SHA-256 (`9fc38a4e…`), launch arguments, caps, and other recorded environment settings. The first follow-up has identical retained messages, 165,645 input tokens and 165,056 hit tokens. At 256 it stops with `InvalidQwenToken` after one output token. At 0 it completes 128 tokens with `2cac6f7e6317`. Prefix off also completes the second follow-up with 128 tokens, hash `24ec0db0fe76`, and 165,632 hit tokens. Its prompt has 165,802 tokens, while the prior error leads the prefix-256 arm to send 165,674. See `raw/a-{bisect,old-pref0}/code-160k/env.txt:2` through line 7, both `session.jsonl:2` and `:3`, the on arm's `server.log:31` and `:36`, and the off arm's `server.log:31`, `:32`, `:34`, and `:35`. The off run exits successfully at `queue/logs/tptf-a-old-pref0.log:14`.

The discriminator shows that disabling prefix attention on the old build avoided the observed error in this conversation. It isolates a setting change at matched build and first-follow-up input, with one run per setting at different times. It includes no same-build prefix-off cache-off baseline or byte comparison, so completion gives no output or state parity claim. In particular the first follow-up hash differs from the patched prefix-on result. At TP2 and 262,144 context the setting also reduces requested prefix scratch from 203,097,600 B to 198,144 B per rank and changes the executed kernels. It cannot distinguish either effect or its interaction with placement, earlier work, or driver state. It supplies no proof of page aliasing, a size threshold, or safety across other prefix-off configurations. See `zig/src/xpu/forward.zig:77`, `:102`, and `:423`.

## Approval conditions

1. Correct the exact replay recipe, bound the discriminator language, and pin the review-build inputs. The existing pushed commit history can stand.
2. When GPU work is authorized, retain a TP2 int8 multi-state eviction/reallocation run and a fitting TP1 bf16 cache-enabled check on the proposed head. Compare restored text and engine hashes with same-build cold requests. Include a partial-save allocation failure and check actual memory reclamation, allocation count, and save/restore/drop timing. Use a larger state count or mixed depths within a verified memory budget. These checks address the blanket default and the new cleanup path.
3. Keep the claim at observed mitigation scope. A root-cause or upstream claim needs same-process evidence that distinguishes dispatch from scratch allocation and placement, plus direct before/after snapshot bytes.

Review validation used `git diff --check 78f42f240 15adb58cf` and Zig 0.17.0 `ast-check zig/src/xpu/slot_states.zig`. Both passed. Evidence recomputation read the retained JSONL, server logs, environment records, and executable digests on CPU.
