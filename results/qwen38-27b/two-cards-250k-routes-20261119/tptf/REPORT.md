# TensorFold TP2: two correctness fixes, one speed change priced and not run

Agent `tptf`, night of 2026-11-18 to 19, two Intel Arc Pro B70 cards. Takes over from the first `tptf` agent.
Ledger ids follow the 01:00 CET renumbering (RL-282 to RL-286). Nothing here was opened or posted upstream.

## 1. Result

**Bug B (`--tp 2 --parallel 4` stream differs from the same prompt alone) is fixed.** The 7,963-token `chat-8k`
prompt gave a different 512-token reply whenever its draft tree shared a round with other streams. The cause is in
our fork, not in PR #3586. With `XPU_ATTN_PREF` set, a lone draft tree took the prefix attention kernels and a shared
tree took the sliced kernels. The two paths differ numerically in more than order:
- the prefix path computes scores and P·V on DPAS with bf16-rounded softmax weights (`decode_prefix.cl:34`, `:42`);
- the sliced path is a scalar reduction with float weights (`decode.cl:125`, `:137`).

Commit `df3baa3ea` sends shared trees through the prefix path. On two cards:

| Build | Requests matching (of 32) |
|---|---|
| Old build | 29 |
| Fixed build | 32 |
| Mitigation `78f42f240` | 32 |
| Review head `15adb58cf` | 32 |
| Old build, prefix path off | 32 |

For this workload, the prefix-off control isolates the choice of attention path as the cause. That makes it a dispatch
bug, not documented batch-dependent numerics. The runs do not isolate which numerical difference flips the token, and
no logits were kept to show a near tie. The older one-card comparison had one differing and one matching drafted
repeat, and per-round co-scheduling was not recorded for it.

**Bug A (int8 K/V, prompt cache, every resumed turn fails `InvalidQwenToken`) has a tested mitigation on our fork:
the failure is absent in the two tested replays, and its cause is unknown.**
- **Depth:** it fails from about 165K prompt tokens; d1 resumed at 131K without an error. The exact first failing depth
  is not known.
- **What goes wrong:** a byte-compare trace shows rank 0's prompt-cache snapshot does not hold what the save copied in
  its first 2 MiB, which is layer 0's GDN state and part of layer 1's. The restored state carries NaN and the argmax
  returns `0x7fffffff`.
- **Mitigation:** commit `78f42f240` gives the snapshot one device buffer per layer. `15adb58cf` (review round 1)
  corrects its comment and drains each rank once on drop.
  - At 165,109 tokens, the resumed text equals the cache-off text on 3 of 3 turns (2 of 2 resumed).
  - At 249,784 tokens, the original coordinate, it equals the cache-off text on 4 of 4 turns (3 of 3 resumed). The
    first token comes in 1.1–1.6 s instead of 205 s, measured cache-on against cache-off on the same patched binary.
  - Turn 0 keeps the engine token hash `b08b75173e3c` of the cache-off runs from before any change.
- **#3586 comparison, a different build and configuration:** PR #3586's head `28ac4011c`, which has no prefix kernel
  and runs at prefix off, completed both follow-ups at 165K. That build has no same-build cache-off baseline.
  `28ac4011c` is also a later base than our fork's `480167c75`, so the comparison cannot attribute the trigger to
  something our fork adds. The mitigation stays on our fork.
- **Discriminator (review round 1, job 1048):** the old fork binary `2286abc0c` at the failing 165K fixture with only
  `XPU_ATTN_PREF=0` completed both follow-ups, with cache hits at 165,056 and 165,632 and hashes `2cac6f7e6317` and
  `24ec0db0fe76`. With the same old binary and the same first follow-up, prefix off completed where prefix 256 failed. The setting changes several things at once: dispatch, prefix
  scratch size (203,097,600 B against 198,144 B per rank), and possibly allocation placement and command timing. One
  run per setting, completion and cache hits only, no cache-off reference at prefix off. It does not name a single
  cause or rule out others.
- **Scratch test (review round 2, job 1048f2):** With prefix attention off and the prefix scratch still allocated at the prefix-256 size
  (`XPU_ATTN_PREF_SCRATCH=256` on the old binary, job 1048f2), the resumed turn failed again: `InvalidQwenToken` after
  1 token, resumed at 165,056. Turn 0 matched its cache-off reference (`c0c67ffd13e5`), and the follow-up's cache-off
  hash was `2cac6f7e6317`. So in this run, prefix execution was not needed for the failure: the scratch allocation
  alone, about 203 MB per rank, which also moves every later allocation, was enough. That points at allocation size or
  placement. It does not name a mechanism (one run, one depth).
- **Churn and memory (review round 2, job 1048g, head `ac346900b`):** three ~32K conversations in the order A, B, A,
  C, B, A, with the cache sized for two states. Arms: two cards int8, the old single-buffer binary at the same depth,
  and one card bf16. In every arm all 6 requests matched the cache-off text and engine hash, with 5 evictions and
  0 failed saves. Only 1 request per arm resumed (A's first follow-up), because each first turn kept 2 states and
  filled the budget. Memory is reclaimed: card 0 on the two-card head had 20,830 MiB free after load and stayed at
  19,549–19,598 MiB while 1,227–1,267 MiB of states were held through 5 evictions, then returned to 31,705 MiB after
  stop (31,706 before). The resumed turn's first token: head 0.305 s, old single-buffer build 0.270 s, at 32,512
  cached tokens. That is one sample each; save, restore and drop are not timed separately.
- **Where it goes wrong:** the bad bytes appear during layer 27's four snapshot copies on rank 0. They are not the
  start of any layer-27 source.
- **Left the trace unchanged:** each intervention below kept the same 2,090,732 bad bytes and save hash. That shows
  these settings did not change the outcome. It does not prove the driver took the requested paths.
  - `NEOReadDebugKeys=1 EnableDirectSubmission=0`;
  - `FlushTlbBeforeCopy=1`;
  - `zeContextMakeMemoryResident` on the snapshot before the copies;
  - splitting copies at 2 GiB of the snapshot offset;
  - splitting copies where either address crosses 4 GiB.
- **Probed and not seen:**
  - rank 1 writing it (rank 0 is wrong before rank 1's copies);
  - aliasing between the two addresses stamped (the 4 GiB-aligned page and page 0). This doesn't exclude other
    aliasing.
- **Standalone:** Level Zero programs outside TensorFold copy the same layout, straddling copies and 4 GiB crossings
  exactly. They use one active device, fresh allocations and synthetic data, so they don't recreate the engine's
  state.

**Speed: priced, not run.** tpdiag's chain microbench (job 106, `tpdiag/raw/chain-summary.tsv:13`) gives a
19.87 µs incremental median per 163,840-byte exchange step today, over 25 repeats with 0 errors. tpdiag derives about
17 µs for the push design. (19.87 − 17) µs × 128 exchanges ≈ 0.367 ms of the 33.2 ms short-prompt drafted round,
about 1.1 %, below this plan's 2 % gate. No timed run was made. Most of the TP2 gap is elsewhere (tpdiag `TP-ROOT-CAUSES.md`, "Final ranked table"). In tpdiag's accounting, measured
parts sum to 24.6 ms of the 33.2 ms round at 68 tokens: split kernels 19.35 ms, exchanges 2.55, rank-0-only work 1.4,
host gaps 1.3. That leaves about 8.6 ms unexplained. The next speed step, not started here, is a per-rank
`XPU_PROFILE`: one timestamp pool per rank, with rank 1's kernel handles attributed (`dev.zig:178-186`,
`qwen27.zig:472-476`). Then profile one card against two cards, to split half-size kernels that don't halve from
waiting on the capped card 0.

## 2. What you need

- **GPUs:** two Intel Arc Pro B70 32 GB, PCIe 4.0 x8/x8.
  - `0000:0b:00.0`: 180 W, with a 2,400 MHz clock cap from the operator's cooler profile.
  - `0000:0f:00.0`: 230 W.
  - Caps were not changed. Every run's `env.txt` records caps and temperatures, and `cleanup.txt` shows them restored.
- **Software:** kernel `7.0.0-28-generic`, compute-runtime `26.05.37020.3-1` (`libze-intel-gpu1`), Level Zero loader
  1.28.2, gcc for the standalone checks.
- **TensorFold worktrees** under `$WORK/tptf/`. Each carries the lab's
  uncommitted ocloc line.

| Worktree | Branch or state | Head | Pushed |
|---|---|---|---|
| `src` | `xpu-tp2-fixes-20261119` | `78f42f240` → `df3baa3ea` → `2286abc0c` | yes, to `fork` |
| `src-3586` | `xpu-tp2-snapshot-layers-3586` | `378bcf94f` on PR #3586 head `28ac4011c` | no (local) |
| `src-3586-base` | detached PR head | `28ac4011c` | — |
| `src-a-probe` | diagnostic, uncommitted | `patches/probe-cache-trace.diff` | — |
| `src-push` | push patch, uncommitted | `patches/push-f16-not-run.diff` | — |

- **Build:** Zig 0.17.0 in `$TF/toolchain`, nGEN from oneDNN. Builds go through the queue with `build.sh`.
- **Models:** `$TF/models/Qwen3.8-27B-MLX-4bit`, drafter `$TF/models/Qwen3.8-27B-DFlash2`.
- **Prompts:**
  - `results/2026/11/b70-tonight-20261108/prompts-qwen38.json`
  - `results/2026/11/tf-prefix-attn-20261113/prompts-deep.json`
  - `$WORK/tptf/prompts/prompts-bisect.json`. This is the code-256k body cut by
    length, with the same closing instruction. Its checksum is in `scripts/prompts-bisect.sha256`.

## 3. Timeline

CET, 2026-11-19. Start and end times come from `queue/sched.log`.
Total: 00:07 to 05:27 for the GPU work.
Time lost: 22 min to the bisection hang, 23 s and one queue slot to a failed probe build, and about 15 min of host
time to `tpllama`/`tpdiag` work that my exclusive builds waited behind. That last item is expected, not lost.

| Time | Job or action | Result |
|---|---|---|
| 00:07–00:30 | Read briefs, plan, job files and scripts; source read | Bug B cause found in source |
| 00:20 | Commit `df3baa3ea` | |
| 00:22–01:30 | Queue rework. Lead's d1/d2/d3b land | d1 (131K int8 TP2) resumes without error; d2 and d3b refused at load |
| 01:36–02:01 | `tptf-a-bisect` (160K/200K/230K) | 165,109 tokens fails like 250K. The script then hung until the 1,500 s timeout |
| 02:01 | Hang cause | `systemd-run` turned `$$` into `$`, so the pid file held `$` and `stop()` could not kill the server |
| 02:07–02:15 | `tptf-build` | `src` and `src-a-probe` built; `src-push` failed to compile |
| 03:06–03:09 | `tptf-a-trace-160k` | Rank 0's snapshot: first 2 MiB differ right after the save; rank 1 exact; NaN from layer 0 |
| 03:09–03:12 | `tptf-build-push` | Built |
| 03:12–03:15 | `tptf-b-prefix` (old build) | Bug B reproduced: 3 of 32 mismatch |
| 03:21–03:24 | `tptf-a-trace2-160k` | Bad bytes do not change in 200 ms; a second copy holds |
| 03:23–03:27 | `tptf-b-fixed` | 32 of 32. rc 2 came from editing `repro.sh` while it ran (the check had already passed) |
| 03:32–03:40 | Arms: default, `EnableDirectSubmission=0`, `XPU_SNAP_RESIDENT=1` | All identical: 2,090,732 bad bytes, same hashes |
| 03:40–03:44 | `tptf-b-slices` (prefix off) | 32 of 32 |
| 03:44–03:47 | `FlushTlbBeforeCopy=1` | Identical failure |
| 03:48–03:53 | Standalone `page_alias`, `copy_alias straddle` and `layout` | 0 bad words everywhere |
| 03:58–04:01 | `XPU_SNAP_SPLIT=1` (split at 2 GiB of the buffer) | Identical failure; rank 0 already wrong before rank 1 copies |
| 04:01–04:10 | `tptf-a-fix-160k` (per-layer buffers) | 3 of 3 turns, resumed == cache-off |
| 04:11 | Commit `78f42f240` | |
| 04:12–04:13 | Probe build 5 | Compile error; the stale-binary guard stopped the dependent run |
| 04:13–04:32 | `tptf-a-fix-250k` | 4 of 4 turns |
| 04:32 | Push `xpu-tp2-fixes-20261119` to `fork` | |
| 04:37–04:40 | Per-layer check of rank 0's first region | Correct after layers 0–26, wrong from layer 27 |
| 04:46–04:59 | #3586 head and #3586 + fix at 165K | Head: follow-ups complete with cache hits. With fix: 3 of 3 equal |
| 04:59–05:02 | `tptf-b-final` (`78f42f240`) | 32 of 32 |
| 05:07–05:10 | VA alias stamp | No alias: the 4 GiB-aligned page and page 0 are distinct |
| 05:14–05:15 | Standalone `copy_alias vacross` | 0 bad words |
| 05:17–05:20 | `XPU_SNAP_SPLIT=2` (split at 4 GiB of the address) | Identical failure |
| 05:24–05:27 | Source bytes at layer 27 | Page 0 matches no sampled layer-27 source. Last GPU run for bug A (lead's decision) |

## 4. Steps

1. All jobs ran through the queue. Job files are in `queue/done/` and `queue/failed/` (`*tptf*`), logs in
   `queue/logs/tptf-*.log`.
2. Bug A reproduction and fix check: a cache-off server, then a cache-on server, with the same history.

   ```bash
   CACHE_GIB=5 PROMPTS=$WORK/tptf/prompts/prompts-bisect.json \
   bash tp/tptf/repro.sh BIN OUT resume 256 code-160k 262144 2   # code-256k with the default PROMPTS for 250K
   ```

   This is the server command it runs, copied from `env.txt`:

   ```bash
   XPU_ATTN_PREF=256 XPU_KV_INT8=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE tensorfold-native serve \
     $TF/models/Qwen3.8-27B-MLX-4bit --drafter $TF/models/Qwen3.8-27B-DFlash2 --tp 2 --parallel 1 \
     --context 262144 --host 127.0.0.1 --port 8046 --prompt-cache-gib 5 --no-thinking --no-update-check
   ```

   `check.py resume` passes a turn only when:
   - it completes;
   - its text sha256 equals the cache-off text;
   - the engine token hash equals the cache-off one;
   - turns after the first have a cache hit.
3. Bug B: `repro.sh BIN OUT conc 256 chat-8k 16384 3`.
   - It serves with `--parallel 4` and runs code-0, chat-0, code-8k and chat-8k alone, then concurrently.
   - It does 3 repeats, each drafted and with `"draft": false`.
   - `check.py conc` requires every reply to equal its solo reply, and one engine token hash per prompt length.
4. Bug A trace: `repro.sh PROBE_BIN OUT probe 256 code-160k 262144 1` with `XPU_CACHE_TRACE_MIN=150000`. Arms add
   `EXTRA_ENV="KEY=VALUE …"`. The probe compares every snapshot byte at save, before restore and after restore. It
   then checks rank 0's first region after each rank and after each layer, and logs the first non-finite layer and
   the argmax.
5. Standalone checks: `alias/run.sh`, `run2.sh` and `run3.sh` build `page_alias.c` and `copy_alias.c` against
   `libze_loader` and run them per card.

## 5. Fix table

| # | Literal symptom | Cause | Fix | Verified by |
|---|---|---|---|---|
| B | `--tp 2 --parallel 4`: `chat-8k` reply `e7d11631…` concurrent, `fab2e881…` alone | Shared trees took `attn_decode`, lone trees `attn_pref` + `attn_tail` (our prefix kernel; PR #3181 closed). The two paths differ numerically (DPAS with bf16 weights against a scalar float reduction) | `df3baa3ea`: `pre = q.attn_pref` for every decode round | Old 29/32; fixed 32/32; head 32/32; prefix off 32/32 |
| A | `ended reason=error (InvalidQwenToken) prompt=165645 tokens=1` from ~165K tokens (int8, TP2, fork) | The first 2 MiB of rank 0's single snapshot buffer are wrong after layer 27's copies. Cause unknown (section 7) | Mitigation `78f42f240`: one snapshot buffer per layer; `15adb58cf` comment and drain | Failure absent in the two tested replays: 165K 3/3 and 250K 4/4, resumed == cache-off |
| H1 | Bisection job hung 22 min after the first depth | `systemd-run` expands `$$`; the pid file held `$` | `--expand-environment=no`; `stop()` stops the scope if the PID is not numeric | CPU check: real PID written, kill ends the wrapper |
| H2 | `repro.sh: line 213: unexpected EOF` after a passing check | Script edited while a job ran | Body wrapped in `{ …; exit; }`; edits by temp file and rename | No later job disturbed |
| H3 | A run would use the previous binary after a failed build | — | `repro.sh` exits 3 when the binary is missing or older than any `.zig`/`.cl`/`.h` file in its tree | Caught probe build 5 |
| H4 | `qwen27.zig:478: expected '?*anyopaque', found '[2]?*anyopaque'` | The push patch put an array in the kernel struct the profiler walks | Two plain fields | Built 03:12 |
| H5 | `bench.py` read the server's `done` line right after the stream closed | Race with the log write | Waits up to 10 s for the new line | Not exercised (no timed run) |

## 6. Verification

All raw paths are under `$WORK/tptf/` unless noted.

**Build inputs.** Every binary is a lab build. The tree is at the named commit plus one uncommitted line in
`zig/build/xpu.zig`: `opts_rows` drops `-cl-intel-128-GRF-per-thread`, which ocloc 26.05 rejects. That flag only
affects the Gemma 4 `rows_g4s`/`rows_g4g` images; the Qwen images use `opts`. The diff is
`patches/lab-ocloc-build.diff` (sha256 `ba1c7866…`).

| Commit (tree) | Binary sha256 | Used by |
|---|---|---|
| `2286abc0c` (`src-tp3586`) | `e6e59bfd64cd449d…` | w1, bisect, b-prefix, b-slices, old-pref0 |
| `78f42f240` (`src`) | `ec9efcab6843c3f5…` | a-fix-160k, a-fix-250k, b-final |
| `15adb58cf` (`src`) | `fb240d8c58b8fcaa…` | a-head-160k, b-head |
| `ac346900b` (`src`) | recorded in `raw/churn-*/*/env.txt` | churn (1048g) |
| `28ac4011c` (`src-3586-base`) | `8b4ed775f74e0c22…` | 3586-base-160k |
| `378bcf94f` (`src-3586`) | `4cc6f4e22ad12d2f…` | 3586-fix-160k |
| `2286abc0c` + `XPU_ATTN_PREF_SCRATCH` (`src-scratch`, uncommitted) | recorded in `raw/a-scratch-160k-2/*/env.txt` | 1048f2 |

- **Bug B:** `raw/b-prefix/check.jsonl` (3 mismatches), `raw/b-fixed/check.jsonl`, `raw/b-slices/check.jsonl`,
  `raw/b-final/check.jsonl` (32 requests each, pass). Engine token hash per prompt length:

  | Prompt length (tokens) | Token hash |
  |---|---|
  | 68 | `1e55952eff61` |
  | 52 | `596a68a0fce5` |
  | 7,976 | `46ebf41df2f3` |
  | 7,963 | `5b5869d7083a` with prefix and fix; `1167cd16ac89` with prefix off |

- **Bug A failure:** traces in `queue/logs/tptf-a-trace-160k.log`, `tptf-a-trace2-160k.log`, `tptf-a-arm-*.log`,
  `tptf-c-arm-layers2.log`, `tptf-i-arm-va.log`, `tptf-l-arm-split-va.log`, `tptf-n-arm-sources.log`. In every run,
  the save at 165,056 shows:
  - rank 0: 3 of 160 regions differ, 2,090,732 bytes, 6,345 non-finite;
  - rank 1: 0 of 160.
- **Bug A fix at 165K:** `raw/a-fix-160k/check.jsonl`, 3 of 3 pass.
- **Bug A fix at 250K:** `raw/a-fix-250k/check.jsonl`, 4 of 4 pass.
  - Cached tokens: 0, 249,728, 250,304, 250,432.
  - Engine token hashes `b08b75173e3c`, `3a9d7ed2d312`, `0eeed4d2659a`, `4edf6e36c80c`, equal on both servers.
  - Binary sha256 `ec9efcab…`; VRAM free and caps as found in `raw/a-fix-250k/cleanup.txt`.
- **#3586:** `raw/3586-base-160k/code-160k/server.log` (both follow-ups `done`, cached 165,056 and 165,632) and
  `raw/3586-fix-160k/check.jsonl` (3 of 3 pass).
- **Standalone:** `alias/out-034850/result.txt`, `alias/out2-035303/result.txt`, `alias/out3-051436/result.txt`.
  0 bad words in every case. Copies of the C sources are in `tp/tptf/alias/`.

## 7. Results and claim boundary

All runs are correctness only: greedy, thinking off, Qwen3.8-27B MLX 4-bit + DFlash2, on two B70s with the
operator's caps. No speed number in this report is new.

Bug B's fix covers `--parallel 4` with bf16 K/V and the prefix kernel at `XPU_ATTN_PREF=256`. Bug A's mitigation was tested with int8
K/V, `--parallel 1`, 262,144-token context and a 5 GiB cache, at 165,109 and 249,784 prompt tokens with 2 or 3
follow-ups.

**Open question (bug A).** What writes about 2 MiB of int8 K/V-like bytes into the first page of rank 0's single
2.8 GB snapshot buffer during layer 27's four copies? These things were established at 165,056 tokens:
1. It is deterministic: three server processes give identical bad bytes.
2. Between the check after layer 26 and the check after layer 27, both cards are synchronized and only layer 27's
   copies run; no kernel runs. Limits of this bisect: it inserts readback copies after every layer, it logs only once
   a region is bad, and it treats a readback error as zero. So "clean after layer 26" is implicit, not an explicit
   logged zero. A rerun should log the clean result and propagate readback errors.
3. The bytes are not the first 32 bytes of layer 27's K rows, K scales, V rows or V scales, nor the rows 2 MiB before
   each row buffer's end. They differ between the saves at 165,056 and 165,632.
4. The trace stayed the same with `EnableDirectSubmission=0`, with `FlushTlbBeforeCopy=1`, with forced residency,
   and with copies split at 2 GiB of the buffer or at 4 GiB of the address. These are host settings; the logs do not
   prove the driver took the requested paths.
5. The two addresses stamped (the 4 GiB-aligned page inside the buffer and page 0) are distinct memory. Other
   aliasing is not excluded.
6. The same copy layout outside TensorFold is exact.
7. PR #3586's head `28ac4011c` completed both follow-ups at 165K. It is a different build and configuration: no prefix
   kernel, a later base than our `480167c75`, and no same-build cache-off baseline. So it does not attribute the
   trigger to our fork's additions. In the same-build discriminator (job 1048, completion only), with the same old binary and the same first follow-up, prefix off completed where prefix 256 failed.
   That setting changes dispatch, scratch size and possibly placement and timing together, so it does not name a single
   cause.

From source, the fork's added buffers (`pmp`/`plp` 787,200 B, `pop` 201,523,200 B per rank) are written within their
sizes with the per-rank shape. With fact 2, that makes a kernel overrun unlikely. Layer 27 is also where the snapshot
crosses a 4 GiB-aligned address (`0xffffd55b00000000`), but splitting there did not help. I treat that as coincidence
until it can be tied to a size.

**Re-entry test (one 3-minute run, probe already built).** When rank 0's first region first goes bad, copy the whole
2 MiB of snapshot page 0 to the host. Search all 32 of rank 0's attention row buffers (16 layers, K and V,
`at × 512` bytes each) for its first 64 bytes, and log layer, K/V and position. That names the exact source and
offset of the stray bytes, and from it the copy and the address it really went to. The next split, after the prefix-off pass (job 1048): reserve the same `pmp`/`plp`/`pop` scratch at prefix off
(jobs 1048e/1048f2). Result: the resumed turn failed with prefix execution off, so the scratch allocation alone was
enough in that run. Re-entry: keep prefix off and vary only placement, for example a dummy allocation of the same
203 MB made before or after the KV caches, and add the byte search above. No GPU run for this was approved tonight.

**Not tested:**
- TP4.
- One card with int8 and the cache beyond 131K (its state does not fit beside a 262K context).
- bf16 K/V with the cache beyond 131K on our fork.
- `--parallel` above 1 with the cache.
- `--parallel 8`.
- Sampled decoding.
- Structured output.
- A soak.
- Speed of shared trees after the bug B fix.
- Cost of 64 allocations per snapshot.

## 8. If you do this again

1. Run the byte-compare trace first; the bisection only bounded the depth.
2. Log the bytes that land, together with every source of the suspect copies, in the same run as the first trace.
3. Keep queue scripts parsed whole, edit them by rename, and make runs refuse stale binaries.
4. Price an exchange change against a measured per-step cost before building it.
5. The shortest path to tonight's state is about 2 h of queue time:
   - build (6 min);
   - trace at 165K (3 min);
   - per-layer bisect (3 min);
   - per-layer fix at 165K (9 min) and 250K (19 min);
   - the three bug B runs (10 min);
   - #3586 pair (13 min).

## 9. Adapting it

- **Large prompt-cache states:** any engine that keeps a multi-GB state in one device allocation on the B70 should
  compare the saved bytes with the live state once at its largest size.
- **New rank count or K/V format:** this changes the per-rank state size, so repeat the same compare.
- **Several decode kernels:** an engine whose `--parallel` path picks a different attention kernel from its
  single-stream path needs the alone-versus-concurrent test from section 4, step 3, for every kernel option.
