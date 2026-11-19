# xpu(qwen27): prompt-cache snapshots take one buffer per layer (mitigation, cause unknown)

Branch: `SergiioB:xpu-tp2-fixes-20261119`. Commits:
- `78f42f240`: the change.
- `15adb58cf`: review round 1. It corrects the code comment, drains each rank once on drop and documents the cache
  budget.
- `ac346900b`: review round 2. `dropSnap` skips empty ranks, and the comments match the tests.

This is a tested mitigation, not a root-cause fix. The failure was absent in the two tested replays, and its cause is
unknown. Not for upstream as it stands. With the same old binary and the same first follow-up, prefix off completed where prefix 256 failed. That setting changes our fork's prefix
dispatch, its scratch sizes and possibly placement and timing; #3586 has no prefix path. No single cause is named
(see "#3586" below). The first commit cherry-picks cleanly onto #3586 (`378bcf94f`, local branch
`xpu-tp2-snapshot-layers-3586`, not pushed).

## Problem

On two Arc Pro B70 cards with `--tp 2`, int8 K/V (`XPU_KV_INT8=1`), our prefix attention (`XPU_ATTN_PREF=256`) and the
prompt cache on, every follow-up turn of a long conversation fails after one token:

```
info: prompt cache: 165645 tokens, resumed at 165056, kept 1; ...
[tensorfold] ended reason=error (InvalidQwenToken) prompt=165645 tokens=1 after=1.16s
```

It was seen at 165,109 and 249,784 prompt tokens. It was not seen at 130,856 (d1, which checked completion only). The
exact first failing depth is not known, and the first turn is fine. At these depths int8 K/V is the only configuration
that fits with a useful cache: bf16 at 262,144 tokens with a 16 GiB cache was refused at load on two cards.

`saveSlot` copied a slot's whole state into one device buffer per rank: 2.8 GB at 165K tokens, 4.2 GB at 250K.
Comparing that buffer with the live state right after the save, rank 0's first 2 MiB (layer 0's GDN state and part of
layer 1's) do not hold what was copied, while rank 1's copy is exact. The restored state carries NaN from layer 0.
Every logit is then NaN, and the argmax returns `0x7fffffff`, which the drafter rejects as `InvalidQwenToken`.

## Change

Each layer gets its own snapshot buffer: K rows, K scales, V rows and V scales for an attention layer (at most
2 × tokens × 520 bytes per rank for TP2 int8, 170 MB at 165K; TP1 int8 is 1,040 bytes per K or V row), or the GDN and conv states. The same bytes are copied in
the same order.

Every configuration takes this layout (any TP degree, K/V format, slot and depth). The trigger is unknown, so no
configuration can be shown to be outside it, and one layout keeps save and restore a single code path.

`dropSnap` waits on each rank's queue once, then frees that rank's buffers.

**Cache budget.** `--prompt-cache-gib` charges `snapBytes` (one rank's payload) plus the drafter context on rank 0,
unchanged. It does not count the driver's rounding and bookkeeping for the 64 buffers per rank. Under a 4 KiB rounding
model that is 114,688 extra bytes per rank at 165,056 tokens; the real driver charge was not measured.

## Measured

Two Arc Pro B70 (180 W and 230 W, as found), Qwen3.8-27B MLX 4-bit + DFlash2, greedy, thinking off, with
`XPU_ATTN_PREF=256 XPU_KV_INT8=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE` and
`--tp 2 --parallel 1 --context 262144 --prompt-cache-gib 5 --no-thinking`.

- **Fixtures:** one conversation per depth. The first turn is `code-160k` (165,109 tokens) from
  `prompts-bisect.json`: the 250K code prompt cut by length, with the same closing instruction. At 250K it is
  `code-256k` (249,784 tokens) from `prompts-deep.json`. Follow-ups use the fixed follow-up list in
  `gates.py session`, with a 512-token first answer and 128-token follow-ups.
- **Replay procedure:** a cache-off server runs the conversation and records every request. A fresh cache-on server
  then replays those exact requests (`gates.py replay`).
- **Pass rule:** a turn passes when it completes, its text sha256 equals the cache-off text, and its engine token
  hash equals the cache-off one.

| Build | Prompt | Result | Resumed at | First token, resumed turns |
|---|---|---|---|---|
| fork base `2286abc0c` | 165,109 | completion only: turn 0 completed, both follow-ups `InvalidQwenToken`; no matched cache-off run on this binary | 165,056 / 165,632 | — |
| fork + `78f42f240` | 165,109 | 3 of 3 pass (2 of 2 resumed) | 165,056 / 165,632 | 1.2 s / 0.8 s (114 s cache off) |
| fork + `78f42f240` | 249,784 | 4 of 4 pass (3 of 3 resumed) | 249,728 / 250,304 / 250,432 | 1.6 / 1.1 / 1.1 s (205 s cache off) |
| review head `15adb58cf` | 165,109 | 3 of 3 pass (2 of 2 resumed); same engine hashes as `78f42f240` | 165,056 / 165,632 | 1.2 s / 0.8 s |

In the base row the second follow-up was 165,674 tokens. The passing replay's was 165,802, because it followed a
successful conversation history. Turn 0 at 249,784 tokens keeps the engine token hash `b08b75173e3c` of the cache-off
runs from before the change. The first-token times compare cache-on with cache-off on the same patched binary. The cost
of the extra allocations and frees was not isolated.

**#3586, a different build and configuration.** It has no prefix kernel (`XPU_ATTN_PREF=0`) and a later base than our
fork's `480167c75`. At 165,109 tokens, #3586's head `28ac4011c` completed both follow-ups with cache hits at 165,056
and 165,632. That run had no same-build cache-off baseline; its two follow-up hashes match the cache-off hashes kept
from `378bcf94f`. With the change applied (`378bcf94f`), it passes 3 of 3. This does not attribute the trigger to
anything our fork adds. In the same-build discriminator, the old fork binary `2286abc0c` with only `XPU_ATTN_PREF=0` completed both 165K
follow-ups with cache hits: with the same old binary and the same first follow-up, prefix off completed where prefix 256 failed. This is completion only, with no cache-off run
on that binary. The setting changes dispatch, scratch sizes (203,097,600 B against 198,144 B per rank) and possibly
placement and timing, so it does not name a single cause.

**Scratch test.** With prefix attention off and the prefix scratch still allocated at the prefix-256 size
(`XPU_ATTN_PREF_SCRATCH=256` on the old binary, job 1048f2), the resumed turn failed again: `InvalidQwenToken` after
1 token, resumed at 165,056. Turn 0 matched its cache-off reference (`c0c67ffd13e5`), and the follow-up's cache-off
hash was `2cac6f7e6317`. So in this run, prefix execution was not needed for the failure: the scratch allocation
alone, about 203 MB per rank, which also moves every later allocation, was enough. That points at allocation size or
placement. It does not name a mechanism (one run, one depth).

**Churn and memory, head `ac346900b`.** Three ~32K conversations in the order A, B, A, C, B, A, with the cache
sized for two states. Arms: two cards int8, the old single-buffer binary at the same depth, and one card (card 1) bf16.
- **Correctness:** in every arm all 6 requests match the cache-off text and engine hash, with 5 evictions and 0 failed
  saves. Only 1 request per arm resumed (each first turn kept 2 states and filled the budget).
- **Memory:** two-card card 0 stayed at 19,549–19,598 MiB free while 1,227–1,267 MiB of states were held through the
  evictions, then returned to 31,705 MiB after stop (31,706 before).
- **Cost:** the resumed turn's first token was 0.305 s on the head and 0.270 s on the old single-buffer build, at
  32,512 cached tokens. One sample each; save, restore and drop were not timed separately.
- **Raw data:** `raw/churn-*/`.

## Not explained

Why the single buffer goes wrong is unknown. Each of these left the trace unchanged (same bad bytes, same save hash):
- `NEOReadDebugKeys=1 EnableDirectSubmission=0`
- `NEOReadDebugKeys=1 FlushTlbBeforeCopy=1`
- `zeContextMakeMemoryResident` on the buffer before the copies
- splitting every copy at 2 GiB of the snapshot offset
- splitting every copy where either address crosses 4 GiB

These are host settings, and the logs do not prove the driver took the requested paths.

The page goes wrong during layer 27's four copies. A per-layer bisect found it correct after layer 26's copies
(implicitly: the probe logs only once a region is bad), and no kernel runs in between. The bytes that land are not
the start of any layer-27 source. Rank 0's first region is already wrong before rank 1 copies anything. Stamping the
4 GiB-aligned address inside the buffer did not change the first page, so those two addresses are distinct memory;
other aliasing is not excluded.

A standalone Level Zero program gets every word right on either card and with both list kinds. It covered:
- the same 160 copies into one 2.8 GB buffer;
- copies straddling 2 GiB and 4 GiB offsets;
- copies crossing a 4 GiB-aligned address at the same point.

It uses one active device, fresh allocations and synthetic data, so it does not recreate the engine's state.

## Reproduce

```bash
XPU_ATTN_PREF=256 XPU_KV_INT8=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE tensorfold-native serve \
  Qwen3.8-27B-MLX-4bit --drafter Qwen3.8-27B-DFlash2 --tp 2 --parallel 1 \
  --context 262144 --prompt-cache-gib 5 --no-thinking
```

1. Send the `code-160k` prompt.
2. Send one follow-up in the same conversation.
3. For the comparison, run the same history against a `--prompt-cache-gib 0` server.

Before the change, the follow-up ends with `InvalidQwenToken` after one token. After it, in the two replays tested, the
follow-up's text equals the cache-off server's. The lab harness command for exactly this replay:

```bash
CACHE_GIB=5 PROMPTS=$WORK/tptf/prompts/prompts-bisect.json \
  bash tp/tptf/repro.sh BIN OUT resume 256 code-160k 262144 2
```

`CACHE_GIB=5` matters: `repro.sh` defaults to 1.5 GiB in resume mode, below this fixture's about 2.67 GiB charged
state, and then no resume happens. The script runs only under the lab queue (`SCHED` must be set, line 4). It also
carries a fixed wall-clock deadline (lines 5–8, 2026-11-19 14:00 CET), which must be edited before any later run.

## Revert

`git revert ac346900b 15adb58cf 78f42f240` (one file, `zig/src/xpu/slot_states.zig`).

## Risk

Low to medium. The change touches prompt-cache save and restore only, on the XPU Qwen3.8-27B runner.
- **More allocations:** a save makes 64 allocations per rank instead of 1. A drop makes 64 frees after one wait per
  rank. Neither was timed separately; the resumed turns' first token stayed at 0.8–1.6 s.
- **Mitigation, not a fix:** with the cause unknown, the failure could return if it follows absolute placement, queue
  or driver state, or memory pressure rather than allocation length. A per-layer buffer can still cross an absolute
  4 GiB address.

Not tested:
- TP1 and TP4 with the cache past 131K.
- bf16 K/V past 131K.
- A TP2 int8 point near the context limit, above the old 2^32 size.
- `--parallel` above 1 with the cache.
- Repeated hits under churn (1 hit per churn arm; 5 evictions each).
- Failure injection during a partial save.
- Separate save, restore and drop timings.
- More than two cached states at once, or mixed depths.
