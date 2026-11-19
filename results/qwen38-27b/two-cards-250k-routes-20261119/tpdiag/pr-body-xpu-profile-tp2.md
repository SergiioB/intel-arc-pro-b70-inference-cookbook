# xpu(qwen27): XPU_PROFILE works on --tp 2

Draft, not posted. Branch `tpdiag-xpu-profile-tp2-3586` (pushed to github.com/SergiioB/TensorFold 2026-11-19 09:30 CET, no PR opened) on top of PR #3586's head `28ac4011c`:
`ac9ddd5c0` (per-rank profiler) and `d02d8758f` (NEO key under XPU_PROFILE). The same two commits on our fork head
`15adb58cf`: branch `tpdiag-xpu-profile-tp2` (`62ab36f38`, `98f081484`).

## Problem

`XPU_PROFILE=1` with `XPU_TP=2` hangs in the first decode step after the prompt: 1 of 1 run on #3586's head,
2 of 2 on `2286abc0c` (#3586 + a prefix-attention commit), and, with only fault 1 fixed, 5 of 5 more (#3586 head 1,
our fork 4, one of them with direct submission off). Two separate faults:

1. Every rank's launches take events from rank 0's single-device timestamp pool, and only rank 0's kernel handles
   are named, so rank 1's kernels and the TP reduce kernels are never attributed (`dev.zig` `Dev.launch`/`sync`,
   `qwen27.zig` profiler init). Draining one rank when the pool fills mid-forward can also wait on a peer event that
   has not been appended yet.
2. On compute-runtime 26.05.37020.3, reusing the host-reset timestamped events on in-order immediate lists with
   cross-device waits never completes when the runtime's implicit counter-based event conversion is on. A standalone
   Level Zero reproducer (no TensorFold) hangs on two devices 3 of 3 times, passes on one device, passes with
   `NEOReadDebugKeys=1 EnableImplicitConvertionToCounterBasedEvents=0`, and passes on runtime 26.40.40196.8.

## Fix

1. One timestamped event pool and kernel-name table per rank (`Dev.rank`); the one-shot reduce kernels signal a
   profile event through a `TpComm.prof_take` hook; a full pool leaves a launch unprofiled and counts it instead of
   draining mid-forward; `syncAll` drains every rank. The report adds, per rank and kernel, the idle time on the
   in-order list right before the kernel, and the drafter's own kernels as `(other)`.
2. When `XPU_PROFILE` is set, the runner sets the two NEO keys before `zeInit` (`setenv` with overwrite 0, so an
   explicit setting wins). Nothing changes without `XPU_PROFILE`. On runtime 26.40 this commit is not needed; drop
   it if the project requires 26.40.

## Measured (2x Arc Pro B70, PCIe 4.0 x8, Qwen3.8-27B MLX 4-bit + DFlash2 tree 15, greedy, 68-token prompt)

| Run | Before | After |
|---|---|---|
| `XPU_PROFILE=1 XPU_TP=2`, #3586 head, 33 steps | hang (timeout 240 s) | completes, drafted == serial, 149.7 tok/s profiled |
| same, our fork, 33 steps | hang | completes, drafted == serial, 150.8 tok/s profiled |
| same, our fork, 256 steps | hang | completes, per-rank report (below) |
| unprofiled `XPU_TP=2`, #3586 head, 33 steps | 178.8 tok/s | 179.3 tok/s (profiler code inactive) |

What the per-rank report shows on this host (drafted round, 68 tokens): rank 0 busy 30.9 ms, rank 1 25.0 ms; rank 1
idles 2.0 ms a round before the reduce kernels; the reduce kernels take 18.6 µs each at 16 rows. The profiler adds
about 19 % on two cards and 8 % on one.

## How to reproduce

```
XPU_PROFILE=1 XPU_TP=2 XPU_DRAFTER=$MODELS/Qwen3.8-27B-DFlash2 XPU_TREE=15 XPU_CTX=40960 \
  zig-out/bin/xpu-qwen27 $MODELS/Qwen3.8-27B-MLX-4bit prompt.ids 33
```
Before: the prompt report prints, then no output until killed. After: the run ends with `equal to serial: true` and a
per-rank report.

## How to revert

`git revert d02d8758f ac9ddd5c0` (two commits, profiler-only code paths).

## Risk

Low: every change is behind `XPU_PROFILE` except a `rank` field on `Dev` (default 0) and an optional hook pointer in
`TpComm` (default null). Tested on two Arc Pro B70 with runtime 26.05.37020.3 only; four-card (`--tp 4`) not tested.
