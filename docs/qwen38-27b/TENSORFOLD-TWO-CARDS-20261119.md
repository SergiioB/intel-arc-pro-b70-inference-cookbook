# Two B70s for one Qwen3.8-27B stream: what scales, where the time goes, two fixes

Measured 2026-11-18 and 2026-11-19 on two Intel Arc Pro B70 cards, TensorFold `--tp 2` (upstream
[PR #3586](https://github.com/ashhart/TensorFold/pull/3586), open) with our prefix attention kernel
(`XPU_ATTN_PREF=256`). Status: **screen**. This page follows the first two-card record in
[TENSORFOLD-B70.md](TENSORFOLD-B70.md#two-cards-for-one-stream-2026-11-18-13-to-32k-15-at-131k-250k-prompt-with-bf16-kv)
(128.1 → 165.5 tok/s). Raw files and `summarize.py`:
[`results/qwen38-27b/two-cards-250k-routes-20261119/`](../../results/qwen38-27b/two-cards-250k-routes-20261119/README.md).

## Result

1. **Serial decode splits well, a drafted round does not.** One card to two cards: serial 1.54× at a 68-token
   prompt and 1.59× at 32K; a drafted round 1.25× and 1.27×. Same binary, interleaved runs.
2. **The two-card round is now fully measured.** At 68 tokens, 33.1 ms = 18.3 half of the one-card work + 4.4
   kernels that do not halve + 4.3 rank-0-only drafter and argmax + 1.5 card 0 slower under its cap + 2.4 exchange
   kernels + 2.2 idle and host. Host submission is rejected as the main cause. At 32K the drafter's attention on
   rank 0 is the largest single item (8.5 ms a round).
3. **Leave Intel's direct submission on.** It is on by default. Forcing it off cost one card 6 % and two cards
   17–23 %.
4. **Two bugs on our fork, both closed for the tested cases.** Bug B (a draft tree that shares a round gave other
   text than alone) is fixed: 29 of 32 → 32 of 32 requests match. Bug A (resumed turns past about 131K with int8
   K/V fail `InvalidQwenToken`) has a **tested mitigation, cause unknown**: resumed == fresh at 165K (3 of 3) and
   250K (4 of 4), follow-ups start in 1.1–1.6 s instead of 205 s. Three review rounds approved the branch at
   `ac346900b`; an eviction test gave 18 of 18 requests equal to cache-off text.
5. **Decode by prompt length on two cards** falls from 121.6 tok/s at 512 tokens to 54.7 at 131,072 and 39.3 at
   245,760, drafted == serial in 21 of 21 pairs ([table](#decode-by-prompt-length-two-cards)).

## What you need

- Two Arc Pro B70 32 GB on PCIe 4.0 x8/x8 (CPU root ports). Card 0 at 180 W with a 2,400 MHz clock cap (the
  operator's cooler profile), card 1 at 230 W and 2,800 MHz. Card 1 drives the desktop.
- Linux 7.0.0-28, Intel compute-runtime 26.05.37020.3, Level Zero loader 1.28.2, Zig 0.17.0, ocloc 26.05.
- TensorFold: our branch [`tp2-attn-prefix`](https://github.com/SergiioB/TensorFold/tree/tp2-attn-prefix)
  (`2286abc0c`, PR #3586 at `480167c75` plus the prefix kernel) for the measurements;
  [`xpu-tp2-fixes-20261119`](https://github.com/SergiioB/TensorFold/tree/xpu-tp2-fixes-20261119) (`ac346900b`)
  for the two fixes; [`tpdiag-xpu-profile-tp2`](https://github.com/SergiioB/TensorFold/tree/tpdiag-xpu-profile-tp2)
  (`98f081484`) for profiling on two cards.
- `TensorFold/Qwen3.8-27B-MLX-4bit` (4-bit, groups of 64) and the `z-lab/Qwen3.8-27B-DFlash2` drafter.

## What scales and what does not

Dev runner (`xpu-qwen27`), 256 decode steps, greedy, DFlash2 tree of 15. Rates are the runner's own decode rates.
Drafted == serial in every completed run (13 of 13 in job 110, 8 of 8 in job 111).

| Prompt | Cards | n | Drafted tok/s | ms a drafted round | Serial tok/s | Serial ms a token |
|---|---|---|---|---|---|---|
| 68 | 1 | 3 | 123.4 / 123.6 / 123.5 | 41.5 | 33.0 | 30.3 |
| 68 | 2 | 3 | 150.8 / 151.6 / 151.8 | 33.2 | 50.8 | 19.7 |
| 32,552 | 1 | 2 | 77.5 / 77.6 | 60.5 | 28.0 | 35.7 |
| 32,552 | 2 | 2 | 94.7 / 94.8 | 47.6 | 44.7 | 22.4 |

Serial: 1.54× and 1.59×. Drafted round: 1.25× and 1.27×. Two cards accept slightly fewer tokens a round (5.02
against 5.12 at 68 tokens) because their text differs from one card's from about token 26, so compare ms a round.
The served path gave the same direction on 2026-11-18 (1.29× at ~60 tokens, 1.51× at 131K).

## Where a two-card round goes (measured)

A per-rank GPU profile on two cards ([profiling fix](#profiling-on-two-cards-a-driver-hang-and-its-fix)): kernel
times from the profiled runs, wall times from unprofiled runs in the same job. One profiled run per arm and depth; a
second 68-token pair agreed within 0.1 ms a round. Rank 0 (card 0) is the critical path. This replaces the
arithmetic budget of the first version of this page, which left 8.6 ms of the round unexplained.

**Drafted round, 68 tokens: 33.1 ms on two cards, 41.4 ms on one.**

| Part (rank 0) | ms a round |
|---|---:|
| Target work split in half (one-card target kernels 36.6 ms ÷ 2) | 18.3 |
| Kernels that do not halve on half the rows: GEMMs +1.8 (gate/up run as two `dec_wide` a layer, 173 µs, instead of `dec_wide64` + `dec_wide`, 152 µs), `gdn_step` +1.15 (the 16 tree rows run in sequence; 3.37 → 2.84 ms), small kernels +1.45 (fixed cost a launch) | 4.4 |
| Rank-0-only work: drafter GEMMs about 2.7, drafter kernels 1.13, drafter norm and tap 0.16, full-vocabulary argmax 0.35 | 4.3 |
| Card 0 slower than card 1 on the same work (180 W, 2,400 MHz cap): GEMMs +3.9 %, `gdn_step` +15 %, small kernels +13 % | 1.5 |
| Exchange kernels `tp2_allreduce_f16`, 128 × 18.6 µs | 2.4 |
| Idle and host time between kernels (one card: 0.9) | 2.2 |
| **Sum** | **33.1**, equal to the measured wall |

- Rank 1 idles 2.0 ms a round right before the exchanges (rank 0: 0.56 ms). Card 1 also has nothing to do during
  the 4.3 ms of rank-0-only work.
- Where the 8.6 ms went: 4.4 ms of kernels that do not halve, 2.7 ms of drafter GEMMs that the arithmetic counted as
  split, 1.5 ms of card 0 being slower, about 0.9 ms more idle and host time than on one card. The drafter is about
  4 ms of GPU work a round, more than the 2–3 ms bound used before.
- **Serial step, 68 tokens: 19.7 ms (one card 30.3).** Half of one card 15.1; GEMMs that do not halve +1.15; small
  kernels +1.17; `gdn_step` +0.10; card 0 slower +0.49; exchange 0.53 (128 × 4.2 µs); argmax 0.05; idle and host
  0.84. Sum 19.4; 0.3 ms is not placed.
- **Drafted round, 32,552 tokens: 47.6 ms (one card 60.5).** Rank 0 is busy 45.2 ms, rank 1 31.3 ms. The gap is the
  drafter: its kernels take 8.5 ms a round on rank 0 (7.7 on one card), and its attention over the context
  (`dattn`) is 7.4 ms of that on one card. The target's attention does split. Kernels that do not halve: GEMMs
  +0.6, `gdn_step` +1.1, small kernels +1.4; card 0 slower about 1.6; exchange 2.4; idle and host 2.4.

Largest removable items, **estimates, not built**: split `dattn` by heads across the two cards (about 4 ms a round
at 32K, 8 %, more at depth); pick a `dec_wide64`-class kernel for the per-rank gate/up shape 8,704 × 5,120 (up to
1.8 ms a round, 5 %, and 1.15 ms a serial step, 6 %).

### What it is not

| Candidate | Measured | Verdict |
|---|---|---|
| Host submission | unitrace host timing only (job 1049): tracing does not slow the run (150.5 against 151.4 tok/s on two cards, 123.5 against 123.5 on one). Host submission work is about 26 % of the active run on two cards and 7 % on one; the rest is waiting for the GPU | **rejected as the main cause**; with direct submission on, the round is GPU-bound |
| PCIe link | Peer copy 14.32 GB/s (91 % of the 15.75 GB/s PCIe 4.0 x8 line rate); kernel peer stores 11.36 GB/s | not the bound |
| Exchange payload | 160 KiB step: 11 µs copy-engine, 14.4 µs peer stores; full TensorFold exchange step +19.87 µs at 160 KiB, +5.12 µs at 10 KiB, event pair alone +2.7 µs | 2.4 ms of a round, measured in the profile |
| Push instead of read | priced from the step cost: (19.87 − 17) µs × 128 = 0.37 ms, 1.1 % of a round | not built into a timed run; below the 2 % gate |
| Host event resets | 512 resets a forward cost 13–24 µs | 0.1 % |
| Event scope | HOST, device-only and SUBDEVICE pools cost the same within 0.01 µs | free; correctness only |

A host round trip per exchange costs 18–29 µs (two-rank allreduce, submit to both done). Designs that wait on the
host per exchange lose; that is the llama.cpp `-sm tensor` case below.

## The driver setting that must stay on

Intel's compute runtime submits work through direct submission (NEO ULLS) by default on these cards: 23
`DRM_IOCTL_XE_EXEC` calls per host test with the default, 544 with it forced off. Interleaved, 68 tokens, job 111:

| Cards | Default drafted | `EnableDirectSubmission=0` drafted | Change | Serial default → off |
|---|---|---|---|---|
| 1 | 123.5, 123.5 | 115.3, 116.7 | −6 % | 33.0 → 31.6 |
| 2 | 151.2, 151.7 | 125.9, 116.8 | −17 to −23 % | 50.75 → 40.0 |

Do not set `NEOReadDebugKeys=1 EnableDirectSubmission=0` for TensorFold. Forcing it on (`=1`) changed nothing
(151.6 tok/s, one run).

## Fix B: a shared draft tree took another attention path

With `XPU_ATTN_PREF` set, a lone draft tree used the prefix attention kernel and a tree that shared a round with
other streams used the sliced kernel. The two paths differ in more than summation order: the prefix path computes
scores and P·V on DPAS with bf16-rounded softmax weights, the sliced path is a scalar float reduction. Under
`--tp 2 --parallel 4` the 7,963-token prose prompt gave another 512-token reply whenever it shared a round.

Commit `df3baa3ea` sends shared trees through the prefix path. 32 requests a run (4 prompts alone and together, 3
repeats, drafted and `"draft": false`), two cards, bf16 K/V:

| Build | Requests equal to the alone reply |
|---|---|
| `2286abc0c` (before) | 29 of 32 |
| `df3baa3ea` (fix) | 32 of 32 |
| `78f42f240` | 32 of 32 |
| `15adb58cf` (branch head) | 32 of 32 |
| `2286abc0c`, `XPU_ATTN_PREF` unset (control) | 32 of 32 |

The control isolates the path choice for this workload. The runs do not show which numerical difference flips the
token; no logits were kept. The same defect showed on one card: one of two drafted repeats on two independent
one-card servers differed from the alone reply (`gates/conc/two-servers`).

## Fix A: prompt-cache snapshot, one buffer per layer (tested mitigation, cause unknown)

`XPU_KV_INT8=1`, `--tp 2`, `--context 262144`, `--prompt-cache-gib 5`: the first turn at 249,784 tokens returns
512 tokens; every resumed turn then reports a cache hit and fails `InvalidQwenToken` after one token. It fails from
about 165K tokens. A 131K session on the same setup resumed without error (`kv256k/d1`), so the first failing
depth is not known.

What a byte-compare trace showed at 165,056 tokens: the first 2 MiB of rank 0's single 2.8 GB snapshot buffer do
not hold what the save copied (layer 0's recurrent state and part of layer 1's). The bad bytes appear during layer
27's four snapshot copies on rank 0, are identical over three server processes, and the restored state carries NaN.
Rank 1 is exact.

Commit `78f42f240` gives the snapshot one device buffer per layer; `15adb58cf` corrects its comment and drains each
rank once on drop; `ac346900b` makes the drop skip ranks that a partial save left empty.

| Check (two cards, int8 K/V, 5 GiB cache) | Result |
|---|---|
| 165,109 tokens, 3 turns | 3 of 3 resumed == cache-off text and token hash (also at `15adb58cf`) |
| 249,784 tokens, 4 turns | 4 of 4; follow-up first token 1.63 / 1.07 / 1.11 s with the cache, 205.1 / 205.4 / 205.5 s without |
| Same fix on PR #3586 head `28ac4011c`, 165K | 3 of 3 |
| Unmodified PR #3586 head `28ac4011c`, 165K | both follow-ups completed with cache hits (no prefix kernel, later base, no same-build cache-off run) |
| Old fork binary with `XPU_ATTN_PREF=0`, 165K | both follow-ups completed (completion only) |
| Old fork binary, `XPU_ATTN_PREF=0` but the prefix scratch still allocated at its 256 size, 165K | **failed**: `InvalidQwenToken` after 1 token, resumed at 165,056 |
| Eviction, branch head `ac346900b`: three ~32K conversations in the order A, B, A, C, B, A, cache sized for two states; arms two cards int8, the old single-buffer binary, one card bf16 | 18 of 18 requests equal to cache-off text and token hash; 5 evictions and 0 failed saves in each arm; card 0 free VRAM back to 31,705 MiB after stop (31,706 before) |

Ruled out or left unchanged by the trace, so not the explanation as tested:

- `EnableDirectSubmission=0`, `FlushTlbBeforeCopy=1`, forced residency, copies split at 2 GiB of the buffer or at
  the 4 GiB address boundary: the same 2,090,732 bad bytes and save hash every time. These are host settings; the
  logs do not prove the driver took the requested path.
- Rank 1 writing the bytes: rank 0 is wrong before rank 1 copies.
- Aliasing between the 4 GiB-aligned page and page 0: distinct memory. Other aliasing is not excluded.
- The first 32 bytes of layer 27's K rows, K scales, V rows and V scales, and the 2 MiB before each row buffer's end:
  no match.
- A kernel overrun of the fork's added scratch buffers: sizes match the per-rank shape in source, and no kernel runs
  between the layer-26 and layer-27 checks.
- The same copy layout in standalone Level Zero programs (`tptf/alias/`): exact, 0 bad words.

Prefix off completed where prefix 256 failed on the same old binary. With prefix attention off but its scratch
buffers still allocated (about 203 MB a rank instead of 198,144 B), the resumed turn failed again. So in that run
allocation size or placement was enough to trigger it, without prefix execution. One run at one depth; it names no
mechanism. **Cause still unknown.** All configurations keep the per-layer layout. In the eviction test only one
request an arm actually resumed (each first turn kept two states and filled the budget), so eviction under real
reuse is thin evidence.

**Review.** Three rounds by an independent model (`tptf/reviews/`). Round 1: 1 blocker (a driver-aliasing claim in
a code comment, removed), 5 should-fix items, answered in `tptf/REVIEW-RESPONSE.md`. Round 2: `df3baa3ea` approved,
`15adb58cf` and the branch approved with changes. Round 3: branch at `ac346900b` **approved, no merge blockers**;
`df3baa3ea` and `ac346900b` approved; the snapshot change approved as a mitigation for the two reproduced failures,
with the cause and the default's allocation cost left as follow-up. Nothing was proposed upstream.

### Reproduce both fixes

```bash
git clone https://github.com/SergiioB/TensorFold && cd TensorFold
git checkout xpu-tp2-fixes-20261119            # ac346900b: df3baa3ea (B); 78f42f240, 15adb58cf, ac346900b (A)
zig build xpu-native -Doptimize=ReleaseFast -Dngen-include=/path/to/onednn/third_party/ngen
# or apply results/qwen38-27b/two-cards-250k-routes-20261119/tptf/patches/000{1,2,3,4}-*.patch on 2286abc0c

# Bug B: four prompts alone, then together, 3 repeats, drafted and "draft": false
XPU_ATTN_PREF=256 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE zig-out/xpu/bin/tensorfold-native serve \
  Qwen3.8-27B-MLX-4bit --drafter Qwen3.8-27B-DFlash2 --tp 2 --parallel 4 --context 16384 \
  --prompt-cache-gib 0 --no-thinking --no-update-check --port 8046
python3 scripts/gates.py conc --base http://127.0.0.1:8046 --prompts prompts-qwen38.json \
  --keys code-0,chat-0,code-8k,chat-8k --out out-b      # equal_alone must be true for every stream

# Bug A: a cache-off server, then a cache-on server, same history (check.py resume compares them)
XPU_ATTN_PREF=256 XPU_KV_INT8=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE zig-out/xpu/bin/tensorfold-native serve \
  Qwen3.8-27B-MLX-4bit --drafter Qwen3.8-27B-DFlash2 --tp 2 --parallel 1 --context 262144 \
  --prompt-cache-gib 5 --no-thinking --no-update-check --port 8046
```

`tptf/repro.sh` in the evidence folder is the exact wrapper (`repro.sh BIN OUT conc 256 chat-8k 16384 3` and
`repro.sh BIN OUT resume 256 code-256k 262144 2`); it refuses a binary older than any of its source files.

## Decode by prompt length, two cards

Both cards, bf16 K/V, `--context 262144 --prompt-cache-gib 0`, binary `2286abc0c`, greedy, 256 tokens out with
`ignore_eos`, code prompts with a fresh nonce at the start of each repeat, prompt cache hits 0. One full-shape
warm-up per length (discarded) and three measured repeats, serial and drafted on the same prompt. Mean, minimum and
maximum of the three repeats, computed from `sweep-tp2-bf16/samples.jsonl` by `summarize.py`.

| Prompt tokens | Drafted tok/s, mean (min–max) | Serial tok/s, mean (min–max) | Time to first token | Prompt tok/s | Drafted == serial | Tokens a round |
|---|---|---|---|---|---|---|
| 512 | 121.6 (112.3–128.6) | 48.1 (48.0–48.3) | 0.25 s | 2,088 | 3 of 3 | 4.33 |
| 4,096 | 109.9 (106.1–113.6) | 44.4 (44.4–44.4) | 1.79 s | 2,287 | 3 of 3 | 4.34 |
| 16,384 | 132.9 (129.8–135.3) | 43.8 (43.6–43.9) | 7.56 s | 2,169 | 3 of 3 | 5.61 |
| 32,768 | 103.6 (98.2–107.2) | 42.6 (42.6–42.7) | 16.07 s | 2,039 | 3 of 3 | 4.58 |
| 65,536 | 93.2 (85.9–101.5) | 38.6 (38.6–38.6) | 36.00 s | 1,821 | 3 of 3 | 4.70 |
| 131,072 | 54.7 (54.3–54.9) | 31.5 (31.5–31.5) | 86.47 s | 1,516 | 3 of 3 | 3.49 |
| 245,760 | 39.3 (39.2–39.6) | 23.6 (23.6–23.6) | 208.04 s | 1,181 | 3 of 3 | 3.48 |

- Serial decode is the clean depth curve: 48.1 → 23.6 tok/s. Drafted decode also follows how many draft tokens
  the text accepts. The 16K prompt accepted 5.61 tokens a round and ran faster than the 4K one.
- The 245,760-token drafted rate (39.3) is below the 62.3 tok/s of the 249,784-token cell on the
  [250K page](LONG-CONTEXT-250K-20261119.md): other prompt text, 256 instead of 512 tokens, 3.48 tokens a round.
  Compare within one table.
- Another agent ran `sycl-ls` once during the first two minutes of this job (server start and the 512-token
  cells). The 512-token drafted spread (112.3–128.6) is the widest in the table.
- One-card and exl3xpu sweeps with the same script: **pending** (queued 2026-11-19, not run).

## Faster first token: int8-activation prefill (screen)

`XPU_PREFILL_INT8=1` runs the prompt GEMMs of the MLX 4-bit (groups of 64) model with int8 activations (W4A8).
Branch [`xpu-qwen27-w4a8-g64-prefill`](https://github.com/SergiioB/TensorFold/tree/xpu-qwen27-w4a8-g64-prefill)
(`4b21412ba`); patch in `prefill8/`. Three interleaved pairs a cell, drafted, median time to first token:

| Setup | Prompt tokens | Flag off | Flag on | Ratio | Decode off → on |
|---|---|---|---|---|---|
| Two cards | 7,976 | 3.604 s | 3.155 s | 1.14× | 140.7 → 142.7 |
| Two cards | 32,552 | 16.170 s | 14.201 s | 1.14× | 115.8 → 116.8 |
| One card (GPU1, 230 W) | 7,976 | 4.652 s | 3.827 s | 1.22× | 114.5 → 111.0 |
| One card | 32,552 | 21.567 s | 18.144 s | 1.19× | 89.9 → 88.9 |
| One card, one diagnostic run | 130,856 | 128.918 s | 115.176 s | 1.12× | 50.7 → 50.7 |

- The flag changes text: 7 of 8 standard prompts give other words (only the 131K code prompt stays equal).
  Drafted == serial holds on 8 of 8 with the flag on.
- 329-item battery, one run each: 282 flag off (card 0, 180 W), 283 flag on (card 1, 230 W); 4 items gained, 3
  lost, 201 of 329 outputs identical. The two arms ran on different cards.
- Status: keep it opt-in. The rebased build's fresh paired timing is **pending** (queued 2026-11-19).

## Upstream receipt for one card (posted 2026-11-18)

The upstream maintainer asked for #2595 (grid order) and #2799 (attention merge) measured together against main.
Main `8cbef9777` against main + both, one card (GPU1, 230 W), 8 prompts, drafted and `"draft": false`: the token
hash is equal in all 8 cells and the accepted-draft counts are equal. Drafted code: 1.06× at 8K (85.0 → 90.2),
1.22× at 32K (35.2 → 42.9), 1.31× at 131K (12.3 → 16.2). These are stock-attention rates, not our prefix-kernel
route. Posted on [issue #2543](https://github.com/ashhart/TensorFold/issues/2543#issuecomment-6978350342); tables in
`upstream-receipt/receipt-tables.md`.

## llama.cpp `-sm tensor`: local patch, correctness measured, speed pending

Stock llama.cpp `-sm tensor` on two B70s stages every allreduce through pinned host memory and blocks the CPU until
both cards drain (two waits per layer). Our patch adds an opt-in F32 peer-copy allreduce (`GGML_SYCL_TP_P2P=1`): no
host wait, no host buffer, no BF16 truncation of prefill partials. It applies cleanly on llama.cpp `78e05298f`
(checked 2026-11-19). Files and steps: [`llamacpp-tp-patch/APPLY.md`](../../results/qwen38-27b/two-cards-250k-routes-20261119/llamacpp-tp-patch/APPLY.md).
It stays a local patch; nothing goes to llama.cpp (operator decision, 2026-11-19).

Measured with a standalone check program on both cards (`llamacpp-tp-patch/evidence/check/`):

| Check | Result |
|---|---|
| Peer path, F32 sums | exact on both ranks after 64 × 16 reductions at 6 sizes (2,560 twice, 32,767, 32,768, 262,144, 1,310,720 values): 210,370,432 values checked |
| Stock host-staged allreduce, decode size | 49.36 and 52.24 µs a call at 5,120 values (2,000 calls, two runs), 52.18 µs at 20,480 |
| Stock, prefill size | 782.42 µs a call at 2,621,440 values (200 calls) |
| Stock, host memory | resident memory grows 2.58 KiB a call (50,000 calls, two runs) |

**Speed: pending (2026-11-19).** The peer path's own timing run produced no timing lines, and the llama-bench
baseline crashes (exit 139; under diagnosis, `evidence/diag/`). Earlier the first build failed to compile (4
errors, `ext_oneapi_can_access_peer` on a `const sycl::device`) and run jobs failed their preflight on `docker ps`
permission. No decode or prefill rate exists yet for stock or patched `-sm tensor` on this model.

## Profiling on two cards: a driver hang and its fix

`XPU_PROFILE=1` with `--tp 2` hung in the first decode step in every run: 1 of 1 on the PR #3586 head, 2 of 2 on our
branch, then 5 of 5 more with only the engine fix. Two faults:

1. **Engine:** every rank took its timing events from rank 0's single-device pool, so rank 1's kernels and the
   reduce kernels were never attributed. Fixed with one event pool and kernel-name table a rank.
2. **Intel compute-runtime 26.05.37020.3:** reused, host-reset timestamped events on in-order immediate lists with
   cross-device waits never complete while the runtime's implicit conversion to counter-based events is on.

A standalone Level Zero reproducer (`tpdiag/neo-repro/`, no TensorFold) shows the driver fault alone:

| Variant | 26.05.37020.3 | 26.40.40196.8 (loaded side by side) |
|---|---|---|
| Two devices, default | **hang** (batch 1, step 127) | pass |
| Two devices, `NEOReadDebugKeys=1 EnableImplicitConvertionToCounterBasedEvents=0` | pass | pass |
| Two devices, `EnableDirectSubmission=0` | **hang** | not run |
| One device | pass | not run |

- The key costs nothing measurable unprofiled: 151.3 / 151.4 against 151.5 / 151.7 tok/s at 68 tokens, 94.5
  against 94.6 at 32K, same binary, interleaved.
- Engine fix, pushed to our fork, not proposed upstream: [`tpdiag-xpu-profile-tp2`](https://github.com/SergiioB/TensorFold/tree/tpdiag-xpu-profile-tp2)
  (`62ab36f38`, `98f081484` on `15adb58cf`) and [`tpdiag-xpu-profile-tp2-3586`](https://github.com/SergiioB/TensorFold/tree/tpdiag-xpu-profile-tp2-3586)
  (`ac9ddd5c0`, `d02d8758f` on PR #3586 head `28ac4011c`). The runner sets the key only when `XPU_PROFILE` is set.
  Profiled runs complete with drafted == serial (149.7 tok/s on the PR head, 150.8 on our fork, 33 steps).
  Patches in `tpdiag/patches-xpu-profile-tp2*/`.
- Not filed with Intel: 26.40.40196.8 does not hang, and Intel's issue template asks for the latest driver. The
  host still runs 26.05.37020.3; nothing was installed. Record: `tpdiag/neo-issue-draft.md`.
- unitrace `--device-timing` still crashes TensorFold on two cards (segfault, exit 139; `DEVICE_LOST` at 32K);
  `--host-timing` alone works. At 07:24:44 CET such a run also left both cards with an xe "Engine memory CAT error"
  and compute-engine resets. A health canary with known output hashes passed afterwards on one card (120.89 tok/s,
  9 of 9 checks) and two cards (164.73 tok/s, 9 of 9) ([DNR-028](../DO-NOT-REPEAT.md)).

## vLLM XPU v0.31.0 with AutoRound int4: does not load here

`vllm/vllm-openai-xpu:v0.31.0`, Qwen3.8-27B AutoRound int4, one and two cards:

- **One card:** the weights load (48.7 s), then the engine process dies: `trap invalid opcode` in
  `auto_round_kernel_xpu` (kernel log, `vllm/kernel-trap.txt`). That library's host code uses AVX-512 mask
  registers (28 instructions: 14 EVEX compares, 14 masked moves; `vllm/ark-avx512-mask-lines.txt`). The Ryzen 7
  5700X3D (Zen 3) has no AVX-512 ([DNR-027](../DO-NOT-REPEAT.md)).
- **Two cards:** `onecclCommInitRank failed with code 11`; oneCCL (`ze_fd_manager.cpp:144`) cannot open
  `/dev/dri/by-path`, which the container did not mount. Mount it with `-v /dev/dri/by-path:/dev/dri/by-path:ro`.
  A dummy-weight two-card run failed the same way before the mount was added.
- A dummy-weight two-card mechanics run with the mount: **pending** (queued 2026-11-19).

## Failures and lost time

| What happened | Cost | Fix |
|---|---|---|
| A fixed 55 °C cool-down gate waited on card 1, which drives the desktop and idles at 65–66 °C | about 40 of 50 min of job 110; 45 min with nothing running (00:50–01:35 CET) | per-card gate: card 0 ≤ 55 °C, card 1 ≤ 68 °C, bounded wait ([DNR-024](../DO-NOT-REPEAT.md)) |
| The PID file written inside `systemd-run` held `$` instead of the PID | bug-A bisection job hung 22 min until its timeout | `--expand-environment=no`; stop the scope when the PID is not numeric ([DNR-026](../DO-NOT-REPEAT.md)) |
| A failed build let its dependent jobs start | llama.cpp build failed in 38 s; 7 dependents each failed in 23 s | run wrappers refuse a missing binary or one older than its sources ([DNR-025](../DO-NOT-REPEAT.md)) |
| `XPU_PROFILE=1` with `--tp 2` | both two-card profile runs hung to the 240 s timeout | engine fix plus a driver key ([profiling](#profiling-on-two-cards-a-driver-hang-and-its-fix)); the measured budget came from it |
| unitrace `--device-timing` with `--tp 2` | segfault at 68 tokens, `DEVICE_LOST` at 32K; at 07:24:44 an xe engine CAT error and resets on both cards | host-only timing works; health canary passed afterwards ([DNR-022](../DO-NOT-REPEAT.md), [DNR-028](../DO-NOT-REPEAT.md)) |
| A script edited while its job ran | `unexpected EOF` after a passing check | scripts parsed whole (`{ …; exit; }`), edits by rename |

## Not tested

- More than two cards, TP4, or another host or link.
- The served path's two-card profile (the measured budget is the dev runner).
- The push exchange, a `dattn` split or a `dec_wide64` gate/up choice in a timed run (priced only).
- Card 0 without its 180 W / 2,400 MHz cap in the profile.
- Bug A: its cause; bf16 K/V with the cache beyond 131K, `--parallel` above 1 with the cache, eviction with more
  than one resumed request an arm, the cost of 64 allocations a snapshot, a soak.
- Compute-runtime 26.40.40196.8 for serving (loaded only for the reproducer).
- vLLM XPU two-card mechanics with `/dev/dri/by-path` mounted (pending).
- Bug B: speed of shared trees after the fix; `--parallel 8` correctness.
- Sampled decoding and thinking on for the sweep and the fixes.
- Card 0 at 2,800 MHz for the root-cause runs.
- llama.cpp `-sm tensor` decode and prefill speed, stock or patched (pending).
