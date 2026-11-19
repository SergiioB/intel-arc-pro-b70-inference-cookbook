# Two Arc Pro B70s · tensor-parallel root causes

Status: **final for tonight, 2026-11-19 07:35 CET; budget rows replaced by measurements 08:50 CET (job 2191).** Every number is labelled measured, estimate or hypothesis.
Sections below the final table are the evidence, in the order they were written (00:35-07:30). Owner: agent `tpdiag`.

## Final ranked table (07:35 CET)

Workload: TensorFold `2286abc0c` dev runner, Qwen3.8-27B MLX 4-bit + DFlash2 tree 15, greedy, 68-token prompt,
two Arc Pro B70 on PCIe 4.0 x8 each (card 0 capped 180 W / 2,400 MHz). One drafted round: one card 41.4 ms,
two cards 33.2 ms (5.02-5.12 tokens per round); serial step 30.3 → 19.7 ms. "Per token" = per committed token
(round / 5.02). "Per layer" = per round / 64 layers.

| Rank | Cause | Where it lives | Cost on two cards | Evidence | Proposed fix | Expected gain (priced) | Owner |
|---:|---|---|---|---|---|---|---|
| 1 | **Kernels that do not halve** on half the rows: gate/up GEMMs run as 2 `dec_wide` per layer on two cards, `gdn_step` over 16 sequential tree rows, small per-layer kernels with fixed launch cost | engine (kernel choice per rank shape; `gdn_step`) | 4.4 ms/round, 0.88 ms/token, 69 µs/layer (**measured**, per-rank profile, job 2191) | `raw/prof/prof2-0-*.log`, `scripts/budget.py` | `dec_wide64`-class kernel for the 8,704 x 5,120 per-rank gate/up shape; fuse small kernels | ≤ 1.8 ms/round from the GEMM choice (5 %), ≤ 1.15 ms per serial step (**estimate** = measured excess) | `tptf` / next |
| 1b | **Drafter attention `dattn` on rank 0 only** (at depth) | engine `drafter.zig` (all launches on rank 0) | 7.4-8.5 ms/round at 32K (16-18 % of the round), 0.46 ms at 68 tokens (**measured**) | `raw/tf-trace/round-budget-32k-tp1.tsv`, `raw/prof/prof2-32k-*.log` | split `dattn` by heads across ranks | about 4 ms/round at 32K (**estimate**), more at 131K | next |
| 1c | **Card 0 slower than card 1** on the same work (operator cap 180 W / 2,400 MHz): every exchange waits for it; rank 1 idles 2.0 ms/round before the exchanges | platform setting | 1.5 ms/round, 0.49 ms per serial step (**measured**) | same | none (operator setting) | — | operator |
| 2 | **Exchange = cross-card event pair + peer read** of 160 KiB per exchange, 128 per forward | engine `tp.zig:126-148`, `tp.cl` `tp2_allreduce_f16` | 2.55 ms/round = 0.51 ms/token = 40 µs/layer; serial 0.65 ms/token (**measured** per step x 128) | `raw/chain-summary.tsv` (replay arms, 25 reps, 0 errors) | push by peer stores (+17 µs/step est.) or copy-engine push overlapped with the next GEMM (+13.7 µs/step est.) | 0.36 ms/round (1 %, store push) to 0.8 ms/round (2.4 %, copy engine) (**estimates**); `tptf` priced the store push below its 2 % gate and did not run it | `tptf` (closed for now) |
| 4 | **Rank-0-only work**: drafter GEMMs and kernels, full-vocab argmax for 16 rows, logit gather (3.97 MB) | engine `drafter.zig`, `forward.zig:558-580` | 4.3 ms/round at 68 tokens (**measured**, per-rank profile), 11+ ms at 32K (row 1b) | `raw/tf-trace/round-budget-0-tp1.tsv` | per-rank argmax on the vocab shard, exchange (value, index); split the drafter | ≤ 0.5 ms/round from argmax/gather; ≤ 0.4 ms from a drafter split (**estimates**) | `tptf` (gather, if wanted) |
| 5 | **Host submission from one thread for both ranks.** About 2.3x the commands of one card; per command 2.6 µs append + about 0.8 µs of arguments | engine `forward.zig:380-514`; runtime L0 immediate lists | host busy about 26 % of the active run on two cards vs 7 % on one (**measured** API totals, job 1049); **not the bound** with direct submission on: the host spends the rest waiting for the GPU. Direct submission off: −17 to −23 % on two cards, −6 % on one (**measured**, n=2) | `raw/tf-hosttrace/0-tp{1,2}-host.*.stats`, `raw/tf/ds-*.log` | keep direct submission on (it is the default); replayed per-round command lists only if a later change makes kernels shorter | small at today's kernel sizes (**estimate**) | `tptf` (no action) |
| 6 | **Event scope SUBDEVICE** on shared cross-card events (comment says DEVICE) | engine `ze.zig:362-366` | 0 µs: HOST scope and a device-only pool measured equal | `raw/chain-summary.tsv` | use HOST or DEVICE scope (correctness, free) | correctness only | `tpdiag` (finding; no TensorFold patch written tonight) |
| 7 | **512 host event resets per forward** | engine `tp.zig:94-103` | 13-24 µs per forward (**measured** 26-47 ns per reset) | `raw/chain-summary.tsv` | none | < 0.1 % | closed |
| 8 | **Link**: PCIe 4.0 x8 per card on CPU root ports (cards are Gen5 x16 capable) | platform (board) | peer copy 14.3 GB/s (91 % of line rate), kernel peer stores 11.4 GB/s (**measured**); payload floor 160 KiB = 11-14 µs | `raw/peer-*.csv`, `evidence/platform.txt` | none in software | — | — |
| 9 | **llama.cpp SYCL `-sm tensor`** stages each allreduce through host memory and waits on the host twice per call | engine `ggml-sycl.cpp:7855-7886` | an isolated host-waited exchange costs 18-29 µs here (**measured**, allreduce primitive); RL-250: ~275 µs/call older build | `raw/allreduce-*.csv`; `tp/tpllama/` | device-resident exchange without host waits | see `tpllama` | `tpllama` |
| 10 | **vLLM XPU v0.31 TP2** | runtime packaging (oneCCL) and image (AutoRound kernel library) | load fails: TP2 `onecclCommInitRank failed with code 11` because `/dev/dri/by-path` is not mounted in the container (oneCCL `ze_fd_manager.cpp:144`); TP1 traps `invalid opcode` in `auto_round_kernel_xpu` (its host code has 14 AVX-512F EVEX compares and 14 masked moves; Zen 3 has no AVX-512) (**measured**, job 215) | `raw/vllm/*.server.log`, `vllm-so/ark.dis` | mount `/dev/dri/by-path`; avoid AutoRound int4 on this CPU | dummy-weight BF16 24-layer rerun queued (2466) | — |

Tools on two cards: unitrace `--device-timing` crashes `--tp 2` (tool-triggered, see below). TensorFold `XPU_PROFILE=1`
hung on `--tp 2`; fixed on branches `tpdiag-xpu-profile-tp2` (fork head) and `tpdiag-xpu-profile-tp2-3586` (#3586 head),
verified (job 2193). Root cause of the hang: compute-runtime 26.05.37020.3 with implicit counter-based event conversion;
a standalone Level Zero reproducer hangs on two devices 3/3, passes on one device, with the key off, and on 26.40.40196.8
(`repro/`, `neo-issue-draft.md`: not to be filed, fixed upstream). PR body for #3586: `pr-body-xpu-profile-tp2.md`.

## Measured budget from the per-rank profile (job 2191, 08:45 CET) — replaces the estimates above

Profiler: branch `tpdiag-xpu-profile-tp2` (`62ab36f38` on `15adb58cf`): one timestamped event pool and name table per
rank, TP reduce kernels profiled, idle before each kernel; run with `EnableImplicitConvertionToCounterBasedEvents=0`
(the hang fix, below). That setting does not change unprofiled speed: 151.3 / 151.4 vs 151.5 / 151.7 tok/s at
68 tokens, 94.5 vs 94.6 at 32K (same binary, interleaved). Profiling costs +19 % on two cards and +8 % on one, so
kernel times come from the profiled runs and walls from the unprofiled runs of the same job. Script `scripts/budget.py`,
raw `$WORK/tpdiag/raw/prof/prof2-*.log`, n = 1 profiled run per arm and depth
(a second profiled 68-token pair agrees within 0.1 ms/round). Rank 0 = card 0 (180 W, 2,400 MHz), the critical path.

**Drafted round, 68 tokens: measured wall 33.1 ms on two cards (41.4 on one card).**

| Part (rank 0 critical path) | ms/round | Kind |
|---|---:|---|
| Target work split in half (one-card target kernels 36.6 ms / 2) | 18.3 | measured parts, ideal split |
| Kernels that do not halve (rank 1 minus half of one card, same card and clock): GEMMs +1.8 (on two cards gate/up run as 2 `dec_wide` per layer, 173 µs, instead of `dec_wide64` + `dec_wide`, 304/2 = 152 µs), `gdn_step` +1.15 (tree rows are sequential; splitting heads barely helps: 3.37 → 2.84 ms), norms/attention/GDN small kernels +1.45 (fixed per-launch cost) | 4.4 | measured |
| Rank-0-only work: drafter GEMMs about 2.7, drafter kernels 1.13 (`dattn` 0.46 of it), drafter norm/tap 0.16, full-vocab argmax 0.35 | 4.3 | measured (drafter GEMMs = rank 0 minus rank 1 GEMM time minus the clock share) |
| Card 0 slower than card 1 on the same target work: GEMMs +3.9 % (0.64), `gdn_step` +15 % (0.42), small kernels +13 % (0.46) | 1.5 | measured |
| Exchange kernels `tp2_allreduce_f16`, 128 x 18.6 µs | 2.4 | measured |
| Idle and host between kernels and forwards (wall − rank 0 busy) | 2.2 | measured (one card: 0.9) |
| **Sum** | **33.1** | matches the measured wall |

Where the earlier "unexplained 8.6 ms" went: 4.4 kernels that do not halve, 2.7 drafter GEMMs that my estimate wrongly
counted as split, 1.5 card 0 being slower, about 0.9 more idle/host than on one card; the drafter itself is about
4 ms of GPU work per round, more than the ≤ 2-3 ms bound used earlier (that bound came from wall minus attributed time).
**Waiting on card 0:** rank 1 idles 2.0 ms per round right before the exchange kernels (rank 0: 0.56 ms); the cost to
the round is the 1.5 ms by which card 0 is slower, plus the 4.3 ms of rank-0-only work during which card 1 has nothing.

**Serial step, 68 tokens: measured 19.7 ms (30.3 on one card).** Half of one card 15.1; GEMMs that do not halve
+1.15; small kernels +1.17; `gdn_step` +0.10; card 0 slower +0.49; exchange kernels 0.53 (128 x 4.2 µs); argmax
0.05; idle/host 0.84. Sum 19.4 (profiled kernel times against the unprofiled wall; 0.3 ms not placed).

**Drafted round, 32,552 tokens: measured 47.6 ms (60.5 on one card).** Rank 0 busy 45.2 ms, rank 1 31.3 ms. The
gap is the drafter: its own kernels take 8.5 ms per round on rank 0 (one card 7.7), of which `dattn`, the drafter's
attention over the context, is 7.4 ms per round on one card (unitrace, `raw/tf-trace/round-budget-32k-tp1.tsv`,
5 calls a round). The target's attention (`attn_pref` 9.8 ms on one card) does split across the cards.
Kernels that do not halve: GEMMs +0.6, `gdn_step` +1.1, small kernels +1.4; card 0 slower on target work about 1.6;
exchange 2.4; idle/host 2.4.

**Single largest removable item:**
- At 32K and beyond: the drafter's attention `dattn` running on rank 0 only (7.4-8.5 ms per round, 16-18 % of the
  round), while card 1 idles. Fix: split `dattn` by heads across the two ranks like the target attention. Expected
  gain about 4 ms per round at 32K (8 %, **estimate**: half of the measured 8.5 ms), growing with context.
- At 68 tokens: the gate/up GEMM shape on two cards (`dec_wide` x 2 instead of `dec_wide64`), +1.8 ms per round and
  +1.15 ms per serial step. Fix: pick the `dec_wide64`-class kernel for the per-rank gate/up shape (8,704 x 5,120).
  Expected gain up to 1.8 ms per round (5 %) and 1.15 ms per serial step (6 %) (**estimate**: the measured excess).

## Headline for `tptf` (03:25 CET, from job 110; details in "Measured 2026-11-19" below)

| # | Finding | Kind | Number |
|---|---|---|---|
| H1 | NEO direct submission (ULLS) is **on by default** on these cards with runtime 26.05; forcing it off costs two cards 24 % | measured, n=1 per forced arm, n=3 default | drafted 151.4 → 115.7 tok/s; serial 50.8 → 38.1 tok/s (68 tokens) |
| H2 | The split works for serial decode; the drafted round does not | measured | serial 30.3 → 19.7 ms/token (1.54x); drafted round 41.4 → 33.2 ms (1.25x); 32K: 1.59x vs 1.27x |
| H3 | One-card drafted round, attributed kernel sum 39.5 ms (rank 0, profiled): almost all of it is the 16-row target verify. Against a 30.3 ms serial step the round adds 9.3 ms: `gdn_step` +2.94, `dec_narrow` +2.74, `dec_wide64` +1.79, `gdn_pre` +0.53, `dec_wide` +0.33, `swiglu` +0.28, `argmax_part` +0.26, rest +0.4 ms | measured (profile `raw/tf/0-7-profile-1.log`) | — |
| H4 | The drafter is **not** the gap (corrected 03:40; an earlier version of this row estimated about 6 ms from its weight size, which was wrong). 2026-11-18 profile: wall minus profiled GPU time is 5.1 ms a round including host waits, drafter under 3 ms (`research/2026/b70-breakthrough-20261118/README.md` section 2). Tonight: 41.4 ms wall unprofiled − 39.5 ms attributed = at most about 2 ms a round for drafter plus host on one card | measured bound | ≤ 2-3 ms per round, the same on one and two cards |
| H5 | Rank-0-only rows in a TP2 round: drafter (≤ 2-3 ms, above), full-vocab `argmax_part` for 16 rows (0.30 ms measured on one card), logit gather from rank 1 (about 0.35 ms, estimate). An ideal TP2 round is then about 39.5/2 + 2 + 0.65 ≈ 22.4 ms. Measured 33.2 ms: **about 11 ms per round is TP2-specific overhead inside the 16-row verify forward**, against 4.6 ms per serial step (19.7 − 30.3/2) | estimate | 11 ms/round ≈ 85 µs per exchange over 128 exchanges of 160 KiB; serial ≈ 36 µs per 10 KiB exchange |
| H6 | The verify forward's extra TP cost (about 11 vs 4.6 ms) grows with the exchange payload, which is 16x larger for a 16-row tree (160 KiB per exchange, about 20 MiB each way per round, about 2 ms at 10 GB/s). Candidates, to be priced: (a) payload bandwidth, (b) the peer **read** in `tp2_allreduce_f16` vs a push, (c) un-split rows (head per shard, gather, argmax, host tree build), (d) attention rows per rank. Budget per part summing to 33.2 ms: job 112 (trace, 16-row rounds) and job 102 (`chain`, 160 KiB arm) | hypothesis | — |

## Drafted-round budget, 68 tokens (05:45 CET)

Two-card per-kernel timelines could not be taken: unitrace with device timing crashes TensorFold `--tp 2`
(segfault at 68 tokens, `DEVICE_LOST 0x70000001` at 32K; `raw/tf-trace/0-tp2.log`, `32k-tp2.log`). One-card trace
works (`raw/tf-trace/round-budget-0-tp1.tsv`, 16 complete rounds, traced wall 46.2 ms, untraced 41.4 ms).
The two-card column is therefore arithmetic from measured parts; every row says which.

| Part | One card, ms/round | Two cards, ms/round | Basis |
|---|---:|---:|---|
| Target weight-streaming GEMMs (`w4k1_d64`, `w4k_dsplit`, `w4k1_decode`) incl. the drafter's share (about 1.6) | 31.4 | 15.7 | measured on one card; two cards = half (ideal split) |
| `gdn_step` (16 tree rows, sequential) | 3.3 | 1.65 | measured; half (heads split) |
| norms, attention, GDN pre/gate, swiglu, small kernels | 4.0 | 2.0 | measured; half |
| drafter-only kernels (`dattn`, `dconv`, `dprep`, `taps_in`, `topk_part`, `tap`) | 0.73 | 0.73 | measured; rank 0 only |
| full-vocab `argmax_part` + `argmax` (16 rows) | 0.31 | 0.31 | measured; rank 0 only |
| peer logit gather, 3.97 MB | — | 0.35 | estimate at 11-14 GB/s |
| 128 exchanges at 160 KiB (event pair + peer read) | — | 2.55 | measured per step in `chain` (19.9 µs) x 128 |
| host and sync gaps (round end, tree build, syncAll) | 1.3 | 1.3 | one card: 41.4 untraced − 40.1 traced device time; assumed equal on two cards |
| **sum** | **41.0** (measured 41.4) | **24.6** | |
| **measured** | 41.4 | **33.2** | job 110, n=3 |
| **unexplained on two cards** | — | **8.6** | |

The unexplained 8.6 ms per round (about 26 % of the two-card round) is what the next measurement has to split.
Candidates, each with what is known:

1. **Half-size kernel efficiency.** Every GEMM reads half the rows on each card; fixed launch and tail costs do not
   halve. Serial decode shows the same pattern: 19.7 ms/token on two cards against 15.1 ideal + 0.65 exchange
   (measured parts) leaves about 3.9 ms per serial token. Unmeasured per kernel (needs the two-card trace).
2. **Rank imbalance at every exchange.** Card 0 runs at 180 W / 2,400 MHz, card 1 at 230 W / 2,800 MHz; every one of
   the 128 exchanges waits for the slower card. Earlier measurement: the clock cap costs 4.3 % on two cards
   (about 1.4 ms per round). The `chain` costs exclude imbalance (both ranks equal work).
3. **Host submission close to the critical path.** Direct submission off costs one card 6 % but two cards 17-23 %
   (job 111, n=2 per arm, measured). With it on, the host appends about 14 commands per layer and rank at
   2.3-2.5 µs (about 35 µs) against roughly 150 µs of GPU work per rank and layer half: not the bound, but at
   about a quarter of it (estimate). With it off (6-8 µs per append) the host exceeds the GPU on two cards, not on one.

## Host submission: test and verdict (05:55 test written, 07:30 verdict)

Measured (job 111, interleaved, n=2 per arm, 68 tokens, drafted tok/s): one card 123.5 / 123.5 default vs
115.3 / 116.7 with direct submission off (**−6 %**); two cards 151.2 / 151.7 default vs 125.9 / 116.8 off
(**−17 to −23 %**); serial: one card 33.0 → 31.6 (−4 %), two cards 50.75 → 40.0 (−21 %). Two cards are about 4x more
sensitive to the cost of a submission. The exchange design is priced at about 1 % of a round (push vs read, row 2),
the link payload at about 1.5-2.5 ms, so host submission of twice as many commands from one thread is now the
leading explanation for the unexplained 8.6 ms.

What job 1049 (unitrace `--host-timing` only, 257 steps, two cards then one card) must show:

| Quantity (per drafted round, from call counts and API times) | Host-bound (confirms) | GPU-bound (rejects) |
|---|---|---|
| Σ append time (`zeCommandListAppendLaunchKernel` + `AppendMemoryCopy` + `AppendSignalEvent` + `AppendWaitOnEvents`) | close to the round wall on two cards | well below the round wall |
| `zeCommandListHostSynchronize` wait | near zero on two cards (the GPU is already idle when the host arrives) | large, roughly round wall − Σ append |
| commands per round, two cards / one card | about 2x | — |
| one card | GPU-bound (large HostSynchronize wait) in both cases | |

Caveat: the tracer adds time to every call, so Σ append is corrected with the untraced append cost from the probe
(2.3-2.5 µs per command, `raw/host-*.csv`) x the traced call counts.

Fix it would justify, in order of expected effect: (1) record each round shape once into regular command lists and
replay them (the `chain` replay arm removes host append entirely: 10.3 → 1.6 µs per step with no exchange; replay
needs per-round arguments such as positions and tree tables to live in device buffers); (2) one submit thread per
rank, which halves the serial host path but keeps per-command cost. Owner: `tptf`.

**Verdict (job 1049, 07:25, measured):** rejected as the main cause. Host-only tracing does not slow the run
(two cards 150.5 tok/s traced vs 151.4 untraced; one card 123.5 vs 123.5). Whole-run API totals, two cards / one card:
`zeCommandListAppendLaunchKernel` 504,745 / 214,940 calls at 2.6 µs (1.31 / 0.57 s), `zeKernelSetArgumentValue`
5.39 M / 2.56 M calls (0.39 / 0.16 s), `AppendSignalEvent` 79,722 (0.10 s) / 0, `zeEventHostReset` 158,928 at 56 ns
(0.009 s) / 0, `zeCommandListHostSynchronize` 1.92 / 1.80 s, `AppendMemoryCopy` 4.47 / 9.00 s (mostly blocking copies:
weight load, and token reads into host stack memory, which wait for the GPU). Host submission work is about 1.8 s of
about 7 s of active run on two cards (about 26 %) and 0.72 s of about 10.5 s on one card (about 7 %) (**estimate**:
active run = total minus about 4 s of load). The host spends the rest waiting for the GPU, so the round is
GPU-bound on average with direct submission on. Per TP2 forward: about 1,580 launches x (2.3 + 0.8) µs ≈ 5 ms of host
work against about 19.7 ms of step (**estimate**). Direct submission off adds about 5 µs per command (7.9 ms per
forward on two cards), which brings the host close to the GPU time: that is why two cards lose 17-23 % and one
card 6 %. Replay or a thread per rank would not recover the 8.6 ms today.

## Finding: unitrace device timing crashes TensorFold `--tp 2`

Command (job 112, `scripts/tf-trace.sh`): `unitrace --chrome-device-logging --device-timing --kernel-submission
--host-timing --chrome-call-logging -o <depth>-tp2.stats xpu-qwen27 Qwen3.8-27B-MLX-4bit code-<depth>.ids 97` with
`XPU_TP=2 XPU_ATTN_PREF=256 XPU_TREE=15 XPU_DRAFTER=… ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`.
Result: segfault (exit 139) after load at 68 tokens; `Level Zero : 0x70000001` (DEVICE_LOST) and "a command list did
not drain" at 32K (`raw/tf-trace/0-tp2.log`, `32k-tp2.log`, `32k-tp2.868115.stats`). The same command with `XPU_TP=1`
works at both depths. The same binary without unitrace completed 26 of 26 two-card runs tonight (jobs 110, 111).
So the crash needs the tool. Job 1049 (07:25): `--host-timing` only works on two cards (256 steps, drafted == serial);
`--device-timing` only segfaults on two cards again (`raw/tf-hosttrace/0-tp2-dev.log`, exit 139). The trigger is
unitrace's device-event instrumentation, not TensorFold alone. Likely mechanism
(**hypothesis**): unitrace attaches its own timestamp events to every append, including the cross-device waits on
TensorFold's shared events, and its event handling does not survive a second device waiting on them.

## Ranked map, 04:10 version (superseded by the final table above)

Costs are per committed output token at 68 prompt tokens unless the row says otherwise.
"Measured" = read from a run on this host. "Estimated" = arithmetic from measured parts. "Unpriced" = no number yet.

| # | Cause | Where it lives | Cost | Evidence | Proposed fix | Expected gain | Owner |
|---|---|---|---|---|---|---|---|
| 1 | **Hypothesis, demoted 03:40:** work that does not split (drafter, tree build, argmax and logit gather on rank 0). Serial decode scales 1.54x, the drafted round 1.25x, but the drafter is measured at ≤ 2-3 ms a round (H4), so the un-split rows explain at most about 3 ms of the 33.2 ms round. The rest of the gap sits in the 16-row verify forward (H5, H6). | Engine: `zig/src/xpu/drafter.zig` (launches on `dr.r.d`), `drafter_tree.zig:131-164`, `forward.zig:558-580` | ≤ 3 ms per round, measured bound (H4) + 0.30 ms argmax measured + about 0.35 ms gather estimated | Budget per part from job 112 | Split or overlap the drafter only if 112 shows more than the bound. | ≤ 1.5 ms per round (half the bound) | open |
| 2 | Every exchange is a cross-card event pair plus a peer **read**: each rank appends a signal, then a reduce kernel that waits on both events and reads the other rank's buffer over PCIe. 128 exchanges per forward. | Engine: `zig/src/xpu/tp.zig:126-148`; kernel `zig/kernels/xpu/tp.cl` `tp2_allreduce_f16` | **Measured** (chain, replayed lists, GPU cost per step above a 1.6 µs no-exchange step): +5.1 µs at 10 KiB, +8.1 at 40 KiB, +19.9 at 160 KiB; the event pair alone is +2.7 µs. Per forward: 0.65 ms serial (128 x 5.1), 2.5 ms per 16-row verify (128 x 19.9) (**estimate** from the measured step). | `raw/chain-summary.tsv` | Push instead of read: remote store + event (estimate 4 µs / 17 µs per step) or copy-engine push (estimate 13.7 µs at 160 KiB); overlap the payload with the next GEMM. | Store push: about 0.14 ms per serial token, 0.36 ms per round. Copy-engine push: about 0.8 ms per round (estimates). | `tptf` (push patch, `XPU_TP_PUSH_F16`) |
| 3 | `syncAll` host-resets `step*2*tp` events after every forward: 512 resets on TP2, half of them never signalled. | Engine: `zig/src/xpu/tp.zig:94-103` | **Measured**: 256 resets take 6.6-12 µs, so 512 resets cost 13-24 µs per forward, about 0.1 % of a round. **Not a lever.** | `raw/chain-summary.tsv` (`reset_us`) | none for speed | < 0.03 ms per round | closed |
| 4 | One host thread appends both ranks' commands in turn (`for (0..tp)` per layer), each append about 2.3-2.5 µs (direct submission on), about 6-8 µs with it off. About 1,800 appends per TP2 forward = about 4.3 ms of host time (**estimate**), hidden behind 15+ ms of GPU work while direct submission is on; with it off (12+ ms) it is not, which matches the 24 % loss. Kernels shorter than the per-rank append cadence (about 5 µs) starve. | Engine: `forward.zig:380-514`; runtime: L0 immediate in-order lists (`ze.zig:450`) | **Measured**: append 2.3 µs (host test), chain immediate arms host-bound at 2.5 µs per command; replay removes it (local step 10.3 → 1.6 µs). | `raw/host-*.csv`, `raw/chain-summary.tsv` | Record the forward once per round shape and replay it (regular list); or one submit thread per rank. | Unpriced end to end: needs the trace (job 112) to see how much GPU idle sits between short kernels. | `tptf` |
| 5 | Rank 0 gathers the other rank's logits for every verified row: 16 rows x 124,160 x 2 B = 3.97 MB per drafted round over PCIe, then a full-vocab argmax on rank 0. | Engine: `zig/src/xpu/forward.zig:558-580` | **Estimated** 0.33-0.36 ms per round (3.97 MB at the measured 11-12 GB/s peer copy); 1.1-1.2 % of a 30.9 ms TP2 round (165.5 tok/s x 5.12 tokens/round). | Argmax per rank on its vocab shard, exchange 16 x (value, index) pairs. | ~0.3 ms per round (estimate). | `tptf` |
| 6 | Shared events use signal/wait scope 1 = `ZE_EVENT_SCOPE_FLAG_SUBDEVICE` (the comment says DEVICE). A sub-device signal does not promise that the producer's writes are visible to another device. Output matched in every run so far: a latent-correctness question, not an observed failure. | Engine: `zig/src/xpu/ze.zig:362-366` | **Measured**: HOST scope and a device-only pool cost the same per step as SUBDEVICE (within 0.01 µs at 10, 40 and 160 KiB). | `raw/chain-summary.tsv` | Use HOST (or DEVICE) scope: free. | Correctness at no cost. | `tptf` |
| 7 | llama.cpp SYCL `-sm tensor` stages every allreduce through pinned host memory and ends each call with two host waits (`e_h2d_0.wait(); e_h2d_1.wait()`), so the GPU drains on every call. | Engine: `ggml/src/ggml-sycl/ggml-sycl.cpp:7855-7886` (tree d5bbcd1f3) | RL-250: ~275 µs per call, 3,584 calls, 0.86x decode (Nanbeige4.3-33B, older build). Qwen3.8-27B on d5bbcd1f3: unpriced until job 115. | Device-resident peer exchange without a host wait (`tpllama` is building it). | Unpriced. | `tpllama` |
| 8 | Link: each card has its own PCIe 4.0 x8 CPU root port. The cards are Gen5 x16 capable (switch upstream ports report "32GT/s x16, downgraded to 16GT/s x8"). | Platform (board slot wiring) | Line ceiling 15.75 GB/s per direction; measured peer copy 11-12 GB/s (earlier). For a 10 KiB exchange the payload time is 0.65-0.93 µs. | None in software. | Under 0.3 µs per 10 KiB leg even at the ideal line rate. | n/a |
| 9 | vLLM XPU TP2 | Engine/runtime (XCCL transport) | Earlier: blocked at load (RL-075 class). v0.31.0 with real int4 weights: job 120. | n/a | n/a | n/a |

## Measured 2026-11-19 03:49 CET: Level Zero primitives (job 104)

Native probe `scripts/peer.cpp` + `peer.cl`, runtime 26.05.37020.3, 25 retained repeats per cell after one discarded
warm-up, every payload checked (0 mismatches in all host/peer/allreduce cells). Raw: `$WORK/tpdiag/raw/{host,peer,allreduce}-{default,off,on}.csv`.
GPU times are kernel/copy timestamps (the `gpu*_us` columns are written in the wrong unit by the probe: true µs = column / 368,640, applied below).

| Primitive (direct submission = default, which is on) | Median | Notes |
|---|---:|---|
| Direct submission state | on by default | strace: 23 `DRM_IOCTL_XE_EXEC` per host test with default and forced on, 544 with it forced off |
| Kernel append, immediate in-order list | 2.3 µs | forced off: 5.8-7.9 µs |
| One kernel, append to host-visible completion | 8.4-8.6 µs | forced off: 46-59 µs |
| Regular list: append 1.05 + execute 2.3-2.5 µs, completion | 11.5-17.7 µs | forced off: 47-61 µs |
| Peer copy, copy engine, 64 MiB, either direction | 14.32 GB/s | 91 % of the 15.75 GB/s Gen4 x8 line rate; regular and immediate equal |
| Peer copy, copy engine, 10 KiB / 160 KiB (GPU time) | 0.57 / 11.0 µs | |
| Kernel peer stores, 64 MiB, 4-byte or 16-byte | 11.36 GB/s | store width makes no difference |
| Kernel peer stores, ≤ 10 KiB / 160 KiB (GPU time) | 1.25-1.35 / 14.4 µs | fixed cost about 1.3 µs |
| Two-rank allreduce 5-40 KiB, host submit to both done, copy engine, immediate | 18-19 µs | regular 22-29 µs; forced off 102-115 µs |
| Same, peer store + flag + poll, immediate | 25-29 µs | flag wait on the GPU 1.2-1.5 µs; forced off 80-94 µs |

Consequences: the link delivers 14.3 GB/s (better than the 11-12 GB/s on file), so a 160 KiB exchange has an 11-14 µs
payload floor; 128 of them are 1.4-1.8 ms per verify forward (**estimate** from measured parts). An isolated round trip
through the host costs 18-29 µs, which is why every design that waits on the host per exchange (llama.cpp) loses.

## Measured 2026-11-19 02:16-03:06 CET: TensorFold dev runner, one card vs two (job 110)

Binary `$TF/src-tp3586/zig-out/bin/xpu-qwen27` (sha256 `50da998b…`), `XPU_ATTN_PREF=256`, DFlash2 tree 15, 256 steps,
greedy. One card = card 1 (`ZE_AFFINITY_MASK=1`, 230 W, 2,800 MHz). Two cards = card 0 at 180 W / 2,400 MHz + card 1.
Interleaved runs, no profiler. Raw: `$WORK/tpdiag/raw/tf/*.log`.

| Prompt | Arm | n | Drafted tok/s | ms per drafted round | Serial tok/s | Serial ms/token |
|---|---|---:|---|---:|---|---:|
| 68 | one card | 3 | 123.4 / 123.5 / 123.6 | 41.4 | 33.0 (x3) | 30.3 |
| 68 | two cards | 3 | 150.8 / 151.6 / 151.8 | 33.2 | 50.7-50.8 | 19.7 |
| 68 | two cards, NEO direct submission forced on | 1 | 151.6 | 33.1 | 50.9 | 19.6 |
| 68 | two cards, NEO direct submission forced off | 1 | 115.7 | 43.4 | 38.1 | 26.2 |
| 32,552 | one card | 2 | 77.5 / 77.6 | 60.5 | 28.0 | 35.7 |
| 32,552 | two cards | 2 | 94.7 / 94.8 | 47.6 | 44.6-44.7 | 22.4 |

Read-out (measured unless marked):

1. Serial decode scales 1.54x (68) and 1.59x (32K); the drafted round scales only 1.25x (68) and 1.27x (32K).
   The drafted path is the problem, not the split itself.
2. Direct submission (NEO ULLS) is already on by default (default == forced on). Forcing it off costs the two-card run
   24 % (151.4 → 115.7 tok/s) and serial 25 % (50.8 → 38.1). Each submission costs host time that the cards wait for.
   One-card sensitivity to the same switch is queued (job 111) to tell whether TP2 is more host-bound than one card.
3. One card is GPU-bound in serial decode: the kernel sum is 30.1 ms per token (profiled run, rank 0) against 30.3 ms
   wall unprofiled. Two cards take 19.7 ms per token against an ideal of about 15.1 ms (half the kernel time,
   **estimate**): about 4.6 ms per token over 128 exchanges, about 36 µs per exchange if the exchanges carried all of
   it (**estimate**, to be split by jobs 102 and 112).
4. One card, drafted rounds: kernel sum 39.5 ms per round (profiled) against 41.4 ms wall unprofiled, so about 2 ms
   per round is host or sync time on one card (**estimate**: the profiler itself adds about 2.5 ms per round).
5. `XPU_PROFILE=1` hangs on two cards (both two-card profile runs timed out at 240 s after the prompt). Cause in
   source: rank 1's launches take events from rank 0's single-device timestamped pool (`dev.zig:178-186`,
   `qwen27.zig:472-476`).
6. Two-card greedy text differs from one-card greedy text from about token 26, at both depths; drafted == serial
   holds within each configuration (6 of 6 two-card runs, 7 of 7 one-card runs). Acceptance differs slightly
   (5.02 vs 5.12 tokens per round at 68), so compare ms per round, not only tok/s.
7. The dev runner gives 151.4 tok/s on two cards at 68 tokens; the served path measured 165.5 earlier (different
   prompt framing and 512-token budget). The served ratio 1.29x and this 1.23x agree in direction.

## Source anchors (TensorFold `2286abc0c`, tree `$TF/src-tp3586`)

| Location | What it does |
|---|---|
| `zig/src/xpu/forward.zig:485-492`, `:508-514` | Mixer and MLP down GEMMs write into the rank's ring send buffer, then `allReduceF16`: 2 exchanges per layer, 128 per forward. Hidden 5,120, so 10 KiB per row; a 16-row tree verify sends 160 KiB per exchange. |
| `zig/src/xpu/tp.zig:126-148` | Per exchange and rank: `zeCommandListAppendSignalEvent(ready[s][r])`, then `tp2_allreduce_f16` waits on both ranks' ready events and reads `send_buf[0]` and `send_buf[1]` (one local, one remote). |
| `zig/src/xpu/tp.zig:94-103` | `syncAll`: host-synchronize both lists, then `zeEventHostReset` for `step*2*tp` events. |
| `zig/src/xpu/ze.zig:358-367` | Shared event pool: `HOST_VISIBLE`, events with signal = wait = 1 (SUBDEVICE). |
| `zig/src/xpu/forward.zig:558-584` | Head GEMM per vocab shard, signal, rank 0 waits, copies every row of the peer shard, argmax on rank 0, `syncAll`. |
| `zig/src/xpu/qwen27.zig:472-476`, `dev.zig:176-200` | `XPU_PROFILE=1` profiles with rank 0's event pool and attributes only kernels launched through `Dev.launch` with rank 0's kernel handles: the TP reduce kernels and every rank-1 kernel are not attributed. Use the unitrace job for both ranks. |

## Platform (read only, `evidence/platform.txt`, captured 2026-11-18 23:45)

| Item | Value | Can it limit peer traffic? |
|---|---|---|
| Link, CPU side | AMD root ports 00:03.1 and 00:03.2: 16 GT/s x8 each | Yes: 15.75 GB/s line ceiling per direction (the card can do 32 GT/s x16). |
| Link, GPU endpoint | 2.5 GT/s x1 reported by the G31 function and its internal switch port | No: internal segment of the card's own switch; peer copies measured 11-12 GB/s. |
| ReBAR | BAR 2 = 32 GB on both cards | No: the whole VRAM is mapped. |
| ACS | Root ports: ReqRedir+ CmpltRedir+ UpstreamFwd+ | Peer traffic goes through the root complex (it would anyway: two root ports). Latency share unpriced. |
| ASPM | Disabled on every link | No. |
| MaxPayload / MaxReadReq | 256 B / 512 B on all 8 functions | Small. Posted-write efficiency about 92 % of line rate before flow control. |
| IOMMU | `iommu=pt`, identity groups | No translation cost expected. |
| PCIe AtomicOps | GPU: no AtomicOp requester or completer | Rules out device-scope atomics across cards; the flag protocols use plain stores and local polls. |

## Earlier served measurements (cited, not re-run)

Client post-first-token rate, one stream, greedy, 512-token budget, prompt cache off. One card = GPU1 at 230 W.
TP2 = GPU0 180 W and 2,400 MHz plus GPU1 230 W. Source: `../../PROCESS.md` section 7.1.

| Prompt | One card ms/token | TP2 ms/token | Ratio | TP2 excess over an ideal 1.9x |
|---|---:|---:|---:|---:|
| 68 tokens, 5 pairs | 7.806 | 6.042 | 1.29x | 1.934 ms/token |
| 32,552 tokens, 5 pairs | 11.111 | 8.511 | 1.31x | 2.663 ms/token |
| 130,856 tokens, 2 per arm | 19.685 | 13.021 | 1.51x | 2.660 ms/token |

The ratio rises with context because attention KV is split by head (2 of 4 KV heads per card): from 32,552 to
130,856 tokens the per-token cost grew 8.574 ms on one card and 4.510 ms on two (1.90x). The fixed TP2 excess
(about 1.9-2.7 ms per token) is what rows 1-6 must explain.
