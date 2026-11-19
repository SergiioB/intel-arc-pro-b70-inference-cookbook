# B70 two-card tensor-parallel diagnostic (tpdiag)

Second `tpdiag` agent, taking over 2026-11-19 00:07 CET from the first one (out of quota after the plan, the platform
capture and eight unrun queue jobs). Ranked causes with owners: [TP-ROOT-CAUSES.md](TP-ROOT-CAUSES.md), section
"Final ranked table". Upstream drafts: pr-body-xpu-profile-tp2.md (lab file `pr-body-xpu-profile-tp2.md`, not published) (not posted),
neo-issue-draft.md (lab file `neo-issue-draft.md`, not published) (not to be filed), reproducer repro/ (lab file `repro/`, not published).

## 1. Result

On two Arc Pro B70 (PCIe 4.0 x8 each), TensorFold's `--tp 2` loses its time in the 16-row drafted round, not in the
link. One drafted round at 68 tokens takes 41.4 ms on one card and 33.1 ms on two (1.25x); serial decode scales 1.54x
(30.3 → 19.7 ms). A per-rank GPU profile (new, fixed `XPU_PROFILE`) closes the two-card round on card 0, the critical
path: 18.3 ms half of the one-card work, **4.4 ms kernels that do not halve** (gate/up GEMM kernel choice +1.8,
`gdn_step` +1.15, small kernels +1.45), **4.3 ms work on card 0 only** (drafter GEMMs and kernels, argmax),
**1.5 ms card 0 slower** (old cap 180 W / 2,400 MHz), 2.4 ms exchange kernels, 2.2 ms idle and host = 33.1 ms.
At 32K the drafter's attention `dattn` runs on card 0 only and costs 7.4-8.5 ms a round. The link is not the limit
(peer copy 14.3 GB/s, 91 % of line rate). NEO direct submission is on by default; switching it off costs two cards
17-23 % and one card 6 %. Runtime 26.40 is +1.6 % on two cards and +0.5 % on one (below the 2 % gate) with identical
text. `XPU_PROFILE` on two cards hung because of a compute-runtime 26.05 bug (fixed in 26.40); a standalone
reproducer proves it, and the TensorFold fix is on two pushed branches.

**Equal caps (operator, 09:40 CET: both cards 230 W / 2,800 MHz).** Dev runner, same binary, 68 tokens: two cards
158.9 / 158.3 tok/s at 31.65 ms a round (was 151.4 tok/s / 33.1 ms), serial 19.0 ms (was 19.7); one card unchanged
(123.6 / 123.5 tok/s, 41.4 ms). Two cards over one card: 1.31x per round, 1.59x serial. Budget: 18.3 half + 4.4 not
halving + 4.2 card 0 only + 2.4 exchange + 2.5 idle/host = 31.7 ms (measured 31.65). Equal caps were worth +4.8 % on a
two-card drafted round on the dev runner (158.6 vs 151.4 tok/s; 4.6 % less time a round). The served re-baseline at
equal caps (lead, `wave3/rebaseline-230w/tp2/cells.jsonl`, same `2286abc0c`, 512-token replies, 2 drafted runs a cell,
drafted == serial by sha256 in every cell) gives code prompts 173.3 / 173.7 tok/s at 68 tokens (was 165.5),
151.4 / 151.5 at 8K (was 143.0), 123.8 / 123.7 at 32K (was 117.5) and 81.6 / 81.3 at 128K (was 76.8 at 131K).

## 2. What you need

| Item | Value |
|---|---|
| Host | b70-host, Ryzen 7 5700X3D (Zen 3, no AVX-512), 60 GiB RAM, Ubuntu 26.04, kernel 7.0.0-28 |
| GPUs | 2x Arc Pro B70 32 GB: index 0 = `0000:0b:00.0`, index 1 = `0000:0f:00.0` (drives the desktop). Caps until 09:40 CET: card 0 180 W / 2,400 MHz, card 1 230 W / 2,800 MHz; from 09:40 both 230 W / 2,800 MHz (operator) |
| Runtime | compute-runtime 26.05.37020.3 (`driverVersion` 17010844), IGC 2.28.4, Level Zero loader 1.28.2, ocloc 26.05.1. Side by side (not installed): 26.40.40196.8 + IGC core 2.42.6 + gmmlib 22.10.2 under `$WORK/tpdiag/neo-26.40/` (sha256 in `deb-sha256.txt`; remove with `rm -r` of that folder) |
| TensorFold | fork head `15adb58cf` (branch `xpu-tp2-fixes-20261119`) and #3586 head `28ac4011c`; measured binaries: `src-tp3586` `2286abc0c` (jobs 110-112, 1049), `tpdiag/tf-prof` `62ab36f38` (2191, 2195), `tf-prof2` `98f081484` and `tf-3586-fix` `d02d8758f` (2193); binary sha256 in `$WORK/tpdiag/tfbuild-manifest.txt` |
| Models | `$TF/models/Qwen3.8-27B-MLX-4bit`, `$TF/models/Qwen3.8-27B-DFlash2`; prompts `ids/code-0.ids` (68 tokens), `ids/code-32k.ids` (32,552) |
| Tools | unitrace (`$ENGINES/pti-gpu-build/install/bin/unitrace`), gdb, strace |

## 3. Timeline (CET, 2026-11-19)

| Time | Step | Outcome |
|---|---|---|
| 00:07-00:40 | Read briefs, all 8 job files, every script | Found: lists not in-order (payload check could race), no arm reproduced TensorFold's exchange, default DS mode missing, cutoff 03:30 too early, `XPU_PROFILE` rank-0 only |
| 00:40-01:00 | Rewrote `peer.cpp` (in-order, `chain`), `run-peer.sh`, `tf.sh`, `tf-trace.sh`, `llama.sh`, `vllm.sh` | Syntax compile + 2 s `ocloc` of `peer.cl` outside the queue (disclosed) |
| 01:00-02:15 | Queue held by a 2 h battery and jobs 013-033 | 0 GPU time |
| 02:15 | 100 build ok; 103 preflight failed (xpu-smi discovery, known) | — |
| 02:18 | Found `nightsched.py` matched `after` against file stems; lead fixed it | — |
| 02:16-03:06 | 110 TensorFold pairs | 14/16 ok; both two-card `XPU_PROFILE` runs hung; 40 min lost to a 55 °C gate on the display card (lead changed `cool()`) |
| 03:18-03:40 | Lead dropped 115 (llama); I corrected my drafter estimate | row demoted |
| 03:47-03:53 | 101/102 failed (`zeDeviceGet` count bug), rerun 104; chain failed (event reuse), rerun 106 | primitives and chain measured, 0 errors |
| 05:33-05:35 | 111 DS screen; 112 unitrace | DS off −6 % / −17-23 %; unitrace device timing crashes two cards |
| 07:23-07:25 | 1049 host-only trace + device-only crash isolation | host submission rejected as the main cause; crash is tool-triggered |
| 07:28-07:40 | Lead: write the per-rank profiler fix. Worktree, commit `62ab36f38`, cherry-pick on #3586 | build 216 ok |
| 07:52-08:10 | 217/218: fix still hangs (#3586 unmodified too); 218 stopped early (my `cool` under `set -e`) | — |
| 08:12-08:30 | vLLM 215 failed (by-path mount, AVX-512 trap in AutoRound lib); 219 gdb stacks + key variants | hang = implicit counter-based events; key off completes |
| 08:30-08:46 | 2191 per-rank profile with key off | budget closed (33.1 ms) |
| 08:46-08:48 | 2193 fix build + verify on both branches | profiled two-card runs complete with no keys |
| 08:50-09:11 | 2192 reproducer stopped after one variant (my `set -e` bug), 2194 rerun | standalone hang on 26.05 two devices, passes on 26.40 / one device / key off |
| 09:30 | Pushed `tpdiag-xpu-profile-tp2`, `tpdiag-xpu-profile-tp2-3586` to the fork (lead-approved) | no PR |
| 09:20-09:43 | 2195 runtime A/B (27 runs) | +1.6 % two cards, +0.5 % one card, identical text |
| 10:17-10:20 | `2195-c` equal-cap profile (after the lead's cap change and re-baseline) | card-0 row 1.5 → 0 ms; round 31.65 ms |

Time lost: about 2 h queue wait (not tpdiag's), about 40 min cool-down gate, about 30 min of queue round trips for
my own script bugs (primitive count bug, chain event reuse, `cool` and `rc` under `set -e`).

## 4. Step by step

All GPU work through the night queue; job files in `evidence/queue/`. `R=~/B70-DOCS/results/2026/11/b70-breakthrough-20261118/tp/tpdiag`,
`W=$WORK/tpdiag`.

1. Platform (read only): `bash $R/scripts/platform.sh > $R/evidence/platform.txt`.
2. Primitives: `bash $R/scripts/build-peer.sh`; `bash $R/scripts/run-peer.sh smoke host peer allreduce`; `bash $R/scripts/run-peer.sh chain`
   (each in three NEO modes: default, `EnableDirectSubmission=0`, `=1` + overrides). Summaries: `scripts/summarize.py prim`
   (divide its GPU columns by 368,640), `scripts/chain_summary.py`.
3. TensorFold pairs and DS screen: `bash $R/scripts/tf.sh` (jobs 110, 111 with `TF_PLAN_0`).
4. Host-only trace: `bash $R/scripts/tf-hosttrace.sh` (unitrace `--host-timing`).
5. Profiler fix: worktree `$W/tf-prof` from `15adb58cf`, commits `62ab36f38` (per-rank pools, reduce-kernel hook,
   idle before each kernel) and `98f081484` (NEO key under `XPU_PROFILE`); same on `28ac4011c` (`ac9ddd5c0`, `d02d8758f`).
   Build: `bash $R/scripts/build-tf.sh TREE...` (records HEAD and sha256 in `$W/tfbuild-manifest.txt`).
6. Per-rank profile: `bash $R/scripts/prof.sh $W/tf-prof prof2 257 "<depth:arm ...>"` (refuses a binary whose sha256 is
   not the manifest's for HEAD); budget: `python3 $R/scripts/budget.py TP1_LOG TP2_LOG UNPROF_TP1_MS UNPROF_TP2_MS "drafted rounds"`.
7. Hang: `bash $R/scripts/prof-hang.sh` (gdb stacks), reproducer `bash $R/repro/run.sh`.
8. Runtime A/B: `prof.sh` arms `n40wall-N` / `n40profile-N` / `n40libs-N` (LD_LIBRARY_PATH to the 26.40 set).

## 5. Every fix

| # | Literal symptom | Cause | Fix | Verified by |
|---|---|---|---|---|
| 1 | (review) `check_payload` appends a kernel then a copy with no event | lists not in-order | in-order queue and list flags | 0 mismatches in every cell |
| 2 | (review) jobs would be killed at 03:30 | queue position | cutoff 04:00 → 06:00 → 08:30 → 14:00 (lead) | jobs ran to 10:xx |
| 3 | 101/102 never started | `nightsched.py` compared `after` names with file stems | reported; lead fixed at 02:18 | 104 waited for 103 |
| 4 | two-card `XPU_PROFILE`: `exit 124` after the prompt report | (a) rank 1 used rank 0's event pool; (b) runtime 26.05 implicit counter-based events with reused, host-reset events (gdb: spin under `forward.zig:583`) | per-rank pools + hook (`62ab36f38`); NEO key under `XPU_PROFILE` (`98f081484`); same on #3586 | 2193: profiled two-card runs complete, drafted == serial; standalone repro 2194 |
| 5 | `zeDeviceGet(... nullptr) failed: 0x78000005` | stale count (1) | `count = 0` | 104 ran |
| 6 | `zeCommandListAppendLaunchKernel ... 0x78000004` at rep 1 of `event_only_immediate` | reused host-reset events on in-order immediate lists (same runtime bug as #4) | sync lists, time the reset, rebuild the pool | 106: 0 errors |
| 7 | `gpu0_us` 460,800 for a 1-word kernel | `timerResolution` is in Hz for properties 1.2 | divide by 368,640 in analysis | 64 MiB copy 4,687 µs = 14.3 GB/s |
| 8 | 40 min of job 110 in cool-down | card 1 idles at 66 °C (display) | lead's `cool()` change | later jobs within bound |
| 9 | Drafter priced at 6 ms, then at ≤ 2-3 ms | first from weight size, then from wall minus attributed time | per-rank profile: about 4 ms GPU per round at 68 tokens | job 2191 |
| 10 | `0 2 exit 139` / `0x70000001` under unitrace on `--tp 2` | unitrace device-timing instrumentation | not fixed (tool); use the fixed `XPU_PROFILE` | job 1049 |
| 11 | job 218 `rc=124` after 7 runs | bare `cool` under `set -e` returned 124 | `cool` never fails, logs the bound | 2191 ran 15/15 |
| 12 | job 2192 ran 1 of 8 variants | `rc=$?` after a failing command under `set -e` | `rc=0; cmd || rc=$?` | 2194 ran 8/8 |
| 13 | job 2191 would refuse its binary after a new commit | stale-binary guard keyed on mtimes and HEAD | guard on manifest HEAD + sha256 + clean `zig/src` | 2191, 2193, 2195 ran |
| 14 | vLLM TP2 `onecclCommInitRank failed with code 11` | `/dev/dri/by-path` not mounted (oneCCL `ze_fd_manager.cpp:144`) | mount it (`vllm2.sh`) | VLLM_PLACEHOLDER |
| 15 | vLLM TP1 engine killed after "Loading weights took 48.67 seconds" | `invalid opcode` in `auto_round_kernel_xpu` (AVX-512F EVEX ops; Zen 3) (lead found the trap in the kernel log) | avoid AutoRound int4 here; dummy BF16 24-layer config | VLLM_PLACEHOLDER |

## 6. Verification

- Primitives: host 3 modes x 4 cells x 26; peer 3 x 13 sizes x 6 arms x 2 directions x 26; allreduce 3 x 4 x 6 x 26; chain 3 x 3 x 12 x 26. 0 errors.
  Raw `$W/raw/{host,peer,allreduce,chain}-*.{csv,log}`; failed attempts in `raw/failed-0347/`, `raw/chain-fail-0349/`, `raw/repro-attempt1/`.
- TensorFold: 110 (16 runs), 111 (8), 1049 (3), 2191 (15), 2193 (5), 2195 (27): drafted == serial in every completed run;
  serial token hashes identical across runtimes in all 12 cells x 3. Logs `raw/tf/`, `raw/prof/`, `raw/tf-hosttrace/` with `.telemetry`
  (caps, temperature, `max_freq` before and after each run).
- Reproducer: 2194, 8 variants (`raw/repro/*.log`). Hang stacks: `raw/prof-hang/`.
- Text copies in `evidence/summaries/`, `evidence/tf-logs/`; raw pointer `evidence/RAW-POINTER.txt`.

## 7. Results and claim boundary

| Quantity (68 tokens unless noted; caps 180/230 W unless noted) | One card | Two cards | n | Kind |
|---|---:|---:|---:|---|
| Drafted round / serial step | 41.4 / 30.3 ms | 33.1 / 19.7 ms | 8+ / 8+ | measured |
| 32,552 tokens: drafted round / serial step | 60.5 / 35.7 ms | 47.6 / 22.4 ms | 5 / 5 | measured |
| Two-card round budget (rank 0, old caps): half 18.3, not halving 4.4, card-0-only 4.3, card 0 slower 1.5, exchange 2.4, idle/host 2.2 | — | 33.1 ms | 1 profiled + controls | measured |
| Drafter attention `dattn` at 32K | 7.4 ms/round | 8.5 ms/round on rank 0 only | 19 rounds / 57 | measured |
| Direct submission off, drafted tok/s | −6 % | −17 to −23 % | 2 / 3 | measured |
| Runtime 26.40 vs 26.05, drafted | +0.5 % | +1.6 % (68 and 32K) | 3 pairs | measured, below the 2 % gate |
| Exchange per step (replayed lists) 10 / 40 / 160 KiB | — | +5.1 / +8.1 / +19.9 µs | 25 | measured |
| Peer copy / kernel peer stores, ≥ 1 MiB | — | 14.3 / 11.4 GB/s | 25 | measured |
| Host append per command (immediate list) | 2.3 µs | 2.6 µs traced | 25 | measured |
| Host event reset | — | 26-47 ns | 25 | measured |
| `XPU_PROFILE` cost | +8 % | +19 % (26.05), +13 % (26.40) | 2 | measured |

| Equal caps (230/230 W, 2,800/2,800 MHz) | One card | Two cards | n | Kind |
|---|---:|---:|---:|---|
| Drafted tok/s / round | 123.5-123.6 / 41.4 ms | 158.3-158.9 / 31.65 ms | 2 / 2 | measured |
| Serial step | 30.3 ms | 19.0 ms | 2 / 2 | measured |
| Rank 1 idle before exchanges | — | 0.61 ms/round (was 2.0) | 1 profiled | measured |
| Card 0 slower on the same work | — | 0.0 ms (was 1.5) | 1 profiled | measured |

Claim boundary: dev runner (`xpu-qwen27`), not the served path (served earlier: 128.1 → 165.5 tok/s); C1, greedy,
256 steps, no prompt cache. Speeds are the dev runner's wall rates over the decode or drafted span. Per-rank kernel
times come from profiled runs (profiling adds 8-19 %), walls from unprofiled runs of the same binary and job. The
budget row "card 0 slower" was measured with card 0 at 180 W / 2,400 MHz. Two-card text differs from one-card text
from about token 26 (drafted == serial holds within each). The 26.40 runs used a mixed library set (runtime 26.40,
IGC core 2.42.6, system IGC front end 2.28.4) and kernels compiled by ocloc 26.05.1: not the same as a real upgrade.

Not tested: llama.cpp `-sm tensor` (dropped by the lead; see `tp/tpllama/`), the served TensorFold path, `--tp 2` at
131K, peer atomics (not supported), power draw, four cards, a rebuild with ocloc 26.40, real-weight vLLM.

Who acts on what (lead, 10:00): `tptf` works on budget row 1 (gate/up GEMM kernel choice, `dec_wide64`-class kernel
for the 8,704 x 5,120 per-rank shape) and row 1b (splitting the drafter's `dattn` across ranks). Row 1c (card 0 cap)
was acted on by the operator (equal caps from 09:40).

## 8. If you do this again

1. Run the native probe with `smoke` alone first; check device enumeration and event reuse before long cells.
2. Use replayed (regular) command lists for GPU-side costs; immediate lists measure the host.
3. For a two-card per-kernel timeline use the fixed `XPU_PROFILE` (branch `tpdiag-xpu-profile-tp2`), not unitrace device timing.
4. On runtime 26.05, set `NEOReadDebugKeys=1 EnableImplicitConvertionToCounterBasedEvents=0` for anything that reuses host-reset events across two devices (no speed change measured).
5. In queue scripts, never let `cool` or a measured command end the script under `set -e`; guard binaries by sha256 in a build manifest.
Expected duration with a free queue: about 1 h 30 min of GPU (primitives 5 min, pairs 30 min, per-rank profile 20 min, A/B 25 min, reproducer 5 min).

## 9. Adapting it

For another model change the payload (hidden size x rows x 2 bytes) and the exchange count (2 per layer x layers) in
`chain` (`chain_slot`, `chain_steps`), and the kernel classes in `scripts/budget.py`. For another engine, profile one
rank against one card per kernel class; the rows that matter are always: work that does not halve, work on one rank
only, the slower card at each sync point, the exchange kernels, and host/idle.
