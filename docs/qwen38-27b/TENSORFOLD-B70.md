# Qwen3.8-27B on one B70 with TensorFold: lossless 123–128 tok/s, and what context does to it

Measured 2026-11-08 on two Intel Arc Pro B70 cards, one stream per card. Status: **screen**
(two repeats per cell, one host). Raw cells, prompts, scripts and patches:
[`results/qwen38-27b/tensorfold-depth-20261108/`](../../results/qwen38-27b/tensorfold-depth-20261108/).

![Decode tok/s vs prompt length for TensorFold, patched TensorFold, EXL3+MTP3 and EXL3 without speculation](../assets/qwen38-27b-depth-20261108.png)

## Pick a route

| Your prompts | Use | Why |
|---|---|---|
| Up to ~8K tokens (chat, short code tasks) | **TensorFold + DFlash2** | 123–128 tok/s on code, 98–102 on prose, and the drafted output is byte-identical to non-drafted greedy output |
| 32K and longer (agents with big context, repo dumps) | **EXL3 4.0 bpw + MTP3** (exl3xpu) | 74 tok/s at 32K and 48 at 131K, with 1.5–1.7× TensorFold's prefill, but MTP3 changes the output |
| 8K–131K, one stream, and an unofficial engine branch is acceptable | **TensorFold + `XPU_ATTN_PREF=256`** ([2026-11-13 screen](#prefix-attention-kernel-2026-11-13-20-at-32k-34-at-131k-262144-context)) | 115 / 90 / 51 tok/s at 8K / 32K / 131K on code, drafted output identical to non-drafted; the text differs from stock TensorFold on some prompts, and the branch is not in upstream |
| Long context and you need the model's own greedy answer | EXL3 without speculation, or TensorFold with drafts off | 23 tok/s at 131K (EXL3) vs 19 (patched TensorFold, serial) |

## Results (code prompts, 512-token answers, decode tok/s after the first token)

Card 1, 230 W cap, stock engines:

| Prompt tokens | TensorFold drafted | TensorFold serial | EXL3 + MTP3 | EXL3 serial |
|---|---|---|---|---|
| 68 | **128.3** | 32.0 | 82.8 | 32.6 |
| 7,976 | **85.1** | 30.4 | 79.4 | 31.5 |
| 32,552 | 37.3 | 23.4 | **73.8** | 29.4 |
| 130,856 | 11.4 | 12.8 | **48.2** | 23.3 |
| Prefill tok/s at 32K | 1,475 | | 2,455 | 2,526 |
| J/token at 68 tokens | **1.70** | 6.46 | 2.79 | 7.06 |

Chat-style prompts run 15–25 % slower on both speculative routes (lower draft acceptance) and
the same on serial ones. Full table with chat rows, power and TTFT:
[`summary-gpu1-230w.md`](../../results/qwen38-27b/tensorfold-depth-20261108/summary-gpu1-230w.md).

### Lossless means byte-identical

Same frozen prompts, greedy, thinking off, compared by sha256 of the answer text:

- **TensorFold:** drafted, drafted again and `"draft": false` gave identical text in 8/8 cells
  (and repeats were identical). This agreement is within the same build. The full
  two-change patch can differ from stock at depth because it changes reduction order.
- **EXL3:** MTP3 output differed from MTP-off output in 8/8 cells, and two MTP-off runs of the
  same prompt differed in 7/8 cells. Its speed is real; its output is not the model's
  reproducible greedy answer.

Exact-answer battery (6 canaries + 3 prompts at ~8K): TensorFold stock 9/9, patched 9/9.

## Why TensorFold slows down with context, and a patch (+49 % at 128K)

Draft acceptance is flat with depth (about 100 verify rounds per 512 tokens at every depth),
but each round costs 42 ms at 68 tokens, 138 ms at 32K and 420 ms at 131K. The decode
attention kernel re-reads the full K/V cache once per tree row (about 15 rows per round) and
writes fp32 partials for every 32-key slice: ~177 GB of traffic per round at 131K.

Our patch (two small changes, [`tree-attn-row-split.patch`](../../results/qwen38-27b/tensorfold-depth-20261108/patches/tree-attn-row-split.patch)):

1. Dispatch the tree rows of one key slice back to back, so they share it in L2.
2. Widen slices for long rows (32 keys below position 8K, 128 below 64K, 256 above), chosen by
   each row's own position, so drafted and serial decoding still use the same slices and
   stay bit-identical.

Same card (0), 230 W, stock v1.2.4 vs patched:

| Prompt tokens | code stock | code patched | chat stock | chat patched | card W stock → patched |
|---|---|---|---|---|---|
| 68 | 123.2 | 123.2 | 98.5 | 98.5 | 179 → 179 |
| 7,976 | 82.8 | **87.5 (+6 %)** | 66.0 | **70.2 (+6 %)** | 188 → 176 |
| 32,552 | 36.9 | **47.3 (+28 %)** | 27.3 | **35.0 (+28 %)** | 204 → 189 |
| 130,856 | 11.4 | **16.8 (+48 %)** | – | 15.0 | 214 → 194 |

J/token at 131K falls from 18.9 to 11.5. Patched serial decode at 131K is 19.0 tok/s (stock
12.8), which is faster than patched drafting (16.8): above ~64K, turn drafts off. A 180 W
repeat on the same card gave the same gains (+29 % / +49 %). A local-memory variant that
loads each slice once for all rows was 5× slower and is not included.

Upstream status, checked 2026-11-10: [PR #2595](https://github.com/ashhart/TensorFold/pull/2595)
contains the grid-order change with separate receipts against its named main revision.
It preserves stock output in those checks. The full local patch in this page also widens
slices and can change stock output; drafted-versus-serial agreement still holds within
that patched build on the measured cells. The two artifacts and their speed results
must stay separate. [Issue #2543](https://github.com/ashhart/TensorFold/issues/2543)
tracks the broader proposal. Apply the local patch only to the documented v1.2.4 base
when reproducing the tables below.

## Prefix attention kernel (2026-11-13): 2.0× at 32K, 3.4× at 131K, 262,144 context

Status: **screen** (one host; 2 repeats at 0–32K, 4 at 131K, 1 for the int8 cells). Raw cells, scripts, patch:
[`results/qwen38-27b/tensorfold-prefix-attn-20261113/`](../../results/qwen38-27b/tensorfold-prefix-attn-20261113/).
Source: [`SergiioB/TensorFold` tag `b70-attn-prefix-20261113`](https://github.com/SergiioB/TensorFold/tree/b70-attn-prefix-20261113)
(commit `4f7042f2d`, on top of upstream PR #2595). This supersedes the row-split patch above for long prompts; that
patch and its tables stay as measured on v1.2.4.

What it does: with `XPU_ATTN_PREF=256` a decode round reads the committed prefix once. A new kernel scores it in
chunks of 256 keys on bf16 DPAS for 8 (tree row, head) queries at a time; the stock scalar kernel keeps each row's
last keys; a merge joins both in key order. The cut depends on the row's own position, so a drafted row and the
same row decoded without drafts sum in the same order. Rows below position 1,040 run the stock kernels alone.

Card 1, 230 W, drafted, 512-token answers, decode tok/s (code / prose). Control = the same binary with the switch
unset, same session:

| Prompt tokens | Switch unset | `XPU_ATTN_PREF=256` | Ratio (code) | J/token (code) |
|---|---|---|---|---|
| 68 | 128.1 / 102.4 | 128.2 / 102.4 | 1.00 | 1.72 → 1.72 |
| 7,976 | 90.0 / 71.6 | **114.7 / 91.5** | 1.27 | 2.48 → 1.87 |
| 32,552 | 45.9 / 33.9 | **90.1 / 66.5** | 1.96 | 5.00 → 2.53 |
| 130,856 | 15.1 / 13.6 | **50.8 / 45.7** | 3.36 | 15.2 → 4.53 |

With `"draft": false`: 30.4 / 23.7 / 13.0 → 30.4 / 27.3 / 18.8 at 8K / 32K / 131K. Card 0 at 180 W: 85.1 → 108.9
at 8K and 42.4 → 84.6 at 32K (code). The control column is #2595's grid-order build, which is why it is above the
stock v1.2.4 table at the top of this page.

### What stays equal, and what does not

| Check | Result |
|---|---|
| Same text on a repeat | yes, every cell |
| Drafted == `"draft": false` under the switch | yes: every depth cell, and 275 of 275 battery items run both ways |
| Same text on the second card | yes, 329 of 329 battery items |
| Same text as the switch-unset build | **no**: 4 of 6 depth cells and 67 of 329 battery items differ |
| Switch unset | text equal to the stock build at every cell |

The kernel sums the same products in another order and keeps attention weights in bf16. A greedy answer takes
another path wherever two tokens were nearly tied; a 512-key chunk instead of 256 changes the text again in 3 of 4
cells. 329 graded items (HumanEval even / odd, knowledge, JSON schema, instruction following), switch unset
against the switch: 78 / 76 / 40 / 55 / 33 against 78 / 76 / 42 / 55 / 33; two items gained, none lost.

For this reason the change is **not in upstream TensorFold and is not proposed there**: upstream requires a speedup
to keep main's tokens, opt-ins included. Use it from the tag above if a changed-but-deterministic text is
acceptable for your work.

### int8 K/V under the switch, and 262,144 context

`XPU_KV_INT8=1 XPU_ATTN_PREF=256`, card 1, 230 W, one repeat, drafted, code / prose:

| Prompt tokens | int8, switch unset | int8 + switch | `"draft": false`, unset → switch |
|---|---|---|---|
| 7,976 | 88.8 / 74.3 | 102.6 / 85.0 | 30.4 → 28.0 |
| 32,552 | 43.6 / 35.0 | 65.2 / 53.5 | 24.0 → 26.4 |
| 130,856 | not run | 30.8 / 24.6 | – → 19.6 |
| 249,784 (`--context 262144`) | not run | 20.7 (code) | – → 15.3 |

- A 249,784-token prompt completes on one card: cold TTFT 330 s, 217 W, 10.5 J/token, drafted == `"draft": false`.
- The int8 kernel is slower than the bf16 one (131K: 30.8 against 50.8) because it widens every K and V byte to
  bf16. In this 8-bit mode the 2026-11-08 EXL3 + MTP3 cells (79.4 / 73.8 / 48.2, another session) are faster from
  about 20K up. For sessions whose length is not known at the start, the EXL3 route keeps its place.

### Not run

The 329-item battery with the position floor or with int8 K/V; int8 controls at 131K and 250K; repeats of the
int8 cells; retrieval probes at depth (the battery only uses short prompts); prompt-cache resumes under the
switch; concurrent streams (tree rounds of several streams keep the stock path); sampled decoding. The bf16 cells
were measured on two earlier builds of the same kernels (hashes in each run's `env.txt`); the tagged build was
checked against them at 0, 8K and 32K (`tag-pref256`).

```bash
git clone https://github.com/SergiioB/TensorFold && cd TensorFold && git checkout b70-attn-prefix-20261113
# build as below, then:
XPU_ATTN_PREF=256 ZE_AFFINITY_MASK=1 zig-out/xpu/bin/tensorfold-native serve Qwen3.8-27B-MLX-4bit \
  --drafter Qwen3.8-27B-DFlash2 --parallel 1 --context 139264 --no-thinking --no-update-check
# 262,144 context: add XPU_KV_INT8=1 and --context 262144
```

## Reproduce

```bash
# TensorFold v1.2.4, Intel Level Zero build (Zig 0.17.0, ocloc 26.05)
git clone https://github.com/ashhart/TensorFold && cd TensorFold && git checkout v1.2.4
git apply /path/to/local-patch.diff            # drops a 128-GRF flag ocloc 26.05 rejects (Gemma 4 kernels only)
git apply /path/to/tree-attn-row-split.patch   # optional: the long-context patch above
zig build xpu-native -Doptimize=ReleaseFast -Dngen-include=/path/to/onednn/third_party/ngen   # ~2.5 min

# weights: TensorFold/Qwen3.8-27B-MLX-4bit @22d8d538, drafter z-lab/Qwen3.8-27B-DFlash2
ZE_AFFINITY_MASK=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE zig-out/xpu/bin/tensorfold-native serve \
  Qwen3.8-27B-MLX-4bit --drafter Qwen3.8-27B-DFlash2 --parallel 1 --context 139264 \
  --prompt-cache-gib 0 --no-thinking --no-update-check --port 8036
# loads in ~7 s, 28.8 GB VRAM at 139K context
python3 depth_bench.py run --base http://127.0.0.1:8036 --model qwen --prompts prompts-qwen38.json \
  --out out --arm drafted --hwmon /sys/class/hwmon/hwmonN --depths 0,8k,32k,128k --reps 2
```

EXL3 rows: `exl3_depth_run.sh` (image `ghcr.io/sergiiob/exl3xpu-mixedk@sha256:933eda54…`,
turboderp 4.00 bpw weights, `max_model_len` 139264, prefix cache off), see [EXL3-XPU.md](EXL3-XPU.md).

## Claim boundary

- One stream, greedy, thinking off, 512-token answers, prefix cache off, two repeats per cell.
  Prompts are frozen text (llama.cpp source for code, this repo's docs for prose), sha256
  `74afe579…` for the whole set.
- Stock rows on card 1; the patch pair on card 0. At short context card 0 runs ~4 % below
  card 1 (123.2 vs 128.3), so compare within one table.
- The patched kernels ran on the bf16 KV cache, one stream. `XPU_KV_INT8=1` and concurrent
  streams were not measured. Quality was checked by the 9-case battery and byte identity,
  not by a task suite; an earlier 329-item screen put TensorFold within about one item of
  EXL3 4.0 bpw (MLX 4-bit g64 is a coarser format).
- Some card-0 cells ran while card 1 served another benchmark, and three low-priority
  (`nice 19`) engine compiles overlapped 11 timed cells (TensorFold serial repeat 1 at
  0–32K, two TensorFold cells around 128K, EXL3+MTP3 repeat 2 at 0–32K). Each of those cells
  matches its non-overlapped repeat within 1.3 %. Timeline:
  [`host-activity.txt`](../../results/qwen38-27b/tensorfold-depth-20261108/host-activity.txt).
