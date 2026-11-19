# Qwen3.8-27B at 250K context and in warm sessions on one and two B70s

Measured 2026-11-18 and 2026-11-19 on two Intel Arc Pro B70 cards with TensorFold and exl3xpu. Status:
**screen** (one repeat a cell unless stated). Raw files and `summarize.py`:
[`results/qwen38-27b/two-cards-250k-routes-20261119/`](../../results/qwen38-27b/two-cards-250k-routes-20261119/README.md).

> **Cap setting (added 2026-11-19).** Every two-card cell on this page ran with card 0 at 180 W / 2,400 MHz and card 1
> at 230 W / 2,800 MHz; one-card cells ran on card 1 at 230 W. At equal caps (both 230 W / 2,800 MHz) the served
> two-card rates rose 5–6 % at 0–131K with the same text; the 250K cells were not rerun yet (pending, 2026-11-19).
> Recipes at the current setting: [RECIPES-B70.md](RECIPES-B70.md#4-250k-context-on-two-cards).

## Result

1. **Fastest 250K setup measured: TensorFold on two cards with bf16 K/V** (180/230 W), 62.31 tok/s drafted at a 249,784-token
   prompt, first token in 218.8 s. On one card, exl3xpu is fastest (36.69 tok/s, 239.1 s); TensorFold needs int8
   K/V there and gives 20.67 tok/s.
2. **Warm 250K sessions work only on two cards with int8 K/V and our fork's fix.** Follow-up turns start in
   1.07–1.63 s instead of 205 s, with resumed text equal to fresh text on 4 of 4 turns. Every other warm 250K
   setup tried was refused at load or failed on the first follow-up.
3. **Retrieval holds:** 5 of 5 planted codes at 32K, 120K and 250K in all 9 cells (int8 and bf16, one and two
   cards).
4. **Warm follow-ups on two cards resume exactly** at 32K (5 of 5 turns, 0.30–0.49 s) and 131K (5 of 5, 0.65–0.83 s).
5. **int8 K/V is a capacity setting, not a speed setting.** At 250K on two cards it decodes at 0.52× of bf16
   (32.46 against 62.31 tok/s) and changes the words.

## What you need

- One or two Intel Arc Pro B70 32 GB. Measured with card 0 at 180 W and a 2,400 MHz clock cap and card 1 at 230 W,
  before 2026-11-19 09:45 CET. Set today: both cards 230 W / 2,800 MHz
  ([recipe 0](RECIPES-B70.md#0-host-both-cards-at-230-w-and-2800-mhz)).
- TensorFold [`tp2-attn-prefix`](https://github.com/SergiioB/TensorFold/tree/tp2-attn-prefix) (`2286abc0c`) for the
  measurements; [`xpu-tp2-fixes-20261119`](https://github.com/SergiioB/TensorFold/tree/xpu-tp2-fixes-20261119)
  (`ac346900b`, approved after three review rounds) for warm sessions past about 131K with int8 K/V.
  `XPU_ATTN_PREF=256` everywhere.
- exl3xpu `ghcr.io/sergiiob/exl3xpu-mixedk@sha256:933eda54…d3a`, EXL3 4.00 bpw, fp8 K/V, `max_model_len` 262144.
- The 249,784-token prompt is key `code-256k` of the lab file `prompts-deep.json` (sha256 `a3220647…`, 1.4 MB of
  C/C++ inference-engine source with a closing instruction; not published here, same file as the 2026-11-13 int8
  cells). The served tokenizer counts 249,784 tokens; the file's own estimate is 249,773.

## What fits on 32 GB cards at 262,144 context

| Setup | K/V | `--prompt-cache-gib` | Loads | Warm follow-up at 250K |
|---|---|---|---|---|
| TensorFold, one card | int8 | 0 | yes | no cache |
| TensorFold, one card | int8 | 4.1, 4.5, 8.5 | **refused**: "cannot hold a stream of 262144 tokens and a 4.1 GiB prompt cache" | – |
| TensorFold, two cards | bf16 | 0 | yes | no cache |
| TensorFold, two cards | bf16 | 16 | **refused** | – (smaller bf16 budgets not tried) |
| TensorFold, two cards, `2286abc0c` | int8 | 5 | yes | **fails**: three cache hits, then `InvalidQwenToken` after one token, empty reply (bug A) |
| TensorFold, two cards, fix branch (measured at `78f42f240`) | int8 | 5 | yes | **works**: 4 of 4 turns equal fresh, 1.07–1.63 s |
| exl3xpu, one card | fp8 | prefix cache off in these cells | yes, 9.58 GiB K/V reserved | not measured |

The two-card int8 server logged one kept state of 4,077 MiB at 249,784 tokens. On one card a 262,144-token int8
stream plus a 4.1 GiB cache does not fit beside the model and the drafter.

## Decode at 250K

249,784-token code prompt, greedy, 512 tokens out, prompt cache off, one repeat. Two-card rows at 180/230 W; one-card
rows on card 1 at 230 W. An earlier two-card bf16 run of the same prompt on 2026-11-18 (a different server start, n=1)
gave 63.29 drafted and 23.40 serial ([TENSORFOLD-B70](TENSORFOLD-B70.md#two-cards-for-one-stream-2026-11-18-13-to-32k-15-at-131k-250k-prompt-with-bf16-kv)).
The 230/230 W rerun is pending (2026-11-19).

| Setup | Drafted tok/s | Serial tok/s | First token | J a token, drafted | Text (drafted == serial) |
|---|---|---|---|---|---|
| TensorFold two cards, bf16 (180/230 W) | **62.31** | 23.07 | 218.8 s | **3.56** | `c0ba6e57…` (yes) |
| TensorFold two cards, int8 (180/230 W) | 32.46 | 22.84 | 206.9 s | 5.73 | `27aae358…` (yes) |
| exl3xpu one card, fp8, MTP3 / off | 36.69 | 18.85 | 239.1 / 227.9 s | 6.26 | differ |
| TensorFold one card, int8 | 20.67 | 15.37 | 314.9 s | 10.47 | `27aae358…` (yes) |

- Serial decode at 250K is the same on two cards with int8 and bf16 (22.84 against 23.07). The drafted gap comes
  from the int8 attention kernel, which widens every K and V byte to bf16 before use.
- int8 and bf16 give different words for the same prompt; each is exact within itself.
- On two cards with bf16 a 245,760-token prompt decoded at 39.3 tok/s in the prompt-length sweep (other text, 256
  tokens out); see [the sweep](TENSORFOLD-TWO-CARDS-20261119.md#decode-by-prompt-length-two-cards).

## Retrieval at depth

Five labelled 8-digit codes planted at 5 / 25 / 50 / 75 / 95 % of a code corpus, then one question for all five.
Greedy, thinking off, 65 tokens out. A hit is a line naming the label with the right 8 digits
(`kv256k/scripts/grade_needles.py`).

| Prompt tokens | One card int8 | Two cards int8 | Two cards bf16 |
|---|---|---|---|
| 32,209 | 5 of 5 (61.3 tok/s, 20.8 s) | 5 of 5 (81.0, 16.0 s) | 5 of 5 (108.5, 16.0 s) |
| 120,209 | 5 of 5 (29.2, 108.0 s) | 5 of 5 (45.2, 75.8 s) | 5 of 5 (82.6, 79.0 s) |
| 249,709 | 5 of 5 (15.3, 315.3 s) | 5 of 5 (20.6, 212.6 s) | 5 of 5 (48.4, 218.8 s) |

All nine answers are the same text (`913397660712…`). Five needles in one prompt test retrieval, not reasoning
over the document.

## Warm follow-up turns on two cards

Seed turn at the depth below (512 tokens out), then follow-ups of 128 tokens (`scripts/gates.py session`), 180/230 W. The
fresh check replays each turn's exact messages to a cache-off server and compares the text.

| Session | Server | Follow-up first token | Resumed == fresh |
|---|---|---|---|
| 32,552 tokens, bf16 | `--tp 2 --context 40960 --prompt-cache-gib 3` | 0.49 / 0.30 / 0.34 / 0.36 s | 5 of 5 turns |
| 130,856 tokens, bf16 | `--tp 2 --context 139264 --prompt-cache-gib 9.5` | 0.83 / 0.65 / 0.66 / 0.73 s | 5 of 5 turns |
| 130,856 tokens, int8 | `XPU_KV_INT8=1 --tp 2 --context 139264 --prompt-cache-gib 3` | 0.78 / 0.64 / 0.62 s | not checked |
| 249,784 tokens, int8, fix branch | `XPU_KV_INT8=1 --tp 2 --context 262144 --prompt-cache-gib 5` | 1.63 / 1.07 / 1.11 s | 4 of 4 turns |

Without the cache the same follow-ups take 16.3–16.8 s at 32K, 88.8–90.7 s at 131K and 205.1–205.5 s at 250K.
Size the cache to hold one state: about 64 KiB a token plus 0.2 GiB with bf16 K/V
([how](TENSORFOLD-B70.md#warm-follow-up-turns-size-the-prompt-cache)).

```bash
# warm 250K on two cards (fix branch)
XPU_ATTN_PREF=256 XPU_KV_INT8=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE zig-out/xpu/bin/tensorfold-native serve \
  Qwen3.8-27B-MLX-4bit --drafter Qwen3.8-27B-DFlash2 --tp 2 --parallel 1 --context 262144 \
  --prompt-cache-gib 5 --no-thinking --no-update-check --port 8046
```

## Other checks of the two-card route

| Check | Result |
|---|---|
| 329-item battery, run twice (01:46 and 22:08 UTC, 2026-11-18) | 281 both times, 329 of 329 outputs identical |
| Sampled, temperature 0.7, seed 7, 68 and 7,976 tokens | drafted == serial at both; two drafted repeats give the same text. 139.8 / 140.2 drafted against 46.7 serial at 68 tokens; 128.3 / 128.8 against 42.7 at 8K |
| Thinking on, 2,048 tokens | 105.8–109.4 tok/s drafted, 46.1 serial, but every reply was still thinking at the limit (8,266 and 5,412 reasoning characters, empty answer), so exactness with thinking is **not shown** |
| `--parallel 4`, 4 streams, before the bug-B fix | 129.3 / 130.2 tok/s aggregate drafted; 1 of 4 streams differed from its alone reply in both repeats; serial 102.6, all equal |
| Two independent one-card servers, same 4 streams | 161.0 / 159.2 tok/s drafted, 88.1 serial |
| 2-hour stability run | **pending** (queued, not started as of 2026-11-19) |

## Not tested

- Warm sessions at 250K on one card (they do not fit with int8 and a one-state cache).
- bf16 warm sessions on two cards with a cache budget between 9.5 and 16 GiB past 131K.
- Cache eviction at depth: the eviction test ran three ~32K conversations (18 of 18 requests equal to cache-off
  text, 5 evictions an arm, 0 failed saves), not 131K or 250K states. `--parallel` above 1 with the cache.
- exl3xpu warm sessions and retrieval at 250K; exl3xpu with prefix cache on at 250K.
- More than one repeat of any 250K cell; a retrieval task harder than five planted codes.
- Sampled decoding at depth; thinking on with a budget large enough to answer.
