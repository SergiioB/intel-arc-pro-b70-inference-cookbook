# TensorFold MLX 4-bit against exl3xpu EXL3 4.00 bpw, greedy, one session

Measured 2026-11-18 07:21 UTC to 2026-11-19 01:07 UTC on Intel Arc Pro B70 card 1 (230 W, 2,800 MHz), one card
unless a row says two. Status: **screen**. Raw cells and `summarize.py`:
[`results/qwen38-27b/two-cards-250k-routes-20261119/`](../../results/qwen38-27b/two-cards-250k-routes-20261119/README.md).

> **Scope.** Everything here is **greedy, thinking off**. The 2026-11-10 route decision
> ([ROUTE-COMPARISON-20261110.md](ROUTE-COMPARISON-20261110.md)) was made at the **model card's sampling**, and that
> comparison was **not re-measured**: its rerun was queued on 2026-11-19 and did not get GPU time (**pending**).
> Greedy and sampled rates are different metric classes and cannot be converted. The first-run route
> ([START.md](../../START.md), `data/routes.v1.json`) stays EXL3 + MTP3 until the sampled rerun reports.

## Result

| Workload | Faster here | Numbers (code prompt, decode tok/s after the first token) | Measured reason |
|---|---|---|---|
| One user, prompts up to 8K | **TensorFold, one card** | 127.8 vs 83.2 at 68 tokens; 114.6 vs 77.8 at 8K | Serial decode is level (32.4 vs 32.7). The DFlash2 tree gives 3.94× over serial, MTP3 2.54× |
| One user, one cold prompt of 32K–131K | **Decode: TensorFold. First token: exl3xpu** | Decode 90.0 vs 66.5 at 32K, 50.7 vs 48.2 at 131K. First token 21.5 vs 13.3 s at 32K (exl3xpu 1.6× faster), 128.9 vs 85.4 s at 131K (1.5×) | exl3xpu prefills 2,449 tok/s at 32K against 1,511. Its fp8 K/V is half the bytes of TensorFold's bf16, so its serial cost grows 12.2 ms a token from 68 to 131K tokens against 22.0 |
| 250K, one card | **exl3xpu** | 36.7 vs 20.7 drafted, 18.9 vs 15.4 serial; first token 239 / 228 s vs 315 s | TensorFold fits 250K on one card only with int8 K/V, and its int8 attention is slower (two cards: int8 32.5 vs bf16 62.3 at the same prompt) |
| 250K, two cards | **TensorFold `--tp 2`, bf16 K/V** | 62.3 drafted, 23.1 serial, 218.8 s to first token | Two cards halve the K/V each card reads. exl3xpu has no measured two-card route here |
| Many streams, one card | **exl3xpu** | 337.2 vs 269.3–279.8 tok/s aggregate at 8 streams; 320.9 vs 294.2–297.2 at 16 | Not measured. exl3xpu ran one pass a level, and its 16-stream cell had a 16.4 s slowest first token |
| Many streams, both cards free | **level** | TensorFold `--tp 2 --parallel 8`: 329.2 and 335.4 at 8 streams, 0.47 J a token, against exl3xpu's one-card 337.2 at 0.68 J | Same aggregate at 31 % less energy a token. Two independent servers were not measured at 8 streams |
| Same answer for the same request | **TensorFold** | Drafted == serial on 8 of 8 prompts; two 329-item runs on two cards gave the same text on 329 of 329 | exl3xpu MTP3 text equals a serial text on 2 of 16 requests; with MTP off, 1 of 8 prompts gave one text on every repeat |
| Answers right (329 items, greedy) | **level** | exl3xpu MTP3 286 (one run); TensorFold 281 (two cards, two runs), 282 and 283 (one card, one run each) | A 2–5 item spread on one run each does not separate the routes |

## What you need

- One Intel Arc Pro B70 32 GB (card 1 here, 230 W, 2,800 MHz). Two for the two-card rows.
- TensorFold: our branch [`tp2-attn-prefix`](https://github.com/SergiioB/TensorFold/tree/tp2-attn-prefix)
  (`2286abc0c`, binary sha256 `e6e59bfd…5383`), `XPU_ATTN_PREF=256`, bf16 K/V unless a row says int8. Model
  `TensorFold/Qwen3.8-27B-MLX-4bit` + `z-lab/Qwen3.8-27B-DFlash2`.
- exl3xpu: image `ghcr.io/sergiiob/exl3xpu-mixedk@sha256:933eda54bb51a2e8a6669a70f5a2d7696fabd12a5f8b7c469df02fdfecc79d3a`,
  turboderp EXL3 4.00 bpw (6-bit head, 4-bit MTP), fp8 K/V, `max_num_seqs` 16, prefix cache off. Server start
  246–351 s.
- Benchmark script `depth_bench.py` (in [`tensorfold-depth-20261108/`](../../results/qwen38-27b/tensorfold-depth-20261108/depth_bench.py)),
  frozen prompts `prompts-qwen38.json` (sha256 `74afe579…`), 512 tokens out, one stream, no prompt cache.

## Decode, one stream, one card

| Prompt | TensorFold drafted, code / prose | exl3xpu MTP3, code / prose | TensorFold serial | exl3xpu MTP off | First token, TF / exl3xpu | J a token (code), TF / exl3xpu |
|---|---|---|---|---|---|---|
| 68 / 52 | 127.77 / 102.15 | 83.16 / 73.97 | 32.40 | 32.70 | 0.12 / 0.18 s | 1.72 / 2.78 |
| 7,976 / 7,963 | 114.56 / 91.05 | 77.78 / 71.45 | 30.72 | 31.63 | 4.63 / 3.45 s | 1.86 / 2.95 |
| 32,552 / 32,540 | 89.99 / 66.25 | 66.50 / 66.16 | 27.59 | 29.47 | 21.54 / 13.29 s | 2.53 / 3.45 |
| 130,856 / 130,846 | 50.73 / 45.60 | 48.16 / 42.51 | 18.93 | 23.38 | 128.85 / 85.42 s | 4.54 / 4.77 |

- Repeats: TensorFold one a cell; every text equals the same cell's text from six hours earlier. exl3xpu MTP3
  two a cell (median), MTP off three (two at 131K).
- Ratio, code: 1.54× at 68 tokens, 1.47× at 8K, 1.35× at 32K, 1.05× at 131K. Prose: 1.38×, 1.27×, 1.00×, 1.07×.
- A 512-token answer to one cold 131K prompt takes about 96 s on exl3xpu and 139 s on TensorFold one card
  (arithmetic from the cells: first token plus 512 tokens at the decode rate). On two cards TensorFold reaches the
  first token in 86.1 s.
- Serial decode reads the weights once a token. From the weight files: MLX 4-bit 14.42 GB a token (4.50 bpw),
  EXL3 13.18 GB (4.12 bpw). Effective: TensorFold 467 GB/s (77 % of 608), exl3xpu 431 GB/s (71 %). EXL3's 9 %
  byte saving goes to trellis decode compute, not to speed.

## 250K on one card

249,784-token code prompt, 512 tokens out, one repeat a cell.

| Route | K/V | Drafted tok/s | Serial tok/s | First token | J a token (drafted) | Drafted == serial |
|---|---|---|---|---|---|---|
| exl3xpu, `max_model_len` 262144 | fp8 | **36.69** | 18.85 | 239.1 s (MTP3), 227.9 s (off) | 6.26 | no (`c1d0a0ec…` vs `9d4294f2…`) |
| TensorFold, `--context 262144` | int8 | 20.67 | 15.37 | 314.9 s | 10.47 | yes |
| TensorFold two cards (for scale) | bf16 | 62.31 | 23.07 | 218.8 s | 3.56 | yes |

More at 250K, including retrieval and warm follow-ups: [LONG-CONTEXT-250K-20261119.md](LONG-CONTEXT-250K-20261119.md).

## Several streams

Short prompts (the 68-token code and 52-token prose prompts, alternating, each with a unique first line so no two
streams share a prefix), 512 tokens a stream, all streams sent at once.

| Route | 1 | 4 | 8 | 16 | J a token at 8 | Slowest first token at 8 |
|---|---|---|---|---|---|---|
| TensorFold one card `--parallel 8` (one pass) | 131.4 | 228.9 | 279.8 | – | 0.78 | 0.94 s |
| TensorFold one card `--parallel 16 --context 8192` (two passes) | 130.8 / 131.6 | 222.4 / 231.3 | 269.3 / 278.4 | 294.2 / 297.2 | 0.79 / 0.78 | 0.89 / 0.80 s |
| TensorFold two cards `--tp 2 --parallel 8 --context 16384` (two passes) | – | – | 329.2 / 335.4 | – | 0.47 / 0.47 | 1.06 / 0.68 s |
| exl3xpu MTP3 one card, `max_num_seqs` 16 (one pass) | 82.5 | 119.6 | **337.2** | 320.9 | 0.68 | 0.46 s |

- Median rate a stream at 8: exl3xpu 49.6, TensorFold one card 40.5 / 39.8, two cards 48.0 / 52.3.
- exl3xpu's 4-stream cell had one 7.8 s first token (unexplained) and its 16-stream cell a 16.4 s first token and a
  27–64 tok/s spread a stream. Treat its levels as one-pass screens.
- At 4 streams, two independent one-card TensorFold servers (two streams each) gave 160.99 / 159.15 tok/s against
  129.27 / 130.23 for one two-card server ([two-card page](TENSORFOLD-TWO-CARDS-20261119.md)). For several users,
  one server a card stays the default.

## Same text for the same request

| Check | TensorFold | exl3xpu |
|---|---|---|
| Drafted text equals serial text, 8 prompts | 8 of 8 | MTP3: 2 of 16 requests equal a serial text |
| Two drafted repeats of one prompt | same text | two texts on 7 of 8 prompts |
| Repeats with speculation off | same text | one text on every repeat for 1 of 8 prompts (3 repeats to 32K, 2 at 131K) |
| 250K, drafted against serial | equal | different |
| 329 items, run twice on two cards (01:46 and 22:08 UTC) | 329 of 329 outputs identical | not run twice |

TensorFold is exact within one configuration, not across them: at 250K int8 K/V gives one text (`27aae358…`) and
bf16 another (`c0ba6e57…`), each with drafted == serial. Nothing here shows that the EXL3 format causes exl3xpu's
variation; it belongs to that engine build.

## 329-item battery

Tune / test / knowledge / JSON / instructions (82 / 82 / 55 / 55 / 55 items), greedy, one run unless stated.

| Route | Passes | Run |
|---|---|---|
| exl3xpu MTP3, one card | 80 / 77 / 41 / 55 / 33 = **286** | 2026-11-19 00:28 CET |
| TensorFold two cards | 77 / 76 / 40 / 55 / 33 = **281**, twice, same text | 2026-11-18 01:46 and 22:08 UTC |
| TensorFold one card, prefill flag off (card 0, 180 W) | 78 / 76 / 40 / 55 / 33 = **282** | 2026-11-18 |
| TensorFold one card, `XPU_PREFILL_INT8=1` | 78 / 76 / 41 / 55 / 33 = **283** | 2026-11-18 |

TensorFold's one-card records from 2026-11-13 are 282 (stock) and 284 (prefix kernel)
([TENSORFOLD-B70.md](TENSORFOLD-B70.md#what-stays-equal-and-what-does-not)). With one run a route (exl3xpu does
not repeat its text), these totals do not rank the routes.

## Reproduce

```bash
# TensorFold, one card
XPU_ATTN_PREF=256 ZE_AFFINITY_MASK=1 zig-out/xpu/bin/tensorfold-native serve Qwen3.8-27B-MLX-4bit \
  --drafter Qwen3.8-27B-DFlash2 --parallel 1 --context 139264 --tp 1 --prompt-cache-gib 0 \
  --no-thinking --no-update-check --port 8046
python3 depth_bench.py run --base http://127.0.0.1:8046 --model qwen --prompts prompts-qwen38.json \
  --out tf1 --arm drafted --depths 0,8k,32k,128k --reps 1          # then --arm serial

# exl3xpu, one card (MTP3; add --set vllm.speculative_config=null for MTP off)
docker run -d --network=host --device=/dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro --shm-size=8g \
  -v /path/to/exl3-4.00bpw:/mnt/exl3model:ro -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  ghcr.io/sergiiob/exl3xpu-mixedk@sha256:933eda54bb51a2e8a6669a70f5a2d7696fabd12a5f8b7c469df02fdfecc79d3a \
  models/qwen3.8-27b-exl3-4.00bpw --gpu 1 --port 8051 --model-path /mnt/exl3model \
  --set vllm.gpu_memory_utilization=0.92 --set vllm.max_model_len=139264 --set vllm.enable_prefix_caching=false
```

Exact launch lines per arm: `routecmp/*/launch*.txt` and `env.*` in the evidence folder. Several streams:
`scripts/conc.py --levels 1,4,8,16`.

## Not tested

- The model card's sampling, thinking on (**pending**: the RL-281 rerun of the 2026-11-10 comparison).
- llama.cpp in this session (its fresh cells never reached a card; see the two-card page).
- exl3xpu on two cards; exl3xpu warm multi-turn at 250K; retrieval at depth on exl3xpu.
- A second pass of every exl3xpu several-stream level; TensorFold one card at 16 streams with `--parallel 8`.
- Two independent TensorFold servers at 8 streams.
- One-card and exl3xpu prompt-length sweeps with the two-card sweep script (**pending**, queued 2026-11-19).
- Prefix cache on for either route (off in every cell here).
