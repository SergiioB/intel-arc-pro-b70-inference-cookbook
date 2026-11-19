# TensorFold MLX 4-bit against exl3xpu EXL3 4.00 bpw, greedy, one session

Measured 2026-11-18 07:21 UTC to 2026-11-19 01:07 UTC on Intel Arc Pro B70 card 1 (230 W, 2,800 MHz), one card
unless a row says two. Status: **screen**. Raw cells and `summarize.py`:
[`results/qwen38-27b/two-cards-250k-routes-20261119/`](../../results/qwen38-27b/two-cards-250k-routes-20261119/README.md).

> **Scope.** Everything here is **greedy, thinking off**. The 2026-11-10 route decision
> ([ROUTE-COMPARISON-20261110.md](ROUTE-COMPARISON-20261110.md)) was made at the **model card's sampling**, and that
> comparison was **not re-measured**: its rerun was queued on 2026-11-19 and did not get GPU time (**pending**).
> Greedy and sampled rates are different metric classes and cannot be converted. The first-run route
> ([START.md](../../START.md), `data/routes.v1.json`) stays EXL3 + MTP3 until the sampled rerun reports.

> **Cap setting (added 2026-11-19).** One-card cells ran on card 1 at 230 W. The two-card rows ran with card 0 at
> 180 W / 2,400 MHz; at equal caps the two-card rates are 5–6 % higher ([RECIPES-B70.md](RECIPES-B70.md)).

## Result

| Workload | Faster here | Numbers (code prompt, decode tok/s after the first token) | Measured reason |
|---|---|---|---|
| One user, prompts up to 8K | **TensorFold, one card** | 127.8 vs 83.2 at 68 tokens; 114.6 vs 77.8 at 8K | Serial decode is level (32.4 vs 32.7). The DFlash2 tree gives 3.94× over serial, MTP3 2.54× |
| One user, one cold prompt of 32K–131K | **Decode: TensorFold. First token: exl3xpu** | Decode 90.0 vs 66.5 at 32K, 50.7 vs 48.2 at 131K. First token 21.5 vs 13.3 s at 32K (exl3xpu 1.6× faster), 128.9 vs 85.4 s at 131K (1.5×) | exl3xpu prefills 2,449 tok/s at 32K against 1,511. Its fp8 K/V is half the bytes of TensorFold's bf16, so its serial cost grows 12.2 ms a token from 68 to 131K tokens against 22.0 |
| 250K, one card | **exl3xpu** | 36.7 vs 20.7 drafted, 18.9 vs 15.4 serial; first token 239 / 228 s vs 315 s | TensorFold fits 250K on one card only with int8 K/V, and its int8 attention is slower (two cards: int8 32.5 vs bf16 62.3 at the same prompt) |
| 250K, two cards | **TensorFold `--tp 2`, bf16 K/V** | 62.3 drafted, 23.1 serial, 218.8 s to first token (180/230 W) | Two cards halve the K/V each card reads. exl3xpu has no measured two-card route here |
| Many streams, one card | **exl3xpu on throughput, TensorFold on repeatable text** | Three passes a level: 342.2 vs 270.9 tok/s aggregate at 8 streams, 333.3 vs 297.7 at 16, 374.8 vs 297.5 at 32; TensorFold leads at 4 (242.4 vs 220.0) | Cause not measured. Every TensorFold stream equals its solo text (183 of 183); exl3xpu 49 of 183 |
| Many streams, both cards free | **level** | TensorFold `--tp 2 --parallel 8`: 329.2 and 335.4 at 8 streams (two passes, 180/230 W), 0.47 J a token, against exl3xpu's one-card 342.2 at 0.67 J | Same aggregate at about 30 % less energy a token. Three-pass two-card and two-server arms are pending |
| Same answer for the same request | **TensorFold** | Drafted == serial on 8 of 8 prompts; two 329-item runs on two cards (180/230 W) gave the same text on 329 of 329 | exl3xpu MTP3 text equals a serial text on 2 of 16 requests; with MTP off, 1 of 8 prompts gave one text on every repeat |
| Answers right (329 items, greedy) | **level** | exl3xpu MTP3 286 (one run); TensorFold 281 (two cards, two runs), 282 and 283 (one card, one run each) | A 2–5 item spread on one run each does not separate the routes |

## What you need

- One Intel Arc Pro B70 32 GB (card 1 here, 230 W, 2,800 MHz). Two for the two-card rows, which ran with card 0 at
  180 W / 2,400 MHz before 2026-11-19 09:45 CET (both cards are 230 W / 2,800 MHz since).
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
| TensorFold two cards (for scale, 180/230 W) | bf16 | 62.31 | 23.07 | 218.8 s | 3.56 | yes |

More at 250K, including retrieval and warm follow-ups: [LONG-CONTEXT-250K-20261119.md](LONG-CONTEXT-250K-20261119.md).

## Several streams

Short prompts (the 68-token code and 52-token prose prompts, alternating, each with a unique first line so no two
streams share a prefix), 512 tokens a stream, all streams sent at once, card 1 at 230 W. Every stream's text is
compared with the same prompt run alone on the same server.

**Three passes a level (2026-11-19 07:20–08:07 UTC).** TensorFold `--parallel 32 --context 4096`; exl3xpu MTP3
`max_num_seqs` 32, `max_model_len` 16384. Median over the three passes, range in brackets.

| Streams | TensorFold aggregate tok/s | exl3xpu aggregate tok/s | Per-stream median, TF / exl3xpu | Slowest first token, TF / exl3xpu | J a token, TF / exl3xpu | Text equal to solo, TF / exl3xpu |
|---|---|---|---|---|---|---|
| 1 | 130.1 (130.0–130.1) | 82.5 (81.5–84.8) | 133.9 / 84.5 | 0.12 / 0.18 s | 1.69 / 2.79 | 3 of 3 / 2 of 3 (the third pass differed) |
| 4 | **242.4** (240.7–242.5) | 220.0 (118.4–220.4) | 66.9 / 63.3 | 0.46 / 7.88 s | 0.89 / 1.05 | 12 of 12 / 5 of 12 |
| 8 | 270.9 (268.5–272.7) | **342.2** (334.7–350.7) | 41.4 / 48.1 | 0.91 / 0.41 s | 0.81 / 0.67 | 24 of 24 / 8 of 24 |
| 16 | 297.7 (295.0–297.8) | **333.3** (318.6–334.3) | 19.2 / 32.2 | 1.82 / 16.48 s | 0.74 / 0.69 | 48 of 48 / 7 of 48 |
| 32 | 297.5 (297.4–297.5) | **374.8** (373.1–375.9) | 19.1 / 31.4 | 29.12 / 33.59 s | 0.74 / 0.61 | 96 of 96 / 27 of 96 |
| All | | | | | | **183 of 183** / 49 of 183 |

- From 8 streams up exl3xpu gives 1.12–1.26× TensorFold's aggregate on one card; at 4 streams TensorFold leads.
  The cause is not measured.
- TensorFold's aggregate is flat from 16 to 32 streams (297.7 and 297.5); at 32 the median first token waits 14.6 s.
- exl3xpu's 4-stream level had one slow pass (118.4) with a 7.9 s first token, unexplained. Its slowest first token
  at 16 streams was 16.5 s.
- Two-card TensorFold and two-server arms with three passes: **pending** (2026-11-19).

Earlier one-pass and two-pass screens (2026-11-18/19), kept for the two-card row:

| Route | 1 | 4 | 8 | 16 | J a token at 8 |
|---|---|---|---|---|---|
| TensorFold one card `--parallel 8` (one pass) | 131.4 | 228.9 | 279.8 | – | 0.78 |
| TensorFold one card `--parallel 16 --context 8192` (two passes) | 130.8 / 131.6 | 222.4 / 231.3 | 269.3 / 278.4 | 294.2 / 297.2 | 0.79 / 0.78 |
| TensorFold two cards `--tp 2 --parallel 8 --context 16384` (two passes, 180/230 W) | – | – | 329.2 / 335.4 | – | 0.47 / 0.47 |
| exl3xpu MTP3 one card, `max_num_seqs` 16 (one pass) | 82.5 | 119.6 | 337.2 | 320.9 | 0.68 |

At 4 streams (180/230 W), two independent one-card TensorFold servers (two streams each) gave 160.99 / 159.15 tok/s
against 129.27 / 130.23 for one two-card server ([two-card page](TENSORFOLD-TWO-CARDS-20261119.md)). For several users,
one server a card stays the default.

## Same text for the same request

| Check | TensorFold | exl3xpu |
|---|---|---|
| Drafted text equals serial text, 8 prompts | 8 of 8 | MTP3: 2 of 16 requests equal a serial text |
| Two drafted repeats of one prompt | same text | two texts on 7 of 8 prompts |
| Repeats with speculation off | same text | one text on every repeat for 1 of 8 prompts (3 repeats to 32K, 2 at 131K) |
| 250K, drafted against serial | equal | different |
| 329 items, run twice on two cards (01:46 and 22:08 UTC, 180/230 W) | 329 of 329 outputs identical | not run twice |

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

## Could EXL3 weights run inside TensorFold?

Not with the kernel that exists. TensorFold upstream has an `exl3_gemv` probe kernel behind `XPU_EXL3_PROBE`.
Layer-0 shapes, card 1 at 230 W, best of 189 int4 and 94 EXL3 calls, launch and sync included:

| Shape | Rows | TensorFold int4, ms (GB/s) | EXL3 probe 4-bit | 3-bit | 2-bit |
|---|---|---|---|---|---|
| gate+up 5,120 × 34,816 | 1 | 0.183 (488) | 5.037 (18) | 5.095 (13) | 5.105 (9) |
| gate+up | 8 | 0.185 (482) | 11.817 (8) | 11.823 (6) | 11.783 (4) |
| down 17,408 × 5,120 | 1 | 0.092 (483) | 2.858 (16) | 2.905 (12) | 2.886 (8) |
| down | 8 | 0.094 (473) | 5.454 (8) | 5.453 (6) | 6.561 (3) |

- Per weight the probe is **28.7× slower than the int4 kernel at one row and 62× at eight rows** (4-bit, weighted
  2/3 gate+up, 1/3 down). It decodes each weight with four sub-group shuffles, a multiply and a byte sum, uses no
  DPAS and skips the Hadamard transforms. This prices that kernel, not the EXL3 format.
- TensorFold's int4 kernel streams 532–549 GB/s when the group scales are counted, 87.5–90.3 % of 608 GB/s. A
  smaller format has to keep that decode speed to pay off.
- A format study ([`quantpath/QUANT-AND-EXL3-PATH.md`](../../results/qwen38-27b/two-cards-250k-routes-20261119/quantpath/QUANT-AND-EXL3-PATH.md))
  prices, as **estimates, not measurements**: EXL3 weights inside TensorFold at 4.0 bpw about level with today
  (116–130 drafted tok/s against 128); a 3.55 bpw format with today's decode path 148–152 drafted on one card
  (+16–19 %). Measured there: the MLX 4-bit codes carry 3.71 bits of entropy, so entropy coding saves at most 7 %.
- Quality of any sub-4-bit format on the 329-item battery: **pending** (2026-11-19). No sub-4-bit speed is claimed.

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
- Three passes of the two-card and two-server several-stream arms (pending, 2026-11-19).
- An EXL3 decode kernel on DPAS inside TensorFold; any sub-4-bit format's quality on the 329-item battery (pending).
- One-card and exl3xpu prompt-length sweeps with the two-card sweep script (**pending**, queued 2026-11-19).
- Prefix cache on for either route (off in every cell here).
