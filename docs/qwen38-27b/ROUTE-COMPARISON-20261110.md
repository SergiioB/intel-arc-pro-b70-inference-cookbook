# Qwen3.8-27B on one B70: which route to run, measured across the whole context

Measured 2026-11-10 on one Intel Arc Pro B70 (GPU 1, 180 W cap), one request at a time,
prefix cache on, at the model card's recommended sampling (not greedy). Status: **screen**.
Raw cells, prompts hashes, scripts and logs:
[`results/qwen38-27b/route-comparison-20261110/`](../../results/qwen38-27b/route-comparison-20261110/).

## Result

**Run EXL3 4.00 bpw with MTP3 (the launcher's default) and prefix cache on.** It is the
only route that serves the full 262,144-token context on one card, it found every needle at
32K, 128K and 240K, it is the fastest route from 32K up, and it scored highest on the
40-item battery. TensorFold is faster at 8K and below, slower from 32K (the crossover between
them was not measured), and cannot hold more than 131K tokens on one card.

```bash
b70 launch qwen38-27b -e exl3 --ctx 32768 --port 8100 --prefill fp16
```

Raise `--ctx` up to 262144 for long documents; the depth rows below ran at 262144.

Use the model card's sampling in your client:

| Mode | temperature | top_p | top_k | min_p | presence_penalty |
|---|---|---|---|---|---|
| Thinking (default) | 1.0 | 0.95 | 20 | 0 | 0 |
| Non-thinking | 0.7 | 0.80 | 20 | 0 | 1.5 |

Source: [Qwen3.8-27B model card, Best Practices](https://huggingface.co/Qwen/Qwen3.8-27B).

## Decode speed across the context

Code corpus prompts, thinking mode, 512-token answers, median of 3 (TensorFold 0 to 32K:
1 sample, the first run stopped there). Cold time to first token in brackets.

| Prompt | EXL3 + MTP3 | TensorFold + DFlash2 |
|---|---|---|
| 108 tokens | 48.6 tok/s (0.2 s) | **78.2** tok/s (0.2 s) |
| 8K | 46.3 (6.1 s) | **54.1** (5.6 s) |
| 32K | **40.4** (21.6 s) | 28.2 (25.3 s) |
| 64K | **34.8** (34.5 s) | 15.9 (58.8 s) |
| 128K | **28.9** (88.3 s) | not measured yet |
| 240K | **21.7** (235 s) | does not fit (131K limit) |

TensorFold refuses a 262,144-token stream with a 2 GiB prompt cache, and a 131,072-token
stream with a 4 GiB prompt cache, on one 32 GB card. It serves 131,072 tokens with 2 GiB.
llama.cpp across the whole context is still running; its short-context numbers are below.

## Short context, both modes

| Route | Thinking 0 / 8K | Non-thinking 0 / 8K | 40-item battery, 3 sampled runs |
|---|---|---|---|
| EXL3 + MTP3 | 49.3 / 50.3 | 59.6 / 61.8 | **35, 35, 36** |
| EXL3, MTP off | 30.2 / 29.4 | 30.3 / 29.2 | 33, 34, 35 |
| TensorFold + DFlash2 | **80.9 / 56.5** | **98.3 / 72.5**¹ | 33, 30, 34 |
| llama.cpp Q4_K_M, q8_0 KV | 26.4 / 25.3 | 26.0 / 25.0 | 33, 33, 30 |

¹ TensorFold returns HTTP 400 for `presence_penalty` above 0, so its non-thinking runs used 0.
The battery's pass/fail gates were written for greedy decoding; under sampling the counts are
descriptive. Three runs do not separate the routes statistically; EXL3 + MTP3 was highest in
every run.

## Long context: retrieval and chat

- **Retrieval at depth (EXL3 + MTP3):** three codes hidden at 10, 50 and 90 % of a 32K, 128K
  and 240K prompt; 3 of 3 found in every one of 9 runs. Repeats of the same prompt answered in
  2.6 to 6.7 s instead of 27 to 381 s because the prefix cache held them.
- **10-turn chat from an 8K document, cache on:** time to first token on follow-up turns,
  median: TensorFold 0.3 s, llama.cpp 0.4 s, EXL3 MTP off 0.8 s, EXL3 + MTP3 1.9 s. Decode
  stayed flat across turns on every route.

## Prefix cache is safe to keep on

Nine exact-answer questions sharing one ~8K prefix, greedy, every reply kept:

| Server | Correct | Degenerate |
|---|---|---|
| EXL3 MTP off, cache on (cold and warm pass) | 9/9, 9/9 | 0 |
| EXL3 MTP off, cache off | 9/9 | 0 |
| EXL3 + MTP3, cache on | 9/9 | 0 |

TensorFold's prompt cache gave byte-identical answers to cache off on all 11 turns of a greedy
session. EXL3 output is not repeat-stable (two cache-on runs of one session agree on 3 of 11
turns), so its check uses exact answers instead of hashes.

## Limits

- One card, one host, one power cap (180 W); 230 W would be faster on every route.
- Code-corpus prompts for depth; chat-style prompts may differ.
- The battery is 40 items; it cannot rank close routes.
- EXL3 MTP3 changes the greedy answer (see [TensorFold page](TENSORFOLD-B70.md)); under the
  card's sampling that is not a correctness question, and its sampled quality was the highest.
