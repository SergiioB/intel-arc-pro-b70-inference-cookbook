# Inkling 2 Mini on one Arc Pro B70: llama.cpp SYCL recipe (research, 2026-11-21)

> Status: research recipe. One card, one stream, measured 2026-11-21 with 3 seeds a cell. Output bit-exact with
> the same tree carrying only patch 0001 (the crash fix): KL 0, greedy 5/5. The engine without any local patch was
> not compared on these prompts. Decode is 29 tok/s at a short prompt, about 23 % of the computed byte ceiling. Work on a
> TensorFold XPU research build decodes faster but is not yet shown to give the same output
> ([TensorFold route](README.md#tensorfold-xpu-research-build-tag-b70-inkling2-mini-20261121)); numbers here will be
> superseded by a new generation, not edited.

Labels used on this page: **measured** (a raw file in
[`results/inkling2-mini/llamacpp-sycl-20261121/`](../../results/inkling2-mini/llamacpp-sycl-20261121/README.md)),
**computed** (arithmetic on measured or counted inputs), **copied** (from a named outside source, not reproduced here).

## 1. What you need

| Item | Value |
|---|---|
| Card | One Intel Arc Pro B70 (32 GB). Measured at 230 W cap and 2,800 MHz `max_freq` |
| Host (as measured) | Ubuntu 26.04, oneAPI 2026.1 (icx/icpx), compute-runtime 26.05.37020.3 |
| Model file | `unsloth/Inkling-2-Mini-GGUF` @ `b08906e3105be6a53c7dfdd4fd8e1e272eb01221`, `Inkling-2-Mini-UD-Q4_K_XL.gguf`, 17.63 GB, sha256 `e62d463c135ffe283df8d94b279a96ca145ac128b52ce854478613045bf4de15` |
| Engine | llama.cpp `master` `919794500` + PR #32695 (looped layers) + PR #32218 (DSpark), merged locally, + our patches 0001–0006 ([APPLY.md](../../patches/inkling2-mini-llamacpp/APPLY.md)) |
| Disk | about 18 GB for the file, about 2 GB for the build |

The file keeps the release's NVFP4 expert weights unchanged (layers 0–19); the 4 MXFP8 layers (20–23) are Q8_0;
attention and the LM head are Q8_0. It is the root-folder file of the unsloth repository (24 stored layers,
`inkling2.block_count = 48`). Upstream `master` after PR #32694 does not load it (DNR-038).

## 2. Build

```bash
git clone https://github.com/ggml-org/llama.cpp.git llama.cpp
benchmarks/inkling2-mini/prepare-llamacpp.sh llama.cpp llama.cpp-inkling2   # GPU-free; checks both tree hashes
benchmarks/inkling2-mini/build-llamacpp-sycl.sh llama.cpp-inkling2          # about 7 minutes
```

The build is SYCL, ahead-of-time for `bmg_g31`, with `GGML_SYCL_F16=ON` and `GGML_SYCL_GRAPH=ON`. The full cmake
line is in [`build-llamacpp-sycl.sh`](../../benchmarks/inkling2-mini/build-llamacpp-sycl.sh). The measured
binaries: `llama-server` sha256 `4f7e3142f02193aa63be63f37d967b8feb249c62c065f031c6dd6d775ae30dbc`,
`libggml-sycl.so` sha256 `a8a090ff42a9e1118bce45e319001468b7cba4bfdd8c2344506e49b32c782506`. Your binary hashes
will differ (paths and compiler state); the source tree hash must not.

## 3. Serve

```bash
curl -L -o Inkling-2-Mini-UD-Q4_K_XL.gguf \
  https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/b08906e3105be6a53c7dfdd4fd8e1e272eb01221/Inkling-2-Mini-UD-Q4_K_XL.gguf
echo "e62d463c135ffe283df8d94b279a96ca145ac128b52ce854478613045bf4de15  Inkling-2-Mini-UD-Q4_K_XL.gguf" | sha256sum -c

source /opt/intel/oneapi/setvars.sh --force
GGML_SYCL_ENABLE_GRAPH=1 GGML_SYCL_XMX_GATHER_TYPES=32767 SYCL_CACHE_PERSISTENT=0 ZE_AFFINITY_MASK=0 \
llama-server -m Inkling-2-Mini-UD-Q4_K_XL.gguf -ngl 99 -lm none -c 49152 -t 8 -np 1 --jinja --port 8050
```

The same line is in [`launch-inkling2-mini-llamacpp.sh`](../../benchmarks/inkling2-mini/launch-inkling2-mini-llamacpp.sh).
The measured cells also passed `--no-cache-prompt --cache-ram 0` so that every prompt was read cold; leave them off
for normal use.

| Setting | Why |
|---|---|
| `GGML_SYCL_ENABLE_GRAPH=1` | SYCL graph replay. Compiled in by `GGML_SYCL_GRAPH=ON` but off at run time unless set. Decode +34.1 / +26.6 / +19.8 % at 76 / 9,172 / 38,955 prompt tokens, greedy output identical (measured, §5) |
| `GGML_SYCL_XMX_GATHER_TYPES=32767` | Keeps patch 0002 off. 32767 sets bits 0–14 (every type the upstream switch already handles) and clears bit 15, the NVFP4 bit patch 0002 adds. With it, output equals the build with patch 0001 only (KL 0, greedy 5/5; §7). **Dropping it gives the lossy fast-prefill variant** (§6) |
| `SYCL_CACHE_PERSISTENT=0` | The persistent kernel cache crashes llama.cpp with this DPC++ release (DNR-031) |
| `ZE_AFFINITY_MASK=0` | One card. Use `1` for the second card; one server per card |
| `-ngl 99 -lm none -c 49152 -t 8 -np 1` | All layers on the card, no mmap, 49,152-token context, 8 CPU threads (1 thread measured the same, DNR-037), one slot |
| `--jinja` | The GGUF's chat template; `reasoning_effort` is a chat-template argument |
| no `-fa on` | No effect: the engine disables flash attention for this model's relative-position bias (DNR-036) |

Model-card sampling: `temperature 1.0`, `top_p 1.0`, `reasoning_effort 0.9` (the chat template's default). Pass the
effort per request as `"chat_template_kwargs": {"reasoning_effort": 0.9}`.

## 4. Measured: decode at model-card sampling

All cells: one card (card 0, 230 W / 2,800 MHz), one request at a time, cold prompt, code prompts, model-card
sampling, 3 seeds (4242–4244), one discarded warmup, median of 3 (range within 0.32 tok/s in every cell). Decode =
streamed rate after the first token. Energy = card counter over the decode phase. Card temperature at cell start
59–74 °C.

| Build | Graph replay | Reply budget | 76 tokens | 9,172 tokens | 38,955 tokens |
|---|---|---|---|---|---|
| 0001 only | off (control 1) | 2,048 | 18.98 | 17.71 | 13.92 |
| 0001 only | **on** | 2,048 | 25.43 | 22.37 | 16.66 |
| 0001 only | off (control 2) | 2,048 | 18.96 | 17.64 | 13.90 |
| 0001 only | on (control A) | 1,024 | 26.07 | 22.51 | not run |
| **0001–0006** | on | 1,024 | **29.09** | **25.14** | not re-measured |
| 0001 only | on (control B) | 1,024 | 26.13 | 22.46 | not run |

Decode tok/s, measured. Rows are in run order: each candidate ran between its two controls. Numbers from different
reply budgets are not comparable (a longer reply decodes deeper into context), so read each group on its own.

- **The recipe's cell: 29.09 tok/s at 76 prompt tokens and 25.14 at 9,172, 1,024-token replies.** The 38,955-token
  cell has not been re-measured with 0001–0006; the last value there is 16.66 tok/s (0001 only, graph on, 2,048 tokens).
- The 0001–0006 row ran with patch 0002 active. Patch 0002 only handles expert batches of more than one token
  (prompt reading), and decode sends one token, so the decode path is the same with 0002 masked (inferred from the
  patch source). Supporting check: greedy decode with 0002 masked ran at 26.9–31.0 tok/s on 46–256-token replies,
  against 28.3–30.2 with 0002 active (measured, `exact.jsonl`). The masked build was not run through the depth cells.
- Energy, decode: 6.20 / 7.98 J/token at 76 / 9,172 tokens (0001–0006) against 6.68 / 8.70 (controls, 0001 only,
  graph on) and 8.28 / 9.89 / 14.64 (graph off). Card power during decode 180 / 201 W (0001–0006).
- **None of these cells contains an answer.** At reasoning effort 0.9 the code prompt kept reasoning past the whole
  budget: 0/3 non-empty answers at 1,024 tokens and 0/3 at 2,048 in every cell. These cells measure decode speed,
  not time to a finished answer. Short tasks do answer: see §7.

## 5. Measured: what each change did

| Change | Same-session result | Output |
|---|---|---|
| Graph replay on (0001 build) | +34.1 / +26.6 / +19.8 % at 76 / 9,172 / 38,955 tokens vs the mean of the two controls | greedy sha256 5/5 identical to graph off |
| Patches 0003–0006 (0002 present, off the decode path) | +11.5 / +11.8 % at 76 / 9,172 tokens vs the mean of controls A and B | 0002 masked: greedy 5/5 identical to the reference |
| Patch 0001 | 38,955-token prompts serve; without it the server crashed at about 14K prompt tokens | same tokens (copy kernel) |

No single run measured the unpatched, graph-off build against the final recipe with the same reply budget, so this
page gives no combined speed-up.

## 6. Prompt reading and time to first token

Prompt reading was measured only for these builds (engine timing field, median of 3, cold):

| Build | 76 tokens | 9,172 tokens | 38,955 tokens |
|---|---|---|---|
| 0001 only, graph on (controls A/B, 1,024-token replies) | 65.6 tok/s, TTFT 1.36 s | 169.5 tok/s, TTFT 54.4 s | – |
| 0001 only, graph on (2,048-token replies) | 64.7 tok/s, TTFT 1.39 s | 169.1 tok/s, TTFT 54.5 s | 150.0 tok/s, TTFT 259.9 s |
| 0001–0006 with 0002 active = **lossy fast-prefill variant** | 157.6 tok/s, TTFT 0.67 s | 263.1 tok/s, TTFT 35.1 s | – |

**The exact default (0001–0006 with 0002 masked) has no measured prompt speed at server level.** Patches 0003 and
0006 also run during prompt reading, so the 0001-only rows are not a stand-in for it. At 76 tokens the rate is
mostly fixed per-request cost, not reading speed.

The lossy fast-prefill variant drops `GGML_SYCL_XMX_GATHER_TYPES=32767`. Its numbers change: mean KL 0.00112
(max 0.146) and top-1 agreement 98.49 % against the 0001 build (wikitext-2, 20 x 512 tokens), and greedy output
differs on 4 of 5 prompts. It fails the KL gate set before the run (mean KL ≤ 1e-4, top-1 ≥ 99.5 %), so it is
opt-in only (DNR-039). The cause of the size of the shift is not established (`unexplained`): the fused GEMM and the
old path both use fp16 operands in this build.

## 7. Quality (measured)

| Check | Result | Build |
|---|---|---|
| KL vs the 0001 build, wikitext-2 20 x 512 | mean KL 0.000000, max 0.000060, top-1 100.000 % | exact default (0002 masked) |
| Greedy output, 5 fixed prompts, up to 256 tokens | 5/5 sha256 identical to the 0001 build with graph off | exact default (0002 masked) |
| Perplexity, wikitext-2 test, 40 x 512 | 6.8087 ± 0.167 | 0001 only |
| Perplexity, same | 6.8036 ± 0.167 | 0001–0006, 0002 active (lossy variant) |
| `test-backend-ops` vs CPU | clip 9/9, NVFP4 `MUL_MAT_ID` 83/83, GQA-3 `MUL_MAT` 4/4, `SCALE` 9/9 pass | 0001–0006 |
| 9-item canary x 3, effort 0.9, model-card sampling | 27/27 correct and non-empty; median 72 reply tokens (max 320), median 2.8 s | 0001–0006, 0002 active |
| Same 9 items at effort 0.1 / 0.2 / 0.7 / 0.9 | 9/9 each; median 49 / 52 / 64 / 64 reply tokens | 0001–0006, 0002 active |

The perplexity of the exact default was not run separately; with KL 0 against the 0001 build its value follows from
that build's 6.8087 (inferred). The canary is a smoke test of short answers, not a quality ranking. The 0001 build
is not "unpatched": 0001 changes only the work-group size of a copy kernel, and greedy output of the unpatched
engine was not compared on the same prompts.

## 8. Memory (measured at `-c 49152`)

Process VRAM 25,810 MiB after load; 26,152 MiB after the 9,172-token cells; 27,068 MiB after the 38,955-token cells
(free visible VRAM 5,272 MiB). Host memory (process peak) 1,865 MiB. **Context fit (the largest context one card holds) is not yet
measured.** Not measured either: 128K or 1M-token context, retrieval at depth.

## 9. Why decode is at 23 % of the byte ceiling

Byte ceiling (computed): the file reads 4.17 GB a token at short context (attention family 1.43 GB, router, loop
controllers and convolutions 0.24, local-window KV 0.17, experts 1.85, LM head 0.49), plus 32 KB of global-layer KV
a context token. At 528 GB/s (measured device copy traffic on this card) that is **127 / 119 / 101 tok/s** at short
context / 8K / 32K. 29.09 tok/s is 23 % of 127; 25.14 is 21 % of 119. Bytes are not the limit.

What limits it instead:

- Kernel count (computed, ±20 %): about 7,700 GPU kernels a token. The model runs 48 layer applications a token
  (24 stored layers; layers 6–17 run three times), each with many small ops.
- CPU round trips (measured): before patch 0003 the server's graph had 194 scheduler splits; the two activation
  scales of every layer pass ran on the CPU, 96 round trips a token. With patch 0003: 2 splits. The number of
  "disabling SYCL graphs" log lines went from 180 to 0.
- Host submit cost: graph replay removes it, which is why it gave the largest single gain.

The kernel time table for the remaining gap is not measured: unitrace hung, and VTune's GPU modes refused this AMD
host. Fused decode kernels for this model run in the TensorFold XPU research build ([hub](README.md#tensorfold-xpu-research-build-tag-b70-inkling2-mini-20261121)).

## 10. Claim boundary

- One B70, one stream (C1), 230 W cap, 2,800 MHz, card 0. No concurrency, no second instance, no two-card run.
- Cold prompts: prompt cache and RAM cache off in every measured cell.
- 3 seeds a cell; this is not the cookbook's five-request publication matrix (BENCHMARK-FORMAT §11). Status in the
  catalog: provisional.
- 38,955-token decode not re-measured with patches 0001–0006. Exact default prompt speed not measured.
- Not tested: time to a finished answer on long tasks (the code prompt needs more than 2,048 reasoning tokens),
  32K and 128K retrieval, context fit, concurrency, DSpark with graph replay, tool calls, vision.

## 11. If you do this again

1. Download the file (about 18 GB), check its sha256.
2. Run `prepare-llamacpp.sh` and `build-llamacpp-sycl.sh` (about 10 minutes together).
3. Serve with the line in §3. First load about 16 s from a cold disk cache, 3 s warm; SYCL compiles kernels on the first request, so discard it.
4. Set `max_tokens` well above 2,048 for coding prompts at effort 0.9, or lower the effort; effort 0.1 still reasons.
