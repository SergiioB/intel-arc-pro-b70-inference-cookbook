# Inkling 2 Mini, llama.cpp SYCL, one Arc Pro B70: raw evidence (2026-11-21)

Raw rows behind the [Inkling 2 Mini llama.cpp recipe](../../../docs/inkling2-mini/INKLING2-MINI-LLAMACPP.md).
`summary.json` is compiled from these rows only:

```bash
python3 benchmarks/inkling2-mini/summarize.py results/inkling2-mini/llamacpp-sycl-20261121 > summary.json
```

## Provenance

Copied on 2026-11-21 from the private lab repository at commit `0697ec705741362622bca28dc820b2a06b7242ca`
(the copied raw files are tracked there; the three `server-log-excerpt.txt` files are `grep` excerpts of server
logs that are not tracked there). Local paths were replaced: `$MODELS` = model folder, `$LLAMA_CPP` = llama.cpp
tree, `$WORK` = scratch folder, `$LAB` = lab repository, `$HOME` = home folder. Nothing else was edited.

## Common shape

- Hardware: one Intel Arc Pro B70 (32 GB), card 0, power cap 230 W, `max_freq` 2,800 MHz (both cards; `env.txt` of
  every run records the cap file and clock). Host: Ryzen 7 5700X3D, 60 GiB RAM, Ubuntu 26.04, oneAPI 2026.1,
  compute-runtime 26.05.37020.3.
- Model: `unsloth/Inkling-2-Mini-GGUF` @ `b08906e3105be6a53c7dfdd4fd8e1e272eb01221`, file
  `Inkling-2-Mini-UD-Q4_K_XL.gguf`, sha256 `e62d463c135ffe283df8d94b279a96ca145ac128b52ce854478613045bf4de15`
  (`sha256-model-files.txt`).
- Server: `llama-server -m <file> -ngl 99 -lm none -c 49152 -t 8 -np 1 --jinja --no-cache-prompt --cache-ram 0`,
  one request at a time (C1), cold prompt (no prompt cache).
- Depth cells (`depth/cells.jsonl`): code prompts of 76, 9,172 and 38,955 prompt tokens, model-card sampling
  (`temperature 1.0`, `top_p 1.0`, `reasoning_effort 0.9`; preset `inkling-2-mini:think`), seeds 4242–4244, one
  discarded warmup per server. `decode_tok_s` = streamed rate after the first token; `prompt_per_second` in
  `last_chunk_extra.timings` = the engine's own prompt timing; `ttft_s` = client time to first token;
  `decode_avg_w` and `decode_j_per_tok` = card energy counter over the decode phase.
- Greedy files (`exact.jsonl`): 5 fixed prompts, temperature 0 (`top_k` 1), up to 256 tokens, reasoning on;
  `sha256` of the full output.
- Memory files (`mem.jsonl`, in `T1-cand-*` and `cand1-graphon-*`): process VRAM, free visible VRAM and host memory
  before load, after load and after each step.

## Runs

| Folder | Build | Arm | What it shows |
|---|---|---|---|
| `inkling-base-20261121T020357Z` | merged base, no local patch | graph replay off, 512 tokens, 1 seed | first baseline; the 38,955-token cell crashed the server (the bug 0001 fixes); flash attention disabled by the engine (`server-log-excerpt.txt`) |
| `inkling-dspark-20261121T020936Z` | merged base, no local patch | DSpark drafter, `--spec-draft-n-max 3`, 512 tokens, 1 seed | DNR-034 |
| `ctl1-graphoff-20261121T024921Z`, `ctl2-graphoff-20261121T040547Z` | 0001 | graph replay off, 2,048 tokens | controls; `ctl1` greedy file is the exactness reference |
| `cand1-graphon-20261121T032935Z` | 0001 | `GGML_SYCL_ENABLE_GRAPH=1`, 2,048 tokens | graph replay candidate, between the two controls |
| `T1-ctlA-20261121T102722Z`, `T1-ctlB-20261121T104708Z` | 0001 | graph on, 1,024 tokens | controls for the patch stack |
| `T1-cand-20261121T103748Z` | 0001–0006, **0002 active** | graph on, 1,024 tokens | patch stack candidate (decode does not call 0002; its prompt speed and TTFT are the lossy variant's) |
| `T1-cand-nop2-20261121T105733Z` | 0001–0006, 0002 masked | greedy only | exact default: greedy output vs the reference |
| `Q1-kl-20261121T120248Z` | 0001 vs 0001–0006 | wikitext-2, 20 x 512 tokens | KL of 0002 on (`kl-loc.txt`) and masked (`kl-loc-nop2.txt`) against the 0001 build (`base-p1.txt`) |
| `Q1-effort-canary-20261121T120543Z` | 0001–0006, 0002 active | 9-item canary x 3 at effort 0.9, 9 items at effort 0.1 / 0.2 / 0.7 / 0.9 | non-empty, correct short answers |
| `VL1-verify-local` | 0001–0006 | `test-backend-ops` vs CPU (`tbo-*.txt`), perplexity, llama-bench arms with switches | kernel tests, PPL with 0002 active |
| `P1-profile` | 0001 | perplexity, llama-bench graph off/on | PPL of the 0001 build |
| `S1-env-sweep` | 0001 | server load | 194 scheduler splits, 180 "disabling SYCL graphs" lines |
| `P2-vtune` | 0001–0006 | server load | 2 scheduler splits (`splits.txt`); the VTune collections failed (not used) |
| `S2-env-sweep-local` | 0001–0006 | llama-bench tg128 / pp512, 5 SYCL switches between 6 controls, `-t 1` vs `-t 8` | DNR-037 |
| `ceiling/` | – | – | byte ceiling inputs and output (computed, not measured) |
| `ledger.jsonl` | – | – | every accepted and rejected lever, in order |

## Not in this folder

Server logs (large, local paths), the 2 GB KL base logits file, the TensorFold port's runs (unreleased research).
