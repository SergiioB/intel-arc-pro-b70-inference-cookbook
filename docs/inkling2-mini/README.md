# Inkling 2 Mini on Intel Arc Pro B70

Thinking Machines' [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on one B70.
Status 2026-11-21: **one research recipe on llama.cpp SYCL, 29 tok/s decode at a short prompt, bit-exact with the
unpatched engine. A faster build is in progress and is not a recipe yet.**

## Pick a route

| Route | Status | Page |
|---|---|---|
| llama.cpp SYCL + 2 upstream PRs + 6 local patches, one card | Research recipe. Measured 2026-11-21, 3 seeds a cell. 29.09 / 25.14 tok/s decode at 76 / 9,172 prompt tokens (model-card sampling, 1,024-token replies). Output identical to the engine without the patches | [INKLING2-MINI-LLAMACPP.md](INKLING2-MINI-LLAMACPP.md) |
| Same, with patch 0002 active (fast-prefill variant) | Lossy, opt-in only. Prompt reading 2.4x at 76 tokens and 1.55x at 9,172 against the patch-0001 build; fails the KL gate | [§6 of the recipe](INKLING2-MINI-LLAMACPP.md#6-prompt-reading-and-time-to-first-token) |
| TensorFold XPU port | Unreleased research, not reproducible from this repository | [What is running now](#what-is-running-now) |

Do not mix: the patch stack in [`patches/inkling2-mini-llamacpp/`](../../patches/inkling2-mini-llamacpp/APPLY.md)
applies to one base tree only, and the GGUF layout of the unsloth root file loads only on that tree. Numbers on this
page belong to this model, this file and this engine build.

## The model (copied from the model card and config, read 2026-11-20 at revision `6e7c8248`)

| Property | Value |
|---|---|
| Size | 26.4B parameters total, 3.4B active a token (5.4B effective active counting the loop) |
| Loop | 24 stored layers; layers 6–17 run 3 passes, so a token makes 48 layer applications |
| Experts | 128 per layer, 8 active a token, expert width 1,152; hidden size 2,304; vocabulary 201,024 |
| Attention | 1,048,576-token context; local window of 1,024 tokens on 20 of 24 layers; relative-position bias |
| Release format | Experts NVFP4 (layers 0–19) and MXFP8 (layers 20–23); attention, embeddings, LM head BF16. No BF16 release |
| Drafter | DSpark block drafter, 3 layers, block 8 (closed on llama.cpp for now: DNR-034) |
| Sampling (card) | `temperature 1.0`, no `top_p`/`top_k` (`generation_config.json`); `reasoning_effort` 0.9 by default; the card says the lowest effort does not turn reasoning off |
| Licence | Apache-2.0. The model card also links an acceptable-use policy; read it on the [model page](https://huggingface.co/thinkingmachines/Inkling-2-Mini) before use |

The release runs NVFP4 activations too (W4A4), which needs FP4 matrix units this card does not have. The GGUF route
keeps the NVFP4 weights and does not use FP4 activations.

## Which file (quant check, 2026-11-21 08:03 UTC, 104 Inkling repositories on Hugging Face)

Copied from the repository listings and model cards; only the chosen file was downloaded and measured here.

| Repository / file | Size | Experts | Engine it needs | Bytes a token vs the chosen file | Note |
|---|---|---|---|---|---|
| **unsloth `Inkling-2-Mini-UD-Q4_K_XL.gguf` (chosen)** | 17.63 GB | NVFP4 as released + Q8_0 for the 4 MXFP8 layers | llama.cpp + PR #32695 (root-folder layout) | 1.00 | the release's expert bits unchanged |
| unsloth UD-Q8_K_XL, UD-Q4_K_M, UD-IQ4_XS | 22.87 / 16.61 / 15.73 GB | NVFP4 kept down to UD-IQ4_XS | same; `Mainline_Support/` copies for upstream | UD-IQ4_XS about 2.6 % fewer | the 4 Q8_0 layers and the head are smaller |
| nace-ai `Inkling-2-Mini-NVFP4.gguf` | 19.10 GB | NVFP4 repacked, not re-quantized | upstream + PR #32704 | more (larger dense parts) | card reports KL 0.004 vs the original with BF16 activations (2,040 positions) |
| Xylolabs `NVFP4-Q8_0` / `NVFP4` | 17.63 / 22.92 GB | NVFP4 | upstream `29b52d398` or later | equal / more | no accuracy table yet |
| bartowski IQ2 … Q6_K_L | – | re-quantized (imatrix) | upstream b12600 | – | expert down tensor is 1,152 wide, so only 32-wide block types apply |
| ggml-org Q8_0 | 27.82 GB | Q8_0 re-encoded from the NVFP4 values | llama.cpp with loop support | 1.34x more (95 vs 127 tok/s ceiling, computed) | no quality gain possible over its NVFP4 source (DNR-040) |
| grant-ai `Inkling-2-Mini-EXL3-4bpw` | 16 GB | EXL3 about 4 bpw from unpacked NVFP4 | TensorFold, CUDA only | – | card reports KL 0.0236, top-1 96.29 % |
| vcruz305 SAGE-REXL3 2.92 / 4.23 bpw | – | EXL3 | gated | – | – |

Conclusion: no file reads fewer bytes a token while keeping the NVFP4 expert weights, so UD-Q4_K_XL stays. Pending,
not run: KL of the chosen file against nace-ai's NVFP4 file with their method.

**EXL3: not converted, and no engine runs it on this card.** exllamav3 upstream has no Inkling architecture (checked
2026-11-21), the exl3xpu engine has none, and upstream TensorFold runs Inkling only with CUDA kernels. Arithmetic
(computed): EXL3 at about 4 bpw for every matrix reads about 3.1 GB a token, a ceiling near 170 tok/s, against about
162 for a GGUF that keeps the NVFP4 experts and shrinks attention and the head to Q4_K: +5 %, with extra error from
re-quantizing 4-bit expert weights. Re-entry: exllamav3 adds `inkling2_loop`, or a measured profile shows the
attention kernels (not bytes) limit decode and exl3xpu kernels are faster on the same shapes (DNR-035).

## What is running now

> **Research build, numbers will change, do not cite as a recipe.** Unpublished lab measurements: the code is a
> local branch, not in upstream TensorFold, and cannot be reproduced from this repository yet. No catalog record.

A TensorFold XPU model family for Inkling 2 Mini (own decode and prompt kernels for this shape, same GGUF file):

- **Correctness so far.** Per-layer-pass output against a float32 reference: worst relative L2 error 9.8e-4 (pass
  rule < 2e-2). Greedy streams equal llama.cpp's on 1 of 5 prompts; the other four diverge after 12, 43, 28 and 88
  common tokens. At each divergence the float32 reference ranks the port's token first, with a top-2 margin of
  0.045–0.215 logits. Whether llama.cpp was equally unsure at those positions is pending, so the divergences are not
  yet explained. No KL gate has run yet.
- **Decode at model-card sampling** (card 0, 230 W / 2,800 MHz, on-card sampler, code prompt, 3 seeds, one stream):
  - an earlier build: 39.9 / 33.4 tok/s at 76 / 9,172 prompt tokens, 2,048-token replies (no answer inside 2,048);
  - a later build: 55.8 tok/s at 76 prompt tokens over 8,192-token replies; 2 of 3 seeds answered within 8,192;
  - a later build again: 71.5 / 64.1 tok/s at 76 / 9,172 prompt tokens, 2,048-token replies.
- **Prompt reading** (first integrated build, greedy, cold, 3 repeats): 1,260 / 1,325 / 1,191 tok/s at 512 / 4,096 /
  32,768 tokens. That build failed the per-layer correctness check (a wrong constant, fixed afterwards), so these are
  kernel throughput numbers, not a working engine's.
- **Targets** (operator): 100 tok/s decode on the model-card cell above, 10,000 tok/s prompt reading.

When a build passes the KL gate and the same-session controls, it gets its own recipe page and catalog records; this
block is then replaced.

Hy4-flash: queued, nothing measured.

## Evidence and records

- Raw rows: [`results/inkling2-mini/llamacpp-sycl-20261121/`](../../results/inkling2-mini/llamacpp-sycl-20261121/README.md)
- Scripts: [`benchmarks/inkling2-mini/`](../../benchmarks/inkling2-mini/README.md)
- Catalog records: `inkling2-mini-llamacpp-*` in [`data/benchmarks.v1.json`](../../data/benchmarks.v1.json)
- Closed arms: DNR-034 to DNR-040 in [DO-NOT-REPEAT.md](../DO-NOT-REPEAT.md)
