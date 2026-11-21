# Inkling 2 Mini on Intel Arc Pro B70

Thinking Machines' [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on one B70.
Status 2026-11-21: **two routes, both research.** A TensorFold XPU research build decodes 71.48 tok/s at a short
prompt (tagged and rebuildable, not upstream, output not yet shown equal to llama.cpp). The llama.cpp SYCL recipe
decodes 29.09 tok/s and is bit-exact with the unpatched engine. Work on both continues.

## Pick a route

| Route | Status | Section |
|---|---|---|
| TensorFold XPU, tag `b70-inkling2-mini-20261121`, one card | Research build, numbers will change; not upstream TensorFold. 71.48 / 64.05 tok/s decode at 76 / 9,172 prompt tokens (model-card sampling, 2,048-token replies). Per-pass error against a float32 reference passes; greedy output equals llama.cpp's on 1 of 5 prompts; KL not yet measured. Prompt reading slower than llama.cpp (TTFT 104.8 s at 9,172 tokens) | [below](#tensorfold-xpu-research-build-tag-b70-inkling2-mini-20261121) |
| llama.cpp SYCL + 2 upstream PRs + 6 local patches, one card | Research recipe, bit-exact. 29.09 / 25.14 tok/s decode at 76 / 9,172 prompt tokens (model-card sampling, 1,024-token replies) | [INKLING2-MINI-LLAMACPP.md](INKLING2-MINI-LLAMACPP.md) |
| llama.cpp with patch 0002 active (fast-prefill variant) | Lossy, opt-in only. Prompt reading 2.4x at 76 tokens and 1.55x at 9,172 against the patch-0001 build; fails the KL gate | [§6 of the recipe](INKLING2-MINI-LLAMACPP.md#6-prompt-reading-and-time-to-first-token) |

Do not mix: both routes read the same GGUF file, but engines, patches and numbers belong to their own route. The
llama.cpp patch stack in [`patches/inkling2-mini-llamacpp/`](../../patches/inkling2-mini-llamacpp/APPLY.md) applies to
one base tree only.

## TensorFold XPU (research build, tag b70-inkling2-mini-20261121)

> Research build, numbers will change; not upstream TensorFold. Its output is not yet shown to equal llama.cpp's
> (see the correctness box). Use the llama.cpp recipe when you need output identical to the reference engine.

A TensorFold model family for Inkling 2 Mini with its own XPU kernels for this model's shapes (decode one token at a
time, prompt in 64-row chunks, sampling on the card). Code: [`SergiioB/TensorFold` tag
`b70-inkling2-mini-20261121`](https://github.com/SergiioB/TensorFold/tree/b70-inkling2-mini-20261121) = commit
`e47dc7ae8e06c9cdefb846e01cf4671bad6cb3db`, 21 commits on upstream `ea5576501`, 8 files (runner
`zig/src/families/inkling2/xpu_main.zig`, kernels `zig/kernels/xpu/inkling2/ink2.cl`, build step
`zig/build/xpu_inkling2.zig`, `build.zig`, `zig/src/core/gguf_tensor.zig`, `tools/xpu/inkling2_{ref,cmp,serve}.py`).

### What you need

- One Arc Pro B70; measured at 230 W cap and 2,800 MHz. Ubuntu 26.04, kernel 7.0.0-29, compute-runtime and `ocloc`
  26.05.37020.3, Zig 0.17.0, Python 3 with `transformers` (measured 5.15.0, `tokenizers` 0.22.2).
- The same model file as the llama.cpp route: `unsloth/Inkling-2-Mini-GGUF` @ `b08906e3`,
  `Inkling-2-Mini-UD-Q4_K_XL.gguf`, sha256 `e62d463c135ffe283df8d94b279a96ca145ac128b52ce854478613045bf4de15`
  ([download line](INKLING2-MINI-LLAMACPP.md#3-serve)).
- The release's `tokenizer.json` (sha256 `7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6`) and
  `chat_template.jinja` (sha256 `53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc`) from
  `thinkingmachines/Inkling-2-Mini` @ `ed40e7da625f594fc9c4990f5f66be24873c22d7`, in one folder (`META` below).

### Build (GPU-free, about 2 minutes)

```bash
ZIG=/path/to/zig-0.17.0/zig benchmarks/inkling2-mini/build-tensorfold-xpu.sh "$PWD/tf-inkling2"
```

The script clones the tag, runs `zig build xpu-inkling2 -Doptimize=ReleaseFast --prefix <dir>/out`, and compares
the binary with the measured one. The build step `xpu-inkling2` (from `zig/build/xpu_inkling2.zig`) compiles
`ink2.cl` twice with `ocloc compile -device bmg` (`-cl-std=CL3.0`, and `-cl-std=CL3.0 -cl-intel-256-GRF-per-thread`
for the 256-register variant), embeds both kernel binaries, and links the runner. No ngen or oneDNN step. Option
`-Dink2-ocloc=<path>` picks another `ocloc`.

Reproducibility (measured 2026-11-21): a clean clone of the tag built this way gives a runner whose `.text`,
`.rodata` (with both kernel binaries) and `.data` are byte-identical to the binary that produced the table below;
only the debug sections differ, because they hold build paths
([`binary-identity.txt`](../../results/inkling2-mini/tensorfold-xpu-20261121/binary-identity.txt)).

### Serve

```bash
TF="$PWD/tf-inkling2" MODEL=/path/to/Inkling-2-Mini-UD-Q4_K_XL.gguf META=/path/to/meta \
  benchmarks/inkling2-mini/launch-inkling2-mini-tensorfold.sh
# = ZE_AFFINITY_MASK=0 python3 tf-inkling2/tf/tools/xpu/inkling2_serve.py --runner tf-inkling2/out/bin/xpu-inkling2 \
#     --model Inkling-2-Mini-UD-Q4_K_XL.gguf --meta META --port 18452 --ctx 16384
```

- Environment: none beyond `ZE_AFFINITY_MASK` (one card). The measured cell used the defaults of `e47dc7ae8`; no
  `INK2_*` variable is needed (the job set `INK2_X=0`, a name the runner never reads).
- `--ctx 16384` is the measured context; prompt plus reply must fit, or the request fails (the 9,172-token cells of an
  8,192-token run failed this way).
- The wrapper serves OpenAI-style `/v1/chat/completions` on 127.0.0.1. It accepts `temperature` and `seed`; `top_p`
  must be 1 and penalties 0, `top_k 1` means greedy; anything else is refused. `chat_template_kwargs.reasoning_effort`
  goes to the template (default 0.9); reasoning cannot be turned off.

### Measured: decode at model-card sampling

One card (card 0, 230 W / 2,800 MHz), one request at a time, a fresh server a cell, cold prompt, code prompts,
`temperature 1.0`, `top_p 1.0`, `reasoning_effort 0.9`, 3 seeds (4242–4244), one discarded warmup, median of 3.
Decode = streamed rate after the first token. Energy = card counter over the decode phase.

| Build and settings | Reply budget | Prompt tokens | Decode tok/s (range) | TTFT | Card W | J/token | Reasoning / answer tokens |
|---|---|---|---|---|---|---|---|
| **Tag `e47dc7ae8`, defaults** | 2,048 | 76 | **71.48** (71.35–71.49) | 0.68 s | 204 | 2.86 (2.83–2.86) | 2,046 / 0 in all 3 |
| **Tag `e47dc7ae8`, defaults** | 2,048 | 9,172 | **64.05** (64.03–64.05) | 104.8 s | 207 | 3.24 | 2,046 / 0 in all 3 |
| Same binary, check2's settings (control, interleaved by seed) | 2,048 | 76 | 39.92 | 0.68 s | 164 | 4.11 | 2,046 / 0 in all 3 |
| Same binary, check2's settings (control) | 2,048 | 9,172 | 33.47 | 104.9 s | 165 | 4.93 | 2,046 / 0 in all 3 |
| Older untagged build `889fdb304` (2 uncommitted files) | 8,192 | 76 | 55.83 | 0.68 s | 194 | 3.48 | 4,209 / 3,978; 8,190 / 0; 5,239 / 2,948 |

Raw rows: [`results/inkling2-mini/tensorfold-xpu-20261121/`](../../results/inkling2-mini/tensorfold-xpu-20261121/README.md).

- **No answer arrived inside 2,048 tokens:** at effort 0.9 the code prompt reasons past the budget. With 8,192
  tokens, on the older build, 2 of 3 seeds answered (at 4,209 and 5,239 reasoning tokens). That row is an older build
  and cannot be rebuilt exactly; it shows that answers arrive, not the tag's speed over long replies.
- Same-binary gain from the new defaults: +79 % at 76 tokens, +91 % at 9,172 (control rows above).
- Prompt reading is the weak side: 117 tok/s at 76 tokens and 87.6 tok/s at 9,172 (engine timing), TTFT 104.8 s at
  9,172 tokens against 54.4 s on the llama.cpp recipe's 0001 build. A faster prompt path is being built separately and
  is not in the tag.

> **Correctness (plain):**
> - Per-layer-pass output against a float32 reference on prompt p0: worst relative L2 error 9.8e-4, **pass** (rule
>   < 2e-2). Measured on the tagged binary.
> - Greedy decode of prompt p0: 24 of 24 tokens equal to the float32 reference. Measured on the tagged binary.
> - Greedy streams against llama.cpp: equal on 1 of 5 prompts; the other four diverge after 12, 43, 28 and 88 tokens
>   (measured on older builds of the port, not re-run on the tag). At each divergence the float32 reference ranks the
>   port's token first, by 0.045–0.215 logits; llama.cpp's side of that margin analysis is pending. So the output is
>   **not shown to be equal** to llama.cpp's.
> - KL against llama.cpp: **not yet measured.**
> - Sampler: on-card Gumbel-max at temperature 1.0 after the logit softcap, noise from a hash of (seed, step, token).
>   Distribution test pending.
> - Status: research build, numbers will change; not upstream TensorFold.

**Against the llama.cpp recipe (different token budgets):** llama.cpp's 29.09 / 25.14 tok/s were measured with
1,024-token replies, this cell with 2,048. Longer replies decode deeper into context and run slower, so the budget
difference does not favour TensorFold. Ratios: 71.48 / 29.09 = 2.46x at 76 prompt tokens, 64.05 / 25.14 = 2.55x at
9,172 (different token budgets; not a same-session A/B). Prompt reading goes the other way (TTFT 104.8 s against
54.4 s at 9,172 tokens).

Not tested on the tag: 32K and longer context, retrieval, context fit, concurrency, DSpark, tool calls, KL, answers on
long tasks within the budget.

## llama.cpp SYCL (research recipe, bit-exact)

The exact route: llama.cpp `master` `919794500` + PR #32695 + PR #32218 + six local SYCL patches, SYCL graph replay,
patch 0002 masked. 29.09 / 25.14 tok/s decode at 76 / 9,172 prompt tokens, KL 0 and greedy 5/5 identical against the
engine without the patches. Full recipe, build, launch line, controls and quality:
[INKLING2-MINI-LLAMACPP.md](INKLING2-MINI-LLAMACPP.md).

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

> Unpublished lab work in progress. No catalog record; nothing here is a recipe.

- TensorFold prompt path: a separate branch with batched expert, dense and attention kernels. Its first integrated
  build read 1,260 / 1,325 / 1,191 tok/s at 512 / 4,096 / 32,768 prompt tokens (greedy, cold, 3 repeats), but that
  build failed the per-layer correctness check (a wrong constant, fixed afterwards), so these are kernel throughput
  numbers, not a working engine's. Not in the tag.
- TensorFold decode: further changes after the tag; the next tagged build replaces the table above as a new
  generation.
- Pending checks: KL against llama.cpp, llama.cpp's side of the greedy margins, a sampling distribution test.
- Targets (operator): 100 tok/s decode on the model-card cell, 10,000 tok/s prompt reading.

Hy4-flash: queued, nothing measured.

## Evidence and records

- Raw rows: [`results/inkling2-mini/tensorfold-xpu-20261121/`](../../results/inkling2-mini/tensorfold-xpu-20261121/README.md) (TensorFold), [`results/inkling2-mini/llamacpp-sycl-20261121/`](../../results/inkling2-mini/llamacpp-sycl-20261121/README.md) (llama.cpp)
- Scripts: [`benchmarks/inkling2-mini/`](../../benchmarks/inkling2-mini/README.md)
- Catalog records: `inkling2-mini-tensorfold-*` and `inkling2-mini-llamacpp-*` in [`data/benchmarks.v1.json`](../../data/benchmarks.v1.json)
- Closed arms: DNR-034 to DNR-040 in [DO-NOT-REPEAT.md](../DO-NOT-REPEAT.md)
