# Inkling 2 Mini, TensorFold XPU research build, one Arc Pro B70: raw evidence (2026-11-21)

Raw rows behind the [TensorFold XPU route](../../../docs/inkling2-mini/README.md#tensorfold-xpu-research-build-tag-b70-inkling2-mini-20261121).
Research build: numbers will change; not upstream TensorFold. `summary.json` is compiled from these rows only:

```bash
python3 benchmarks/inkling2-mini/summarize.py results/inkling2-mini/tensorfold-xpu-20261121 > summary.json
```

## Provenance

Copied on 2026-11-21 from the port's scratch folder in the lab (these runs are not tracked in a lab repository).
Local paths were replaced: `$MODELS` = model folder, `$TF` = TensorFold source tree, `$TF_WORK` = the port's scratch
folder, `$WORK` = other scratch, `$HOME` = home folder. Nothing else was edited.

## Which binary

| Run | Binary | Source |
|---|---|---|
| `check4-20261121T134525Z` | `xpu-inkling2-c5`, sha256 `7175a4f58909aaf080a3cf63b3af3e6dc4748c1bce0916a91a9f19072f739ce6` | commit `e47dc7ae8e06c9cdefb846e01cf4671bad6cb3db` = public tag `b70-inkling2-mini-20261121` of `github.com/SergiioB/TensorFold`. `env.txt` shows `git_head=641507c5`: that is the tree's HEAD when the job ran, not the build. `binary-identity.txt` proves the build: a rebuild from the public tag has byte-identical `.text`, `.rodata` (both kernel binaries inside) and `.data`; only debug sections (build paths) differ |
| `speed3-20261121T133350Z` | `xpu-inkling2-c4`, sha256 `03de62e21b0ab48c38d217031958da048efd09b54d38e9030dffad4d9acf9042` | an older untagged build: `git_head=889fdb304` with 2 uncommitted files at run time (`dirty=2`). Cannot be rebuilt exactly; context only |

## Common shape

- One Intel Arc Pro B70, card 0, 230 W cap, 2,800 MHz `max_freq`; Ubuntu 26.04, kernel 7.0.0-29, compute-runtime and
  ocloc 26.05.37020.3, Zig 0.17.0.
- Model: `unsloth/Inkling-2-Mini-GGUF` @ `b08906e3105be6a53c7dfdd4fd8e1e272eb01221`, `Inkling-2-Mini-UD-Q4_K_XL.gguf`,
  sha256 `e62d463c135ffe283df8d94b279a96ca145ac128b52ce854478613045bf4de15` (same file as the llama.cpp route).
- Server: `tools/xpu/inkling2_serve.py --runner <binary> --model <gguf> --meta <dir> --port <n> --ctx 16384`; one
  request at a time, a fresh server for every cell, cold prompt. The wrapper samples on the card (Gumbel-max after the
  logit softcap).
- Cells: `depth_bench.py run --sampling inkling-2-mini:think` (temperature 1.0, top_p 1.0, reasoning effort 0.9),
  code prompts of 76 and 9,172 tokens, seeds 4242–4244, one discarded warmup a server. `decode_tok_s` = streamed rate
  after the first token. `completion_tokens_details` splits reasoning and answer tokens.

## Files

| File | What it shows |
|---|---|
| `check4-*/cell-after/cells.jsonl` | Tagged build, defaults (the `after` arm set only `INK2_X=0`, a name the runner never reads): 2,048-token replies |
| `check4-*/cell-before/cells.jsonl` | Same binary with check2's settings (`INK2_CH256=1 INK2_SPLIT=2048 INK2_SPLITSWA=100000 INK2_ROWS=2 INK2_DOWN1=1 INK2_CONV256=1`), interleaved by seed with `after`: same-binary control |
| `check4-*/cmp-v2.txt`, `cmp-fd2.txt`, `check-v2.log` | Tagged build: per-layer-pass check against the float32 reference (worst relative L2 9.82e-4) and 24 greedy tokens of prompt p0 |
| `check4-*/ab-short.txt`, `ab-long.txt` | Greedy step-speed A/Bs on the tagged build (not published as speeds) |
| `speed3-*/depth/cells.jsonl` | Older build: 76-token prompt, 8,192-token replies. The three 9,172-token cells failed: server context 16,384 < 9,172 + 8,192 |
| `speed3-*/exact.jsonl`, `exact-vs-llamacpp.txt` | Older build: 5 greedy prompts against llama.cpp's greedy file: 1 of 5 equal |
| `tmp/diag/margins.log` | Float32 reference at each divergence, teacher-forced on the common prefix |
| `ref/p0.ids.json`, `ref/p0.gen.json` | Prompt p0 tokens and the reference's 24 greedy tokens |
| `binary-identity.txt` | Tag rebuild against the measured binary |
