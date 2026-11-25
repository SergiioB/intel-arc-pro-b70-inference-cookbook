# Inkling 2 Mini: local llama.cpp SYCL patches 0001–0006

Local patches for the [Inkling 2 Mini llama.cpp recipe](../../docs/inkling2-mini/INKLING2-MINI-LLAMACPP.md).
They are not upstream and have not been proposed upstream. Apply them only on the base below.

## Base

| Stage | Commit | Git tree after the stage |
|---|---|---|
| ggml-org/llama.cpp `master` | `919794500475862f6768c353255469dc661cb734` | |
| merge PR [#32695](https://github.com/ggml-org/llama.cpp/pull/32695) head (looped layers) | `6196d9802e0757a5218f16f91d9a6830793ba800` | |
| merge PR [#32218](https://github.com/ggml-org/llama.cpp/pull/32218) head (DSpark drafter) | `78d779424d83e17a450807fe42e28445510134b4` | `9998cc28d9e06bc342397065fd324d2d6c4589c9` |
| `git am` 0001–0006 below, in order | | `b1d97b7bfd0f2b997c53db4a96f2b76f1dec4597` |

Both merges are clean. The measured binaries were built from exactly the last tree (source commit `a96c0ef1`
in the lab clone; merge commits differ in author and date, trees do not).

## Patches, in order

| # | File | sha256 | What it changes | Output |
|---|---|---|---|---|
| 0001 | `0001-local-SYCL-get_rows-float-size-the-work-group-to-the.patch` | `4b8cd55235a197c5e2cc3d9902b8c6dcffc292644b3085378da3d684f5a4a3c9` | `get_rows` (float): work-group sized to the row. Fixes the crash past about 16K tokens of context (the relative-bias gather overflowed the `int` launch range at 16,384 KV rows) | same tokens (copy kernel) |
| 0002 | `0002-local-SYCL-fused-dequant-GEMM-NVFP4-A-stage-grouped-.patch` | `d0b9dfc099b410fd18996c6903ec55b55d89a334f319668f478e25c5669c0981` | NVFP4 stage of the fused dequant GEMM: grouped expert GEMM for batches of more than one token (prompt reading). Adds bit 15 to `GGML_SYCL_XMX_GATHER_TYPES` | **changes output** (KL gate failed). Off in the default recipe |
| 0003 | `0003-local-SYCL-scale-strided-kernel-for-row-contiguous-v.patch` | `421c9a2199f51758ada767d826f72589478eb3142fedbb9e744b8730b2bddd79` | `SCALE` on row-contiguous strided views runs on the GPU. Before, the two SITU activation scales of every layer pass ran on the CPU: 96 round trips a token | same tokens |
| 0004 | `0004-local-SYCL-GQA-grouped-p021-nc-decode-attention-mat-.patch` | `67b9dee9201ba8dcbc3b62efd352483e709f5e7b6b6e264b8c9c1e805d516063` | Decode attention mat-vecs read each KV row once per KV head, not once per query head (GQA 3). Switch: `GGML_SYCL_NO_GQA_MV=1` turns it off | same tokens |
| 0005 | `0005-local-GGML_SYCL_NO_STRIDED_SCALE-disables-the-stride.patch` | `7a54b79880d32f0f122179fc13c035780af2dac19edd06d3041d07ded803db4c` | Switch `GGML_SYCL_NO_STRIDED_SCALE=1` turns 0003 off (for A/B runs) | none unless set |
| 0006 | `0006-local-SYCL-fusion-of-the-Inkling-clip-chain-SQR.MUL-.patch` | `fca225287f385b7ba504d600a2b0e0ebc7efb80a8c7be90a08c2e33f69316348` | Fuses Inkling's `clip()` chain (8 graph nodes into 1 kernel, same per-node arithmetic). Switch: `GGML_SYCL_NO_CLIP_FUSION=1` | same tokens |

`SHA256SUMS` in this folder lists the same hashes. "Same tokens" means: greedy output on 5 fixed prompts was
identical to the build with 0001 only, with 0002 masked (`GGML_SYCL_XMX_GATHER_TYPES=32767`). Evidence:
[`results/inkling2-mini/llamacpp-sycl-20261121/`](../../results/inkling2-mini/llamacpp-sycl-20261121/README.md).

## Apply

```bash
git clone https://github.com/ggml-org/llama.cpp.git llama.cpp
benchmarks/inkling2-mini/prepare-llamacpp.sh llama.cpp llama.cpp-inkling2
```

The script fetches the base and both pinned PR commits by SHA (PR #32218's head moved to `88025eb` on 2026-11-21;
the pinned `78d77942` is fetched by its SHA, with the PR ref as a fallback), merges them, checks `SHA256SUMS`, runs
`git am` on 0001–0006 and compares both tree hashes above. It exits 1 on any difference. Relative paths are
resolved against the current directory, so the new tree lands beside the clone.

## Apply test (GPU-free)

2026-11-21 16:42 UTC, exactly the commands above, run in an empty folder: a fresh full clone of
`ggml-org/llama.cpp` (master `19bc41dbb`, which does not contain `78d77942`), then the script with relative paths.
Exit 0 in 5 s:

```text
OK   merged base tree 9998cc28d9e06bc342397065fd324d2d6c4589c9
0001-local-SYCL-get_rows-float-size-the-work-group-to-the.patch: OK
0002-local-SYCL-fused-dequant-GEMM-NVFP4-A-stage-grouped-.patch: OK
0003-local-SYCL-scale-strided-kernel-for-row-contiguous-v.patch: OK
0004-local-SYCL-GQA-grouped-p021-nc-decode-attention-mat-.patch: OK
0005-local-GGML_SYCL_NO_STRIDED_SCALE-disables-the-stride.patch: OK
0006-local-SYCL-fusion-of-the-Inkling-clip-chain-SQR.MUL-.patch: OK
OK   patched (0001-0006) tree b1d97b7bfd0f2b997c53db4a96f2b76f1dec4597
```

The first version of this script (head `a5ba80d`) failed on a fresh clone: the relative target path resolved inside
the clone, and it fetched only the moved PR head. An earlier run against the lab clone, which already held all
commits, had hidden both defects.

This proves source compatibility only: the patched tree is byte-identical to the tree of the measured build.
It does not prove runtime correctness, stability or speed. Those come from the measured runs in the recipe.

## Never apply

- On upstream `master` after PR #32694 (merged 2026-11-20). Copied, not run here: a user report says that tree fails
  to load the root-folder file, and unsloth's `Mainline_Support` README says its loop path loses the per-expert NVFP4
  scales without PR #32704 (see DNR-038).
- On any other model family's llama.cpp tree in this cookbook.
