# vLLM XPU kernel source fixes

These are **source** edits in `vllm-xpu-kernels`. They are **not** Python
runtime patches. They change C++ / kernel-config and need a
`vllm-xpu-kernels` rebuild (or a future image that includes the PR).

The DFlash / no-spec launchers already apply the **Python** router fix
(`../patch_xpu_grouped_topk_native_v2.py`). That is enough to *serve*
Nemotron DFlash from the public digest. [vllm#52159](https://github.com/vllm-project/vllm/pull/52159)
closed without merge; the fused XPU route is [vllm#53580](https://github.com/vllm-project/vllm/pull/53580)
(open). See [IMAGE-AND-PATCH-MATRIX.md](../../docs/IMAGE-AND-PATCH-MATRIX.md)
Upstream status.

| File | What | Needed for DFlash serve? |
|---|---|---|
| `0001-zero-xe2-grouped-gemm-atomic.py` | `at::empty` → `at::zeros` on the Xe2 grouped-GEMM scheduler counter. [vllm-xpu-kernels#524](https://github.com/vllm-project/vllm-xpu-kernels/pull/524) still open; kernels `main` still has `at::empty`. | No. Needed for temperature-0 graph-replay determinism |
| `0002-muse-paged-decode-tuple.py` | add `16,128,64,false,true,false` | **No. Already in kernels `main` via [vllm-xpu-kernels#526](https://github.com/vllm-project/vllm-xpu-kernels/pull/526) (merged 2026-08-14).** Do not apply on a current checkout. Keep the script for old trees; it no-ops if the tuple is present. |

## Apply to a kernels checkout

```bash
git clone --depth=1 https://github.com/vllm-project/vllm-xpu-kernels.git
python3 patches/vllm-xpu-kernels/0001-zero-xe2-grouped-gemm-atomic.py \
  --root vllm-xpu-kernels
# then rebuild vllm-xpu-kernels per that repo's README
```

Do **not** run `0002-muse-paged-decode-tuple.py` against current `main`.
`0001` fails closed if the public-HEAD `at::empty` anchor is gone (PR already
merged, or the file moved). Do **not** treat a local Docker tag as the
reproduce default.
