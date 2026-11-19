# Apply the SYCL peer-copy allreduce patch

Base: llama.cpp master `78e05298f0d68abff41d08d79fa7469264b2aff0` (2026-11-19). Both patches apply cleanly
there (checked with `git apply --check`, 2026-11-19 06:22 CET).

| File | What | sha256 |
|---|---|---|
| `0001-sycl-opt-in-F32-peer-copy-allreduce.patch` | The change (commit `3688ed8`): `ggml-sycl.cpp` +68/-5, one line in `docs/backend/SYCL.md`. Off unless `GGML_SYCL_TP_P2P=1`. | `b84ac9f2956d90fce75f16a6eff2a4fb058bcdd0d2c080abe3c706119d10800c` |
| `measurement-only-GGML_SYCL_TP_DIAG_SKIP.patch` | Not for use. `GGML_SYCL_TP_DIAG_SKIP=1` skips every allreduce (wrong output) to measure how much decode time the reduction costs. Measured builds carry it, unset. | `107efd0f7919d0772436bb93de3ce7b57e7151c6cb859e4a740e2ba768c73fbe` |

## 1. Apply

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 78e05298f0d68abff41d08d79fa7469264b2aff0
git am /path/to/0001-sycl-opt-in-F32-peer-copy-allreduce.patch   # or: git apply
```

## 2. Build (the command used for the measured binary)

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DGGML_SYCL=ON -DGGML_SYCL_F16=OFF -DGGML_SYCL_DNN=ON -DGGML_SYCL_GRAPH=ON \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_MTMD=OFF
cmake --build build --parallel 10 --target llama-bench llama-server llama-perplexity test-backend-ops
```

About 15 minutes on the Ryzen 7 5700X3D (run under a 20 GiB memory cap).

## 3. Enable

All four variables are needed; without the shared context the patch logs
`peer-copy allreduce unavailable, using host staging` and falls back to the stock path.

```bash
export SYCL_ENABLE_DEFAULT_CONTEXTS=1      # one SYCL context for both cards (Linux)
export UR_LOADER_USE_LEVEL_ZERO_V2=0       # Level Zero v1 adapter, as measured
export ONEAPI_DEVICE_SELECTOR='level_zero:*'
export GGML_SYCL_TP_P2P=1
./build/bin/llama-server -m MODEL.gguf -dev SYCL0,SYCL1 -sm tensor -ts 1,1 --fit off -ngl 99 -fa on ...
```

Check the server log for `ggml_backend_sycl_comm_init: using F32 peer-copy allreduce`.

## 4. Revert

- At run time: unset `GGML_SYCL_TP_P2P` or set it to `0`. The code path is then byte-for-byte the stock one
  except one host wait when the exchange buffer grows.
- In the tree: `git revert <commit>` or `git apply -R 0001-sycl-opt-in-F32-peer-copy-allreduce.patch`.
