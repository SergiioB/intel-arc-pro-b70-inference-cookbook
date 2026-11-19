# Apply the SYCL tensor-split allreduce patches

Base: llama.cpp master `78e05298f0d68abff41d08d79fa7469264b2aff0` (2026-11-19). All three patches apply
cleanly there in this order (checked with `git apply`, 2026-11-19 11:20 CET). Local only: nothing is sent to
ggml-org/llama.cpp (operator decision 2026-11-19 06:20 CET).

| File | What | sha256 |
|---|---|---|
| `0001-sycl-opt-in-F32-peer-copy-allreduce-for-two-GPU-tens.patch` | Commit `3688ed8`. Adds `GGML_SYCL_TP_P2P` and mode 1 (peer `memcpy` ordered by cross-queue events). `ggml-sycl.cpp` +68/-5, `docs/backend/SYCL.md` +1. **Mode 1 is not for use:** it leaks Level Zero events and segfaulted in the driver after about 100,000 calls (REPORT.md section 5). | `5f25dfed9bb7a1f30f4a6a1cee2814a6c0bf698870a75da121dc9ac142ffc71b` |
| `0002-sycl-peer-store-allreduce-mode-GGML_SYCL_TP_P2P-2.patch` | Commit `64eb6f6`. Adds mode 2: each GPU stores its F32 partial into the peer's inbox, a one-item kernel raises the peer's flag, the add kernel polls the local flag. No cross-queue events, no host wait. `ggml-sycl.cpp` +107/-5, `docs/backend/SYCL.md` +1/-1. **This is the measured candidate.** | `9a66ca4949e61b3aa344c1d2ee600c72b33639fe671a125277e4f40daca9da2a` |
| `measurement-only-GGML_SYCL_TP_DIAG_SKIP.patch` | Commit `1626bf5`. Not for use. `GGML_SYCL_TP_DIAG_SKIP=1` skips every allreduce (wrong output) to measure what the reduction costs. The measured binaries carry it, unset except in the `diag` arm. | `1433688b1ba081985671a6ede613d7e7cfa8fd8dc876779a78852008321881ee` |

## 1. Apply

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 78e05298f0d68abff41d08d79fa7469264b2aff0
git am /path/to/0001-*.patch /path/to/0002-*.patch      # or: git apply, in that order
```

## 2. Build (the command used for the measured binary)

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DGGML_SYCL=ON -DGGML_SYCL_F16=OFF -DGGML_SYCL_DNN=ON -DGGML_SYCL_GRAPH=ON \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_MTMD=OFF
cmake --build build --parallel 10 --target llama-bench llama-server llama-perplexity
```

About 4 minutes incremental, about 15 minutes clean on the Ryzen 7 5700X3D, under a 20 GiB memory cap.

## 3. Enable

```bash
export SYCL_ENABLE_DEFAULT_CONTEXTS=1      # one SYCL context for both cards (Linux); required
export UR_LOADER_USE_LEVEL_ZERO_V2=0       # Level Zero v1 adapter, as measured
export ONEAPI_DEVICE_SELECTOR='level_zero:*'
export SYCL_CACHE_PERSISTENT=0             # DPC++ 2026.1 persistent cache segfaults llama.cpp on this host
export GGML_SYCL_TP_P2P=2
./build/bin/llama-server -m MODEL.gguf -dev SYCL0,SYCL1 -sm tensor -ts 1,1 --fit off -ngl 99 -fa on ...
```

Without the shared context the code logs `peer-copy allreduce unavailable, using host staging` and runs the stock
path. `llama-server` hides that backend log at default verbosity; `llama-bench -v` and other tools show
`ggml_backend_sycl_comm_init: using F32 peer-store allreduce` when mode 2 is active.

## 4. Revert

- At run time: unset `GGML_SYCL_TP_P2P` or set it to `0`. The stock path then runs unchanged except one host
  wait when the exchange buffer grows.
- In the tree: `git revert` the two commits, or `git apply -R` the patches in reverse order (0002, then 0001).
