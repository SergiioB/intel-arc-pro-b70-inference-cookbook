#!/usr/bin/env bash
# Build the prepared tree (prepare-llamacpp.sh) with SYCL, ahead-of-time for the Arc Pro B70 (bmg_g31).
#
#   benchmarks/inkling2-mini/build-llamacpp-sycl.sh <prepared tree>
#
# Measured build: oneAPI 2026.1 (icx/icpx), compute-runtime 26.05.37020.3, Ubuntu 26.04, about 7 minutes
# with -j 8. The binaries go to <tree>/build-sycl/bin.
set -euo pipefail
S=${1:?prepared llama.cpp tree}
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null; set -u
export SYCL_CACHE_PERSISTENT=0
cmake -S "$S" -B "$S/build-sycl" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
  -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_DEVICE_ARCH=bmg_g31 \
  -DGGML_SYCL_F16=ON -DGGML_SYCL_GRAPH=ON -DGGML_SYCL_DNN=OFF \
  -DGGML_SYCL_HOST_MEM_FALLBACK=OFF -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON
cmake --build "$S/build-sycl" --target llama-server llama-bench llama-perplexity -j 8
sha256sum "$S"/build-sycl/bin/llama-server "$S"/build-sycl/bin/libggml-sycl.so
