<!-- GENERATED: source=data/benchmarks.v1.json command='python3 scripts/render-benchmark-catalog.py'; DO NOT EDIT -->

# Public benchmark catalog

> Generated from `data/benchmarks.v1.json`. Edit the JSON, then run
> `python3 scripts/render-benchmark-catalog.py`. Do not edit this table directly.

Benchmark rows require exact workload coordinates, sample counts, metric semantics, and commit-pinned evidence. Capability notes without complete coordinates stay outside rankings.

## Benchmarks

| Model | Hardware | Engine | Workload | Result | Evidence |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B MTP4 | 1× Intel Arc Pro B70 | vllm | P512 / G128 / C1 · n=5 median | 170.91 token/s (client post first output tps) | [Official lab](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/a4ba875413728bd787a2a01a2fc0cac46b035701/docs/qwen36-35a3/QWEN36-MOE-VLLM-XPU.md) |
| Qwen3.8-27B Draft INT4 | 1× Intel Arc Pro B70 | vllm | P512 / G128 / C1 · n=5 median | 106.7 token/s (client post first output tps) | [Official lab](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/a4ba875413728bd787a2a01a2fc0cac46b035701/docs/qwen38-27/QWEN38-VLLM-XPU.md) |
| Qwen3.6-27B MTP4 | 1× Intel Arc Pro B70 | vllm | P512 / G128 / C1 · n=5 median | 69.3 token/s (client post first output tps) | [Official lab](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/a4ba875413728bd787a2a01a2fc0cac46b035701/docs/qwen36-27/QWEN36-DENSE-VLLM-XPU.md) |
| Nemotron-3.5-Lightning DFlash | 1× Intel Arc Pro B70 | vllm | P2048 / G128 / C1 · n=5 median | 186.61 token/s (client post first output tps) | [Official lab](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/a4ba875413728bd787a2a01a2fc0cac46b035701/docs/nemotron35-30a3/NEMOTRON-DFLASH-B70.md) |
| Muse-Glimmer-30B DFlash | 1× Intel Arc Pro B70 | llama.cpp | P512 / G128 / C1 · n=5 median | 26.8 token/s (engine predicted output tps) | [Official lab](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/a4ba875413728bd787a2a01a2fc0cac46b035701/docs/muse-glimmer/MUSE-GLIMMER-B70.md) |
| Ornith-1.5-35B-A3B MTP1 | 1× Intel Arc Pro B70 | vllm | P512 / G128 / C1 · n=5 median | 106.27 token/s (client post first output tps) | [Official lab](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/a4ba875413728bd787a2a01a2fc0cac46b035701/docs/ornith15-35a3/ORNITH-VLLM-XPU.md) |
| Qwen3.8-27B FP8 TP2 MTP8 decode | 2× Intel Arc Pro B70 | vllm | P1024 / G128 / C1 · n=5 median | 60.13 token/s (client e2e output tps) · 71.06 token/s (client e2e output tps max) | [Official lab](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/8c69e5f5143aff60f0bc1cabc97c2fcc364697b2/docs/qwen38-27/FP8-TP2-W8A16.md) |
| Qwen3.8-27B FP8 TP2 cold-input prefill | 2× Intel Arc Pro B70 | vllm | P1024 / G128 / C1 · n=5 median | 1736.2 token/s (cold input rate tps) | [Official lab](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/8c69e5f5143aff60f0bc1cabc97c2fcc364697b2/docs/qwen38-27/FP8-TP2-W8A16.md) |
| Qwen3.8-27B FP8 TP2 MTP8 model-card sampling | 2× Intel Arc Pro B70 | vllm | P512 / G128 / C1 · n=5 median | 22.16 token/s (client e2e output tps) · 36.8 token/s (client e2e output tps max) | [Official lab](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/2c6de11bd7ab097040a11e3746d915b99c5552f0/results/qwen38-27-fp8-tp2-k8-n5/bench/model-card-sampling-p512-g128.json) |
| Qwen3.8-27B FP8 TP2 MTP8 model-card sampling | 2× Intel Arc Pro B70 | vllm | P1024 / G128 / C1 · n=5 median | 11.66 token/s (client e2e output tps) · 28.21 token/s (client e2e output tps max) | [Official lab](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/2c6de11bd7ab097040a11e3746d915b99c5552f0/results/qwen38-27-fp8-tp2-k8-n5/bench/model-card-sampling-p1024-g128.json) |
| Qwen3.8-27B FP8 TP2 LocalMaxxing speed test | 2× Intel Arc Pro B70 | vllm | P74 / G256 / C1 · n=3 median | 31.1 token/s (client post first output tps) · 189.93 ms (client ttft ms) | [Community reviewed](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/2c6de11bd7ab097040a11e3746d915b99c5552f0/results/qwen38-27-fp8-tp2-k8-n5/localmaxxing-approved-receipt.json) |
| Qwen3.8-Flash-Next fused-FP32 C1 decode | 2× Intel Arc Pro B70 | llama.cpp | P512 / G128 / C1 · n=5 median | 23.38 token/s (engine predicted output tps) · 23.73 token/s (engine predicted output tps max) · 183.98 token/s (engine prompt tps) | [Official lab](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/53836d9df7225a83ea6dcef3a38dcdb6abf2453e/results/qwen38-flash-next-dual-b70-c1/n5-fused-fp32-c8k-p512-g128/summary.json) |
| Qwen3.8-Flash-Next fused-F16 C1 cold input | 2× Intel Arc Pro B70 | llama.cpp | P9096 / G128 / C1 · n=5 median | 594.49 token/s (engine prompt tps) · 20.34 token/s (engine predicted output tps) | [Official lab](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/53836d9df7225a83ea6dcef3a38dcdb6abf2453e/results/qwen38-flash-next-dual-b70-c1/n5-fused-f16-c16k-p9096-g128/summary.json) |
| Qwen3.8-Flash-Next MTP + fused multi-token MoE kernel (decode) | 2× Intel Arc Pro B70 | llama.cpp | P512 / G128 / C1 · n=5 median | 33.25 token/s (engine native decode tps) · 30.68 token/s (engine native decode tps prose task) | [Official lab](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/61bc9478424f5f240cef1cee6678864e55ebeceb/results/qwen38-flash-next-mtp-kernel-v1/summary.json) |
| Qwen3.8-Flash-Next prefill (F16 no-spec) | 2× Intel Arc Pro B70 | llama.cpp | P9096 / G128 / C1 · n=3 median | 585.9 token/s (engine native cold input tps) · 22.96 token/s (engine native decode tps) | [Official lab](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/61bc9478424f5f240cef1cee6678864e55ebeceb/results/qwen38-flash-next-mtp-kernel-v1/summary.json) |
| Qwen3.8-Flash-Next 128K near-boundary (first completed 120K on dual B70) | 2× Intel Arc Pro B70 | llama.cpp | P120768 / G128 / C1 · n=5 median | 18.38 token/s (engine native decode tps) · 222.9 token/s (engine native cold input tps) | [Official lab](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/61bc9478424f5f240cef1cee6678864e55ebeceb/results/qwen38-flash-next-mtp-kernel-v1/summary.json) |

## Non-ranking capabilities

| Capability | Hardware | Status | What is established | Evidence |
|---|---|---|---|---|
| Dual-B70 TP2 / PP2 serving | 2× Intel Arc Pro B70 | Validated | Validated infrastructure path for dual-B70 TP2 / PP2 using spawn-time worker affinity, Docker SYS_PTRACE/ipc=host, and oneCCL simple-threshold variables. Model-specific numeric cards, exact stacks, and evidence are published separately. | [source](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/a4ba875413728bd787a2a01a2fc0cac46b035701/docs/DUAL-B70-TP2.md) |
| Qwen3.8-27B FP8 W8A16 | 1× Intel Arc Pro B70 | Provisional | The XPU reroute removes dynamic activation quantization and reports roughly 34% speedup and 48 token/s at p1024. Output length, sample count, and full public result coordinates are not yet recorded, so this is excluded from benchmark rankings. Superseded by qwen38-fp8-tp2-k8-p1024-g128-c1 (same recipe lineage, n=5, dual-card). | [source](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/0f876e141c7a45b9db6081fba53b207b150206ca/patches/patch_fp8_w8a16.py) |
| Qwen3.8-Flash-Next 128K C1 capacity | 2× Intel Arc Pro B70 | Provisional | C1 llama.cpp SYCL completed p512/g128 and actual-p9096/g128 at -c 131072 -b/-ub 1024 (b1536+ rejected). n=3 only. p512 decode ~23.06-23.54 engine tok/s. c128 is context, not concurrency. Not a ranking cell. | [source](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/53836d9df7225a83ea6dcef3a38dcdb6abf2453e/results/qwen38-flash-next-dual-b70-c1/n3-context-map.json) |
| Qwen3.8-Flash-Next GPU MUL_MAT_ID grouping | 2× Intel Arc Pro B70 | Provisional | Matched C1 n=3 fused-F16 16K actual-p9096: GPU grouping 599.48 vs disable 594.67 cold-input (+0.81%). Decode unchanged. Occupancy D2H remains. Not a ranking cell. | [source](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/53836d9df7225a83ea6dcef3a38dcdb6abf2453e/results/qwen38-flash-next-dual-b70-c1/n3-f16-gpu-group-c16k-p9096-g128/summary.json) |

## Trust labels

- **Official lab:** measured under the cookbook's documented process.
- **Community reviewed:** schema and evidence reviewed; contributor controls the machine.
- **Validated:** an operational capability worked; any model-specific numeric records are published separately.
- **Provisional:** an early numeric claim is documented but lacks one or more required workload coordinates.
