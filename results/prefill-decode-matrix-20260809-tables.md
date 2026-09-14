## Phase-separated vLLM benchmark

Tested stack: vLLM `0.26.1rc1.dev457+gc810e5ee9.xpu`, `vllm-xpu-kernels 0.1.12`, C1, `n=5`, scheduler 8192, context 131072, prefix cache enabled with zero hit delta, configured 165 W cap.

### Cold input rate (actual input tokens / TTFT, tok/s)

| Mode | p512 | p2048 | p4096 | p6144 | p8192 | Full p131071 |
|---|---:|---:|---:|---:|---:|---:|
| No spec | 5,156 | 6,674 | 7,197 | 7,451 | 7,576 | 3,144 |
| MTP1 | 4,840 | 7,377 | 6,999 | 7,189 | 7,264 | 2,679 |
| MTP2 | 4,843 | 7,341 | 7,002 | 7,140 | 7,229 | 2,683 |
| MTP4 | 4,532 | 7,401 | 6,868 | 7,057 | 7,197 | 2,678 |

### Decode at p512 (client post-first tok/s)

| Mode | g32 | g128 | g256 | g512 |
|---|---:|---:|---:|---:|
| No spec | 97.43 | 96.79 | 96.60 | 96.13 |
| MTP1 | 122.21 | 124.57 | 123.82 | 120.58 |
| MTP2 | 162.90 | 153.17 | 148.31 | 141.80 |
| MTP4 | 178.34 | 170.91 | 167.85 | 148.35 |

### Decode at p8192 (client post-first tok/s)

| Mode | g32 | g128 | g256 | g512 |
|---|---:|---:|---:|---:|
| No spec | 85.92 | 90.34 | 90.91 | 91.26 |
| MTP1 | 108.41 | 118.41 | 118.49 | 117.45 |
| MTP2 | 143.95 | 145.43 | 143.82 | 135.61 |
| MTP4 | 156.28 | 164.36 | 163.89 | 138.03 |

### Matched historical control (p9445/g128)

| Mode | Client post-first median (tok/s) |
|---|---:|
| No spec | 89.68 |
| MTP1 | 116.85 |
| MTP2 | 142.02 |
| MTP4 | 160.42 |

### Full-context decode

| Mode | p130944/g128 (tok/s) | MTP accept | p130560/g512 (tok/s) | MTP accept |
|---|---:|---:|---:|---:|
| No spec | 57.35 | n/a | 57.14 | n/a |
| MTP1 | 84.88 | 89.22% | 82.74 | 85.32% |
| MTP2 | 101.64 | 85.81% | 94.01 | 76.45% |
| MTP4 | 93.53 | 66.91% | 93.83 | 59.81% |

Input rate includes request scheduling and first-token work; it is not llama-bench engine-native `pp`. Decode is client-observed and not engine-native vLLM throughput. Exact output rows use the requested completion length; exclusions and replacements remain in `summary.json`.
