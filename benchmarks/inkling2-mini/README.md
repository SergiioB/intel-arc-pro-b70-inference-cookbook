# Inkling 2 Mini benchmark slice

Scripts for the Inkling 2 Mini routes: [vLLM XPU primary recipe](../../docs/inkling2-mini/INKLING2-MINI-VLLM-XPU.md), [TensorFold XPU research build](../../docs/inkling2-mini/INKLING2-MINI-TENSORFOLD.md), and [llama.cpp SYCL recipe](../../docs/inkling2-mini/INKLING2-MINI-LLAMACPP.md).

| File | Does | GPU |
|---|---|---|
| `launch-inkling2-mini-vllm-xpu.sh` | Serves the unmodified native checkpoint on one B70 with vLLM XPU + `vllm-inkling2-xpu` (supports 90K daily, 200K max-ctx, and optional DSpark profiles) | yes |
| `prepare-llamacpp.sh` | Builds the exact source tree (upstream base + 2 PR merges + local patches 0001–0006) and checks both git tree hashes | no |
| `build-llamacpp-sycl.sh` | SYCL build, ahead-of-time for `bmg_g31` | no |
| `launch-inkling2-mini-llamacpp.sh` | Serves the exact default on one card | yes |
| `build-tensorfold-xpu.sh` | Clones the TensorFold tag `b70-inkling2-mini-20261121`, builds `xpu-inkling2`, and checks `.text` / `.rodata` / `.data` against the measured binary | no |
| `launch-inkling2-mini-tensorfold.sh` | Serves the TensorFold research build on one card | yes |
| `summarize.py` | Compiles either raw folder into `summary.json` (stdlib only) | no |

## Raw files

All in [`results/inkling2-mini/llamacpp-sycl-20261121/`](../../results/inkling2-mini/llamacpp-sycl-20261121/README.md):

| Published number | Raw file |
|---|---|
| Decode, prompt, TTFT, watts, J/token, patch stack (1,024 tokens) | `T1-ctlA-20261121T102722Z/depth/cells.jsonl`, `T1-cand-20261121T103748Z/depth/cells.jsonl`, `T1-ctlB-20261121T104708Z/depth/cells.jsonl` |
| Decode, graph replay off/on (2,048 tokens, three depths) | `ctl1-graphoff-20261121T024921Z/depth/cells.jsonl`, `cand1-graphon-20261121T032935Z/depth/cells.jsonl`, `ctl2-graphoff-20261121T040547Z/depth/cells.jsonl` |
| Greedy exactness (sha256 of 5 outputs) | `exact.jsonl` in `ctl1-graphoff-*`, `cand1-graphon-*`, `T1-cand-*`, `T1-cand-nop2-*` |
| KL of patch 0002 | `Q1-kl-20261121T120248Z/kl-loc.txt`, `kl-loc-nop2.txt` |
| Canary and effort sweep | `Q1-effort-canary-20261121T120543Z/*.jsonl` |
| Perplexity | `P1-profile/ppl-wikitext2-c512-40.txt` (0001), `VL1-verify-local/ppl-local.txt` (0001–0006, 0002 active) |
| Kernel tests vs CPU | `VL1-verify-local/tbo-*.txt` |
| Scheduler splits 194 → 2 | `S1-env-sweep/server-log-excerpt.txt`, `P2-vtune/splits.txt` |
| Closed arms DNR-034, DNR-036, DNR-037 | `inkling-dspark-*`, `inkling-base-*/server-log-excerpt.txt`, `S2-env-sweep-local/*.jsonl` |
| VRAM at `-c 49152` | `T1-cand-20261121T103748Z/mem.jsonl`, `cand1-graphon-20261121T032935Z/mem.jsonl` |
| Byte ceiling (computed) | `ceiling/inkling-udq4kxl.out` |

TensorFold XPU research build, in [`results/inkling2-mini/tensorfold-xpu-20261121/`](../../results/inkling2-mini/tensorfold-xpu-20261121/README.md):

| Published number | Raw file |
|---|---|
| 71.48 / 64.05 tok/s decode, TTFT, J/token (tagged build) | `check4-20261121T134525Z/cell-after/cells.jsonl` |
| 39.92 / 33.47 tok/s, same binary with older settings | `check4-20261121T134525Z/cell-before/cells.jsonl` |
| 55.83 tok/s over 8,192-token replies (older build) | `speed3-20261121T133350Z/depth/cells.jsonl` |
| Per-pass error 9.82e-4, 24/24 tokens | `check4-20261121T134525Z/cmp-v2.txt`, `check-v2.log.txt` |
| Greedy 1 of 5 equal to llama.cpp | `speed3-20261121T133350Z/exact-vs-llamacpp.txt`, `tmp/diag/margins.log.txt` |
| Tag rebuild = measured binary | `binary-identity.txt` |
