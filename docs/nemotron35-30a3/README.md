# Nemotron-3.5-Lightning-30B-A3B on the B70

Family index. Keep Qwen and Muse numbers on their own pages.

| Path | What it is | Status |
|---|---|---|
| [NEMOTRON-DFLASH-B70.md](NEMOTRON-DFLASH-B70.md) | Working speculative recipe (`method=dflash` n=7) | Isolated n=5 |
| [NEMOTRON-B70.md](NEMOTRON-B70.md) | No-spec XPU-graph floor | n=5 decode; graph determinism caveat |
| [CLAIMS.md](CLAIMS.md) | Observed numbers only — copy from here | Source of truth for this family |
| `benchmarks/nemotron35-30a3/launch-nemotron-dflash.sh` | DFlash launcher | Public digest + runtime patches |
| `benchmarks/nemotron35-30a3/launch-nemotron-graph.sh` | No-spec launcher | Same digest; no spec config |

## Artifacts (do not rename)

Published under **`SergiioB`** (two i's). Keep these ids: they already
appear on LocalMaxxing notes, HF cards, and the isolated n=5 evidence.

- Target: https://huggingface.co/SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym
- Draft: https://huggingface.co/SergiioB/Nemotron-3.5-Lightning-30B-A3B-DFlash-BF16

Renaming would break every link and every card that already says “local
conversion.” The names already encode format (`GPTQ-INT4-G64-sym`,
`DFlash-BF16`). That is enough.

## Hard rules for this family

1. Different public digest than Qwen3.6 Pi. See
   [IMAGE-AND-PATCH-MATRIX.md](../IMAGE-AND-PATCH-MATRIX.md).
2. Native vLLM MTP is **0%** acceptance. Do not advertise MTP.
3. Representative decode is **186.61** at **p2048/g128 n=5**, not p512 194.6.
4. Lane 1 input is **7160** at **p8192/g1 n=5**. The ~10.3k figure is a
   no-spec n=3 TTFT on a **g128** cell.
5. LocalMaxxing `cmsr9po4w000ams01e4fc5qhj` is APPROVED.
6. [vllm#52159](https://github.com/vllm-project/vllm/pull/52159) closed
   2026-09-02 without merge. The DFlash launcher still applies the Python
   router patch on the pinned `0.1.12.3` image. Kernel `at::zeros` is
   optional (graph-replay determinism) via `0001` /
   [vllm-xpu-kernels#524](https://github.com/vllm-project/vllm-xpu-kernels/pull/524)
   (still open). Do not apply `0002`; that tuple is in kernels `main`
   ([#526](https://github.com/vllm-project/vllm-xpu-kernels/pull/526)).
