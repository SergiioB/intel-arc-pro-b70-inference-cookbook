# Nemotron 3.5 Bolt 74B-A9B on B70

Family index. This is a different model and engine from
[Nemotron-3.5-Lightning-30B-A3B](../nemotron35-30a3/README.md) (vLLM XPU).
Do not mix images, patches or numbers between the two families.

| Route | Hardware | Start here |
|---|---|---|
| llama.cpp SYCL, NVIDIA 24GB GGUF + mmproj, no drafter | One B70, C1, 64K default, up to 262K context | [NEMOTRON-BOLT-B70-LLAMACPP.md](NEMOTRON-BOLT-B70-LLAMACPP.md) |
| vLLM XPU TP2, AutoRound W4A16 int4 g128 | Two B70, TP2, both 180 W, 64K default, up to 262K context; C1 decode 60.5 tok/s at 886 prompt tokens (median of 5) | [VLLM-TP2-AUTOROUND-W4A16.md](VLLM-TP2-AUTOROUND-W4A16.md) |

What Bolt is good for compared with Qwen3.8-27B and Qwen3.6-35B-A3B on the same 94 tasks
(agentic work, tool use and long context strong; physics and strict JSON weaker; two cards for a
lower total than Qwen3.8-27B on one card): [MODEL-COMPARISON.md](../MODEL-COMPARISON.md).

The two routes use different weights, engines and patches. The claim boundary
below covers the llama.cpp route; the vLLM TP2 route has its own boundary,
launcher ([`launch-nemotron-bolt-vllm-tp2.sh`](../../benchmarks/nemotron35-bolt-74b/launch-nemotron-bolt-vllm-tp2.sh)),
record (`nemotron35-bolt-74b-vllm-tp2-c1-20261016`) and evidence in its recipe.
How its int4 weights were made and checked, step by step, is in
[AUTOROUND-W4A16-PROCESS.md](AUTOROUND-W4A16-PROCESS.md).

Both routes set model-card text sampling: temperature `1.0`, top_p `0.95`.
See the [llama.cpp sampling settings](NEMOTRON-BOLT-B70-LLAMACPP.md#3-sampling-and-thinking)
and [vLLM sampling settings](VLLM-TP2-AUTOROUND-W4A16.md#sampling-and-thinking)
for server defaults and client examples. Clients such as omp that omit sampling
settings rely on the server defaults.

| Item | Authority |
|---|---|
| Image and patch | [IMAGE-AND-PATCH-MATRIX.md](../IMAGE-AND-PATCH-MATRIX.md), row "Nemotron 3.5 Bolt 74B-A9B" |
| Launcher | [`benchmarks/nemotron35-bolt-74b/launch-nemotron-bolt-llamacpp.sh`](../../benchmarks/nemotron35-bolt-74b/launch-nemotron-bolt-llamacpp.sh) |
| Public numbers | record `nemotron35-bolt-74b-llamacpp-c1-v1` in [`data/benchmarks.v1.json`](../../data/benchmarks.v1.json) |
| Evidence | [`results/nemotron35-bolt-74b/llamacpp-c1-20261015-summary.json`](../../results/nemotron35-bolt-74b/llamacpp-c1-20261015-summary.json) |

## Claim boundary

- Artifact: `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF`, the 20.5 GiB
  GGUF plus the BF16 mmproj. Other quantizations of this model are not covered.
- Engine: the patched image
  `ghcr.io/sergiiob/llama.cpp-sycl-b70:b11669-bolt@sha256:c1b871ee87d2790e92605e89d8fa40c41a61afa128bdce55da0f9fba22e3d042`.
  The catalog numbers come from a native build of the same source on host
  compute-runtime 26.05 / IGC 2.28.4; the image (bundled 26.18 / 2.34.4)
  decodes about 6% slower on the same card, cause not isolated. The upstream
  `server-intel-b11669` image loads the model but lacks the fixes and was not
  measured. Numbers from it are not this route.
- One card, one request at a time (`-np 1`). Speed numbers use a 65,536-token
  context; 262,144 tokens (the GGUF maximum) loads in 25.8 GiB and found a
  needle in a 250K-token prompt. No concurrency or multi-card claim, and no
  long-context quality claim beyond that retrieval check.
- Evidence tier E2: self-reported, one host, one card, not independently
  reproduced. The quality evidence is a 6-case greedy smoke and a KLD screen
  against the stock build. That does not establish general quality.
