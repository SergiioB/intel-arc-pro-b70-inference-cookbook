# Nemotron 3.5 Bolt 74B-A9B on one B70

Family index. This is a different model and engine from
[Nemotron-3.5-Lightning-30B-A3B](../nemotron35-30a3/README.md) (vLLM XPU).
Do not mix images, patches or numbers between the two families.

| Route | Hardware | Start here |
|---|---|---|
| llama.cpp SYCL, NVIDIA 24GB GGUF + mmproj, no drafter | One B70, C1, 64K context | [NEMOTRON-BOLT-B70-LLAMACPP.md](NEMOTRON-BOLT-B70-LLAMACPP.md) |

| Item | Authority |
|---|---|
| Image and patch | [IMAGE-AND-PATCH-MATRIX.md](../IMAGE-AND-PATCH-MATRIX.md), row "Nemotron 3.5 Bolt 74B-A9B" |
| Launcher | [`benchmarks/nemotron35-bolt-74b/launch-nemotron-bolt-llamacpp.sh`](../../benchmarks/nemotron35-bolt-74b/launch-nemotron-bolt-llamacpp.sh) |
| Public numbers | record `nemotron35-bolt-74b-llamacpp-c1-v1` in [`data/benchmarks.v1.json`](../../data/benchmarks.v1.json) |
| Evidence | [`results/nemotron35-bolt-74b/llamacpp-c1-20261015-summary.json`](../../results/nemotron35-bolt-74b/llamacpp-c1-20261015-summary.json) |

## Claim boundary

- Artifact: `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF`, the 20.5 GiB
  GGUF plus the BF16 mmproj. Other quantizations of this model are not covered.
- Engine: the patched image `ghcr.io/sergiiob/llama.cpp-sycl-b70:b11669-bolt`.
  The upstream `server-intel-b11669` image loads the model but lacks the
  fixes and was not measured. Numbers from it are not this route.
- One card, one request at a time (`-np 1`), 65,536-token context. No
  concurrency, multi-card or longer-context claim.
- Evidence tier E2: self-reported, one host, one card, not independently
  reproduced. The quality evidence is a 6-case greedy smoke and a KLD screen
  against the stock build. That does not establish general quality.
