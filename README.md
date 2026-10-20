# Intel Arc Pro B70 Inference Cookbook

Pinned vLLM XPU and llama.cpp SYCL recipes, failure notes, and measurement records for Intel Arc Pro B60/B70. Start with [what worked, what did not, and what to test next](docs/WHAT-WORKED.md). For a first local run, use [START.md](START.md).

## Pick one path

| Your goal | Start here | Status |
|---|---|---|
| Choose a model for a job | [Model comparison](docs/MODEL-COMPARISON.md) | 94-task quality screen of Bolt 74B-A9B (two cards), Qwen3.8-27B and Qwen3.6-35B-A3B (one card each) with power and heat; one sample per task, Qwen3.6 partial. |
| Serve on one B70 | [Qwen3.8-27B EXL3 route](docs/qwen38-27b/EXL3-XPU.md) | Primary route: trellis 4bpw + MTP3 + fp8 KV on the exl3xpu engine; the only measured route at full 262,144 context and clean on the exact-answer battery. The GPTQ/W4A16 route is legacy — that class fails the canary on this model. |
| Run GGUF or vision | [Muse-Glimmer llama.cpp route](docs/muse-glimmer/MUSE-GLIMMER-B70.md) | Measured on its own engine and workload; no cross-engine speed ranking without matched timing. |
| Run MoE or long context | [Qwen3.6-35B-A3B recipe](docs/qwen36-35a3/QWEN36-MOE-VLLM-XPU.md) | Historical MTP speed record is stack-specific; a [later matched comparison](docs/qwen36-35a3/MOE-INT4-THREE-ENGINE-CEILING-20260917.md) did not reproduce it. |
| Serve on two B70s | [Independent-card setup](docs/DUAL-B70-TP2.md) | Default: one server per card. [FP8 TP2](docs/qwen38-27b/FP8-TP2-W8A16.md) is a research route, not an automatic single-request win. |
| Diagnose or reproduce a number | [Image and patch matrix](docs/IMAGE-AND-PATCH-MATRIX.md) → [family recipe](docs/WHAT-WORKED.md) → [benchmark catalog](docs/BENCHMARK-CATALOG.md) | Match image, patch, model, context, cache, power, and timing definition. |

**Read the [decision guide](docs/WHAT-WORKED.md) before tuning.** It separates measured improvements, failed attempts, unpublished lab screens, and stop/go gates. A passing smoke canary does not establish model quality. No public performance claim from a different model, engine, host, or result generation transfers to this setup.

## Reproduce and operate

1. Check your card and host with [`scripts/doctor.py`](scripts/doctor.py), then use the [quick start](START.md) for a pinned single-card run.
2. Check the [image and ordered patch matrix](docs/IMAGE-AND-PATCH-MATRIX.md) for the exact family before launching. Do not mix family patch stacks.
3. Connect a client using [client setup](docs/CONNECTING-CLIENTS.md); use the [reliability guide](docs/RELIABILITY-REPORT.md) if the server hangs.
4. Confirm a recipe with [the reproduction procedure](docs/CONFIRM-A-RECIPE.md) and [benchmark format](docs/BENCHMARK-FORMAT.md). The numeric authority is [`data/benchmarks.v1.json`](data/benchmarks.v1.json); [`docs/BENCHMARK-CATALOG.md`](docs/BENCHMARK-CATALOG.md) is generated.

More routes: [Nemotron DFlash](docs/nemotron35-30a3/README.md), [Nemotron Bolt 74B](docs/nemotron35-bolt-74b/README.md) (one card on llama.cpp, or [two cards on vLLM](docs/nemotron35-bolt-74b/VLLM-TP2-AUTOROUND-W4A16.md)), [Flash-Next](docs/qwen38-flash-next/README.md), [Ornith-1.5-122B GGUF](docs/ornith15-122b/README.md), [Windows](docs/qwen38-27b/WINDOWS-STANDALONE.md), and [family-specific research](docs/WHAT-WORKED.md). Research routes are not interchangeable with the first-run recipe.

Quantize a model yourself: [how the Bolt 74B AutoRound W4A16 quant was made and checked on two B70s](docs/nemotron35-bolt-74b/AUTOROUND-W4A16-PROCESS.md), with every command, fix and script.

## Request or add a recipe

Need a model or configuration not yet covered? Submit a [recipe request](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/issues/new?template=recipe-request.yml) with your target Hugging Face model and hardware setup.

To contribute a recipe directly to the B70 Launcher catalog, follow the [recipe format specification](https://github.com/SergiioB/b70-launcher/blob/main/docs/recipe-format.md) to edit `recipes.json`, verify with `python3 -m unittest tests.test_recipes`, and open a pull request.

## For contributors and agents

Use [`data/system-map.v1.json`](data/system-map.v1.json) to select an intent. [`docs/AGENT-SYSTEM.md`](docs/AGENT-SYSTEM.md) explains the authority layers; [`docs/ADDING-A-RECIPE.md`](docs/ADDING-A-RECIPE.md) explains publication. Keep historical evidence intact. Validate changes with `python3 scripts/check-repo.py`.

Code: MIT. Measurement reports and prose: CC BY 4.0. See [LICENSE](LICENSE).
