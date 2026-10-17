# Which model for which job: Bolt 74B-A9B vs Qwen3.8-27B vs Qwen3.6-35B-A3B

A quality screen on one set of 94 tasks, run on Intel Arc Pro B70 cards with each model's own
recipe. Use it to choose a model. It is not a leaderboard: one sample per task, 4 to 8 tasks per
area.

Per-task outcomes: [`results/2026/10/model-compare-v1-20261017/summary.json`](../results/2026/10/model-compare-v1-20261017/summary.json).
Catalog records in [`data/benchmarks.v1.json`](../data/benchmarks.v1.json):
`nemotron35-bolt-74b-vllm-tp2-quality-v1compare-20261017`,
`qwen38-27b-exl3-4bpw-quality-v1compare-20261017`,
`qwen36-35b-a3b-ovms-int4-quality-v1compare-20261017`.

## 1. Pick by job

| Job | Use | Why (from this screen) |
|---|---|---|
| Default local agent and coding on one card | **Qwen3.8-27B EXL3** ([recipe](qwen38-27b/EXL3-XPU.md)) | Highest total (81 of 94), 6 of 6 agentic repository tasks, needs only one card |
| Physics, numeric STEM, strict JSON output | **Qwen3.8-27B EXL3** | Physics 8 of 8 vs Bolt 5; structured output 8 of 8 vs Bolt 4 |
| Agent work when both cards are free, long documents, broad knowledge, images | **Bolt 74B-A9B** ([vLLM TP2](nemotron35-bolt-74b/VLLM-TP2-AUTOROUND-W4A16.md), or [llama.cpp one card](nemotron35-bolt-74b/NEMOTRON-BOLT-B70-LLAMACPP.md)) | 6 of 6 agentic, 8 of 8 tool use, 4 of 4 long context, best humanities score (6 of 8); the model accepts images and has a 262K context; finished the run fastest (52 min) |
| Low-cost question answering (chemistry, coding Q&A, professional knowledge) | **Qwen3.6-35B-A3B** (3B active per token) | 47 of 54 on the tasks it finished, level with the other two; **not** on OVMS under load (see section 4). Speed was not measured here |
| One-shot web pages, games, 3D | No clear choice | 0 of 4 for every model at a 32K output budget with thinking on. Give a larger budget or turn thinking off |

Plain verdict: **Bolt does not beat the smaller models on this set.** Qwen3.8-27B on one card
matches or beats Bolt on two cards in 11 of 13 areas. Bolt earns its second card when you need
its 262K context, image input, or a second opinion from a different model family, not for
higher scores.

## 2. Results

Same 94 tasks for every model: coding (HumanEval and bug-fix tasks, run against tests), 6 agentic
repository tasks driven by the omp agent and graded by hidden tests, 4 web/3D pages checked in a
headless browser, physics, chemistry/biology (GPQA Diamond, MMLU-Pro), math (MATH-500),
humanities and professional knowledge (MMLU-Pro), instruction following (IFEval), tool calls,
JSON structured output, five languages, and long-context retrieval at about 16K and 64K tokens.

Tasks passed (one sample per task):

| Area (tasks) | Bolt 74B-A9B W4A16 | Qwen3.8-27B EXL3 | Qwen3.6-35B-A3B (finished) |
|---|---:|---:|---:|
| Agentic coding (6) | 6 | 6 | not scored |
| Coding (8) | **7** | 6 | **7** |
| Physics (8) | 5 | **8** | not scored |
| Chemistry / biology (8) | 5 | 5 | **6** |
| Math (8) | 8 | 8 | 6 of 6 |
| Humanities knowledge (8) | **6** | 5 | 5 |
| Professional knowledge (8) | 6 | **7** | **7** |
| Instruction following (8) | 8 | 8 | 8 |
| Tool use (8) | 8 | 8 | not scored |
| Structured output (8) | 4 | **8** | not scored |
| Multilingual (8) | 8 | 8 | 8 |
| Long context (4) | 4 | 4 | not scored |
| Web / 3D pages (4) | 0 | 0 | not scored |
| **Total** | **75 of 94** | **81 of 94** | **47 of 54** |

On the 54 tasks all three finished: Qwen3.6 47, Bolt 46, Qwen3.8 45. Differences of one or two
tasks are noise at this size.

Web/3D detail: Bolt's four pages loaded but threw JavaScript errors. Qwen3.8 hit the 32,768-token
limit while still thinking on all four; with a 65,536-token limit it passed 1 of 4 and still ran
out on 2.

## 3. Cost, power and heat

All runs: both cards capped at 180 W (`power1_cap`), energy from the `energy1_input` card counter
sampled once a second, average watts = energy delta / elapsed time.

| Model | Cards | Run time | Average draw | Energy | Per passed task | Hottest package / VRAM |
|---|---:|---:|---|---:|---:|---|
| Bolt 74B-A9B | 2 | 51.6 min | 142 W + 167 W | 266 Wh | 3.5 Wh | 77 °C / 86 °C |
| Qwen3.8-27B | 1 | 56.7 min | 179 W | 169 Wh | 2.1 Wh | 77 °C / 86 °C |
| Qwen3.6-35B-A3B | 1 | 57.0 min (to last finished task) | 157 W | 149 Wh | 3.2 Wh | 77 °C / 76 °C |

Idle with a model loaded: 45-49 W per card. Qwen3.8 keeps its card at the 180 W cap for the
whole run; Bolt spreads the load over two cards and finished first, but used 1.6 times the energy
of Qwen3.8 for fewer passed tasks. Run times compare recipes at their own concurrency (Bolt and
Qwen3.8 at 8 parallel requests, Qwen3.6 at 4), not model speed.

## 4. What went wrong

- **Qwen3.6-35B-A3B on OVMS 2026.4.1 crashed twice** with the same OpenVINO GenAI error when its
  KV cache filled: `Check 'm_block_table.count(seq_id) > 0' failed ... block_manager.hpp:759
  Invalid sequence group`, then a segfault. First at 8 parallel requests after 10 minutes, then
  at 4 after 54 finished tasks. The other 40 tasks (agentic, physics, tool use, structured output,
  long context, web) have no result for this model. For long thinking runs or several clients,
  use the [vLLM recipe](qwen36-35a3/QWEN36-MOE-VLLM-XPU.md) (its artifact is an uncensored
  derivative, not Qwen's release) or a GGUF of the official weights on llama.cpp instead.
- One Bolt web task was graded twice: the browser checker failed while cleaning up its temporary
  profile. After the fix the page failed on its own JavaScript error.

## 5. Exact setup

| Model | Engine and image | Weights | Context, parallel requests |
|---|---|---|---|
| Bolt 74B-A9B | vLLM 0.31.0 XPU `vllm/vllm-openai-xpu@sha256:95ac815038c4b3537b173798bfe3b89bdf525a52bd653cd14d4deb01f1062be7`, TP2, fp8 KV, graphs on, prefix cache on | `SergiioB/Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym` | 131,072 and 8 (the recipe defaults are 65,536 and 4) |
| Qwen3.8-27B | exl3xpu `ghcr.io/0xsero/exl3xpu:0.1.0@sha256:21412bdd7535e9c653eeb3d099dce3bc79a83d440def2cd9556111c79c870fa8`, MTP3, fp8 KV, int8 prefill, prefix cache on | EXL3 4.00bpw of `Qwen/Qwen3.8-27B` (see [EXL3-XPU.md](qwen38-27b/EXL3-XPU.md)) | 98,304 and 8 |
| Qwen3.6-35B-A3B | `openvino/model_server:2026.4.1-gpu@sha256:a928903fe429f1e3e27107cc45932d30d95febfd2d1159dc50869c3eac76fc58`, prefix cache on, qwen3 reasoning and qwen3coder tool parsers | `OpenVINO/Qwen3.6-35B-A3B-int4-ov` | 131,072 and 4 |

Sampling: each model card's thinking settings (temperature 1.0, top_p 0.95; Qwen also top_k 20,
min_p 0, presence_penalty 0 for Qwen3.8 and 1.5 for Qwen3.6), fixed seed per task, thinking on.
Output limit 16K or 32K tokens per task depending on area; agentic tasks 15 minutes each.

## 6. Claim boundary

- Quality screen, evidence tier E2: self-reported, one host, per-task outcomes published, model
  answers and the eval harness not published, not independently reproduced.
- One sample per task at temperature 1.0. A second Bolt run on 87 of the same tasks changed the
  pass/fail result on 6 of them, so treat one- or two-task gaps as noise.
- Engines differ per model, so run time and energy compare these recipes, not the models.
- Qwen3.6-35B-A3B covers 54 tasks in 7 areas only.
- Not tested: thinking off, more than one sample, other quantizations, other engines for
  Qwen3.6, power caps other than 180 W.
- Public benchmarks (GPQA, MMLU-Pro, MATH-500, HumanEval, IFEval) may be in the models' training
  data.
