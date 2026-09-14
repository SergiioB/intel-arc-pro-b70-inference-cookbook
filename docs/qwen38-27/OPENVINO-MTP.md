# Qwen3.8-27B INT8 OpenVINO MTP — 3.29x decode on one B70 (2026-09-14)

Follow-up to [OPENVINO-CASCADIA-REPORT.md](OPENVINO-CASCADIA-REPORT.md).
Same artifact (`OpenVINO/Qwen3.8-27B-int8-ov`), GenAI **2026.5.0.0-3412** nightly,
one B70 (GPU.1), 230 W cap, greedy, ChatML rendered manually, n=1 smoke ladder.
Reproduces the external ~61 tok/s "latest image" claim exactly.

![MTP ladder](../assets/b70-qwen38-ov-mtp-ladder.svg)

## Numbers

| Config | wall tok/s | engine tps | TTFT ms | speedup |
|---|---:|---:|---:|---:|
| no MTP (VLMPipeline) | 18.72 | 18.83 | 112 | 1.00x |
| MTP depth 2 | 42.81 | 43.66 | 109 | 2.29x |
| MTP depth 3 | 52.67 | 54.01 | 108 | 2.81x |
| MTP depth 4 | 61.56 | 64.03 | 136 | 3.29x |

- Output text byte-identical across all depths and vs no-MTP (greedy parity holds).
- Prose prompt (explain speculative decoding), MTP3: 36.43 vs 18.68 base (1.95x).
  Counting tasks accept drafted tokens better than prose — pick depth by workload.
- 384-token request: 54.0 tok/s (EOS at 234 tokens).
- VRAM after load: 25.8 / 32 GB (weights + draft). Load ~57-61 s, cache warm after first.
- Raw JSON: `results/qwen38-ov-mtp-20260914/` on the lab host (probe5/probe6).

## The five blockers and their fixes

1. **Wrong pipeline class.** The Hub int8-ov export is
   `Qwen3_5ForConditionalGeneration` — a VL split export. The language graph
   takes `inputs_embeds` only (no `input_ids` port), so `LLMPipeline.generate()`
   dies with `Port for tensor name input_ids was not found`.
   Fix: `og.VLMPipeline(model_dir, 'GPU.1')`. Text-only prompts work fine.
2. **Minja chat-template bug.** Nightly 2026.5 cannot compile this model's
   template (`Unknown type for 'is' operator ... enable_thinking is undefined`).
   Fix: render ChatML yourself and set `apply_chat_template=False`.
3. **MTP silently needs two switches.** `GenerationConfig.num_assistant_tokens > 0`
   (mtp_strategy.cpp:28) **and** `SchedulerConfig.enable_prefix_caching = false`
   — the linear-attention verifier rejects prefix caching (pipeline_impl.cpp:388).
   VLMPipeline defaults prefix caching on, so MTP fails until you pass a scheduler
   config that turns it off.
4. **Do not set `cache_size` yourself.** With 25.8 GB of weights resident the
   KV budget check dies at load (cache_orchestrator.hpp:868). Defaults fit.
5. **Python 3.14 forkserver** re-imports your script as `__mp_main__`; open with
   `if __name__ != '__main__': raise SystemExit(0)` or the child reruns the probe.

## Working snippet

```python
import openvino as ov, openvino_genai as og
sc = og.SchedulerConfig(); sc.max_num_seqs = 1; sc.enable_prefix_caching = False
draft = og.draft_model(MODEL_DIR, 'GPU.1')          # bundled openvino_mtp_model.xml
pipe = og.VLMPipeline(MODEL_DIR, 'GPU.1', draft_model=draft, scheduler_config=sc)
cfg = og.GenerationConfig()
cfg.apply_chat_template = False                       # render ChatML yourself
cfg.num_assistant_tokens = 4                          # MTP depth
```

## Next targets

1. Export an MTP draft unit for `SergiioB/Qwen3.8-27B-int4-gdn8-ov` (the fixed
   INT4) — INT4 weights + MTP is the real size/speed target. No draft unit exists yet.
2. Cross-GPU draft: `og.draft_model(dir, 'GPU.0')` + main on GPU.1.
3. `num_assistant_tokens_schedule` and depth 5-6 ceiling on prose workloads.
