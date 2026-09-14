# OpenVINO GenAI on Arc Pro B70 — tuning guide (2026-09-14)

Everything below is measured on B70 hardware with OpenVINO 2026.5.0-22967 /
GenAI 2026.5.0.0-3412 nightly, Qwen3.8-27B (int8-ov VL split export), unless
marked otherwise. Model-specific numbers live in
[OPENVINO-MTP.md](OPENVINO-MTP.md); this page is the how-to.

## 1. Pipeline selection: read the export layout first

The IR's input ports decide the class, not the model name:

| Export layout | Language-graph input | Class that works |
|---|---|---|
| Standard LLM (`openvino_model.xml`) | `input_ids` | `og.LLMPipeline` |
| VL split (`openvino_language_model.xml` + `openvino_text_embeddings_model.xml` + vision units) | `inputs_embeds` [-1,-1,hidden] | `og.VLMPipeline` |

Hub `Qwen3.8-27B-int8-ov` is a VL split (`Qwen3_5ForConditionalGeneration`)
even though most users only run text. `LLMPipeline.generate()` then dies with
`Port for tensor name input_ids was not found`. `VLMPipeline` handles text-only
prompts fine — pass strings, not image inputs.

Check before loading: does the dir contain `openvino_text_embeddings_model.xml`?
If yes → VLMPipeline. `config.json` `architectures: *ForConditionalGeneration`
is the other tell.

## 2. Chat template: bypass Minja on 2026.5 nightly

The nightly's Minja engine fails to compile templates using
`X is undefined` Jinja tests (`enable_thinking is undefined`) with
`Unknown type for 'is' operator`. Two options:

1. Render the template yourself (ChatML for Qwen3.x) and set
   `cfg.apply_chat_template = False` — **recommended, deterministic**.
2. `pipe.get_tokenizer().set_chat_template()` with a simplified template —
   only if you must use the built-in path.

## 3. MTP speculative decoding: the required switch set

All four or it fails silently-to-loudly:

```python
sc = og.SchedulerConfig()
sc.max_num_seqs = 1
sc.enable_prefix_caching = False   # hard requirement: linear-attention verifier
props = {}                         # see §4 for KV precision
draft = og.draft_model(MODEL_DIR, 'GPU.1', **props)
pipe = og.VLMPipeline(MODEL_DIR, 'GPU.1', draft_model=draft,
                      scheduler_config=sc, **props)
cfg = og.GenerationConfig()
cfg.apply_chat_template = False
cfg.num_assistant_tokens = 5        # static count ONLY on 2026.5
cfg.do_sample = False               # MTP path is greedy-verified
```

- `num_assistant_tokens > 0` or mtp_strategy asserts at load.
- `enable_prefix_caching = False` or pipeline_impl.cpp rejects the combination
  (GatedDeltaNet/linear-attention verifier; by design, no upstream fix).
- The old `num_assistant_tokens_schedule` constant/heuristic/dynamic triad is
  **gone** in 2026.5 — static counts only; `assistant_confidence_threshold` is
  rejected for MTP.
- Draft unit must exist: `openvino_mtp_model.xml` in the model dir. If absent
  you must export one (none existed for our INT4-GDN8).
- Verify window scheduling is atomic: keep `max_num_batched_tokens` >= nat+1
  or windows get deferred (default unlimited via LLMPipeline is fine).

## 4. KV cache precision — the property is real, wire it like this

`KV_CACHE_PRECISION` is a GPU-plugin property (values `f16`, `u8`/`i8`,
`u4`/`i4`; u↔i normalized at compile). Two hard facts:

- **Default on XMX parts is f16** (matches inference precision) — your KV is
  NOT compressed unless you ask (upstream commit ecdecf3 changed this from
  the older INT8-by-default behavior; old blog text saying "INT8 by default"
  is outdated).
- It is passed as a **pipeline property dict**, not a SchedulerConfig field:

```python
# u8 on the MAIN model only. Do NOT pass the prop to draft_model() —
# a u8 draft KV corrupts long-context MTP (>32K -> empty decode).
draft = og.draft_model(MODEL_DIR, 'GPU.1')
pipe = og.VLMPipeline(MODEL_DIR, 'GPU.1', draft_model=draft,
                      scheduler_config=sc,
                      KV_CACHE_PRECISION='u8')
```

Notes: INT4 (`u4`) upstream requires the Paged Attention backend for by-channel
key quantization; on SDPA it falls back to by-token keys (worse accuracy).
Measure, don't assume — see §7 for what u8 did to our ceiling and speed.

## 5. Scheduler budget rules (or: how to OOM at load)

- **Never set `SchedulerConfig.cache_size` when weights are ~26 GB.** The
  budget check (`budget_in_bytes <= total_available_memory`,
  cache_orchestrator.hpp) fails the whole load. Defaults fit.
- `max_num_seqs=1` for latency benchmarks; raise only for throughput work.
- `num_linear_attention_blocks` must cover live rows + speculative window;
  undersizing silently defers verify windows, then drops requests.

## 6. Python 3.14 forkserver guard

Scripts touching multiprocessing get re-imported as `__mp_main__` by the
forkserver. Without a guard the child re-runs your probe (or trips
assert-never-overwrite). First lines of every script:

```python
if __name__ != '__main__':
    raise SystemExit(0)
```

## 7. What each knob did on Qwen3.8-27B int8-ov, 1× B70, 230 W, greedy

| Knob | Result |
|---|---|
| MTP off → nat5 | 18.72 → 68.36 tok/s wall (n=5: 67.91/71.01 post-first), TTFT ~96 ms |
| nat 8 | 42.83 — rejections dominate; nat5 is the sweet spot on this model |
| f16 KV ceiling | clean through 49152 input (916 t/s prefill, 30.2 decode); CL_EXEC crash at 65536 |
| KV u8 ≤32K | identical speed to f16 (decode 41.3→41.8, prefill 1119→1117), coherent — free memory win |
| KV u8 64K–128K + MTP (u8 on BOTH main and draft) | prefill WORKS where f16 crashed (775→465 t/s), but decode emits ~1 empty token — broken |
| KV u8 64K no MTP | prefill 828 t/s, decode 15.15 t/s, 128 tokens coherent — defect isolated to draft KV precision |
| **KV mixed: main=u8, draft=f16** | **THE FIX: 64K decode 25.0 tok/s with MTP nat5, 128 tokens coherent — u8 only on the main model, draft stays default f16** |
| Prefix caching | must stay off with MTP (linear-attn verifier); ON is fine without MTP |
| Cross-GPU draft (GPU.0 draft / GPU.1 main) | OpenCL runtime abort (`enqueue_svm.h:308`) — driver-level, do not retry as-is |

## 7b. Long-context recipe (which config per length, 1× B70 32GB)

| Context | Config | Decode |
|---|---|---|
| ≤32K | MTP nat5, main KV u8 or f16, draft f16 | 41.8 tok/s @32K |
| 32–48K | MTP nat5 + f16 KV (both) | 30.2 tok/s @48K |
| 48–~80K | **MTP nat5, main=u8 draft=f16 (mixed)** | **25.0 tok/s @64K** |
| >~80K | NO MTP + u8 KV everywhere | 15.2 tok/s @64K-class; prefill reaches 128K |

**The mixed-KV rule:** when passing `KV_CACHE_PRECISION='u8'`, pass it to the
MAIN pipeline only — `og.draft_model(MODEL, 'GPU.1')` with NO precision prop.
u8 on the draft's own KV corrupts its attention at >32K (garbage drafts ->
verifier rejects -> `gen_tokens==1`, empty text). With draft at f16 the main
model still gets the full u8 memory saving; 96K then dies on true VRAM
exhaustion (CL_OUT_OF_RESOURCES), not on the defect — mixed config ceiling is
~64-80K on one 32GB card.

File-able upstream signature (u8-on-draft defect): genai 2026.5.0.0-3412,
Qwen3.8-27B-int8-ov, u8 KV on draft+main, nat5, >32K input, TTFT normal,
`gen_tokens==1`, empty text; same prompt with draft at f16 decodes normally.

## 8. Failure fingerprints (what the error means)

| Error | Meaning | Fix |
|---|---|---|
| `Port for tensor name input_ids was not found` | VL split export through LLMPipeline | use VLMPipeline (§1) |
| `Unknown type for 'is' operator` | Minja can't compile template | manual render + apply_chat_template=False (§2) |
| `num_assistant_tokens > 0` assert | MTP depth never set | cfg.num_assistant_tokens (§3) |
| `enable_prefix_caching=false` assert | prefix cache on with linear-attn MTP | scheduler flag (§3) |
| `budget_in_bytes <= total_available_memory` | cache_size override too big | drop the override (§5) |
| `Option not found: ATTENTION_BACKEND` | not a pipeline property on this build | ignore; default backend hits 68 t/s |
| `clWaitForEvents ... -14 CL_EXEC_STATUS_ERROR` | out of GPU resources at this context | lower context or compress KV (§4) |
| u8 KV + MTP >32K: TTFT ok, `gen_tokens==1`, empty text | MTP verify × u8 KV interaction defect | drop MTP or stay f16 (§7b) |

## 9. Power and measurement discipline

- Default cap 150 W; raise to 230 W only during measurement; **always restore
  in `finally`** (a driver abort skips finally — check the cap after crashes).
- Watts: `energy1_input` (µJ counter) delta / wall time = true mean draw.
  `power1_average` does not exist on B70 hwmon. Our runs: 202–211 W mean at
  the 230 W cap — the card does not saturate the cap at C1.
- Preflight before every load: `visible_avail >= 31000 MiB` on the target GPU,
  no `docker ps` containers, package temp <= 65 C.

## 10. Cross-GPU split: what died

`og.draft_model(dir, 'GPU.0')` + `VLMPipeline(dir, 'GPU.1', draft_model=...)`
aborts inside the OpenCL compute runtime (`enqueue_svm.h:308`) before the
first generate. Not a Python/API issue — driver level. If retrying later,
try (a) both devices in one L0 context via ONEAPI_DEVICE_SELECTOR, (b) newer
compute-runtime, (c) file upstream with the abort signature.
