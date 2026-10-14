---
title: "Qwen3.8-27B OpenVINO INT4-GDN8 — how it was fixed (2026-09-12)"
type: research
date: 2026-09-12
status: final
---

# Qwen3.8-27B OpenVINO INT4-GDN8 — how it was fixed (2026-09-12)

Private evidence. Public IR: https://huggingface.co/SergiioB/Qwen3.8-27B-int4-gdn8-ov
GitHub: https://github.com/openvinotoolkit/openvino.genai/issues/4467

## What was broken

Hub `OpenVINO/Qwen3.8-27B-int4-ov` (experimental):

- `ignored_scope: null`
- INT4_ASYM `ratio=1.0` on the whole language graph, including GatedDeltaNet / `linear_attn`
- Same garbage on GPU.0 and CPU (`")-+&-&)&/-,'/!&*%!'.33)...`)
- Dummy image → token id 0 (`!`)
- Fresh JIT cache, `DYNAMIC_QUANTIZATION_GROUP_SIZE=0`, official kwargs: no change
- Class of [openvino.genai#3870](https://github.com/openvinotoolkit/openvino.genai/issues/3870) (optimum-intel dropped ignored scopes)

Flags cannot fix a collapsed logit tensor.

## What did not work

1. nncf INT8→INT4 with `.*linear_attn.*` ignored: **0 INT4 layers** (already quantized)
2. Cascadia stock `ov-genai` 0.2.3: SIGSEGV, bundled OpenVINO 2026.2 vs IR needing 2026.4+
3. Cascadia `--ov-*` on `qwen35`: ignored by design
4. Rebuilding Cascadia vs 2026.3.1: `ov-genai` still SIGSEGV; `qwen35` works but copies f32 logits to host (~3× slower)

## Conversion that works

Need ~52 GB BF16 + ~48 GB FP16 IR peak + 21 GB INT4 out. `/mnt/<models-disk>` 24 GB free was not enough for the FP16 save (`basic_ios::clear`). Export output went to `/mnt/<models-disk>`.

```text
1. Transformers 5.2.0 (5.4.0 refuses the exporter)
2. optimum main_export Qwen/Qwen3.8-27B → FP16 VLM IR (language.bin ~48 GB)
3. nncf.compress_weights(
     language_model,
     mode=INT4_ASYM, group_size=128, ratio=1.0,
     ignored_scope=IgnoredScope(patterns=[".*linear_attn.*"]),
   )
   → 73% INT4 / 22% float GDN / 1 INT8 layer, language.bin 21 GB
4. convert_tokenizer --with-detokenizer
5. VLMPipeline GPU.0 LATENCY f16, generation_config= keyword
```

Scripts on B70: `/mnt/<models-disk>/ov-vs-cascadia/export_int4_gdn8.py`, `compress_lang_int4.py`.

## N5 OpenVINO (150 W, LATENCY f16, greedy, ai2d image) — superseded if 230 W exists

| P | G | n | median wall tok/s | W | Tmax | quality |
|---|---|---|---|---|---|---|
| 128 | 128 | 5 | 13.76 | 150 | 70 | PASS |
| 512 | 128 | 5 | 14.65 | 150 | 73 | PASS |
| 1024 | 128 | 5 | 13.90 | 150 | 74 | PASS |
| 512 | 64 | 5 | 11.34 | 150 | 74 | PASS |
| 512 | 256 | 5 | 19.14 | 150 | 74 | PASS |

G64 is slower wall tok/s because TTFT is amortized over fewer tokens. G256 looks faster for the same reason. Headline catalog row should stay **P512/G128 n=5 median**.

150 W was the cap. Dense 27B should also be measured at **230 W**.
