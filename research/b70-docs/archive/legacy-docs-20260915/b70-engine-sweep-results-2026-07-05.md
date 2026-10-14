# B70 Engine Sweep Results - 2026-07-05

Host: Intel Arc Pro B70 32GB, Ubuntu 26.04, llama.cpp SYCL/Level-Zero b9851.

## Engine Status

| Engine | Status | Notes |
|---|---|---|
| llama.cpp SYCL/Level-Zero | Baseline complete | All runs used `-fa on -ctk q5_0 -ctv q4_1 -ngl 99 -ncmoe 0 -t 8 --no-mmap`. |
| Intel LLM Scaler vLLM | Scout in progress | Docker image `intel/llm-scaler-vllm:0.14.0-b8.3.2` works for XPU. Qwen 27B AutoRound requires `--dtype float16` to avoid the BF16/Half GDN attention crash. vLLM does not expose llama.cpp `q5_0/q4_1` KV; scout runs used vLLM FP8 KV. |
| SGLang XPU | Not installed yet | Official XPU path is source/Python 3.12 oriented and should be tested after vLLM blockers are pinned. |
| Megakernels/custom kernels | Research track | Not a separate serving engine here. For B70 this means kernel paths inside vLLM/SGLang/IPEX/Triton XPU: graph capture, fused linear/attention/MoE, TurboQuant-style kernels, or custom Triton XPU ops. |

## llama.cpp Power and Temperature Sweep

Source result file: `b70-bench-results/engine-sweeps/20260705-164340/results.jsonl`.

All rows are averages of measured non-warmup rounds. Temperature is average round start, peak observed end temperature, and average round end temperature.

| Candidate | Cap | Context | Decode tg/s | Prefill t/s | TTFT ms | Card W | Temp start->max->end | VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| `gemma4-26b-q4-256k` | 150W | 256K | 55.61 | 133.4 | 235 | 130.9 | 63->65->64C | 19.55 |
| `gemma4-26b-q4-256k` | 165W | 256K | 55.52 | 132.5 | 236 | 130.7 | 63->65->64C | 19.56 |
| `gemma4-26b-q4-256k` | 180W | 256K | 55.43 | 133.9 | 235 | 130.5 | 63->65->64C | 19.56 |
| `gemma4-26b-q4-256k` | 200W | 256K | 55.42 | 132.6 | 237 | 130.4 | 63->65->64C | 19.56 |
| `ornith35-q5-256k` | 150W | 256K | 72.29 | 107.5 | 235 | 139.8 | 64->66->66C | 26.34 |
| `ornith35-q5-256k` | 165W | 256K | 72.21 | 107.0 | 236 | 139.7 | 64->66->66C | 26.34 |
| `ornith35-q5-256k` | 180W | 256K | 72.27 | 106.8 | 237 | 139.7 | 64->66->66C | 26.34 |
| `ornith35-q5-256k` | 200W | 256K | 72.12 | 107.4 | 236 | 139.5 | 64->66->66C | 26.34 |
| `qwen27-mtp-128k` | 150W | 128K | 22.79 | 42.7 | 612 | 149.8 | 66->69->68C | 25.89 |
| `qwen27-mtp-128k` | 165W | 128K | 25.15 | 43.9 | 602 | 164.6 | 68->71->70C | 25.89 |
| `qwen27-mtp-128k` | 180W | 128K | 26.76 | 43.8 | 602 | 178.4 | 70->73->72C | 25.89 |
| `qwen27-mtp-128k` | 200W | 128K | 28.33 | 43.9 | 601 | 196.8 | 71->75->74C | 25.89 |
| `qwen27-mtp-200k` | 150W | 200K | 22.74 | 42.9 | 613 | 149.7 | 66->70->68C | 27.81 |
| `qwen27-mtp-200k` | 165W | 200K | 25.13 | 44.0 | 599 | 164.6 | 68->71->70C | 27.80 |
| `qwen27-mtp-200k` | 180W | 200K | 26.77 | 44.0 | 599 | 178.4 | 69->73->72C | 27.80 |
| `qwen27-mtp-200k` | 200W | 200K | 28.27 | 43.9 | 602 | 196.9 | 72->75->74C | 27.80 |
| `qwen35-q4-256k` | 150W | 256K | 65.22 | 99.0 | 253 | 130.0 | 63->65->64C | 24.28 |
| `qwen35-q4-256k` | 165W | 256K | 64.81 | 97.5 | 255 | 129.4 | 62->65->64C | 24.30 |
| `qwen35-q4-256k` | 180W | 256K | 65.13 | 99.0 | 254 | 129.7 | 63->65->64C | 24.30 |
| `qwen35-q4-256k` | 200W | 256K | 65.07 | 98.8 | 253 | 129.7 | 62->65->64C | 24.28 |
| `qwen35-q5-256k` | 150W | 256K | 63.53 | 96.2 | 257 | 131.6 | 63->65->64C | 27.81 |
| `qwen35-q5-256k` | 165W | 256K | 63.35 | 97.1 | 255 | 131.7 | 63->65->64C | 27.75 |
| `qwen35-q5-256k` | 180W | 256K | 63.10 | 96.2 | 258 | 131.4 | 63->65->64C | 27.75 |
| `qwen35-q5-256k` | 200W | 256K | 63.31 | 96.5 | 257 | 131.6 | 63->65->64C | 27.75 |

## Conclusions

For MoE models, 150W is the correct default. Qwen 35B, Gemma 4 26B, and Ornith 35B do not gain useful decode speed from 165W, 180W, or 200W. They also do not pull the higher cap: observed card power stays roughly 130-140W and temperatures stay around 64-66C after warmup.

For dense Qwen 27B MTP, power cap matters. Moving from 150W to 200W improves decode from about 22.8 tg/s to about 28.3 tg/s, but raises sustained temperature from roughly 68-70C to roughly 74-75C. 165W remains the efficiency point; 180W is the sustained-performance point; 200W is a faster short-run profile.

Qwen 27B 200K context has the same decode speed as 128K in this sweep, with about 1.9 GiB more VRAM use. The 200K profile is viable on the B70 with q5_0/q4_1 KV.

## vLLM Scout Notes

The Intel LLM Scaler vLLM container works on this host and starts XPU inference. Current findings:

- `Intel/Qwen3.6-27B-int4-AutoRound` starts at 256K with `--enforce-eager --dtype float16 --kv-cache-dtype fp8`.
- The same model previously crashed generation under the default BF16 path with `dt_bias dtype must match core_attn_out dtype`; explicit FP16 avoids that crash.
- First successful non-stream request returned 128 tokens in 11.1s end-to-end.
- Streamed 150W smoke rounds completed without crashing; card draw was about 140-143W and temperature rose from 57C to 69C.
- Streamed chunks did not include token usage, so a cleaner vLLM throughput harness needs either non-streamed usage timing, metrics scraping, or tokenizer-side token counting.

Do not compare vLLM FP8-KV results directly against llama.cpp q5_0/q4_1-KV results as if the cache precision were identical.

## OVMS Ornith Scout Notes

Tested `openvino/model_server:2026.2.1-gpu` with
`Wondernutts/Ornith-1.0-35B-AEON-Ultimate-Uncensored-BF16-int4-ov`, the model
behind the Reddit/HF 89-106 tok/s claims.

Download notes:

- OVMS built-in HF/LFS pull was slow on the 18.65 GB language-model blob.
- `aria2c -c -x16 -s16` resumed the partial file and reached roughly 30-90 MiB/s
  early in the transfer, but HF/Xet returned some 403s for ranged chunks.
- Best practical pattern was: start with `aria2c -c -x16 -s16`, then resume with
  a fresh URL at `aria2c -c -x4 -s4` when surviving connections drop near the end.
- Remaining sidecar files also benefited from `aria2c -c -x4 -s4`.

AUTO/VLM path:

- Full model directory loads as a Visual Language Model Continuous Batching
  servable. Load time was about 3 minutes from local disk/cache.
- VRAM after load was about 19.8-20.3 GiB used, leaving about 12.4-12.8 GiB
  visible free. Temperature stayed about 53-58C during the smoke runs.
- OVMS logged `BOS token was not found in model files`.
- A 2332-token prompt plus `max_tokens=32` produced about 34 tok/s
  end-to-end after warmup. Streaming TTFT averaged about 0.409s.
- Output quality/template was poor for the smoke prompt: the model emitted a
  visible "thinking process" instead of obeying `Say exactly: READY`.

Text-only LM path:

- A hard-linked text-only copy without `openvino_vision_*` files allowed OVMS to
  initialize `Language Model Legacy servable`.
- The full directory cannot be forced to `--pipeline_type LM`; OVMS rejects it
  with `Models directory content indicates VLM pipeline`.
- The text-only LM path registered, but `/v3/chat/completions` failed with
  `Chat template not loaded correctly`, and `/v3/completions` failed because the
  graph did not expose the expected `input_ids` tensor.

Current conclusion: this specific Wondernutts OV model is not a drop-in win on
this host yet. It proves OVMS can load the model on the B70, but the measured
AUTO/VLM serving path is much slower than the Reddit tg32 claim and has template
quality issues. Next OVMS test should use the official `OpenVINO/Qwen3.6-35B-A3B-int4-ov`
model and the documented parser flags before spending more time on this Ornith
conversion.

## OVMS Official Qwen 35B A3B Scout Notes

Tested `OpenVINO/Qwen3.6-35B-A3B-int4-ov` with
`openvino/model_server:2026.2.1-gpu`, matching Intel/OpenVINO's documented B70
path more closely than the Wondernutts Ornith conversion.

Download notes:

- The 18 GiB `openvino_language_model.bin` completed after multiple `aria2c`
  resumes. `-x16 -s16` reached roughly 43-53 MiB/s early but HF/Xet range URLs
  started returning 403s near the tail.
- The best stable pattern was `aria2c -c -x8 -s8` for the main body, then
  `aria2c -c -x4 -s4` or letting the final surviving connection finish for the
  last few percent.
- The bottleneck was Hugging Face/Xet signed ranged URLs, not local network or
  disk throughput. More connections were not always faster because dead ranges
  had to be restarted with fresh signed URLs.
- Final local repo size is about 19 GiB at
  `/mnt/<models-disk>/ovms-repo/OpenVINO/Qwen3.6-35B-A3B-int4-ov`.

Full VLM/AUTO path:

- Full model directory loads as `Visual Language Model Continuous Batching
  servable`; startup took about 2m36s before `/v3/models` listed `qwen35`.
- VRAM after load was about 19.2 GiB used, leaving about 12.7 GiB visible free.
  Temperature during smoke tests was about 60-61C at the 150W cap.
- OVMS again logged `BOS token was not found in model files`.
- Short chat replies are not usable with low `max_tokens`: `max_tokens=16` and
  `max_tokens=64` returned empty `content` because generated tokens were captured
  as `reasoning_content`. With `max_tokens=256`, the final content appeared only
  after 73 completion tokens.
- Timed chat smoke, `max_tokens=128`, prompt sizes 586-794 tokens:
  29.9, 40.8, and 36.9 completion tok/s wall-rate. All three hit `length` while
  still emitting reasoning text.
- `tg32`-style smoke with a repeated 3790-token prompt:
  warmup 7.3 tok/s wall-rate, then 25.9, 25.9, and 26.3 tok/s wall-rate. Output
  remained in `reasoning_content` with empty `content`.

Text-only hardlink attempt:

- A hard-linked copy with `openvino_vision_*` and processor files removed made
  OVMS select `Language Model Continuous Batching servable`, but initialization
  failed because the model config still references
  `openvino_vision_embeddings_model.xml`.
- Unlike the Ornith conversion, this official Qwen repo cannot be converted to a
  text-only directory by simply removing the vision sidecars.

Current conclusion: official Qwen OVMS loads and runs on the B70, but the local
observed chat path is still far below the Reddit `tg32` 100+ tok/s claim and has
reasoning/content extraction issues. Before treating OVMS as a replacement for
llama.cpp SYCL, test with the same benchmark harness used in the claim
(`genai-bench` pp2048/tg32 if available) and inspect whether their run used a
different chat template, parser setting, or non-chat generation path.
