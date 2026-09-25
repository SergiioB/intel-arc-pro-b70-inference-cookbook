# Qwen3.8-Flash-Next — AP-Q4_K_XL dual-card placement (capability, no performance claim)

Status: **capability — load + coherence verified; no performance claim on this placement.**

The AP-Q4_K_XL GGUF is ~94 GB — larger than one 32 GB card. This recipe places it across
two Arc Pro B70s with a bounded CPU offload and is the only known-working load for this
artifact on this hardware class.

## Requirements

- Binary: a `qwen4exp`-capable llama.cpp SYCL build, icx/icpx 2026.1.1
  (sha256 `7c2d6adc4ac5dfa6f1deb8db8d545d8e4cae390ed9c1cd255d1e7f71a83e84a1`).
- Model: `Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf`
  (101,142,769,984 bytes, sha256 `427a7a10dd5a98d77c71be62a89d14e7408fc61a84c6efe889393852282cd3c6`).

## The placement that works (and the one that does not)

A plain 49/51 layer split **overfills both cards** (per-card weight use ≈34.6 / 32.3 GiB
before KV) and dies at padded-tensor allocation. The corrected placement offloads the
PLE and the expert weights of the first and last six blocks to CPU:

```bash
llama-server -m Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf \
  --device SYCL0,SYCL1 --tensor-split 49,51 --split-mode layer -ngl 99 \
  -ot 'per_layer_token_embd=CPU,blk\.([0-5]|4[2-7])\.ffn_(up|gate|down)_exps\.weight=CPU' \
  -c 8192 -fa on -t 6 -tb 14 -np 1 -b 512 -ub 512 \
  --cache-type-k q8_0 --cache-type-v q4_1 \
  --no-warmup --no-cache-prompt --metrics --slots --jinja \
  --host 127.0.0.1 --port 8010
```

Observed load state: /health in ~75–90 s, >3.3 GiB free on the fuller card, no
DEVICE_LOST/OOM lines. For an 8K-prompt cell use `-c 9216`.

## Verification gates this recipe passed

1. Fixed three-prompt coherence trio at temperature 0 (answers: `B70-FLASH-OK`,
   `703`, `Paris` — this model emits a short reasoning block first; give it
   `max_tokens ≥ 256` or visible content can come back empty).
2. Offline exact-token calibration (512 and 8192) with a live `/tokenize`
   id-sequality gate before any measured request.

## Scope and limits

- This is a **capacity** recipe (94 GB across two 32 GB cards). It makes no speed
  claim; treat any per-stream numbers you measure on this placement as
  placement-specific, not as the model's or the hardware's ceiling.
- The MTP GGUF exists for this model but is not part of this placement record.
- Failure signature of the wrong placement (early-block-only offload): instant
  `UR_RESULT_ERROR_DEVICE_LOST` in `ggml_backend_sycl_buffer_init_tensor` memset,
  from padded-tensor allocation overfill.
