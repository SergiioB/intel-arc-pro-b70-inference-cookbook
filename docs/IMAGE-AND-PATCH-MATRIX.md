# vLLM XPU Image and Patch Matrix

This cookbook has **separate vLLM generations**. Pick the **family** first,
then apply **only** that row’s “Apply” list. The last column is a
**denylist**, not a second patch list.

- **Qwen** (3.6 or 3.8) uses the **Qwen MTP** patches. Never put Nemotron
  grouped-topk / SSU on a Qwen image.
- **Nemotron DFlash** uses the **Nemotron** router + SSU patches. Never put
  Qwen `patch_mtp_nightly.py` / `patch_mtp_boundary.py` on a Nemotron image.

| Family | Image digest | Apply, in this order | Never apply on this family |
|---|---|---|---|
| Qwen3.6 Pi / dense 27B | `…2c427ef477da…` | Qwen MTP: `patch_mtp_nightly.py` then `patch_mtp_boundary.py` | Nemotron grouped-topk / SSU (Nemotron-only) |
| Nemotron-3.5-Lightning DFlash | `…1da0a9548545…` | Nemotron: `patch_xpu_grouped_topk_native_v2.py` then SSU B8/W4 | Qwen MTP patches (Qwen-only) |
| Qwen3.8-27B | `…f01e24f6c7ff…` | Qwen MTP: `patch_mtp_nightly.py` then `patch_mtp_boundary.py`. Optional mixed-batch: compact+scatter `patch_gdn_mixed_split_v5.py`. Optional MTP speed: `patch_draft_lmhead_int4.py` then `patch_draft_mtp_int4.py` (`B70_DRAFT_LMHEAD_INT4=1`, `B70_DRAFT_MTP_INT4=1`) | Nemotron grouped-topk / SSU. Original full-buffer GDN split (OOB on kernels 0.1.12.3). DFlash 2 / PR #52816 is **not** an apply-list item: overlay can load, measured accept **0%** |
| Ornith-1.5-35B-A3B | `…f01e24f6c7ff…` (same nightly as Qwen3.8) | `patch_mtp_boundary.py` for exact max-len MTP. `patch_mtp_nightly.py` already-in-image if `dynamic` excludes `mtp`. **Default DraftINT4** (`B70_DRAFT_LMHEAD_INT4=1`, `B70_DRAFT_MTP_INT4=1`): launcher `DRAFT_INT4=1`. Weights: MixedCal-v2 experts-only GPTQ ([MIXEDCAL-V2.md](ornith15-35a3/MIXEDCAL-V2.md)). Recipe: [ORNITH-VLLM-XPU.md](ornith15-35a3/ORNITH-VLLM-XPU.md). | Nemotron grouped-topk / SSU. GDN mixed-split v5 on C1. Do not expect the v0.21 native-int4moe 204.6 path on this image. |

The **Windows 11 standalone kits** ([qwen38-27/WINDOWS-STANDALONE.md](qwen38-27/WINDOWS-STANDALONE.md),
[`windows/`](../windows/)) are not a new generation: they build from the same
Qwen3.8 `f01e24f6` digest. Image tag **2026.08.19** applies the Qwen MTP pair
(SHA-256-verified against `5c6b6b1`) **and** the optional mixed-split v5 +
draft-INT4 S+M1 patches, with prefix cache **on** for serving. `DRAFT_INT4=0`
recovers the 18 August BF16-draft path. Display-attached VRAM pin (0.75 +
4.30 GiB fp8 KV, 100K; bumped from 4.25 on 2026-08-22 for headroom) is
otherwise unchanged.

Nemotron-3.5-Lightning DFlash is a **second generation**. It uses a newer
public digest than the Qwen3.6 Pi matrix below. Do not mix patch lists.

## Nemotron DFlash generation (2026-08-13)

```text
vllm/vllm-openai-xpu@sha256:1da0a95485455f08588c11080b9718992fd7d434c6a965d74654903a9d999c57
vLLM 0.26.1rc1.dev668+g3ee2df303
vllm-xpu-kernels 0.1.12.3
```

Runtime patches, in order: `patch_xpu_grouped_topk_native_v2.py`, then copy
`ssu-b70-b8w4/*.json` into the SSU config dir. That Python router change is
also [vllm#52159](https://github.com/vllm-project/vllm/pull/52159).

Optional **source** kernel edits (need a `vllm-xpu-kernels` rebuild, **not**
a local Docker tag): `patches/vllm-xpu-kernels/0001-zero-xe2-grouped-gemm-atomic.py`
then `0002-muse-paged-decode-tuple.py` —
[vllm-xpu-kernels#524](https://github.com/vllm-project/vllm-xpu-kernels/pull/524).
`0001` is for temperature-0 graph-replay determinism. `0002` is Muse-only.

Models: `SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym` +
`SergiioB/Nemotron-3.5-Lightning-30B-A3B-DFlash-BF16`.
Launcher: `benchmarks/nemotron35-30a3/launch-nemotron-dflash.sh`.
Recipe: `docs/nemotron35-30a3/NEMOTRON-DFLASH-B70.md`.
Do **not** apply `patch_mtp_nightly.py` / `patch_mtp_boundary.py` to this
generation (those are Qwen3.6 MTP).

## Current supported recipe (Qwen3.6 Pi matrix)

Use this exact public image:

```text
vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97
```

Observed version inside the image:

```text
vLLM 0.26.1rc1.dev457+gc810e5ee9.xpu
vllm-xpu-kernels 0.1.12
```

PyPI published `vllm-xpu-kernels 0.1.12.2` on August 5, 2026. That newer patch
wheel has not been measured in this recipe. The current tables are specifically
`0.1.12` results from the pinned image; changing the wheel creates a new
comparison generation.

Verify package metadata inside a running container:

```bash
sudo docker exec b70-mtp2-cache-on python -c \
  "import importlib.metadata as m; print(m.version('vllm')); print(m.version('vllm-xpu-kernels'))"
```

Use one of these models:

```text
MoE:    llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4
Dense:  llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4
```

The local model directory must contain the preserved `mtp.*` tensors. The official `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` v2 now ships those draft weights in a separate `mtp.safetensors` (785 per-expert-split MTP keys) and was verified byte-exact on this stack; older snapshots of that repo lack the file and would fall back to no-spec.

**Dense 27B additional requirements (verified Run 29/31):** the same two patches
apply unchanged to the dense `Qwen3_5ForConditionalGeneration` (same shared
`qwen3_5_mtp.py` / `gdn_attn.py`). Dense launches MUST add
`--kv-cache-dtype fp8` (fp16 KV needs 9.5 GiB at 128K and does not fit) and use
`gpu-memory-utilization=0.88` for MTP4 (0.90 fills the card with spec buffers).
Use `benchmarks/qwen36-27/launch-dense27-128k-mode.sh` for the dense track; the MoE
launcher is `benchmarks/qwen36-35a3/launch-vllm-128k-mode.sh`.

Apply these patches in order:

| Order | File | Purpose |
|---:|---|---|
| 1 | `patches/patch_mtp_nightly.py` | Build the preserved BF16 MTP draft outside the target GPTQ quant config |
| 2 | `patches/patch_mtp_boundary.py` | Complete exact 128K when the final MTP4 group is shorter than five tokens |

Do not apply `patch_xpu_int4_moe_v4.py` or `patch_mtp_bf16_draft.py` to this nightly. Those files target the historical vLLM 0.21 stack and use different source anchors.

## Launch any tested mode

The generic launcher supports no-spec, MTP1, MTP2, and MTP4 with cache explicitly on or off:

```bash
git pull --ff-only
export MODEL_DIR=/path/to/Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4
bash benchmarks/qwen36-35a3/launch-vllm-128k-mode.sh "$MODEL_DIR" mtp2 on 8000
```

The launcher pins the image by digest, mounts both current patches, applies the BF16 MTP patch first and the exact-boundary patch second, sets `--max-model-len 131072`, and sets `--max-num-batched-tokens 8192`.

Follow startup:

```bash
sudo docker logs -f b70-mtp2-cache-on
```

Expected first lines include both:

```text
patched .../qwen3_5_mtp.py
patched .../gdn_attn.py
```

Check health after model load:

```bash
curl -f http://127.0.0.1:8000/health
```

Full model download, patch verification, all eight launch commands, the complete Docker command, and the matched benchmark command are in [`FULL-SETUP-COMMANDS.md`](FULL-SETUP-COMMANDS.md).

## Historical image warning

```text
intel/vllm:0.21.0-xpu-int4moe
```

was a local derived image used during the 2026-08-06 research campaign. It was not published to Docker Hub. A reader cannot pull it.

`intel/vllm:0.21.0-xpu` is not a drop-in replacement. The old cookbook patches expected changes present in the local derived image, so applying them to the public base tag can fail on missing source anchors or runtime APIs.

Historical files remain in this repo to preserve the evidence trail:

- `patches/patch_xpu_int4_moe_v4.py`
- `patches/patch_mtp_bf16_draft.py`
- `benchmarks/qwen36-35a3/launch-mtp-bf16draft.sh`

Do not use them for the current quick start.

## Exact 128K proof

The current two-patch path completed:

| Field | Value |
|---|---:|
| Exact prompt | 130,944 tokens |
| Requested/completed output | 128 / 128 tokens |
| Total sequence | 131,072 tokens |
| TTFT | 48.601 s |
| Client post-first rate | 96.87 tok/s |
| MTP acceptance | 72.31% |
| Prefix-cache hit delta | 0 |
| Free VRAM after load | 2,619 MiB |
| Logged KV capacity | 165,961 tokens |

`Client post-first rate` is not an engine-native timing field. It is `(completion_tokens - 1) / (request_end - first_token)`.

Full commands and real-world tables: [`REAL-WORLD-PI-BENCHMARKS.md`](REAL-WORLD-PI-BENCHMARKS.md).
