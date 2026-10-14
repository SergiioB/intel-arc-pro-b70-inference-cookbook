---
title: "Conversion and download status — 2026-08-12"
type: research
date: 2026-08-12
status: final
---

# Conversion and download status — 2026-08-12

## Decision

No GPU model load, vLLM test, or conversion run was started in this status pass.
The next conversion must be validated on CPU before it is allowed to consume the
exclusive B70 inference slot.

## Live storage and transfer snapshot

Observed while the MLX Nemotron and RedHat Muse downloads were active:

| Filesystem | Available | Decision relevance |
|---|---:|---|
| `/` | 14 GiB | Insufficient for an XPU image build without separate cleanup/storage work |
| `/mnt/<models-disk>` | 82 GiB | Enough to retain the ~17.8 GB MLX source and stage a similarly sized GPTQ output only after the transfer is complete and a fresh reserve check passes |
| `/mnt/<models-disk>` | 21 GiB | Insufficient for the complete 23 GB Muse compressed-tensors source plus another full GPTQ output; do not begin this conversion here |

These are transient observations, not an allocation promise. Every future action
must re-run `df -h / /mnt/<models-disk> /mnt/<models-disk>` and `findmnt`.

## Artifact decisions

### Keep: MLX Nemotron 4-bit

`/mnt/<models-disk>/mlx-nemotron-4bit/` is the intended source for the experimental
MLX affine INT4 → GPTQ conversion. Its source index declares 17,775,339,392 bytes
across four shards. A later live check found `aria2c` still owns three shard
transfers and the apparent shard total is 17,647,107,807 bytes—128,231,585 bytes
short of the index total. The `.aria2` sidecars were retained, so it must not be
converted, moved, or deleted until the transfer exits and its completion is
verified.

The conversion is approximate: MLX uses `q * scale + bias`; GPTQ uses
`(q - zero) * scale`, requiring `round(-bias / scale)` plus clamping to 0–15.
A successful file rewrite is not evidence of parity or vLLM support.

### Keep: complete Muse compressed-tensors group-32 source

`/mnt/<models-disk>/muse-glimmer-awq-int4/` is complete and declares
compressed-tensors `pack-quantized`, asymmetric, group size 32. Its `AWQ` repo
name is not its actual checkpoint format. It remains useful only as a staged
future source: Muse vLLM support still requires the architecture, exact-quant
text-only dispatch, and vision gates.

### Deleted 2026-08-13: RedHat Muse W4A16 partial

`/mnt/<models-disk>/muse-glimmer-w4a16-redhat/` was deleted after the download was
stopped and its metadata/provenance was retained under
`research/references/deleted-model-artifacts-20260813/muse-redhat-partial/`.
The partial declared compressed-tensors `pack-quantized`, **symmetric**, group
size **128**, and static activation ordering, so it could not feed the current
asymmetric group-32 scratch repacker. A future symmetric-group-128 effort must
start from a fresh, revision-pinned download rather than attempting to resume
the deleted sparse shard.

### Defer deletion: incomplete MLX Muse cache

`~/.cache/huggingface/hub/models--mlx-community--Muse-Glimmer-30B-4bit/`
is an incomplete cache with no current conversion plan. It can be deleted only
after checking that no Hugging Face downloader owns it. It was deliberately not
deleted during this status pass.

## Conversion order and gates

1. Complete and verify the MLX Nemotron source transfer without touching its
   `.aria2` controls.
2. Run the hardened CPU-only converter into a fresh, capacity-gated destination.
   It rewrites the safetensors index and emits conversion provenance.
3. Run static/index and sampled tensor dequantization validation. Record the
   recovered-zero rounding/clipping error; do not call the result lossless.
4. Only after validation: ensure the B70 is empty per `AGENTS.md` §8.1, then
   perform an isolated vLLM architecture/load/coherence smoke test. Benchmarking
   is forbidden until that test passes.
5. For Muse: wait for vLLM PR #51655 to merge and pin a post-merge XPU digest.
   Test supported raw compressed-tensors text-only dispatch before considering a
   GPTQ repack. Vision remains a distinct final gate.

## Related work

- Format/engine selection: `AGENTS.md` §5.1
- Muse gates: `research/2026/b70-vllm-muse-int4/PLAN.md`
- Q4_0 artifact provenance: `08-q4_0-breakthrough.md`
- CPU conversion: `scripts/tmp/muse-gptq-convert/mlx_to_gptq.py`
- CPU validation: `scripts/tmp/muse-gptq-convert/validate_mlx_gptq.py`
