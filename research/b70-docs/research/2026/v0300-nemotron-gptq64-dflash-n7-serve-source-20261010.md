---
title: "vLLM XPU v0.30.0 pinned source: Nemotron GPTQ-INT4 G64 + DFlash n7 serve confirmation (Phase A3)"
type: research-note
status: provisional
date: 2026-10-10
---

# v0.30.0 source confirmation: Nemotron GPTQ-INT4 G64 + DFlash n7 on XPU, capture on

Phase A3 of `docs/plans/2026-10-09-zcode-workflow-prompt.md` (row A3, §3/§4). Read-only
source research; no GPU work, no server start, no measurements. Verdict: **supported**;
no missing feature found. All three parts of the ask check out from the pinned source.

## 1. What was inspected (primary source identity)

- Image `vllm/vllm-openai-xpu:v0.30.0@sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d`,
  listed by `sudo -n docker -H unix:///run/<exl3-docker>.sock images --digests`
  (isolated daemon, socket `/run/<exl3-docker>.sock`, unit `<exl3-dockerd>d5`
  running the exact argv documented in the workflow prompt §2; started 2026-10-10
  09:06:26 by the documented `systemd-run` command — my own `systemd-run` call
  reported the unit already loaded, so the daemon may have been started
  concurrently by another Phase-A session; argv verified identical via
  `systemctl cat` + `ps`).
- Image config: `Entrypoint ["vllm","serve"]`, `VLLM_TARGET_DEVICE=xpu`,
  `VLLM_WORKER_MULTIPROC_METHOD=spawn` (`docker image inspect ... --format
  '{{json .Config.Entrypoint}} {{json .Config.Cmd}} {{json .Config.Env}}'`).
- Installed package: `/opt/venv/lib/python3.12/site-packages/vllm`,
  dist-info `vllm-0.30.0+xpu.dist-info` (layer
  `…/<exl3-dockerd>/overlay2/78607364…/diff/…`). `vllm/_version.py:21-24`:
  `__version__ = '0.30.0'`, `commit_id = 'gced6857af'`. No `.pth`/`direct_url.json`
  → real (copied) install, so site-packages is the code that runs.
- Extraction (read-only, container never started): `docker create` (container
  `a3-src-inspect`, removed afterwards) + one `docker export | tar -x` pass of
  `opt/venv/.../vllm/*` + dist-info into `/tmp/a3-v0300` (ephemeral scratch).
  Line numbers below are from that copy; byte-identical to the image layer.
- torch installed in the same image: `torch-2.13.0+xpu.dist-info` (layer
  `…/overlay2/29e7f502…/diff/opt/venv/lib/python3.12/site-packages/`, lower in
  the stack than the vllm layer). Version read from the dist-info directory
  name; torch was not executed.

## 2. Ask (1): Nemotron-3.5-Lightning-30B-A3B GPTQ-INT4 G64 serves on XPU with XPU graph capture on

**Artifact on disk** — `/mnt/<sata-disk>/nemotron-lightning-gptq-sym64-20260812/config.json`:
`architectures: ["NemotronHForCausalLM"]` (lines 2-4), `model_type: "nemotron_h"`
(line 84), 52 hybrid mamba/moe/attention layers (lines 22-75, 101), 128 routed
experts / 6 per tok (lines 95, 100), `quantization_config: {quant_method: "gptq",
bits: 4, group_size: 64, sym: true, desc_act: false, lm_head_quantized: false,
modules_to_not_convert: [backbone.embeddings, lm_head]}` (lines 107-119),
`max_position_embeddings: 262144` (line 81).

**Model support** (paths inside the installed `vllm` package of the pinned image):
- `model_executor/models/registry.py:185` maps `NemotronHForCausalLM` →
  `nemotron_h.NemotronHForCausalLM`; class at `nemotron_h.py:711`; `NemotronHModel`
  at `:558` (with `EagleModelMixin` — the hook DFlash needs for head/aux-state
  sharing). MTP-block weights (`mtp.*`, `num_nextn_predict_layers: 1` in the
  target config) are dropped for this architecture:
  `nemotron_h.py:657-659` (`is_spec_layer` → `weight_name.startswith("mtp.")`),
  `:728` (`orig_to_new_prefix={"backbone": "model", "mtp": None}`).
- **GPTQ dispatch**: `model_executor/layers/quantization/__init__.py:167` maps
  `"gptq"` → `AutoGPTQConfig`; `auto_gptq.py:216-234` (`override_quantization_method`)
  accepts user `--quantization gptq`; `auto_gptq.py:97-118,149,197` parse
  group_size/desc_act (`normalize_and_validate_gptq_desc_act(desc_act=False, 64)` OK).
- **Dense W4A16 on XPU**: `auto_gptq.py:338-354` builds an `MPLinearLayerConfig`
  and calls `choose_mp_linear_kernel`; `model_executor/kernels/linear/__init__.py:444-447`
  lists the XPU candidates `[XPUW4A8IntLinearKernel, XPUwNa16LinearKernel]`;
  `kernels/linear/mixed_precision/xpu.py:20-54` (`XPUwNa16LinearKernel.can_implement`):
  XPU-only, BF16/FP16 activations (line 30), weight `uint4`/`uint4b8` (line 15, 33),
  **group_size multiple of 32 → 64 accepted** (lines 40-45), input size multiple
  of 32 (lines 47-52). Apply calls the native XPU op
  `torch.ops._xpu_C.int4_gemm_w4a16` (`xpu.py:92-109`).
- **MoE experts on XPU**: `auto_gptq.py:240-255` — for `RoutedExperts` layers the
  marlin check cannot hold on XPU, so it warns "Falling back to Moe WNA16
  kernels" and returns `MoeWNA16Config`; `moe_wna16.py:239-251` supports int4,
  group 64 (target sizes 2688 / 1856 are divisible by 64; the divisor loop at
  `:276-286` keeps ≥32). `nemotron_h.py:223-231` creates the experts via
  `FusedMoEFactory(... quant_config=quant_config ...)` with
  `use_grouped_topk=True`, matching the config's `norm_topk_prob`.

**XPU graph capture on** — three conditions, all in source:
- Do NOT pass `--enforce-eager`: default is `False` (`config/model.py:242`);
  `enforce_eager=True` forces `CUDAGraphMode.NONE` (`config/vllm.py:1780-1784`).
- Env `VLLM_XPU_ENABLE_XPU_GRAPH=1` is **required on XPU**:
  `platforms/xpu.py:293-306` — torch without XPU-graph support → forced NONE;
  `elif not envs.VLLM_XPU_ENABLE_XPU_GRAPH:` → forced NONE with the warning
  "please set VLLM_XPU_ENABLE_XPU_GRAPH=1 to enable it."; else "experimental,
  single-GPU execution". Default False (`envs.py:318`, `:2112-2114`).
- torch gate: `utils/torch_utils.py:1050-1051` `supports_xpu_graph()` =
  torch ≥ 2.11.0.dev; the image ships torch **2.13.0+xpu** → passes.
  `v1/worker/xpu_model_runner.py:42-64` then aliases
  `torch.cuda.graph/​CUDAGraph/graph_pool_handle` → `torch.xpu.*` (lines 59-63),
  so the standard V1 capture path (`XPUModelRunner(GPUModelRunner)`, `:16`;
  `XPUWorker`, `xpu_worker.py:17,24-25`) records real XPU graphs. Default v1
  cudagraph mode is FULL_AND_PIECEWISE (`config/compilation.py:614-631`); DFlash
  narrows the draft to FULL_DECODE_ONLY (see §3). `XPUPlatform.support_static_graph_mode()`
  → True (`platforms/xpu.py:443-444`), so the graph branch in
  `config/vllm.py:1725` is active.

**Operational corroboration (not the source of this verdict)**:
`results/2026/10/b70-v030-262k-230w-full/provenance.json` — this exact digest
(`image_override`, line 21) served `gptq` + float16 + `xpu_graph: "1"` +
`prefix_caching: false` + `async_scheduling: true` + `spec: "dflash n7"` on
2026-10-01 with working acceptance counters (50.5 % / 55.0 % / 33-55 % across
cells) — the stack ran end-to-end. The `patch_applied: true` in that record is
the vxk **FP8-GEMM scratchpad pooling** patch (`scripts/tmp/apply_scratchpad_patch.py:1-19`,
patches `/mnt/<models-disk>/vxk-build/src` FP8 GEMM headers) — orthogonal to this
GPTQ-INT4 stack; the n5 finals ran no-patch
(`results/2026/10/b70-final3-np-n5/provenance.json`, `patch_applied: false`).

## 3. Ask (2): exact serve flags for DFlash n7 with the BF16 drafter

**Drafter artifact on disk** —
`/mnt/<sata-disk>/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-DFlash-BF16/config.json`:
`architectures: ["DFlashDraftModel"]` (lines 2-4), `dtype: "bfloat16"` (line 26),
6 full-attention layers (41-48, 54), `has_lm_head: false` (line 36 — shares the
target head, as recorded in the model dossier), `dflash_config: {mask_token_id:
990, target_layer_ids: [1,5,19,29,41,51], causal: false, use_swa: false,
sample_from_anchor: false}` (lines 7-21), `eagle_aux_hidden_state_layer_ids:
[2,6,20,30,42,52]` (lines 27-34).

**Source support**:
- `model_executor/models/registry.py:636`: `"DFlashDraftModel"` →
  `qwen3_dflash.DFlashQwen3ForCausalLM`.
- `config/speculative.py:67-70`: `DFlashModelTypes = Literal["dflash"]` is a
  member of `EagleModelTypes`, hence a valid `SpeculativeMethod` (`:72-81`).
  `num_speculative_tokens: int = Field(default=None, gt=0)` (`:381`).
  Auto-detection: `"dflash" in draft_model_config.model.lower()` →
  `method = "dflash"` (`:1319-1323`); explicit method is passed through at
  `:1307`. Drafter-side fields are read from the drafter's own `dflash_config`
  (`:136-138` override precedence, `:242` target_layer_ids, `:336` mask_token_id).
- Worker side: `v1/worker/gpu/spec_decode/dflash/{speculator,utils,cudagraph}.py`
  exist; the XPU runner is `GPUModelRunner`-derived so this is the same path.
  `speculator.py:122-141 init_cudagraph_manager`: DFlash does **not** support
  PIECEWISE — with a mode whose decode_mode is FULL and UNIFORM_BATCH attention
  support it sets `CUDAGraphMode.FULL_DECODE_ONLY` for the draft query graphs,
  else eager; `capture()` at `:144+` records the draft via
  `DFlashCudaGraphManager`. So the draft is captured, not eager, when the
  §2 conditions hold.

**Exact serve invocation** (engine flags every one of which exists in v0.30.0;
non-flag values from the 2026-10-01 v0.30.0 run and `scripts/bench/b70-final-run3.py:159-166`):

```bash
VLLM_XPU_ENABLE_XPU_GRAPH=1 vllm serve \
  /mnt/<sata-disk>/nemotron-lightning-gptq-sym64-20260812 \
  --dtype float16 \
  --quantization gptq \
  --speculative-config '{"method":"dflash","model":"/mnt/<sata-disk>/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-DFlash-BF16","num_speculative_tokens":7}' \
  --async-scheduling \
  --max-model-len <declare per campaign; historical 16384 / probe 262144> \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching
```

Flag provenance: `--speculative-config`/`-sc` JSON
(`engine/arg_utils.py:1696-1698`); equivalent shorthand
`--spec-method dflash --spec-model <path> --spec-tokens 7`
(`arg_utils.py:1701-1704`; per-key mutually exclusive with the JSON form,
`:1933-1947`). `--async-scheduling` (`arg_utils.py:1644`;
`config/scheduler.py:190-194`). `--enforce-eager` exists but must NOT be passed.
Deployment wrap (host): run in the pinned image by digest with
`--device=/dev/dri --group-add <render gid>` and `ZE_AFFINITY_MASK=level_zero:<card>`,
per the host's established pattern; the image entrypoint is `vllm serve`.

## 4. Ask (3): the flag that guarantees prefix cache stays off

**`--no-enable-prefix-caching`** — and it is required, not optional:

- `config/cache.py:130`: `enable_prefix_caching: bool = True` — **ON by
  default** in this V1 engine.
- `engine/arg_utils.py:538`: CLI `enable_prefix_caching: bool | None = None`;
  bool-typed args get `argparse.BooleanOptionalAction`, which "Creates
  `--no-<name>` and `--<name>` flags" (`arg_utils.py:360-361`); the arg is
  registered at `:1281-1286` with `default: None`. When left as None it resolves
  to the engine default True (`arg_utils.py:2828-2829`).
- Therefore an explicit `--no-enable-prefix-caching` is the only CLI guarantee;
  it satisfies AGENTS.md §7 / standing-rules §1.4 (off on vLLM-XPU speculative
  stacks until the #53912 class is fixed and re-verified).

## 5. What this note is not

- No server was started and no measurement was taken in this phase (Phase A3 is
  source research; the GPU run is Phase G). "Serves" here means: architecture,
  quant path, speculation method, draft model class and capture path all exist
  and are wired in the pinned source, plus the 2026-10-01 provenance record of
  this exact digest running the stack.
- The historical launcher `/tmp/b70-sweep.py` that produced the 2026-10-01 runs
  is gone; the flag record cited is `scripts/bench/b70-final-run3.py:159-166`
  plus `provenance.json` files, not a replayable launcher.
- `supports_xpu_graph` was verified against the installed torch's dist-info
  directory name (2.13.0+xpu), not by executing torch.

## 6. Cleanup

`a3-src-inspect` (created and never started by this phase) was removed from the
isolated daemon: `docker -H unix:///run/<exl3-docker>.sock rm -f a3-src-inspect`.
The isolated daemon was left running (started with the documented argv at
09:06:26; the Phase-A runtime lane owns its lifecycle). An unrelated exited
container `nemotron-hermes` on the same daemon belongs to another session and
was not touched. `/tmp/a3-v0300` is ephemeral scratch.
