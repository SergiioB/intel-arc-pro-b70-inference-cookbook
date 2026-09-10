# Feature flags

Registry of the environment-gated runtime flags the tested recipes actually
set. These are recipe gates, not application feature flags: launchers export
them, and runtime patches read them inside the serving container.

This page exists so agents and operators do not have to grep launchers and
patch files to learn what a flag does. Defaults below are what happens when
the flag is unset. The authoritative compatibility order of patches per image
stays in [IMAGE-AND-PATCH-MATRIX.md](IMAGE-AND-PATCH-MATRIX.md); the
authoritative watchdog operational envs stay in
[watchdog/README.md](../watchdog/README.md) and are not duplicated here.

## Model-draft gates (patch-injected)

| Flag | Default | Effect | Family / route | Source |
|---|---|---|---|---|
| `B70_MTP_BF16_DRAFT=1` | off | Loads the preserved BF16 MTP draft weights outside the GPTQ quantization config. Required by every MTP recipe on GPTQ checkpoints. | qwen36-35a3, qwen36-27, qwen38-27, ornith15-35a3 | [launch-vllm-128k-mode.sh](../benchmarks/qwen36-35a3/launch-vllm-128k-mode.sh), [FULL-SETUP §11](FULL-SETUP-COMMANDS.md) |
| `B70_DRAFT_LMHEAD_INT4=1` | off | Draft-INT4 overlay: runtime RTN of the draft LM head. Target verify stays BF16. Set together with `B70_DRAFT_MTP_INT4`. Blocked under tensor parallelism by a guard. | qwen38-27 S+M1 overlay | [§11](FULL-SETUP-COMMANDS.md), [patch_draft_lmhead_int4.py](../patches/patch_draft_lmhead_int4.py) |
| `B70_DRAFT_MTP_INT4=1` | off | Draft-INT4 overlay: runtime RTN of the five MTP linears. Same pairing and TP guard as above. | qwen38-27 S+M1 overlay | [§11](FULL-SETUP-COMMANDS.md), [patch_draft_mtp_int4.py](../patches/patch_draft_mtp_int4.py) |
| `B70_GDN_MIXED_SPLIT_V5` | off | Patch marker for the mixed-batch GDN split v5; keeps the patch idempotent and lets logs prove it applied. | qwen38-27 v5 overlay | [patch_gdn_mixed_split_v5.py](../patches/patch_gdn_mixed_split_v5.py) |
| `B70_MTP_NIGHTLY_DRAFT` | patch marker | Identifies the BF16 MTP draft build inserted by `patch_mtp_nightly.py`. | every route applying `patch_mtp_nightly.py` | [patch_mtp_nightly.py](../patches/patch_mtp_nightly.py) |
| `B70_MTP_PARTIAL_FINAL_GROUP` | patch marker | Handles the partial final speculative group at the exact 131,072-token boundary. | every route applying `patch_mtp_boundary.py` | [patch_mtp_boundary.py](../patches/patch_mtp_boundary.py), [FULL-SETUP §5](FULL-SETUP-COMMANDS.md) |

## Multi-GPU gates (dual-B70 TP2 / PP2)

| Flag | Default | Effect | Family / route | Source |
|---|---|---|---|---|
| `B70_WORKER_AFFINITY=1` | off | Patches vLLM to spawn one worker per card and set `ZE_AFFINITY_MASK=<rank>` per rank. Without it, TP2 aborts at `zeMemOpenIpcHandle` on dual-B70 hosts. | dual-B70 TP2 | [§12](FULL-SETUP-COMMANDS.md), [DUAL-B70-TP2.md](DUAL-B70-TP2.md) |
| `ZE_AFFINITY_MASK=0` / `0,1` | unset | Pins the process to one card (single-card recipes) or both cards at container level (TP2 launch). | all recipes, §11 and §12 | [DUAL-B70-TP2.md](DUAL-B70-TP2.md) |
| `CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296` | unset | Lifts the oneCCL simple-path threshold above the max message size. One of the four required TP2 variables. | dual-B70 TP2 | [§12](FULL-SETUP-COMMANDS.md), [DUAL-B70-TP2.md](DUAL-B70-TP2.md) |
| `CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296` | unset | Same lift for reduce-scatter. Required TP2 variable. | dual-B70 TP2 | [§12](FULL-SETUP-COMMANDS.md) |
| `CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296` | unset | Same lift for allgatherv. Required TP2 variable. | dual-B70 TP2 | [§12](FULL-SETUP-COMMANDS.md) |
| `CCL_SYCL_ALLTOALL_TMP_BUF=1` | unset | Enables the oneCCL alltoall temporary buffer in the tested TP2 stack. | dual-B70 TP2 | [§12](FULL-SETUP-COMMANDS.md) |
| `CCL_ZE_IPC_EXCHANGE`, `CCL_ATL_TRANSPORT`, `CCL_ATL_SHM` | unset | Do NOT set: they select handle-exchange mechanics and the tested stack deliberately leaves them alone. | dual-B70 TP2 (negative guidance) | [DUAL-B70-TP2.md](DUAL-B70-TP2.md) |

## Runtime and allocator switches

| Flag | Default | Effect | Family / route | Source |
|---|---|---|---|---|
| `VLLM_TARGET_DEVICE=xpu` | image default (cpu) | Selects the XPU backend inside the serving image. Set by every launcher. | all GPU routes | [launch-vllm-128k-mode.sh](../benchmarks/qwen36-35a3/launch-vllm-128k-mode.sh) |
| `VLLM_XPU_ENABLE_XPU_GRAPH=1` | 0 | Enables XPU graph capture. Set to `0` for TP2: graph capture is refused on multi-GPU on every tested build. | single-card MTP and graph routes; TP2 forces 0 | [§12](FULL-SETUP-COMMANDS.md), [Start-Qwen38.ps1](../windows/Qwen38-WSLC-Standalone/Start-Qwen38.ps1) |
| `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE` | unset | Flat COMPOSITE Level-Zero enumeration so the single card shows as one device. | all launchers | [Start-Qwen38-Docker.ps1](../windows/Qwen38-Docker-Standalone/Start-Qwen38-Docker.ps1) |
| `PYTORCH_ALLOC_CONF=expandable_segments:True` | unset | Expandable allocator segments; part of the tested 128K long-context stack. | qwen36-35a3, qwen36-27, qwen38-27 | [launch-dense27-128k-mode.sh](../benchmarks/qwen36-27/launch-dense27-128k-mode.sh) |
| `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` | driver default | Disables Level-Zero immediate command lists in the tested Nemotron stack. | nemotron35-30a3 | [launch-nemotron-dflash.sh](../benchmarks/nemotron35-30a3/launch-nemotron-dflash.sh) |
| `SYCL_CACHE_PERSISTENT=0` | driver default | Disables the persistent SYCL kernel cache in the same stack. | nemotron35-30a3 | [launch-nemotron-dflash.sh](../benchmarks/nemotron35-30a3/launch-nemotron-dflash.sh) |
| `SYCL_UR_USE_LEVEL_ZERO_V2=0` | driver default | Remediation switch when Level-Zero V2 enumeration breaks. Documented as a recovery step, not a default. | reliability remediation | [RELIABILITY-REPORT.md](RELIABILITY-REPORT.md) |
| `SYCL_PI_LEVEL_ZERO_USM_RESIDENT=0` | driver default | Recovery step when the legacy L0 adapter reports false USM residency pressure (a 2 MiB allocation OOMs while tens of GiB are free). Pairs with the switch above. | reliability remediation | [RELIABILITY-REPORT.md](RELIABILITY-REPORT.md) |

## What is not here

- Watchdog operational envs (`HEALTH_URL`, `TRIGGER_MODE`, `FAIL_STREAK`,
  `RECOVERY_TIMEOUT_S`, and the rest): owned by the
  [watchdog/README.md](../watchdog/README.md) config reference.
- vLLM serve CLI flags (`--max-model-len`, `--kv-cache-dtype`, and friends):
  owned by each family recipe under `docs/<family>/`.
- Flags without a tested route are not registered. Add a row only together
  with the route that sets it, and link that route in the Source column.
