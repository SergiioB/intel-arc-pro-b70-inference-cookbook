---
title: "Current B70 Host State"
type: operation
date: 2026-08-27
status: living
---

---
doc_id: DOC-OPS-CURRENT-HOST-STATE
doc_type: operational_state
title: Current B70 Host State
lifecycle: active
created: 2026-08-19
updated: 2026-09-12
last_verified: 2026-09-12
canonical_for:
  - current-host-state
review_on:
  - model_inventory_change
  - storage_change
  - engine_change
  - reboot_recovery
---

# Current B70 Host State

This document records **volatile observed state**. It is not a capacity promise or a substitute for live discovery. Re-run the commands below before a download, conversion, model load, or benchmark.

## Host

| Field | Observed value |
|---|---|
| Hostname | `b70-host` |
| Kernel | `7.0.0-28-generic` (observed 2026-10-09) |
| GPU | Two Intel Arc Pro B70, 32 GB each. Dual-card default is independent stacks (ADR-0008); board: ASUS ROG STRIX X570-F GAMING dual CPU-attached x8/x8 (B450-F swapped out 2026-08-27); topology: `research/2026/x570f-dual-b70-pcie-topology-20260830.md` |
| Driver family | `xe` |
| Active profile pointer | `nemotron-vllm-dflash7-120k` |
| Observation date | 2026-08-19 (host fields); kernel and storage re-observed 2026-10-09 |

Agents running on this host must operate locally. Do **not** SSH to `<lab-host>` from `b70-host`.

## Live storage observation

Re-observed 2026-10-13 after the storage cleanup and remount
(`results/2026/10/storage-cleanup-20261013/`). Layout and current artifact
locations: `storage-layout.md`.

| Mount | Source | Filesystem | Size | Available | Use |
|---|---|---|---:|---:|---:|
| `/` | `/dev/sda4` | ext4 | 214 GiB | ~19 GiB | 92% |
| `/data/nvme` | `/dev/nvme1n1p1` | ext4 | 458 GiB | 188 GiB | 57% |
| `/data/sata1` | `/dev/sdb2` | ext4 | 458 GiB | 203 GiB | 56% |
| `/data/sata2` | `/dev/sda3` | ext4 | 229 GiB | 95 GiB | 57% |

Previous observations (2026-10-09 and earlier) used the retired `/mnt` names;
they stay in this file's history and in the dated notes. No conversion or
download may be planned from an old capacity figure.

Verify live state:

```bash
df -h / /data/nvme /data/sata1 /data/sata2
python3 tools/storage_inventory.py scan --no-size
```

Artifact presence does not imply that an artifact is recommended,
quality-validated, or safe to load. See `registry/model-status.json` and the
model dossiers.

## Inference state at migration baseline

At the documentation-migration baseline:

- no inference container was running;
- no `llama-server`, `llama-bench`, vLLM server, or GPTQ quantization process was active;
- the improved Ornith calibration workspace existed on disk but was not actively writing at the time of inspection.

Before GPU work, apply the isolation gates in `AGENTS.md` and `docs/methods/benchmark-protocol.md`.

## Volatile-state rules

1. Never copy this inventory into a stable method or storage-layout document.
2. Record `last_verified` whenever observed state changes.
3. Keep artifact availability separate from correctness, quality, and performance status.
4. Treat an active-profile pointer as control-plane state, not proof that its model is loaded or healthy.
5. There is no LVM anymore (retired 2026-09-13); missing mount content after boot points at fstab/device issues, not partial-VG activation.
6. Since 2026-09-12 mount points use physical-disk identity names (see `storage-and-vram-safety.md` §9); the old workload names are compatibility symlinks.
