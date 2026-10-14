---
title: "Storage layout"
type: operation
status: living
date: 2026-10-13
---

# Storage layout

Authority: `registry/storage-layout.json`. Check it with:

```bash
python3 tools/storage_inventory.py check          # mounts + top-level folders match the registry
python3 tools/storage_inventory.py scan           # every folder: size, last change, LIVE if a config or process uses it
python3 tools/storage_inventory.py resolve PATH   # where an old path from a dated note lives now
```

## 1. Disks

| Mount | Disk | Use |
|---|---|---|
| `/` | WD Blue SATA 500 GB, `sda4` | OS, home, B70-DOCS, default Docker images. Keep >= 10 GB free. |
| `/data/nvme` | WD_BLACK SN770 NVMe 500 GB, `nvme1n1p1` | Fastest disk: models that are served, EXL3 docker root |
| `/data/sata1` | T-FORCE SATA 500 GB, `sdb2` | Model library, Strata install |
| `/data/sata2` | WD Blue SATA 500 GB, `sda3` (same physical disk as `/`) | Caches, archive, small models; `comfyui/` (not B70) |
| `/data/<other-disk>` | external 1 TB NTFS, external | Usually absent (`nofail`) |

`nvme0n1` (1 TB) is the Windows disk and is never mounted. All fstab entries use
`UUID=`. There are no aliases or symlinks under `/mnt`.

## 2. Folders on every data disk

| Folder | Holds | Rule |
|---|---|---|
| `models/` | One folder or file per model artifact, original artifact name kept | The launcher scans `/data/*/models`. A new download goes here. |
| `engines/` | Engine source trees, builds, venvs, self-contained installs (`sata1/engines/strata`) | Name by engine and commit or date. |
| `docker/` | Private Docker data roots | `nvme/docker/exl3-docker` belongs to the EXL3 daemon. |
| `cache/` | Re-creatable caches (Hugging Face hub, OpenVINO, compile caches) | Safe to delete when space is short. |
| `work/` | Scratch for one running campaign: `work/<campaign-slug>/` | Delete or move to `archive/` when the campaign closes. |
| `archive/` | Kept but inactive data | Record why it is kept in the campaign or cleanup note. |

Anything else at the top level of a data disk fails `storage_inventory.py check`.

## 3. Where current things live

| What | Path |
|---|---|
| Qwen3.8-27B EXL3 4.00bpw (route of record) | `/data/nvme/models/b70-exl3-qwen38-4bpw-20260926` |
| Qwen3.8-27B SC_5.00bpw_H6 (quality tier) | `/data/sata1/models/qwen38-27b-exl3-SC_5.00bpw_H6` |
| Qwen3.8-27B BF16 (KL reference) | `/data/nvme/models/b70-bf16-qwen38-20260926` |
| Flash-Next Strata IQ3_XXS + engine | `/data/sata1/engines/strata` (model in `models-iq3xxs/`) |
| Flash-Next legacy GGUF + MTP draft | `/data/sata1/models/signal-3.8-flash-next`, `/data/nvme/models/Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf` |
| Pinned Flash-Next llama.cpp build | `/data/sata2/engines/qwenflash-llama-pinned-9723942` |
| EXL3 Docker data root | `/data/nvme/docker/exl3-docker` (launcher setting `exl3_data_root=/data/nvme/docker/exl3`) |
| Hugging Face cache (`HF_HOME`) | `/data/sata2/cache/huggingface` |
| Bulky files moved out of B70-DOCS | `/data/sata2/archive/b70-docs-bulk-20261013/<same relative path>` |
| Old home folders (cascadia, b70-lab, llama.cpp-muse) | `/data/sata2/archive/home-sergio-20261013/` |

## 4. Rules

1. New campaign data goes to `/data/<disk>/work/<campaign-slug>/`, never into
   B70-DOCS and never at a disk's top level.
2. B70-DOCS keeps text evidence only. Files over 20 MB, compile caches,
   `node_modules` and profiler traces go to `/data/<disk>/archive/` with a
   `BULK-MOVED-*.txt` pointer and sha256 next to where they were.
3. When a research line closes, delete its `work/` folder or move it to
   `archive/` in the same session, and write the paths into the closing note.
4. Before deleting anything, run `scan` (it marks a folder LIVE when a config,
   container, process or a B70-DOCS symlink points at it) and keep the deleted
   list with sizes in `results/<yyyy>/<mm>/storage-cleanup-<date>/`.
5. No B70-DOCS symlink may point onto a data disk; use a pointer file.
6. Old notes keep their old paths (no mass rewrite of evidence). Use `resolve`.

## 5. History

- 2026-10-13: 413 GiB of closed-line data deleted (170 items), disks remounted
  from `/mnt/<brand>-<size>-data` to `/data/<disk>`, `/mnt/<models-disk>*` aliases
  removed, folders sorted into the six role folders, 5.3 GiB of bulky files
  moved out of B70-DOCS. Record:
  `results/2026/10/storage-cleanup-20261013/` (`tierA.txt`, `deleted.tsv`,
  `moves.tsv`, `bulk-moved.tsv`, `fstab.before`, `fstab.after`, `images.log`).
- 2026-09-12/13: brand-named mounts introduced, LVM retired. See
  `storage-and-vram-safety.md` section 9 (historical).
