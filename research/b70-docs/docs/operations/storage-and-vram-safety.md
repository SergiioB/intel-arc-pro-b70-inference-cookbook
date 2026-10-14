---
title: "Storage and VRAM Safety"
type: operation
date: 2026-09-13
status: living
---

---
doc_id: DOC-OPS-STORAGE-VRAM-SAFETY
doc_type: operational_method
title: Storage and VRAM Safety
lifecycle: stable
created: 2026-08-19
updated: 2026-08-19
last_verified: 2026-08-19
canonical_for:
  - storage-safety
  - vram-load-safety
supersedes:
  - DOC-LEGACY-LVM-AND-STORAGE-CURRENT-GUIDANCE
---

# Storage and VRAM Safety

This document contains stable operating rules. Volatile mount sizes, free space, and model inventory belong in [Current B70 Host State](current-host-state.md).

## 1. Discover storage immediately before acting

Before every download, conversion, image build, or artifact copy:

```bash
df -h / /data/nvme /data/sata1 /data/sata2
python3 tools/storage_inventory.py check
```

Mount points and folder roles are defined in `storage-layout.md` (authority:
`registry/storage-layout.json`). Paths in older notes (`/mnt/<models-disk>`,
`/mnt/<models-disk>`, `/mnt/<brand>-<size>-data`) were retired on 2026-10-13; translate
them with `python3 tools/storage_inventory.py resolve <path>`.

Budget every simultaneously retained component:

- source checkpoint;
- output artifact;
- partial downloads;
- disk offload and temporary staging;
- RAM and swap requirements;
- container/image layers on `/`;
- rollback reserve;
- an explicit safety margin on each affected mount.

A historical free-space figure is never authorization to begin.

## 2. Preserve source artifacts

Derived-artifact work must be fail-closed:

1. Keep the source unchanged.
2. Use a new, explicit output name.
3. Do not symlink any file the converter will overwrite.
4. Refuse to write an output index through a symlink.
5. Verify source and output index-to-file coverage independently.
6. Retain conversion configuration, exclusions, hashes, and numerical validation.
7. If an artifact is deleted, retain provenance without retaining secret material.

See [Quantization Decision Record](../reference/quantization-decision-record.md).

## 3. Partial-VG boot trap (retired 2026-09-13)

Historical: `vg_data` on `sdb3` carried a missing ~50 GiB `lv_data` LV on a lost PV, which blocked all VG metadata changes and required the `vg-data-lv-models.service` boot workaround. On 2026-09-13 the LV content was migrated into the sdb2 ext4 volume (checksum-verified), and the volume group was retired entirely (`vgreduce --removemissing`, `vgremove`, `pvremove`) with explicit user authorization. The systemd service was removed. There is no LVM on the T-FORCE disk anymore; if a future setup reintroduces LVM, re-derive guidance then.

## 4. Empty-GPU gate

Before any model load, all checks must pass:

```bash
pgrep -af 'llama-server|llama-bench|vllm|vllm-xpu'
docker ps --format '{{.ID}} {{.Image}} {{.Names}}'
# Discover and read the verified B70 hwmon power cap.
sudo cat /sys/kernel/debug/dri/0000:0b:00.0/tile0/vram_mm
```

Expected empty-card `visible_avail` is approximately 31,000 MiB or more. If a process/container exists, VRAM is materially below the empty-card baseline, or a check fails, stop and diagnose.

## 5. Post-load VRAM gates

| Free VRAM after load | Allowed interpretation |
|---:|---|
| below 500 MiB | Abort; do not issue a request |
| 500–1,023 MiB | Isolated C1 capacity observation only |
| 1,024–2,047 MiB | Staged C1 research |
| at least 2,048 MiB | Preferred C1 performance work |
| at least 3,072 MiB | Target for mixed/concurrent serving |

A successful server launch is not a context-capacity result. Capacity requires staged real requests that complete at the declared prompt-plus-output coordinate.

## 6. Large GGUF safety

The B70 exposes 32,656 MiB visible VRAM. A single GGUF larger than roughly 30 GB with full GPU offload can overflow VRAM and hard-crash the host rather than returning a clean OOM.

For an artifact above that boundary:

- calculate a partial GPU-layer budget;
- leave explicit KV and compute headroom;
- do not use `--no-mmap` for the CPU-offloaded portion;
- stage the load conservatively;
- never infer safety from file size alone.

## 7. Conversion-specific storage lesson

A 72 GB BF16 model cannot be assumed to materialize in 30 GB system RAM. GPTQ conversions of this class require disk offload on a mount that has room for offload independently from the destination artifact. A swapfile placed on the destination/source volume can consume the very reserve the conversion needs.

Record source, output, offload, temporary, RAM, and swap locations in the artifact manifest before launch.

**Host-RAM pressure (2026-10-11/12).** systemd-oomd watches `user@1000.service` and kills its largest unit when that slice's memory pressure stays above 50 % for 20 s. It killed the terminal hosting the agent session twice (19:39 and 00:15 local), and on 2026-10-12 a K4 checker capped at 8 GiB inside a `systemd-run --user` scope thrashed at its cap twice (~11:20 and 12:29; scope 86-94 % full pressure), lifting the whole user slice to 70-82 %. A cap inside the user slice does not protect the session: thrashing in any child raises the parent's pressure. Rules: run any job above 1 GB RSS, or any job reading model shards, in the system slice: inside a container with `--memory=<cap> --memory-swap=<cap>`, or under `sudo -n systemd-run --scope -p MemoryMax=<cap> -p MemorySwapMax=0 --uid=sergio --gid=sergio`; size caps with at least 30 % headroom over the expected peak; serving containers that mmap weights need a cap above the weight size; start only when `MemAvailable` is at least 20 GiB; stream tensors with `safe_open`, never copy shards through the page cache. `scripts/tmp/lane_watchdog_20261012.py --protect` checks pressure every 5 s and terminates a thrashing `app.slice/run-*.scope` agent job when the user slice passes 35 %, before oomd acts.

**Disk space for long quantization runs (2026-10-12).** The SN770 reached 0 bytes free at the start of a full-model trellis quantization (per-layer 5 GiB checkpoint plus about 14 GiB of tensors), and the supervisor could not even write its exit record. Before launch, budget output plus checkpoint plus one atomic temporary copy, keep at least 7 GiB free before every checkpoint write, and check free space before each automatic relaunch. Agent scratch moved off the SN770 is listed in `/mnt/<sata-disk>/b70-scratch-archive-20261012/README.txt`. `scripts/tmp/lane_watchdog_20261012.py --disk PATH:MIN_GIB` alerts below a floor.

## 8. Cleanup evidence

Every automated campaign or conversion should retain cleanup state and restore the 150 W cap. Benchmark scripts follow the stronger requirements in `docs/methods/benchmark-protocol.md`.

## 9. Mount naming history (retired 2026-10-13)

**Superseded 2026-10-13** by `storage-layout.md`: disks now mount at `/data/nvme`, `/data/sata1`, `/data/sata2` and every alias below is gone. The table is kept so old notes can be read; translate paths with `python3 tools/storage_inventory.py resolve <path>`.

Until 2026-10-13, mount points were named by physical disk identity (interface-brand-capacity-role). The fstab source is always `UUID=`. Map:

| Disk | Partition / LV | Canonical mount point | Legacy path (symlink) |
|---|---|---|---|
| `sda` WDC WDS500G2B0A (SATA 500 GB, serial 194970801031) | `sda3` ext4 | `/mnt/<sata-disk>` | `/mnt/<models-disk>` |
| `sdb` T-FORCE 500GB (SATA 500 GB, serial TM11-19112604100403) | `sdb2` ext4, single full-disk volume (former NTFS partition converted 2026-09-12; LVM `vg_data` retired 2026-09-13 with `lv_models` content migrated in and checksum-verified) | `/mnt/<sata-disk>` | `/mnt/<models-disk>`, `/mnt/<other-disk>` |
| `nvme1n1` WD_BLACK SN770 (NVMe 500 GB, serial 22073R486206) | `nvme1n1p1` ext4 | `/mnt/<nvme-disk>` | `/mnt/<nvme-disk>` |

Root (`/`), `/boot/efi`, and swap keep standard names and are documented per-disk in `/etc/fstab`.

**Windows disk exclusion:** `nvme0n1` (WDC WDS100T3X0C, NVMe 1 TB) is the Windows OS disk and is fully excluded from Linux: no fstab entry, no mount, no mountpoint directories. Do not mount it read-write; its filesystem has a history of unclean closure. Mount read-only only for explicit diagnostic tasks and clean up afterward.

**LVM retirement record (2026-09-13):** the historical partial `vg_data` (phantom ~50 GiB `lv_data` on a lost PV, `IJ2FDM-lD2Y-Yh0V-sjO4-IJCi-fERg-hcjY0k`) was removed with explicit user authorization after `lv_models` content was migrated to `sdb2` and verified 0-diff. The `vg-data-lv-models.service` boot workaround was removed with it. There is no LVM on the T-FORCE disk.

**Known broken content (pre-existing):** `Qwen3.8-27B-BF16/` on the T-FORCE volume holds shard symlinks targeting `/mnt/<models-disk>/qw38-bf16/...`, a directory that no longer exists. The symlinks were already broken before the merge and were faithfully preserved; that model is unusable until re-derived or re-linked.

**Removal record:** the former ex-NTFS working-set content (95 GiB, incl. the `qwen38-flash-next-ad-4.27bpw-q4-k-m-m64` artifact) was removed from the T-FORCE on 2026-09-13 by explicit user request. A checksum-verified copy remains at `/mnt/<nvme-disk>/tforce-shared-staging-20260912/`; delete it to reclaim ~95 GiB on the SN770.
