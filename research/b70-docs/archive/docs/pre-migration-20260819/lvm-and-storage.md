# Disk Layout, LVM, and VRAM Rules

Storage topology and the critical VRAM safety limits for the B70.

Last updated: 2026-08-18 (lv_models remount + partial-VG boot trap).

---

## 0. ⚠️ 2026-08-05 event: T-Force wiped by Windows

The T-Force 500 GB disk was repartitioned in Windows (~343 GB NTFS for games).
This DESTROYED the LVM PV that hosted `lv_models` (458 GB) → `/mnt/<models-disk>`.
All models on it were lost (Q6_K-MTP, Q5_K-MTP, Ornith Q5, MXFP4-native, BF16 source).

Recovery performed same day:
1. Recreated partition `sda3` (146.5 GB) in the remaining free space
2. `pvcreate /dev/sda3` → `vgreduce --removemissing --force vg_data` (dropped the dead
   PV + broken lv_models) → `vgextend vg_data /dev/sda3`
3. `lvcreate -n lv_models -l 100%FREE` (146.5 GB) → ext4 → mounted at `/mnt/<models-disk>`
4. Re-downloaded (aria2c -x16 on resolved Xet CDN, ~86 MB/s): Q6_K-MTP (20.5 GB),
   Q5_K_M-MTP (19 GB) from `bytkim/Qwen3.6-27B-MTP-pi-tune-GGUF`; Ornith Q5_K_M
   (21.8 GB) from `ornith-ai/Ornith-1.0-35B-GGUF`. All GGUF-magic verified.
5. `/mnt/<models-disk>` now has **~106 GB free** — budget: Q6 20.7 + Q5 19 + Ornith 21.8
   (already there) + MXFP4-native 22.4 (rebuildable) + BF16 67 (too big — do NOT
   re-download unless needed).

**Losses not restored:** Qwen3.6-35B-A3B-MXFP4-native (our vLLM checkpoint — rebuildable
from community MXFP4 + scripts/tmp/b70-mxfp4-merge.py if the vLLM path is needed again)
and the 67 GB BF16 source (deferred — not needed for serving).

---

## 1. Physical disks (roles, not names — /dev names shifted after the wipe)

| Role | Device (current) | Size | Contents |
|------|------------------|------|----------|
| Linux system | `/dev/sdc` | 466 GB | root (sdc4), /mnt/<models-disk> (sdc3), swap, EFI |
| Windows NVMe (C:) | `/dev/nvme0n1` | 931 GB | NTFS 930 GB |
| Windows D: | `/dev/sdb` | 931 GB | NTFS 882 GB (mounted /mnt/<other-disk>, ro) + sdb3 LVM PV 49.6 GB (lv_data) |
| T-FORCE 500 GB | `/dev/sda` | 466 GB | sda2 NTFS 343 GB (games) + **sda3 LVM PV 146.5 GB (lv_models)** |

---

## 2. Mount points and usage

| Mount | Device | Size | Used | Free | Purpose |
|-------|--------|------|------|------|---------|
| `/` | `/dev/sdc4` | 214 GB | 185 GB (92%) | ~18 GB | Root filesystem — **tight** |
| `/mnt/<models-disk>` | `/dev/sdc3` | 229 GB | 182 GB (84%) | 35 GB | Primary GGUF models (raw ext4) |
| `/mnt/<models-disk>` | `/dev/vg_data/lv_models` | **146.5 GB** | 40 GB | **~106 GB** | Extra quants (LVM, recreated 2026-08-05) |
| `/mnt/<other-disk>` | `/dev/sdb2` | 882 GB | — | — | Windows D: (read-only) |
| `/mnt/<other-disk>` | `/dev/vg_data/lv_data` | 49.6 GB | — | — | Inference experiments (INTACT on sdb3) |

**Symlinks:**
- `~/models-cache` → `/mnt/<models-disk>/gguf/` (primary models)
- `~/models2-cache` → `/mnt/<models-disk>/` (extra storage)

---

## 3. LVM state

```
Volume Group: vg_data (196 GB total, after 2026-08-05 recreation)
  ├─ PV /dev/sda3  (146.5 GB)  ← NEW 2026-08-05 (replaced the wiped 465.8 GB PV)
  └─ PV /dev/sdb3  (49.6 GB)   ← intact (lv_data)

Logical Volumes:
  ├─ lv_data   (49.6 GB)   → /mnt/<other-disk>  (preserved)
  └─ lv_models (146.5 GB)  → /mnt/<models-disk>            (recreated)
```

**fstab entry** (UUID-based, `nofail` so a future Windows wipe won't block boot):
```
UUID=<blkid -s UUID -o value /dev/vg_data/lv_models> /mnt/<models-disk> ext4 defaults,nofail 0 2
```

**Lesson:** the LVM PV lived on a dual-boot disk that Windows can repartition at any
time. Keep only re-downloadable quants on the T-Force; primary models on `/mnt/<models-disk>`
are safer.

### 3.1 2026-08-18 remount (this host, live)

Device names drifted again. Current Linux root is `/dev/sda4`; `/mnt/<models-disk>`
is `/dev/sda3`. The models2 LV is **`/dev/sdb3` → `vg_data/lv_models`**
(146.48 GiB, ext4 UUID `a1957aa3-1877-4a57-a024-ccdca74481ae`).

`vg_data` is **partial**: missing PV `IJ2FDM-lD2Y-Yh0V-sjO4-IJCi-fERg-hcjY0k`.
`lv_data` is NOT available. systemd therefore skipped auto-activation of the
intact `lv_models`, so `/mnt/<models-disk>` was an empty mountpoint after boot
(`fstab` `nofail` succeeded with nothing underneath). That is why every
`models2` path looked deleted.

Fix (do **not** activate `lv_data`):
```
sudo lvchange -ay vg_data/lv_models
sudo mount /mnt/<models-disk>
```
Persisted as `vg-data-lv-models.service` (oneshot, before `mnt-models2.mount`).

Live after remount: 143.1G, **53.1G free**, 82.7G used. Present: Qwen3.6-27B
GPTQ-INT4 (19G), Qwen3.6-35B GPTQ-INT4 (23G), Qwen3.8 nogidx GPTQ (19G),
Qwen3.8 BF16 shards (15G), Nemotron DFlash-BF16 (1.6G), leftover
`Qwen3.8-27B-DSpark-vLLM` (2.6G — not a served cell). **Still absent:**
ThinkingCap Q4/Q5/Q6 GGUF, Qwen3.6 UD GGUF, Laguna, Ornith Q5. Rock 5B
(`radxa@<lab-host>`) has none of those either.

---

## 4. Models on disk (2026-08-05, post-wipe re-download)

**Location: `/mnt/<models-disk>/gguf/` (primary, ~100 GB total) — untouched by the wipe**

| Model | Size | arch | Type | Status |
|-------|------|------|------|--------|
| ThinkingCap-Qwen3.6-27B-Q4_K_M | 15 GB | qwen35 | Dense 27B (MTP) | ✅ Available (fallback profile) |
| Qwen3.6-35B-A3B-UD-Q5_K_M | 24 GB | qwen35moe | MoE 35B | ✅ Available |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | 20 GB | qwen35moe | MoE 35B | ✅ Available |
| Laguna-S-2.1-UD-IQ2_XXS | 34 GB | laguna | Dense | ⚠️ See VRAM rule below |
| laguna-s-2.1-DFlash-BF16 | 2 GB | dflash | Spec decoder | Companion to Laguna |
| dflash-draft-3.6-q8_0 | 2 GB | dflash-draft | Spec decoder | Not standalone-loadable |
| mmproj-ThinkingCap-Qwen3.6-27B-f16 | 889 MB | clip | Vision projector | For ThinkingCap 27B |
| mmproj-deepreinforce-ai_Ornith-1.0-35B-f16 | 859 MB | clip | Vision projector | For Ornith (on models2) |

**Location: `/mnt/<models-disk>/` (re-downloaded 2026-08-05, ~62 GB total, 106 GB free)**

| Model | Size | Source repo | Status |
|-------|------|-------------|--------|
| ThinkingCap-Qwen3.6-27B-Q6_K-MTP | 20.5 GB | `bytkim/Qwen3.6-27B-MTP-pi-tune-GGUF` (file `Qwen3.6-27B-MTP-pi-tune-Q6_K.gguf`) | ✅ **ACTIVE PRODUCTION** @128K MTP |
| ThinkingCap-Qwen3.6-27B-Q5_K_M-MTP | 19 GB | `bytkim/Qwen3.6-27B-MTP-pi-tune-GGUF` (`...-Q5_K_M.gguf`) | ✅ Available (200K no-MTP) |
| ornith-1.0-35b-Q5_K_M | 21.8 GB | `ornith-ai/Ornith-1.0-35B-GGUF` | ✅ Available |

**LOST in the wipe (not restored):** Qwen3.6-35B-A3B-MXFP4-native (vLLM checkpoint,
rebuildable), Qwen3.6-35B-A3B-bf16 (67 GB — do not re-download unless rebuilding MXFP4).

**Download new models/quants to** `/mnt/<models-disk>/` (106 GB free — half the old capacity).

---

## 5. ⚠️ Critical VRAM rule: never load >30 GB models

The B70 has **32,656 MiB visible VRAM**. Loading a single GGUF larger than ~30 GB
**overflows VRAM and causes a hard system crash** (full reboot, not a cleanable OOM).

**Verified 2026-08-03:** `Laguna-S-2.1-UD-IQ2_XXS.gguf` (34 GB) crashed the PC
on load. The system recovered after reboot with all volumes intact.

**If you must run a >30 GB model:**
```bash
# Offload tail layers to CPU (reduces VRAM usage)
-ngl 40    # instead of -ngl 99 — keeps only first 40 layers on GPU
# OR use the Laguna profile which offloads experts:
-ot "blk.(4[0-9]).ffn_.*_exps.=CPU"
```

---

## 6. Reading VRAM (debugfs)

```bash
sudo cat /sys/kernel/debug/dri/0000:0b:00.0/tile0/vram_mm
```

Key fields:
- `visible_avail: 31730MiB` — free VRAM (strip non-digits if parsing)
- `usage: 28247109632` — used VRAM in bytes (divide by 1048576 for MiB)
- `visible_size: 32656MiB` — total VRAM

**Before loading any model**, check free VRAM:
```bash
free_mib=$(sudo cat /sys/kernel/debug/dri/0000:0b:00.0/tile0/vram_mm | awk -F': ' '/visible_avail/ {gsub(/[^0-9]/,"",$2); print $2}')
```

Model size (GB) + ~4 GB KV cache (at 128K, q8_0-q4_1) must be < free VRAM.

---

## 7. Reading power and temperature

```bash
# Power cap (microwatts)
cat /sys/class/hwmon/hwmon4/power1_cap
# Set: echo 150000000 | sudo tee /sys/class/hwmon/hwmon4/power1_cap

# Temperature (millidegrees C — divide by 1000)
cat /sys/class/hwmon/hwmon4/temp2_input

# Idle: ~49°C. Under load at 150W: 63-70°C. At 230W: 76-78°C.
```

⚠️ **Use `temp2_input`, not `temp1_input`** — `temp1_input` does not exist on hwmon4.
Several launcher scripts had this bug (fixed 2026-08-03).
