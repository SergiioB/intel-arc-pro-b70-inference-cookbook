---
title: "ASUS ROG STRIX B450-F Gaming — dual-B70 PCIe topology and capability map (2026-08-24)"
type: research
date: 2026-08-24
status: final
---

# ASUS ROG STRIX B450-F Gaming — dual-B70 PCIe topology and capability map (2026-08-24)

Status: `SUPERSEDED 2026-08-30` — the motherboard was swapped to an ASUS ROG STRIX X570-F GAMING (dual CPU-attached x8/x8) on 2026-08-27; every measurement below was taken on the previous B450-F board. Current topology authority: `research/2026/x570f-dual-b70-pcie-topology-20260830.md`. Historical value: records the B450-F Gen3 x8 + Gen3 x4 second-slot bottleneck that the X570-F x8/x8 wiring removes.

Status at capture: `active` — live-observed host evidence, 2026-08-24 (b70-host, boot 11:25:50).
The two bandwidth numbers in §4 are diagnostic single-run best-of-5 microbenchmarks
(`scripts/tmp/pcie_bw_sycl.cpp`), not campaign-grade medians; they match link-theory
efficiency (≈90 %) and are sufficient for placement decisions.

Dual-card **product** choice (independent stacks vs vLLM TP2 vs llama split) is
[ADR-0008](../../decisions/adr-0008-dual-b70-independent-default.md). External TP2
recipe capture: [`humble-b70-llm-dual-tp2-20260824.md`](humble-b70-llm-dual-tp2-20260824.md).

## 1. Why this document exists

Two Arc Pro B70s are installed. Whether "dual stack inference" means two independent
engines (one per card) or one model split across both cards is decided by the
motherboard's slot wiring, link widths, and memory — all mapped below from the live
host, not the spec sheet.

## 2. Host inventory

| Component | Observed |
|---|---|
| Board | ASUSTeK ROG STRIX B450-F GAMING, Rev 1.xx |
| BIOS | AMI 6001, 2026-05-21 |
| CPU | Ryzen 7 5700X3D (Vermeer, 8C/16T), 1 NUMA node |
| RAM | 4 × 8 GB DDR4-2133 (JEDEC 2133, 1.2 V, all four slots populated), 30 Gi usable, dual channel ≈ 34 GB/s theoretical |
| GPUs | 2 × Arc Pro B70 (`8086:e223`), `xe` driver, 32,656 MiB visible VRAM each |
| IOMMU | AMD IOMMU active, 32 groups |
| Resizable BAR | Active on both B70s — GPU function BAR 2 current size **32 GB** (full VRAM); above-4G decoding working |

## 3. Live PCIe topology (Matisse/Vermeer root complex)

```text
CPU root complex (0000:00)
├─ 00:01.1 → NVMe SSD (SanDisk/WD SN750 class)          CPU lanes, not chipset
├─ 00:01.3 → B450 chipset (FCH)                          xHCI, SATA, I211 GbE,
│                                                          empty x1 buses 05–08
├─ 00:03.1 → [Gen3 x8] → B70 #0 bridge 09:00.0 (Gen5 x16 cap)
│                            └─ 0a:01.0 → 0b:00.0 GPU0 + 0c:00.0 HDMI audio
└─ 00:03.2 → [Gen3 x4] → B70 #1 bridge 0d:00.0 (Gen5 x16 cap)
                             └─ 0e:01.0 → 0f:00.0 GPU1 + 10:00.0 HDMI audio
```

Key facts, all from `lspci -vv` LnkCap/LnkSta on 2026-08-24:

| Path | Root port cap | Card bridge LnkSta | Theoretical |
|---|---|---|---|
| GPU 0 (`0b:00.0`) | 00:03.1 = Gen3 **x8** | Gen3 x8 (downgraded from card's Gen5 x16 cap) | 7.88 GB/s |
| GPU 1 (`0f:00.0`) | 00:03.2 = Gen3 **x4** | Gen3 x4 | 3.94 GB/s |
| NVMe | CPU-attached | — | Gen3 x4 class |

- The B70's on-card switch advertises PCIe 5.0 x16; the **board/CPU wiring is the
  limiter on both slots**. Neither card runs Gen4/Gen5 on B450.
- The primary physical x16 slot runs **x8**, and the second slot **x4** — both
  CPU-attached (not chipset). NVMe and the chipset uplink are separate CPU lanes,
  so GPU traffic never crosses the FCH.
- GPU function-level `LnkSta` (2.5 GT/s x1) is the SR-IOV PF representation
  ("Running in SR-IOV PF mode"); the authoritative host link is the card bridge
  above each GPU, as tabulated.
- No ACS on the GPU bridges; consumer AM4 root ports provide no usable P2P —
  inter-GPU copies bounce through host RAM.

## 4. Measured host↔device bandwidth (Level Zero, 256 MiB pinned, best of 5)

| GPU | H2D | D2H | Link efficiency |
|---|---|---|---|
| GPU 0 (Gen3 x8) | 7.10 GB/s | 7.09 GB/s | ≈ 90 % |
| GPU 1 (Gen3 x4) | 3.60 GB/s | 3.57 GB/s | ≈ 91 % |

Source: `scripts/tmp/pcie_bw_sycl.cpp`, run 2026-08-24, `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`.

## 5. The GPU 1 flapping incident (root cause + fix)

After the first PCI remove/rescan power cycles, GPU 1 flapped: present → silent link
drop 30–60 s later → hotplug re-add → drop again (dashboard log 12:01–12:03;
kernel showed a clean teardown at each drop with **no** AER/fault/reset messages).

Root cause: the whole GPU 1 chain (`0f:00.0`, on-card bridges, root port 00:03.2)
was in `power/control = auto`; runtime suspend of the chain on this slot silently
kills the link, and the next resume/enum cycle repeats it. GPU 0 never flapped
because the desktop keeps its chain runtime-active.

Fix (verified): pin `power/control = on` for the GPU and every ancestor bridge.
`b70-gpu1 on` now does this automatically (2026-08-24); with the pin applied the
card has been stable across subsequent off/on cycles and idle periods.

## 6. What this means for dual-stack inference decisions

**Recommended: two independent stacks, one card per GPU.**
Zero inter-GPU traffic, each card keeps its full 32 GB, and the existing serving
lanes (llama.cpp SYCL profile service, vLLM containers) already work per-card via
`ZE_AFFINITY_MASK`. The x4 link of GPU 1 is irrelevant to steady-state serving
because weights are resident; it only slows cold model loads.

Decision-relevant consequences of the measured topology:

1. **Cold model load on GPU 1 is ~2× slower.** A 30 GB Q4 GGUF streams at
   ≈ 3.6 GB/s → ≈ 8–9 min load on GPU 1 vs ≈ 4 min on GPU 0 (plus conversion
   overheads). Plan loads ahead; keep frequently-served models resident on GPU 1.
2. **Single model across both cards (llama.cpp layer/tensor split) is viable for
   decode-focused use.** Per-token cross-card traffic is one activation
   (a few MB/token on 27–35 B models) — negligible on Gen3 x4. Prefill batches
   stream larger activations; expect prefill scaling well below 2×. Combined
   64 GB VRAM opens 60 GB+ single-model placements otherwise impossible on this
   host. This is topology-viable, not a current launcher product.
3. **vLLM TP=2 is a gated research path, not the dual-card default.** Stock
   tensor-parallel allreduce still has to bounce through host RAM over the x4
   link with no P2P; do not expect 2× and do not launch stock Docker TP2.
   An external recipe ([humble-b70-llm](https://github.com/JP-devv/humble-b70-llm),
   captured in [`humble-b70-llm-dual-tp2-20260824.md`](humble-b70-llm-dual-tp2-20260824.md))
   shows host-staged collectives can still scale decode ~1.4× vs TP1 on this
   board class (second slot x4, no peer IPC), with **worse TTFT and prefill**.
   Those rows are not this host. Policy: [`decisions/adr-0008-dual-b70-independent-default.md`](../../decisions/adr-0008-dual-b70-independent-default.md).
4. **CPU-offload paths (partial `-ngl`) degrade on GPU 1.** Any scheme that
   streams weights from host RAM per token is capped at the link speed; GPU 1's
   x4 makes partial offload there strictly worse than on GPU 0.
5. **KV spill / swap-to-host** (vLLM `--swap-space`, llama mmap paging) inherits
   the same link ceilings.
6. **GPU 0 carries the desktop** (display + Brave ≈ 0.7–1 GB VRAM, compositor
   load). For clean Cn campaigns on GPU 0, remember the display is attached
   (cannot be powered off); GPU 1 is the clean card for contained experiments —
   including the b70-gpu1 power-off workflow.

## 7. Idle power draw (2026-08-24, xe hwmon energy counters)

Method per observability standard: `Δ energy / Δ monotonic time`, single 60.00 s
quiescent window per card, resolved paths `hwmon4`/`hwmon5` → `energy1_input`
(card domain, µJ) and `energy2_input` (pkg domain, µJ). Diagnostic single-window
evidence, not campaign-grade.

| Card | State | Card domain | Pkg domain | Temp / cap |
|---|---|---|---|---|
| GPU 0 | D0 idle, desktop attached | **51.31 W** | 30.17 W | 56 °C / 180 W |
| GPU 1 | D0 idle, runtime PM pinned on | **44.66 W** | 25.83 W | 55 °C / 150 W |
| GPU 1 | "suspended" (function-level D3, bridges pinned) | **44.51 W** | 38.40 W | resume-contaminated |

Findings:

1. Function-level runtime suspend (`power/control=auto` on `0f:00.0` only) saves
   **nothing measurable** — the card keeps drawing ≈ 45 W while
   `runtime_status` reports `suspended`. Real power gating on this board only
   happens with the full chain down (which caused the link flapping in §5) or
   with PCI remove.
2. Reading `energy*_input` **resumes** a runtime-suspended GPU, so D3 windows
   are self-corrupting unless the end-read is the only wake event.
3. The off state (`b70-gpu1 off`, PCI remove) is **not measurable in software** —
   the counters vanish with the device. Quantifying it needs a wall meter.
4. Practical cost of leaving GPU 1 powered on at idle: ≈ 45 W continuous
   (≈ 0.95 kWh/day at 21 h idle).
5. Why D0 idle is ≈ 45 W and not review-style ~20 W: `dgfx_pkg_residencies`
   (xe debugfs, since boot) shows the die **never** reaches deep package states —
   PC6 = 0, PC10 = 0, only PC2/PC8 — and PCIe L1.2 residency = 0. Deep idle
   (PC6+/L1.2/D3) is gated behind runtime suspend, which this board cannot do
   stably, and the card is also touched every 2 s by dashboard telemetry.
   The ≈ 45 W is the stable-idle price on this host, not a defect of the card.

Raw evidence (GPU 1, D0 window): card `132897192199 → 135576849121 µJ` over
`60.00 s`; GPU 0: `150286864501 → 153365520874 µJ` over `60.00 s`.

## 8. Operational notes

- `b70-gpu1 on` restores 150 W cap and pins runtime PM `on` up the PCI chain;
  both are required for stability on this board.
- Render minors and hwmon indices renumber across power cycles (observed
  renderD129 → renderD130); always re-discover paths.
- Do not run `b70-gpu1 off` during a live campaign; its guards refuse when
  inference processes hold the card, and `--force` bypasses that protection.
- The power toggle is driver-level (PCI remove, D3); it is not a physical
  power cut — the card's standby draw remains.
