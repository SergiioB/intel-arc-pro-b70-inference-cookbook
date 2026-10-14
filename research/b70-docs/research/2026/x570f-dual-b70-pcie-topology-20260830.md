---
title: "Dual-B70 PCIe Topology — ASUS ROG STRIX X570-F GAMING (current host)"
type: research
date: 2026-08-30
status: final
---

# Dual-B70 PCIe Topology — ASUS ROG STRIX X570-F GAMING (current host)

**Date:** 2026-08-30 · **Status:** Active topology authority (supersedes `research/2026/b450f-dual-b70-pcie-topology-20260824.md` — the B450-F board was swapped out on 2026-08-27; all campaign results from 2026-08-27T23:10Z onward were measured on THIS board)

## Board verification (dmidecode)

- Manufacturer: ASUSTeK COMPUTER INC. · Product: **ROG STRIX X570-F GAMING** (Rev X.0x)
- Slots (dmidecode type 9): PCIEX16_1 → `00:03.1` · PCIEX16_2 → `00:03.2` · PCIEX16_3 (chipset) → unused for GPUs

## GPU placement (lspci hierarchy)

| GPU | BDF | Upstream path | Slot |
|---|---|---|---|
| GPU 0 | `0000:0b:00.0` | `00:03.1` → `09:00.0` → `0a:01.0` → `0b:00.0` | PCIEX16_1 (CPU-attached) |
| GPU 1 | `0000:0f:00.0` | `00:03.2` → `0d:00.0` → `0e:01.0` → `0f:00.0` | PCIEX16_2 (CPU-attached) |

Both cards hang off **CPU root ports** (AMD Starship/Matisse GPP bridges `00:03.1`/`00:03.2`) — dual x8/x8 CPU-attached slots. No GPU-P2P interconnect exists on this topology (same conclusion as the B450-F; TP2 collectives remain host-staged, ADR-0008 unchanged).

> **Correction — 2026-10-07 (RL-059):** the "no GPU-P2P" sentence above is wrong as a
> capability claim on this board. Direct-peer P2P **copy** is verified: bidirectional,
> checksum-clean, ~11.1–11.9 GB/s vs 5.86 GB/s host-staged, ACCESS flag set, **ATOMICS
> absent**. Evidence:
> `results/2026/09/b70-improvement-probes-20260920/p2-levelzero-p2p-probe-20260924T194500Z/`.
> TP2 policy is unchanged (ADR-0008): copy-based collectives stay gated research
> (`research/PROGRAM.md` RQ-06).

## Measured bandwidth (2026-08-30, torch H2D copy probe, 512 MiB × 8)

| GPU | H2D |
|---|---:|
| GPU 0 (`0b:00.0`) | 8.85 GB/s |
| GPU 1 (`0f:00.0`) | 7.89 GB/s |

**Both cards now measure ≥7.9 GB/s.** The B450-F era "GPU 1 = Gen3 x4 ≈ 3.6 GB/s" second-slot bottleneck is gone. The probe is CPU-copy-limited — treat these as floors; a proper multi-GB bandwidth audit is pending. (Gen4 x8 slots: theoretical 15.8 GB/s each.)

## lspci LnkSta is unreliable on these endpoints

`lspci -vv` reports `LnkSta: Speed 2.5GT/s, Width x1` on BOTH cards even while the devices are active and delivering ~8-9 GB/s (also observed during the 08-29 campaigns). This is an endpoint firmware/runtime-PM reporting quirk, not a real link state. **Do not use LnkSta on this host; use measured H2D/D2H bandwidth.**

## Runtime-PM rules (unchanged)

Pin `power/control=on` up the PCI chain for both GPUs after any power/rebind event (the `b70-gpu1` tool does this for GPU 1; GPU 0 was pinned manually on 2026-08-30). With `control=auto` the cards drop into a state where LnkSta misreads and post-resume link flapping was observed on the previous board (incident 2026-08-24). Post-pin verification: H2D probe above.

## Cross-references

- B450-F historical measurements (Gen3 x8 + Gen3 x4, H2D 7.1/3.6 GB/s): `research/2026/b450f-dual-b70-pcie-topology-20260824.md` (SUPERSEDED for current-host claims)
- TP2 product decision (independent stacks default, host-staged collectives): `decisions/adr-0008-dual-b70-independent-default.md` (conclusions unchanged on X570-F)
- All campaign results from 2026-08-27T23:10Z onward (bat0-bat4, W8A16/small-M/k-screen series, dflash2) were measured on THIS board
