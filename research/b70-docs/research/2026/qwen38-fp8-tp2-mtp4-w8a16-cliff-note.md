---
title: "Cliff probe addendum — 2026-08-29 ~12:00 UTC"
type: research
date: 2026-09-15
status: final
---

# Cliff probe addendum — 2026-08-29 ~12:00 UTC

Sweep at the claimed boundary (`LABEL=cliff`,
`results/2026/08/qw38-fp8-tp2-mtp4-w8a16-20260829T104757Z/`, same W8A16 server,
random draws, seed-free): p1024 44.35 / p2040 35.30 / p2047 40.94 /
**p2048 35.74** / p2049 24.21 / p2056 38.01 tok/s.

**The 0.0%-acceptance p2048 collapse did NOT reproduce.** Same server, same
patch, same bench — only the content draw differed. Reclassified: the
p2048/p6144 collapse in the first W8A16 bench was a **draw-dependent
acceptance collapse** (numerics x content interaction at longer contexts),
not a deterministic M>=2048 structural bug. W8A16 beyond ~1.5k tokens remains
RISKY (occasional total acceptance collapse on some draws); the operational
routing rule (W8A16 <=1536, W8A8 beyond) stays, and any W8A16 long-context
use must gate on live acceptance metrics.
