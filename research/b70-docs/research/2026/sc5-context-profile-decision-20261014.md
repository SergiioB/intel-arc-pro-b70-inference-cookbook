---
title: "SC5 context profiles: operator-selected 196K inference default"
type: research
date: 2026-10-14
status: validated_internal
---

# 190K+ context is tested, not a desktop-only feature

The operator explicitly selected the 196K single-request inference setting after reviewing its lower free-VRAM reserve. The initial capacity report's conservative proposed default is not the current launcher recipe.

| Profile | Configured context | Highest checked occupied tokens | Memory utilization | Free VRAM under load |
|---|---:|---:|---:|---:|
| Preferred single-request inference | 196,608 | 196,438 | 0.94 | 1,069 MiB |
| Higher-reserve alternate | 131,072 | 130,872 | 0.90 | 2,327 MiB |

Both profiles used the same Qwen3.8-27B SC_5.00bpw_H6 artifact, image 20261014 digest, FP8 KV, MTP3 and exact retrieval/multi-turn checks. API and headless CLI use the same serving path as the desktop launcher. The larger setting is a low-headroom capacity observation, not a general concurrency or performance recommendation.

Retained capacity evidence (`../../results/2026/10/ctx-capacity-sc5-20261014/REPORT.md`) remains unchanged. Its budget ceiling and top-rung timeouts are not completed-request evidence. Cache-hit counters were unavailable; no cold-speed claim is made.

Launcher 0.6.7 implements context-profile dispatch using recipe fields: 0.90 at or below the 131,072-token alternate, 0.94 above it. Explicit memory overrides are never silently increased. A low-utilization override with context beyond its known pool budget is refused. The 206,000-token configured budget ceiling remains unverified above the tested 196K stage.

Plain EXL3 4.0bpw's separate 262,144-token route and Flash-Next EXL3 port results must not be conflated with SC5 capacity.
