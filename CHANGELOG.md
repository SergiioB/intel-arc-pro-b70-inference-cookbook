# Changelog

## [1.1.1](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/compare/v1.1.0...v1.1.1) (2026-09-11)


### Bug Fixes

* **flash-next:** repin evidence to full post-rebase SHA ([0565f3a](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/0565f3a52896a06dbba605b93a9690584c0dfaab))

## [1.1.0](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/compare/v1.0.0...v1.1.0) (2026-09-10)


### Features

* add canonical benchmark catalog ([c0aaa71](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/c0aaa713ce63a6db86f6ced2bd7ce898cac4d6f5))
* **gates:** readiness hardening - CI, quality gates, structured logging, docs ([e32d3fd](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/e32d3fdd6ad61524e87d809241d471f6f7bbb205))
* **patches:** add worker affinity (TP2/PP2) and FP8 W8A16 linear reroute patches ([0f876e1](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/0f876e141c7a45b9db6081fba53b207b150206ca))
* **templates:** hardware result issue template for B50/B60/B65 and other B70 counts ([e4fef2e](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/e4fef2ea9a99c5767a17a2f26c494ac812f43881))


### Bug Fixes

* guard draft INT4 under tensor parallelism ([d9c95e5](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/d9c95e5cb54fa5588f871b09e1fdd0162180da39))
* **qwen38:** compact+scatter GDN split (v5) and int64-only ptr wrap ([db20e00](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/db20e002fee4bc14180e7f851fd0d82473fa6273))
* **qwen38:** concurrency + pointer-safety patches for concurrent MTP4 serving ([455e913](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/455e9133dfcf1b67b1e4aa678868ae55f0e8ec86))
* **scripts:** make host-preflight stdlib-only (psutil -&gt; /proc/meminfo) ([99a170c](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/99a170c388ee7d4006fa63457cdcff6e3d9b4f62))


### Performance Improvements

* **qwen38:** INT4 speculative draft (Phase S+M1) for higher MTP decode ([aa363ca](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/aa363cad7b484173b6a228224cb005f8c16bcd4d))
