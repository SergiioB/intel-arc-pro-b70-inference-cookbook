# Changelog

## Unreleased

- Put measured wins, failed routes, unpublished lab screens, and ranked stop/go decisions in `docs/WHAT-WORKED.md`; shorten the README chooser to one entry per goal.
- Qualify historical MoE and TP2 headlines, distinguish the quick-start BF16 drafter from the draft-INT4 result, and add required tool-call parser flags to `START.md`.
- Close the own-corpus shortlist as an invalid thermal screen without a speed claim; point readers to the already verified draft-INT4 result instead of scheduling a duplicate A/B.

## [1.3.0](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/compare/v1.2.4...v1.3.0) (2026-09-20)


### Features

* **ux:** product overhaul with human START.md, inverted README, stack doctor, canary probes, and APC+MTP guards ([e91d2c3](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/e91d2c328a3270c368a385ba255654dfa38085d0))


### Bug Fixes

* **lint:** format, mypy types, and line lengths for doctor and canary scripts ([b24d260](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/b24d2601fd6bba576b4a93e9fc2c99b302865cdc))
* **qwen38-27b:** tool-calling flags on all serve commands; qwen3_xml parser guidance ([6dab23f](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/6dab23fcb0b7faa19a3fc6c0c6d409ccf390b6e8))
* **qwen38-27b:** tool-calling flags on all serve commands; qwen3_xml parser guidance ([dfd132b](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/dfd132bd45b9dcc31a3c88269028e5f6e247958b))

## [1.2.4](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/compare/v1.2.3...v1.2.4) (2026-09-15)


### Bug Fixes

* **data:** stale OV numbers swept; Cascadia split as its own engine ([150bed4](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/150bed477cba66d18f656a91afec13e2a14c6f2b))
* **data:** stale OV numbers swept; Cascadia split as its own engine ([676c371](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/676c371f8e16a74ae505638486fbaa7b1e4b2098))

## [1.2.3](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/compare/v1.2.2...v1.2.3) (2026-09-15)


### Bug Fixes

* **docs:** finish folder unification — remaining bare qwen38-27 mentions in FEATURE-FLAGS now point at qwen38-27b ([443362e](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/443362e9592cc6aa3ee0d3eccf25137c36385d93))

## [1.2.2](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/compare/v1.2.1...v1.2.2) (2026-09-15)


### Bug Fixes

* **ci:** repair BENCHMARK-CATALOG link + add FP8 TP2 measured sweep as pinned record (26 rows) ([83fbdb2](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/83fbdb270746c9b9c95fc8551332f66770883f95))
* **ov:** correct 256K overclaim — full-protocol ceiling 128K; capacity probes labeled ([4583e84](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/4583e84e192b25467f75afd9d4b1619cca4499f9))
* **ov:** correct 256K overclaim — full-protocol ceiling is 128K; 196K-256K are 40-token capacity probes ([9b34742](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/9b347420b132da1e15e755d35467d4b65f591009))
* shorten FP8 TP2 catalog record wording for mobile rendering ([472cb40](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/472cb40c13a222233a0d791c93d7d4863c6e4b1c))

## [1.2.1](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/compare/v1.2.0...v1.2.1) (2026-09-15)


### Bug Fixes

* **ci:** repair BENCHMARK-CATALOG link + add FP8 TP2 measured sweep as pinned record (26 rows) ([83fbdb2](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/83fbdb270746c9b9c95fc8551332f66770883f95))
* shorten FP8 TP2 catalog record wording for mobile rendering ([472cb40](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/472cb40c13a222233a0d791c93d7d4863c6e4b1c))

## [1.2.0](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/compare/v1.1.1...v1.2.0) (2026-09-13)


### Features

* **flash-next:** publish MTP draft-head + fused multi-token MoE kernel patches ([72dc153](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/72dc15382656bec35d2a7e4d6518f8a7298ba01d))


### Bug Fixes

* **flash-next:** document multi-token kernel ids-transpose fix (wrong experts in verify; quality restored, speed unchanged) ([31b3707](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/31b370710d95a023fe7d1c05f536983f59594790))

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
