# Changelog

## Unreleased

- Issue triage: stop labelling from maintainer replies and from issue-form headings. A reply saying "only acts on TP2 or PP2" kept re-adding `pp:2` to the TP4 report #38, and the bug form's "Logs / dmesg / coredump" heading tagged every bug report `driver-hang` (#15). Re-classifying all 60 issues changes only #38, #15 and #8, each by dropping a false label.
- Windows Docker kit: update the SHA-256 pins for `patch_draft_lmhead_int4.py` and `patch_draft_mtp_int4.py`. The TP guard commit `d9c95e5` changed both files but not the pins, so `Build-Qwen38Image-Docker.ps1` stopped with a hash mismatch (#15).
- Windows Docker and WSLC kits: add `patch_xpu_only_platform.py`. On hosts that also have an NVIDIA GPU, WSL exposes NVML in the container and vLLM stopped with `Only one platform plugin can be activated, but got: ['cuda', 'xpu']` (#15). Rebuild with `Upgrade-Qwen38-Docker.ps1`.
- `patch_vllm_worker_affinity.py`: make the `xpu_worker.py` device-remap hunk idempotent. Its marker never matched the injected text, so a second run in the same container (any restart) failed with `anchor count 0 != 1` and the startup chain exited. Checked on the `f01e24f6` image trees with the #38 order (affinity, accepted-sync, eagle-drop, backward-copy) applied twice: all exit 0, 3/3 backward-copy guards (#38).
- Qwen3.8-27B SC5: measure int8 vs fp16 prefill as served. int8 shifts the held-out chat distribution by KL 0.00099, about 70 % of SC5's weight KL to BF16. fp16 prefill reads cold prompts 1.63×/1.54×/1.34×/1.28× slower at 8K/32K/131K/190K. Document the opt-in `--prefill fp16` mode (b70-launcher 0.6.10) and the `--set env.EXL3_INT8_PREFILL=0` serve flag.
- Put measured wins, failed routes, unpublished lab screens, and ranked stop/go decisions in `docs/WHAT-WORKED.md`; shorten the README chooser to one entry per goal.
- Qualify historical MoE and TP2 headlines, distinguish the quick-start BF16 drafter from the draft-INT4 result, and add required tool-call parser flags to `START.md`.
- Close the own-corpus shortlist as an invalid thermal screen without a speed claim; point readers to the already verified draft-INT4 result instead of scheduling a duplicate A/B.
- Qwen3.8-27B SC5: publish the KL screen vs BF16 (SC5 KL 0.418×/0.161× of plain 4.00bpw), the SC_5.00bpw_H6_V6 check (text-identical, does not load), and pin `exl3xpu-mixedk:20261014` (same layers as `20261013`; source: [SergiioB/exl3xpu-mixedk](https://github.com/SergiioB/exl3xpu-mixedk); prior `20261012c` behaves the same) with the one-command launcher path.

## [1.8.2](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/compare/v1.8.1...v1.8.2) (2026-10-29)


### Bug Fixes

* **ledgers:** satisfy ruff E501/C901, ruff format, and mypy strict ([2097d3b](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/2097d3b008060e746af1cfd90b44a500cf7fa72f))

## [1.8.1](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/compare/v1.8.0...v1.8.1) (2026-10-25)


### Bug Fixes

* **triage:** ignore maintainer replies and issue-form headings ([85a84ab](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/85a84ab7afdedebd394447023538cc0598ebdb92))

## [1.8.0](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/compare/v1.7.1...v1.8.0) (2026-10-25)


### Features

* **catalog:** Ornith-1.5-122B IQ3_XXS dual-B70 capability record ([1d24085](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/1d24085b214a7bbb151fdadc0d2c8356550317fd))
* **catalog:** Ornith-1.5-122B IQ3_XXS pp2048 700 and tg128 28.1 ([5179916](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/51799160aef3509e6105ffdfe1aedd3a86474348))


### Bug Fixes

* **patches:** make worker-affinity xpu_worker hunk idempotent ([f631905](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/f6319055fd3614e67e4d25760c11a632003575cb)), closes [#38](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/issues/38)
* **windows:** rebuild pins and XPU-only platform on NVIDIA hosts ([3755535](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/37555353bc590a1818c5d6d8df9b88d1b8944a2e)), closes [#15](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/issues/15)

## [1.7.1](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/compare/v1.7.0...v1.7.1) (2026-10-17)


### Bug Fixes

* **recipes:** set Bolt model-card sampling on both routes ([#47](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/issues/47)) ([5a00d06](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/5a00d06a9f0581136e9be0d0dcd5fc0b0610fc7f))

## [1.7.0](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/compare/v1.6.0...v1.7.0) (2026-10-16)


### Features

* **recipes:** Nemotron 3.5 Bolt 74B-A9B on two B70s with vLLM TP2 (AutoRound W4A16) and the full quantization guide ([5194d73](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/5194d73f1bd7a82675d995535e5883f73577f24d))

## [1.6.0](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/compare/v1.5.1...v1.6.0) (2026-10-15)


### Features

* **recipes:** Nemotron 3.5 Bolt 74B-A9B on one B70 (llama.cpp SYCL) ([57cd376](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/57cd3767ba5e409ef3dd004d5a6f6f7d166cb898))
* **recipes:** Nemotron 3.5 Bolt 74B-A9B on one B70 with patched llama.cpp SYCL ([415809b](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/415809b9d8a389941ea8556c23456f1948813c78))
* **recipes:** Nemotron Bolt context up to 262,144 tokens (CTX setting, capability record), launcher bound to localhost ([a67ed3e](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/a67ed3ee32365aa7931fbf4ce9f0b5ab98efbca1))
* **recipes:** pin the Nemotron Bolt image digest and report the image check ([622aca8](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/622aca8267fbb3f170bad32f0f7491317b883190))

## [1.5.1](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/compare/v1.5.0...v1.5.1) (2026-10-15)


### Bug Fixes

* **serving:** prefix cache on for Nemotron DFlash after its second host check; one agent per card on EXL3 ([81d4c78](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/81d4c7842ec45a12671f5e51d094dfa85917f484))

## [1.5.0](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/compare/v1.4.1...v1.5.0) (2026-10-15)


### Features

* **catalog:** SC_5.00bpw_H6 mixed-bit records — paired A/B + capability, pinned to sc5-mixedk-ab-20261012 ([b4464f7](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/b4464f7ebda578027b898781e349643c5e94b38c))
* **catalog:** SC5 context-capacity capability record (2026-10-14) ([a3bced5](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/a3bced5e62d3d860cc9c1b52dad3438e48a4d487))
* **catalog:** SC5 int8-vs-fp16 prefill served-KL and cold-prompt cost capability records (2026-10-14) ([2412995](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/2412995ecbb89363b51ffde7264ddd056d42ec7f))
* **catalog:** SC5 KL screen, V6 check and exl3xpu-mixedk 20261013 pin records; EXL3-XPU + image matrix update ([c79bb75](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/c79bb751801b07e7b62c6f0d511895ae330421a8))

### Other changes

Titles release-please does not list: a type it has no section for (`serving`, `research`, `nemotron`) or no type at all. They still ship.

* nemotron: keep DFlash serving cache-off after the 2026-10-15 host check (answer tests failed); other recipes stay cache-on ([3d80fc8](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/3d80fc818f054923ad99e68d0d37cff26cbe6bf5))
* serving: prefix caching on by default in every serving recipe; cache-off kept only on benchmark commands (labelled cold-input benchmarks only) ([0b7669f](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/0b7669f89ac2acde90ccc873c7135b59d65bb238))
* research: correct FP8 KV size (32 KiB/token: 4 GiB at 131K) and 'byte floor' wording per 2026-10-14 audit ([fbdbd19](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/fbdbd19b76ec77d687c74da85e035d79bbda4b33))
* Add recipe request issue form ([5f3b0e2](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/5f3b0e249b98f9c2f94781cec0bb017c03eed397))
* research: import sanitized B70-DOCS lab knowledge base (research/b70-docs/) ([299831c](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/299831c7724eed84d86dfbe6de174038baa9bf2d))

## [1.4.1](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/compare/v1.4.0...v1.4.1) (2026-09-27)


### Bug Fixes

* **claims:** +33-42% paired-median cap scaling (was +38-42 on stale 17.8 baseline); ALU-bound -&gt; unproven upstream hypothesis; W4A16 closure narrowed to tested export paths; 262K = capacity vs 131K proven completion ([8217736](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/82177369801f15c6749c2b2865babbbafa99010c))
* **evidence:** EXL3 262K claim upgraded — real 261,920-token completion w/ needle retrieval (was capacity-only) ([732efc0](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/732efc0949c4b95418a002f4df7a8f818bfdeb9a))

## [1.4.0](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/compare/v1.3.0...v1.4.0) (2026-09-27)


### Features

* **assets:** commit rendered comparison SVGs ([d71532f](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/d71532f2c9aab19a2b3ae99ae456d03ae8d5f2a1))
* **catalog:** EXL3 route records + OpenVINO GenAI envelope + Cascadia capability; exl3/autoround-vs-exl3 campaign evidence; regenerate catalog ([deac5c5](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/deac5c561156bac8d74dd52169d933b4f3fc85c8))
* **catalog:** EXL3 route records for CAM-2026-AUTOROUND-VS-EXL3 ([55eefea](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/55eefeaeaa1e47b9e34b2fd20f55ffc1a8125c6c))
* **catalog:** OpenVINO GenAI envelope + Cascadia capability records (uncommitted prior work landed); fix evidence pins ([59aa043](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/59aa043cc2f617d66b1caa45a13ce6bbfac8c06f))
* **charts:** exl3xpu decode-ctx + vllm-w4a16 decode-ctx SVGs (rendered from chart-inputs, vision-QA pass); EXL3-XPU doc embeds; 150W cappair median corrected 17.8-&gt;18.4 ([76b2c92](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/76b2c92afeec2e45568f2aa5e5ac6a6aef2258d2))


### Bug Fixes

* untrack stray brag-output/ work dir committed accidentally (exceeds 1MiB file gate) ([4a128a1](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/commit/4a128a1d2f18e571321fe0c46f8bc7fada31ba48))

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
