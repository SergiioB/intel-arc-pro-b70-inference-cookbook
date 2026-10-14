---
title: "Findings 2026-10-13: divergence, quality, speed, launcher runtime"
type: research
date: 2026-10-13
status: provisional
---

# Findings 2026-10-13: divergence, quality, speed, launcher runtime

**PROVISIONAL - NOT FOR PUBLIC HEADLINE.** One place for every number measured or
concluded on 2026-10-12/13. Each row names its ledger entry and evidence root;
the ledger stays the authority. Model: Qwen3.8-27B on Arc Pro B70.

## 1. Divergence (final-logit KL vs BF16, native heads)

Eval A = WikiText prose; eval B = held-out chat. Lower is better. Ratio is
against turboderp plain EXL3 4.00bpw on the same harness.

| Artifact | avg bpw | KL A (wiki) | KL B (chat) | ratio A | ratio B | Top-1 A | Top-1 B | Ledger |
|---|---|---|---|---|---|---|---|---|
| turboderp plain 4.00bpw (route) | 4.0 | 0.01391 | 0.00876 | 1.000 | 1.000 | 94.65 % | 96.41 % | RL-179 |
| turboderp SC_4.00bpw_H5 | 4.0 | 0.01602 | 0.00653 | 1.152 | 0.746 | 94.29 % | 96.80 % | RL-179 |
| turboderp SC_5.00bpw_H6 | 5.0 | 0.00582 | 0.00141 | 0.418 | 0.161 | 96.57 % | 98.11 % | RL-179 |
| Our XTQ Q2 (K4 mul1, full 64 layers) | 4.0 | 0.012353 | 0.009504 | 0.888 | 1.085 | 95.07 % | 95.43 % | RL-197 |
| Our XTQ, 13-layer prefix only | 4.0 | 0.0014177 | 0.0007809 | 1.159 | 1.412 | 96.45 % | 99.22 % | RL-191 |

Per-tensor held-out Hessian-weighted error vs turboderp's own tensors (RL-195,
partial, L00-L02 only): Q2 recipe 1.034-1.040x; turboderp's method
reimplemented 0.991-0.998x; best variant (F_svrefine) 0.968-0.991x. No variant
reached the predeclared 10 % win rule.

Conclusions:
1. SC5 is the only artifact clearly closer to BF16 than plain 4.0 on both slices.
   It is 25 % larger.
2. SC4 trades: better on chat, worse on prose. It fails the frozen-set gate.
3. Our full quant is better on prose and worse on chat. The gate needs both
   ratios at or below 1.0, so it **fails** (RL-197). No adoption, no claim.
4. Per-tensor error did not predict output KL: the 13-layer prefix had lower
   in-sample error than turboderp on every tensor yet 1.16-1.41x the KL (RL-191).

## 2. Quality (task checks)

| Artifact | Canaries (exact answer) | Long prompts (~8.2K) | Multi-turn cache on | Kernel checks | Ledger |
|---|---|---|---|---|---|
| plain 4.00bpw | 6/6 (h03 step-0 margin wrong: -0.06) | 3/3 | pass | - | RL-179, RL-186 |
| SC_4.00bpw_H5 | 6/6 (h03 margin -0.24) | 3/3 | pass | 46/46 | RL-193 |
| SC_5.00bpw_H6 | 6/6, all step-0 margins right | 3/3 | pass, 4 turns to 15.6K tokens | 46/46 | RL-182, cookbook launcher smoke |
| XTQ Q2 export | not served (gate failed first) | - | - | writer torch.equal on 3 tensors | RL-197 |

## 3. Speed (one B70, MTP3, C1, 128 output tokens)

Metric: client post-first token rate = usage completion_tokens / (t_last - t_first).
Not an engine decode figure, not an SSE chunk rate.

| Comparison | Card / cap | Sampling | p512 | p8192 | Ratio | Ledger |
|---|---|---|---|---|---|---|
| plain 4.0 vs SC5 (recipe of record) | GPU1 / 230 W | T0.7 top_p0.8 top_k20 pres1.5 | 63.56 vs 56.62 | 64.34 vs 56.82 | 0.89 / 0.88 | RL-186 |
| plain 4.0 vs SC5 greedy, single cell | GPU1 / 230 W | greedy | 85.4 vs 58.1 | - | 0.68 | RL-186 |
| plain 4.0 vs SC4 | GPU0 / 180 W | recipe | 56.53 vs 56.80 | 60.11 vs 55.01 | 1.005 / 0.915 | RL-193 |
| SC5 dispatch fix (odd-K to GEMV) | GPU1 / 230 W | greedy, n=3 | 57.0 -> 70.3 | - | +23 % | RL-185 |

Energy per completion token (window includes prefill): p512 plain 4.21 J vs SC5
4.75 J; p8192 9.13 J vs 9.88 J (RL-186).

## 4. External claim check (sybil-solutions, "over 100 tok/s on 2x B70")

Full audit: `research/2026/sybil-100tps-claim-audit-20261013.md`.

1. No published number matches our metric. 140.8 tok/s is a C4 aggregate wall
   rate including prefill on a TP2 AutoRound INT4 stack with graphs off (~35
   tok/s per stream). 114-116 tok/s is C1 decode on a different model
   (Qwen3.6-35B-A3B MoE). The GLM post has no stack.
2. Same artifact on one card: their 91.2 tok/s C1 is thinking-on prose at
   T0.7, an aggregate window rate; ours is 63.25-64.34 at p512/g128.
3. Correction to the audit: the two env levers it ranked first
   (`EXL3_DRAFT_VOCAB`, `EXL3_KV_BLOCK_EXACT=1`) are already active in our
   launches. Logs show "MTP draft head uses 512 of 1940 vocab blocks" and
   "keeps the hybrid negotiation (EXL3_KV_BLOCK_EXACT=1)".
4. Open: a protocol A/B (greedy p512 vs thinking prose T0.7 on one server) and
   a single-cell MTP5 probe. Nothing measured yet.

## 5. Launcher 0.6.0 runtime findings (GPU 1, 2026-10-13)

Evidence: `results/2026/10/launcher-06-verification-20261013/recipes/SUMMARY.md`.

| Recipe | Ready (s) | Answer check | 17*23=391 | Stop |
|---|---|---|---|---|
| qwen36-35b llama.cpp (native) | 31.9 | pass | pass | clean |
| qwen38-27b llama.cpp (native) | 42.8 | pass | pass | clean |
| qwen38-27b exl3 plain 4.0 | 328.1 | pass | pass | clean |
| qwen38-27b-sc5 exl3 | 405.7 | pass | pass | clean |

Not run: OVMS and nemotron vLLM (images not local; launcher now refuses instead
of pulling), qwen36-35b vLLM (artifact absent), Flash-Next and FP8 TP2 (need
GPU 0, which another study holds).

Bugs found by running, all fixed in 0.6.0:
1. Every container mapped all of `/dev/dri`: an engine on GPU 1 also opened
   GPU 0. Containers now get only the selected card and its by-path links.
   Mounting the full host by-path broke oneCCL init ("fill_device_fds ... open
   failed"), so the launcher writes a by-path directory per card set.
2. `docker stop` on a `--rm` container made the next `docker rm` fail ("removal
   already in progress"), leaving the card reserved.
3. EXL3 served name was the config id, so the name given to clients returned 404;
   no reasoning parser, so thinking text and `</think>` landed in the answer.
4. A transient child process (`gh`) that inherited a desktop app's render fd made
   GPU 1 look busy. Clients are now matched by DRM client id.
5. Missing engine images would have been pulled with ~3 GiB free on `/`.
6. The EXL3 recipe's 64K context was raised to 262K by `config.json`.

## 5a. External claim gap measured on our host (RL-200, screen; see section 4)

Plain EXL3 4.00bpw through the launcher, GPU 1 at 230 W, C1, same metric for both
arms (tokens after the first, reasoning and answer chunks both counted).

| Session | Greedy p512/g128 (thinking off) | Prose, thinking on, T0.7 (267-512 tokens) |
|---|---|---|
| MTP3 (recipe) | 78.66 tok/s (76.83-82.86) | 54.85 tok/s (50.64-63.90) |
| MTP5 (`--set num_speculative_tokens=5`) | 94.71 tok/s (93.14-99.30), +20 % | 52.84 tok/s (43.68-63.93), -4 % |

1. Thinking prose at T0.7 is slower than greedy here, so the published 91.2
   prose figure is not explained by protocol.
2. MTP5 speeds up only the high-acceptance greedy prompts. Session acceptance was
   31.8 % of draft tokens. Not adopted: it needs an interleaved same-content
   confirmation and a greedy output identity check.
3. Unexplained remainder: their host stack (kernel 7.1.8, compute-runtime
   26.31, IGC 2.40.13) vs ours (kernel 7.0.0-28, container runtime 26.14), an
   unstated power cap, and their windowed aggregate metric.

## 6. Host fault: copy-engine (BCS) page fault during weight upload

During SC5 loads on GPU 1 (0000:0f:00.0) the engine hung at checkpoint shard
2/3 while EngineCore spun at ~95 % CPU in `urEventWait` under
`at::native::xpu::_copy_xpu`. dmesg at the same minute: "Engine memory CAT
error [18]: class=bcs", engine reset, "Timedout job ... in python3".

- Rate today: 2 hangs in 4 SC5 loads on the final mappings (12:41Z, 13:16Z);
  the other loads finished shards in 50-78 s.
- Not new and not caused by the device mapping: the same BCS CAT fault appears
  on GPU 1 on 2026-08-27, 2026-08-28, 2026-10-12 (RL-178 incident) and
  2026-10-13 00:42/01:07, all before today's launcher change.
- Screen (RL-198): with `UR_L0_USE_COPY_ENGINE=0` and
  `SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0`, 4 of 4 SC5 loads were ready in
  360-386 s with zero new BCS dmesg events. Decode was unchanged: greedy
  p512/g128 median 70.24 tok/s post-first (68.75 / 70.24 / 70.54), wall 59.24,
  vs 70.3 / 59.2 in RL-185 on the same image. Four loads cannot prove the fault
  is gone; it is now the launcher default for EXL3 containers.
- Every EXL3 load also spends ~180 s compiling kernels, because the JIT cache
  lives inside a `--rm` container. A persistent cache mount is the next load-time
  lever (not measured).
- Evidence: `results/2026/10/launcher-06-verification-20261013/bcs-hang-20261013/`.
