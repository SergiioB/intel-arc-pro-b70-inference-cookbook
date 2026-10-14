<!-- Retained 2026-10-09 from a delegated agent run (model: GLM-5.3). Repo inventory of tried vs untried levers (agent survey, read-only). Not an authority; see research/2026/b70-untried-levers-synthesis-20261009.md. Known corrections: Staleness flags in section 3 were folded into docs/governance/standing-rules.md. -->

# B70-DOCS — tried-vs-untried matrix (2026-10-09)

Sources: `registry/research-ledger.json` (RL-001..RL-126), `registry/model-status.json`, `registry/research-program.json`, `engines/README.md`, `models/**`, `research/2026/b70-engine-capability-audit-20260929.md`, `research/2026/b70-breakthrough-architect-20260928.md` §A (L56–149), `research/2026/b70-format-route-lessons-20260930.md`, `decisions/adr-*`. Every row cites an RL id or path. "Measured" = on this dual-B70 host unless marked external.

## 1. Engine × quant × model × lever → outcome

### vLLM-XPU (engine dossier engines/vllm-xpu/README.md)

| Quant / model | Lever | Outcome | Evidence |
|---|---|---|---|
| Block-FP8 W8A8/W8A16, Qwen3.8-27B-FP8, TP2 | flag tuning, W8A16 reroute (`fp8_gemm_w8a16`), small-M Xe2 kernel M≤40, k-ladder to k=8, hybrid W8A8-prefill/W8A16-decode | adopted for the ≤1536-token FP8 route of record (51.54/56.85 tok/s p512/p1024; p2048 collapse fixed by hybrid, 28.5–32.2) | RL-002 (blocked) → RL-003 (validated) → RL-004 (blocked, kernel build owed); `research/2026/qwen38-fp8-tp2-speed-attack-consolidated-20260830.md`; registry/model-status.json:20-32 (best 48.08 p1024) |
| FP8 TP2, DFlash2 draft | DFlash2 official draft vs native MTP | rejected/blocked — K-norm bug (context path), selector-NaN eager artifact, drafter weaker than MTP4 on GPTQ; compile fix validated upstream (#56787) | RL-005, RL-006 (failed), RL-027 (failed, confounded), RL-028 (validated), RL-029 (validated) |
| GPTQ INT4 G128 (removed champion), Qwen3.8-27B, TP1 | MTP4 + draft-side INT4 pair + fp8 KV + XPU graphs | adopted as champion 2026-08→09 (+39.6/+48.6 %, iso-power), published LocalMaxxing; **checkpoint REMOVED 2026-09-27 (30-defect)** | RL-035, RL-036, RL-037 (validated); removal RL-087; registry/model-status.json:9,17 |
| same stack | depth k=6 vs k=4; k=3 clamp | k=6 flat (RL-026 validated); clamp measured −5.45 % (RL-042/RL-044 failed) | RL-026, RL-042, RL-044 |
| same stack | probabilistic draft sampling; bf16 draft-KV; presence-penalty audit; FastMTP head retrain; shortlist draft heads (foreign + own-corpus); verify-head INT4; host-prep cache; fused QK-RoPE #53989/#56096; max-num-batched-tokens; QK-norm | all rejected/closed — see §A.3 closed list | RL-046/RL-048, RL-045, RL-047, RL-056/RL-058, RL-074/RL-077, RL-041, RL-061, RL-054/RL-063, RL-064 |
| same stack | TP2 stock (oneCCL), TP2 + allgather overlay | rejected −45…−71 %; collectives exact but host-staged ~102 µs/4 KB | RL-075, RL-071, RL-008/RL-009 (ADR-0008) |
| same stack | PP2 host-RAM collectives | blocked | RL-016 |
| AutoRound W4A16 compressed-tensors + auto_gptq local re-cal, Qwen3.8-27B | serving route | closed — 30-defect in weights (RL-092), +6 %/−23 % vs EXL3 at matched caps | RL-087, RL-117, RL-116, RL-092; format-route-lessons §4 |
| bf16, Qwen3.8-27B, TP2 | quality reference | validated — base answers 14, defect is quant-induced | RL-084 |
| GPTQ INT4 G64 + DFlash n7, Nemotron-3.5-Lightning-30B-A3B | MoE serving; Mamba2 fusion | serving adopted (186.61 tok/s p2048/g128, registry/model-status.json:49-60); fusion withdrawn — fusable family 8 % of device time | RL-032 (failed → withdrawn); ADR-0002 |
| GPTQ INT4 + MTP1/MTP4, Ornith-1.5-35B | adoption | quality-rejected/closed 2026-08-24 | RL-017, RL-018 (validated 94.3), RL-019 blocked; ADR-0003/0007 |
| GPTQ-INT4 G128 + BF16 MTP, Qwen3.6-35B-A3B | 131K ctx MTP4 mixed load | capacity validated; causal_conv1d mixed-load open | RL-015; registry/model-status.json:34-46 |
| AWQ (cyankiwi Qwen3.8-27B-AWQ-INT4) | iso-byte alternative | blocked/open — engine loads it as compressed-tensors, AWQ loader not exercised, 1.079× bytes | RL-034 |
| EXL3 4bpw trellis + native MTP3 + fp8 KV (exl3xpu plugin), Qwen3.8-27B | primary route, graphs ON | **ADOPTED (ADR-0010)** — 63.25 tok/s p512 n=5 @230 W, 262K native, 6/6 battery, cache 5.4× | RL-082, RL-085, RL-088, RL-090; registry/model-status.json:5-19 |

### exl3xpu kernel program (Qwen3.8-27B EXL3 route)

| Lever | Outcome | Evidence |
|---|---|---|
| K=3 (3bpw) kernel extension (patches, P3 screen, P4) | authored+validated but serving FAIL at P3 → not adopted; quality-first 3bpw trial passed battery | RL-095, RL-097 (failed), RL-099, RL-100, RL-094 |
| MB=4 DpasKernel dispatch; NT strip geometry; ESIMD_REGISTER bank binding C2-V2 | all negative (+3–6 % regressions / miss gates) | RL-101, RL-102, RL-111, RL-112 (failed) |
| M4 flag ablation (B_FAKE, PF256, FOLD) | diagnostic only; no serving promotion | RL-119 |
| pruned draft head (EXL3_DRAFT_VOCAB) | static discovery only — premise superseded (512 of 1940 blocks present) | RL-106, RL-118 |
| margin-aware requant rounds 1–3 (scale-only, calibration pools) | canary repairable, heldout transfer fails; budget-blocked r3 | RL-113, RL-114 (failed), RL-115 (blocked) |
| TP2 | impossible at load (plugin guard) — custom TP2 conditional only | RL-103, RL-104, RL-105; registry/model-status.json:15 |
| kernel-probe program (VERIM4, thread sweep, residual split) | discarded by operator 2026-10-07 | RL-122 |

### llama.cpp SYCL (engines/llamacpp-sycl/README.md)

| Quant / model | Lever | Outcome | Evidence |
|---|---|---|---|
| GGUF Q4_K_M/mixed/Q5_K_M/Q8_0, Qwen3.8-27B | quant-format showdown vs GPTQ, eager-vs-compile, 150/230 W | validated — Q4_K_M quality reference (fails arithmetic 59-vs-60) | RL-024, RL-079 |
| hybrid-GDN qwen3_5 PPL | F16 reference + chunk geometry; upstream bug #28879 | validated | RL-025 |
| K216 (DeepSeek-V4 MoE) IQ3_XXS/IQ3_S | dual-SYCL decode (device-ID repair +3.54 %), ubatch grouped prefill kernel, fused IQ3 MMVQ, GPU-resident DSpark draft, CPU-MoE | repair kept; ≥10 % gate FAILED; ubatch prefill validated; DSpark failed; from-scratch grouped-IQ3 kernels 1.39–1.41× slower (closed) | RL-010, RL-011, RL-012, RL-030, RL-031, RL-040, RL-050 |
| K216 / #29245 upstream contribution | IQ3_XXS fused-gemm extension spec review | validated read-only; contribution scoped | RL-038 |
| q8_0 K + q4_1 V KV (fleet default) + quantized-KV TILE #26689 | TILE decode at 32K depth | validated_screen +13.0 % (merged cell) | ADR-0001; RL-081 |
| sparse FA #28796, Flash-Next dual-split placement | occupied long-context decode; AP-Q4_K_XL placement + calibration | sparse arm validated_screen; placement measured 21.6 tok/s = "capacity route" → superseded by Strata | RL-080, RL-052, RL-055, RL-057, RL-068, RL-070, RL-121 |
| Q4_K_XL + DFlash n2, Muse-Glimmer-30B | multimodal serving | measured 26.8 tok/s @128K (registry/model-status.json:76-88) | ADR per models/muse/muse-glimmer-30b/README.md |
| MTP/ngram stacking | measured −2–3 % (B70-specific, confirmed) | AGENTS.md "repo data" row 6; docs/speculative-decoding-on-b70.md |

### Strata (new engine, adopted 2026-10-09)

| Quant / model | Lever | Outcome | Evidence |
|---|---|---|---|
| IQ3_XXS GSQ-RCO + int8 KV + MTP --spec 4, Qwen3.8-Flash-Next, ONE card | `--stream-experts`, expert cache, prompt-cache, R4 hit path | **ADOPTED (ADR-0011)** 48.03 decode / 989.11 prefill (2.22×/5.1–5.3× vs RL-070 anchors), battery 3/3; quality gates still open | RL-125; registry/model-status.json:90-102; RL-121 (external assessment) |
| two-card expert-cache arm | rejected — slower (33.44 decode, host-staged) | RL-125 |

### OpenVINO GenAI / OVMS (engines/openvino-genai/README.md)

| Quant / model | Lever | Outcome | Evidence |
|---|---|---|---|
| int4-ov + grafted MTP5, Qwen3.8-27B | serving, long context | measured 28.5 tok/s @150 W clean; ≥196K = per-generation memory-growth OOM (real, classified); capacity ceiling 128K-class wording corrected | RL-086, RL-033; `research/2026/ov-cascadia-qwen38-bakeoff-20260911.md`; registry/model-status.json:17 |

### Cascadia (engines/cascadia/README.md)

| Lever | Outcome | Evidence |
|---|---|---|
| PP2 shard pipeline Qwen3.8-27B; llama.cpp RPC split | PP2 ~100 % retention / −0.26 %; RPC split −46 % (never-RPC-split rule); two independent servers 47.8 aggregate | RL-051, RL-083 |

### Host/topology (engine-agnostic)

| Lever | Outcome | Evidence |
|---|---|---|
| copy-only P2P capability (Level Zero probe) | validated — 11.1–11.9 GB/s bidirectional, no atomics | RL-059 |
| xccl collectives correctness | validated exact, latency-priced | RL-071 |
| megakernel/persistent fusion policy | closed class (ADR-0009), contingent on B6 | RL-007 |
| power caps 150/230 W, thermal wall, idle physics | measured; ≤55 °C protocol substitution declared | RL-065, RL-069, RL-086, RL-013 (failed engine sweep) |
| xe over-allocation incident | over-allocation evicts+wedges, GT resets | RL-123 |
| storage pricing (sybil tier) | failed gate — RQ-15 closed | RL-124 |
| stale-number sweep | validated fix pass | RL-039 |

## 2. Mentioned as ideas but NEVER measured locally

| Idea | Where mentioned | Status |
|---|---|---|
| W8A8/Quark INT8 MoE (as a *route*; 4× B70 TP4 93.55) | `research/external/neural-download-20260831.md:47` (external, Steve Seguin lab); "watch for 2× INT8/W8A8 on X570-F" in models/STATUS.md backups | external only; FP8 W8A8 dense *is* locally measured (RL-002/003) |
| MXFP4 (llama.cpp PR #29809 MoE decode; vLLM gpt-oss graph corruption #48946; vllm-xpu platform lists mxfp4) | RL-126; `research/2026/b70-quant-lever-hunt-20261009-summary.md` §4 P2 ("author-measured 55.84→106.64 on exactly one B70" — external) | never measured here; #29809 author numbers external |
| NVFP4 (QUASAR-QAT artifact; convert #28636 merged; no vLLM-XPU support #553) | RL-126; lever-hunt §4 P5 | never measured here |
| llama.cpp Vulkan backend | `archive/B70-MULTIGPU-REDDIT-RESEARCH-2026-07-21.md:56-60` ("Mesa 26.1+ Vulkan NV_coopmat2 can match or beat SYCL… not a safe production switch"); `archive/B70-MEGAKERNEL-ARCHITECTURE-2026-07-14.md:349` | never measured here |
| AWQ via real AWQ loader (iso-byte artifact) | RL-034 (branch open); lever-hunt §4 "Rejected with precedent: RL-034" | loader path never exercised |
| OpenVINO int4 on *other* models / OVMS production | engines/openvino-genai/README.md | measured only on Qwen3.8-27B |
| Prefill/decode disaggregation across the 2 cards | `research/b70-inference-strategy.md:21` ("treat the B70 as the compute-heavy Prefill Engine… stream KV to RTX decode engines") | idea only, never measured |
| Draft-on-other-card (B10 cross-card meta-speculation) | `research/2026/b70-breakthrough-architect-20260928.md` §B10 (L339–366) | priced only; kill-probe (zero-GPU stats) never run |
| Flag-slot seqno collectives (B11) | architect §B11 (L367–394); static graph-safety pre-check in capability-audit §C (RL-091) | microbench never run |
| Expert parallel B14 | architect §B14 (L422–448); capability-audit §A (RL-091: "no usable XPU EP dispatch… it IS TP2's transport") | static audit only, never measured |
| GDN constant-state exploits B15 (mamba prefix-cache "all", connector offload) | architect §B15; capability-audit §B (RL-091) | static audit only |
| EAGLE-3 | AGENTS.md spec-method list (`draft-eagle3`); .agents skill docs (vllm-xpu-run multi-gpu-and-tuning.md:97-106) | never measured on this host (MTP/DFlash/ngram only) |
| n-gram as vLLM speculative method | AGENTS.md list (`ngram-simple`…`ngram-cache`); vllm-xpu-run skill | n-gram measured only in llama.cpp (stacking −2–3 %, AGENTS "repo data" row 6); vLLM ngram never run |
| Suffix decoding (speculative class) | not present as a lever anywhere except CPU "suffix replay" margin pilot RL-110 (different mechanism) | never proposed/measured here |
| KV offload / tiering (mooncake/nixl mamba offload, GPU1 tiering) | capability-audit §B2 (RL-091: "a P2P-copy connector to GPU1 is writable but does not exist") | static only |
| Invented formats Q3DT / M5X / DSM | `research/2026/b70-quant-lever-hunt-20261009-summary.md` §4; `research/2026/b70-invented-quant-abis-20261009.md` | CPU-simulated, killed/deferred before GPU |
| INT4-G128 dual-path DPAS kernel (P3 falsifier), UD-Q4_K_M dense screen (P4), IQ2 experts (P1) | lever-hunt §4; `research/2026/b70-overnight-quant-campaign-plan-20261009.md` | planned for tonight, not yet run |
| B1/B2 acceptance-distribution mining, fork drafting, B16 KV read-cap | architect §B1/B2/B16 | RL-093 did the zero-GPU forensics (bimodality non-universal); predictors/forks never built |

## 3. Standing invariants (architect §A, L56–149) and staleness

### A.1 ALWAYS-ON (L58–100)
1. "XPU graph capture always on for vLLM XPU serving… eager torch-profiler traces are blind to graph-replayed kernels (RL-072 lesson)" (L60–64) — **current** (EXL3 route runs graphs ON, RL-090).
2. "Locked spec stack: stock MTP weights, MTP4, greedy drafting, draft-side INT4 pair, fp8 KV, model-card recommended sampling (presence_penalty 1.5 non-thinking)" (L65–68) — **STALE**: describes the *removed GPTQ champion* (artifact removed RL-087). Current incumbent is EXL3 4bpw **MTP3**, fp8 KV (only dtype), no draft-INT4 pair (registry/model-status.json:11).
3. "Exact-answer battery + spec-canary before any speed claim" (L69–71) — **current** (battery re-pinned RL-079; ADR-0011 adopted on 3/3 exact with open quality gates).
4. "Prefix-cache policy: OFF on W4A16-class stacks until vLLM #53912 is fixed" (L72–75) — **half-stale**: W4A16-class is closed/removed entirely (RL-087); EXL3 route runs cache ON with measured 5.4× (RL-088).
5. Output preservation (L76–78), 6. metric naming (L79–82), 7. α bookkeeping (L83–86), 8. single-client C1 gate (L87–88), 10. quantization axes independence (L92–94), 11. attribution honesty (L95–97) — **current** (reinforced by RL-090/RL-093 metric corrections).
9. "Placement: independent dual-card stacks (ADR-0008). TP2 only behind the gated matched-control campaign; never a stock Docker TP2 launch" (L89–91) — **current**; RL-103 further hard-blocks EXL3 TP2 at load.
12. "Thermal/power: … ≤55 °C protocol or a formally declared substitution, 150 W restored on exit" (L98–100) — **partially stale**: the ≤55 °C gate was proven unreachable (RL-065/RL-069 awake-idle wall) and substitution is now the standing norm; operating point since 2026-09-30 is 200 W/2400 MHz both cards (format-route-lessons §5.4), not "150 W restored".

### A.2 ALWAYS-TRY FIRST MOVES (L102–127)
1. Graph capture first (L104–105) — current. 2. Draft-side quant pair "+39.6/+48.6 %" (L106–107) — **stale as standing advice**: measured on the removed GPTQ stack; the EXL3 drafter has no separately-quantized INT4 pair. 3. Arithmetic before building (L108–110, k=3 clamp −5.45 %) — current. 4. Canary first (L111) — current. 5. Draft k+1 re-read bytes (L112–113) — current (RL-106 shortlist lever was this axis). 6. Acceptance economics +6.25 %/+0.05α, bimodal profile (L114–116) — **partially stale**: RL-093 showed the bimodal 0.98/0.98/0.12/0.09 stock profile is *not universal* (27/76 cliff, 38 gradual; EXL3-MTP3 route is flat decay). 7. Attribution vs named floor (L117–119) — current. 8. KV-read growth term "32 KiB/token × N on this geometry" (L120–122) — dense-27B-specific; measured 38.2 KB/tok fp8 on Qwen3.8-27B (RL-085) — roughly current for that model. 9. Metric audit (L123–124) — current (RL-090 chunk-rate correction). 10. Placement arithmetic before "one card" (L125–127) — current; two-card arm just measured slower on Strata (RL-125).

### A.3 CLOSED — DO NOT RETRY (L129–146)
Quote (L131–146): "Single-card champion decode kernel/host hunting (byte floor, three levels, subject to correction-2 caveat); whole W4A16/GPTQ-class symmetric int4 on Qwen3.8-27B (30-defect ×3 exports; kernel-vs-weights discriminator still owed); acceptance frontier — MTP-head small-corpus retrain (D11 −2.5 %), probabilistic drafting (D12 finalist-refuted), bf16 draft-KV (D14 α −0.042), foreign-corpus shortlist (RL-074 finalist −1.5/−5.5 %), verify-head INT4 (top-1 72 %), depth >4 (flat ±1.4 %); TP2 on this topology as served by stock oneCCL (RL-075 −45…−71 %; peer-IPC/graph collectives segfault); grouped-IQ3 from-scratch kernel (1.39–1.41× slower); #29107 (+0.08 %); vllm-xpu-kernels 0.1.15 (regression −4.4/−14.5 %); #56096/#53989 fused QK-RoPE (wrong model class); QuaRot→INT3 (no INT3 XPU kernel; ideal 1.215×); W4A8 prefill (closed-by-attribution, ≤1.27×); Cascadia PP2 (−0.26 %); DFlash2 on GPTQ; host-prep cache (+1.5 % vs +4 % gate); max-num-batched-tokens (±0.25 %); CPU-MoE DSpark; `-sm row`; stock `--tensor-parallel-size 2`. Re-entry only via a named changed variable."

Staleness flags:
- "kernel-vs-weights discriminator **still owed**" (L133–134) — **resolved after writing**: RL-092 ran it, verdict WEIGHTS (fp32 oracle reproduces the flip). 
- "Single-card champion decode kernel/host hunting … closed" (L131–132) — written against the GPTQ stack; the architect's own B6 notes "zero attribution exists for the adopted route… the verdict does NOT cover EXL3" (L272–291), and RL-098/101/102/111/112 have since measured EXL3 kernel levers (all negative so far, RL-122 discarded the program). Class-closure on the *champion* needs re-anchoring to the EXL3 route.
- "DFlash2 on GPTQ" (L143) — GPTQ is removed; class entry now moot/historical.
- "TP2 … stock oneCCL" — still current and extended by RL-103 (EXL3 TP2 impossible at load).
- Everything else in A.3 remains valid as of 2026-10-09.
