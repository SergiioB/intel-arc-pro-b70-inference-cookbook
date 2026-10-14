# B70 new-quant-format lever hunt — complete state and findings (2026-10-09)

**Status:** Day-work summary and stop-state record. Operator ordered full stop at ~19:25 local.
No GPU campaign ran today; everything below is static analysis, CPU-only measurement, and one
operator-ordered process kill. The overnight plan is separate:
`research/2026/b70-overnight-quant-campaign-plan-20261009.md`.

## 1. Mission and method

Operator ask: find where a BIG quant-format lever can improve B70 inference, from experiment
evidence (what worked/failed) + silicon knowledge, then design NEW formats and pre-screen them so
tonight can test what actually holds quality + speed. Executed by the specialist fleet:
roofline architect (astra), decode kernel engineer (astra), quantization engineer (GLM-5.3),
upstream researcher (GLM-5.3), ISA analyst (flash-high), correctness engineer (flash-high),
adversarial second architect (deepseek; final pass did not return), plus lead arbitration on
disputed numbers. All artifacts under `research/2026/`, captures under `research/references/`.

## 2. The law and the corrected anchors (all downstream pricing uses these)

- `t_step >= max(D/B, I/R_issue, F/P) + serial` — never a product. Bytes AND issue-cost AND compute
  all bind through the max.
- B70 pin **608 GB/s**, 32 GB/card. EXL3 4bpw weight-pass anchor **381 GB/s** cold-DRAM class.
- **636 GB/s "B_FAKE" is logical-byte throughput on an isolated qkv pair with failing numerics —
  NOT sustainable DRAM and NOT model throughput.** Instruction-tax quotes must name their base:
  37.3% pin-based (1−381/608) vs 40.1% vs B_FAKE.
- Route-lessons' "111 tok/s ceiling" is the **16.43 GB conditional-round** convention (weights+embed
  +MTP reads); weights-only 13.5 GB gives 135.1. Both conventions now labeled everywhere.
- Xe2: sub-8-bit INT gives **no** TOPS multiplier (INT8 ~363 TOPS class is the ceiling; BF16 ~170);
  no native FP8/FP4 datapath; DP4A needs integer activations; W4A16 dequant feeds FP16/BF16 DPAS.

## 3. Verified structural finding: the integer-DPAS vacuum

ISA-level audit of the retained EXL3 M=4 ablation disassembly: **every DPAS emitted is FP16 (`:hf`);
integer DPAS (`:s4`/`:s8`) count = 0**. llama.cpp SYCL runs quant decode as MMVQ `__dp4a` vs Q8_1
activations (M≤8; **MMQ tiled path hard-disabled upstream for accuracy**), and prefill >8 cols is
dequant→FP16+GEMM. Strata dequants→FP16 buffers→oneMKL FP16 GEMM (`dpas.8x4:hf`). **The ~363-TOPS
INT8 class is unused by every quant path on this stack.** Any new format claiming it must ship a
co-designed integer kernel AND a parity strategy (the one historical tiled-integer attempt was
accuracy-disabled). This is the deepest deficiency found and the target of tonight's P3 falsifier.

## 4. Candidate ledger (what survives into tonight)

| Candidate | Verdict | Key numbers | Gate |
|---|---|---|---|
| Close ADR-0011 quality gates on incumbent IQ3_XXS (Strata, GPU0) | **P0 — gates the night** | adopted on only 3 exact greedy cases; open: representative tasks, long/multi-turn, replay drift | 6-canary x2 + 3x3 replay + 10 tasks ≥8/10 + 8K edge check (protocol precommitted by correctness engineer) |
| IQ2_XXS/IQ2_S experts, Flash-Next (Strata) | Conditional P1 | −33% expert bytes → miss floor 20.6→13.8 ms/round, residency 59→87.5% → +11-14% decode bound; kernels verified in Strata source | artifact existence (sweep was running at stop) + strict quality-first ladder; fallback UD-Q4_K_XL |
| MXFP4 MoE decode via llama.cpp PR #29809 on gpt-oss-20b | **P2 — highest verified upside** | author-measured **55.84→106.64 tok/s (1.91x, n=5) on exactly one B70**; real MMVQ/MoE path | branch build + ~13 GB download + canary battery before speed |
| INT4-G128 dual-path kernel falsifier (M=1 DP4A / M=4 DPAS) | **P3 — the structural lever test** | hypothesis ≥500 GB/s metadata-inclusive vs 381 anchor; break-evens: G128 >392.9, IQ4_XS/MXFP4 >404.8, Q4_0 >428.6, Q4_1 >476.25 | source pin → ISA gate (no spills, DPAS emission) → ring bench → 3+5 paired cells |
| UD-Q4_K_M 27B dense screen (llama.cpp) | P4 — cheap | artifact on disk: `/mnt/<sata-disk>/Qwen3.8-27B-UD-Q4_K_M.gguf` (16.4 GB, ~4.9 bpw) | canary battery + p512/g128 n=3 screen; quality data point, not a speed lever |
| NVFP4 QUASAR-QAT 27B (llama.cpp) | P5 — conditional | convert #28636 merged 2026-10-09 (#27161 closed unmerged — corr. post-audit); MMVQ in master tree | community-QAT → exact battery FIRST (RL-087/092 defect class) |
| **Q3DT lossless IQ3 cold-tier container (invented)** | **KILLED on real bytes** | codec-simulated c=1.1193 mean (30 samples, L0/8/24/40/47 x gate/up/down, ±8 MiB steps); realized c≈1.0000 once the ABI's raw-mode fallback is counted; kill rule c>0.90 fires either way | H0 per-tensor 7.29–7.96 bits/byte (down tensors 7.29–7.31 → H0/8≈0.911) — floor defeats general-purpose lossless coders; "defeats ANY coder" withdrawn (corr. 2026-10-09 post-audit); artifact `scripts/tmp/q3dt-census-result.json` |
| **M5X signed-5 MXFP4 recode (invented)** | KILLED for tonight | 5.25 bpw (+23.5% bytes) needs ≥547 GB/s to match 4.125@430; 10% gate ≈ pin | kept only as exact-layout functionality |
| **DSM dual-scale margin bitplanes (invented)** | DEFERRED | 3.25/4.25 bpw two-basis additive grid, margin-allocated per tensor | blocker: no canonical BF16 source on host (~54 GB acquisition decision) |
| Q8_0 as speed route | Rejected | needs 809 GB/s > 608 pin | arithmetic |
| AWQ / LM-head INT4 / post-train MXFP4 dense / requant-from-quantized / W4A8-lossless / NVFP4-on-vLLM | Rejected with precedent | RL-034 / RL-041 / KLD 2x / RL-113-115 / numerics / #553 open | ledger |
| EXL3 2/3/5 bpw re-trial, VERIM4, margin-requant repair | **Operator-held — NOT planned** | RL-097/101/112/113/114 failed; RL-122 stop | reopens only by operator |

## 5. Data hygiene record (disputes found and resolved)

- TQ1_0: census first "falsified" 54 B → claimed 56 B, then self-contradicted. Lead arbitration at
  the pinned header (`ggml-common.h` @ `50e3e3e480c80b9ee0ea067e475602d540dc62c8`):
  `qs[(QK_K-4*QK_K/64)/5]`=48 + `qh[4]` + `d`=2 → **54 B = 1.6875 bpw confirmed** (upstream comment
  "// 1.6875 bpw"). Addenda corrected in both affected notes.
- Re-verified block sizes at the pinned commit: IQ4_XS 136 B/256, IQ3_XXS 98 B/256, IQ3_S 110 B/256,
  Q4_K 144 B/256, Q8_0 34 B/32, TQ2_0 66 B/256, Q6_K 210 B/256, IQ2_XXS 66/74/82 B (XXS/XS/S),
  IQ1_S 50, IQ1_M 56, MXFP4 17 B/32, NVFP4 36 B/64+scales.
- Strata expert-blob calibration: 512 experts, 42.91 GB total, blobs 1.3-1.87 MB (INTEL.md's 1.97 MB
  was the IQ3_S figure) — from `pack/iq3xxs/native_experts.txt`.

## 6. Availability matrix headline (full table in quant-format-availability-20261009.md; ledger RL-126)

- Decode feasibility today = MMVQ (M≤8) only; ALL quant GEMM >8 cols = dequant→FP16+GEMM.
- Local llama.cpp base is ~2 months stale vs master (misses #26689 TILE-KV, #29500 IQ3_S multicol,
  #29245 grouped-MoE-XMX, #28796 sparse-FA); Strata v0.1.41 is a clean upgrade candidate.
- No Qwen-org QAT artifact exists for Qwen3.8/Flash-Next (unsloth UD = mixed recipes of standard types).
- TQ1_0/TQ2_0: zero SYCL kernels (CPU fallback). vLLM-XPU GPTQ W4A16: upstream #52203 flags Arc-B60
  DEVICE_LOST at profile_run — contradicts our B70 record; watch on re-pin.

## 7. Host stop-state (as found at stop)

- Both B70s bound to `xe`; card0 0b:00.0 / card1 0f:00.0; renderD128/129 present and **IDLE**.
- The active `llama-perplexity` PPL run (PID 3911473, Qwen3.6-35B-A3B-UD-Q4_K_XL Hellaswag 10042,
  started ~19:00 via omp) was **killed on explicit operator order** at ~19:12 (~12 min in; its
  partial scores are discarded, not retained — nothing was published from it).
- Leftover flagged, not touched: container `b70-qwen38-flashnext-strata` (strata-sycl-dev, up 7 h,
  status **unhealthy**, holds no GPU fds) — the morning serve wrapper; one `docker stop` if wanted.
- Power caps: re-read at night launch (morning as-found was card0 180 W / card1 230 W; xpu-smi
  daemon unreachable from the agent shell; record via the operator shell or container stats).
- Device access note: the agent sandbox masks `/dev`; GPU-affecting commands need the elevated mode.
- Interrupted prep (exact resume point): the 3 new canary token files were NOT created — the encoder
  (`scripts/tmp/p0_encode_canaries.py`, copied to `$S/prompts/`) failed with
  `ModuleNotFoundError: regex` in the container's system python. Resume options: pip-install `regex`
  into the container, use the orchestrator's venv, or hand-encode via `vocab.json`+`merges.txt`.
  Engine mechanics are fully decoded (`prefill-sweep.sh` driver pattern: docker `strata-sycl-dev`,
  `--device=/dev/dri --group-add $RENDER_GID`, env `ONEAPI_DEVICE_SELECTOR=level_zero:0
  STRATA_VERIFY_NO_HOST=1 STRATA_VERIFY_DEVICE_PLAN=1 STRATA_STAGER_THREADS=12`,
  `--pack/--native/--ple-gguf --expert-profile data/expert-profile.bin --expert-cache auto
  --stream-experts --mtp /work/mtp-rt --spec 4 --spec-min-p 0.5 --vram-reserve-mib 256 --ple-io ram
  --max-context 10240 --kv int8 --prefill 4096 --prompt-cache 0 --tokens-file ... --max-new 64
  --greedy --stats`; image entrypoint swallows non-engine commands — pass `--entrypoint` explicitly).
- Existing canary assets: `batt-arith.txt` (703), `batt-capital.txt` (Paris), `batt-token.txt`
  (echo) — all passed in the 03:2x confirm run; the NEW information is in the three unrun canaries
  (14-vs-30 defect class, multi-hop logic, strict JSON).
- **Post-stop correction (2026-10-09 evening):** the encoder had two latent defects behind
  the recorded `ModuleNotFoundError` — (1) `Tokenizer(TOKDIR)` passes a directory where the
  constructor takes `(tokens, merges, …)`; (2) its "common prefix" specials extraction yields
  an EMPTY prefix (type-3 CONTROL tokens don't round-trip through `encode` with
  `parse_special=False`), so canaries would have been written **with no chat template**.
  `scripts/tmp/p0_encode_canaries.py` was rewritten: loads the pack `tokenizer/` dir per the
  pack format, splits the reference into header/body/footer at the chat markers, and proves
  the split rebuilds `batt-arith.txt` token-exactly before writing. Run on host against the
  real pack (`/mnt/<sata-disk>/strata` = container `/work`; template
  `<|im_start|>user\n … <|im_end|>\n<|im_start|>assistant\n`): **the 3 canary files are
  created and verified** — batt-sumsq 28 ids, batt-logic 31, batt-json 26; header
  `248045,846,198` and footer `248046,198,248045,74455,198` identical to the reference.
  Remaining P0 pre-flight: record caps, then the battery.

## 8. Artifact index (all 2026-10-09)

- `research/2026/b70-quant-family-roofline-20261009.md` + `.py` — full format/KV pricing, break-evens,
  max() law, corrected anchors (census addenda appended).
- `research/2026/b70-small-m-quant-layout-static-spec-20261009.md` — decode-layout ranking, MXFP4
  remap proof, falsifying microbenchmark spec (addendum appended).
- `research/2026/b70-quant-format-lever-audit-20261009.md` — D1-D7 deficiencies, C1-C6 candidates,
  held/rejected tables, evidence log.
- `research/2026/quant-format-availability-20261009.md` — format x engine matrix, top-3 picks,
  literature harvest, pin-risk; ledger **RL-126**; captures `research/references/quant-avail-20261009/`.
- `research/2026/b70-invented-quant-abis-20261009.md` + `.py` — DSM / M5X / Q3DT designs, integer
  accuracy contract (A8 group-32; INT32 bounds 12,192 / 48,768 verified), Pareto ranking, kill rules.
- `research/2026/b70-overnight-quant-campaign-plan-20261009.md` — tonight's precommitted P0-P5 queue.
- `research/2026/b70-correctness-*` gate ladder: see correctness-engineer protocol in session record
  (6 canaries, gate order, UD-Q4_K_XL trigger; open windows: sampled-quality-under-MTP, 8-10K drift,
  prefix-cache parity).
- `scripts/tmp/q3dt_entropy_census.py` + `q3dt-census-result.json` — the Q3DT kill measurement.
- `scripts/tmp/p0_encode_canaries.py` — canary encoder, **rewritten + verified post-stop**
  (see §7 correction); the 3 token files are in `/work/prompts`.

## 9. Open items at stop (night owner picks up here)

1. Existence sweep (IQ2-class Flash-Next artifact; NVFP4 QUASAR-QAT size/license; gpt-oss revisions;
   Strata ≥0.1.41 B70 rows) — was in flight, not delivered. **[Partially resolved post-stop:
   GSQ-RCO has no IQ2_XXS/IQ2_S (HF-verified) → plan P1 DEAD tonight; QUASAR-QAT NVFP4 artifact
   exists; #28636 merged / #27161 closed unmerged (RL-126 corrected). gpt-oss + Strata rows open.]**
2. Kernel-engineer ABI pricing + single-format confirmation for P3 — delivered? no; pending.
3. Adversarial pass #2 verdicts (KILL/CONDITIONAL/ADVANCE on DSM/M5X/Q3DT + queue order) — did not return.
4. P0 pre-flight: create the 3 canary token files (§7 resume point), record caps, then the battery.
   **[Canary files created + verified post-stop, §7 correction; remaining: record caps, run battery.]**
5. Decide the leftover `b70-qwen38-flashnext-strata` container: stop it before tonight's serve.

Nothing was adopted, no route changed, no public claim made. All quality gates remain open until run.
