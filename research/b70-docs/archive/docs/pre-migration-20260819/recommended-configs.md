# B70 Recommended Configurations

Per-model optimal configs for the Intel Arc Pro B70 32GB.
All configs use **q8_0 K + q4_1 V** KV cache (KL-divergence-optimal, see `kv-cache-quantization.md`).

Last updated: 2026-08-18. Historical Nemotron Q4_0 evidence and current artifact
availability are distinguished below; do not infer a current local file from a
historical benchmark. Latest local llama.cpp source checked: 89e0aa6fd with
`GGML_SYCL_F16=ON`. vLLM DFlash n=5 (Run 36) is a separate engine/path from the
llama.cpp MTP rows. Qwen3.8-27B vLLM MTP4 is Run 40 (n=5); mixed-split v5 is
infra (Run 41), not a speed replacement. ThinkingCap GGUF paths in the table
below are historical — those files are not on disk as of 2026-08-18.

---

## Quant Selection Guide (NEW — read first)

**Nemotron-specific llama.cpp/SYCL finding — not a universal Xe2 rule.** The
locally requantized Nemotron 3.5 Lightning 30B-A3B Q4_0 artifact outperformed
Q4_K_M in its recorded campaigns (see
`research/nemotron-lightning-optimization-20260811/08-q4_0-breakthrough.md`):

| Quant/artifact | BPW | Recorded no-spec decode | Recorded MTP4 decode | Interpretation |
|----------------|----:|------------------------:|---------------------:|----------------|
| **Local Q4_0 from Q4_K_M** | 4.5 | **68.0 t/s** | **87.4 t/s median, n=5** | Simpler 32-weight blocks reduced this model's GEMV/dequant overhead |
| Native-source Q4_K_M | 5.44 | 38.2 t/s | ~50 t/s | Different baseline campaigns; quality-oriented artifact |

- The Q4_0 result is a model-weight requantization made with
  `--allow-requantize`, **not** a benchmark of the absent `-native.gguf` file.
  It may carry an additional quality loss versus direct BF16 → Q4_0.
- For throughput-critical Nemotron serving, validate Q4_0 on the intended task;
  for quality-critical work, prefer a native Q4_K_M/Q5_K_M or a validated native
  Q4_0. Community PPL deltas are not a B70 task-quality measurement.
- Re-benchmark every model/build/quant pair. Muse-Glimmer Q4_0 was not tested;
  its current production route remains Q4_K_XL + DFlash.
- **Model-weight Q4_0 is unrelated to KV-cache `q4_0 K`, which remains
  catastrophic and prohibited.**
- Before downloading or duplicating a conversion source, check live `df -h` and
  budget retained source + output + staging space plus reserve; historical disk
  snapshots in this repository are not capacity guarantees.

### vLLM checkpoint gate

GPTQ-INT4 is the documented B70 XPU INT4 path only after the **exact
architecture/checkpoint** loads. Determine source format from `config.json` and
tensor names, not a repository's `AWQ` label. A conversion does not establish
quality parity or add an absent architecture implementation. Muse-Glimmer remains
blocked on three independent gates: post-merge architecture support (PR #51655),
exact quant text-only dispatch, then vision/XPU validation. See
`research/b70-vllm-muse-int4/PLAN.md`.

---

## Per-Model Recommended Configs

| Model | Quant | KV | Context | Power | MTP/Spec | Decode t/s | VRAM free | Best for |
|-------|-------|-----|---------|-------|----------|-----------|-----------|----------|
| **Nemotron 3.5 Lightning 30B-A3B** | **vLLM GPTQ-INT4-G64-sym + DFlash-BF16** | auto | **120K capacity** (speed card still 16K n=5) | 150W | ✅ DFlash n=7 | **186.6 median, n=5** (p2048/g128 client post-first @16K) | 5.8 GB @16K / 5.3 GB @120K | Fastest isolated Nemotron vLLM path; 119904 tokens completed; not 128K |
| **Nemotron 3.5 Lightning 30B-A3B** | vLLM GPTQ-INT4-G64-sym (no-spec graphs) | auto | 16K | 150W | — | **93.00 / 87.25** p512/p8192 g128 n=5 | campaign-specific | Graph baseline; native MTP still 0% |
| **Nemotron 3.5 Lightning 30B-A3B** | **Local Q4_0 from Q4_K_M** (llama.cpp) | q8_0-q4_1 | 16K-128K | 230W | ✅ MTP-4 | **87.4 median, n=5** | campaign-specific | Fastest recorded llama.cpp Nemotron cell; artifact currently absent |
| **Nemotron 3.5 Lightning 30B-A3B** | Native-source Q4_K_M (llama.cpp) | q8_0-q4_1 | 16K-128K | 230W | ✅ MTP-4 | ~50 | campaign-specific | Quality-oriented baseline; separate campaign |
| **Muse-Glimmer-30B** | Q4_K_XL | q8_0-q4_1 | **128K** | 230W | ✅ DFlash n2 | **26.8** p512/g128 n=5 (29.2 was the 64K cell) | ~2.3–4 GB at 128K / 1 slot | Dense multimodal (vision+reasoning) |
| **Qwen 35B MoE** | Q4_K_XL | q8_0-q4_1 | **512K** | 150W | — | **72.6** | 1.6 GB | **Max context + throughput** |
| **Qwen 35B MoE** | Q5_K_M | q8_0-q4_1 | **256K** | 150W | — | ~68¹ | 2.2 GB | Best quality MoE |
| **ThinkingCap 27B** | Q4_K_M | q8_0-q4_1 | **128K** | 200W | ✅ MTP-4 | **29.75** | ~3 GB | Historical GGUF — **file absent 2026-08-18** |
| **ThinkingCap 27B** | Q5_K_M | q8_0-q4_1 | **128K** | 200W | ✅ MTP-4 | **26.3** | 3.0 GB | Historical GGUF — **file absent 2026-08-18** |
| **ThinkingCap 27B** | Q6_K | q8_0-q4_1 | **128K** | 200W | ✅ MTP-4 | **27.85** | 0.4 GB | Historical production GGUF — **file absent 2026-08-18** |
| **Qwen3.8-27B** | **vLLM GPTQ-INT4-sym-G128 + BF16 MTP** | fp8 | **128K** (U=0.88) | 230W | ✅ MTP-4 | **83.7 median, n=5** (p512/g128 client post-first) | ~1.0 GB after load | Current dense vLLM **default** (Run 40 LMX). Do not overwrite with the overlay. Mixed spec+prefill needs `patch_gdn_mixed_split_v5.py` (Run 41). |
| **Qwen3.8-27B** | **vLLM GPTQ + draft-INT4 S+M1** (optional) | fp8 | 128K isolated C1 | 230W | ✅ MTP-4 | **112.65** p512/g128 n=5 vs matched BF16-draft **81.20**; prefix-on agentic **58.86 vs 43.81** short / **61.02 vs 46.34** 32K / **56.20 vs 44.83** 64K | 816–871 MiB after load | Optional overlay (Run 42–44). Synthetic p130944/g128 **62.52** n=5 isolated C1 — not serving. Prefix-on ~128K t1 **37.48 vs 43.82** (n=1, S+M1 slower) — not a headline. |
| **Qwen3.8-27B** | **vLLM GPTQ + v5 + draft-INT4 S+M1** (concurrent serving) | fp8 | 131072 (research reserve) | 230W | ✅ MTP-4 | Realistic C5 8K/g512: **27.7/user, Σ 138.3** (LMX `cmt023kzf0egfms01rxx5jhv7`); C32 Σ-streams **903**; lmx short-prompt C5 **203.8** / C32 **224.2** | 631–682 MiB after load | Concurrent recipe (Run 45–46): v5 REQUIRED at C≥2 mixed; accept 46–56% at Cn; Cn ≠ C1. **Prompt-content sensitive**: Pi 100.2 C1 / INDEX 71 |
| **Qwen3.8-27B** | llama.cpp Q4_K_M | q8_0-q4_1 | 128K | 230W | — (converter stripped MTP) | **22.30 ±0.01** tg128 n=5 | campaign-specific | Local GGUF present (`/mnt/<models-disk>/gguf/Qwen3.8-27B-Q4_K_M.gguf`); not the serving champion |

**Qwen3.8-27B client sampling — selected by thinking level (mandatory, Qwen model card):**
`off` = `temperature 0.7, top_p 0.80, top_k 20, presence_penalty 1.5` + `chat_template_kwargs {"enable_thinking": false, "preserve_thinking": true}`;
`low|medium|xhigh` = `temperature 1.0, top_p 0.95, top_k 20, presence_penalty 0` + `chat_template_kwargs {"enable_thinking": true, "preserve_thinking": true}` + `reasoning_effort: "<level>"`.
Serve with `--generation-config auto` (checkpoint ships the thinking preset).
Measured cost vs greedy: C1 −4% (accept 91.8%), C5 Σ −15%. Greedy = diagnostics only.
Desktop launcher does the mapping: `~/Desktop/b70-launcher.sh [off|low|medium|xhigh]`.

¹ Dense MTP-4 decode = MEASURED engine rate @200W/128K ctx (Run 12, 2026-08-05), not estimated.
MTP VRAM ceiling: dense 27B + spec buffers fits ONLY at 128K — Q4@256K and Q5@200K with MTP
OVERFLOW (verified; see benchmark-history Run 12). MoE Q5 row = b10222-era, not re-run on 0804.
Base decode (no MTP): Q4=21.3, Q5=16.2, Q6=16.0 t/s. Prefill ~700-936 t/s dense.

**Nemotron 3.5 Lightning details (2026-08-13):** hybrid Mamba2-Transformer MoE
(46 SSM + 7 attention layers, 128 experts/7 active).

**vLLM XPU (canonical now):** local GPTQ INT4 G64 + local DFlash BF16,
`method=dflash` n_spec=7, graphs + native grouped-topk v2, cache off, 150 W.
Isolated n=5 representative decode **186.61 t/s** at p2048/g128; p8192/g1 cold
input **7160** (not engine prefill). Serving limit after Run 38:
`--max-model-len 120000` (completed 119,904 tokens). **Not 128K.** HF:
`SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym` +
`SergiioB/Nemotron-3.5-Lightning-30B-A3B-DFlash-BF16`. Cookbook:
`intel-arc-pro-b70-inference-cookbook/docs/nemotron35-30a3/NEMOTRON-DFLASH-B70.md`.
Open PRs (2026-08-13): [vllm#52159](https://github.com/vllm-project/vllm/pull/52159)
(XPU grouped_topk compile disable) and
[vllm-xpu-kernels#524](https://github.com/vllm-project/vllm-xpu-kernels/pull/524)
(`at::zeros` + Muse tuple). Native vLLM MTP remains 0%. Do not cite the
superseded n=3 DFlash screen or the no-spec n=3 “10k prefill” as this path.

**llama.cpp SYCL (historical):** the recorded Q4_0 evidence used
`/mnt/<models-disk>/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf`, made locally
from Q4_K_M with `llama-quantize ... Q4_0 --allow-requantize`; it was later
removed during storage cleanup. Do **not** use the historical result as a launch
path. A verified native source such as
`bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` needs its own completed
file/hash check and, for quality-sensitive use, a fresh quality/serving check.
Recorded no-spec: 68.0 t/s at tg128/150W. Recorded MTP4: 87.4 t/s median at
230W (`n=5`, 95–99% acceptance); measured active decode draw was 134W. MTP
launch: `--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.75`.
Native MTP (nextn) is built in. Full analysis:
`research/nemotron-lightning-optimization-20260811/`.

**Muse-Glimmer details (2026-08-10, Run 33-35):** dense 30B multimodal,
Q4_K_XL + DFlash n-max=2, 128K, 230W. GGML_SYCL_F16=ON build MANDATORY
(loses 3.4× prefill without it). p512/g128 **26.8** t/s at **128K** (max 28.9,
n=5, Run 35). The **29.2** figure is the earlier **64K** n=5 cell (Run 34) —
do not mix them.
File: `/mnt/<models-disk>/muse-glimmer/Muse-Glimmer-30B-UD-Q4_K_XL.gguf` +
`/mnt/<models-disk>/muse-glimmer/dflash-kquant.gguf`.

---

## Use Case Picks

| Use Case | Model | Profile | Why |
|----------|-------|---------|-----|
| **Fastest isolated Nemotron vLLM path** | GPTQ-INT4-G64 + DFlash-BF16 | cookbook `launch-nemotron-dflash.sh` | **186.6 t/s** p2048/g128 n=5; local conversions; C1 cache-off |
| **Fastest recorded Nemotron llama.cpp path** | Local Q4_0 from Q4_K_M | Historical only; no current launch path | **87.4 t/s MTP4 median, n=5** in its named campaign; do not compare directly with Qwen or vLLM DFlash cells |
| **Quality-oriented Nemotron baseline** | Native-source Q4_K_M | Validate current file/profile first | Separate campaign around ~50 t/s MTP4; quality difference not measured locally |
| **Multimodal + reasoning** | Muse-Glimmer-30B Q4_K_XL | `muse-glimmer-dflash-230w` | Vision + reasoning, DFlash **26.8** t/s at 128K (29.2 at 64K) |
| **Best quality chat** | Qwen 35B Q5_K_M | `qwen35-q5-256k` | Highest quality quant, full 256K context |
| **Fastest MoE** | Qwen 35B Q4_K_XL | `qwen35-q4-256k` | +3 t/s over Q5, still strong quality |
| **Agent / tool use (current)** | Qwen3.8-27B GPTQ + MTP4 | `scripts/tmp/qwen38-pi-serve.sh` | MTP4 + GDN mixed-split v5 + `qwen3_xml`. Sequential agentic ~38 t/s. GGUF catalog profiles point at missing files. |
| **Agent / tool use (historical MoE)** | Qwen 35B Q5_K_M | `qwen35-q5-256k` | Long context + best quality for code/tools — **GGUF absent 2026-08-18** |
| **Dense + reasoning** | ThinkingCap 27B Q6 | `thinkingcap27-mtp-q6-128k-165w` | Historical production GGUF — **file absent 2026-08-18** |
| **Fastest dense (historical GGUF)** | ThinkingCap 27B Q4 | `thinkingcap27-mtp-q4-128k-vision-180w` | File absent 2026-08-18. Current dense serve is Qwen3.8 vLLM MTP4. |
| **Vision + reasoning** | ThinkingCap 27B Q4 | `thinkingcap27-mtp-q4-128k-vision-180w` | Projector still on `/mnt/<models-disk>/gguf/`; weights absent |
| **Best dense quality (historical)** | ThinkingCap 27B Q5 | `thinkingcap27-mtp-q5-200k-165w` | File absent 2026-08-18 |
| ⚠️ NOT for dense | ThinkingCap 27B Q4 @ 200K | `thinkingcap27-mtp-q4-200k-165w` | 200K+MTP borderline — verify VRAM ≥1 GiB free before use |

---

## Server Launch Template

```bash
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1

# Binary (latest master + kernel fusion + MMQ/XMX + F16, 2026-08-12)
BIN=~/llama.cpp-sycl-latest/build-sycl/bin/llama-server

# Nemotron 3.5 Lightning 30B-A3B Q4_0 + MTP4.
# Substitute only a verified, existing Q4_0 path. The historical 87.4 t/s
# artifact was locally requantized and removed; this is not a copy-paste launch
# command until a fresh source/hash/quality check is complete.
NEMOTRON_Q4_0=/path/to/verified/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf
$BIN \
  -m "$NEMOTRON_Q4_0" \
  --host 0.0.0.0 --port 8765 \
  -ngl 99 -ncmoe 0 \
  -fa on -ctk q8_0 -ctv q4_1 \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.75 \
  -t 8 --no-mmap \
  -c 16384 -b 8192 -ub 4096

# Muse-Glimmer-30B Q4_K_XL + DFlash (multimodal, 26.8 t/s at 128K)
$BIN \
  -m /mnt/<models-disk>/muse-glimmer/Muse-Glimmer-30B-UD-Q4_K_XL.gguf \
  -md /mnt/<models-disk>/muse-glimmer/dflash-kquant.gguf \
  --spec-type draft-dflash --spec-draft-n-max 2 \
  -fa on --jinja \
  -ngl 99 -ngld 99 -ctk q8_0 -ctv q4_1 \
  -t 8 --no-mmap \
  -c 131072 -b 8192 -ub 8192
# Add vision: --mmproj ~/models-cache/mmproj-muse-glimmer-f16.gguf

# MoE model (e.g., Qwen 35B Q5)
$BIN \
  -m ~/models-cache/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf \
  --host 0.0.0.0 --port 8765 \
  -ngl 99 -ncmoe 0 \
  -fa on -ctk q8_0 -ctv q4_1 \
  -t 8 --no-mmap \
  -c 262144 -b 8192 -ub 4096

# Dense + MTP (ThinkingCap 27B)
$BIN \
  -m ~/models-cache/ThinkingCap-Qwen3.6-27B-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8765 \
  -ngl 99 -ncmoe 0 \
  -fa on -ctk q8_0 -ctv q4_1 \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.75 \
  -t 8 --no-mmap \
  -c 204800 -b 8192 -ub 4096

# Add vision: --mmproj ~/models-cache/mmproj-MODEL-f16.gguf
```

# MoE model (e.g., Qwen 35B Q5)
$BIN \
  -m ~/models-cache/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf \
  --host 0.0.0.0 --port 8765 \
  -ngl 99 -ncmoe 0 \
  -fa on -ctk q8_0 -ctv q4_1 \
  -t 8 --no-mmap \
  -c 262144 -b 8192 -ub 4096

# Dense + MTP (ThinkingCap 27B)
$BIN \
  -m ~/models-cache/ThinkingCap-Qwen3.6-27B-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8765 \
  -ngl 99 -ncmoe 0 \
  -fa on -ctk q8_0 -ctv q4_1 \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.75 \
  -t 8 --no-mmap \
  -c 204800 -b 8192 -ub 4096

# Add vision: --mmproj ~/models-cache/mmproj-MODEL-f16.gguf
```

**Or use the launcher** (preferred): `~/inference/switch-profile qwen35-q5-256k`

---

## Context Ceilings (q8_0-q4_1 KV — MEASURED 2026-08-03)

| Model | Quant | Max safe context | VRAM free at max | Notes |
|-------|-------|-----------------|------------------|-------|
| Qwen 35B MoE | Q4_K_XL | **512K** | 1.6 GB | Context champion! |
| Qwen 35B MoE | Q5_K_M | **256K** | 2.2 GB | 512K = OOM crash |
| ThinkingCap 27B | Q4_K_M | **256K** | 0.9 GB | **WITH MTP = OVERFLOW** (verified 2026-08-05) — 128K is the MTP ceiling |
| ThinkingCap 27B | Q5_K_M | **200K** | 2.7 GB | **WITH MTP = OVERFLOW** (verified) — 128K is the MTP ceiling |
| ThinkingCap 27B | Q6_K | **128K** | 0.7 GB | MTP fits (404 MiB spare) — production config |

_Contexts above 128K apply to llama-bench/base decode only. MTP spec buffers add ~2-3 GB
(see AGENTS.md §15.3, benchmark-history Run 12)._

---

## VRAM Budget Rule of Thumb (q8_0-q4_1 KV, 128K context)

| Model type | Weights | KV cache (128K) | Total | Fits 32 GB? |
|-----------|---------|-----------------|-------|-------------|
| MoE 35B (3B active) | 20-25 GB | ~4 GB | 24-29 GB | ✅ Yes |
| Dense 27B | 15-19 GB | ~4 GB | 19-23 GB | ✅ Yes |
| Dense 27B + MTP + vision | 15 GB + 0.9 GB | ~4 GB | ~20 GB | ✅ Yes |
| Dense 35B (non-MoE) | 24-34 GB | ~8 GB | 32-42 GB | ❌ Tight/crash |

KV at 256K doubles to ~8 GB. Plan accordingly.

---

## KV Cache Quantization Reference

| Config | KL-div | VRAM savings | Verdict |
|--------|--------|--------------|---------|
| **q8_0 K + q4_1 V** | **0.0044** | **~50%** | **✅ Fleet standard** |
| q8_0 K + q8_0 V | 0.0044 | ~25% | Conservative |
| q5_0 K + q4_1 V | 0.008 | ~56% | Acceptable fallback |
| q4_0 K + q4_0 V | >0.03 | ~75% | ❌ Quality cliff |

See `kv-cache-quantization.md` for the full analysis.
