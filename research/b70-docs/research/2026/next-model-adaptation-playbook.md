---
title: "Next-model adaptation playbook (B70)"
type: research
date: 2026-09-15
status: final
---

# Next-model adaptation playbook (B70)

Read this before porting a new architecture. It is the compressed memory of
Nemotron 3.5 Lightning, Qwen3.6, and Muse-Glimmer — what worked, what lied,
and which gate to run first.

## 0. Four separate questions

1. **Does the engine implement the architecture?** Conversion cannot add a
   missing model class (Muse vLLM was blocked on PR #51655).
2. **What is the actual checkpoint format?** Read `config.json` and tensor
   names. Repo titles (`AWQ`, `NVFP4`, `GPTQ`) lie.
3. **Does the exact quant dispatch on XPU?** GPTQ-INT4 / compatible
   compressed-tensors W4A16 through `XPUwNa16*` is the proven family.
   NVFP4, MLX affine, bitsandbytes are not drop-in.
4. **Is the number honest?** Architecture support ≠ speed ≠ quality.

## 1. Pick the speculation path last

| Path | When it works | When it dies |
|---|---|---|
| Native MTP | Dense Qwen3.6 on vLLM and llama.cpp; GGUF Nemotron MTP-4 | NemotronH **vLLM MTP = 0%** (bad draft logits). Do not assume the head is wired. |
| DFlash | Muse llama.cpp (n_max=2); **Nemotron vLLM XPU n_spec=7** after NVFP4→BF16 | Official NVFP4 draft will not load; leftover `hf_quant_config.json` makes vLLM treat BF16 as NVFP4. Over-drafting kills acceptance (Muse n8 crashes). |
| N-gram | never as production on this XPU stack | Temp-0 nondeterministic. |
| No-spec + XPU graphs | Nemotron 21.8 → 93 t/s | Needs native grouped-topk; compiled `grouped_topk` asserts. |

**Screen speculation. Do not infer it from the architecture label.**

## 2. Format playbook

- Prefer a **native** artifact that already matches the engine.
- GPTQ for vLLM XPU: **symmetric**, group 64 (Nemotron) or the published
  group, `desc_act=false`. Asymmetric GPTQ: ordinary linear can carry
  qzeros after a narrow patch; **fused MoE WNA16 is symmetric and drops
  expert qzeros**. Never relabel `sym=true`.
- NVFP4 → BF16 for drafts is a **reconstruction**, not a quant. Keep the
  source. Rename leftover ModelOpt json.
- MLX INT4 → GPTQ is dequant/requant, not a transpose.
- GGUF Q4_0 helped **this** Nemotron on llama.cpp. It is not a Xe2 law.
  Never confuse weight Q4_0 with KV `q4_0 K` (catastrophic).

## 3. Measurement playbook (the lies we already paid for)

- HTTP prefill + prefix cache = silent 5×. Unique entropy-first prefix,
  zero `prefix_cache_hits` delta, or it is not a prefill number.
- `prompt_tokens / TTFT` = **cold input rate**, not engine prefill.
- vLLM client post-first ≠ llama-bench `tg`.
- n=3 screens are provisional. Isolated n=5 is the public median.
- `Restart=on-failure` will respawn `b70dense` if you only `docker rm`.
  Pin `Restart=no` for the whole load+matrix. pgrep of `vllm` matches the
  agent shell — match real binaries.
- Production unit needs `XDG_RUNTIME_DIR` + `DBUS_SESSION_BUS_ADDRESS`.
- llama-bench hangs after printing. Kill after the last line.
- `GGML_SYCL_F16=ON` missing = 3.4× fake Muse “slow prefill”.
- Do not headline a wide family cell (Nemotron DFlash p512 140–220). Use
  the tighter real-workload cell (p2048/g128 186.6).

## 4. Kernel / graph playbook

Eager Nemotron was 21.8 t/s because of **~950 launches/token**, not because
MoE is slow. Graphs need:

1. `at::zeros` on grouped-GEMM atomic (else graph MoE is non-deterministic).
2. Native XPU `grouped_topk` + `torch.compiler.disable` (else compile assert).
3. `VLLM_XPU_ENABLE_XPU_GRAPH=1` + PIECEWISE+FULL.
4. Optional SSU B8/W4 (small).

`--enforce-eager` is the safe compiled-mode bypass, not a fix.

## 5. Isolation / VRAM

Empty card ~31k+ MiB `visible_avail`. One engine. After load: <500 abort;
500–1023 capacity only; ≥2048 preferred C1. Dense 27B 128K needs **fp8 KV**.
Never `-ngl 99` a single GGUF >30 GB (hard crash).

## 6. First day on a new model

1. Live `df -h` / `findmnt`. Do not use remembered free space.
2. Architecture gate in the **exact** image digest.
3. Metadata + tensor audit. Preserve the source.
4. Coherence smoke (greedy, ignore_eos).
5. Isolated Lane 1 n=5 (or n=3 screen → fresh n=5).
6. Spec screen only after no-spec is honest.
7. Claims audit before any public number.

Canonical DFlash n=5:
`results/2026/08/nemotron-dflash-bf16-n7-n5-20260813T082203Z-1407994/`  
PR list: `research/2026/vllm-xpu-pr-readiness-20260813.md`  
10k input myth: `research/2026/nemotron-cold-input-vs-10k.md`
