---
title: "P8 AWQ entry gate — loader and XPU kernel probe (read-only, no GPU)"
type: plan
date: 2026-09-21
status: COMPLETE
phase: P8
branch: AWQ
verdict: loader+kernel supported on this stack
ledger: RL-TBD-P8
---

# P8 AWQ entry gate — loader + kernel probe

Read-only source investigation, CPU only; no GPU, model load, docker, or conversion. Decides the AWQ branch's first step (P8 plan:113,116). Trees: `~/vllm-src/vllm` @ `066ae99e` (clean; descendant of `origin/main` = `ac7509e2` = deployed engine line `0.27.2rc1.dev77+gac7509e2b`, `models/qwen/qwen3.8-27b/serving.md`:32; AWQ wiring verified present at both) and `~/vllm-src/vllm-xpu-kernels` @ `4043e76` = `v0.1.13-5-g4043e76` (deployed image carries kernels `0.1.12.3`, `serving.md:32` — see §5 caveat).

## 1. Verdict

**loader+kernel supported on this stack.**

`auto_awq.py:304` — `if current_platform.is_cpu() or current_platform.is_xpu():` → `return AutoAWQMarlinLinearMethod(self)`: an explicit named XPU branch whose own comment (`:299-303`) says it selects "XPUwNa16LinearKernel on XPU". That resolves via `kernels/linear/__init__.py:492-495` (`PlatformEnum.XPU: [XPUW4A8IntLinearKernel, XPUwNa16LinearKernel]`) to `kernels/linear/mixed_precision/xpu.py:15,33` (accepts `uint4`) and `:113` (`torch.ops._xpu_C.int4_gemm_w4a16`). The op has executed on this fleet: `results/2026/08/nemotron-gptq-builtin-profile-20260812T171658Z-671822/profiler_out_0.txt`:7 shows `_xpu_C::int4_gemm_w4a16` in a captured profiler trace. **AWQ needs no new backend.** Caveat: that trace is a *GPTQ* serve on an earlier image (`sha256:1da0a954`; champion image is `f01e24f6`, `serving.md:106`), and the AWQ load-time repack is unproven at runtime.

## 2. AWQ-relevant code paths and their gates

| Path | Symbol | XPU-gated? | Gate (quoted) |
|---|---|---|---|
| `quantization/__init__.py:140-142`; `platforms/xpu.py:113-114`; `platforms/interface.py:964` | registration + admission | yes | `"awq"`/`"auto_awq"` present in `supported_quantization`; `if cls.supported_quantization and quant not in cls.supported_quantization:` |
| `auto_awq.py:230` + `platforms/xpu.py:249-255` + `kernels/linear/__init__.py:814-817` | `get_min_capability` vs capability filter | yes, via platform | `return 75`, but XPU `get_device_capability` → `return None` ("capacity format differs from cuda's"), and the filter is `if compute_capability is not None and …` — **75 is not an XPU blocker** |
| `auto_awq.py:304-305` | `get_quant_method` XPU branch | **yes** | `if current_platform.is_cpu() or current_platform.is_xpu():` → `AutoAWQMarlinLinearMethod` |
| `auto_awq.py:427` | `AutoAWQMarlinLinearMethod.quant_type` | no | `scalar_types.uint4` |
| `auto_awq.py:432-437` | marlin verification skip | **yes** | `if not (current_platform.is_cpu() or current_platform.is_xpu()):` |
| `kernels/linear/__init__.py:774,802` | `choose_mp_linear_kernel` | yes | `platform_kernels = _POSSIBLE_KERNELS.get(current_platform._enum, [])` |
| `kernels/linear/__init__.py:492-495` | XPU kernel list | **yes** | `PlatformEnum.XPU: [XPUW4A8IntLinearKernel, XPUwNa16LinearKernel]` |
| `mixed_precision/xpu.py:20,27` | `XPUwNa16LinearKernel.can_implement` | **yes** | `if not current_platform.is_xpu(): return False, "XPUwNa16 only supported on XPU"` |
| `xpu.py:15,33,40,47` + `:56-121` | constraints, weight prep, dispatch | yes | `(scalar_types.uint4, scalar_types.uint4b8)`; group/input size multiples of 32; transposes scales, synthesizes `w_zp = 8` when symmetric; `torch.ops._xpu_C.int4_gemm_w4a16(reshaped_x, w_q.t(), …)` |
| `xpu.py:142-146` | `XPUW4A8IntLinearKernel` weight gate | yes | requires `scalar_types.int4` — **unreachable from AWQ** (`uint4`) |
| `auto_awq.py:93,533` | `_convert_awq_to_standard_format` | no | called from `process_weights_after_loading` |

Kernel side: decl `csrc/xpu/ops.h:43`; bind `csrc/xpu/torch_bindings.cpp:37` (`xpu_ops.impl("int4_gemm_w4a16", torch::kXPU, &int4_gemm_w4a16)`); impl `csrc/xpu/onednn/onednn_matmul.cpp:199` (→ `oneDNN::dnnl_matmul_w4a16_int4`); built via glob `CMakeLists.txt:650`; exercised by `tests/test_int4_gemm_onednn.py:45` (SYM/ASYM × act_order).

## 3. Kernel situation: packing change, not a new kernel

AWQ W4A16 compute maps onto the **same XPU GEMM the GPTQ-INT4 champion already uses**. All differences are load-time: `qweight` is packed along the output dim in AWQ bit order, and `_convert_awq_to_standard_format` (`auto_awq.py:93`) unpacks, reorders by `_REVERSE_AWQ_PACK_ORDER` (`:77,121-127`) and repacks along the input dim — a bit permutation, no arithmetic; `qzeros` likewise (`:138-158`, target stated at `:142-144` as "matches the CompressedTensors layout expected by the kernels"); `scales` are untouched by the converter and transposed by the kernel (`xpu.py:70-81`). Honest reading: **"just a packing change"** — conditional on the repack proving lossless (P8:114) and the asymmetric zero-point path working, neither established by source reading alone. Since the arithmetic equals the champion's, P8's *speed ≈ 1.0×* expectation (`P8:41-42`) holds by construction; the only possible win is quality-at-iso-bytes.

## 4. Prior-attempt check

- **AWQ: never attempted.** `grep -i awq registry/research-ledger.json` → 0 hits; no AWQ artifacts or measurements anywhere in the repo; `serving.md` has 0 AWQ mentions. `docs/reference/quantization-decision-record.md`:132-134: "**AWQ itself was not measured** (no artifacts; Decision 3 still requires exact loader/kernel demonstration before any AWQ claim) ... it remains unevaluated on this host with no route to displace the GPTQ-packed path." The "no route" clause is **superseded by this probe**.
- **W4A8: attempted and rejected — via a local patch on the very kernel AWQ uses.** `scripts/tmp/patch_dense_w4a8.py`:1-17 adds an env-gated (`B70_DENSE_W4A8=1`) W4A8 path "to `XPUwNa16LinearKernel` (used by dense GPTQ/AWQ)". Outcome, `results/2026/08/qwen38-gguf-20260814T150404/manifest.txt`:110: "W4A8 (`B70_DENSE_W4A8=1`, patch applied): NO prefill gain (1398/1654 cold) AND output corruption (Paris prompt degraded to repeated exclamation marks) ... **W4A8 confirmed DEAD for this artifact family. Do not retry.**" An earlier 230 W arm had shown +40-43 % prefill (`archive/docs/pre-migration-20260819/benchmark-history.md`:214-232); the corruption finding closed it. No W4A8 ledger entry exists either. Implication: W4A8 stays gated behind a *changed hypothesis* (P8:55-61); the default AWQ path is unaffected.
- **Do not conflate engines.** `awq` is "silently broken on XPU (HTTP 200, garbage content)" in **SGLang** (`.agents/skills/sglang-xpu-run/SKILL.md:46,262,301`) — a different engine, not evidence about vLLM.
- **Stale skill guidance, flagged.** `.agents/skills/vllm-xpu-profile/SKILL.md`:130-131 and `.agents/skills/vllm-xpu-run/references/quantization.md`:24-25 tell readers to expect `int4_gemm_w4a8` for AWQ and to treat `int4_gemm_w4a16` as a "silent fall-through (vLLM #38064)". At the deployed revision that is wrong: W4A8 requires `weight_type == scalar_types.int4` (`xpu.py:142-146`) while AWQ sets `uint4` (`auto_awq.py:427`), so `int4_gemm_w4a16` **is** AWQ's correct W4A16 destination. `research/2026/intel-vllm-xpu-skills-vs-b70-observations-20260822.md`:144 already recorded this failure mode as "Cleared on current image".

## 5. Smallest falsifying test, and the stop rule

**Falsifier (one command, no model load, no conversion).** Ask the pinned image whether the op is registered in its shipped kernels wheel:

```sh
docker run --rm --device=/dev/dri -e ZE_AFFINITY_MASK=0 --entrypoint python3 \
  vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f \
  -c "import torch, torch.xpu; torch.xpu.init(); print(torch.ops._xpu_C.int4_gemm_w4a16)"
```

Op resolves → kernel half confirmed on the deployed wheel; `AttributeError` falsifies the verdict → downgrade to `loader only / no XPU kernel`. This addresses the one real provenance gap: the local kernels tree is `v0.1.13-5-g4043e76` while the deployed wheel is `0.1.12.3` (`serving.md:32`), so source-level presence is not by itself proof for the deployed wheel.

**Confirming probe (the informative positive, when GPU work resumes).** Load any AWQ W4-G128 checkpoint for one model, read logs for `Using XPUwNa16LinearKernel for AutoAWQMarlinLinearMethod` (`auto_awq.py:471-472`), then a 2-3 sample coherence check plus output comparison against a reconstructed reference. Per P8:113-115 that is the gate before any conversion is paid for.

**Stop rule if negative.** Close the AWQ branch at this gate — do **not** write a new AWQ/W4A8 backend. Record a ledger non-recommendation naming the missing symbol, citing `P8-bounded-quant-screen.md:116` ("Stop cheaply if no credible compatible route exists") and `:149` ("Stop AWQ if compatibility cost becomes a new backend project"), and spend the remaining P8 budget on the rotated-GPTQ branch (P8:45-52).

**Not covered.** No runtime execution of any kind; the AWQ repack (`auto_awq.py:93`) and asymmetric zero-point handling (`xpu.py:83-87`) are unproven at runtime here; no AWQ checkpoint exists locally; the MoE path (`AutoAWQMoEMethod`, `auto_awq.py:674`) is out of scope for dense Qwen3.8-27B (`serving.md:392`).

## Falsifying test — executed 2026-09-21 (result: verdict HOLDS)

The named smallest falsifying test was run against the **deployed local image** (not the
champion digest, which is being pulled separately):

```bash
docker run --rm --device=/dev/dri -e ZE_AFFINITY_MASK=1 --entrypoint python3 \
  vllm/vllm-openai-xpu@sha256:2c427ef4… -c "import torch, torch.xpu; torch.xpu.init(); print(torch.ops._xpu_C.int4_gemm_w4a16)"
```

Observed:

```text
vllm 0.26.1rc1.dev457+gc810e5ee9
vllm-xpu-kernels 0.1.12
OP_PRESENT int4_gemm_w4a16 = _xpu_C.int4_gemm_w4a16
int4/w4a ops: ['int4_gemm_w4a16']
```

So the op the AWQ path resolves to **exists in the deployed wheel**, and `int4_gemm_w4a16` is
the only int4/w4a variant present — which also confirms there is no separate `w4a8` op in this
wheel (consistent with the earlier note that W4A8 was a local patch, not a shipped path).

Version position recorded: deployed wheel **0.1.12** vs local source tree **v0.1.13-5-g4043e76**.
Any later AWQ conversion must be checked against the wheel that will actually serve it, not the
tree — this gap is the one open item before the iso-byte screen.
