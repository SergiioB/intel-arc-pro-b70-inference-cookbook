---
title: "SGLang Xe2 dense E=1 and custom W4A16 tile — 2026-08-17"
type: research
date: 2026-08-17
status: final
---

# SGLang Xe2 dense E=1 and custom W4A16 tile — 2026-08-17

Status: measured. Decode custom tile **rejected**. E=1 dispatch **works**. Does not beat vLLM GPTQ+MTP 83.7 t/s.

Host: desktop-b70, Arc Pro B70 (`0xe223`), 32 GB, 608 GB/s, 256 XMX.
Workspace: `~/inference/research/sglang-xe2-dense/`
Model shapes: Qwen3.8-27B `text_config` hidden=5120, intermediate=17408, 64 layers (48 GDN + 16 attn).
Checkpoint: `/mnt/<models-disk>/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16-nogidx` (GPTQ 4-bit, group 128, symmetric, desc_act=false).

## 1. Question

Can SGLang’s hand-written Xe2 W4A16 kernel (`moe_grouped_mm_nt_xe20_w4a16`) run a **dense** 27B GPTQ model, and can a custom tile raise XMX use toward ~90%?

## 2. What E=1 means

SGLang only wired that kernel for MoE. A dense GEMM is the same math as MoE with `num_experts=1` and `rows_per_expert=[M]`.

Call contract (`GroupGemmW4A16Xe20.cpp`):

- activations `[M, K]` bf16/fp16
- packed_weights `[E, N, K/2]` int8 (two nibbles/byte)
- scales `[E, N, K/group]` same dtype as activations
- `is_int4=true`, group_size in {32,64,128,256}

E=1 is a **calling convention**, not a speed trick. It does not fill XMX by itself.

Stock policy menu (runtime `avg_m = total_m / E`):

| avg_m | policy | WG tile |
|---|---|---|
| ≤4 | `w4a16_policy_m_8` | 8×64×32 |
| ≤8 | `w4a16_policy_m_16` | 16×64×32 |
| ≤128 | `w4a16_policy_m_32` | 32×64×32 |
| >128 | `w4a16_policy` | 128×256×32 |

Kernel path: unpack INT4 → scale in registers → `cute::gemm` as **BF16/FP16 DPAS** (W4A16). Peak is not 367 INT8 TOPS for this path.

## 3. Hardware / format (unchanged)

Xe2 DPAS native: INT4, INT8, FP16, BF16. No `dpas.f4`. NVFP4 on B70 is software unpack. GPTQ INT4 remains the native format. vLLM dense path is `XPUwNa16LinearKernel` / oneDNN `dnnl_matmul_w4a16_int4`.

Production to beat (Run 40, 2026-08-15): Qwen3.8-27B GPTQ+MTP4 on `vllm/vllm-openai-xpu` **83.7 t/s** decode p512/g128, **1774 t/s** 8K cold prefill. Run 39 note: MTP4 decode sits at the 608 GB/s ceiling; 95 t/s needs multi-layer MTP or the MoE track, not a new GEMM.

## 4. Tries (in order)

1. **Host compile `sgl-kernel-xpu`** on oneAPI 2026.0 (`20260331`). CMake wants `SYCL >= 20250806`. After a `sycl_add_library`→`add_library` hack: `No SOURCES` on generated W4A16 insts. Abandoned. Restored `BuildOnLinux.cmake`.

2. **Full `docker/xpu.Dockerfile`** (`sglang-xpu-local`). No Hub XPU image. First build SIGTERM on SSH drop. Disk hit 100% on `/` (fixed by prune, **deleted production vLLM images**). Second build died compiling FMHA: `[Memory] OOM guard … avail=0.50GiB` while building 2019 TUs (`USE_FMHA=ON`).

3. **W4A16-only image** from cached torch layer `sglang-xpu-torch:base` (29.1 GB, torch 2.12.0+xpu, oneAPI 2025.3). `USE_FMHA=OFF USE_MLA=OFF USE_MOE=ON -j2`. Dirty local tree failed cmake; **fresh git clone** succeeded. First working wheel: `sgl-kernel 0.11.0`.

4. **E=1 microbench** (stock tiles). Op registered. Numbers in §5.

5. **Slim rebuild** (skip FMHA, LoRA insts, `GroupGemmXe20`). n256 `.so` compiled, but `common_ops.abi3.so` still references `sgemm_lora_a_fwd` / `GroupGemmXe20` launchers → import failed, op missing. A ctypes “drop broken .so” sweeper then deleted **good** W4A16 libs too (`sos_left []`). Slim path rejected.

6. **Custom decode tile `w4a16_policy_dense_n256`**: WG `8×256×32`, SGLayout `<_1,_8,_1>`. Dispatch: `avg_m<=4` uses n256 instead of m_8. Full ~483-file rebuild (stock recipe + 4 n256 insts). Compiled, imported, benched. Numbers in §6. **M=1 slower.**

7. **vLLM image restore:** pulled `vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97` (production dense launcher). Not re-run of the 83.7 matrix.

## 5. Stock E=1 microbench (15:xx first run)

Env: `sglang-xpu-torch:base`, `sgl-kernel 0.11.0`, symmetric packed `[1,N,K/2]` int8, scales `[1,N,K/128]` bf16, zeros=None.
Op: `torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_w4a16`.

| shape | K×N | M=1 ms / TF | M=8 | M=16 | M=32 | M=256 |
|---|---|---:|---:|---:|---:|---:|
| mlp_gateup | 5120×34816 | 0.58 / 0.6 | 0.43 / 6.6 | 0.26 / 22.1 | 0.22 / 51.0 | 1.08 / **84.6** |
| mlp_down | 17408×5120 | 0.28 / 0.6 | 0.27 / 5.2 | 0.20 / 14.5 | 0.20 / 28.6 | 0.81 / 56.5 |
| attn_o | 5120×5120 | 0.08 / 0.6 | 0.08 / 5.4 | 0.06 / 13.5 | 0.06 / 26.5 | 0.31 / 43.0 |

M=1 effective BW ~154–164 GB/s (**25–27% of 608**).
64-layer GEMM-only (these 3 mats, no QKV/GDN/norm): M=1 → **16.6 t/s ceiling**.

## 6. Custom 8×256 tile vs stock 8×64 (15:13)

Same harness, same shapes. n256 only affects `avg_m<=4` (the M=1 column).

| shape | M | stock 8×64 | n256 8×256 | note |
|---|---:|---:|---:|---|
| gate_up | 1 | **0.58 ms** | 0.67 ms | custom tile **worse** |
| gate_up | 8 | 0.43 ms | 0.22 ms | still old `m_16` policy; treat as noise/warmup |
| gate_up | 32 | 0.22 ms | 0.22 ms | same |
| gate_up | 256 | 1.08 ms | 1.07 ms | same |
| down | 1 | **0.28 ms** | 0.46 ms | worse |
| attn_o | 1 | **0.08 ms** | 0.10 ms | worse |

Wider N at M=1 adds empty work. Does not raise XMX occupancy on a 1-row GEMV.

## 7. XMX occupancy (what 90% actually means)

- Single-token decode (M=1) cannot fill DPAS. The array wants fat M.
- This kernel is W4A16: INT4 unpack + **BF16 DPAS**. 367 INT8 TOPS is the wrong ceiling.
- M=256 gate_up **85 TFLOP/s** is the high-M point. If BF16 XMX peak is ~90–180 TFLOPS (INT8/4 or INT8/2), that is already a large fraction of **BF16** peak — not 23% of INT8.
- Real ways to load XMX: MTP/batch so GEMM sees M=16–32; fuse GDN (48/64 layers) so XMX is not idle; optional later oneDNN vs this kernel at M≥256 for **prefill** only.

## 8. Build pitfalls (do not repeat)

- Do not `sycl_add_library`→`add_library` on oneAPI 2026.0. Use Docker 2025.3.
- Do not `docker system prune -a` — it removed the 83.7 vLLM images. Restored digest `2c427ef…`.
- Full XPU Dockerfile + `USE_FMHA=ON` OOMs 30 GB RAM (2019 TUs). `USE_FMHA=OFF` + `-j2` is the working compile.
- Slimming LoRA/`GroupGemmXe20` breaks `common_ops` import. Keep the full MOE+LoRA set if you need `import sgl_kernel`.
- Isolated `ctypes.CDLL` of every `.so` is not a liveness test; it deleted working W4A16 libs.
- SSH `nohup` without `< /dev/null` / `setsid` SIGTERMs docker build.
- `DOCKER_BUILDKIT=1` fails here (no buildx). Use `DOCKER_BUILDKIT=0`.
- oneAPI `setvars.sh` under `set -u` dies on `OCL_ICD_FILENAMES`. Isolate with `set +eu`.
- Qwen3.8-27B is **5120 / 17408**, not 3584 / 18944.

## 9. Files

On B70:

- `~/inference/research/sglang-xe2-dense/dense_wrapper.py` — E=1 wrapper, layout `[E,N,K/2]`
- `~/inference/research/sglang-xe2-dense/sglang_intercept.py` — GPTQ apply() monkeypatch (not E2E tested)
- `~/inference/research/sglang-xe2-dense/kernel_microbench.py` — shape sweep
- `~/inference/research/sglang-xe2-dense/apply_dense_n256.py` — policy + dispatch + cmake inst
- `~/inference/research/sglang-xe2-dense/build_dense_n256.sh` — docker rebuild
- logs: `build_docker.log`, `w4a16_build.log`, `dense_n256.log`

On Radxa:

- `B70-DOCS/results/sglang-xe2-dense-e1-microbench-20260817.md`
- `B70-DOCS/research/decision-sglang-e1-vs-vllm-mtp-20260817.md`

Image still on host: `sglang-xpu-torch:base`, plus restored `vllm/vllm-openai-xpu@sha256:2c427ef…`.

## 10. Verdict

1. E=1 dense dispatch on the Xe2 W4A16 kernel **works**.
2. Custom 8×256 decode tile **does not** beat stock 8×64 at M=1. Stop that line.
3. Do not replace vLLM GPTQ+MTP4 with an SGLang E2E server for this 27B. SGLang XPU MTP is ~1.3× vs vLLM ~3.3×.
4. Keep production on vLLM `XPUwNa16` + MTP4. Next decode lever is a multi-layer MTP checkpoint or the 35B MoE track. Next kernel lever worth measuring later: this op vs oneDNN at **M≥256 prefill** only, and GDN Xe20 so 48 layers do not leave XMX idle.
