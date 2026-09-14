#!/usr/bin/env python3
"""Apply the Xe2 small-M BF16xBlockFP8 kernel patches to the build tree
(b70-architect authored, 2026-08-29). Run on the host AFTER the baseline
wheel build completes; then run an incremental rebuild.

Patches: P1 block-scaled inner loop (gemm_xe2.hpp); P2 launcher/scheduler
template param (interface.hpp + xe2.hpp); P3 E=1 impl (interface.hpp);
P4 wrappers (xe2.h/.cpp); P5 arch dispatch (grouped_gemm_interface.*);
P6 TORCH_LIBRARY registration (torch_bindings.cpp).
Fails loudly on any anchor mismatch. Idempotent per-patch via marker greps.

Source checkout: set VXK_SRC to the vllm-xpu-kernels source root
(e.g. VXK_SRC=$PWD/vllm-xpu-kernels/src), anchored at the exact source
generation the wheel was built from.
"""
from pathlib import Path
import sys
import os

SRC = Path(os.environ.get("VXK_SRC", "vllm-xpu-kernels/src/csrc/xpu"))
bad = []


def apply(path: Path, old: str, new: str, name: str, expect: int = 1,
          marker: str | None = None):
    t = path.read_text()
    if marker and marker in t:
        print(f"[B70_SMALLM] {name}: already applied — ok")
        return
    if old == "@EOF@" or old == "@HASH_END@":
        # EOF-append semantics (no count check needed on the sentinel)
        if old == "@HASH_END@" and t.count("#endif") != 0:
            bad.append(f"{name} (#endif present, append unsafe)")
            return
        path.write_text(t.rstrip() + "\n" + new)
        print(f"[B70_SMALLM] {name}: appended at EOF")
        return
    n = t.count(old)
    if n != expect:
        bad.append(f"{name} (found {n}, want {expect})")
        return
    path.write_text(t.replace(old, new))
    print(f"[B70_SMALLM] {name}: applied")

# ---------- P1: block-scaled inner loop (gemm_xe2.hpp) ----------
G = SRC / "grouped_gemm/xe_2/gemm_xe2.hpp"
BLOCK_FP8_FN = r'''
template <
    class GmemTiledCopyA,
    class GmemTiledCopyB,
    class GmemTiledCopyC,
    class ATensor,
    class BTensor,
    class DTensor,
    class TiledMMA,
    typename ElementS,
    typename ElementBI>
CUTE_DEVICE void xe_gemm_block_fp8(
    ATensor const& A,  // (M,K)
    BTensor const& B,  // (N,K)
    const ElementS* Scales,  // contiguous [K/128, N/128]
    const ElementBI* Bias,
    DTensor& C,  // (M,N)
    Coord<int, int, cute::Underscore, int> blk_coord,
    TiledMMA const& mma) {
  using TB = typename BTensor::element_type;
  static_assert(
      std::is_same_v<TB, cutlass::float_e4m3_t>,
      "xe_gemm_block_fp8 only supports FP8 E4M3 weights");
  static_assert(
      std::is_same_v<ElementS, float>,
      "xe_gemm_block_fp8 requires FP32 scales");

  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto wg_m = get<0>(blk_coord);
  auto wg_n = get<1>(blk_coord);
  int local_id = item.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());
  Tensor cC = make_identity_tensor(C.shape());

  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(
      cA, select<0, 2>(wg_tile), make_coord(wg_m, _));  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(
      cB, select<1, 2>(wg_tile), make_coord(wg_n, _));  // (BLK_N,BLK_K,k)
  Tensor gC =
      local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{});  // (BLK_M,BLK_N)

  auto copy_a = get_block_2d_copy_A<GmemTiledCopyA>(mma, A);
  auto copy_b = get_block_2d_copy_B<GmemTiledCopyB>(mma, B);
  auto copy_c = get_block_2d_copy_D<GmemTiledCopyC>(mma, C);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);
  auto thr_copy_c = copy_c.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  auto tCrC = thr_mma.partition_sg_fragment_C(gC);
  auto tCrC_block = make_fragment_like(tCrC);
  auto tCrC_out = thr_copy_c.partition_sg_fragment_S(gC);
  auto tCgC = thr_copy_c.partition_D(gC);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  constexpr int prefetch_dist = 3;
  constexpr int barrier_scope = 2;
  constexpr int block_k = 128;
  constexpr int block_n = 128;

  static constexpr auto tile_m = get<0>(wg_tile);
  static constexpr auto tile_n = get<1>(wg_tile);
  static constexpr auto tile_k = get<2>(wg_tile);

  static_assert(
      block_k % tile_k == 0,
      "The Xe2 FP8 K tile must divide the K=128 scale block");
  static_assert(
      tile_n <= block_n && block_n % tile_n == 0,
      "The Xe2 FP8 N tile must fit within and divide the N=128 scale block");

  constexpr int k_tiles_per_scale = block_k / tile_k;

  int k_tile_count = ceil_div(shape<1>(A), tile_k);
  int k_tile_prefetch = 0;

  const int n_blocks = shape<0>(B) / block_n;
  const int n_block_idx = (wg_n * tile_n) / block_n;

  clear(tCrC);
  clear(tCrC_block);

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; ++k_tile_prefetch) {
    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }
  }

  for (int k_tile = 0; k_tile < k_tile_count;
       ++k_tile, ++k_tile_prefetch) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    // DPAS/XMX accumulation for one part of the current K=128 block.
    cute::gemm(mma, tCrA, tCrB, tCrC_block);

    const bool end_of_scale_block =
        ((k_tile + 1) % k_tiles_per_scale) == 0;

    if (end_of_scale_block) {
      const int k_block_idx = (k_tile * tile_k) / block_k;
      const float B_scale =
          Scales[k_block_idx * n_blocks + n_block_idx];

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tCrC.size(); ++i) {
        tCrC(i) += tCrC_block(i) * B_scale;
      }
      clear(tCrC_block);
    }

    barrier_wait(barrier_scope);
  }

  if (Bias != nullptr) {
    static constexpr auto ATOM_M =
        get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_N =
        get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());

    auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;

    static constexpr auto SG_M = tile_m / ATOM_M;
    static constexpr auto SG_N = tile_n / ATOM_N;

    int sg_local_id = cutlass::get_sub_group_local_id();
    static constexpr int sg_local_range = 16;

    int n_tile_start = wg_n * tile_n;
    int n_sg_start = sg_local_n_coord * SG_N;

    CUTLASS_PRAGMA_UNROLL
    for (int sn = 0; sn < SG_N / sg_local_range; ++sn) {
      int sg_local_n = sn * sg_local_range + sg_local_id;
      float b_float = Bias[n_tile_start + n_sg_start + sg_local_n];

      CUTLASS_PRAGMA_UNROLL
      for (int sm = 0; sm < SG_M; ++sm) {
        tCrC(sn * SG_M + sm) += b_float;
      }
    }
  }

  reorder(tCrC, tCrC_out);
  copy(copy_c, tCrC_out, tCgC);
}
'''
apply(G, "CUTE_DEVICE void xe_gemm_4bits(",
      BLOCK_FP8_FN + "\nCUTE_DEVICE void xe_gemm_4bits(",
      "P1 block-scaled inner loop", 1, marker="xe_gemm_block_fp8")

# ---------- P2: interface launcher/scheduler template param ----------
I = SRC / "grouped_gemm/xe_2/grouped_gemm_xe2_interface.hpp"
apply(I, "template <typename, typename, typename, typename, char, char, class>\nclass GemmCuteName;",
      "template <typename, typename, typename, typename, char, char, class, bool>\nclass GemmCuteName;",
      "P2a GemmCuteName fwd param", 1, marker="char, char, class, bool>")
apply(I, "template <\n    char layoutA,\n    char layoutB,\n    class policy,\n    typename ElementA,",
      "template <\n    char layoutA,\n    char layoutB,\n    class policy,\n    bool BlockScaledFP8,\n    typename ElementA,",
      "P2b launcher template param", 1)
apply(I, "        layoutA,\n        layoutB,\n        policy>>(",
      "        layoutA,\n        layoutB,\n        policy,\n        BlockScaledFP8>>(",
      "P2c kernel name arg", 1)
apply(I, "              layoutA,\n              layoutB,\n              'R'>(",
      "              layoutA,\n              layoutB,\n              'R',\n              BlockScaledFP8>(",
      "P2d MoEGEMM call arg", 1)
apply(I, "  MoEGEMMLauncher<LayoutA, LayoutB, Policy>(",
      "  MoEGEMMLauncher<LayoutA, LayoutB, Policy, false>(",
      "P2e preserve grouped behavior", 1)

# ---------- P2.2: scheduler scale ptr + gemm call (xe2.hpp) ----------
H = SRC / "grouped_gemm/xe_2/grouped_gemm_xe2.hpp"
apply(H, "    char LayoutKindA,\n    char LayoutKindB,\n    char LayoutKindD,\n    class TiledMMA,",
      "    char LayoutKindA,\n    char LayoutKindB,\n    char LayoutKindD,\n    bool BlockScaledFP8,\n    class TiledMMA,",
      "P2f scheduler template param", 1)
apply(H, "    ElementS* ptr_Scales_curr_batch = const_cast<ElementS*>(Scales) + expert_id;\n    if constexpr (is_B_4bits) {\n      ptr_Scales_curr_batch =\n          const_cast<ElementS*>(Scales) + B_offset * 2 / group_size;\n    }",
      "    ElementS* ptr_Scales_curr_batch = const_cast<ElementS*>(Scales);\n    if constexpr (is_B_4bits) {\n      ptr_Scales_curr_batch =\n          const_cast<ElementS*>(Scales) + B_offset * 2 / group_size;\n    } else if constexpr (!BlockScaledFP8) {\n      // Existing FP8 grouped GEMM uses one scalar scale per expert.\n      ptr_Scales_curr_batch += expert_id;\n    }",
      "P2g scale ptr selection", 1)
apply(H, "      } else {\n        xe_gemm<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD>(\n            A_tensor,\n            B_tensor,\n            ptr_Scales_curr_batch,\n            ptr_Bias_curr_batch,\n            D_tensor,\n            tile_coord,\n            mma);\n      }",
      "      } else {\n        if constexpr (BlockScaledFP8) {\n          xe_gemm_block_fp8<\n              GmemTiledCopyA,\n              GmemTiledCopyB,\n              GmemTiledCopyD>(\n              A_tensor,\n              B_tensor,\n              ptr_Scales_curr_batch,\n              ptr_Bias_curr_batch,\n              D_tensor,\n              tile_coord,\n              mma);\n        } else {\n          xe_gemm<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD>(\n              A_tensor,\n              B_tensor,\n              ptr_Scales_curr_batch,\n              ptr_Bias_curr_batch,\n              D_tensor,\n              tile_coord,\n              mma);\n        }\n      }",
      "P2h block-scaled gemm dispatch", 1)

# ---------- P3: E=1 impl (interface.hpp, before MoE close) ----------
IMPL = r'''
at::Tensor xe2_block_fp8_small_m_impl(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& B_scale,
    const c10::optional<at::Tensor>& bias) {
  const at::DeviceGuard device_guard(A.device());

  TORCH_CHECK(A.is_xpu(), "A must be an XPU tensor");
  TORCH_CHECK(B.is_xpu(), "B must be an XPU tensor");
  TORCH_CHECK(B_scale.is_xpu(), "B_scale must be an XPU tensor");
  TORCH_CHECK(B.device() == A.device(), "A and B must be on the same XPU device");
  TORCH_CHECK(
      B_scale.device() == A.device(),
      "A and B_scale must be on the same XPU device");

  TORCH_CHECK(A.dim() == 2, "A must be 2D [M, K]");
  TORCH_CHECK(B.dim() == 2, "B must be 2D [K, N]");
  TORCH_CHECK(
      B_scale.dim() == 2,
      "B_scale must be 2D [K/128, N/128]");

  const int64_t M = A.size(0);
  const int64_t K = A.size(1);

  TORCH_CHECK(M >= 1 && M <= 40, "xe2_block_fp8_small_m requires 1 <= M <= 40");
  TORCH_CHECK(K % 128 == 0, "K must be divisible by 128");
  TORCH_CHECK(B.size(0) == K, "B must have shape [K, N]");

  const int64_t N = B.size(1);

  TORCH_CHECK(N % 128 == 0, "N must be divisible by 128");
  TORCH_CHECK(
      A.scalar_type() == at::kBFloat16 || A.scalar_type() == at::kHalf,
      "A must be BF16 or FP16");
  TORCH_CHECK(
      B.scalar_type() == at::kFloat8_e4m3fn,
      "B must be FP8 E4M3FN");
  TORCH_CHECK(
      B_scale.scalar_type() == at::kFloat,
      "B_scale must be FP32");

  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(
      B_scale.is_contiguous(),
      "B_scale must be contiguous [K/128, N/128]");
  TORCH_CHECK(
      B_scale.size(0) == K / 128,
      "B_scale.size(0) must equal K/128");
  TORCH_CHECK(
      B_scale.size(1) == N / 128,
      "B_scale.size(1) must equal N/128");

  const bool B_is_kn_contiguous = B.is_contiguous();
  const bool B_is_transposed_nk =
      B.stride(0) == 1 && B.stride(1) == K;

  TORCH_CHECK(
      B_is_kn_contiguous || B_is_transposed_nk,
      "B must be contiguous [K, N] or a transpose view of contiguous [N, K]");

  if (bias.has_value()) {
    TORCH_CHECK(bias->is_xpu(), "bias must be an XPU tensor");
    TORCH_CHECK(
        bias->device() == A.device(),
        "bias must be on the same XPU device as A");
    TORCH_CHECK(bias->dim() == 1, "bias must be 1D [N]");
    TORCH_CHECK(bias->size(0) == N, "bias must have N elements");
    TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(
        bias->scalar_type() == A.scalar_type(),
        "bias dtype must match A dtype");
  }

  auto& dpcpp_queue =
      at::xpu::getCurrentXPUStream(A.device().index()).queue();

  at::Tensor output =
      at::empty({M, N}, A.options().dtype(at::kBFloat16));
  at::Tensor rows_per_expert =
      at::full({1}, M, A.options().dtype(at::kInt));
  at::Tensor atomic_buffer =
      at::empty({1}, A.options().dtype(at::kInt));

#define XE2_BLOCK_FP8_LAUNCH(LayoutB, Policy, ElementAType)                    \
  MoEGEMMLauncher<'R', LayoutB, Policy, true>(                                 \
      dpcpp_queue,                                                             \
      reinterpret_cast<const ElementAType*>(A.data_ptr()),                     \
      reinterpret_cast<const float_e4m3_t*>(B.data_ptr()),                     \
      reinterpret_cast<const float*>(B_scale.data_ptr()),                      \
      bias.has_value()                                                         \
          ? reinterpret_cast<const ElementAType*>(bias->data_ptr())            \
          : static_cast<const ElementAType*>(nullptr),                         \
      reinterpret_cast<bfloat16_t*>(output.data_ptr()),                        \
      static_cast<int>(N),                                                     \
      static_cast<int>(K),                                                     \
      reinterpret_cast<const int*>(rows_per_expert.data_ptr()),                \
      1,                                                                       \
      128,                                                                     \
      reinterpret_cast<int32_t*>(atomic_buffer.data_ptr()));

#define XE2_BLOCK_FP8_SELECT_POLICY(LayoutB, ElementAType) \
  if (M <= 8) {                                            \
    using policy = w8a16_policy_m_16;                      \
    XE2_BLOCK_FP8_LAUNCH(LayoutB, policy, ElementAType);   \
  } else {                                                 \
    using policy = w8a16_policy_m_32;                      \
    XE2_BLOCK_FP8_LAUNCH(LayoutB, policy, ElementAType);   \
  }

  if (B_is_kn_contiguous) {
    if (A.scalar_type() == at::kBFloat16) {
      XE2_BLOCK_FP8_SELECT_POLICY('R', bfloat16_t);
    } else {
      XE2_BLOCK_FP8_SELECT_POLICY('R', half_t);
    }
  } else {
    if (A.scalar_type() == at::kBFloat16) {
      XE2_BLOCK_FP8_SELECT_POLICY('C', bfloat16_t);
    } else {
      XE2_BLOCK_FP8_SELECT_POLICY('C', half_t);
    }
  }

#undef XE2_BLOCK_FP8_SELECT_POLICY
#undef XE2_BLOCK_FP8_LAUNCH

  return output;
}
'''
apply(I, "}  // namespace MoE", IMPL + "\n}  // namespace MoE",
      "P3 E=1 impl", 1, marker="xe2_block_fp8_small_m_impl")

# ---------- P4: xe2 wrappers (EOF append; headers have no #endif) ----------
H4 = SRC / "grouped_gemm/xe_2/grouped_gemm_xe2.h"
C4 = SRC / "grouped_gemm/xe_2/grouped_gemm_xe2.cpp"
apply(H4, "@HASH_END@",
      "\ntorch::Tensor xe2_block_fp8_small_m_xe2(\n"
      "    const torch::Tensor& A,\n"
      "    const torch::Tensor& B,\n"
      "    const torch::Tensor& B_scale,\n"
      "    const c10::optional<torch::Tensor>& bias);\n",
      "P4a xe2.h decl", 1, marker="xe2_block_fp8_small_m_xe2")
apply(C4, "@EOF@",
      "\ntorch::Tensor xe2_block_fp8_small_m_xe2(\n"
      "    const torch::Tensor& A,\n"
      "    const torch::Tensor& B,\n"
      "    const torch::Tensor& B_scale,\n"
      "    const c10::optional<torch::Tensor>& bias) {\n"
      "  return MoE::xe2_block_fp8_small_m_impl(A, B, B_scale, bias);\n}\n",
      "P4b xe2.cpp def", 1, marker="MoE::xe2_block_fp8_small_m_impl")

# ---------- P5: arch dispatch (EOF append on .h; .cpp EOF append) ----------
H5 = SRC / "grouped_gemm/grouped_gemm_interface.h"
C5 = SRC / "grouped_gemm/grouped_gemm_interface.cpp"
apply(H5, "@HASH_END@",
      "\ntorch::Tensor xe2_block_fp8_small_m(\n"
      "    const torch::Tensor& A,\n"
      "    const torch::Tensor& B,\n"
      "    const torch::Tensor& B_scale,\n"
      "    const c10::optional<torch::Tensor>& bias);\n",
      "P5a interface.h decl", 1, marker="xe2_block_fp8_small_m(")
apply(C5, "@EOF@",
      "\ntorch::Tensor xe2_block_fp8_small_m(\n"
      "    const torch::Tensor& A,\n"
      "    const torch::Tensor& B,\n"
      "    const torch::Tensor& B_scale,\n"
      "    const c10::optional<torch::Tensor>& bias) {\n"
      "  TORCH_CHECK(\n"
      "      !vllm::xpu::force_xe_default_kernel(),\n"
      "      \"xe2_block_fp8_small_m cannot use the XE-default kernel path\");\n"
      "  TORCH_CHECK(\n"
      "      vllm::xpu::is_xe2_arch() || vllm::xpu::is_xe3_arch(),\n"
      "      \"xe2_block_fp8_small_m requires an Xe2/Xe3 architecture\");\n"
      "#ifdef VLLM_XPU_ENABLE_XE2\n"
      "  return xe2_block_fp8_small_m_xe2(A, B, B_scale, bias);\n"
      "#else\n"
      "  TORCH_CHECK(\n"
      "      false,\n"
      "      \"xe2_block_fp8_small_m requires VLLM_XPU_ENABLE_XE2\");\n"
      "#endif\n}\n",
      "P5b interface.cpp def", 1, marker="requires VLLM_XPU_ENABLE_XE2")

# ---------- P6: TORCH_LIBRARY registration ----------
B = SRC / "torch_bindings.cpp"
apply(B, '  xpu_ops.impl(\n      "cutlass_grouped_gemm_interface",\n      torch::kXPU,\n      &cutlass_grouped_gemm_interface);\n#endif',
      '  xpu_ops.impl(\n      "cutlass_grouped_gemm_interface",\n      torch::kXPU,\n      &cutlass_grouped_gemm_interface);\n\n'
      '  xpu_ops.def(\n      "xe2_block_fp8_small_m(Tensor A, Tensor B, Tensor B_scale, "\n      "Tensor? bias) -> Tensor");\n'
      '  xpu_ops.impl(\n      "xe2_block_fp8_small_m",\n      torch::kXPU,\n      &xe2_block_fp8_small_m);\n#endif',
      "P6 op registration", 1, marker="xe2_block_fp8_small_m")

if bad:
    print(f"[B70_SMALLM] FATAL — anchor problems: {bad}")
    sys.exit(1)
print("[B70_SMALLM] all patches applied")