#include "csrc/sycl_first.h"
#include <torch/extension.h>
#include "csrc/utils.h"

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/float8.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/numeric_types.h"
#include "cutlass/platform/platform.h"

#include "csrc/xpu/grouped_gemm/xe_2/gemm_xe2_policy.hpp"
#include "csrc/xpu/grouped_gemm/xe_2/gemm_xe2.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace InklingNVFP4 {
using namespace cute;
using namespace MoE;

CUTE_DEVICE float fast_e4m3_to_f32(uint32_t u) {
  // Exact branchless IEEE-754 conversion for non-negative FP8 E4M3FN scales (sign=0).
  // Normal (u >= 8): exp = (u >> 3) + 120, mant = (u & 7) << 20 -> (u << 20) + (120u << 23).
  // Subnormal (u < 8): u * 2^-9.
  uint32_t norm_bits = (u << 20) + (120u << 23);
  float norm_val = sycl::bit_cast<float>(norm_bits);
  float sub_val = static_cast<float>(u) * (1.0f / 512.0f);
  return (u < 8u) ? sub_val : norm_val;
}

template <
    class GmemTiledCopyA,
    class GmemTiledCopyB,
    class GmemTiledCopyC,
    class ATensor,
    class BTensor,
    class DTensor,
    class TiledMMA>
CUTE_DEVICE void xe_gemm_nvfp4_gs16(
    ATensor const& A,
    BTensor const& B,
    const cutlass::float_e4m3_t* Scales,
    float expert_scale,
    DTensor& C,
    Coord<int, int, cute::Underscore, int> blk_coord,
    TiledMMA const& mma) {
  using TA = typename ATensor::element_type;
  static constexpr int group_size = 16;
  static constexpr int sg_local_range = 16;
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto wg_m = get<0>(blk_coord);
  auto wg_n = get<1>(blk_coord);
  int local_id = item.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());
  Tensor cC = make_identity_tensor(C.shape());

  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));
  Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{});

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
  auto tCrC_out = thr_copy_c.partition_sg_fragment_S(gC);
  auto tCgC = thr_copy_c.partition_D(gC);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);
  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);
  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  constexpr SPIRVScope barrier_scope = ScopeWorkgroup;
  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  static constexpr auto ATOM_N = get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto tile_n = get<1>(wg_tile);
  static constexpr auto tile_k = get<2>(wg_tile);
  static constexpr auto SG_N = tile_n / ATOM_N;
  static constexpr int prefetch_dist = 6;

  static constexpr auto thr_N = get<1>(tCrB.shape());
  static constexpr auto thr_K = get<2>(tCrB.shape());
  static constexpr auto channel_num = get<0>(get<0>(tCrB.shape()));
  static_assert(tile_k == 32, "NVFP4 kernel assumes tile_k == 32");
  static_assert(thr_K == 2, "NVFP4 kernel assumes thr_K == 2 (two 16-wide K groups)");

  auto n_tile_start = wg_n * tile_n;
  int sg_id = local_id / sg_local_range;
  int sg_local_id = local_id % sg_local_range;
  int sg_local_n_coord = sg_id % ATOM_N;
  int n_sg_start = sg_local_n_coord * SG_N;
  int group_num = get<1>(A.shape()) / group_size;
  int x_idx = sg_local_id / channel_num;

  float scales0[thr_N * channel_num];
  float scales1[thr_N * channel_num];

  clear(tCrC);

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    int g0 = k_tile * 2;
    const uint16_t* Scales16 = reinterpret_cast<const uint16_t*>(Scales);
    int group_pairs = group_num >> 1;
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < thr_N; ++n) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < channel_num; ++c) {
        int real_idx = x_idx + c * (sg_local_range / channel_num);
        int sg_local_n = n * sg_local_range + real_idx;
        uint32_t pair = Scales16[(n_tile_start + n_sg_start + sg_local_n) * group_pairs + k_tile];
        scales0[n * channel_num + c] = fast_e4m3_to_f32(pair & 0xFFu);
        scales1[n * channel_num + c] = fast_e4m3_to_f32((pair >> 8) & 0xFFu);
      }
    }

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < thr_N; ++n) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < channel_num; ++c) {
        float s0 = scales0[n * channel_num + c];
        float s1 = scales1[n * channel_num + c];
        auto slice0 = tCrB(cute::tuple(c, _), n, _0{});
        auto slice1 = tCrB(cute::tuple(c, _), n, _1{});
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < slice0.size(); ++i) {
          slice0[i] = apply_scale(slice0[i], s0);
          slice1[i] = apply_scale(slice1[i], s1);
        }
      }
    }

    cute::gemm(mma, tCrA, tCrB, tCrC);
    barrier_wait(barrier_scope);
  }

  if (expert_scale != 1.0f) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tCrC.size(); ++i) {
      tCrC(i) *= expert_scale;
    }
  }

  reorder(tCrC, tCrC_out);
  copy(copy_c, tCrC_out, tCgC);
}

template <class Policy>
class NVFP4GemmName;

template <class Policy>
void launch_nvfp4(
    sycl::queue& stream,
    const bfloat16_t* activations,
    const uint8_t* weights,
    const cutlass::float_e4m3_t* scales,
    const float* expert_scales,
    bfloat16_t* outputs,
    int gemm_n,
    int gemm_k,
    const int* rows_per_expert,
    int num_experts,
    int32_t* atomic_buffer) {
  auto op = XE_DPAS_TT<8, float, bfloat16_t>{};
  using WGTile = typename Policy::WGTile;
  using SGLayout = typename Policy::SGLayout;
  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>, SGLayout>::TiledMMA;
  auto mma = MMA{};

  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  auto MaxThreadsPerWorkgroup = size(mma);
  static constexpr int MaxThreadsPerSM = 512;

  sycl::range<3> local(1, 1, MaxThreadsPerWorkgroup);
  sycl::range<3> global(1, sm_count * MaxThreadsPerSM / MaxThreadsPerWorkgroup, 1);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties kernel_props{
      syclex::sub_group_size<16>, intelex::grf_size<256>};

  stream.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> slm_mem_const(sycl::range<1>(1), cgh);
    cgh.parallel_for<NVFP4GemmName<Policy>>(
        sycl::nd_range<3>(global * local, local),
        kernel_props,
        [=](sycl::nd_item<3> item) {
          auto wg_tile = mma.tile_mnk();
          auto wg_tile_m = get<0>(wg_tile);
          auto wg_tile_n = get<1>(wg_tile);

          int group_id = item.get_group_linear_id();
          int gemm_n_pad = (gemm_n + wg_tile_n - 1) / wg_tile_n * wg_tile_n;
          int group_m_id = (group_id * wg_tile_n) / gemm_n_pad;
          int local_id = item.get_local_linear_id();

          int pre_rows = 0;
          int pre_tiles = 0;
          int32_t* slm_mem = static_cast<int32_t*>(
              slm_mem_const.template get_multi_ptr<sycl::access::decorated::no>().get());

          for (int i = 0; i < num_experts; ++i) {
            int gemm_m = rows_per_expert[i];
            int cumsum_rows = pre_rows + gemm_m;
            int cumsum_tiles = (gemm_m + wg_tile_m - 1) / wg_tile_m + pre_tiles;
            if (group_m_id >= cumsum_tiles) {
              pre_rows = cumsum_rows;
              pre_tiles = cumsum_tiles;
              continue;
            }
            int expert_id = i;
            int64_t B_offset = static_cast<int64_t>(expert_id) * gemm_n * gemm_k / 2;
            bfloat16_t* ptr_A = const_cast<bfloat16_t*>(activations) + static_cast<int64_t>(pre_rows) * gemm_k;
            uint8_t* ptr_B = const_cast<uint8_t*>(weights) + B_offset;
            bfloat16_t* ptr_D = outputs + static_cast<int64_t>(pre_rows) * gemm_n;
            const cutlass::float_e4m3_t* ptr_S = scales + (B_offset * 2) / 16;
            float escale = expert_scales ? expert_scales[expert_id] : 1.0f;

            auto A_tensor = make_tensor(make_gmem_ptr(ptr_A), make_layout(make_shape(gemm_m, gemm_k), make_stride(gemm_k, _1{})));
            auto B_tensor = make_tensor(make_gmem_ptr(reinterpret_cast<float_e2m1_t*>(ptr_B)), make_layout(make_shape(gemm_n, gemm_k), make_stride(gemm_k, _1{})));
            auto D_tensor = make_tensor(make_gmem_ptr(ptr_D), make_layout(make_shape(gemm_m, gemm_n), make_stride(gemm_n, _1{})));

            while (group_m_id < cumsum_tiles) {
              int n_coord = (group_id * wg_tile_n) % gemm_n_pad / wg_tile_n;
              int m_coord = group_m_id - pre_tiles;
              auto tile_coord = make_coord(m_coord, n_coord, _, 0);

              xe_gemm_nvfp4_gs16<
                  typename Policy::GmemTiledCopyA,
                  typename Policy::GmemTiledCopyB,
                  typename Policy::GmemTiledCopyD>(
                  A_tensor, B_tensor, ptr_S, escale, D_tensor, tile_coord, mma);

              if (local_id == 0) {
                slm_mem[0] = cutlass::atomicAdd(atomic_buffer, 1);
              }
              item.barrier(sycl::access::fence_space::local_space);
              group_id = slm_mem[0] + item.get_group_range(1);
              group_m_id = (group_id * wg_tile_n) / gemm_n_pad;
            }
            pre_rows = cumsum_rows;
            pre_tiles = cumsum_tiles;
          }
        });
  });
}

torch::Tensor nvfp4_grouped_gemm_xe2(
    torch::Tensor ptr_A,
    torch::Tensor ptr_B,
    torch::Tensor ptr_scales,
    c10::optional<torch::Tensor> expert_scales,
    torch::Tensor ptr_D,
    torch::Tensor rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    c10::optional<torch::Tensor> atomic_buf = c10::nullopt) {
  auto& dpcpp_queue = vllm::xpu::vllmGetQueue(ptr_A.device().index());
  at::Tensor atomic_buffer = atomic_buf.has_value()
      ? *atomic_buf
      : at::zeros({1}, ptr_A.options().dtype(at::kInt));

  int A_avg_M = ptr_A.size(0) / num_experts;
  const float* esc = expert_scales.has_value()
      ? reinterpret_cast<const float*>(expert_scales->data_ptr())
      : nullptr;

  if (A_avg_M <= 4) {
    launch_nvfp4<w4a16_policy_m_8>(
        dpcpp_queue,
        reinterpret_cast<const bfloat16_t*>(ptr_A.data_ptr()),
        reinterpret_cast<const uint8_t*>(ptr_B.data_ptr()),
        reinterpret_cast<const cutlass::float_e4m3_t*>(ptr_scales.data_ptr()),
        esc,
        reinterpret_cast<bfloat16_t*>(ptr_D.data_ptr()),
        N, K, reinterpret_cast<const int*>(rows_per_expert.data_ptr()),
        num_experts, static_cast<int32_t*>(atomic_buffer.data_ptr()));
  } else if (A_avg_M <= 8) {
    launch_nvfp4<w4a16_policy_m_16>(
        dpcpp_queue,
        reinterpret_cast<const bfloat16_t*>(ptr_A.data_ptr()),
        reinterpret_cast<const uint8_t*>(ptr_B.data_ptr()),
        reinterpret_cast<const cutlass::float_e4m3_t*>(ptr_scales.data_ptr()),
        esc,
        reinterpret_cast<bfloat16_t*>(ptr_D.data_ptr()),
        N, K, reinterpret_cast<const int*>(rows_per_expert.data_ptr()),
        num_experts, static_cast<int32_t*>(atomic_buffer.data_ptr()));
  } else if (A_avg_M <= 64) {
    launch_nvfp4<w4a16_policy_m_32>(
        dpcpp_queue,
        reinterpret_cast<const bfloat16_t*>(ptr_A.data_ptr()),
        reinterpret_cast<const uint8_t*>(ptr_B.data_ptr()),
        reinterpret_cast<const cutlass::float_e4m3_t*>(ptr_scales.data_ptr()),
        esc,
        reinterpret_cast<bfloat16_t*>(ptr_D.data_ptr()),
        N, K, reinterpret_cast<const int*>(rows_per_expert.data_ptr()),
        num_experts, static_cast<int32_t*>(atomic_buffer.data_ptr()));
  } else {
    launch_nvfp4<w4a16_policy>(
        dpcpp_queue,
        reinterpret_cast<const bfloat16_t*>(ptr_A.data_ptr()),
        reinterpret_cast<const uint8_t*>(ptr_B.data_ptr()),
        reinterpret_cast<const cutlass::float_e4m3_t*>(ptr_scales.data_ptr()),
        esc,
        reinterpret_cast<bfloat16_t*>(ptr_D.data_ptr()),
        N, K, reinterpret_cast<const int*>(rows_per_expert.data_ptr()),
        num_experts, static_cast<int32_t*>(atomic_buffer.data_ptr()));
  }
  return ptr_D;
}

class RouteSortSmallName;
class RouteHistLargeName;
class RoutePrefixLargeName;
class RouteScatterLargeName;
class SwiGLUInterleavedName;

// Zero-sync GPU counting sort for MoE routes (E <= 256).
// Outputs: counts [E], order [routes] (int32), inv_order [routes] (int32), atomic2 [2] (zeroed int32).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
moe_route_sort(
    torch::Tensor ids,
    c10::optional<torch::Tensor> is_padding,
    int64_t num_experts) {
  auto& q = vllm::xpu::vllmGetQueue(ids.device().index());
  int64_t tokens = ids.size(0);
  int64_t topk = ids.size(1);
  int64_t routes = tokens * topk;
  int E = static_cast<int>(num_experts);
  auto i32_opts = ids.options().dtype(at::kInt);

  torch::Tensor counts = torch::empty({E}, i32_opts);
  torch::Tensor order = torch::empty({routes}, i32_opts);
  torch::Tensor inv_order = torch::empty({routes}, i32_opts);
  torch::Tensor atomic2 = torch::empty({2}, i32_opts);

  const int32_t* ids_ptr = ids.data_ptr<int32_t>();
  const bool* pad_ptr = is_padding.has_value() ? is_padding->data_ptr<bool>() : nullptr;
  int32_t* cnt_ptr = counts.data_ptr<int32_t>();
  int32_t* ord_ptr = order.data_ptr<int32_t>();
  int32_t* inv_ptr = inv_order.data_ptr<int32_t>();
  int32_t* atm_ptr = atomic2.data_ptr<int32_t>();

  if (routes <= 256) {
    // Single-workgroup SLM counting sort: ~5 us total (replaces 6 PyTorch ops!)
    q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<int32_t, 1> sm_cnt(sycl::range<1>(257), cgh);
      sycl::local_accessor<int32_t, 1> sm_off(sycl::range<1>(257), cgh);
      cgh.parallel_for<RouteSortSmallName>(
          sycl::nd_range<1>(256, 256),
          [=](sycl::nd_item<1> it) {
            int tid = it.get_local_id(0);
            if (tid < 2) atm_ptr[tid] = 0;
            if (tid <= E) sm_cnt[tid] = 0;
            it.barrier(sycl::access::fence_space::local_space);

            int my_e = E;
            if (tid < routes) {
              int tok = tid / static_cast<int>(topk);
              int e = ids_ptr[tid];
              if (pad_ptr && pad_ptr[tok]) e = E;
              if (e < 0 || e >= E) e = E;
              my_e = e;
              sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                               sycl::memory_scope::work_group,
                               sycl::access::address_space::local_space>
                  a(sm_cnt[my_e]);
              a.fetch_add(1);
            }
            it.barrier(sycl::access::fence_space::local_space);

            if (tid == 0) {
              int acc = 0;
              for (int e = 0; e <= E; ++e) {
                int c = sm_cnt[e];
                sm_off[e] = acc;
                acc += c;
                if (e < E) cnt_ptr[e] = c;
              }
            }
            it.barrier(sycl::access::fence_space::local_space);

            if (tid < routes) {
              sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                               sycl::memory_scope::work_group,
                               sycl::access::address_space::local_space>
                  a(sm_off[my_e]);
              int pos = a.fetch_add(1);
              ord_ptr[pos] = tid;
              inv_ptr[tid] = pos;
            }
          });
    });
  } else {
    // Large prefill path (e.g. routes=16384): 3 tiny SYCL passes (hist -> prefix -> scatter)
    torch::Tensor offs = torch::zeros({E + 1}, i32_opts);
    int32_t* off_ptr = offs.data_ptr<int32_t>();
    int wg = 256;
    int ng = static_cast<int>((routes + wg - 1) / wg);
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<RouteHistLargeName>(
          sycl::nd_range<1>(ng * wg, wg),
          [=](sycl::nd_item<1> it) {
            int gid = it.get_global_id(0);
            if (gid < 2) atm_ptr[gid] = 0;
            if (gid < routes) {
              int tok = gid / static_cast<int>(topk);
              int e = ids_ptr[gid];
              if (pad_ptr && pad_ptr[tok]) e = E;
              if (e < 0 || e >= E) e = E;
              sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::global_space>
                  a(off_ptr[e]);
              a.fetch_add(1);
            }
          });
    });
    q.submit([&](sycl::handler& cgh) {
      cgh.single_task<RoutePrefixLargeName>([=]() {
        int acc = 0;
        for (int e = 0; e <= E; ++e) {
          int c = off_ptr[e];
          off_ptr[e] = acc;
          acc += c;
          if (e < E) cnt_ptr[e] = c;
        }
      });
    });
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<RouteScatterLargeName>(
          sycl::nd_range<1>(ng * wg, wg),
          [=](sycl::nd_item<1> it) {
            int gid = it.get_global_id(0);
            if (gid < routes) {
              int tok = gid / static_cast<int>(topk);
              int e = ids_ptr[gid];
              if (pad_ptr && pad_ptr[tok]) e = E;
              if (e < 0 || e >= E) e = E;
              sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::global_space>
                  a(off_ptr[e]);
              int pos = a.fetch_add(1);
              ord_ptr[pos] = gid;
              inv_ptr[gid] = pos;
            }
          });
    });
  }
  return std::make_tuple(counts, order, inv_order, atomic2);
}

torch::Tensor fused_swiglu_interleaved(
    torch::Tensor inp,
    double gate_cap_d,
    double up_cap_d) {
  auto& q = vllm::xpu::vllmGetQueue(inp.device().index());
  int64_t rows = inp.size(0);
  int64_t inter = inp.size(1) / 2;
  torch::Tensor out = torch::empty({rows, inter}, inp.options());
  const bfloat16_t* in_ptr = reinterpret_cast<const bfloat16_t*>(inp.data_ptr());
  bfloat16_t* out_ptr = reinterpret_cast<bfloat16_t*>(out.data_ptr());
  float gate_cap = static_cast<float>(gate_cap_d);
  float up_cap = static_cast<float>(up_cap_d);
  int64_t total = rows * inter;
  int wg = 256;
  int64_t ng = (total + wg - 1) / wg;

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<SwiGLUInterleavedName>(
        sycl::nd_range<1>(ng * wg, wg),
        [=](sycl::nd_item<1> it) {
          int64_t idx = it.get_global_id(0);
          if (idx < total) {
            float g = static_cast<float>(in_ptr[2 * idx]);
            float u = static_cast<float>(in_ptr[2 * idx + 1]);
            float sg = (gate_cap > 0.0f) ? (gate_cap * sycl::tanh(g / gate_cap)) : g;
            float su = (up_cap > 0.0f) ? (up_cap * sycl::tanh(u / up_cap)) : u;
            float sig = 1.0f / (1.0f + sycl::exp(-g));
            out_ptr[idx] = static_cast<bfloat16_t>((sg * sig) * su);
          }
        });
  });
  return out;
}

}  // namespace InklingNVFP4

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nvfp4_grouped_gemm_xe2", &InklingNVFP4::nvfp4_grouped_gemm_xe2,
        py::arg("ptr_A"), py::arg("ptr_B"), py::arg("ptr_scales"),
        py::arg("expert_scales"), py::arg("ptr_D"), py::arg("rows_per_expert"),
        py::arg("N"), py::arg("K"), py::arg("num_experts"),
        py::arg("atomic_buf") = c10::nullopt,
        "Native NVFP4 (e2m1 + per-16 e4m3 + expert_scale) CUTLASS Xe2 grouped GEMM");
  m.def("moe_route_sort", &InklingNVFP4::moe_route_sort,
        "Zero-sync GPU counting sort for MoE routes (returns counts, order, inv_order, atomic2)");
  m.def("fused_swiglu_interleaved", &InklingNVFP4::fused_swiglu_interleaved,
        "Fused SYCL softcapped SwiGLU for interleaved gate/up pairs");
}
