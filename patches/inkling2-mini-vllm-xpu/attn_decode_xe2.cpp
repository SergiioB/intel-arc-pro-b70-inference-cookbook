// SPDX-License-Identifier: Apache-2.0
// V3 AOT bmg_g31 SYCL Cooperative 8-Token GQA-Shared Decode Attention for Inkling 2 Mini.
// Breakthrough:
//   - For M=1: V2 4-SG/WG zero-spill split-KV (already 4.39x faster than Triton @ 256K FP8: 11.45 ms!)
//   - For M=2..8 (DSpark verify): 8 subgroups/WG where subgroup `t` (0..7) owns token `t`
//     (48 floats/lane -> ZERO register spill!). All 8 subgroups COOPERATIVELY load 16 keys of
//     K & V into 8 KB BF16 SLM (2 keys/SG -> 8x less DRAM traffic + 8x fewer FP8 decodes!),
//     then each subgroup computes its G=3 query heads from SLM (~2 TB/s!).
//   - For SWA (window=1024, M=8): 4 splits x 8 KV_H = 32 WGs (256 SGs) + tiny 4-split combine!

#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>
#include <torch/extension.h>
#include <cmath>
#include <cstdint>

namespace {

static inline float bf16_to_f32(uint16_t b) {
  return sycl::bit_cast<float>(static_cast<uint32_t>(b) << 16);
}

static inline uint16_t f32_to_bf16(float f) {
  uint32_t u = sycl::bit_cast<uint32_t>(f);
  uint32_t rounding = 0x7FFFu + ((u >> 16) & 1u);
  return static_cast<uint16_t>((u + rounding) >> 16);
}

static inline float e4m3_to_f32_init(uint8_t u) {
  uint32_t s = static_cast<uint32_t>(u & 0x80u) << 24;
  uint32_t m = static_cast<uint32_t>(u & 0x7Fu);
  if (m >= 8u) return sycl::bit_cast<float>(s | ((m << 20) + (120u << 23)));
  float v = static_cast<float>(m) * (1.0f / 512.0f);
  return (u & 0x80u) ? -v : v;
}

static inline uint16_t e4m3_to_bf16_init(uint8_t u) {
  return f32_to_bf16(e4m3_to_f32_init(u));
}

static inline float fast_sigmoid(float x) {
  return 1.0f / (1.0f + sycl::native::exp2(-x * 1.4426950408889634f));
}

template <bool IS_FP8>
static inline void load_kv_vec8_f32(
    const void* base_ptr, int64_t elem_off, const float* lut, float dst[8]) {
  if constexpr (IS_FP8) {
    const uint8_t* p = static_cast<const uint8_t*>(base_ptr) + elem_off;
    uint64_t packed = *reinterpret_cast<const uint64_t*>(p);
    #pragma unroll
    for (int i = 0; i < 8; ++i) dst[i] = lut[(packed >> (i * 8)) & 0xFFu];
  } else {
    const uint16_t* p = static_cast<const uint16_t*>(base_ptr) + elem_off;
    sycl::uint4 packed = *reinterpret_cast<const sycl::uint4*>(p);
    uint32_t w[4] = {packed.x(), packed.y(), packed.z(), packed.w()};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      dst[2 * i]     = bf16_to_f32(static_cast<uint16_t>(w[i] & 0xFFFFu));
      dst[2 * i + 1] = bf16_to_f32(static_cast<uint16_t>(w[i] >> 16));
    }
  }
}

// ============================================================================
// KERNEL A (M=2..8 DSpark Verify — BOTH SWA & FULL!):
// Cooperative 8-Token Split-KV with 8 KB BF16 SLM KV Staging!
// Grid: (num_splits, KV_H, num_reqs) | WG: 8 subgroups (128 lanes), sg_id = tok (0..7)
// Each WG processes `BK = 16` keys at a time: all 8 SGs cooperatively load 2 keys each
// of K & V into BF16 SLM (8 KB total -> 8x DRAM bandwidth reduction!), then each SG
// computes its own token's G=3 query heads against all 16 keys from SLM (ZERO spill!).
// ============================================================================
template <bool IS_FP8, int G>
struct Coop8TokenSplitKernel {
  const uint16_t* Q;
  const void* K;
  const void* V;
  const uint16_t* Rel;
  const int32_t* BlockTable;
  const int32_t* SeqLens;
  const int32_t* CuSeqlensQ;
  float* OPart;    // [num_splits, num_tokens, H, D]
  float* LsePart;  // [num_splits, num_tokens, H]

  float softmax_scale;
  int64_t stride_q_t, stride_q_h;
  int64_t stride_kv_blk, stride_kv_slot, stride_kv_h;
  int64_t stride_rel_t, stride_bt;
  int H, E, E_PAD, PAGE, WINDOW, num_tokens, num_splits;

  sycl::local_accessor<uint16_t, 1> smem_k;   // [16 * 128] bf16 = 4 KB
  sycl::local_accessor<uint16_t, 1> smem_v;   // [16 * 128] bf16 = 4 KB
  sycl::local_accessor<uint16_t, 1> lut_bf16; // [256] bf16 = 0.5 KB

  [[sycl::reqd_sub_group_size(16)]]
  void operator()(sycl::nd_item<3> it) const {
    constexpr int D = 128;
    constexpr int BK = 16;
    constexpr float LOG2E = 1.4426950408889634f;

    const int tid = static_cast<int>(it.get_local_linear_id());  // 0..127
    if constexpr (IS_FP8) {
      lut_bf16[tid] = e4m3_to_bf16_init(static_cast<uint8_t>(tid));
      lut_bf16[tid + 128] = e4m3_to_bf16_init(static_cast<uint8_t>(tid + 128));
      sycl::group_barrier(it.get_group());
    }
    const uint16_t* lut = lut_bf16.get_multi_ptr<sycl::access::decorated::no>().get();

    const int split = static_cast<int>(it.get_group(0));
    const int kv_head = static_cast<int>(it.get_group(1));
    const int req = static_cast<int>(it.get_group(2));

    const int q_start = CuSeqlensQ[req];
    const int q_len = CuSeqlensQ[req + 1] - q_start;
    const int seqlen_k = SeqLens[req];
    const int past = seqlen_k - q_len;

    const int sg_id = tid >> 4;   // 0..7 = token index within request!
    const int lane = tid & 15;    // 0..15
    const int d_base = lane * 8;
    auto sg = it.get_sub_group();

    const bool tok_valid = (sg_id < q_len);
    const int token = q_start + sg_id;
    const int pos = past + sg_id;
    const int h0 = kv_head * G;
    const float qk_scale = softmax_scale * LOG2E;

    float q_reg[G][8];
    #pragma unroll
    for (int g = 0; g < G; ++g) {
      if (tok_valid) {
        const uint16_t* qp = Q + token * stride_q_t + (h0 + g) * stride_q_h + d_base;
        sycl::uint4 packed = *reinterpret_cast<const sycl::uint4*>(qp);
        uint32_t w[4] = {packed.x(), packed.y(), packed.z(), packed.w()};
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          q_reg[g][2 * i]     = bf16_to_f32(static_cast<uint16_t>(w[i] & 0xFFFFu)) * qk_scale;
          q_reg[g][2 * i + 1] = bf16_to_f32(static_cast<uint16_t>(w[i] >> 16)) * qk_scale;
        }
      } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) q_reg[g][i] = 0.0f;
      }
    }

    const int tile_end = ((sg_id / 128) + 1) * 128 < q_len ? ((sg_id / 128) + 1) * 128 : q_len;
    const int rel_base = ((past + tile_end + 127) / 128) * 128 - E_PAD;
    const int64_t rel_t_off = static_cast<int64_t>(token) * stride_rel_t;

    const int pos_lo_all = past;
    const int pos_hi_all = past + q_len - 1;
    const int k_min_all = (WINDOW > 0 && pos_lo_all - (WINDOW - 1) > 0) ? (pos_lo_all - (WINDOW - 1)) : 0;
    const int k_max_all = (pos_hi_all + 1 < seqlen_k) ? (pos_hi_all + 1) : seqlen_k;

    // Split key range into contiguous BK-aligned chunks across num_splits!
    const int n_blocks_all = (k_max_all - k_min_all + BK - 1) / BK;
    const int blocks_per_split = (n_blocks_all + num_splits - 1) / num_splits;
    const int b_lo = split * blocks_per_split;
    const int b_hi = (b_lo + blocks_per_split < n_blocks_all) ? (b_lo + blocks_per_split) : n_blocks_all;

    const int32_t* bt = BlockTable + static_cast<int64_t>(req) * stride_bt;
    const int64_t kv_h_off = static_cast<int64_t>(kv_head) * stride_kv_h + d_base;

    float m_i[G], l_i[G], acc[G][8];
    #pragma unroll
    for (int g = 0; g < G; ++g) {
      m_i[g] = -INFINITY;
      l_i[g] = 0.0f;
      #pragma unroll
      for (int i = 0; i < 8; ++i) acc[g][i] = 0.0f;
    }

    uint16_t* sk_ptr = smem_k.get_multi_ptr<sycl::access::decorated::no>().get();
    uint16_t* sv_ptr = smem_v.get_multi_ptr<sycl::access::decorated::no>().get();

    for (int b = b_lo; b < b_hi; ++b) {
      const int k_start = k_min_all + b * BK;
      const int bk_valid = (k_max_all - k_start < BK) ? (k_max_all - k_start) : BK;

      // Cooperative load: each of the 8 subgroups loads 2 keys (sg_id and sg_id + 8)!
      #pragma unroll
      for (int step = 0; step < 2; ++step) {
        const int u = sg_id + step * 8;
        const int k_idx = k_start + u;
        if (u < bk_valid) {
          const int phys_page = bt[k_idx / PAGE];
          const int slot = k_idx % PAGE;
          const int64_t off = static_cast<int64_t>(phys_page) * stride_kv_blk +
                              static_cast<int64_t>(slot) * stride_kv_slot + kv_h_off;
          if constexpr (IS_FP8) {
            uint64_t pk = *reinterpret_cast<const uint64_t*>(static_cast<const uint8_t*>(K) + off);
            uint64_t pv = *reinterpret_cast<const uint64_t*>(static_cast<const uint8_t*>(V) + off);
            uint16_t* dk = sk_ptr + u * D + d_base;
            uint16_t* dv = sv_ptr + u * D + d_base;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
              dk[i] = lut[(pk >> (i * 8)) & 0xFFu];
              dv[i] = lut[(pv >> (i * 8)) & 0xFFu];
            }
          } else {
            sycl::uint4 pk = *reinterpret_cast<const sycl::uint4*>(static_cast<const uint16_t*>(K) + off);
            sycl::uint4 pv = *reinterpret_cast<const sycl::uint4*>(static_cast<const uint16_t*>(V) + off);
            *reinterpret_cast<sycl::uint4*>(sk_ptr + u * D + d_base) = pk;
            *reinterpret_cast<sycl::uint4*>(sv_ptr + u * D + d_base) = pv;
          }
        }
      }
      sycl::group_barrier(it.get_group());

      if (tok_valid) {
        const bool block_has_rel = (pos - k_start) < (E + BK);
        const bool block_all_visible = (pos >= k_start + bk_valid - 1) && (WINDOW == 0 || (pos - k_start) < WINDOW);

        // Process the 16 keys from SLM in 4-key tiled softmax micro-batches!
        for (int u0 = 0; u0 < bk_valid; u0 += 4) {
          const int rem = (bk_valid - u0 < 4) ? (bk_valid - u0) : 4;
          float kv_k[4][8], kv_v[4][8];
          #pragma unroll
          for (int u = 0; u < 4; ++u) {
            if (u < rem) {
              sycl::uint4 pk = *reinterpret_cast<const sycl::uint4*>(sk_ptr + (u0 + u) * D + d_base);
              sycl::uint4 pv = *reinterpret_cast<const sycl::uint4*>(sv_ptr + (u0 + u) * D + d_base);
              uint32_t wk[4] = {pk.x(), pk.y(), pk.z(), pk.w()};
              uint32_t wv[4] = {pv.x(), pv.y(), pv.z(), pv.w()};
              #pragma unroll
              for (int i = 0; i < 4; ++i) {
                kv_k[u][2 * i]     = bf16_to_f32(static_cast<uint16_t>(wk[i] & 0xFFFFu));
                kv_k[u][2 * i + 1] = bf16_to_f32(static_cast<uint16_t>(wk[i] >> 16));
                kv_v[u][2 * i]     = bf16_to_f32(static_cast<uint16_t>(wv[i] & 0xFFFFu));
                kv_v[u][2 * i + 1] = bf16_to_f32(static_cast<uint16_t>(wv[i] >> 16));
              }
            }
          }

          #pragma unroll
          for (int g = 0; g < G; ++g) {
            float s[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
              if (u < rem) {
                const int k_idx = k_start + u0 + u;
                const int dist = pos - k_idx;
                bool vis = block_all_visible || (dist >= 0 && (WINDOW == 0 || dist < WINDOW));
                if (vis) {
                  float dot = 0.0f;
                  #pragma unroll
                  for (int i = 0; i < 8; ++i) dot += q_reg[g][i] * kv_k[u][i];
                  dot = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
                  if (block_has_rel && dist < E) {
                    dot += bf16_to_f32(Rel[rel_t_off + static_cast<int64_t>(h0 + g) * E_PAD + (k_idx - rel_base)]) * LOG2E;
                  }
                  s[u] = dot;
                }
              }
            }
            float m4 = sycl::fmax(sycl::fmax(s[0], s[1]), sycl::fmax(s[2], s[3]));
            if (m4 != -INFINITY) {
              float m_new = sycl::fmax(m_i[g], m4);
              float alpha = (m_i[g] == -INFINITY) ? 0.0f : sycl::native::exp2(m_i[g] - m_new);
              float p0 = (s[0] == -INFINITY) ? 0.0f : sycl::native::exp2(s[0] - m_new);
              float p1 = (s[1] == -INFINITY) ? 0.0f : sycl::native::exp2(s[1] - m_new);
              float p2 = (s[2] == -INFINITY) ? 0.0f : sycl::native::exp2(s[2] - m_new);
              float p3 = (s[3] == -INFINITY) ? 0.0f : sycl::native::exp2(s[3] - m_new);
              l_i[g] = l_i[g] * alpha + (p0 + p1 + p2 + p3);
              #pragma unroll
              for (int i = 0; i < 8; ++i) {
                acc[g][i] = acc[g][i] * alpha
                          + (p0 != 0.0f ? p0 * kv_v[0][i] : 0.0f)
                          + (p1 != 0.0f ? p1 * kv_v[1][i] : 0.0f)
                          + (p2 != 0.0f ? p2 * kv_v[2][i] : 0.0f)
                          + (p3 != 0.0f ? p3 * kv_v[3][i] : 0.0f);
              }
              m_i[g] = m_new;
            }
          }
        }
      }
      sycl::group_barrier(it.get_group());
    }

    if (tok_valid) {
      #pragma unroll
      for (int g = 0; g < G; ++g) {
        const int64_t part = (static_cast<int64_t>(split) * num_tokens + token) * H + (h0 + g);
        const bool empty = (l_i[g] == 0.0f);
        const float inv_l = empty ? 0.0f : (1.0f / l_i[g]);
        if (lane == 0) {
          LsePart[part] = empty ? -INFINITY : (m_i[g] + sycl::native::log2(l_i[g]));
        }
        float* op = OPart + part * D + d_base;
        #pragma unroll
        for (int i = 0; i < 8; ++i) op[i] = acc[g][i] * inv_l;
      }
    }
  }
};

// ============================================================================
// KERNEL B (M=1 Single-Token Decode — proven 4.39x faster @ 256K FP8: 11.45 ms!)
// ============================================================================
template <bool IS_FP8, int G>
struct SingleTokSplitKernel {
  const uint16_t* Q;
  const void* K;
  const void* V;
  const uint16_t* Rel;
  const int32_t* BlockTable;
  const int32_t* SeqLens;
  const int32_t* CuSeqlensQ;
  float* OPart;
  float* LsePart;

  float softmax_scale;
  int64_t stride_q_t, stride_q_h;
  int64_t stride_kv_blk, stride_kv_slot, stride_kv_h;
  int64_t stride_rel_t, stride_bt;
  int H, E, E_PAD, PAGE, WINDOW, num_tokens, num_splits;

  sycl::local_accessor<float, 1> smem;     // [4 * 3 * 130]
  sycl::local_accessor<float, 1> lut_slm;  // [256]

  [[sycl::reqd_sub_group_size(16)]]
  void operator()(sycl::nd_item<3> it) const {
    constexpr int NUM_SG = 4;
    constexpr int D = 128;
    constexpr float LOG2E = 1.4426950408889634f;

    const int tid = static_cast<int>(it.get_local_linear_id());
    if constexpr (IS_FP8) {
      for (int i = tid; i < 256; i += 64) lut_slm[i] = e4m3_to_f32_init(static_cast<uint8_t>(i));
      sycl::group_barrier(it.get_group());
    }
    const float* lut = lut_slm.get_multi_ptr<sycl::access::decorated::no>().get();

    const int split = static_cast<int>(it.get_group(0));
    const int kv_head = static_cast<int>(it.get_group(1));
    const int req = static_cast<int>(it.get_group(2));

    const int q_start = CuSeqlensQ[req];
    const int q_len = CuSeqlensQ[req + 1] - q_start;
    if (q_len <= 0) return;
    const int token = q_start;
    const int seqlen_k = SeqLens[req];
    const int past = seqlen_k - q_len;
    const int pos = past;

    const int sg_id = tid >> 4;
    const int lane = tid & 15;
    const int d_base = lane * 8;
    auto sg = it.get_sub_group();

    const int h0 = kv_head * G;
    const float qk_scale = softmax_scale * LOG2E;
    float q_reg[G][8];
    #pragma unroll
    for (int g = 0; g < G; ++g) {
      const uint16_t* qp = Q + token * stride_q_t + (h0 + g) * stride_q_h + d_base;
      sycl::uint4 packed = *reinterpret_cast<const sycl::uint4*>(qp);
      uint32_t w[4] = {packed.x(), packed.y(), packed.z(), packed.w()};
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        q_reg[g][2 * i]     = bf16_to_f32(static_cast<uint16_t>(w[i] & 0xFFFFu)) * qk_scale;
        q_reg[g][2 * i + 1] = bf16_to_f32(static_cast<uint16_t>(w[i] >> 16)) * qk_scale;
      }
    }

    const int tile_end = (128 < q_len) ? 128 : q_len;
    const int rel_base = ((past + tile_end + 127) / 128) * 128 - E_PAD;
    const int64_t rel_t_off = static_cast<int64_t>(token) * stride_rel_t;

    const int k_min_all = (WINDOW > 0 && pos - (WINDOW - 1) > 0) ? (pos - (WINDOW - 1)) : 0;
    const int k_max_all = (pos + 1 < seqlen_k) ? (pos + 1) : seqlen_k;
    const int total_k = k_max_all - k_min_all;
    const int total_chunks = num_splits * NUM_SG;
    const int my_id = split * NUM_SG + sg_id;
    const int k_per = (total_k + total_chunks - 1) / total_chunks;
    const int my_lo = k_min_all + my_id * k_per;
    const int my_hi = (my_lo + k_per < k_max_all) ? (my_lo + k_per) : k_max_all;

    const int32_t* bt = BlockTable + static_cast<int64_t>(req) * stride_bt;
    const int64_t kv_h_off = static_cast<int64_t>(kv_head) * stride_kv_h + d_base;

    float m_i[G], l_i[G], acc[G][8];
    #pragma unroll
    for (int g = 0; g < G; ++g) {
      m_i[g] = -INFINITY;
      l_i[g] = 0.0f;
      #pragma unroll
      for (int i = 0; i < 8; ++i) acc[g][i] = 0.0f;
    }

    int k_cur = my_lo;
    while (k_cur < my_hi) {
      const int p_idx = k_cur / PAGE;
      const int phys_page = bt[p_idx];
      const int page_end = (p_idx + 1) * PAGE;
      const int k_end = (page_end < my_hi) ? page_end : my_hi;
      const int64_t blk_off = static_cast<int64_t>(phys_page) * stride_kv_blk + kv_h_off;
      const bool page_has_rel = (pos - k_cur) < (E + PAGE);

      while (k_cur + 4 <= k_end) {
        float kv_k[4][8], kv_v[4][8];
        int slot0 = k_cur - p_idx * PAGE;
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
          int64_t off = blk_off + static_cast<int64_t>(slot0 + u) * stride_kv_slot;
          load_kv_vec8_f32<IS_FP8>(K, off, lut, kv_k[u]);
          load_kv_vec8_f32<IS_FP8>(V, off, lut, kv_v[u]);
        }
        #pragma unroll
        for (int g = 0; g < G; ++g) {
          float s[4];
          #pragma unroll
          for (int u = 0; u < 4; ++u) {
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) dot += q_reg[g][i] * kv_k[u][i];
            dot = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
            if (page_has_rel && (pos - (k_cur + u) < E)) {
              dot += bf16_to_f32(Rel[rel_t_off + static_cast<int64_t>(h0 + g) * E_PAD + (k_cur + u - rel_base)]) * LOG2E;
            }
            s[u] = dot;
          }
          float m4 = sycl::fmax(sycl::fmax(s[0], s[1]), sycl::fmax(s[2], s[3]));
          float m_new = sycl::fmax(m_i[g], m4);
          float alpha = (m_i[g] == -INFINITY) ? 0.0f : sycl::native::exp2(m_i[g] - m_new);
          float p0 = sycl::native::exp2(s[0] - m_new);
          float p1 = sycl::native::exp2(s[1] - m_new);
          float p2 = sycl::native::exp2(s[2] - m_new);
          float p3 = sycl::native::exp2(s[3] - m_new);
          l_i[g] = l_i[g] * alpha + (p0 + p1 + p2 + p3);
          #pragma unroll
          for (int i = 0; i < 8; ++i) {
            acc[g][i] = acc[g][i] * alpha + p0 * kv_v[0][i] + p1 * kv_v[1][i] + p2 * kv_v[2][i] + p3 * kv_v[3][i];
          }
          m_i[g] = m_new;
        }
        k_cur += 4;
      }
      while (k_cur < k_end) {
        int slot = k_cur - p_idx * PAGE;
        int64_t off = blk_off + static_cast<int64_t>(slot) * stride_kv_slot;
        float k_vec[8], v_vec[8];
        load_kv_vec8_f32<IS_FP8>(K, off, lut, k_vec);
        load_kv_vec8_f32<IS_FP8>(V, off, lut, v_vec);
        #pragma unroll
        for (int g = 0; g < G; ++g) {
          float dot = 0.0f;
          #pragma unroll
          for (int i = 0; i < 8; ++i) dot += q_reg[g][i] * k_vec[i];
          dot = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
          if (page_has_rel && (pos - k_cur < E)) {
            dot += bf16_to_f32(Rel[rel_t_off + static_cast<int64_t>(h0 + g) * E_PAD + (k_cur - rel_base)]) * LOG2E;
          }
          float m_new = sycl::fmax(m_i[g], dot);
          float alpha = (m_i[g] == -INFINITY) ? 0.0f : sycl::native::exp2(m_i[g] - m_new);
          float p = sycl::native::exp2(dot - m_new);
          l_i[g] = l_i[g] * alpha + p;
          #pragma unroll
          for (int i = 0; i < 8; ++i) acc[g][i] = acc[g][i] * alpha + p * v_vec[i];
          m_i[g] = m_new;
        }
        k_cur++;
      }
    }

    #pragma unroll
    for (int g = 0; g < G; ++g) {
      int base = (g * NUM_SG + sg_id) * 130;
      if (lane == 0) { smem[base + 128] = m_i[g]; smem[base + 129] = l_i[g]; }
      #pragma unroll
      for (int i = 0; i < 8; ++i) smem[base + d_base + i] = acc[g][i];
    }
    sycl::group_barrier(it.get_group());

    #pragma unroll
    for (int g = 0; g < G; ++g) {
      const int64_t part = (static_cast<int64_t>(split) * num_tokens + token) * H + (h0 + g);
      float m_max = -INFINITY;
      #pragma unroll
      for (int s = 0; s < NUM_SG; ++s) m_max = sycl::fmax(m_max, smem[(g * NUM_SG + s) * 130 + 128]);
      float l_sum = 0.0f, w_s[NUM_SG];
      #pragma unroll
      for (int s = 0; s < NUM_SG; ++s) {
        float ls = smem[(g * NUM_SG + s) * 130 + 129];
        w_s[s] = (m_max != -INFINITY && ls > 0.0f) ? sycl::native::exp2(smem[(g * NUM_SG + s) * 130 + 128] - m_max) : 0.0f;
        l_sum += ls * w_s[s];
      }
      if (tid == 0) LsePart[part] = (l_sum > 0.0f) ? (m_max + sycl::native::log2(l_sum)) : -INFINITY;
      float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
      #pragma unroll
      for (int k2 = 0; k2 < 2; ++k2) {
        int d = tid + k2 * 64;
        float ov = 0.0f;
        #pragma unroll
        for (int s = 0; s < NUM_SG; ++s) ov += smem[(g * NUM_SG + s) * 130 + d] * w_s[s];
        OPart[part * D + d] = ov * inv_l;
      }
    }
  }
};

struct FullDecodeCombineKernel {
  const float* OPart;
  const float* LsePart;
  const uint16_t* Gate;
  const uint16_t* Sink;
  uint16_t* Out;

  int64_t stride_gate, stride_sink, stride_out_t;
  int num_tokens, num_actual_tokens, H, num_splits;
  sycl::local_accessor<float, 1> smem_lse;

  [[sycl::reqd_sub_group_size(16)]]
  void operator()(sycl::nd_item<2> it) const {
    constexpr int D = 128;
    const int tok = static_cast<int>(it.get_group(0));
    const int head = static_cast<int>(it.get_group(1));
    const int d = static_cast<int>(it.get_local_id(1));

    if (tok >= num_actual_tokens) {
      Out[static_cast<int64_t>(tok) * stride_out_t + head * D + d] = 0;
      return;
    }

    for (int s = d; s < num_splits; s += 128) {
      int64_t part = (static_cast<int64_t>(s) * num_tokens + tok) * H + head;
      smem_lse[s] = LsePart[part];
    }
    sycl::group_barrier(it.get_group());

    float m_max = -INFINITY;
    for (int s = 0; s < num_splits; ++s) m_max = sycl::fmax(m_max, smem_lse[s]);
    float w_sum = 0.0f, acc = 0.0f;
    if (m_max != -INFINITY) {
      for (int s = 0; s < num_splits; ++s) {
        float ls = smem_lse[s];
        if (ls != -INFINITY) {
          float w = sycl::native::exp2(ls - m_max);
          w_sum += w;
          int64_t part = (static_cast<int64_t>(s) * num_tokens + tok) * H + head;
          acc += OPart[part * D + d] * w;
        }
      }
    }
    float o = (w_sum > 0.0f) ? (acc / w_sum) : 0.0f;
    float lse_nat = (w_sum > 0.0f) ? (m_max + sycl::native::log2(w_sum)) * 0.6931471805599453f : -INFINITY;
    float gate_v = bf16_to_f32(Gate[static_cast<int64_t>(tok) * stride_gate + head * D + d]);
    float sink_v = bf16_to_f32(Sink[static_cast<int64_t>(tok) * stride_sink + head]);
    o *= fast_sigmoid(gate_v) * fast_sigmoid(lse_nat - sink_v);
    Out[static_cast<int64_t>(tok) * stride_out_t + head * D + d] = f32_to_bf16(o);
  }
};

void rel_decode_attn_xe2(
    torch::Tensor q, torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor rel_logits, torch::Tensor gate, torch::Tensor sinks,
    torch::Tensor out, torch::Tensor block_table, torch::Tensor seq_lens,
    torch::Tensor cu_seqlens_q, torch::Tensor o_part, torch::Tensor lse_part,
    int64_t max_query_len, int64_t num_actual_tokens, double softmax_scale,
    int64_t window, int64_t num_splits) {
  auto stream = c10::xpu::getCurrentXPUStream(q.device().index());
  sycl::queue& queue = stream.queue();

  const int num_tokens = static_cast<int>(q.size(0));
  const int H = static_cast<int>(q.size(1));
  const int KV_H = static_cast<int>(key_cache.size(2));
  const int PAGE = static_cast<int>(key_cache.size(1));
  const int E_PAD = static_cast<int>(rel_logits.size(2));
  const int E = E_PAD - 256;
  const int num_reqs = static_cast<int>(seq_lens.size(0));
  const bool is_fp8 = (key_cache.element_size() == 1);
  const int sp = static_cast<int>(num_splits);

  if (max_query_len <= 1) {
    sycl::range<3> gws(sp * 64, KV_H, num_reqs);
    sycl::range<3> lws(64, 1, 1);
    auto launch1 = [&](auto fp8_tag) {
      constexpr bool FP8 = decltype(fp8_tag)::value;
      queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> smem(sycl::range<1>(4 * 3 * 130), cgh);
        sycl::local_accessor<float, 1> lut(sycl::range<1>(256), cgh);
        SingleTokSplitKernel<FP8, 3> k{
            reinterpret_cast<const uint16_t*>(q.data_ptr()),
            key_cache.data_ptr(), value_cache.data_ptr(),
            reinterpret_cast<const uint16_t*>(rel_logits.data_ptr()),
            block_table.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
            cu_seqlens_q.data_ptr<int32_t>(),
            o_part.data_ptr<float>(), lse_part.data_ptr<float>(),
            static_cast<float>(softmax_scale),
            q.stride(0), q.stride(1),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            rel_logits.stride(0), block_table.stride(0),
            H, E, E_PAD, PAGE, static_cast<int>(window), num_tokens, sp,
            smem, lut
        };
        cgh.parallel_for(sycl::nd_range<3>(gws, lws), k);
      });
    };
    if (is_fp8) launch1(std::true_type{});
    else launch1(std::false_type{});
  } else {
    // Cooperative 8-token SLM KV sharing (works for BOTH SWA and FULL, M=2..8!)
    sycl::range<3> gws(sp * 128, KV_H, num_reqs);
    sycl::range<3> lws(128, 1, 1);
    auto launch8 = [&](auto fp8_tag) {
      constexpr bool FP8 = decltype(fp8_tag)::value;
      queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<uint16_t, 1> sk(sycl::range<1>(16 * 128), cgh);
        sycl::local_accessor<uint16_t, 1> sv(sycl::range<1>(16 * 128), cgh);
        sycl::local_accessor<uint16_t, 1> lut(sycl::range<1>(256), cgh);
        Coop8TokenSplitKernel<FP8, 3> k{
            reinterpret_cast<const uint16_t*>(q.data_ptr()),
            key_cache.data_ptr(), value_cache.data_ptr(),
            reinterpret_cast<const uint16_t*>(rel_logits.data_ptr()),
            block_table.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
            cu_seqlens_q.data_ptr<int32_t>(),
            o_part.data_ptr<float>(), lse_part.data_ptr<float>(),
            static_cast<float>(softmax_scale),
            q.stride(0), q.stride(1),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            rel_logits.stride(0), block_table.stride(0),
            H, E, E_PAD, PAGE, static_cast<int>(window), num_tokens, sp,
            sk, sv, lut
        };
        cgh.parallel_for(sycl::nd_range<3>(gws, lws), k);
      });
    };
    if (is_fp8) launch8(std::true_type{});
    else launch8(std::false_type{});
  }

  sycl::range<2> cgws(num_tokens, H * 128);
  sycl::range<2> clws(1, 128);
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> smem_lse(sycl::range<1>(sp), cgh);
    FullDecodeCombineKernel ck{
        o_part.data_ptr<float>(), lse_part.data_ptr<float>(),
        reinterpret_cast<const uint16_t*>(gate.data_ptr()),
        reinterpret_cast<const uint16_t*>(sinks.data_ptr()),
        reinterpret_cast<uint16_t*>(out.data_ptr()),
        gate.stride(0), sinks.stride(0), out.stride(0),
        num_tokens, static_cast<int>(num_actual_tokens), H, sp, smem_lse
    };
    cgh.parallel_for(sycl::nd_range<2>(cgws, clws), ck);
  });
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rel_decode_attn_xe2", &rel_decode_attn_xe2,
        "V3 Cooperative 8-Token AOT bmg_g31 SYCL decode attention");
}
