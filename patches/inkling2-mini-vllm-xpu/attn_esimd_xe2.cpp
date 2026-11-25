// SPDX-License-Identifier: Apache-2.0
// Opt-in Intel Xe2 ESIMD XMX Paged Decode Attention for vllm-inkling2-xpu
// (_inkling_esimd_attn_xe2).
//
// Two-pass ESIMD XMX paged decode attention tailored to the Inkling 2
// rel_logits / gate / sink contract (both BF16 and FP8 E4M3 KV caches):
//   - One ESIMD work-item per (split, kv_head, 8-slot row tile); workgroup = 1,
//     no subgroups, no SLM, no barriers, all state in GRFs
//     (-ze-opt-large-register-file).
//   - Hardware lsc_load_2d for K (transposed) and V (VNNI) straight out of the
//     HND paged KV cache: physical tensor [NB, KV_H=8, PAGE=64, 2*D=256] where
//     key_cache = transpose(1,2)[...,:128] / value_cache = [...,128:]. The 2D
//     surface row width is D elems, pitch = stride_kv_slot elems (256 = 2*D).
//   - FP8 E4M3 via the 3-op SIMD dequant ((u&127)<<7 | (u&128)<<8) * 256, exact
//     for normals AND subnormals; BF16 via u16 -> (u32<<16) -> f32 -> fp16,
//     which is exact because bf16's 8-bit mantissa fits inside fp16.
//   - xmx::dpas<8,8> fp16 for both QK^T (8 slots x 16 keys) and P.V
//     (8 slots x 16 dims per c-chunk); single fp16 datapath for both KV dtypes.
//   - Our sheared bf16 rel_logits bias (per-row rel_base from tok's shear
//     tile), causal + optional sliding-window mask, per-(tok,head) gate/sink
//     applied by the ESIMD combine (NOT folded into softmax; Inkling's sink is
//     a sigmoid gate on the LSE, not a softmax sink).
//   - LsePart is log2-domain (m + log2 den) to match _rel_decode_combine_kernel
//     and the SIMT extension; the combine converts to natural log for
//     sigmoid(lse - sink).
//
// Exposed: rel_decode_esimd_xe2 — same signature as
// attn_decode_xe2.cpp's rel_decode_attn_xe2. Phase-A scope: single request
// (num_reqs == 1), q_len <= 8 (decode + DSpark verify).

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <c10/xpu/XPUStream.h>
#include <torch/extension.h>
#include <cmath>
#include <cstdint>

namespace {

namespace xesimd = sycl::ext::intel::experimental::esimd;
using namespace sycl::ext::intel::esimd;
using fp16 = sycl::half;

constexpr float LOG2E_F = 1.4426950408889634f;
constexpr float LN2_F = 0.6931471805599453f;
constexpr float NEG = -1.0e30f;

// E4M3 and binary16 share the significand after a seven-bit left shift;
// multiplying by 256 adjusts the exponent bias for both normals and subnormals.
template <int N>
ESIMD_INLINE simd<fp16, N> from_e4m3(simd<uint8_t, N> bytes) {
    simd<uint16_t, N> u = convert<uint16_t>(bytes);
    simd<uint16_t, N> bits = ((u & 127) << 7) | ((u & 128) << 8);
    return bits.template bit_cast_view<fp16>().read() * fp16(256.f);
}

ESIMD_INLINE float sexp2(float x) { return exp2(simd<float, 1>(x))[0]; }
ESIMD_INLINE float slog2(float x) { return log2(simd<float, 1>(x))[0]; }
ESIMD_INLINE float sigmoid1(float x) {
    return 1.f / (1.f + sexp2(-x * LOG2E_F));
}
ESIMD_INLINE simd<float, 32> sigmoid32(simd<float, 32> x) {
    return 1.f / (1.f + exp2(-x * LOG2E_F));
}
ESIMD_INLINE float bf16_to_f32(uint16_t b) {
    return sycl::bit_cast<float>(static_cast<uint32_t>(b) << 16);
}

// ============================================================================
// PASS 1 — ESIMD XMX split-KV paged decode attention.
// Grid: nd_range<3>((num_splits, KV_H=8, M_TILES), (1,1,1)); M_TILES covers
// R = q_len*3 GQA-packed slots (slot r = query row r/3, head hk*3 + r%3).
// Writes normalized partials OPart[splits,T,H,128] f32 and log2-domain
// LsePart[splits,T,H] f32 (empty splits get o=0, lse=-inf).
// ============================================================================
template <bool IS_FP8, int M_TILES>
struct PagedXmxEsimd {
    const uint16_t* q;          // bf16 [T, H, 128]
    const uint8_t* k;           // kv elems, key_cache view
    const uint8_t* v;           // kv elems, value_cache view
    const uint16_t* rel;        // bf16 [T+128, H, E_PAD] sheared rel logits
    const int32_t* block_table; // [1, max_pages]
    const int32_t* seq_lens;    // [1]
    const int32_t* cu_seqlens;  // [2]
    float* o_part;
    float* lse_part;

    float softmax_scale;
    int64_t stride_q_t, stride_q_h;
    int64_t stride_kv_blk, stride_kv_slot, stride_kv_h;
    int64_t stride_rel_t;
    int H, E, E_PAD, PAGE, WINDOW, num_tokens, num_splits;

    void operator()(sycl::nd_item<3> it) const SYCL_ESIMD_KERNEL {
        constexpr int BK = 16;
        const int part = static_cast<int>(it.get_global_id(0));
        const int hk = static_cast<int>(it.get_global_id(1));
        const int tile = static_cast<int>(it.get_global_id(2));

        const int q_start = cu_seqlens[0];
        const int q_len = cu_seqlens[1] - q_start;
        const int R = q_len * 3;
        const int r0 = tile * 8;
        if (r0 >= R) return;

        const int seqlen_k = seq_lens[0];
        const int past = seqlen_k - q_len;
        // Position range covered by this tile's query rows.
        const int tok0 = r0 / 3;
        const int tok1 = ((r0 + 7) / 3 < q_len) ? (r0 + 7) / 3 : (q_len - 1);
        const int pos_lo = past + tok0;
        const int pos_hi = past + tok1;

        int lo = 0;
        if (WINDOW > 0) {
            lo = pos_lo - (WINDOW - 1);
            if (lo < 0) lo = 0;
        }
        const int hi = (pos_hi + 1 < seqlen_k) ? (pos_hi + 1) : seqlen_k;
        const int lo_chunk = lo / BK;
        const int hi_chunk = (hi + BK - 1) / BK;
        const int nchunk = hi_chunk - lo_chunk;
        const int per = (nchunk + num_splits - 1) / num_splits;
        const int first = lo_chunk + part * per;
        const int last = (first + per < hi_chunk) ? (first + per) : hi_chunk;

        simd<fp16, 1024> Q = 0;
        simd<float, 1024> O = 0;
        simd<float, 8> m = NEG, total = 0.f;
#pragma unroll
        for (int row = 0; row < 8; ++row) {
            const int r = r0 + row;
            if (r < R) {
                const int64_t qoff = int64_t(q_start + r / 3) * stride_q_t +
                                     int64_t(hk * 3 + r % 3) * stride_q_h;
                // bf16 -> fp16 is exact (bf16 mantissa fits inside fp16).
                auto q16 = block_load<uint16_t, 128>(q + qoff);
                simd<uint32_t, 128> qb = convert<uint32_t>(q16) << 16;
                simd<fp16, 128> qf =
                    convert<fp16>(qb.template bit_cast_view<float>().read());
#pragma unroll
                for (int c = 0; c < 8; ++c)
                    Q.template select<16, 1>((c * 8 + row) * 16) =
                        qf.template select<16, 1>(c * 16);
            }
        }

        const unsigned surf_w = IS_FP8 ? 127u : 255u;   // row width, bytes - 1
        const unsigned pitch = unsigned(stride_kv_slot) * (IS_FP8 ? 1u : 2u) - 1u;

        for (int chunk = first; chunk < last; ++chunk) {
            const int index = chunk * BK;
            const int64_t base =
                int64_t(block_table[index / PAGE]) * stride_kv_blk +
                int64_t(index % PAGE) * stride_kv_slot +
                int64_t(hk) * stride_kv_h;

            // ---- QK^T : score[8 slots][16 keys] ----------------------------
            simd<float, 128> score = 0;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                simd<uint32_t, 128> k2;
                if constexpr (IS_FP8) {
                    auto k4 = xesimd::lsc_load_2d<uint32_t, 4, 16, 1, true, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                        reinterpret_cast<const uint32_t*>(k + base), surf_w, 15,
                        pitch, c * 4, 0);
#pragma unroll
                    for (int d = 0; d < 4; ++d) {
                        simd<uint32_t, 16> four = k4.template select<16, 1>(d * 16);
                        simd<uint16_t, 16> lo = convert<uint16_t>(four);
                        simd<uint16_t, 16> hiw = convert<uint16_t>(four >> 16);
                        auto val_lo = from_e4m3<32>(lo.template bit_cast_view<uint8_t>().read());
                        k2.template select<16, 1>(d * 32) =
                            val_lo.template bit_cast_view<uint32_t>().read();
                        auto val_hi = from_e4m3<32>(hiw.template bit_cast_view<uint8_t>().read());
                        k2.template select<16, 1>(d * 32 + 16) =
                            val_hi.template bit_cast_view<uint32_t>().read();
                    }
                } else {
                    auto kt = xesimd::lsc_load_2d<uint32_t, 8, 16, 1, true, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                        reinterpret_cast<const uint32_t*>(k + 2 * base), surf_w,
                        15, pitch, c * 8, 0);
                    simd<uint32_t, 256> bits =
                        convert<uint32_t>(kt.template bit_cast_view<uint16_t>().read()) << 16;
                    simd<fp16, 256> vals =
                        convert<fp16>(bits.template bit_cast_view<float>().read());
                    k2 = vals.template bit_cast_view<uint32_t>().read();
                }
                score = xmx::dpas<8, 8, float, float, fp16, fp16>(
                    score, k2.template bit_cast_view<fp16>().read(),
                    simd<fp16, 128>(Q.template select<128, 1>(c * 128)));
            }
            score *= LOG2E_F * softmax_scale;

            // ---- bias + mask + online softmax ------------------------------
            simd<float, 8> alpha;
            simd<fp16, 128> P;
            const bool need_bias = (index + BK - 1) >= (pos_lo - E + 1);
#pragma unroll
            for (int row = 0; row < 8; ++row) {
                const int r = r0 + row;
                simd<float, 16> s = score.template select<16, 1>(row * 16);
                simd<int, 16> key(index, 1);
                const int tok_r = r / 3;
                const int pos_r = past + tok_r;
                simd<int, 16> delta = pos_r - key;
                simd_mask<16> valid = (delta >= 0) & (key < seqlen_k) & (r < R);
                if (WINDOW > 0) valid &= (delta < WINDOW);
                if (need_bias) {
                    simd_mask<16> near = valid & (delta < E);
                    const int token = q_start + tok_r;
                    const int head = hk * 3 + r % 3;
                    // Sheared rel_logits: per-row rel_base from tok's shear tile.
                    const int tile_end =
                        (((tok_r / 128) + 1) * 128 < q_len)
                            ? ((tok_r / 128) + 1) * 128
                            : q_len;
                    const int rel_base =
                        ((past + tile_end + 127) / 128) * 128 - E_PAD;
                    simd<int, 16> col = key - rel_base;
                    simd<uint32_t, 16> off16 =
                        convert<uint32_t>(max(col, simd<int, 16>(0))) * 2;
                    if (r < R) {
                        auto b16 = gather<uint16_t, 16>(
                            rel + int64_t(token) * stride_rel_t +
                                int64_t(head) * E_PAD,
                            off16, near);
                        simd<uint32_t, 16> bb = convert<uint32_t>(b16) << 16;
                        s.merge(s + bb.template bit_cast_view<float>().read() *
                                        LOG2E_F,
                                near);
                    }
                }
                s.merge(NEG, !valid);
                const float peak = hmax<float>(s);
                const float mx = m[row] > peak ? float(m[row]) : peak;
                alpha[row] = sexp2(m[row] - mx);
                simd<float, 16> p = exp2(s - mx);
                p.merge(0.f, s < NEG * 0.5f);
                P.template select<16, 1>(row * 16) = convert<fp16>(p);
                total[row] = total[row] * alpha[row] + reduce<float>(p, std::plus<>());
                m[row] = mx;
            }

            // ---- P . V : O[8 slots][128 dims] ------------------------------
            auto rescale = alpha.template replicate_vs_w_hs<8, 1, 16, 0>(0);
            // Last partial chunk may hold slots >= seqlen_k whose bytes are
            // stale; masked scores give P=0, and 0*NaN = NaN, so zero the V
            // payload of out-of-range tokens on the BF16 path (FP8's dequant
            // output is always finite: e4m3 0x7F maps to +/-256.0).
            const bool oob_chunk = (index + BK > seqlen_k);
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                simd<uint32_t, 128> v2;
                if constexpr (IS_FP8) {
                    simd<uint8_t, 256> v4 =
                        xesimd::lsc_load_2d<uint8_t, 16, 16, 1, false, true,
                            xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                            v + base, surf_w, 15, pitch, c * 16, 0);
#pragma unroll
                    for (int d = 0; d < 4; ++d) {
                        simd<uint32_t, 16> four =
                            v4.template bit_cast_view<uint32_t>().template select<16, 1>(16 * d);
                        simd<uint16_t, 16> lo = convert<uint16_t>(four);
                        simd<uint16_t, 16> hiw = convert<uint16_t>(four >> 16);
                        auto val_lo = from_e4m3<32>(lo.template bit_cast_view<uint8_t>().read());
                        v2.template select<16, 1>(d * 32) =
                            val_lo.template bit_cast_view<uint32_t>().read();
                        auto val_hi = from_e4m3<32>(hiw.template bit_cast_view<uint8_t>().read());
                        v2.template select<16, 1>(d * 32 + 16) =
                            val_hi.template bit_cast_view<uint32_t>().read();
                    }
                } else {
                    simd<uint16_t, 256> vt =
                        xesimd::lsc_load_2d<uint16_t, 16, 16, 1, false, true,
                            xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                            reinterpret_cast<const uint16_t*>(v + 2 * base),
                            surf_w, 15, pitch, c * 16, 0);
                    if (oob_chunk) {
                        // vt layout [tok pair][dim][tok-in-pair]: element j
                        // holds token index + 2*(j/32) + (j%2).
                        simd<int, 256> lane(0, 1);
                        simd<int, 256> tv = index + (lane >> 5) * 2 + (lane & 1);
                        vt.merge(uint16_t(0), tv >= seqlen_k);
                    }
                    simd<uint32_t, 256> bits = convert<uint32_t>(vt) << 16;
                    simd<fp16, 256> vals =
                        convert<fp16>(bits.template bit_cast_view<float>().read());
                    v2 = vals.template bit_cast_view<uint32_t>().read();
                }
                simd<float, 128> acc =
                    simd<float, 128>(O.template select<128, 1>(c * 128)) * rescale;
                O.template select<128, 1>(c * 128) =
                    xmx::dpas<8, 8, float, float, fp16, fp16>(
                        acc, v2.template bit_cast_view<fp16>().read(), P);
            }
        }

        // ---- epilogue: normalized partial + log2-domain lse ----------------
#pragma unroll
        for (int row = 0; row < 8; ++row) {
            const int r = r0 + row;
            if (r >= R) break;
            const int token = q_start + r / 3;
            const int64_t dest =
                (int64_t(part) * num_tokens + token) * H + (hk * 3 + r % 3);
            simd<float, 128> o;
            const float den = total[row];
#pragma unroll
            for (int c = 0; c < 8; ++c)
                o.template select<16, 1>(16 * c) =
                    O.template select<16, 1>((c * 8 + row) * 16);
            block_store<float, 128>(o_part + dest * 128,
                                    o * (den > 0.f ? 1.f / den : 0.f));
            lse_part[dest] =
                den > 0.f ? (float(m[row]) + slog2(den)) : -INFINITY;
        }
    }
};

// ============================================================================
// PASS 2 — ESIMD combine + fused gate/sink.
// Grid: nd_range<2>((num_tokens, H*4), (1,1)); each item owns 32 output dims of
// one (token, head): o = sum_s w_s * o_part_s / sum_s w_s,
// w_s = exp2(lse_s - max), then o *= sigmoid(gate) * sigmoid(lse_nat - sink).
// Matches _rel_decode_combine_kernel + FullDecodeCombineKernel semantics.
// ============================================================================
struct EsimdCombineGated {
    const float* o_part;
    const float* lse_part;
    const uint16_t* gate;   // bf16 [T, H*D]
    const uint16_t* sink;   // bf16 [T, H]
    uint16_t* out;          // bf16 [T, H*D]

    int64_t stride_gate, stride_sink, stride_out_t;
    int num_tokens, num_actual_tokens, H, num_splits;

    void operator()(sycl::nd_item<2> it) const SYCL_ESIMD_KERNEL {
        const int tok = static_cast<int>(it.get_global_id(0));
        const int h4 = static_cast<int>(it.get_global_id(1));
        const int head = h4 >> 2;
        const int dc = h4 & 3;
        uint16_t* orow =
            out + int64_t(tok) * stride_out_t + head * 128 + dc * 32;
        if (tok >= num_actual_tokens) {
            block_store<uint16_t, 32>(orow, simd<uint16_t, 32>(0));
            return;
        }
        const int64_t lse_off = int64_t(tok) * H + head;
        float mx = -INFINITY;
        for (int s = 0; s < num_splits; ++s) {
            const float ls = lse_part[int64_t(s) * num_tokens * H + lse_off];
            mx = (ls > mx) ? ls : mx;
        }
        float w_sum = 0.f;
        simd<float, 32> acc = 0.f;
        if (mx != -INFINITY) {
            for (int s = 0; s < num_splits; ++s) {
                const float ls = lse_part[int64_t(s) * num_tokens * H + lse_off];
                if (ls != -INFINITY) {
                    const float w = sexp2(ls - mx);
                    w_sum += w;
                    const float* op = o_part +
                        ((int64_t(s) * num_tokens + tok) * H + head) * 128 +
                        dc * 32;
                    acc += w * block_load<float, 32>(op);
                }
            }
        }
        simd<float, 32> o = acc * (w_sum > 0.f ? 1.f / w_sum : 0.f);
        const float lse_nat =
            w_sum > 0.f ? (mx + slog2(w_sum)) * LN2_F : -INFINITY;
        auto g16 = block_load<uint16_t, 32>(
            gate + int64_t(tok) * stride_gate + head * 128 + dc * 32);
        simd<uint32_t, 32> gb = convert<uint32_t>(g16) << 16;
        simd<float, 32> gv = gb.template bit_cast_view<float>().read();
        const float sink_v =
            bf16_to_f32(sink[int64_t(tok) * stride_sink + head]);
        o = o * (sigmoid32(gv) * sigmoid1(lse_nat - sink_v));
        // f32 -> bf16 with round-to-nearest-even (matches SIMT f32_to_bf16).
        simd<uint32_t, 32> u = o.template bit_cast_view<uint32_t>().read();
        u += 0x7FFFu + ((u >> 16) & 1u);
        simd<uint16_t, 32> o16 = convert<uint16_t>(u >> 16);
        block_store<uint16_t, 32>(orow, o16);
    }
};

void rel_decode_esimd_xe2(
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
    const int sp = static_cast<int>(num_splits);
    const int tiles = (static_cast<int>(max_query_len) * 3 + 7) / 8;

    TORCH_CHECK(seq_lens.size(0) == 1 && cu_seqlens_q.size(0) == 2,
                "esimd decode: single request only (num_reqs == 1)");
    TORCH_CHECK(max_query_len >= 1 && max_query_len <= 8,
                "esimd decode: max_query_len must be in [1, 8]");
    TORCH_CHECK(q.size(2) == 128 && q.stride(2) == 1 && q.stride(1) == 128,
                "esimd decode: q must be [T, H, 128] with contiguous head_dim");
    TORCH_CHECK(H == KV_H * 3 && KV_H == 8, "esimd decode: H = 3 * KV_H = 24");
    TORCH_CHECK(PAGE >= 16 && PAGE % 16 == 0,
                "esimd decode: page size must be a multiple of 16");
    TORCH_CHECK(key_cache.stride(3) == 1 && value_cache.stride(3) == 1,
                "esimd decode: kv head_dim must be contiguous");
    TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0) &&
                key_cache.stride(1) == value_cache.stride(1) &&
                key_cache.stride(2) == value_cache.stride(2),
                "esimd decode: key/value cache strides must match");
    TORCH_CHECK(rel_logits.stride(1) == E_PAD && rel_logits.stride(2) == 1,
                "esimd decode: sheared rel_logits layout mismatch");
    const bool is_fp8 = key_cache.element_size() == 1;
    TORCH_CHECK(is_fp8 == (value_cache.element_size() == 1),
                "esimd decode: key/value dtype mismatch");
    TORCH_CHECK(key_cache.element_size() == 1 || key_cache.element_size() == 2,
                "esimd decode: kv cache must be fp8 e4m3 or bf16");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 &&
                rel_logits.scalar_type() == at::kBFloat16 &&
                gate.scalar_type() == at::kBFloat16 &&
                sinks.scalar_type() == at::kBFloat16 &&
                out.scalar_type() == at::kBFloat16,
                "esimd decode: q/rel_logits/gate/sinks/out must be bf16");
    TORCH_CHECK(gate.stride(1) == 1 && out.stride(1) == 1 &&
                sinks.stride(1) == 1 && block_table.stride(1) == 1,
                "esimd decode: gate/sinks/out/block_table must be inner-contiguous");
    TORCH_CHECK(block_table.scalar_type() == at::kInt &&
                seq_lens.scalar_type() == at::kInt &&
                cu_seqlens_q.scalar_type() == at::kInt,
                "esimd decode: block_table/seq_lens/cu_seqlens_q must be int32");
    TORCH_CHECK(o_part.scalar_type() == at::kFloat &&
                lse_part.scalar_type() == at::kFloat &&
                o_part.is_contiguous() && lse_part.is_contiguous() &&
                o_part.numel() >= int64_t(sp) * num_tokens * H * 128 &&
                lse_part.numel() >= int64_t(sp) * num_tokens * H,
                "esimd decode: o_part/lse_part must be contiguous fp32 "
                "scratch of shape [num_splits, T, H, 128] / [num_splits, T, H]");
    TORCH_CHECK(num_actual_tokens >= 0 && num_actual_tokens <= num_tokens,
                "esimd decode: num_actual_tokens out of range");

    auto launch = [&]<bool FP8, int TILES>() {
        PagedXmxEsimd<FP8, TILES> kern{
            reinterpret_cast<const uint16_t*>(q.data_ptr()),
            reinterpret_cast<const uint8_t*>(key_cache.data_ptr()),
            reinterpret_cast<const uint8_t*>(value_cache.data_ptr()),
            reinterpret_cast<const uint16_t*>(rel_logits.data_ptr()),
            block_table.data_ptr<int32_t>(),
            seq_lens.data_ptr<int32_t>(),
            cu_seqlens_q.data_ptr<int32_t>(),
            o_part.data_ptr<float>(), lse_part.data_ptr<float>(),
            static_cast<float>(softmax_scale),
            q.stride(0), q.stride(1),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            rel_logits.stride(0),
            H, E, E_PAD, PAGE, static_cast<int>(window), num_tokens, sp};
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(sp, KV_H, TILES),
                                  sycl::range<3>(1, 1, 1)),
                kern);
        });
    };
    if (is_fp8) {
        switch (tiles) {
            case 1: launch.template operator()<true, 1>(); break;
            case 2: launch.template operator()<true, 2>(); break;
            default: launch.template operator()<true, 3>(); break;
        }
    } else {
        switch (tiles) {
            case 1: launch.template operator()<false, 1>(); break;
            case 2: launch.template operator()<false, 2>(); break;
            default: launch.template operator()<false, 3>(); break;
        }
    }

    queue.submit([&](sycl::handler& cgh) {
        EsimdCombineGated ck{
            o_part.data_ptr<float>(), lse_part.data_ptr<float>(),
            reinterpret_cast<const uint16_t*>(gate.data_ptr()),
            reinterpret_cast<const uint16_t*>(sinks.data_ptr()),
            reinterpret_cast<uint16_t*>(out.data_ptr()),
            gate.stride(0), sinks.stride(0), out.stride(0),
            num_tokens, static_cast<int>(num_actual_tokens), H, sp};
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(num_tokens, H * 4),
                              sycl::range<2>(1, 1)),
            ck);
    });
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rel_decode_esimd_xe2", &rel_decode_esimd_xe2,
          "Opt-in ESIMD XMX Paged Decode Attention (AOT bmg_g31)");
}
