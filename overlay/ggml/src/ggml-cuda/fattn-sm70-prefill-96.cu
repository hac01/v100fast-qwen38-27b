// High-occupancy 12-warp (384-thread, REG<=164, STACK:0) Volta HMMA.884 Prefill FlashAttention (Stage 6F).
// Processes Br=96 consecutive Q tokens of one head per CTA, Bc=32 KV keys per tile, D=256.
// 3x higher warp occupancy than sm70_d256_splitd_dense_kernel (12 warps vs 4 warps @ REG:255),
// zero split-K merge, causal tile skipping, incremental KV pointers, and zero-branch q8_0 commit.
#include "fattn-sm70-prefill-96.cuh"
#include <math_constants.h>

namespace sm70_prefill96 {

constexpr int kDSlice  = kD / kSubWarps; // 64
constexpr int kDChunks = kD / 8;         // 32
constexpr int kDLocal  = kDSlice / 8;    // 8

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

static __device__ __forceinline__ int qp_get_i() { return threadIdx.x & 31; }
static __device__ __forceinline__ int k_get_i() {
    const int lane = threadIdx.x & 31;
    return ((lane / 16) * 4) + (lane % 4);
}
static __device__ __forceinline__ int d_get_i(int l) {
    const int lane = threadIdx.x & 31;
    return (l & 2) + (lane & ~2);
}
static __device__ __forceinline__ int d_get_j(int l) {
    const int lane = threadIdx.x & 31;
    return (lane & 2) + (l & (4 + 1));
}
static __device__ __forceinline__ void softmax_to_half2(half2 (&p)[4], const float (&d)[8]) {
    const bool swap_hi = ((threadIdx.x & 2) == 0);
    const half2 a0 = __floats2half2_rn(d[0], d[1]);
    const half2 a1 = __floats2half2_rn(d[2], d[3]);
    const half2 a2 = __floats2half2_rn(d[4], d[5]);
    const half2 a3 = __floats2half2_rn(d[6], d[7]);
    const half2 s0 = __shfl_xor_sync(0xFFFFFFFFu, swap_hi ? a1 : a0, 2, 32);
    const half2 s1 = __shfl_xor_sync(0xFFFFFFFFu, swap_hi ? a3 : a2, 2, 32);
    p[0] = swap_hi ? a0 : s0;
    p[1] = swap_hi ? s0 : a1;
    p[2] = swap_hi ? a2 : s1;
    p[3] = swap_hi ? s1 : a3;
}

struct KvPrefetchF16 { uint4 r0, r1, r2; };
struct KvPrefetchQ8  { uint16_t s0, s1, s2; uint32_t a0, b0, c0, a1, b1, c1, a2, b2, c2; };

template <int KVMode> struct KvPrefetch { KvPrefetchF16 f; };
template <>           struct KvPrefetch<2> { KvPrefetchQ8 q; };

static __device__ __forceinline__ void prefetch_q8_row(
        const uint8_t * __restrict__ blk, int q_off, bool q_unal, bool valid,
        uint16_t & sc, uint32_t & u0, uint32_t & u1, uint32_t & u2) {
    if (valid) {
        const uint32_t * p32 = reinterpret_cast<const uint32_t *>(blk + q_off - (q_unal ? 2 : 0));
        sc = __ldcs(reinterpret_cast<const uint16_t *>(blk));
        u0 = __ldcs(p32 + 0);
        u1 = __ldcs(p32 + 1);
        u2 = q_unal ? __ldcs(p32 + 2) : 0u;
    } else {
        sc = 0; u0 = 0x80808080u; u1 = 0x80808080u; u2 = 0x80808080u;
    }
}

static __device__ __forceinline__ void commit_q8_row(
        uint4 * __restrict__ dst_u4, bool q_unal,
        uint16_t sc, uint32_t u0, uint32_t u1, uint32_t u2, const half2 neg1152) {
    const uint32_t w0 = q_unal ? __byte_perm(u0, u1, 0x5432u) : u0;
    const uint32_t w1 = q_unal ? __byte_perm(u1, u2, 0x5432u) : u1;
    const half  dh = *reinterpret_cast<const half *>(&sc);
    const half2 hd = __half2half2(dh);
    const uint32_t x0 = w0 ^ 0x80808080u;
    const uint32_t x1 = w1 ^ 0x80808080u;
    const uint32_t h0 = __byte_perm(x0, 0x00006400u, 0x5150u);
    const uint32_t h1 = __byte_perm(x0, 0x00006400u, 0x5352u);
    const uint32_t h2 = __byte_perm(x1, 0x00006400u, 0x5150u);
    const uint32_t h3 = __byte_perm(x1, 0x00006400u, 0x5352u);
    const half2 o0 = __hmul2(__hadd2(*reinterpret_cast<const half2 *>(&h0), neg1152), hd);
    const half2 o1 = __hmul2(__hadd2(*reinterpret_cast<const half2 *>(&h1), neg1152), hd);
    const half2 o2 = __hmul2(__hadd2(*reinterpret_cast<const half2 *>(&h2), neg1152), hd);
    const half2 o3 = __hmul2(__hadd2(*reinterpret_cast<const half2 *>(&h3), neg1152), hd);
    *dst_u4 = make_uint4(
        *reinterpret_cast<const uint32_t *>(&o0),
        *reinterpret_cast<const uint32_t *>(&o1),
        *reinterpret_cast<const uint32_t *>(&o2),
        *reinterpret_cast<const uint32_t *>(&o3));
}

template <int KVMode>
static __device__ __forceinline__ void prefetch_kv_tile(
        KvPrefetch<KVMode> & pf,
        const uint8_t * __restrict__ ptr0, int64_t rstride_b,
        int k_rem, int tid) {
    const int kl0 = tid >> 5;
    const int d0  = (tid & 31) << 3;
    const bool full = (k_rem >= kBc);
    if constexpr (KVMode == 2) {
        const int q_off = 2 + (d0 & 31);
        const bool q_unal = (((d0 >> 5) & 1) == 0);
        prefetch_q8_row(ptr0,                  q_off, q_unal, full || (kl0      < k_rem), pf.q.s0, pf.q.a0, pf.q.b0, pf.q.c0);
        prefetch_q8_row(ptr0 + 12 * rstride_b, q_off, q_unal, full || (kl0 + 12 < k_rem), pf.q.s1, pf.q.a1, pf.q.b1, pf.q.c1);
        if (kl0 < 8) {
            prefetch_q8_row(ptr0 + 24 * rstride_b, q_off, q_unal, full || (kl0 + 24 < k_rem), pf.q.s2, pf.q.a2, pf.q.b2, pf.q.c2);
        }
    } else {
        auto ld1 = [&](const uint8_t * p, bool valid, uint4 & r) {
            r = valid ? *reinterpret_cast<const uint4 *>(p) : make_uint4(0u, 0u, 0u, 0u);
        };
        ld1(ptr0,                  full || (kl0      < k_rem), pf.f.r0);
        ld1(ptr0 + 12 * rstride_b, full || (kl0 + 12 < k_rem), pf.f.r1);
        if (kl0 < 8) { ld1(ptr0 + 24 * rstride_b, full || (kl0 + 24 < k_rem), pf.f.r2); }
    }
}

template <int KVMode>
static __device__ __forceinline__ void commit_kv_tile(
        half * __restrict__ kv_s, const KvPrefetch<KVMode> & pf, int tid) {
    const int kl0 = tid >> 5;
    const int d0  = (tid & 31) << 3;
    uint4 * dst0 = reinterpret_cast<uint4 *>(&kv_s[kl0 * kSmemStride + d0]);
    constexpr int s12 = (12 * kSmemStride) / 8;
    if constexpr (KVMode == 2) {
        const unsigned neg1152_bits = 0xE480E480u;
        const half2 neg1152 = *reinterpret_cast<const half2 *>(&neg1152_bits);
        const bool q_unal = (((d0 >> 5) & 1) == 0);
        commit_q8_row(dst0,           q_unal, pf.q.s0, pf.q.a0, pf.q.b0, pf.q.c0, neg1152);
        commit_q8_row(dst0 + s12,     q_unal, pf.q.s1, pf.q.a1, pf.q.b1, pf.q.c1, neg1152);
        if (kl0 < 8) {
            commit_q8_row(dst0 + 2 * s12, q_unal, pf.q.s2, pf.q.a2, pf.q.b2, pf.q.c2, neg1152);
        }
    } else {
        *dst0 = pf.f.r0;
        *(dst0 + s12) = pf.f.r1;
        if (kl0 < 8) { *(dst0 + 2 * s12) = pf.f.r2; }
    }
}

static __device__ __forceinline__ void load_qp_128(half2 (&dst)[4], const half2 * __restrict__ base, int stride_h2) {
    *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(base + qp_get_i() * stride_h2);
}
static __device__ __forceinline__ void load_k_128(half2 (&dst)[4], const half2 * __restrict__ base, int stride_h2) {
    *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(base + k_get_i() * stride_h2);
}
static __device__ __forceinline__ void load_v_64(half2 (&dst)[4], const half2 * __restrict__ base, int stride_h2) {
    const int lane = threadIdx.x & 31;
    const int i0 = lane & 3;
    const int i1 = 4 + i0;
    const int j0 = (lane >> 4) << 1;
    *reinterpret_cast<uint2 *>(&dst[0]) = *reinterpret_cast<const uint2 *>(base + i0 * stride_h2 + j0);
    *reinterpret_cast<uint2 *>(&dst[2]) = *reinterpret_cast<const uint2 *>(base + i1 * stride_h2 + j0);
}
static __device__ __forceinline__ void mma_qk_k0(float (&d)[8], const half2 (&q)[4], const half2 (&k)[4]) {
    const int * A = reinterpret_cast<const int *>(q);
    const int * B = reinterpret_cast<const int *>(k);
    int * D       = reinterpret_cast<int *>(d);
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3]), "+r"(D[4]), "+r"(D[5]), "+r"(D[6]), "+r"(D[7])
        : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]));
}
static __device__ __forceinline__ void mma_qk_k1(float (&d)[8], const half2 (&q)[4], const half2 (&k)[4]) {
    const int * A = reinterpret_cast<const int *>(q);
    const int * B = reinterpret_cast<const int *>(k);
    int * D       = reinterpret_cast<int *>(d);
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3]), "+r"(D[4]), "+r"(D[5]), "+r"(D[6]), "+r"(D[7])
        : "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]));
}
static __device__ __forceinline__ void mma_pv_f32_k0(float (&d)[8], const half2 (&p)[4], const half2 (&v)[4]) {
    const int * P = reinterpret_cast<const int *>(p);
    const int * V = reinterpret_cast<const int *>(v);
    int * D       = reinterpret_cast<int *>(d);
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3]), "+r"(D[4]), "+r"(D[5]), "+r"(D[6]), "+r"(D[7])
        : "r"(P[0]), "r"(P[1]), "r"(V[0]), "r"(V[1]));
}
static __device__ __forceinline__ void mma_pv_f32_k1(float (&d)[8], const half2 (&p)[4], const half2 (&v)[4]) {
    const int * P = reinterpret_cast<const int *>(p);
    const int * V = reinterpret_cast<const int *>(v);
    int * D       = reinterpret_cast<int *>(d);
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3]), "+r"(D[4]), "+r"(D[5]), "+r"(D[6]), "+r"(D[7])
        : "r"(P[2]), "r"(P[3]), "r"(V[2]), "r"(V[3]));
}

#endif

template <int KMode, int VMode, bool MaskMD>
__global__ void __launch_bounds__(kThreads, 1) sm70_prefill96_kernel(
        const float * __restrict__ q,
        int64_t q_row_stride, int64_t q_head_stride,
        const half * __restrict__ k, const half * __restrict__ v,
        int64_t k_row_stride, int64_t k_head_stride,
        int64_t v_row_stride, int64_t v_head_stride,
        float * __restrict__ out,
        int64_t o_row_stride, int64_t o_head_stride,
        int q_len, int kv_len, int gqa, float scale,
        const half * __restrict__ mask_last_q, int64_t mask_q_stride,
        const half * __restrict__ mask_kv, int64_t mask_kv_stride) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
    extern __shared__ char smem_raw[];
    half   * q_s    = reinterpret_cast<half   *>(smem_raw);
    half   * kv_s   = reinterpret_cast<half   *>(smem_raw + kQBytes);
    uint4  * p_s    = reinterpret_cast<uint4  *>(smem_raw + kQBytes + kKVBytes);
    float2 * stat_s = reinterpret_cast<float2 *>(smem_raw + kQBytes + kKVBytes + kPBytes);
    float2 * sum_s  = reinterpret_cast<float2 *>(smem_raw + kQBytes + kKVBytes + kPBytes + kStatBytes);

    auto p_ptr    = [&](int rg, int sub) -> uint4  * { return p_s    + ((rg * 4 + sub) * 32); };
    auto stat_ptr = [&](int rg, int sub) -> float2 * { return stat_s + ((rg * 4 + sub) * 32); };
    auto sum_ptr  = [&](int rg, int sub) -> float2 * { return sum_s  + ((rg * 4 + sub) * 32); };

    constexpr float Log2E   = 1.4426950408889634074f;
    constexpr unsigned Full = 0xffffffffu;

    const int q_block = (gridDim.x - 1) - blockIdx.x; // reverse order: longest causal blocks launch first!
    const int head_q  = blockIdx.y;
    const int kv_head = head_q / gqa;
    const int q0      = q_block * kBr;
    const int q_rows  = min(kBr, q_len - q0);

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int rg   = warp >> 2;
    const int lw   = warp & 3;

    const int row_base  = q0 + rg * 32;
    const int rows_here = max(0, min(32, q_len - row_base));

    int live = kv_len;
    if constexpr (!MaskMD) {
        int lo = 0, hi = kv_len;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (__half2float(mask_last_q[(int64_t) mid * mask_q_stride]) > -30000.0f) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        live = lo;
    }
    const int kv_offset = live - q_len;
    // Causal upper bound for this Q block [q0, q0 + q_rows): no key > kv_offset + q0 + q_rows - 1 is attended!
    const int k_end = MaskMD ? live : min(live, kv_offset + q0 + q_rows);

    const int64_t k_rstride_b = (KMode == 0) ? (k_row_stride * (int64_t) sizeof(half)) : k_row_stride;
    const int64_t v_rstride_b = (VMode == 0) ? (v_row_stride * (int64_t) sizeof(half)) : v_row_stride;
    const int d0_init = lane << 3;
    const int64_t k_extra = (KMode == 2) ? ((d0_init >> 5) * 34) : (d0_init * (int64_t) sizeof(half));
    const int64_t v_extra = (VMode == 2) ? ((d0_init >> 5) * 34) : (d0_init * (int64_t) sizeof(half));
    const int64_t k_hoff_b = (KMode == 0) ? ((int64_t) kv_head * k_head_stride * (int64_t) sizeof(half)) : ((int64_t) kv_head * k_head_stride);
    const int64_t v_hoff_b = (VMode == 0) ? ((int64_t) kv_head * v_head_stride * (int64_t) sizeof(half)) : ((int64_t) kv_head * v_head_stride);
    const uint8_t * k_ptr = reinterpret_cast<const uint8_t *>(k) + (int64_t) warp * k_rstride_b + k_hoff_b + k_extra;
    const uint8_t * v_ptr = reinterpret_cast<const uint8_t *>(v) + (int64_t) warp * v_rstride_b + v_hoff_b + v_extra;

    // Stage Q [q0..q0+q_rows) for head_q into q_s (zero GQA division!)
    const float * q_head_ptr = q + (int64_t) head_q * q_head_stride;
    for (int chunk = tid; chunk < kBr * (kD / 8); chunk += kThreads) {
        const int r = chunk >> 5;
        const int d = (chunk & 31) << 3;
        half * dst  = &q_s[r * kSmemStride + d];
        if (r < q_rows) {
            const float * src = q_head_ptr + (int64_t) (q0 + r) * q_row_stride + d;
            const float4 v0 = *reinterpret_cast<const float4 *>(src);
            const float4 v1 = *reinterpret_cast<const float4 *>(src + 4);
            const half2 h0 = __floats2half2_rn(v0.x, v0.y);
            const half2 h1 = __floats2half2_rn(v0.z, v0.w);
            const half2 h2 = __floats2half2_rn(v1.x, v1.y);
            const half2 h3 = __floats2half2_rn(v1.z, v1.w);
            *reinterpret_cast<uint4 *>(dst) = make_uint4(
                *reinterpret_cast<const uint32_t *>(&h0),
                *reinterpret_cast<const uint32_t *>(&h1),
                *reinterpret_cast<const uint32_t *>(&h2),
                *reinterpret_cast<const uint32_t *>(&h3));
        } else {
            *reinterpret_cast<uint4 *>(dst) = make_uint4(0u, 0u, 0u, 0u);
        }
    }

    float m_lo = -CUDART_INF_F, m_hi = -CUDART_INF_F;
    float l_lo = 0.0f, l_hi = 0.0f;
    float acc_f[kDLocal][8];
#pragma unroll
    for (int c = 0; c < kDLocal; ++c) {
#pragma unroll
        for (int j = 0; j < 8; ++j) { acc_f[c][j] = 0.0f; }
    }

    const int row_lo = d_get_i(0);
    const int row_hi = d_get_i(2);
    const int tok_lo = row_base + row_lo;
    const int tok_hi = row_base + row_hi;
    const bool r_lo_ok = (row_lo < rows_here);
    const bool r_hi_ok = (row_hi < rows_here);
    // Max causal key for the lowest row in this row-group (all keys < rg_causal_lim are unmasked for all 32 rows!)
    const int rg_causal_lim = kv_offset + row_base;

    const half2 * q_rg_h2 = reinterpret_cast<const half2 *>(q_s + rg * 32 * kSmemStride);
    const half2 * kv_h2   = reinterpret_cast<const half2 *>(kv_s);

    KvPrefetch<KMode> pf_k;
    if (k_end > 0) {
        prefetch_kv_tile<KMode>(pf_k, k_ptr, k_rstride_b, k_end, tid);
    }
    __syncthreads();

    for (int k0 = 0; k0 < k_end; k0 += kBc) {
        const int k_rem = k_end - k0;
        commit_kv_tile<KMode>(kv_s, pf_k, tid);
        KvPrefetch<VMode> pf_v;
        prefetch_kv_tile<VMode>(pf_v, v_ptr, v_rstride_b, k_rem, tid);
        __syncthreads();

        float my_s[8];
        if (rows_here > 0) {
            float d0[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            float d1[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            const half2 * k_lw_h2 = kv_h2 + lw * 8 * (kSmemStride / 2);
#pragma unroll 4
            for (int c = 0; c < kDChunks; c += 2) {
                half2 qf0[4], kf0[4], qf1[4], kf1[4];
                load_qp_128(qf0, q_rg_h2 + (c + 0) * 4, kSmemStride / 2);
                load_k_128 (kf0, k_lw_h2 + (c + 0) * 4, kSmemStride / 2);
                load_qp_128(qf1, q_rg_h2 + (c + 1) * 4, kSmemStride / 2);
                load_k_128 (kf1, k_lw_h2 + (c + 1) * 4, kSmemStride / 2);
                mma_qk_k0(d0, qf0, kf0);
                mma_qk_k0(d1, qf1, kf1);
                mma_qk_k1(d0, qf0, kf0);
                mma_qk_k1(d1, qf1, kf1);
            }
            const int sub_k0 = k0 + lw * 8;
            if ((!MaskMD) && (sub_k0 + 8 <= k_end) && (sub_k0 + 8 <= rg_causal_lim)) {
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const bool r_ok = ((l & 2) != 0) ? r_hi_ok : r_lo_ok;
                    my_s[l] = r_ok ? (d0[l] + d1[l]) * scale : -CUDART_INF_F;
                }
            } else {
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    float val = d0[l] + d1[l];
                    const bool is_hi = (l & 2) != 0;
                    const bool r_ok  = is_hi ? r_hi_ok : r_lo_ok;
                    const int  key   = sub_k0 + d_get_j(l);
                    if (r_ok && key < k_end) {
                        const int tok = is_hi ? tok_hi : tok_lo;
                        if (MaskMD || key >= kv_offset) {
                            const float mv = __half2float(mask_kv[(int64_t) tok * mask_kv_stride + key]);
                            val = (mv > -30000.0f) ? (val * scale + mv) : -CUDART_INF_F;
                        } else {
                            val *= scale;
                        }
                    } else {
                        val = -CUDART_INF_F;
                    }
                    my_s[l] = val;
                }
            }
            float sm_lo = fmaxf(fmaxf(my_s[0], my_s[1]), fmaxf(my_s[4], my_s[5]));
            float sm_hi = fmaxf(fmaxf(my_s[2], my_s[3]), fmaxf(my_s[6], my_s[7]));
            sm_lo = fmaxf(sm_lo, __shfl_xor_sync(Full, sm_lo, 2, 32));
            sm_hi = fmaxf(sm_hi, __shfl_xor_sync(Full, sm_hi, 2, 32));
            stat_ptr(rg, lw)[lane] = make_float2(sm_lo, sm_hi);
        }
        __syncthreads();

        commit_kv_tile<VMode>(kv_s, pf_v, tid);
        k_ptr += 32 * k_rstride_b;
        v_ptr += 32 * v_rstride_b;
        if (k_rem > kBc) {
            prefetch_kv_tile<KMode>(pf_k, k_ptr, k_rstride_b, k_rem - kBc, tid);
        }

        float a_lo = 1.0f, a_hi = 1.0f;
        if (rows_here > 0) {
            const float2 sm0 = stat_ptr(rg, 0)[lane];
            const float2 sm1 = stat_ptr(rg, 1)[lane];
            const float2 sm2 = stat_ptr(rg, 2)[lane];
            const float2 sm3 = stat_ptr(rg, 3)[lane];
            const float bm_lo = fmaxf(fmaxf(sm0.x, sm1.x), fmaxf(sm2.x, sm3.x));
            const float bm_hi = fmaxf(fmaxf(sm0.y, sm1.y), fmaxf(sm2.y, sm3.y));
            const float nm_lo = fmaxf(m_lo, bm_lo);
            const float nm_hi = fmaxf(m_hi, bm_hi);
            a_lo = (m_lo == -CUDART_INF_F) ? 0.0f : ((nm_lo == m_lo) ? 1.0f : exp2f((m_lo - nm_lo) * Log2E));
            a_hi = (m_hi == -CUDART_INF_F) ? 0.0f : ((nm_hi == m_hi) ? 1.0f : exp2f((m_hi - nm_hi) * Log2E));
            m_lo = nm_lo;
            m_hi = nm_hi;

#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const float nm = ((l & 2) == 0) ? nm_lo : nm_hi;
                my_s[l] = (nm > -CUDART_INF_F && my_s[l] > -CUDART_INF_F)
                              ? exp2f((my_s[l] - nm) * Log2E) : 0.0f;
            }
            float bl_lo = my_s[0] + my_s[1] + my_s[4] + my_s[5];
            float bl_hi = my_s[2] + my_s[3] + my_s[6] + my_s[7];
            bl_lo += __shfl_xor_sync(Full, bl_lo, 2, 32);
            bl_hi += __shfl_xor_sync(Full, bl_hi, 2, 32);
            sum_ptr(rg, lw)[lane] = make_float2(bl_lo, bl_hi);

            half2 p_my[4];
            softmax_to_half2(p_my, my_s);
            p_ptr(rg, lw)[lane] = *reinterpret_cast<const uint4 *>(p_my);
        }
        __syncthreads();

        if (rows_here > 0) {
            const float2 s0 = sum_ptr(rg, 0)[lane];
            const float2 s1 = sum_ptr(rg, 1)[lane];
            const float2 s2 = sum_ptr(rg, 2)[lane];
            const float2 s3 = sum_ptr(rg, 3)[lane];
            l_lo = l_lo * a_lo + (s0.x + s1.x + s2.x + s3.x);
            l_hi = l_hi * a_hi + (s0.y + s1.y + s2.y + s3.y);

            if (a_lo != 1.0f || a_hi != 1.0f) {
#pragma unroll
                for (int c = 0; c < kDLocal; ++c) {
                    acc_f[c][0] *= a_lo; acc_f[c][1] *= a_lo; acc_f[c][4] *= a_lo; acc_f[c][5] *= a_lo;
                    acc_f[c][2] *= a_hi; acc_f[c][3] *= a_hi; acc_f[c][6] *= a_hi; acc_f[c][7] *= a_hi;
                }
            }

            const half2 * v_slice = kv_h2 + (lw * kDSlice) / 2;
            constexpr int v_sub_stride = 8 * (kSmemStride / 2);
#pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                half2 p_sub[4];
                *reinterpret_cast<uint4 *>(p_sub) = p_ptr(rg, sub)[lane];
                const half2 * vsub = v_slice + sub * v_sub_stride;
#pragma unroll
                for (int c = 0; c < kDLocal; c += 2) {
                    half2 vf0[4], vf1[4];
                    load_v_64(vf0, vsub + (c + 0) * 4, kSmemStride / 2);
                    load_v_64(vf1, vsub + (c + 1) * 4, kSmemStride / 2);
                    mma_pv_f32_k0(acc_f[c + 0], p_sub, vf0);
                    mma_pv_f32_k0(acc_f[c + 1], p_sub, vf1);
                    mma_pv_f32_k1(acc_f[c + 0], p_sub, vf0);
                    mma_pv_f32_k1(acc_f[c + 1], p_sub, vf1);
                }
            }
        }
        __syncthreads();
    }

    // Write normalized output directly (ZERO split-K partials or merge kernel!)
    if (rows_here > 0) {
        const float inv_lo = (l_lo > 0.0f) ? (1.0f / l_lo) : 0.0f;
        const float inv_hi = (l_hi > 0.0f) ? (1.0f / l_hi) : 0.0f;
        const int d_base = lw * kDSlice;
        float * out_head = out + (int64_t) head_q * o_head_stride + d_base;
        float * out0 = r_lo_ok ? (out_head + (int64_t) tok_lo * o_row_stride) : nullptr;
        float * out1 = r_hi_ok ? (out_head + (int64_t) tok_hi * o_row_stride) : nullptr;
        const int j0 = d_get_j(0), j1 = d_get_j(1), j4 = d_get_j(4), j5 = d_get_j(5);
#pragma unroll
        for (int c = 0; c < kDLocal; ++c) {
            const int co = c * 8;
            if (out0 != nullptr) {
                out0[co + j0] = acc_f[c][0] * inv_lo;
                out0[co + j1] = acc_f[c][1] * inv_lo;
                out0[co + j4] = acc_f[c][4] * inv_lo;
                out0[co + j5] = acc_f[c][5] * inv_lo;
            }
            if (out1 != nullptr) {
                out1[co + j0] = acc_f[c][2] * inv_hi;
                out1[co + j1] = acc_f[c][3] * inv_hi;
                out1[co + j4] = acc_f[c][6] * inv_hi;
                out1[co + j5] = acc_f[c][7] * inv_hi;
            }
        }
    }
#endif
}

template <int KM, int VM, bool MM>
static void run_p96(
        dim3 grid, cudaStream_t stream,
        const float * q, int64_t q_row, int64_t q_head,
        const half * k, const half * v,
        int64_t k_row, int64_t k_head, int64_t v_row, int64_t v_head,
        float * out, int64_t o_row, int64_t o_head,
        int q_len, int kv_len, int gqa, float scale,
        const half * mask_last_q, int64_t mask_q_stride,
        const half * mask_kv, int64_t mask_kv_stride) {
    auto kfn = sm70_prefill96_kernel<KM, VM, MM>;
    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kSmemBytes));
        attr_set = true;
    }
    kfn<<<grid, kThreads, kSmemBytes, stream>>>(
        q, q_row, q_head, k, v, k_row, k_head, v_row, v_head,
        out, o_row, o_head, q_len, kv_len, gqa, scale,
        mask_last_q, mask_q_stride, mask_kv, mask_kv_stride);
}

void launch_sm70_prefill96(
        int k_mode, int v_mode, bool mask_md,
        cudaStream_t stream,
        const float * q, int64_t q_row, int64_t q_head,
        const half * k, const half * v,
        int64_t k_row, int64_t k_head, int64_t v_row, int64_t v_head,
        float * out, int64_t o_row, int64_t o_head,
        int q_len, int kv_len, int hq, int gqa, float scale,
        const half * mask_last_q, int64_t mask_q_stride,
        const half * mask_kv, int64_t mask_kv_stride) {
    const dim3 grid((unsigned) ((q_len + kBr - 1) / kBr), (unsigned) hq, 1);
    auto disp = [&](auto mm) {
        constexpr bool MM = decltype(mm)::value;
        if (k_mode == 2 && v_mode == 2) {
            run_p96<2, 2, MM>(grid, stream, q, q_row, q_head, k, v, k_row, k_head, v_row, v_head,
                out, o_row, o_head, q_len, kv_len, gqa, scale, mask_last_q, mask_q_stride, mask_kv, mask_kv_stride);
        } else {
            run_p96<0, 0, MM>(grid, stream, q, q_row, q_head, k, v, k_row, k_head, v_row, v_head,
                out, o_row, o_head, q_len, kv_len, gqa, scale, mask_last_q, mask_q_stride, mask_kv, mask_kv_stride);
        }
    };
    if (mask_md) { disp(std::true_type{}); } else { disp(std::false_type{}); }
}

} // namespace sm70_prefill96
