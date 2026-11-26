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


// ============================================================================
// Stage 9 Prefill Bc128 (Br=64, Bc=128, D=256, 8 warps = 256 threads, F16 KV)
// Solves the ncu-identified SMEM/MIO bottleneck of sm70_d256_splitd_dense_kernel:
//   - QK^T: each warp computes 32x32 (4 K-strips per Q load) -> 1.0 SMEM wf/HMMA (2.5x lower!)
//   - Double-buffered sK[2][128][72] (0% store bank conflicts, 1 barrier per D-chunk)
//   - Double-buffered sV[2][32][264] reusing the exact same 36.8 KB KV buffer
//   - Bc=128: 4x fewer online softmaxes, exp2f, O-rescales, and barriers per 128 KV tokens
//   - Zero Q staging, zero Os staging, zero SplitKV3 merge, zero cudaMemsetAsync
// ============================================================================

constexpr int kB128_Br        = 64;
constexpr int kB128_Bc        = 128;
constexpr int kB128_Threads   = 256;
constexpr int kB128_QStride   = 264; // 528 B (4 mod 32 banks)
constexpr int kB128_KStride   = 72;  // 144 B (4 mod 32 banks -> conflict-free load_k_128 & STS.128)
constexpr int kB128_VStride   = 264; // 528 B
constexpr size_t kB128_QBytes    = kB128_Br * kB128_QStride * sizeof(half);           // 33,792 B
constexpr size_t kB128_KBufElems = kB128_Bc * kB128_KStride;                          //  9,216 halfs
constexpr size_t kB128_VBufElems = 32 * kB128_VStride;                                //  8,448 halfs
constexpr size_t kB128_KVBytes   = 2 * kB128_KBufElems * sizeof(half);                // 36,864 B (holds 2x K or 2x V)
constexpr size_t kB128_PBytes    = 2 * 4 * 4 * 32 * sizeof(uint4);                    // 16,384 B
constexpr size_t kB128_StatBytes = 2 * 4 * 32 * sizeof(float2);                       //  2,048 B
constexpr size_t kB128_SumBytes  = 2 * 4 * 32 * sizeof(float2);                       //  2,048 B
constexpr size_t kB128_SmemBytes = kB128_QBytes + kB128_KVBytes + kB128_PBytes + kB128_StatBytes + kB128_SumBytes; // 91,136 B (< 96 KB)

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
static __device__ __forceinline__ uint4 ldg_cg_u4(const uint4 * ptr) {
    uint4 ret;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
                 : "l"(ptr));
    return ret;
}

static __device__ __forceinline__ void b128_prefetch_k_chunk(
        uint4 (&pf)[4], const half * __restrict__ k_head_ptr,
        int64_t k_row_stride, int k0, int k_rem, int dc, int tid) {
    const int r0 = tid >> 3;           // 0..31
    const int d0 = dc * 64 + ((tid & 7) << 3);
    const half * base = k_head_ptr + (int64_t) (k0 + r0) * k_row_stride + d0;
    const int64_t s32 = (int64_t) 32 * k_row_stride;
    if (k_rem >= 128) {
        pf[0] = ldg_cg_u4(reinterpret_cast<const uint4 *>(base));
        pf[1] = ldg_cg_u4(reinterpret_cast<const uint4 *>(base + s32));
        pf[2] = ldg_cg_u4(reinterpret_cast<const uint4 *>(base + 2 * s32));
        pf[3] = ldg_cg_u4(reinterpret_cast<const uint4 *>(base + 3 * s32));
    } else {
        pf[0] = (r0      < k_rem) ? ldg_cg_u4(reinterpret_cast<const uint4 *>(base))           : make_uint4(0u,0u,0u,0u);
        pf[1] = (r0 + 32 < k_rem) ? ldg_cg_u4(reinterpret_cast<const uint4 *>(base + s32))     : make_uint4(0u,0u,0u,0u);
        pf[2] = (r0 + 64 < k_rem) ? ldg_cg_u4(reinterpret_cast<const uint4 *>(base + 2 * s32)) : make_uint4(0u,0u,0u,0u);
        pf[3] = (r0 + 96 < k_rem) ? ldg_cg_u4(reinterpret_cast<const uint4 *>(base + 3 * s32)) : make_uint4(0u,0u,0u,0u);
    }
}

static __device__ __forceinline__ void b128_commit_k_chunk(
        half * __restrict__ sk_buf, const uint4 (&pf)[4], int tid) {
    const int r0 = tid >> 3;
    const int d0 = (tid & 7) << 3;
    uint4 * dst = reinterpret_cast<uint4 *>(&sk_buf[r0 * kB128_KStride + d0]);
    constexpr int s32 = (32 * kB128_KStride) / 8;
    dst[0]       = pf[0];
    dst[s32]     = pf[1];
    dst[2 * s32] = pf[2];
    dst[3 * s32] = pf[3];
}

static __device__ __forceinline__ void b128_prefetch_v_tile(
        uint4 (&pf)[4], const half * __restrict__ v_head_ptr,
        int64_t v_row_stride, int kv_base, int v_rem, int tid) {
    const int r0 = tid >> 5;           // 0..7
    const int d0 = (tid & 31) << 3;    // 0..248
    const half * base = v_head_ptr + (int64_t) (kv_base + r0) * v_row_stride + d0;
    const int64_t s8 = (int64_t) 8 * v_row_stride;
    if (v_rem >= 32) {
        pf[0] = ldg_cg_u4(reinterpret_cast<const uint4 *>(base));
        pf[1] = ldg_cg_u4(reinterpret_cast<const uint4 *>(base + s8));
        pf[2] = ldg_cg_u4(reinterpret_cast<const uint4 *>(base + 2 * s8));
        pf[3] = ldg_cg_u4(reinterpret_cast<const uint4 *>(base + 3 * s8));
    } else {
        pf[0] = (r0      < v_rem) ? ldg_cg_u4(reinterpret_cast<const uint4 *>(base))          : make_uint4(0u,0u,0u,0u);
        pf[1] = (r0 +  8 < v_rem) ? ldg_cg_u4(reinterpret_cast<const uint4 *>(base + s8))     : make_uint4(0u,0u,0u,0u);
        pf[2] = (r0 + 16 < v_rem) ? ldg_cg_u4(reinterpret_cast<const uint4 *>(base + 2 * s8)) : make_uint4(0u,0u,0u,0u);
        pf[3] = (r0 + 24 < v_rem) ? ldg_cg_u4(reinterpret_cast<const uint4 *>(base + 3 * s8)) : make_uint4(0u,0u,0u,0u);
    }
}

static __device__ __forceinline__ void b128_commit_v_tile(
        half * __restrict__ sv_buf, const uint4 (&pf)[4], int tid) {
    const int r0 = tid >> 5;
    const int d0 = (tid & 31) << 3;
    uint4 * dst = reinterpret_cast<uint4 *>(&sv_buf[r0 * kB128_VStride + d0]);
    constexpr int s8 = (8 * kB128_VStride) / 8;
    dst[0]      = pf[0];
    dst[s8]     = pf[1];
    dst[2 * s8] = pf[2];
    dst[3 * s8] = pf[3];
}
#endif

template <bool MaskMD>
__global__ void __launch_bounds__(kB128_Threads, 1) sm70_prefill_bc128_kernel(
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
    half   * kv_s   = reinterpret_cast<half   *>(smem_raw + kB128_QBytes);
    uint4  * p_s    = reinterpret_cast<uint4  *>(smem_raw + kB128_QBytes + kB128_KVBytes);
    float2 * stat_s = reinterpret_cast<float2 *>(smem_raw + kB128_QBytes + kB128_KVBytes + kB128_PBytes);
    float2 * sum_s  = reinterpret_cast<float2 *>(smem_raw + kB128_QBytes + kB128_KVBytes + kB128_PBytes + kB128_StatBytes);

    half * sk_buf0 = kv_s;
    half * sk_buf1 = kv_s + kB128_KBufElems;
    half * sv_buf0 = kv_s;
    half * sv_buf1 = kv_s + kB128_VBufElems;

    auto p_ptr    = [&](int rg, int sw, int strip) -> uint4 * { return p_s + (((rg * 4 + sw) * 4 + strip) * 32); };
    auto stat_ptr = [&](int rg, int sw) -> float2 * { return stat_s + ((rg * 4 + sw) * 32); };
    auto sum_ptr  = [&](int rg, int sw) -> float2 * { return sum_s  + ((rg * 4 + sw) * 32); };

    constexpr float Log2E   = 1.4426950408889634074f;
    constexpr unsigned Full = 0xffffffffu;

    const int q_block = (gridDim.x - 1) - blockIdx.x; // longest-job-first
    const int head_q  = blockIdx.y;
    const int kv_head = head_q / gqa;
    const int q0      = q_block * kB128_Br;
    const int q_rows  = min(kB128_Br, q_len - q0);

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int rg   = warp >> 2; // 0..1
    const int lw   = warp & 3;  // 0..3

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
    const int k_end = MaskMD ? live : min(live, kv_offset + q0 + q_rows);

    const half * k_head_ptr = k + (int64_t) kv_head * k_head_stride;
    const half * v_head_ptr = v + (int64_t) kv_head * v_head_stride;

    // Stage Q [q0..q0+q_rows) into q_s
    const float * q_head_ptr = q + (int64_t) head_q * q_head_stride;
    for (int chunk = tid; chunk < kB128_Br * (kD / 8); chunk += kB128_Threads) {
        const int r = chunk >> 5;
        const int d = (chunk & 31) << 3;
        half * dst  = &q_s[r * kB128_QStride + d];
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
    const int rg_causal_lim = kv_offset + row_base;

    const half2 * q_rg_h2 = reinterpret_cast<const half2 *>(q_s + rg * 32 * kB128_QStride);

    uint4 pf_k[4];
    if (k_end > 0) {
        b128_prefetch_k_chunk(pf_k, k_head_ptr, k_row_stride, 0, k_end, 0, tid);
    }
    __syncthreads();

    for (int k0 = 0; k0 < k_end; k0 += kB128_Bc) {
        const int k_rem = k_end - k0;

        float acc_s[4][8];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
#pragma unroll
            for (int l = 0; l < 8; ++l) { acc_s[s][l] = 0.0f; }
        }

        // ---- QK^T: 4 D-chunks of 64 dims, double-buffered in sk_buf0/sk_buf1 ----
        b128_commit_k_chunk(sk_buf0, pf_k, tid);
        __syncthreads();

#pragma unroll
        for (int dc = 0; dc < 4; ++dc) {
            if (dc + 1 < 4) {
                b128_prefetch_k_chunk(pf_k, k_head_ptr, k_row_stride, k0, k_rem, dc + 1, tid);
            }
            const half2 * sk_cur = reinterpret_cast<const half2 *>((dc & 1) ? sk_buf1 : sk_buf0)
                                 + lw * 32 * (kB128_KStride / 2);
            if (rows_here > 0) {
                constexpr int k_strip_stride = 8 * (kB128_KStride / 2);
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    half2 qf[4];
                    load_qp_128(qf, q_rg_h2 + (dc * 8 + c) * 4, kB128_QStride / 2);
                    const half2 * kcol = sk_cur + c * 4;
                    half2 kf0[4], kf1[4];
                    load_k_128(kf0, kcol + 0 * k_strip_stride, kB128_KStride / 2);
                    load_k_128(kf1, kcol + 1 * k_strip_stride, kB128_KStride / 2);
                    mma_qk_k0(acc_s[0], qf, kf0);
                    mma_qk_k1(acc_s[0], qf, kf0);
                    load_k_128(kf0, kcol + 2 * k_strip_stride, kB128_KStride / 2);
                    mma_qk_k0(acc_s[1], qf, kf1);
                    mma_qk_k1(acc_s[1], qf, kf1);
                    load_k_128(kf1, kcol + 3 * k_strip_stride, kB128_KStride / 2);
                    mma_qk_k0(acc_s[2], qf, kf0);
                    mma_qk_k1(acc_s[2], qf, kf0);
                    mma_qk_k0(acc_s[3], qf, kf1);
                    mma_qk_k1(acc_s[3], qf, kf1);
                }
            }
            if (dc + 1 < 4) {
                b128_commit_k_chunk(((dc + 1) & 1) ? sk_buf1 : sk_buf0, pf_k, tid);
                __syncthreads();
            }
        }

        // Prefetch V tile 0 while doing score masking & local max reduction
        uint4 pf_v[4];
        b128_prefetch_v_tile(pf_v, v_head_ptr, v_row_stride, k0, k_rem, tid);

        if (rows_here > 0) {
            const int sw_k0 = k0 + lw * 32;
            const bool unmasked_all = (!MaskMD) && (sw_k0 + 32 <= k_end) && (sw_k0 + 32 <= rg_causal_lim);
            float sm_lo = -CUDART_INF_F;
            float sm_hi = -CUDART_INF_F;
            if (unmasked_all) {
#pragma unroll
                for (int s = 0; s < 4; ++s) {
#pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        const bool r_ok = ((l & 2) != 0) ? r_hi_ok : r_lo_ok;
                        acc_s[s][l] = r_ok ? (acc_s[s][l] * scale) : -CUDART_INF_F;
                    }
                    sm_lo = fmaxf(sm_lo, fmaxf(fmaxf(acc_s[s][0], acc_s[s][1]), fmaxf(acc_s[s][4], acc_s[s][5])));
                    sm_hi = fmaxf(sm_hi, fmaxf(fmaxf(acc_s[s][2], acc_s[s][3]), fmaxf(acc_s[s][6], acc_s[s][7])));
                }
            } else {
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const int sub_k0 = sw_k0 + s * 8;
#pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        float val = acc_s[s][l];
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
                        acc_s[s][l] = val;
                    }
                    sm_lo = fmaxf(sm_lo, fmaxf(fmaxf(acc_s[s][0], acc_s[s][1]), fmaxf(acc_s[s][4], acc_s[s][5])));
                    sm_hi = fmaxf(sm_hi, fmaxf(fmaxf(acc_s[s][2], acc_s[s][3]), fmaxf(acc_s[s][6], acc_s[s][7])));
                }
            }
            sm_lo = fmaxf(sm_lo, __shfl_xor_sync(Full, sm_lo, 2, 32));
            sm_hi = fmaxf(sm_hi, __shfl_xor_sync(Full, sm_hi, 2, 32));
            stat_ptr(rg, lw)[lane] = make_float2(sm_lo, sm_hi);
        }
        __syncthreads();

        // Commit V tile 0 into sv_buf0 (sk_buf is dead now!)
        b128_commit_v_tile(sv_buf0, pf_v, tid);

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

            float bl_lo = 0.0f, bl_hi = 0.0f;
#pragma unroll
            for (int s = 0; s < 4; ++s) {
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const float nm = ((l & 2) == 0) ? nm_lo : nm_hi;
                    acc_s[s][l] = (nm > -CUDART_INF_F && acc_s[s][l] > -CUDART_INF_F)
                                      ? exp2f((acc_s[s][l] - nm) * Log2E) : 0.0f;
                }
                bl_lo += acc_s[s][0] + acc_s[s][1] + acc_s[s][4] + acc_s[s][5];
                bl_hi += acc_s[s][2] + acc_s[s][3] + acc_s[s][6] + acc_s[s][7];
                half2 p_my[4];
                softmax_to_half2(p_my, acc_s[s]);
                p_ptr(rg, lw, s)[lane] = *reinterpret_cast<const uint4 *>(p_my);
            }
            bl_lo += __shfl_xor_sync(Full, bl_lo, 2, 32);
            bl_hi += __shfl_xor_sync(Full, bl_hi, 2, 32);
            sum_ptr(rg, lw)[lane] = make_float2(bl_lo, bl_hi);
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
        }

        // ---- PV: 4 V-tiles of 32 rows x 256 dims, double-buffered in sv_buf0/sv_buf1 ----
#pragma unroll
        for (int vt = 0; vt < 4; ++vt) {
            const int v_off_next = (vt + 1) * 32;
            if (vt + 1 < 4) {
                if (v_off_next < k_rem) {
                    b128_prefetch_v_tile(pf_v, v_head_ptr, v_row_stride, k0 + v_off_next, k_rem - v_off_next, tid);
                }
            } else if (k_rem > kB128_Bc) {
                b128_prefetch_k_chunk(pf_k, k_head_ptr, k_row_stride, k0 + kB128_Bc, k_rem - kB128_Bc, 0, tid);
            }

            if (rows_here > 0 && vt * 32 < k_rem) {
                const half2 * v_slice = reinterpret_cast<const half2 *>((vt & 1) ? sv_buf1 : sv_buf0)
                                      + (lw * kDSlice) / 2;
                constexpr int v_sub_stride = 8 * (kB128_VStride / 2);
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    half2 p_sub[4];
                    *reinterpret_cast<uint4 *>(p_sub) = p_ptr(rg, vt, s)[lane];
                    const half2 * vsub = v_slice + s * v_sub_stride;
#pragma unroll
                    for (int c = 0; c < kDLocal; c += 2) {
                        half2 vf0[4], vf1[4];
                        load_v_64(vf0, vsub + (c + 0) * 4, kB128_VStride / 2);
                        load_v_64(vf1, vsub + (c + 1) * 4, kB128_VStride / 2);
                        mma_pv_f32_k0(acc_f[c + 0], p_sub, vf0);
                        mma_pv_f32_k0(acc_f[c + 1], p_sub, vf1);
                        mma_pv_f32_k1(acc_f[c + 0], p_sub, vf0);
                        mma_pv_f32_k1(acc_f[c + 1], p_sub, vf1);
                    }
                }
            }

            if (vt + 1 < 4) {
                if (v_off_next < k_rem) {
                    b128_commit_v_tile(((vt + 1) & 1) ? sv_buf1 : sv_buf0, pf_v, tid);
                }
                __syncthreads();
            } else {
                __syncthreads();
            }
        }
    }

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

template <bool MM>
static void run_bc128(
        dim3 grid, cudaStream_t stream,
        const float * q, int64_t q_row, int64_t q_head,
        const half * k, const half * v,
        int64_t k_row, int64_t k_head, int64_t v_row, int64_t v_head,
        float * out, int64_t o_row, int64_t o_head,
        int q_len, int kv_len, int gqa, float scale,
        const half * mask_last_q, int64_t mask_q_stride,
        const half * mask_kv, int64_t mask_kv_stride) {
    auto kfn = sm70_prefill_bc128_kernel<MM>;
    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kB128_SmemBytes));
        CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        attr_set = true;
    }
    kfn<<<grid, kB128_Threads, kB128_SmemBytes, stream>>>(
        q, q_row, q_head, k, v, k_row, k_head, v_row, v_head,
        out, o_row, o_head, q_len, kv_len, gqa, scale,
        mask_last_q, mask_q_stride, mask_kv, mask_kv_stride);
}

void launch_sm70_prefill_bc128(
        bool mask_md,
        cudaStream_t stream,
        const float * q, int64_t q_row, int64_t q_head,
        const half * k, const half * v,
        int64_t k_row, int64_t k_head, int64_t v_row, int64_t v_head,
        float * out, int64_t o_row, int64_t o_head,
        int q_len, int kv_len, int hq, int gqa, float scale,
        const half * mask_last_q, int64_t mask_q_stride,
        const half * mask_kv, int64_t mask_kv_stride) {
    const dim3 grid((unsigned) ((q_len + kB128_Br - 1) / kB128_Br), (unsigned) hq, 1);
    if (mask_md) {
        run_bc128<true>(grid, stream, q, q_row, q_head, k, v, k_row, k_head, v_row, v_head,
            out, o_row, o_head, q_len, kv_len, gqa, scale, mask_last_q, mask_q_stride, mask_kv, mask_kv_stride);
    } else {
        run_bc128<false>(grid, stream, q, q_row, q_head, k, v, k_row, k_head, v_row, v_head,
            out, o_row, o_head, q_len, kv_len, gqa, scale, mask_last_q, mask_q_stride, mask_kv, mask_kv_stride);
    }
}


// ============================================================================
// Stage 9 Prefill Br128_Bc64 v4 (Br=128, Bc=64, D=256, 8 warps = 256 threads, STACK:0)
//   - Constant-stride padded SMEM: sQ[128][264]=67,584 B, kv_s=9,216 B, stat_s+sum_s=4,096 B, p_s=16,384 B
//     Total = 97,280 B <= 98,304 B (96 KB exact V100 limit).
//   - Zero-cost double-buffering via dead-region aliasing:
//       * QK^T: sk_buf0 = kv_s (9,216 B), sk_buf1 = p_s (16,384 B, 100% dead during QK^T!) -> cuts QK barriers 7 -> 4!
//       * PV  : p_s is laid out sw-major: p_s[sw=0..3] (4,096 B each).
//               sv_buf0 = kv_s (8,448 B), sv_buf1 = stat_s (stat_s + sum_s + p_s[sw=0] = 8,192 B;
//               put 256 B pad before p_s -> [76,800..85,248) = 8,448 B, touching ONLY dead stat_s+sum_s+p_s[sw=0] after vt=0!)
//               -> cuts PV barriers 7 -> 5! Total barriers per Bc=64 tile: 16 -> 11 (-31%)!
//   - Exact (s - nm) * Log2E softmax for 100% greedy SHA parity across 12K/64K/100K.
// ============================================================================

constexpr int kB128x64_Br       = 128;
constexpr int kB128x64_Bc       = 64;
constexpr int kB128x64_Threads  = 256;
constexpr int kB128x64_QStride  = 264;
constexpr int kB128x64_KStride  = 72;
constexpr int kB128x64_VStride  = 264;
constexpr size_t kB128x64_QBytes    = 128 * kB128x64_QStride * sizeof(half); // 67,584 B
constexpr size_t kB128x64_KVBytes   = 64  * kB128x64_KStride * sizeof(half); //  9,216 B (>= 16*264*2 = 8,448 B)
constexpr size_t kB128x64_StatBytes = 2 * 2 * 4 * 16 * sizeof(float2);       //  2,048 B
constexpr size_t kB128x64_VPadBytes = 256;                                   // makes stat_s+sum_s+pad+p_s[sw=0] = 8,448 B exact!
constexpr size_t kB128x64_PBytes    = 4 * 2 * 2 * 2 * 32 * sizeof(uint4);    // 16,384 B (sw-major: 4,096 B per sw)
constexpr size_t kB128x64_SmemBytes = kB128x64_QBytes + kB128x64_KVBytes + 2 * kB128x64_StatBytes + kB128x64_VPadBytes + kB128x64_PBytes; // 97,536 B <= 98,304 B

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
static __device__ __forceinline__ void b128x64_prefetch_k(
        uint4 (&pf)[2], const half * __restrict__ k_head_ptr,
        int64_t k_row_stride, int k0, int k_rem, int dc, int tid) {
    const int r0 = tid >> 3;        // 0..31
    const int d0 = dc * 64 + ((tid & 7) << 3);
    const half * base = k_head_ptr + (int64_t) (k0 + r0) * k_row_stride + d0;
    const int64_t s32 = (int64_t) 32 * k_row_stride;
    if (k_rem >= 64) {
        pf[0] = ldg_cg_u4(reinterpret_cast<const uint4 *>(base));
        pf[1] = ldg_cg_u4(reinterpret_cast<const uint4 *>(base + s32));
    } else {
        pf[0] = (r0      < k_rem) ? ldg_cg_u4(reinterpret_cast<const uint4 *>(base))       : make_uint4(0u,0u,0u,0u);
        pf[1] = (r0 + 32 < k_rem) ? ldg_cg_u4(reinterpret_cast<const uint4 *>(base + s32)) : make_uint4(0u,0u,0u,0u);
    }
}

static __device__ __forceinline__ void b128x64_commit_k(half * __restrict__ sk_buf, const uint4 (&pf)[2], int tid) {
    const int r0 = tid >> 3;
    const int d0 = (tid & 7) << 3;
    uint4 * dst = reinterpret_cast<uint4 *>(&sk_buf[r0 * kB128x64_KStride + d0]);
    constexpr int s32 = (32 * kB128x64_KStride) / 8;
    dst[0]   = pf[0];
    dst[s32] = pf[1];
}

static __device__ __forceinline__ void b128x64_prefetch_v(
        uint4 (&pf)[2], const half * __restrict__ v_head_ptr,
        int64_t v_row_stride, int kv_base, int v_rem, int tid) {
    const int r0 = tid >> 5;        // 0..7
    const int d0 = (tid & 31) << 3; // 0..248
    const half * base = v_head_ptr + (int64_t) (kv_base + r0) * v_row_stride + d0;
    const int64_t s8 = (int64_t) 8 * v_row_stride;
    if (v_rem >= 16) {
        pf[0] = ldg_cg_u4(reinterpret_cast<const uint4 *>(base));
        pf[1] = ldg_cg_u4(reinterpret_cast<const uint4 *>(base + s8));
    } else {
        pf[0] = (r0     < v_rem) ? ldg_cg_u4(reinterpret_cast<const uint4 *>(base))      : make_uint4(0u,0u,0u,0u);
        pf[1] = (r0 + 8 < v_rem) ? ldg_cg_u4(reinterpret_cast<const uint4 *>(base + s8)) : make_uint4(0u,0u,0u,0u);
    }
}

static __device__ __forceinline__ void b128x64_commit_v(half * __restrict__ sv_buf, const uint4 (&pf)[2], int tid) {
    const int r0 = tid >> 5;
    const int d0 = (tid & 31) << 3;
    uint4 * dst = reinterpret_cast<uint4 *>(&sv_buf[r0 * kB128x64_VStride + d0]);
    constexpr int s8 = (8 * kB128x64_VStride) / 8;
    dst[0]  = pf[0];
    dst[s8] = pf[1];
}
#endif

template <bool MaskMD>
__global__ void __launch_bounds__(kB128x64_Threads, 1) sm70_prefill_br128_bc64_kernel(
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
    half   * kv_s   = reinterpret_cast<half   *>(smem_raw + kB128x64_QBytes);
    float2 * stat_s = reinterpret_cast<float2 *>(smem_raw + kB128x64_QBytes + kB128x64_KVBytes);
    float2 * sum_s  = reinterpret_cast<float2 *>(smem_raw + kB128x64_QBytes + kB128x64_KVBytes + kB128x64_StatBytes);
    uint4  * p_s    = reinterpret_cast<uint4  *>(smem_raw + kB128x64_QBytes + kB128x64_KVBytes + 2 * kB128x64_StatBytes + kB128x64_VPadBytes);

    // Dead-region double-buffer aliases:
    half * sk_buf0 = kv_s;
    half * sk_buf1 = reinterpret_cast<half *>(p_s);    // 16,384 B dead during QK^T!
    half * sv_buf0 = kv_s;
    half * sv_buf1 = reinterpret_cast<half *>(stat_s); // 8,448 B covers stat_s(2048)+sum_s(2048)+pad(256)+p_s[sw=0](4096) — all dead after vt=0!

    auto lane16   = [](int ln) -> int { return ((ln & ~3) >> 1) | (ln & 1); };
    // sw-major layout: sw=0 lives in first 4,096 B of p_s (dead right after vt=0!)
    auto p_ptr    = [&](int rg, int rh, int sw, int strip) -> uint4 * { return p_s + ((((sw * 2 + rg) * 2 + rh) * 2 + strip) * 32); };
    auto stat_ptr = [&](int rg, int rh, int sw) -> float2 * { return stat_s + (((rg * 2 + rh) * 4 + sw) * 16); };
    auto sum_ptr  = [&](int rg, int rh, int sw) -> float2 * { return sum_s  + (((rg * 2 + rh) * 4 + sw) * 16); };

    constexpr float Log2E   = 1.4426950408889634074f;
    constexpr unsigned Full = 0xffffffffu;

    const int q_block = (gridDim.x - 1) - blockIdx.x;
    const int head_q  = blockIdx.y;
    const int kv_head = head_q / gqa;
    const int q0      = q_block * kB128x64_Br;
    const int q_rows  = min(kB128x64_Br, q_len - q0);

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int ln16 = lane16(lane);
    const int rg   = warp >> 2;
    const int lw   = warp & 3;

    const int row_base0  = q0 + rg * 64;
    const int row_base1  = row_base0 + 32;
    const int rows_here0 = max(0, min(32, q_len - row_base0));
    const int rows_here1 = max(0, min(32, q_len - row_base1));

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
    const int k_end = MaskMD ? live : min(live, kv_offset + q0 + q_rows);

    const half * k_head_ptr = k + (int64_t) kv_head * k_head_stride;
    const half * v_head_ptr = v + (int64_t) kv_head * v_head_stride;

    // Stage Q [q0..q0+q_rows) into q_s[128][264]
    const float * q_head_ptr = q + (int64_t) head_q * q_head_stride;
    for (int chunk = tid; chunk < kB128x64_Br * 32; chunk += kB128x64_Threads) {
        const int r = chunk >> 5;
        const int d = (chunk & 31) << 3;
        half * dst  = &q_s[r * kB128x64_QStride + d];
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

    float m_lo[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float m_hi[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float l_lo[2] = {0.0f, 0.0f};
    float l_hi[2] = {0.0f, 0.0f};
    float acc_f[2][kDLocal][8];
#pragma unroll
    for (int rh = 0; rh < 2; ++rh) {
#pragma unroll
        for (int c = 0; c < kDLocal; ++c) {
#pragma unroll
            for (int j = 0; j < 8; ++j) { acc_f[rh][c][j] = 0.0f; }
        }
    }

    const int row_lo = d_get_i(0);
    const int row_hi = d_get_i(2);
    const int tok_lo[2] = { row_base0 + row_lo, row_base1 + row_lo };
    const int tok_hi[2] = { row_base0 + row_hi, row_base1 + row_hi };
    const bool r_lo_ok[2] = { row_lo < rows_here0, row_lo < rows_here1 };
    const bool r_hi_ok[2] = { row_hi < rows_here0, row_hi < rows_here1 };
    const int rg_causal_lim = kv_offset + row_base0;

    // Precompute per-lane constant SMEM pointers for BOTH double-buffers!
    const uint4 * q0_lane   = reinterpret_cast<const uint4 *>(q_s + (rg * 64 + lane) * kB128x64_QStride);
    const uint4 * q1_lane   = reinterpret_cast<const uint4 *>(q_s + (rg * 64 + 32 + lane) * kB128x64_QStride);
    const int k_lane_off0   = (lw * 16 + k_get_i()) * kB128x64_KStride;
    const int k_lane_off1   = (lw * 16 + 8 + k_get_i()) * kB128x64_KStride;
    const uint4 * k0_buf0   = reinterpret_cast<const uint4 *>(sk_buf0 + k_lane_off0);
    const uint4 * k1_buf0   = reinterpret_cast<const uint4 *>(sk_buf0 + k_lane_off1);
    const uint4 * k0_buf1   = reinterpret_cast<const uint4 *>(sk_buf1 + k_lane_off0);
    const uint4 * k1_buf1   = reinterpret_cast<const uint4 *>(sk_buf1 + k_lane_off1);
    const int v_i0          = lane & 3;
    const int v_j0          = (lane >> 4) << 2;
    const int v_lane_off0   = v_i0 * kB128x64_VStride + lw * kDSlice + v_j0;
    const int v_lane_off1   = (v_i0 + 4) * kB128x64_VStride + lw * kDSlice + v_j0;
    const uint2 * v0_buf0   = reinterpret_cast<const uint2 *>(sv_buf0 + v_lane_off0);
    const uint2 * v1_buf0   = reinterpret_cast<const uint2 *>(sv_buf0 + v_lane_off1);
    const uint2 * v0_buf1   = reinterpret_cast<const uint2 *>(sv_buf1 + v_lane_off0);
    const uint2 * v1_buf1   = reinterpret_cast<const uint2 *>(sv_buf1 + v_lane_off1);
    constexpr int v_s8_u2   = (8 * kB128x64_VStride) / 4;

    uint4 pf_kv[2];
    if (k_end > 0) {
        b128x64_prefetch_k(pf_kv, k_head_ptr, k_row_stride, 0, k_end, 0, tid);
    }
    __syncthreads();

    for (int k0 = 0; k0 < k_end; k0 += kB128x64_Bc) {
        const int k_rem = k_end - k0;

        float acc_s[2][2][8];
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
#pragma unroll
            for (int s = 0; s < 2; ++s) {
#pragma unroll
                for (int l = 0; l < 8; ++l) { acc_s[rh][s][l] = 0.0f; }
            }
        }

        // ---- QK^T: DOUBLE-BUFFERED sk_buf0 / sk_buf1 (ONLY 4 barriers total!) ----
        b128x64_commit_k(sk_buf0, pf_kv, tid);
        __syncthreads();

#pragma unroll
        for (int dc = 0; dc < 4; ++dc) {
            if (dc + 1 < 4) {
                b128x64_prefetch_k(pf_kv, k_head_ptr, k_row_stride, k0, k_rem, dc + 1, tid);
            } else {
                b128x64_prefetch_v(pf_kv, v_head_ptr, v_row_stride, k0, k_rem, tid);
            }
            const uint4 * k0_lane = (dc & 1) ? k0_buf1 : k0_buf0;
            const uint4 * k1_lane = (dc & 1) ? k1_buf1 : k1_buf0;
            if (rows_here0 > 0) {
                half2 qf0[4], qf1[4], kf0[4], kf1[4];
                *reinterpret_cast<uint4 *>(qf0) = q0_lane[dc * 8 + 0];
                *reinterpret_cast<uint4 *>(qf1) = q1_lane[dc * 8 + 0];
                *reinterpret_cast<uint4 *>(kf0) = k0_lane[0];
                *reinterpret_cast<uint4 *>(kf1) = k1_lane[0];
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    mma_qk_k0(acc_s[0][0], qf0, kf0);
                    mma_qk_k0(acc_s[1][0], qf1, kf0);
                    mma_qk_k0(acc_s[0][1], qf0, kf1);
                    mma_qk_k0(acc_s[1][1], qf1, kf1);
                    mma_qk_k1(acc_s[0][0], qf0, kf0);
                    mma_qk_k1(acc_s[1][0], qf1, kf0);
                    if (c + 1 < 8) {
                        *reinterpret_cast<uint4 *>(kf0) = k0_lane[c + 1];
                    }
                    mma_qk_k1(acc_s[0][1], qf0, kf1);
                    if (c + 1 < 8) {
                        *reinterpret_cast<uint4 *>(qf0) = q0_lane[dc * 8 + c + 1];
                    }
                    mma_qk_k1(acc_s[1][1], qf1, kf1);
                    if (c + 1 < 8) {
                        *reinterpret_cast<uint4 *>(qf1) = q1_lane[dc * 8 + c + 1];
                        *reinterpret_cast<uint4 *>(kf1) = k1_lane[c + 1];
                    }
                }
            }
            if (dc + 1 < 4) {
                b128x64_commit_k(((dc + 1) & 1) ? sk_buf1 : sk_buf0, pf_kv, tid);
                __syncthreads();
            }
        }

        if (rows_here0 > 0) {
            const int sw_k0 = k0 + lw * 16;
            const bool unmasked_all = (!MaskMD) && (sw_k0 + 16 <= k_end) && (sw_k0 + 16 <= rg_causal_lim);
#pragma unroll
            for (int rh = 0; rh < 2; ++rh) {
                float sm_lo = -CUDART_INF_F;
                float sm_hi = -CUDART_INF_F;
                if (unmasked_all) {
#pragma unroll
                    for (int s = 0; s < 2; ++s) {
#pragma unroll
                        for (int l = 0; l < 8; ++l) {
                            const bool r_ok = ((l & 2) != 0) ? r_hi_ok[rh] : r_lo_ok[rh];
                            acc_s[rh][s][l] = r_ok ? (acc_s[rh][s][l] * scale) : -CUDART_INF_F;
                        }
                        sm_lo = fmaxf(sm_lo, fmaxf(fmaxf(acc_s[rh][s][0], acc_s[rh][s][1]), fmaxf(acc_s[rh][s][4], acc_s[rh][s][5])));
                        sm_hi = fmaxf(sm_hi, fmaxf(fmaxf(acc_s[rh][s][2], acc_s[rh][s][3]), fmaxf(acc_s[rh][s][6], acc_s[rh][s][7])));
                    }
                } else {
#pragma unroll
                    for (int s = 0; s < 2; ++s) {
                        const int sub_k0 = sw_k0 + s * 8;
#pragma unroll
                        for (int l = 0; l < 8; ++l) {
                            float val = acc_s[rh][s][l];
                            const bool is_hi = (l & 2) != 0;
                            const bool r_ok  = is_hi ? r_hi_ok[rh] : r_lo_ok[rh];
                            const int  key   = sub_k0 + d_get_j(l);
                            if (r_ok && key < k_end) {
                                const int tok = is_hi ? tok_hi[rh] : tok_lo[rh];
                                if (MaskMD || key >= kv_offset) {
                                    const float mv = __half2float(mask_kv[(int64_t) tok * mask_kv_stride + key]);
                                    val = (mv > -30000.0f) ? (val * scale + mv) : -CUDART_INF_F;
                                } else {
                                    val *= scale;
                                }
                            } else {
                                val = -CUDART_INF_F;
                            }
                            acc_s[rh][s][l] = val;
                        }
                        sm_lo = fmaxf(sm_lo, fmaxf(fmaxf(acc_s[rh][s][0], acc_s[rh][s][1]), fmaxf(acc_s[rh][s][4], acc_s[rh][s][5])));
                        sm_hi = fmaxf(sm_hi, fmaxf(fmaxf(acc_s[rh][s][2], acc_s[rh][s][3]), fmaxf(acc_s[rh][s][6], acc_s[rh][s][7])));
                    }
                }
                sm_lo = fmaxf(sm_lo, __shfl_xor_sync(Full, sm_lo, 2, 32));
                sm_hi = fmaxf(sm_hi, __shfl_xor_sync(Full, sm_hi, 2, 32));
                if ((lane & 2) == 0) {
                    stat_ptr(rg, rh, lw)[ln16] = make_float2(sm_lo, sm_hi);
                }
            }
        }
        // Commit V tile 0 into sv_buf0 (sk_buf0 was last read in dc=2, ALREADY synced before dc=3!)
        b128x64_commit_v(sv_buf0, pf_kv, tid);
        __syncthreads();

        float a_lo[2] = {1.0f, 1.0f};
        float a_hi[2] = {1.0f, 1.0f};
        if (rows_here0 > 0) {
#pragma unroll
            for (int rh = 0; rh < 2; ++rh) {
                const float2 sm0 = stat_ptr(rg, rh, 0)[ln16];
                const float2 sm1 = stat_ptr(rg, rh, 1)[ln16];
                const float2 sm2 = stat_ptr(rg, rh, 2)[ln16];
                const float2 sm3 = stat_ptr(rg, rh, 3)[ln16];
                const float bm_lo = fmaxf(fmaxf(sm0.x, sm1.x), fmaxf(sm2.x, sm3.x));
                const float bm_hi = fmaxf(fmaxf(sm0.y, sm1.y), fmaxf(sm2.y, sm3.y));
                const float nm_lo = fmaxf(m_lo[rh], bm_lo);
                const float nm_hi = fmaxf(m_hi[rh], bm_hi);
                a_lo[rh] = (m_lo[rh] == -CUDART_INF_F) ? 0.0f : ((nm_lo == m_lo[rh]) ? 1.0f : exp2f((m_lo[rh] - nm_lo) * Log2E));
                a_hi[rh] = (m_hi[rh] == -CUDART_INF_F) ? 0.0f : ((nm_hi == m_hi[rh]) ? 1.0f : exp2f((m_hi[rh] - nm_hi) * Log2E));
                m_lo[rh] = nm_lo;
                m_hi[rh] = nm_hi;

                float bl_lo = 0.0f, bl_hi = 0.0f;
#pragma unroll
                for (int s = 0; s < 2; ++s) {
#pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        const float nm = ((l & 2) == 0) ? nm_lo : nm_hi;
                        acc_s[rh][s][l] = (nm > -CUDART_INF_F && acc_s[rh][s][l] > -CUDART_INF_F)
                                          ? exp2f((acc_s[rh][s][l] - nm) * Log2E) : 0.0f;
                    }
                    bl_lo += acc_s[rh][s][0] + acc_s[rh][s][1] + acc_s[rh][s][4] + acc_s[rh][s][5];
                    bl_hi += acc_s[rh][s][2] + acc_s[rh][s][3] + acc_s[rh][s][6] + acc_s[rh][s][7];
                    half2 p_my[4];
                    softmax_to_half2(p_my, acc_s[rh][s]);
                    p_ptr(rg, rh, lw, s)[lane] = *reinterpret_cast<const uint4 *>(p_my);
                }
                bl_lo += __shfl_xor_sync(Full, bl_lo, 2, 32);
                bl_hi += __shfl_xor_sync(Full, bl_hi, 2, 32);
                if ((lane & 2) == 0) {
                    sum_ptr(rg, rh, lw)[ln16] = make_float2(bl_lo, bl_hi);
                }
            }
        }
        __syncthreads();

        if (rows_here0 > 0) {
#pragma unroll
            for (int rh = 0; rh < 2; ++rh) {
                const float2 s0 = sum_ptr(rg, rh, 0)[ln16];
                const float2 s1 = sum_ptr(rg, rh, 1)[ln16];
                const float2 s2 = sum_ptr(rg, rh, 2)[ln16];
                const float2 s3 = sum_ptr(rg, rh, 3)[ln16];
                l_lo[rh] = l_lo[rh] * a_lo[rh] + (s0.x + s1.x + s2.x + s3.x);
                l_hi[rh] = l_hi[rh] * a_hi[rh] + (s0.y + s1.y + s2.y + s3.y);
                if (a_lo[rh] != 1.0f || a_hi[rh] != 1.0f) {
                    const float al = a_lo[rh], ah = a_hi[rh];
#pragma unroll
                    for (int c = 0; c < kDLocal; ++c) {
                        acc_f[rh][c][0] *= al; acc_f[rh][c][1] *= al; acc_f[rh][c][4] *= al; acc_f[rh][c][5] *= al;
                        acc_f[rh][c][2] *= ah; acc_f[rh][c][3] *= ah; acc_f[rh][c][6] *= ah; acc_f[rh][c][7] *= ah;
                    }
                }
            }
        }

        // ---- PV: DOUBLE-BUFFERED sv_buf0 / sv_buf1 (ONLY 5 barriers total!) ----
#pragma unroll
        for (int vt = 0; vt < 4; ++vt) {
            const int v_off_next = (vt + 1) * 16;
            if (vt + 1 < 4) {
                if (v_off_next < k_rem) {
                    b128x64_prefetch_v(pf_kv, v_head_ptr, v_row_stride, k0 + v_off_next, k_rem - v_off_next, tid);
                }
            } else if (k_rem > kB128x64_Bc) {
                b128x64_prefetch_k(pf_kv, k_head_ptr, k_row_stride, k0 + kB128x64_Bc, k_rem - kB128x64_Bc, 0, tid);
            }

            if (rows_here0 > 0 && vt * 16 < k_rem) {
                const uint2 * vb0 = (vt & 1) ? v0_buf1 : v0_buf0;
                const uint2 * vb1 = (vt & 1) ? v1_buf1 : v1_buf0;
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    half2 p0[4], p1[4];
                    *reinterpret_cast<uint4 *>(p0) = p_ptr(rg, 0, vt, s)[lane];
                    *reinterpret_cast<uint4 *>(p1) = p_ptr(rg, 1, vt, s)[lane];
                    const uint2 * vl0 = vb0 + s * v_s8_u2;
                    const uint2 * vl1 = vb1 + s * v_s8_u2;
                    half2 vfa[4], vfb[4], vfc[4], vfd[4];
                    *reinterpret_cast<uint2 *>(&vfa[0]) = vl0[0];
                    *reinterpret_cast<uint2 *>(&vfa[2]) = vl1[0];
                    *reinterpret_cast<uint2 *>(&vfb[0]) = vl0[2];
                    *reinterpret_cast<uint2 *>(&vfb[2]) = vl1[2];
#pragma unroll
                    for (int c = 0; c < kDLocal; c += 2) {
                        half2 (&cur0)[4] = ((c & 2) == 0) ? vfa : vfc;
                        half2 (&cur1)[4] = ((c & 2) == 0) ? vfb : vfd;
                        half2 (&nxt0)[4] = ((c & 2) == 0) ? vfc : vfa;
                        half2 (&nxt1)[4] = ((c & 2) == 0) ? vfd : vfb;
                        if (c + 2 < kDLocal) {
                            *reinterpret_cast<uint2 *>(&nxt0[0]) = vl0[(c + 2) * 2];
                            *reinterpret_cast<uint2 *>(&nxt0[2]) = vl1[(c + 2) * 2];
                        }
                        mma_pv_f32_k0(acc_f[0][c + 0], p0, cur0);
                        mma_pv_f32_k0(acc_f[1][c + 0], p1, cur0);
                        mma_pv_f32_k0(acc_f[0][c + 1], p0, cur1);
                        mma_pv_f32_k0(acc_f[1][c + 1], p1, cur1);
                        if (c + 2 < kDLocal) {
                            *reinterpret_cast<uint2 *>(&nxt1[0]) = vl0[(c + 3) * 2];
                            *reinterpret_cast<uint2 *>(&nxt1[2]) = vl1[(c + 3) * 2];
                        }
                        mma_pv_f32_k1(acc_f[0][c + 0], p0, cur0);
                        mma_pv_f32_k1(acc_f[1][c + 0], p1, cur0);
                        mma_pv_f32_k1(acc_f[0][c + 1], p0, cur1);
                        mma_pv_f32_k1(acc_f[1][c + 1], p1, cur1);
                    }
                }
            }

            if (vt == 0) {
                // Must sync before committing sv_buf1 (aliases sum_s + p_s[sw=0] which warps just finished reading!)
                __syncthreads();
                if (v_off_next < k_rem) {
                    b128x64_commit_v(sv_buf1, pf_kv, tid);
                }
                __syncthreads();
            } else if (vt + 1 < 4) {
                // Alternate buffer was last read 2 stages ago (already synced!) -> commit directly, 1 sync!
                if (v_off_next < k_rem) {
                    b128x64_commit_v(((vt + 1) & 1) ? sv_buf1 : sv_buf0, pf_kv, tid);
                }
                __syncthreads();
            } else {
                // vt==3 read sv_buf1; sk_buf0=kv_s is already free! Wait: sync ensures sv_buf1 reads done before next k0!
                __syncthreads();
            }
        }
    }

    if (rows_here0 > 0) {
        const int d_base = lw * kDSlice;
        float * out_head = out + (int64_t) head_q * o_head_stride + d_base;
        const int j0 = d_get_j(0), j1 = d_get_j(1), j4 = d_get_j(4), j5 = d_get_j(5);
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
            const float inv_lo = (l_lo[rh] > 0.0f) ? (1.0f / l_lo[rh]) : 0.0f;
            const float inv_hi = (l_hi[rh] > 0.0f) ? (1.0f / l_hi[rh]) : 0.0f;
            float * out0 = r_lo_ok[rh] ? (out_head + (int64_t) tok_lo[rh] * o_row_stride) : nullptr;
            float * out1 = r_hi_ok[rh] ? (out_head + (int64_t) tok_hi[rh] * o_row_stride) : nullptr;
#pragma unroll
            for (int c = 0; c < kDLocal; ++c) {
                const int co = c * 8;
                if (out0 != nullptr) {
                    out0[co + j0] = acc_f[rh][c][0] * inv_lo;
                    out0[co + j1] = acc_f[rh][c][1] * inv_lo;
                    out0[co + j4] = acc_f[rh][c][4] * inv_lo;
                    out0[co + j5] = acc_f[rh][c][5] * inv_lo;
                }
                if (out1 != nullptr) {
                    out1[co + j0] = acc_f[rh][c][2] * inv_hi;
                    out1[co + j1] = acc_f[rh][c][3] * inv_hi;
                    out1[co + j4] = acc_f[rh][c][6] * inv_hi;
                    out1[co + j5] = acc_f[rh][c][7] * inv_hi;
                }
            }
        }
    }
#endif
}

template <bool MM>
static void run_br128_bc64(
        dim3 grid, cudaStream_t stream,
        const float * q, int64_t q_row, int64_t q_head,
        const half * k, const half * v,
        int64_t k_row, int64_t k_head, int64_t v_row, int64_t v_head,
        float * out, int64_t o_row, int64_t o_head,
        int q_len, int kv_len, int gqa, float scale,
        const half * mask_last_q, int64_t mask_q_stride,
        const half * mask_kv, int64_t mask_kv_stride) {
    auto kfn = sm70_prefill_br128_bc64_kernel<MM>;
    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kB128x64_SmemBytes));
        CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        attr_set = true;
    }
    kfn<<<grid, kB128x64_Threads, kB128x64_SmemBytes, stream>>>(
        q, q_row, q_head, k, v, k_row, k_head, v_row, v_head,
        out, o_row, o_head, q_len, kv_len, gqa, scale,
        mask_last_q, mask_q_stride, mask_kv, mask_kv_stride);
}

void launch_sm70_prefill_br128_bc128(
        bool mask_md,
        cudaStream_t stream,
        const float * q, int64_t q_row, int64_t q_head,
        const half * k, const half * v,
        int64_t k_row, int64_t k_head, int64_t v_row, int64_t v_head,
        float * out, int64_t o_row, int64_t o_head,
        int q_len, int kv_len, int hq, int gqa, float scale,
        const half * mask_last_q, int64_t mask_q_stride,
        const half * mask_kv, int64_t mask_kv_stride) {
    const dim3 grid((unsigned) ((q_len + kB128x64_Br - 1) / kB128x64_Br), (unsigned) hq, 1);
    if (mask_md) {
        run_br128_bc64<true>(grid, stream, q, q_row, q_head, k, v, k_row, k_head, v_row, v_head,
            out, o_row, o_head, q_len, kv_len, gqa, scale, mask_last_q, mask_q_stride, mask_kv, mask_kv_stride);
    } else {
        run_br128_bc64<false>(grid, stream, q, q_row, q_head, k, v, k_row, k_head, v_row, v_head,
            out, o_row, o_head, q_len, kv_len, gqa, scale, mask_last_q, mask_q_stride, mask_kv, mask_kv_stride);
    }
}

} // namespace sm70_prefill96
