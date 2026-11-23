// Single-pass Br=96, Bc=32, 12-warp Volta HMMA.884 FlashAttention decode kernel v4 (v100fast).
// Breakthrough optimizations over v1-v3:
//   1. Direct FP32 Tensor-Core PV (mma.sync.m8n8k4.row.row.f32.f16.f16.f32) straight into acc_f:
//      ZERO H2F conversions, ZERO acc+contrib adds, ZERO pv temp regs, skips rescale when alpha==1.0!
//   2. 4x split softmax across the 4 warps of each row-group:
//      Each warp computes ONLY 8 exp2f + 8 F2F (for its own sub=lw) instead of 32 -> 4x less SFU, 9x less F2F!
//      Publishes p_sub[4] to p_s in smem (zero p[4][4] reg pressure, 87 KB smem < 96 KB).
//   3. Zero-ALU async q8_0 KvPrefetch: raw uint16_t loads only before QK^T/PV, deferring all
//      XOR+PRMT+HADD2+HMUL2 to commit after HMMA finishes -> 100% HBM2 overlap!
//   4. 3 barriers per 32 keys (down from 4) + 2x HMMA ILP in QK^T & PV + LDS.128 Q/K/P + LDS.64 V.
#include "fattn-sm70-q8-96.cuh"
#include <math_constants.h>

namespace sm70_fa96 {

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

static __device__ __forceinline__ void dequant8_q4_local(
        const uint8_t * __restrict__ row_base, int d, uint4 & out_u4) {
    const uint8_t * blk = row_base + (d >> 5) * 18;
    const float s = __half2float(*reinterpret_cast<const half *>(blk));
    const int off = 2 + (d & 15);
    uint64_t b8 = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        b8 |= (uint64_t) *reinterpret_cast<const uint16_t *>(blk + off + 2 * k) << (16 * k);
    }
    const bool low = (d & 16) == 0;
    half h[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint8_t byte = (uint8_t) ((b8 >> (8 * j)) & 0xFF);
        const int q = low ? (byte & 0xF) : (byte >> 4);
        h[j] = __float2half_rn(((float) q - 8.0f) * s);
    }
    out_u4 = *reinterpret_cast<const uint4 *>(h);
}

// Zero-ALU async prefetch structs: NO dependent ALU on loaded words during prefetch!
struct KvPrefetchF16 {
    uint4 r0, r1, r2;
};
struct KvPrefetchQ8 {
    uint16_t s0, s1, s2;
    uint32_t a0, b0, c0;
    uint32_t a1, b1, c1;
    uint32_t a2, b2, c2;
};

template <int KVMode>
struct KvPrefetch {
    KvPrefetchF16 f;
};
template <>
struct KvPrefetch<2> {
    KvPrefetchQ8 q;
};

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
            if (valid) {
                if constexpr (KVMode == 1) { dequant8_q4_local(p, d0, r); }
                else { r = *reinterpret_cast<const uint4 *>(p); }
            } else { r = make_uint4(0u, 0u, 0u, 0u); }
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
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3]), "+r"(D[4]), "+r"(D[5]), "+r"(D[6]), "+r"(D[7])
        : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]));
}

static __device__ __forceinline__ void mma_qk_k1(float (&d)[8], const half2 (&q)[4], const half2 (&k)[4]) {
    const int * A = reinterpret_cast<const int *>(q);
    const int * B = reinterpret_cast<const int *>(k);
    int * D       = reinterpret_cast<int *>(d);
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3]), "+r"(D[4]), "+r"(D[5]), "+r"(D[6]), "+r"(D[7])
        : "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]));
}

// Direct FP32 PV accumulation straight into acc_f[8] -- ZERO H2F, ZERO temp pv regs!
static __device__ __forceinline__ void mma_pv_f32_k0(float (&d)[8], const half2 (&p)[4], const half2 (&v)[4]) {
    const int * P = reinterpret_cast<const int *>(p);
    const int * V = reinterpret_cast<const int *>(v);
    int * D       = reinterpret_cast<int *>(d);
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3]), "+r"(D[4]), "+r"(D[5]), "+r"(D[6]), "+r"(D[7])
        : "r"(P[0]), "r"(P[1]), "r"(V[0]), "r"(V[1]));
}

static __device__ __forceinline__ void mma_pv_f32_k1(float (&d)[8], const half2 (&p)[4], const half2 (&v)[4]) {
    const int * P = reinterpret_cast<const int *>(p);
    const int * V = reinterpret_cast<const int *>(v);
    int * D       = reinterpret_cast<int *>(d);
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3]), "+r"(D[4]), "+r"(D[5]), "+r"(D[6]), "+r"(D[7])
        : "r"(P[2]), "r"(P[3]), "r"(V[2]), "r"(V[3]));
}

#endif // __CUDA_ARCH__ == 700

template <int GQA, int KMode, int VMode, bool MaskMD>
__global__ void __launch_bounds__(kThreads, 1) sm70_fa96_decode_kernel(
        const float * __restrict__ q,
        int64_t q_row_stride, int64_t q_head_stride,
        const half * __restrict__ k, const half * __restrict__ v,
        int64_t k_row_stride, int64_t k_head_stride,
        int64_t v_row_stride, int64_t v_head_stride,
        int q_len, int kv_len, float scale, int splits,
        const half * __restrict__ mask_last_q, int64_t mask_q_stride,
        float * __restrict__ partial_acc, float * __restrict__ partial_m,
        float * __restrict__ partial_l,
        const half * __restrict__ mask_kv, int64_t mask_kv_stride) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
    extern __shared__ char smem_raw[];
    half   * q_s      = reinterpret_cast<half   *>(smem_raw);
    half   * kv_s     = reinterpret_cast<half   *>(smem_raw + kQBytes);
    float  * scores_s = reinterpret_cast<float  *>(smem_raw + kQBytes + kKVBytes);
    uint4  * p_s      = reinterpret_cast<uint4  *>(smem_raw + kQBytes + kKVBytes + kScoresBytes);
    float2 * stat_s   = reinterpret_cast<float2 *>(smem_raw + kQBytes + kKVBytes + kScoresBytes + kPBytes);

    auto score_ptr = [&](int rg, int sub, int j) -> float * {
        return scores_s + (((rg * 4 + sub) * 8 + j) * 32);
    };
    auto p_ptr = [&](int rg, int sub) -> uint4 * {
        return p_s + ((rg * 4 + sub) * 32);
    };
    auto stat_ptr = [&](int rg, int sub) -> float2 * {
        return stat_s + ((rg * 4 + sub) * 32);
    };

    constexpr float Log2E   = 1.4426950408889634074f;
    constexpr unsigned Full = 0xffffffffu;

    const int rows_total   = q_len * GQA;
    const int n_row_passes = (rows_total + kBr - 1) / kBr;
    const int kv_head      = blockIdx.x / n_row_passes;
    const int row_pass     = blockIdx.x % n_row_passes;
    const int pass_base    = row_pass * kBr;
    const int hkv          = gridDim.x / n_row_passes;
    const int split        = blockIdx.y;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int rg   = warp >> 2;
    const int lw   = warp & 3;

    const int row_base  = pass_base + rg * 32;
    const int rows_here = max(0, min(32, rows_total - row_base));
    const int pass_rows = min(kBr, rows_total - pass_base);

    int live = 0;
    if constexpr (MaskMD) {
        live = kv_len;
    } else {
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

    const int live_tiles      = (live + kBc - 1) / kBc;
    const int tiles_per_split = (live_tiles + splits - 1) / splits;
    const int k_start         = min(split * tiles_per_split * kBc, live);
    const int k_end           = min(k_start + tiles_per_split * kBc, live);

    const int64_t k_head_off = (int64_t) kv_head * k_head_stride;
    const int64_t v_head_off = (int64_t) kv_head * v_head_stride;

    // Stage Q into q_s
    for (int chunk = tid; chunk < kBr * (kD / 8); chunk += kThreads) {
        const int r = chunk >> 5;
        const int d = (chunk & 31) << 3;
        half * dst  = &q_s[r * kSmemStride + d];
        if (r < pass_rows) {
            const int r_gl   = pass_base + r;
            const int token  = r_gl / GQA;
            const int c      = r_gl - token * GQA;
            const int head_q = kv_head * GQA + c;
            const float * src = q + d + (int64_t) token * q_row_stride
                                      + (int64_t) head_q * q_head_stride;
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
    const int tok_lo = (row_base + row_lo) / GQA;
    const int tok_hi = (row_base + row_hi) / GQA;
    const bool r_lo_ok = (row_lo < rows_here);
    const bool r_hi_ok = (row_hi < rows_here);

    const half2 * q_rg_h2 = reinterpret_cast<const half2 *>(q_s + rg * 32 * kSmemStride);
    const half2 * kv_h2   = reinterpret_cast<const half2 *>(kv_s);

    const int64_t k_rstride_b = (KMode == 0) ? (k_row_stride * (int64_t) sizeof(half)) : k_row_stride;
    const int64_t v_rstride_b = (VMode == 0) ? (v_row_stride * (int64_t) sizeof(half)) : v_row_stride;
    const int d0_init = lane << 3;
    const int64_t k_extra = (KMode == 2) ? ((d0_init >> 5) * 34) : ((KMode == 0) ? (d0_init * (int64_t) sizeof(half)) : 0);
    const int64_t v_extra = (VMode == 2) ? ((d0_init >> 5) * 34) : ((VMode == 0) ? (d0_init * (int64_t) sizeof(half)) : 0);
    const int64_t k_hoff_b = (KMode == 0) ? (k_head_off * (int64_t) sizeof(half)) : k_head_off;
    const int64_t v_hoff_b = (VMode == 0) ? (v_head_off * (int64_t) sizeof(half)) : v_head_off;
    const uint8_t * k_ptr = reinterpret_cast<const uint8_t *>(k) + (int64_t) (k_start + warp) * k_rstride_b + k_hoff_b + k_extra;
    const uint8_t * v_ptr = reinterpret_cast<const uint8_t *>(v) + (int64_t) (k_start + warp) * v_rstride_b + v_hoff_b + v_extra;

    // Prologue: prefetch first K tile while Q stages!
    KvPrefetch<KMode> pf_k;
    if (k_start < k_end) {
        prefetch_kv_tile<KMode>(pf_k, k_ptr, k_rstride_b, k_end - k_start, tid);
    }
    __syncthreads();

    for (int k0 = k_start; k0 < k_end; k0 += kBc) {
        const int k_rem = k_end - k0;
        // 1) Commit K into kv_s + start zero-ALU async prefetch of V for THIS tile!
        commit_kv_tile<KMode>(kv_s, pf_k, tid);
        KvPrefetch<VMode> pf_v;
        prefetch_kv_tile<VMode>(pf_v, v_ptr, v_rstride_b, k_rem, tid);
        __syncthreads();

        // 2) QK^T (2x HMMA ILP) -> scale+mask ONCE -> keep my_s[8] in regs + publish sub_max (2 floats) to stat_s!
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
            if ((!MaskMD) && (sub_k0 + 8 <= k_end) && (sub_k0 + 8 <= kv_offset)) {
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

        // 3) Commit V into kv_s + 4x-split softmax (each warp computes ONLY its 8 exp2f & publishes p_s!)
        //    + start zero-ALU async prefetch of NEXT tile's K!
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

            // Only 8 exp2f + 8 F2F per warp (4x reduction!), keeping my_s[8] in regs (zero scores_s traffic!)
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
            // Store my sub's sum into scores_s[rg][lw][0..1][lane] (disjoint from stat_s!)
            score_ptr(rg, lw, 0)[lane] = bl_lo;
            score_ptr(rg, lw, 1)[lane] = bl_hi;

            half2 p_my[4];
            softmax_to_half2(p_my, my_s);
            p_ptr(rg, lw)[lane] = *reinterpret_cast<const uint4 *>(p_my);
        }
        __syncthreads();

        // 4) Direct FP32 PV straight into acc_f (ZERO H2F, 2x HMMA ILP, LDS.128 P + LDS.64 V)!
        if (rows_here > 0) {
            if (lw == 0) {
                const float bl_lo = score_ptr(rg, 0, 0)[lane] + score_ptr(rg, 1, 0)[lane]
                                  + score_ptr(rg, 2, 0)[lane] + score_ptr(rg, 3, 0)[lane];
                const float bl_hi = score_ptr(rg, 0, 1)[lane] + score_ptr(rg, 1, 1)[lane]
                                  + score_ptr(rg, 2, 1)[lane] + score_ptr(rg, 3, 1)[lane];
                l_lo = l_lo * a_lo + bl_lo;
                l_hi = l_hi * a_hi + bl_hi;
            }

            // Skip ~95% of rescaling FMAs when running max did not change!
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

    // Write split partials (note: FP32 mma_pv_f32 output uses d_get_i(l) / d_get_j(l) layout!)
    const int rows_pad = hkv * rows_total;
    if (lw == 0 && lane < rows_here) {
        const int token  = (row_base + lane) / GQA;
        const int c      = (row_base + lane) - token * GQA;
        const int row_gl = (kv_head * GQA + c) * q_len + token;
        const int64_t st = (int64_t) split * rows_pad + row_gl;
        partial_m[st] = ((lane & 2) == 0) ? m_lo : m_hi;
        partial_l[st] = ((lane & 2) == 0) ? l_lo : l_hi;
    }
    if (rows_here > 0) {
        const int r0 = row_lo;
        const int r1 = row_hi;
        const int d_base = lw * kDSlice;
        float * out0 = nullptr;
        float * out1 = nullptr;
        if (r0 < rows_here) {
            const int tok = (row_base + r0) / GQA;
            const int cix = (row_base + r0) - tok * GQA;
            const int rgl = (kv_head * GQA + cix) * q_len + tok;
            out0 = partial_acc + ((int64_t) split * rows_pad + rgl) * kD + d_base;
        }
        if (r1 < rows_here) {
            const int tok = (row_base + r1) / GQA;
            const int cix = (row_base + r1) - tok * GQA;
            const int rgl = (kv_head * GQA + cix) * q_len + tok;
            out1 = partial_acc + ((int64_t) split * rows_pad + rgl) * kD + d_base;
        }
        const int j0 = d_get_j(0); // (lane&2) + 0
        const int j1 = d_get_j(1); // (lane&2) + 1
        const int j4 = d_get_j(4); // (lane&2) + 4
        const int j5 = d_get_j(5); // (lane&2) + 5
#pragma unroll
        for (int c = 0; c < kDLocal; ++c) {
            const int co = c * 8;
            if (out0 != nullptr) {
                out0[co + j0] = acc_f[c][0];
                out0[co + j1] = acc_f[c][1];
                out0[co + j4] = acc_f[c][4];
                out0[co + j5] = acc_f[c][5];
            }
            if (out1 != nullptr) {
                out1[co + j0] = acc_f[c][2];
                out1[co + j1] = acc_f[c][3];
                out1[co + j4] = acc_f[c][6];
                out1[co + j5] = acc_f[c][7];
            }
        }
    }
#endif // __CUDA_ARCH__ == 700
}

template <int GQA, int KM, int VM, bool MM>
static void run_fa96_t(
        dim3 grid, cudaStream_t stream,
        const float * q, int64_t q_row, int64_t q_head,
        const half * k, const half * v,
        int64_t k_row, int64_t k_head, int64_t v_row, int64_t v_head,
        int q_len, int kv_len, float scale, int splits,
        const half * mask_last_q, int64_t mask_q_stride,
        float * partial_acc, float * partial_m, float * partial_l,
        const half * mask_kv, int64_t mask_kv_stride) {
    auto kfn = sm70_fa96_decode_kernel<GQA, KM, VM, MM>;
    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kSmemBytes));
        attr_set = true;
    }
    kfn<<<grid, kThreads, kSmemBytes, stream>>>(
        q, q_row, q_head, k, v, k_row, k_head, v_row, v_head,
        q_len, kv_len, scale, splits, mask_last_q, mask_q_stride,
        partial_acc, partial_m, partial_l, mask_kv, mask_kv_stride);
}

void launch_sm70_fa96_decode(
        int gqa, int k_mode, int v_mode, bool mask_md,
        dim3 grid, cudaStream_t stream,
        const float * q, int64_t q_row, int64_t q_head,
        const half * k, const half * v,
        int64_t k_row, int64_t k_head, int64_t v_row, int64_t v_head,
        int q_len, int kv_len, float scale, int splits,
        const half * mask_last_q, int64_t mask_q_stride,
        float * partial_acc, float * partial_m, float * partial_l,
        const half * mask_kv, int64_t mask_kv_stride) {
    auto disp_kv = [&](auto mm) {
        constexpr bool MM = decltype(mm)::value;
        const int kv_pair = k_mode * 3 + v_mode;
        auto go = [&](auto km, auto vm) {
            constexpr int KM = decltype(km)::value;
            constexpr int VM = decltype(vm)::value;
            if (gqa == 6) {
                run_fa96_t<6, KM, VM, MM>(grid, stream, q, q_row, q_head, k, v, k_row, k_head, v_row, v_head,
                    q_len, kv_len, scale, splits, mask_last_q, mask_q_stride, partial_acc, partial_m, partial_l, mask_kv, mask_kv_stride);
            } else {
                run_fa96_t<8, KM, VM, MM>(grid, stream, q, q_row, q_head, k, v, k_row, k_head, v_row, v_head,
                    q_len, kv_len, scale, splits, mask_last_q, mask_q_stride, partial_acc, partial_m, partial_l, mask_kv, mask_kv_stride);
            }
        };
        switch (kv_pair) {
            case 0: go(std::integral_constant<int,0>{}, std::integral_constant<int,0>{}); break;
            case 1: go(std::integral_constant<int,0>{}, std::integral_constant<int,1>{}); break;
            case 2: go(std::integral_constant<int,0>{}, std::integral_constant<int,2>{}); break;
            case 3: go(std::integral_constant<int,1>{}, std::integral_constant<int,0>{}); break;
            case 4: go(std::integral_constant<int,1>{}, std::integral_constant<int,1>{}); break;
            case 6: go(std::integral_constant<int,2>{}, std::integral_constant<int,0>{}); break;
            case 8: go(std::integral_constant<int,2>{}, std::integral_constant<int,2>{}); break;
            default: GGML_ABORT("sm70_fa96: unsupported kv_pair %d", kv_pair);
        }
    };
    if (mask_md) { disp_kv(std::true_type{}); } else { disp_kv(std::false_type{}); }
}

} // namespace sm70_fa96
