#include "q8-small.cuh"

#include <cstdint>
#include <cstdlib>

#define Q8S_MAX_N 64
#define Q8S_MAX_M 16

// Single-pass float4-vectorized multi-row Q8_0 small-N matmul (Stage 7E-v2).
// Each CTA computes RN output rows x MT batch columns, reusing each float4 activation
// load across RN weight rows in registers (cutting L2 activation traffic by RNx and
// LDG instruction count by 4x) with 8 lanes per Q8_0 block (4 blocks/warp/iter).
template <int RN, int MT, int NWARPS>
static __global__ void __launch_bounds__(NWARPS * 32)
q8_small_tiled_kernel(
        const block_q8_0 * __restrict__ w,
        const float      * __restrict__ x,
        float            * __restrict__ dst,
        const int n, const int m, const int nblk_row, const int64_t x_stride) {
    const int r0    = blockIdx.x * RN;
    const int m0    = blockIdx.y * MT;
    const int lane  = threadIdx.x; // 0..31
    const int warp  = threadIdx.y; // 0..NWARPS-1
    const int sub_b = lane >> 3;   // 0..3 (4 blocks per warp)
    const int sub_q = lane & 7;    // 0..7 (8 float4 groups per 32-elem block)

    if (r0 >= n || m0 >= m) {
        return;
    }
    const int r_rem = n - r0;
    const int m_rem = m - m0;
    const float * x_base = x + (int64_t) m0 * x_stride;

    float acc[RN][MT];
#pragma unroll
    for (int ri = 0; ri < RN; ++ri) {
#pragma unroll
        for (int j = 0; j < MT; ++j) {
            acc[ri][j] = 0.0f;
        }
    }

    const int b_stride = NWARPS * 4;
    if (r_rem >= RN && m_rem >= MT) {
        for (int b = warp * 4 + sub_b; b < nblk_row; b += b_stride) {
            const int k_vec = b * 8 + sub_q;
            float4 xv[MT];
#pragma unroll
            for (int j = 0; j < MT; ++j) {
                xv[j] = reinterpret_cast<const float4 *>(x_base + (int64_t) j * x_stride)[k_vec];
            }
#pragma unroll
            for (int ri = 0; ri < RN; ++ri) {
                const block_q8_0 * blk = w + ((int64_t) (r0 + ri) * nblk_row + b);
                const float d = __half2float(blk->d);
                const uint16_t * qp = reinterpret_cast<const uint16_t *>(blk->qs + sub_q * 4);
                const uint16_t q01 = qp[0];
                const uint16_t q23 = qp[1];
                const float q0 = (float) (int8_t) (q01 & 0xFFu);
                const float q1 = (float) (int8_t) (q01 >> 8);
                const float q2 = (float) (int8_t) (q23 & 0xFFu);
                const float q3 = (float) (int8_t) (q23 >> 8);
#pragma unroll
                for (int j = 0; j < MT; ++j) {
                    const float dot = fmaf(q0, xv[j].x, fmaf(q1, xv[j].y, fmaf(q2, xv[j].z, q3 * xv[j].w)));
                    acc[ri][j] = fmaf(d, dot, acc[ri][j]);
                }
            }
        }
    } else {
        for (int b = warp * 4 + sub_b; b < nblk_row; b += b_stride) {
            const int k_vec = b * 8 + sub_q;
            float4 xv[MT];
#pragma unroll
            for (int j = 0; j < MT; ++j) {
                if (j < m_rem) {
                    xv[j] = reinterpret_cast<const float4 *>(x_base + (int64_t) j * x_stride)[k_vec];
                }
            }
#pragma unroll
            for (int ri = 0; ri < RN; ++ri) {
                if (ri < r_rem) {
                    const block_q8_0 * blk = w + ((int64_t) (r0 + ri) * nblk_row + b);
                    const float d = __half2float(blk->d);
                    const uint16_t * qp = reinterpret_cast<const uint16_t *>(blk->qs + sub_q * 4);
                    const uint16_t q01 = qp[0];
                    const uint16_t q23 = qp[1];
                    const float q0 = (float) (int8_t) (q01 & 0xFFu);
                    const float q1 = (float) (int8_t) (q01 >> 8);
                    const float q2 = (float) (int8_t) (q23 & 0xFFu);
                    const float q3 = (float) (int8_t) (q23 >> 8);
#pragma unroll
                    for (int j = 0; j < MT; ++j) {
                        if (j < m_rem) {
                            const float dot = fmaf(q0, xv[j].x, fmaf(q1, xv[j].y, fmaf(q2, xv[j].z, q3 * xv[j].w)));
                            acc[ri][j] = fmaf(d, dot, acc[ri][j]);
                        }
                    }
                }
            }
        }
    }

    __shared__ float smem[NWARPS][RN * MT];
#pragma unroll
    for (int ri = 0; ri < RN; ++ri) {
#pragma unroll
        for (int j = 0; j < MT; ++j) {
            float v = acc[ri][j];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_xor_sync(0xffffffffu, v, off);
            }
            if (lane == 0) {
                smem[warp][ri * MT + j] = v;
            }
        }
    }
    __syncthreads();

    if (warp == 0 && lane < RN * MT) {
        const int ri = lane / MT;
        const int j  = lane % MT;
        if (ri < r_rem && j < m_rem) {
            float sum = 0.0f;
#pragma unroll
            for (int w_i = 0; w_i < NWARPS; ++w_i) {
                sum += smem[w_i][lane];
            }
            dst[(int64_t) (m0 + j) * n + (r0 + ri)] = sum;
        }
    }
}

bool ggml_cuda_q8_small_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    static const bool enabled = [] {
        const char * e = getenv("LLAMA_SM70_Q8_SMALL");
        return e == nullptr || atoi(e) != 0;
    }();
    if (!enabled) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = src1->ne[1];
    if (n < 8 || n > Q8S_MAX_N || m < 2 || m > Q8S_MAX_M || k < QK8_0 || k % QK8_0 != 0 || src1->ne[0] != k) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int64_t x_stride = (int64_t) (src1->nb[1] / sizeof(float));
    if ((((uintptr_t) src1->data) & 15u) != 0 || (x_stride & 3) != 0) {
        return false;
    }
    GGML_ASSERT(dst->ne[0] == n && dst->ne[1] == m);

    const int nblk_row = (int) (k / QK8_0);
    cudaStream_t stream = ctx.stream();

    // Choose (RN, MT) so typical n=48, m=16 launches 64 CTAs (1 clean wave on 80 SMs,
    // 3x less activation L2 traffic, 4x fewer LDG ops via float4).
    if (m <= 4) {
        constexpr int RN = 2;
        constexpr int MT = 4;
        constexpr int NWARPS = 8;
        const dim3 grid((uint32_t) ((n + RN - 1) / RN), 1, 1);
        const dim3 block(32, NWARPS, 1);
        q8_small_tiled_kernel<RN, MT, NWARPS><<<grid, block, 0, stream>>>(
            (const block_q8_0 *) src0->data, (const float *) src1->data, (float *) dst->data,
            (int) n, (int) m, nblk_row, x_stride);
    } else if (m <= 8) {
        constexpr int RN = 2;
        constexpr int MT = 4;
        constexpr int NWARPS = 8;
        const dim3 grid((uint32_t) ((n + RN - 1) / RN), (uint32_t) ((m + MT - 1) / MT), 1);
        const dim3 block(32, NWARPS, 1);
        q8_small_tiled_kernel<RN, MT, NWARPS><<<grid, block, 0, stream>>>(
            (const block_q8_0 *) src0->data, (const float *) src1->data, (float *) dst->data,
            (int) n, (int) m, nblk_row, x_stride);
    } else {
        constexpr int RN = 3;
        constexpr int MT = 4;
        constexpr int NWARPS = 8;
        const dim3 grid((uint32_t) ((n + RN - 1) / RN), (uint32_t) ((m + MT - 1) / MT), 1);
        const dim3 block(32, NWARPS, 1);
        q8_small_tiled_kernel<RN, MT, NWARPS><<<grid, block, 0, stream>>>(
            (const block_q8_0 *) src0->data, (const float *) src1->data, (float *) dst->data,
            (int) n, (int) m, nblk_row, x_stride);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

template <int RN, int MT, int NWARPS>
static __global__ void __launch_bounds__(NWARPS * 32)
q8_small_dual_tiled_kernel(
        const block_q8_0 * __restrict__ wa,
        const block_q8_0 * __restrict__ wb,
        const float      * __restrict__ x,
        float            * __restrict__ dsta,
        float            * __restrict__ dstb,
        const int n, const int m, const int nblk_row, const int64_t x_stride) {
    const int r0    = blockIdx.x * RN;
    const int m0    = blockIdx.y * MT;
    const int lane  = threadIdx.x;
    const int warp  = threadIdx.y;
    const int sub_b = lane >> 3;
    const int sub_q = lane & 7;

    if (r0 >= n || m0 >= m) {
        return;
    }
    const int r_rem = n - r0;
    const int m_rem = m - m0;
    const float * x_base = x + (int64_t) m0 * x_stride;

    float acca[RN][MT];
    float accb[RN][MT];
#pragma unroll
    for (int ri = 0; ri < RN; ++ri) {
#pragma unroll
        for (int j = 0; j < MT; ++j) {
            acca[ri][j] = 0.0f;
            accb[ri][j] = 0.0f;
        }
    }

    const int b_stride = NWARPS * 4;
    if (r_rem >= RN && m_rem >= MT) {
        for (int b = warp * 4 + sub_b; b < nblk_row; b += b_stride) {
            const int k_vec = b * 8 + sub_q;
            float4 xv[MT];
#pragma unroll
            for (int j = 0; j < MT; ++j) {
                xv[j] = reinterpret_cast<const float4 *>(x_base + (int64_t) j * x_stride)[k_vec];
            }
#pragma unroll
            for (int ri = 0; ri < RN; ++ri) {
                const int64_t boff = (int64_t) (r0 + ri) * nblk_row + b;
                {
                    const block_q8_0 * blk = wa + boff;
                    const float d = __half2float(blk->d);
                    const uint16_t * qp = reinterpret_cast<const uint16_t *>(blk->qs + sub_q * 4);
                    const uint16_t q01 = qp[0];
                    const uint16_t q23 = qp[1];
                    const float q0 = (float) (int8_t) (q01 & 0xFFu);
                    const float q1 = (float) (int8_t) (q01 >> 8);
                    const float q2 = (float) (int8_t) (q23 & 0xFFu);
                    const float q3 = (float) (int8_t) (q23 >> 8);
#pragma unroll
                    for (int j = 0; j < MT; ++j) {
                        const float dot = fmaf(q0, xv[j].x, fmaf(q1, xv[j].y, fmaf(q2, xv[j].z, q3 * xv[j].w)));
                        acca[ri][j] = fmaf(d, dot, acca[ri][j]);
                    }
                }
                {
                    const block_q8_0 * blk = wb + boff;
                    const float d = __half2float(blk->d);
                    const uint16_t * qp = reinterpret_cast<const uint16_t *>(blk->qs + sub_q * 4);
                    const uint16_t q01 = qp[0];
                    const uint16_t q23 = qp[1];
                    const float q0 = (float) (int8_t) (q01 & 0xFFu);
                    const float q1 = (float) (int8_t) (q01 >> 8);
                    const float q2 = (float) (int8_t) (q23 & 0xFFu);
                    const float q3 = (float) (int8_t) (q23 >> 8);
#pragma unroll
                    for (int j = 0; j < MT; ++j) {
                        const float dot = fmaf(q0, xv[j].x, fmaf(q1, xv[j].y, fmaf(q2, xv[j].z, q3 * xv[j].w)));
                        accb[ri][j] = fmaf(d, dot, accb[ri][j]);
                    }
                }
            }
        }
    } else {
        for (int b = warp * 4 + sub_b; b < nblk_row; b += b_stride) {
            const int k_vec = b * 8 + sub_q;
            float4 xv[MT];
#pragma unroll
            for (int j = 0; j < MT; ++j) {
                if (j < m_rem) {
                    xv[j] = reinterpret_cast<const float4 *>(x_base + (int64_t) j * x_stride)[k_vec];
                }
            }
#pragma unroll
            for (int ri = 0; ri < RN; ++ri) {
                if (ri < r_rem) {
                    const int64_t boff = (int64_t) (r0 + ri) * nblk_row + b;
                    {
                        const block_q8_0 * blk = wa + boff;
                        const float d = __half2float(blk->d);
                        const uint16_t * qp = reinterpret_cast<const uint16_t *>(blk->qs + sub_q * 4);
                        const uint16_t q01 = qp[0];
                        const uint16_t q23 = qp[1];
                        const float q0 = (float) (int8_t) (q01 & 0xFFu);
                        const float q1 = (float) (int8_t) (q01 >> 8);
                        const float q2 = (float) (int8_t) (q23 & 0xFFu);
                        const float q3 = (float) (int8_t) (q23 >> 8);
#pragma unroll
                        for (int j = 0; j < MT; ++j) {
                            if (j < m_rem) {
                                const float dot = fmaf(q0, xv[j].x, fmaf(q1, xv[j].y, fmaf(q2, xv[j].z, q3 * xv[j].w)));
                                acca[ri][j] = fmaf(d, dot, acca[ri][j]);
                            }
                        }
                    }
                    {
                        const block_q8_0 * blk = wb + boff;
                        const float d = __half2float(blk->d);
                        const uint16_t * qp = reinterpret_cast<const uint16_t *>(blk->qs + sub_q * 4);
                        const uint16_t q01 = qp[0];
                        const uint16_t q23 = qp[1];
                        const float q0 = (float) (int8_t) (q01 & 0xFFu);
                        const float q1 = (float) (int8_t) (q01 >> 8);
                        const float q2 = (float) (int8_t) (q23 & 0xFFu);
                        const float q3 = (float) (int8_t) (q23 >> 8);
#pragma unroll
                        for (int j = 0; j < MT; ++j) {
                            if (j < m_rem) {
                                const float dot = fmaf(q0, xv[j].x, fmaf(q1, xv[j].y, fmaf(q2, xv[j].z, q3 * xv[j].w)));
                                accb[ri][j] = fmaf(d, dot, accb[ri][j]);
                            }
                        }
                    }
                }
            }
        }
    }

    constexpr int RM = RN * MT;
    __shared__ float smem[NWARPS][2 * RM];
#pragma unroll
    for (int ri = 0; ri < RN; ++ri) {
#pragma unroll
        for (int j = 0; j < MT; ++j) {
            float va = acca[ri][j];
            float vb = accb[ri][j];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                va += __shfl_xor_sync(0xffffffffu, va, off);
                vb += __shfl_xor_sync(0xffffffffu, vb, off);
            }
            if (lane == 0) {
                smem[warp][ri * MT + j]      = va;
                smem[warp][RM + ri * MT + j] = vb;
            }
        }
    }
    __syncthreads();

    if (warp == 0 && lane < 2 * RM) {
        const int is_b = lane >= RM;
        const int idx  = is_b ? (lane - RM) : lane;
        const int ri   = idx / MT;
        const int j    = idx % MT;
        if (ri < r_rem && j < m_rem) {
            float sum = 0.0f;
#pragma unroll
            for (int w_i = 0; w_i < NWARPS; ++w_i) {
                sum += smem[w_i][lane];
            }
            float * out = is_b ? dstb : dsta;
            out[(int64_t) (m0 + j) * n + (r0 + ri)] = sum;
        }
    }
}

bool ggml_cuda_q8_small_mul_mat_dual(ggml_backend_cuda_context & ctx,
                                     const ggml_tensor * src0_a, const ggml_tensor * src0_b,
                                     const ggml_tensor * src1,
                                     ggml_tensor * dst_a, ggml_tensor * dst_b) {
    static const bool enabled = [] {
        const char * e = getenv("LLAMA_SM70_Q8_SMALL_DUAL");
        return e == nullptr || e[0] != '0';
    }();
    if (!enabled) { return false; }
    if (src0_a->type != GGML_TYPE_Q8_0 || src0_b->type != GGML_TYPE_Q8_0 ||
        src1->type != GGML_TYPE_F32 || dst_a->type != GGML_TYPE_F32 || dst_b->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t k = src0_a->ne[0];
    const int64_t n = src0_a->ne[1];
    const int64_t m = src1->ne[1];
    if (src0_b->ne[0] != k || src0_b->ne[1] != n ||
        n < 8 || n > Q8S_MAX_N || m < 2 || m > Q8S_MAX_M || k < QK8_0 || k % QK8_0 != 0 || src1->ne[0] != k) {
        return false;
    }
    if (src0_a->ne[2] != 1 || src0_a->ne[3] != 1 || src0_b->ne[2] != 1 || src0_b->ne[3] != 1 ||
        src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(src0_a) || !ggml_is_contiguous(src0_b) || !ggml_is_contiguous(src1) ||
        !ggml_is_contiguous(dst_a) || !ggml_is_contiguous(dst_b)) {
        return false;
    }
    if (dst_a->ne[0] != n || dst_a->ne[1] != m || dst_b->ne[0] != n || dst_b->ne[1] != m) {
        return false;
    }
    const int64_t x_stride = (int64_t) (src1->nb[1] / sizeof(float));
    if ((((uintptr_t) src1->data) & 15u) != 0 || (x_stride & 3) != 0) {
        return false;
    }
    const int nblk_row = (int) (k / QK8_0);
    cudaStream_t stream = ctx.stream();
    if (m <= 4) {
        constexpr int RN = 2, MT = 4, NWARPS = 8;
        const dim3 grid((uint32_t) ((n + RN - 1) / RN), 1, 1);
        const dim3 block(32, NWARPS, 1);
        q8_small_dual_tiled_kernel<RN, MT, NWARPS><<<grid, block, 0, stream>>>(
            (const block_q8_0 *) src0_a->data, (const block_q8_0 *) src0_b->data,
            (const float *) src1->data, (float *) dst_a->data, (float *) dst_b->data,
            (int) n, (int) m, nblk_row, x_stride);
    } else if (m <= 8) {
        constexpr int RN = 2, MT = 4, NWARPS = 8;
        const dim3 grid((uint32_t) ((n + RN - 1) / RN), (uint32_t) ((m + MT - 1) / MT), 1);
        const dim3 block(32, NWARPS, 1);
        q8_small_dual_tiled_kernel<RN, MT, NWARPS><<<grid, block, 0, stream>>>(
            (const block_q8_0 *) src0_a->data, (const block_q8_0 *) src0_b->data,
            (const float *) src1->data, (float *) dst_a->data, (float *) dst_b->data,
            (int) n, (int) m, nblk_row, x_stride);
    } else {
        constexpr int RN = 3, MT = 4, NWARPS = 8;
        const dim3 grid((uint32_t) ((n + RN - 1) / RN), (uint32_t) ((m + MT - 1) / MT), 1);
        const dim3 block(32, NWARPS, 1);
        q8_small_dual_tiled_kernel<RN, MT, NWARPS><<<grid, block, 0, stream>>>(
            (const block_q8_0 *) src0_a->data, (const block_q8_0 *) src0_b->data,
            (const float *) src1->data, (float *) dst_a->data, (float *) dst_b->data,
            (int) n, (int) m, nblk_row, x_stride);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
