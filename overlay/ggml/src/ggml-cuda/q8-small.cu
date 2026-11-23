#include "q8-small.cuh"

#include <cstdlib>

#define Q8S_CHUNK  128 // K values per CUDA block, 4 q8_0 blocks
#define Q8S_MAX_N  64
#define Q8S_MAX_M  16
#define Q8S_STRIDE (Q8S_CHUNK + 1) // odd row stride: the threads of a warp read different rows, this keeps them on different banks

// One CUDA block per chunk of 128 K values: the chunk of all weight rows and of all input columns goes to
// shared memory, then every (row, column) pair gets its partial dot product. part is [chunks][m][n].
static __global__ void q8_small_partial_kernel(
        const block_q8_0 * __restrict__ w, const float * __restrict__ x, float * __restrict__ part,
        const int n, const int m, const int nblk_row, const int64_t x_stride) {
    __shared__ float ws[Q8S_MAX_N * Q8S_STRIDE];
    __shared__ float xs[Q8S_MAX_M * Q8S_CHUNK];

    const int chunk = blockIdx.x;
    const int tid   = threadIdx.x;

    for (int i = tid; i < n * (Q8S_CHUNK / QK8_0); i += blockDim.x) {
        const int r = i / (Q8S_CHUNK / QK8_0);
        const int c = i % (Q8S_CHUNK / QK8_0);
        const block_q8_0 * blk = w + (int64_t) r * nblk_row + chunk * (Q8S_CHUNK / QK8_0) + c;
        const float d = __half2float(blk->d);
        float * o = ws + r * Q8S_STRIDE + c * QK8_0;
#pragma unroll
        for (int e = 0; e < QK8_0; ++e) {
            o[e] = d * (float) blk->qs[e];
        }
    }
    for (int i = tid; i < m * Q8S_CHUNK; i += blockDim.x) {
        const int j = i / Q8S_CHUNK;
        const int e = i % Q8S_CHUNK;
        xs[i] = x[j * x_stride + (int64_t) chunk * Q8S_CHUNK + e];
    }
    __syncthreads();

    const int nm = n * m;
    for (int p = tid; p < nm; p += blockDim.x) {
        const float * wr = ws + (p % n) * Q8S_STRIDE;
        const float * xj = xs + (p / n) * Q8S_CHUNK;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
#pragma unroll 8
        for (int e = 0; e < Q8S_CHUNK; e += 4) {
            s0 += wr[e + 0] * xj[e + 0];
            s1 += wr[e + 1] * xj[e + 1];
            s2 += wr[e + 2] * xj[e + 2];
            s3 += wr[e + 3] * xj[e + 3];
        }
        part[(int64_t) chunk * nm + p] = (s0 + s1) + (s2 + s3);
    }
}

// Sums the chunks in a fixed order, so the result does not depend on scheduling.
static __global__ void q8_small_reduce_kernel(
        const float * __restrict__ part, float * __restrict__ dst, const int nm, const int nchunks) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nm) {
        return;
    }
    float s = 0.0f;
    for (int c = 0; c < nchunks; ++c) {
        s += part[(int64_t) c * nm + p];
    }
    dst[p] = s;
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
    if (n < 8 || n > Q8S_MAX_N || m < 2 || m > Q8S_MAX_M || k < Q8S_CHUNK || k % Q8S_CHUNK != 0 || src1->ne[0] != k) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    GGML_ASSERT(dst->ne[0] == n && dst->ne[1] == m);

    const int nchunks = (int) (k / Q8S_CHUNK);
    const int nm      = (int) (n * m);
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) nchunks * nm);

    q8_small_partial_kernel<<<nchunks, 256, 0, stream>>>(
        (const block_q8_0 *) src0->data, (const float *) src1->data, part.get(),
        (int) n, (int) m, (int) (k / QK8_0), (int64_t) (src1->nb[1] / sizeof(float)));
    CUDA_CHECK(cudaGetLastError());
    q8_small_reduce_kernel<<<(nm + 255) / 256, 256, 0, stream>>>(part.get(), (float *) dst->data, nm, nchunks);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
