#pragma once
#include "common.cuh"
#include <cuda_fp16.h>
#include <cstdint>

namespace sm70_fa96 {
constexpr int kBr         = 96;
constexpr int kRowGroups  = 3;
constexpr int kSubWarps   = 4;
constexpr int kWarps      = kRowGroups * kSubWarps; // 12 warps
constexpr int kThreads    = kWarps * 32;            // 384 threads
constexpr int kBc         = 32;
constexpr int kD          = 256;
constexpr int kSmemPad    = 8;
constexpr int kSmemStride = kD + kSmemPad;          // 264 halfs = 528 B (16B aligned)
constexpr size_t kQBytes      = kBr * kSmemStride * sizeof(half);         // 50,688 B
constexpr size_t kKVBytes     = kBc * kSmemStride * sizeof(half);         // 16,896 B
constexpr size_t kScoresBytes = kRowGroups * 4 * 8 * 32 * sizeof(float);  // 12,288 B
constexpr size_t kPBytes      = kRowGroups * 4 * 32 * 4 * sizeof(half2);  //  6,144 B
constexpr size_t kStatBytes   = kRowGroups * 4 * 32 * 2 * sizeof(float);  //  3,072 B (lo/hi max or sum)
constexpr size_t kSmemBytes   = kQBytes + kKVBytes + kScoresBytes + kPBytes + kStatBytes; // 89,088 B (< 96 KB)

void launch_sm70_fa96_decode(
        int gqa, int k_mode, int v_mode, bool mask_md,
        dim3 grid, cudaStream_t stream,
        const float * q, int64_t q_row, int64_t q_head,
        const half * k, const half * v,
        int64_t k_row, int64_t k_head, int64_t v_row, int64_t v_head,
        int q_len, int kv_len, float scale, int splits,
        const half * mask_last_q, int64_t mask_q_stride,
        float * partial_acc, float * partial_m, float * partial_l,
        const half * mask_kv, int64_t mask_kv_stride);
} // namespace sm70_fa96
