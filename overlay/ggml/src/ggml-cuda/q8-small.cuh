// Q8_0 weights with few rows times a few input columns, computed in float.
// Written for the 5120 -> 48 projections of Qwen3.8's delta-net layers (ssm_alpha, ssm_beta), which
// the general quantized kernels handle as three launches (quantize, multiply, fixup).
#pragma once

#include "common.cuh"

// Returns true if it computed dst = src0 * src1. Off with LLAMA_SM70_Q8_SMALL=0.
bool ggml_cuda_q8_small_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Stage 8E-2: Dual fused Q8_0 small-N matmul (e.g. ssm_beta + ssm_alpha sharing src1).
bool ggml_cuda_q8_small_mul_mat_dual(ggml_backend_cuda_context & ctx, const ggml_tensor * src0_a, const ggml_tensor * src0_b, const ggml_tensor * src1, ggml_tensor * dst_a, ggml_tensor * dst_b);
