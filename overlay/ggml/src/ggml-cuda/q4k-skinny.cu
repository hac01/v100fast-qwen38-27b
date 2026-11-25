// Q2_K/Q3_K/Q4_K/Q5_K/Q6_K skinny GEMM for sm_70 (Volta).
//
// Design follows the QPN8 execution layout of q8-skinny.cu (itself adapted from 1Cat-vLLM
// fp8_qpn8_sm70.cu, Apache-2.0) and the int4 scale/bias folding of 1Cat's awq_qpn_m1_sm70.cu
// and ninfer-v100's q4 Volta kernels. The QPN8 layout is derived from dnv2003/v100-skinny
// (MIT) and its block-scale adaptation in haohervchb/sglang-V100. See LICENSE.v100-skinny in
// this directory and docs/design/v106-q4k-skinny.md for the layout and the numeric analysis.

#include "common.cuh"
#include "convert.cuh"
#include "q-skinny-common.cuh"
#include "q4k-skinny.cuh"
#include "iq-skinny.cuh"

#include <algorithm>
#include <vector>

// sentinel object whose address tags a repacked Q4_K tensor (not const: it is stored in tensor->extra)
static char q4k_skinny_marker = 0;

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// ---- codec: per-type code decode plus scale/min parsing ----

// the record struct + qpn8 code get/put helpers live in q-skinny-common.cuh

// get_scale_min_k4 from dequantize.cuh on two little-endian 64-bit words of scale bytes
static __device__ __forceinline__ uint8_t qskinny_scale_byte(const unsigned long long s_lo,
                                                             const unsigned long long s_hi, const int i) {
    return (uint8_t) ((i < 8 ? s_lo : s_hi) >> (8 * (i & 7)));
}

// One record holds values_per_record raw codes in qpn8 physical slot order; meta holds the raw
// super-block header. 16*record_bytes + meta_bytes == block size, so a repacked tensor keeps the
// byte count of its quantized type and the repack stays in place. decode() turns the two records
// of one 32-value sub-block plus the meta of its super-block into 16 half2 weights.
//
// Q4_K: w = d*sc*q - dmin*m folded into one half FMA. Q2_K/Q3_K/Q5_K/Q6_K fold the same way
// (their raw code lands exact in the 1024 + code bit pattern), so one hsub2 + one half FMA or
// multiply per value pair; the f16 result matches the float expression of dequantize.cuh up to
// the single f16 rounding of the scale, exactly as the Q4_K decoder has always done.

// ---- Q4_K: 8 code bytes per 16 K values and lane (16 nibbles, physical slot s in byte s/2,
// half s&1) and 16 meta bytes per 256 K values (the raw super-block header). ----

template <> struct qskinny_codec<GGML_TYPE_Q4_K> {
    using record_t = uint2;
    using meta_t   = uint4;
    static constexpr int record_bytes      = 8;
    static constexpr int meta_bytes        = 16;
    static constexpr int bits_per_value    = 4;
    static constexpr int block_bytes       = 144;
    static constexpr int values_per_record = 16;
    static constexpr int values_per_meta   = 256;
    static constexpr int values_per_sub_block = 32;

    static __host__ __device__ constexpr size_t codes_bytes(size_t n, size_t k) {
        return n * k * record_bytes / values_per_record;
    }

    static __device__ __forceinline__ qskinny_code load_record(const record_t * p) {
        const uint2 r = __ldcs(p);
        return { (unsigned long long) r.x | ((unsigned long long) r.y << 32), 0 };
    }

    static __device__ __forceinline__ meta_t load_meta(const uint8_t * p) {
        return __ldcs(reinterpret_cast<const meta_t *>(p));
    }

    static __device__ __forceinline__ void store_record(const qskinny_code rec, uint8_t * p) {
        *reinterpret_cast<uint2 *>(p) = make_uint2((unsigned) rec.lo, (unsigned) (rec.lo >> 32));
    }

    static __device__ __forceinline__ void store_meta(const meta_t meta, uint8_t * p) {
        *reinterpret_cast<meta_t *>(p) = meta;
    }

    // the raw 16-byte super-block header (dm + scales), copied unchanged
    static __device__ __forceinline__ meta_t pack_meta(const uint8_t * block) {
        return *reinterpret_cast<const meta_t *>(block);
    }

    // a 32-value sub-block uses 32 consecutive qs bytes: low nibbles for even sub-blocks, high
    // nibbles for odd ones; group g covers the first (g&1 == 0) or second 16 values
    static __device__ __forceinline__ qskinny_code pack_record(const uint8_t * block, const int g) {
        const uint8_t * qs = block + 16;
        const int qs_base  = 32 * (g >> 2) + ((g & 1) ? 16 : 0);
        const int qs_shift = (g & 2) ? 4 : 0;
        qskinny_code rec = { 0, 0 };
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int slot = qpn8_physical_k(j);
            qskinny_code_put(rec, (qs[qs_base + j] >> qs_shift) & 0xF, 4 * slot, 4);
        }
        return rec;
    }

    static __device__ __forceinline__ uint8_t meta_byte(const uint4 meta, const int i) {
        // scales[12] starts at byte 4 of the super-block header
        const unsigned word = i < 4 ? meta.y : i < 8 ? meta.z : meta.w;
        return (uint8_t) (word >> (8 * (i & 3)));
    }

    // get_scale_min_k4 from dequantize.cuh, applied to the raw meta bytes
    static __device__ __forceinline__ void get_scale_min(const uint4 meta, const int j,
                                                         uint8_t & sc, uint8_t & m) {
        if (j < 4) {
            sc = meta_byte(meta, j) & 63;
            m  = meta_byte(meta, j + 4) & 63;
        } else {
            sc = (meta_byte(meta, j + 4) & 0x0F) | ((meta_byte(meta, j - 4) >> 6) << 4);
            m  = (meta_byte(meta, j + 4) >> 4)   | ((meta_byte(meta, j)     >> 6) << 4);
        }
    }

    // One record holds 16 nibbles in physical order. The m8n8k4 B fragment pairs physical
    // slots (s, s+4), exactly like the Q8_0 decoder pairs bytes (i, i+4).
    static __device__ __forceinline__ void decode_record(const qskinny_code rec, half2 out[8]) {
        const unsigned rx = (unsigned) rec.lo;
        const unsigned ry = (unsigned) (rec.lo >> 32);
        // the second byte goes to result byte 2, where the mask below reads it
        const unsigned pairs[4] = {
            __byte_perm(rx, 0, 0x0200),  // bytes (0, 2)
            __byte_perm(rx, 0, 0x0301),  // bytes (1, 3)
            __byte_perm(ry, 0, 0x0200),  // bytes (4, 6)
            __byte_perm(ry, 0, 0x0301),  // bytes (5, 7)
        };
        const unsigned offset_bits = 0x64006400u; // f16 1024 in both halves
        const half2 offset = *reinterpret_cast<const half2 *>(&offset_bits);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned lo = (pairs[i] & 0x000F000Fu) | offset_bits;
            const unsigned hi = ((pairs[i] >> 4) & 0x000F000Fu) | offset_bits;
            out[2*i + 0] = __hsub2(*reinterpret_cast<const half2 *>(&lo), offset);
            out[2*i + 1] = __hsub2(*reinterpret_cast<const half2 *>(&hi), offset);
        }
    }

    static __device__ __forceinline__ void decode(const qskinny_code records[2], const uint4 meta,
                                                  const int sub_block, half2 out[16]) {
        decode_record(records[0], out);
        decode_record(records[1], out + 8);

        uint8_t sc, m;
        get_scale_min(meta, sub_block, sc, m);
        const half2 dm = *reinterpret_cast<const half2 *>(&meta.x);
        const float d    = __half2float(__low2half(dm));
        const float dmin = __half2float(__high2half(dm));
        // f32 folding, rounded once to f16 (see 3.4 of the design note)
        const half2 scale2 = __float2half2_rn(d * (float) sc);
        const half2 bias2  = __float2half2_rn(-dmin * (float) m);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            out[i] = __hfma2(out[i], scale2, bias2);
        }
    }
};

// ---- Q5_K: same 16-byte meta as Q4_K (dm + scales), 5 code bits per value: the low 4 bits as
// in Q4_K, the high bit from qh at byte j_in_sub, bit sub_block. ----

template <> struct qskinny_codec<GGML_TYPE_Q5_K> {
    using record_t = qskinny_rec10;
    using meta_t   = uint4;
    static constexpr int record_bytes      = 10;
    static constexpr int meta_bytes        = 16;
    static constexpr int bits_per_value    = 5;
    static constexpr int block_bytes       = 176;
    static constexpr int values_per_record = 16;
    static constexpr int values_per_meta   = 256;
    static constexpr int values_per_sub_block = 32;

    static __host__ __device__ constexpr size_t codes_bytes(size_t n, size_t k) {
        return n * k * record_bytes / values_per_record;
    }

    static __device__ __forceinline__ qskinny_code load_record(const record_t * p) {
        const unsigned short * w = reinterpret_cast<const unsigned short *>(p);
        qskinny_code rec = { 0, 0 };
        rec.lo = (unsigned long long) w[0] | ((unsigned long long) w[1] << 16) |
                 ((unsigned long long) w[2] << 32) | ((unsigned long long) w[3] << 48);
        rec.hi = w[4];
        return rec;
    }

    static __device__ __forceinline__ meta_t load_meta(const uint8_t * p) {
        return __ldcs(reinterpret_cast<const meta_t *>(p));
    }

    static __device__ __forceinline__ void store_record(const qskinny_code rec, uint8_t * p) {
        unsigned short * w = reinterpret_cast<unsigned short *>(p);
        for (int i = 0; i < 4; ++i) {
            w[i] = (unsigned short) (rec.lo >> (16 * i));
        }
        w[4] = (unsigned short) rec.hi;
    }

    static __device__ __forceinline__ void store_meta(const meta_t meta, uint8_t * p) {
        *reinterpret_cast<meta_t *>(p) = meta;
    }

    static __device__ __forceinline__ meta_t pack_meta(const uint8_t * block) {
        return *reinterpret_cast<const meta_t *>(block);
    }

    // SoA in-place 320B group layout: lo[32] (uint2, 256B, Q4_K physical slot order) + hi[32] (uint16_t, 64B, bit slot = high bit)
    static __device__ __forceinline__ qskinny_code pack_record(const uint8_t * block, const int g) {
        const int sb = g >> 1;
        const uint8_t * ql = block + 48;
        const uint8_t * qh = block + 16;
        const int ql_base  = 32 * (g >> 2) + ((g & 1) ? 16 : 0);
        const int ql_shift = (g & 2) ? 4 : 0;
        const int qh_base  = 16 * (g & 1);
        qskinny_code rec = { 0, 0 };
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int slot = qpn8_physical_k(j);
            const unsigned lo4 = (ql[ql_base + j] >> ql_shift) & 0xF;
            const unsigned hi1 = (qh[qh_base + j] >> sb) & 1;
            rec.lo |= (unsigned long long) lo4 << (4 * slot);
            rec.hi |= (unsigned long long) hi1 << slot;
        }
        return rec;
    }

    static __device__ __forceinline__ void decode_record(const qskinny_code rec, half2 out[8]) {
        const unsigned rx = (unsigned) rec.lo;
        const unsigned ry = (unsigned) (rec.lo >> 32);
        const unsigned rh = (unsigned) rec.hi;
        const unsigned pairs[4] = {
            __byte_perm(rx, 0, 0x0200),  // bytes (0, 2) -> lo=(0,4), hi=(1,5)
            __byte_perm(rx, 0, 0x0301),  // bytes (1, 3) -> lo=(2,6), hi=(3,7)
            __byte_perm(ry, 0, 0x0200),  // bytes (4, 6) -> lo=(8,12), hi=(9,13)
            __byte_perm(ry, 0, 0x0301),  // bytes (5, 7) -> lo=(10,14), hi=(11,15)
        };
        const unsigned rh_pairs[2] = {
            (rh & 0x0Fu) | ((rh & 0xF0u) << 12),
            ((rh >> 8) & 0x0Fu) | (((rh >> 8) & 0xF0u) << 12),
        };
        const unsigned offset_bits = 0x64006400u; // f16 1024 in both halves
        const half2 offset = *reinterpret_cast<const half2 *>(&offset_bits);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned rhp = rh_pairs[i >> 1];
            const int sh_lo = 4 - 2 * (i & 1);
            const int sh_hi = 3 - 2 * (i & 1);
            const unsigned lo = (pairs[i] & 0x000F000Fu)        | ((rhp << sh_lo) & 0x00100010u) | offset_bits;
            const unsigned hi = ((pairs[i] >> 4) & 0x000F000Fu) | ((rhp << sh_hi) & 0x00100010u) | offset_bits;
            out[2*i + 0] = __hsub2(*reinterpret_cast<const half2 *>(&lo), offset);
            out[2*i + 1] = __hsub2(*reinterpret_cast<const half2 *>(&hi), offset);
        }
    }

    static __device__ __forceinline__ void decode(const qskinny_code records[2], const meta_t meta,
                                                  const int sub_block, half2 out[16]) {
        decode_record(records[0], out);
        decode_record(records[1], out + 8);

        uint8_t sc, m;
        qskinny_codec<GGML_TYPE_Q4_K>::get_scale_min(meta, sub_block, sc, m);
        const half2 dm = *reinterpret_cast<const half2 *>(&meta.x);
        const float dall = __half2float(__low2half(dm));
        const float dmin = __half2float(__high2half(dm));
        const half2 scale2 = __float2half2_rn(dall * (float) sc);
        const half2 bias2  = __float2half2_rn(-dmin * (float) m);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            out[i] = __hfma2(out[i], scale2, bias2);
        }
    }
};

// ---- Q6_K: 18 meta bytes per 256 K values (16 int8 scales + d), 6 code bits per value: the
// low 4 bits as in Q4_K, the high 2 bits from qh at byte 32*(sub_block>>2) + j_in_sub. ----

template <> struct qskinny_codec<GGML_TYPE_Q6_K> {
    using record_t = qskinny_rec12;
    struct meta_t {
        unsigned long long s_lo, s_hi;
        unsigned short d;
    };
    static constexpr int record_bytes      = 12;
    static constexpr int meta_bytes        = 18;
    static constexpr int bits_per_value    = 6;
    static constexpr int block_bytes       = 210;
    static constexpr int values_per_record = 16;
    static constexpr int values_per_meta   = 256;
    static constexpr int values_per_sub_block = 32;

    static __host__ __device__ constexpr size_t codes_bytes(size_t n, size_t k) {
        return n * k * record_bytes / values_per_record;
    }

    static __device__ __forceinline__ qskinny_code load_record(const record_t * p) {
        const unsigned * w = reinterpret_cast<const unsigned *>(p);
        return { (unsigned long long) w[0] | ((unsigned long long) w[1] << 32), w[2] };
    }

    static __device__ __forceinline__ meta_t load_meta(const uint8_t * p) {
        const unsigned short * w = reinterpret_cast<const unsigned short *>(p);
        meta_t meta;
        meta.s_lo = (unsigned long long) w[0] | ((unsigned long long) w[1] << 16) |
                    ((unsigned long long) w[2] << 32) | ((unsigned long long) w[3] << 48);
        meta.s_hi = (unsigned long long) w[4] | ((unsigned long long) w[5] << 16) |
                    ((unsigned long long) w[6] << 32) | ((unsigned long long) w[7] << 48);
        meta.d = w[8];
        return meta;
    }

    static __device__ __forceinline__ void store_record(const qskinny_code rec, uint8_t * p) {
        unsigned * w = reinterpret_cast<unsigned *>(p);
        w[0] = (unsigned) rec.lo;
        w[1] = (unsigned) (rec.lo >> 32);
        w[2] = (unsigned) rec.hi;
    }

    static __device__ __forceinline__ void store_meta(const meta_t meta, uint8_t * p) {
        unsigned short * w = reinterpret_cast<unsigned short *>(p);
        for (int i = 0; i < 4; ++i) {
            w[i]     = (unsigned short) (meta.s_lo >> (16 * i));
            w[4 + i] = (unsigned short) (meta.s_hi >> (16 * i));
        }
        w[8] = meta.d;
    }

    // scales[16] + d, copied unchanged
    static __device__ __forceinline__ meta_t pack_meta(const uint8_t * block) {
        return load_meta(block + 192);
    }

    // SoA in-place 384B group layout: lo[32] (uint2, 256B, Q4_K physical slot order) + hi[32] (uint32_t, 128B, bits 2*slot..2*slot+1)
    static __device__ __forceinline__ qskinny_code pack_record(const uint8_t * block, const int g) {
        const int sb = g >> 1;
        const uint8_t * ql = block;
        const uint8_t * qh = block + 128;
        const int ql_base  = 32 * (sb & 1) + 16 * (g & 1) + 64 * (sb >> 2);
        const int ql_shift = (sb & 2) ? 4 : 0;
        const int qh_base  = 16 * (g & 1) + 32 * (sb >> 2);
        const int qh_shift = 2 * (sb & 3);
        qskinny_code rec = { 0, 0 };
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int slot = qpn8_physical_k(j);
            const unsigned lo4 = (ql[ql_base + j] >> ql_shift) & 0xF;
            const unsigned hi2 = (qh[qh_base + j] >> qh_shift) & 3;
            rec.lo |= (unsigned long long) lo4 << (4 * slot);
            rec.hi |= (unsigned long long) hi2 << (2 * slot);
        }
        return rec;
    }

    static __device__ __forceinline__ void decode_record(const qskinny_code rec, half2 out[8], const half2 scale2) {
        const unsigned rx = (unsigned) rec.lo;
        const unsigned ry = (unsigned) (rec.lo >> 32);
        const unsigned rh = (unsigned) rec.hi;
        const unsigned pairs[4] = {
            __byte_perm(rx, 0, 0x0200),
            __byte_perm(rx, 0, 0x0301),
            __byte_perm(ry, 0, 0x0200),
            __byte_perm(ry, 0, 0x0301),
        };
        const unsigned rh_pairs[2] = {
            __byte_perm(rh, 0, 0x4140),
            __byte_perm(rh, 0, 0x4342),
        };
        const unsigned base_bits   = 0x64006400u; // f16 1024
        const unsigned offset_bits = 0x64206420u; // f16 1056 (1024 + 32)
        const half2 offset = *reinterpret_cast<const half2 *>(&offset_bits);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned rhp = rh_pairs[i >> 1];
            const unsigned h_lo = (i & 1) ? (rhp & 0x00300030u)        : ((rhp << 4) & 0x00300030u);
            const unsigned h_hi = (i & 1) ? ((rhp >> 2) & 0x00300030u) : ((rhp << 2) & 0x00300030u);
            const unsigned lo = (pairs[i] & 0x000F000Fu)        | h_lo | base_bits;
            const unsigned hi = ((pairs[i] >> 4) & 0x000F000Fu) | h_hi | base_bits;
            out[2*i + 0] = __hmul2(__hsub2(*reinterpret_cast<const half2 *>(&lo), offset), scale2);
            out[2*i + 1] = __hmul2(__hsub2(*reinterpret_cast<const half2 *>(&hi), offset), scale2);
        }
    }

    static __device__ __forceinline__ void decode(const qskinny_code records[2], const meta_t meta,
                                                  const int sub_block, half2 out[16]) {
        const float d = __half2float(__ushort_as_half(meta.d));
        const int sc0 = (int8_t) qskinny_scale_byte(meta.s_lo, meta.s_hi, 2 * sub_block + 0);
        const int sc1 = (int8_t) qskinny_scale_byte(meta.s_lo, meta.s_hi, 2 * sub_block + 1);
        decode_record(records[0], out,     __float2half2_rn(d * (float) sc0));
        decode_record(records[1], out + 8, __float2half2_rn(d * (float) sc1));
    }
};

// ---- Q2_K: 20 meta bytes per 256 K values (dm + 16 scale/min bytes), 2 code bits per value. ----

template <> struct qskinny_codec<GGML_TYPE_Q2_K> {
    using record_t = uint32_t;
    struct meta_t {
        unsigned long long s_lo, s_hi;
        unsigned dm;
    };
    static constexpr int record_bytes      = 4;
    static constexpr int meta_bytes        = 20;
    static constexpr int bits_per_value    = 2;
    static constexpr int block_bytes       = 84;
    static constexpr int values_per_record = 16;
    static constexpr int values_per_meta   = 256;
    static constexpr int values_per_sub_block = 32;

    static __host__ __device__ constexpr size_t codes_bytes(size_t n, size_t k) {
        return n * k * record_bytes / values_per_record;
    }

    static __device__ __forceinline__ qskinny_code load_record(const record_t * p) {
        return { __ldcs(p), 0 };
    }

    static __device__ __forceinline__ meta_t load_meta(const uint8_t * p) {
        const unsigned * w = reinterpret_cast<const unsigned *>(p);
        meta_t meta;
        meta.s_lo = (unsigned long long) w[1] | ((unsigned long long) w[2] << 32);
        meta.s_hi = (unsigned long long) w[3] | ((unsigned long long) w[4] << 32);
        meta.dm   = w[0];
        return meta;
    }

    static __device__ __forceinline__ void store_record(const qskinny_code rec, uint8_t * p) {
        *reinterpret_cast<uint32_t *>(p) = (uint32_t) rec.lo;
    }

    static __device__ __forceinline__ void store_meta(const meta_t meta, uint8_t * p) {
        unsigned * w = reinterpret_cast<unsigned *>(p);
        w[0] = meta.dm;
        w[1] = (unsigned) meta.s_lo;
        w[2] = (unsigned) (meta.s_lo >> 32);
        w[3] = (unsigned) meta.s_hi;
        w[4] = (unsigned) (meta.s_hi >> 32);
    }

    // dm lives after qs in the block, the scales before it: meta keeps dm first
    static __device__ __forceinline__ meta_t pack_meta(const uint8_t * block) {
        return load_meta_from(block + 80, block);
    }

    static __device__ __forceinline__ meta_t load_meta_from(const uint8_t * dm_src, const uint8_t * scales_src) {
        meta_t meta;
        meta.dm   = *reinterpret_cast<const unsigned *>(dm_src);
        meta.s_lo = (unsigned long long) reinterpret_cast<const unsigned *>(scales_src)[0] |
                    ((unsigned long long) reinterpret_cast<const unsigned *>(scales_src)[1] << 32);
        meta.s_hi = (unsigned long long) reinterpret_cast<const unsigned *>(scales_src)[2] |
                    ((unsigned long long) reinterpret_cast<const unsigned *>(scales_src)[3] << 32);
        return meta;
    }

    static __device__ __forceinline__ qskinny_code pack_record(const uint8_t * block, const int g) {
        const int sb = g >> 1;
        const uint8_t * qs = block + 16;
        const int qs_base  = 32 * (sb >> 2) + 16 * (g & 1);
        const int qs_shift = 2 * (sb & 3);
        qskinny_code rec = { 0, 0 };
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int slot = qpn8_physical_k(j);
            qskinny_code_put(rec, (qs[qs_base + j] >> qs_shift) & 3, 2 * slot, 2);
        }
        return rec;
    }

    static __device__ __forceinline__ void decode(const qskinny_code records[2], const meta_t meta,
                                                  const int sub_block, half2 out[16]) {
        const float dall = __half2float(__ushort_as_half((unsigned short) meta.dm));
        const float dmin = __half2float(__ushort_as_half((unsigned short) (meta.dm >> 16)));
        const unsigned offset_bits = 0x64006400u; // f16 1024 in both halves
        const half2 offset = *reinterpret_cast<const half2 *>(&offset_bits);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int is = 2 * sub_block + r;
            const int sc = qskinny_scale_byte(meta.s_lo, meta.s_hi, is) & 0xF;
            const int mn = qskinny_scale_byte(meta.s_lo, meta.s_hi, is) >> 4;
            // f16 folding, rounded once to f16 like the Q4_K decoder
            const half2 scale2 = __float2half2_rn(dall * (float) sc);
            const half2 bias2  = __float2half2_rn(-dmin * (float) mn);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int s = (i & 3) + 8 * (i >> 2);
                const unsigned q0 = qskinny_code_get(records[r], 2 * s, 2);
                const unsigned q1 = qskinny_code_get(records[r], 2 * (s + 4), 2);
                const unsigned h = (0x6400u | q0) | ((0x6400u | q1) << 16);
                out[8*r + i] = __hfma2(__hsub2(*reinterpret_cast<const half2 *>(&h), offset),
                                       scale2, bias2);
            }
        }
    }
};

// ---- Q3_K: 14 meta bytes per 256 K values (12 scale bytes + d), 3 code bits per value: the
// low 2 bits as in Q2_K, the sign bit from hmask at byte j_in_sub, bit sub_block. ----

template <> struct qskinny_codec<GGML_TYPE_Q3_K> {
    using record_t = qskinny_rec6;
    struct meta_t {
        unsigned long long s_lo, s_hi;
    };
    static constexpr int record_bytes      = 6;
    static constexpr int meta_bytes        = 14;
    static constexpr int bits_per_value    = 3;
    static constexpr int block_bytes       = 110;
    static constexpr int values_per_record = 16;
    static constexpr int values_per_meta   = 256;
    static constexpr int values_per_sub_block = 32;

    static __host__ __device__ constexpr size_t codes_bytes(size_t n, size_t k) {
        return n * k * record_bytes / values_per_record;
    }

    static __device__ __forceinline__ qskinny_code load_record(const record_t * p) {
        const unsigned short * w = reinterpret_cast<const unsigned short *>(p);
        return { (unsigned long long) w[0] | ((unsigned long long) w[1] << 16) |
                 ((unsigned long long) w[2] << 32), 0 };
    }

    static __device__ __forceinline__ meta_t load_meta(const uint8_t * p) {
        const unsigned short * w = reinterpret_cast<const unsigned short *>(p);
        meta_t meta;
        meta.s_lo = (unsigned long long) w[0] | ((unsigned long long) w[1] << 16) |
                    ((unsigned long long) w[2] << 32) | ((unsigned long long) w[3] << 48);
        // scales[8..11] then d, zero padded into the second word
        meta.s_hi = (unsigned long long) w[4] | ((unsigned long long) w[5] << 16) |
                    ((unsigned long long) w[6] << 32);
        return meta;
    }

    static __device__ __forceinline__ void store_record(const qskinny_code rec, uint8_t * p) {
        unsigned short * w = reinterpret_cast<unsigned short *>(p);
        w[0] = (unsigned short) rec.lo;
        w[1] = (unsigned short) (rec.lo >> 16);
        w[2] = (unsigned short) (rec.lo >> 32);
    }

    static __device__ __forceinline__ void store_meta(const meta_t meta, uint8_t * p) {
        unsigned short * w = reinterpret_cast<unsigned short *>(p);
        for (int i = 0; i < 4; ++i) {
            w[i] = (unsigned short) (meta.s_lo >> (16 * i));
        }
        for (int i = 0; i < 3; ++i) {
            w[4 + i] = (unsigned short) (meta.s_hi >> (16 * i));
        }
    }

    // scales[12] + d, copied unchanged
    static __device__ __forceinline__ meta_t pack_meta(const uint8_t * block) {
        return load_meta(block + 96);
    }

    static __device__ __forceinline__ qskinny_code pack_record(const uint8_t * block, const int g) {
        const int sb = g >> 1;
        const uint8_t * qs    = block + 32;
        const uint8_t * hmask = block;
        const int qs_base  = 32 * (sb >> 2) + 16 * (g & 1);
        const int qs_shift = 2 * (sb & 3);
        const int hm_base  = 16 * (g & 1);
        qskinny_code rec = { 0, 0 };
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int slot = qpn8_physical_k(j);
            const unsigned code = ((qs[qs_base + j] >> qs_shift) & 3) |
                                  (((hmask[hm_base + j] >> sb) & 1) << 2);
            qskinny_code_put(rec, code, 3 * slot, 3);
        }
        return rec;
    }

    static __device__ __forceinline__ void decode(const qskinny_code records[2], const meta_t meta,
                                                  const int sub_block, half2 out[16]) {
        const float d_all = __half2float(__ushort_as_half((unsigned short) (meta.s_hi >> 32)));
        // f16 folding: the effective code is code2 - (hbit ? 0 : 4), in [-4, 3]. 1024 + code2 + 8
        // (hbit set) or 1024 + code2 + 4 (hbit clear) minus 1032 gives it exactly, in one hsub2.
        const unsigned offset_bits = 0x64086408u; // f16 1032 in both halves
        const half2 offset = *reinterpret_cast<const half2 *>(&offset_bits);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int is = 2 * sub_block + r;
            // the 6-bit scale unpack of dequantize.cuh
            const int8_t us = is <  4 ? (qskinny_scale_byte(meta.s_lo, meta.s_hi, is - 0) & 0xF) |
                                        (((qskinny_scale_byte(meta.s_lo, meta.s_hi, is + 8) >> 0) & 3) << 4) :
                              is <  8 ? (qskinny_scale_byte(meta.s_lo, meta.s_hi, is - 0) & 0xF) |
                                        (((qskinny_scale_byte(meta.s_lo, meta.s_hi, is + 4) >> 2) & 3) << 4) :
                              is < 12 ? (qskinny_scale_byte(meta.s_lo, meta.s_hi, is - 8) >>  4) |
                                        (((qskinny_scale_byte(meta.s_lo, meta.s_hi, is + 0) >> 4) & 3) << 4) :
                                        (qskinny_scale_byte(meta.s_lo, meta.s_hi, is - 8) >>  4) |
                                        (((qskinny_scale_byte(meta.s_lo, meta.s_hi, is - 4) >> 6) & 3) << 4);
            const half2 scale2 = __float2half2_rn(d_all * (float) (us - 32));
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int s = (i & 3) + 8 * (i >> 2);
                const unsigned c0 = qskinny_code_get(records[r], 3 * s, 3);
                const unsigned c1 = qskinny_code_get(records[r], 3 * (s + 4), 3);
                const unsigned h0 = 0x6400u | ((c0 & 3) + ((c0 & 4) ? 8 : 4));
                const unsigned h1 = 0x6400u | ((c1 & 3) + ((c1 & 4) ? 8 : 4));
                const unsigned h = h0 | (h1 << 16);
                out[8*r + i] = __hmul2(__hsub2(*reinterpret_cast<const half2 *>(&h), offset), scale2);
            }
        }
    }
};


// ---- IQ4_XS: 8 code bytes per 16 K values and lane (16 nibbles in physical slot order, as Q4_K) and 8 meta
// bytes per 256 K values (the raw super-block header: d, scales_h, scales_l[4]). A nibble indexes the
// non-linear table kvalues_iq4nl, a 32-value sub-block has one 6-bit scale: w = d*(ls - 32)*kvalues_iq4nl[q].
// 16*8 + 8 == sizeof(block_iq4_xs), so the repack stays in place. ----

// kvalues_iq4nl as f16 bit patterns (the 16 values are exact in f16)
// kvalues_iq4nl decoded via 4-register __byte_perm LUT (0 CALL, 0 SHFL, bit-exact f16)

template <> struct qskinny_codec<GGML_TYPE_IQ4_XS> {
    using record_t = uint2;
    using meta_t   = uint2;
    static constexpr int record_bytes      = 8;
    static constexpr int meta_bytes        = 8;
    static constexpr int bits_per_value    = 4;
    static constexpr int block_bytes       = 136;
    static constexpr int values_per_record = 16;
    static constexpr int values_per_meta   = 256;
    static constexpr int values_per_sub_block = 32;

    static __host__ __device__ constexpr size_t codes_bytes(size_t n, size_t k) {
        return n * k * record_bytes / values_per_record;
    }

    static __device__ __forceinline__ qskinny_code load_record(const record_t * p) {
        const uint2 r = __ldcs(p);
        return { (unsigned long long) r.x | ((unsigned long long) r.y << 32), 0 };
    }

    static __device__ __forceinline__ meta_t load_meta(const uint8_t * p) {
        return __ldcs(reinterpret_cast<const meta_t *>(p));
    }

    static __device__ __forceinline__ void store_record(const qskinny_code rec, uint8_t * p) {
        *reinterpret_cast<uint2 *>(p) = make_uint2((unsigned) rec.lo, (unsigned) (rec.lo >> 32));
    }

    static __device__ __forceinline__ void store_meta(const meta_t meta, uint8_t * p) {
        *reinterpret_cast<meta_t *>(p) = meta;
    }

    // the raw 8-byte super-block header (d, scales_h, scales_l), copied unchanged
    static __device__ __forceinline__ meta_t pack_meta(const uint8_t * block) {
        return *reinterpret_cast<const meta_t *>(block);
    }

    // sub-block g/2 uses 16 consecutive qs bytes: the low nibbles are its first 16 values (g&1 == 0),
    // the high nibbles its second 16
    static __device__ __forceinline__ qskinny_code pack_record(const uint8_t * block, const int g) {
        const uint8_t * qs = block + 8;
        const int qs_base  = 16 * (g >> 1);
        const int qs_shift = (g & 1) ? 4 : 0;
        qskinny_code rec = { 0, 0 };
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int slot = qpn8_physical_k(j);
            qskinny_code_put(rec, (qs[qs_base + j] >> qs_shift) & 0xF, 4 * slot, 4);
        }
        return rec;
    }

    // 4-register __byte_perm LUT for kvalues_iq4nl + 128 (0 CALL, 0 SHFL, exact f16).
    static __device__ __forceinline__ void decode_4nibbles(const unsigned v, const half2 neg1152,
                                                           half2 & out_even, half2 & out_odd) {
        const unsigned t    = v & 0x07070707u;
        const unsigned a    = (t | (t >> 4)) & 0x00770077u;
        const unsigned sel  = a | (a >> 8);
        const unsigned neg4 = __byte_perm(0x3F2D1801u, 0x766A5D4Fu, sel);
        const unsigned pos4 = __byte_perm(0xA6998D81u, 0xF1D9C5B5u, sel);
        const unsigned m4   = ((v >> 3) & 0x01010101u) * 0xFFu;
        const unsigned u4   = (pos4 & m4) | (neg4 & ~m4);
        const unsigned h02  = __byte_perm(u4, 0x6400u, 0x5250u);
        const unsigned h13  = __byte_perm(u4, 0x6400u, 0x5351u);
        out_even = __hadd2(*reinterpret_cast<const half2 *>(&h02), neg1152);
        out_odd  = __hadd2(*reinterpret_cast<const half2 *>(&h13), neg1152);
    }

    // Same record layout and slot pairing as Q4_K, each nibble mapped through the 4-reg PRMT table.
    static __device__ __forceinline__ void decode_record(const qskinny_code rec, half2 out[8]) {
        const unsigned rx = (unsigned) rec.lo;
        const unsigned ry = (unsigned) (rec.lo >> 32);
        const unsigned neg1152_bits = 0xE480E480u;
        const half2 neg1152 = *reinterpret_cast<const half2 *>(&neg1152_bits);
        decode_4nibbles(rx,      neg1152, out[0], out[2]);
        decode_4nibbles(rx >> 4, neg1152, out[1], out[3]);
        decode_4nibbles(ry,      neg1152, out[4], out[6]);
        decode_4nibbles(ry >> 4, neg1152, out[5], out[7]);
    }

    static __device__ __forceinline__ void decode(const qskinny_code records[2], const uint2 meta,
                                                  const int sub_block, half2 out[16]) {
        decode_record(records[0], out);
        decode_record(records[1], out + 8);

        // meta.x = d | scales_h << 16, meta.y = scales_l[0..3]
        const float    d        = __half2float(__ushort_as_half((unsigned short) (meta.x & 0xFFFF)));
        const unsigned scales_h = meta.x >> 16;
        const int      ls       = (int) (((meta.y >> (8 * (sub_block >> 1) + 4 * (sub_block & 1))) & 0xF) |
                                         (((scales_h >> (2 * sub_block)) & 3) << 4));
        // f32 folding of the scale, rounded once to f16 (as the k-quant decoders do)
        const half2 scale2 = __float2half2_rn(d * (float) (ls - 32));
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            out[i] = __hmul2(out[i], scale2);
        }
    }
};

static_assert(qskinny_codec<GGML_TYPE_IQ4_XS>::block_bytes == sizeof(block_iq4_xs), "iq4_xs block size");
static_assert(qskinny_codec<GGML_TYPE_IQ4_XS>::record_bytes * 16 + qskinny_codec<GGML_TYPE_IQ4_XS>::meta_bytes == qskinny_codec<GGML_TYPE_IQ4_XS>::block_bytes, "iq4_xs packing");

// ---- IQ4_NL: 8 consecutive block_iq4_nl (18B each = 144B per 256 values) -> 16*8B codes + 16B meta (8 half scales). ----
template <> struct qskinny_codec<GGML_TYPE_IQ4_NL> {
    using record_t = uint2;
    using meta_t   = uint4;
    static constexpr int record_bytes         = 8;
    static constexpr int meta_bytes           = 16;
    static constexpr int bits_per_value       = 4;
    static constexpr int block_bytes          = 144;
    static constexpr int values_per_record    = 16;
    static constexpr int values_per_meta      = 256;
    static constexpr int values_per_sub_block = 32;

    static __host__ __device__ constexpr size_t codes_bytes(size_t n, size_t k) {
        return n * k * record_bytes / values_per_record;
    }

    static __device__ __forceinline__ qskinny_code load_record(const record_t * p) {
        const uint2 r = __ldcs(p);
        return { (unsigned long long) r.x | ((unsigned long long) r.y << 32), 0 };
    }

    static __device__ __forceinline__ meta_t load_meta(const uint8_t * p) {
        return __ldcs(reinterpret_cast<const meta_t *>(p));
    }

    static __device__ __forceinline__ void store_record(const qskinny_code rec, uint8_t * p) {
        *reinterpret_cast<uint2 *>(p) = make_uint2((unsigned) rec.lo, (unsigned) (rec.lo >> 32));
    }

    static __device__ __forceinline__ void store_meta(const meta_t meta, uint8_t * p) {
        *reinterpret_cast<meta_t *>(p) = meta;
    }

    static __device__ __forceinline__ meta_t pack_meta(const uint8_t * block) {
        unsigned d[8];
#pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            d[ib] = *reinterpret_cast<const uint16_t *>(block + 18 * ib);
        }
        return make_uint4(d[0] | (d[1] << 16), d[2] | (d[3] << 16), d[4] | (d[5] << 16), d[6] | (d[7] << 16));
    }

    static __device__ __forceinline__ qskinny_code pack_record(const uint8_t * block, const int g) {
        const int ib = g >> 1;
        const uint8_t * qs = block + 18 * ib + 2;
        const int qs_shift = (g & 1) ? 4 : 0;
        qskinny_code rec = { 0, 0 };
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int slot = qpn8_physical_k(j);
            qskinny_code_put(rec, (qs[j] >> qs_shift) & 0xF, 4 * slot, 4);
        }
        return rec;
    }

    static __device__ __forceinline__ void decode(const qskinny_code records[2], const uint4 meta,
                                                  const int sub_block, half2 out[16]) {
        qskinny_codec<GGML_TYPE_IQ4_XS>::decode_record(records[0], out);
        qskinny_codec<GGML_TYPE_IQ4_XS>::decode_record(records[1], out + 8);
        const unsigned word = (sub_block < 2) ? meta.x : (sub_block < 4) ? meta.y : (sub_block < 6) ? meta.z : meta.w;
        const uint16_t d_u16 = (uint16_t) ((sub_block & 1) ? (word >> 16) : (word & 0xFFFF));
        const half2 scale2 = __half2half2(__ushort_as_half(d_u16));
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            out[i] = __hmul2(out[i], scale2);
        }
    }
};

static_assert(qskinny_codec<GGML_TYPE_IQ4_NL>::block_bytes == 8 * sizeof(block_iq4_nl), "iq4_nl 256-val super-block size");
static_assert(qskinny_codec<GGML_TYPE_IQ4_NL>::record_bytes * 16 + qskinny_codec<GGML_TYPE_IQ4_NL>::meta_bytes == qskinny_codec<GGML_TYPE_IQ4_NL>::block_bytes, "iq4_nl packing");

// ---- group-level record load/store (default AoS; SoA coalesced __ldcs for Q5_K and Q6_K) ----
template <ggml_type T>
static __device__ __forceinline__ qskinny_code qskinny_load_group_record(const uint8_t * group_base, const int lane) {
    using codec = qskinny_codec<T>;
    return codec::load_record(reinterpret_cast<const typename codec::record_t *>(group_base) + lane);
}

template <ggml_type T>
static __device__ __forceinline__ void qskinny_store_group_record(const qskinny_code rec, uint8_t * group_base, const int lane) {
    using codec = qskinny_codec<T>;
    codec::store_record(rec, group_base + lane * codec::record_bytes);
}

template <>
__device__ __forceinline__ qskinny_code qskinny_load_group_record<GGML_TYPE_Q5_K>(const uint8_t * group_base, const int lane) {
    const uint2 r = __ldcs(reinterpret_cast<const uint2 *>(group_base) + lane);
    const uint16_t h = __ldcs(reinterpret_cast<const uint16_t *>(group_base + 256) + lane);
    return { (unsigned long long) r.x | ((unsigned long long) r.y << 32), (unsigned long long) h };
}

template <>
__device__ __forceinline__ void qskinny_store_group_record<GGML_TYPE_Q5_K>(const qskinny_code rec, uint8_t * group_base, const int lane) {
    *reinterpret_cast<uint2 *>(group_base + lane * 8) = make_uint2((unsigned) rec.lo, (unsigned) (rec.lo >> 32));
    *reinterpret_cast<uint16_t *>(group_base + 256 + lane * 2) = (uint16_t) rec.hi;
}

template <>
__device__ __forceinline__ qskinny_code qskinny_load_group_record<GGML_TYPE_Q6_K>(const uint8_t * group_base, const int lane) {
    const uint2 r = __ldcs(reinterpret_cast<const uint2 *>(group_base) + lane);
    const unsigned h = __ldcs(reinterpret_cast<const unsigned *>(group_base + 256) + lane);
    return { (unsigned long long) r.x | ((unsigned long long) r.y << 32), (unsigned long long) h };
}

template <>
__device__ __forceinline__ void qskinny_store_group_record<GGML_TYPE_Q6_K>(const qskinny_code rec, uint8_t * group_base, const int lane) {
    *reinterpret_cast<uint2 *>(group_base + lane * 8) = make_uint2((unsigned) rec.lo, (unsigned) (rec.lo >> 32));
    *reinterpret_cast<unsigned *>(group_base + 256 + lane * 4) = (unsigned) rec.hi;
}

static_assert(qskinny_codec<GGML_TYPE_Q2_K>::block_bytes == sizeof(block_q2_K), "q2_K block size");
static_assert(qskinny_codec<GGML_TYPE_Q3_K>::block_bytes == sizeof(block_q3_K), "q3_K block size");
static_assert(qskinny_codec<GGML_TYPE_Q4_K>::block_bytes == sizeof(block_q4_K), "q4_K block size");
static_assert(qskinny_codec<GGML_TYPE_Q5_K>::block_bytes == sizeof(block_q5_K), "q5_K block size");
static_assert(qskinny_codec<GGML_TYPE_Q6_K>::block_bytes == sizeof(block_q6_K), "q6_K block size");
static_assert(qskinny_codec<GGML_TYPE_Q2_K>::record_bytes * 16 + qskinny_codec<GGML_TYPE_Q2_K>::meta_bytes == qskinny_codec<GGML_TYPE_Q2_K>::block_bytes, "q2_K packing");
static_assert(qskinny_codec<GGML_TYPE_Q3_K>::record_bytes * 16 + qskinny_codec<GGML_TYPE_Q3_K>::meta_bytes == qskinny_codec<GGML_TYPE_Q3_K>::block_bytes, "q3_K packing");
static_assert(qskinny_codec<GGML_TYPE_Q4_K>::record_bytes * 16 + qskinny_codec<GGML_TYPE_Q4_K>::meta_bytes == qskinny_codec<GGML_TYPE_Q4_K>::block_bytes, "q4_K packing");
static_assert(qskinny_codec<GGML_TYPE_Q5_K>::record_bytes * 16 + qskinny_codec<GGML_TYPE_Q5_K>::meta_bytes == qskinny_codec<GGML_TYPE_Q5_K>::block_bytes, "q5_K packing");
static_assert(qskinny_codec<GGML_TYPE_Q6_K>::record_bytes * 16 + qskinny_codec<GGML_TYPE_Q6_K>::meta_bytes == qskinny_codec<GGML_TYPE_Q6_K>::block_bytes, "q6_K packing");

// ---- repack: row-major block -> codes + meta ----

// One CTA per tile. src points at the first tile of the range, codes at its codes
// destination and meta at the full meta array indexed by the global tile.
template <ggml_type T>
__global__ void qskinny_repack_kernel(const uint8_t * __restrict__ src, uint8_t * __restrict__ codes,
                                      uint8_t * __restrict__ meta, int n, int k, int tile0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    using codec = qskinny_codec<T>;
    const int blocks_k = k >> 8;
    const int tile = tile0 + blockIdx.x;
    uint8_t * codes_tile = codes + (size_t) blockIdx.x * 32 * (k >> 4) * codec::record_bytes;
    for (int i = threadIdx.x; i < 32 * blocks_k; i += blockDim.x) {
        const int row = i / blocks_k;
        const int kb  = i - row * blocks_k;
        const uint8_t * block = src + ((size_t) blockIdx.x * 32 + row) * blocks_k * codec::block_bytes
                                         + (size_t) kb * codec::block_bytes;
        const int lane = qpn8_lane_from_col(row & 31);

        codec::store_meta(codec::pack_meta(block),
                          meta + ((size_t) kb * n + (size_t) tile * 32 + lane) * codec::meta_bytes);

        // 16 groups of 16 values per super-block, group g holds the first (g&1 == 0) or second
        // 16 values of sub-block g/2
#pragma unroll
        for (int g = 0; g < 16; ++g) {
            qskinny_store_group_record<T>(codec::pack_record(block, g),
                                          codes_tile + ((size_t) kb * 16 + g) * 32 * codec::record_bytes, lane);
        }
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(src, codes, meta, n, k, tile0);
#endif
}

// ---- dequantize: codes + meta -> dense F16 [N][K], same layout as the Q4_K to F16 conversion ----

// One CTA per (tile of 32 rows, 256-K super-block). A warp decodes one 32-value sub-block per
// lane; a code record decodes to one output row, so the warps stage the decoded values in
// shared memory first. That keeps the stores to global memory coalesced along K instead of one
// 4-byte store per row.
template <ggml_type T>
__global__ void qskinny_to_f16_kernel(half * __restrict__ output, const uint8_t * __restrict__ codes,
                                      int n, int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    using codec = qskinny_codec<T>;
    // values_per_meta / 2 half2 per row plus 4 half2 of padding
    __shared__ __align__(16) half2 tile_smem[32][codec::values_per_meta / 2 + 4];
    __shared__ __align__(16) typename codec::meta_t meta_smem[32];

    const int lane   = threadIdx.x & 31;
    const int warp   = threadIdx.x >> 5;
    const int tile   = blockIdx.y;
    const int kb     = blockIdx.x;
    const int groups = k >> 4;

    // one meta per lane, for the row this lane holds the codes of
    if (warp == 0) {
        meta_smem[lane] = codec::load_meta(codes + codec::codes_bytes(n, k) +
                                           (size_t) codec::meta_bytes * ((size_t) kb * n + tile * 32 + lane));
    }
    __syncthreads();

    // sub-block `warp` of this super-block: records 2*warp and 2*warp+1
    const int group = kb * (codec::values_per_meta / codec::values_per_record) + 2 * warp;
    const qskinny_code records[2] = {
        qskinny_load_group_record<T>(codes + ((size_t) tile * groups + group + 0) * 32 * codec::record_bytes, lane),
        qskinny_load_group_record<T>(codes + ((size_t) tile * groups + group + 1) * 32 * codec::record_bytes, lane),
    };
    half2 weights[codec::values_per_sub_block / 2];
    codec::decode(records, meta_smem[lane], warp, weights);

    half2 * smem_row = tile_smem[qpn8_col_from_lane(lane)];
#pragma unroll
    for (int i = 0; i < codec::values_per_sub_block / 2; ++i) {
        smem_row[warp * (codec::values_per_sub_block / 2) + i] = weights[i];
    }
    __syncthreads();

    // 4 uint4 (four half2 each) per thread
    half * out_tile = output + (size_t) tile * 32 * k + (size_t) kb * codec::values_per_meta;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int idx = threadIdx.x + i * 256;
        const int row = idx >> 5;
        const int col = (idx & 31) << 2;
        const uint4 value = *reinterpret_cast<const uint4 *>(&tile_smem[row][col]);
        *reinterpret_cast<uint4 *>(out_tile + (size_t) row * k + col * 2) = value;
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(output, codes, n, k);
#endif
}

// ---- 1Cat fp8_qpn8_sm70.cu:168-175 (mma.m8n8k4 macro, unchanged) ----

#define Q4K_SKINNY_MMA_8N8K4(C, A0, A1, B0, B1)                      \
  asm volatile(                                                      \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "              \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                 \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]),  \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                           \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// ---- 1Cat fp8_qpn8_sm70.cu:207-432 (main kernel, adapted to the codec) ----

template <ggml_type T, int SplitK, int NAcc, bool M1Only = false, int RowTiles = 1>
__global__ void q4k_skinny_kernel(
    const uint8_t * __restrict__ codes, const half * __restrict__ input,
    float * __restrict__ output, int n, int k, int m) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    using codec = qskinny_codec<T>;
    static_assert(RowTiles == 1 || RowTiles == 2,
                  "Q4_K skinny supports one or two 8-row tiles");
    static_assert(!M1Only || RowTiles == 1,
                  "Q4_K skinny M=1 specialization uses one row tile");
    __shared__ float partials[SplitK][M1Only ? 32 : RowTiles * 256];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tile = blockIdx.x;
    const int quadpair = (lane >> 2) & 3;
    const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_k16 = k >> 4;
    // one iteration per 32-value sub-block; SplitK divides the sub-block count, the meta is
    // reloaded whenever the 256-value super-block index changes
    const int sub_blocks = k >> 5;
    const int sub_blocks_per_warp = sub_blocks / SplitK;
    const int sub_block_begin = warp * sub_blocks_per_warp;
    const uint8_t * code_base = codes + (size_t) tile * groups_k16 * 32 * codec::record_bytes;
    // meta stride and offset are per codec type
    const uint8_t * meta_ptr = codes + codec::codes_bytes(n, k) +
                               (size_t) codec::meta_bytes * (tile * 32 + lane);

    float accum[RowTiles][NAcc][8];
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
        for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                accum[row_tile][chain][i] = 0.0f;
            }
        }
    }
    int loaded_meta_kb = -1;
    typename codec::meta_t meta = {};

#pragma unroll 2
    for (int sub_block = sub_block_begin; sub_block < sub_block_begin + sub_blocks_per_warp; ++sub_block) {
        const int kb = sub_block >> 3;
        if (kb != loaded_meta_kb) {
            // one meta per 256 values; shared by the 8 sub-blocks that follow
            meta = codec::load_meta(meta_ptr + (size_t) kb * n * codec::meta_bytes);
            loaded_meta_kb = kb;
        }

        const int group = sub_block << 1;
        const qskinny_code records[2] = { // record load per codec type
            qskinny_load_group_record<T>(code_base + (size_t) (group + 0) * 32 * codec::record_bytes, lane),
            qskinny_load_group_record<T>(code_base + (size_t) (group + 1) * 32 * codec::record_bytes, lane),
        };
        half2 weights[codec::values_per_sub_block / 2];
        codec::decode(records, meta, sub_block & 7, weights);

        const unsigned * b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
            uint4 input01 = make_uint4(0, 0, 0, 0);
            uint4 input23 = make_uint4(0, 0, 0, 0);
            uint4 input45 = make_uint4(0, 0, 0, 0);
            uint4 input67 = make_uint4(0, 0, 0, 0);
            const int input_row_idx = row_tile * 8 + row;
            if (input_row_idx < m) {
                const half * input_row = input + (size_t) input_row_idx * k;
                input01 = *reinterpret_cast<const uint4 *>(input_row + group * 16);
                input23 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 8);
                input45 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 16);
                input67 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 24);
            }

            const unsigned * a0 = reinterpret_cast<const unsigned *>(&input01);
            const unsigned * a1 = reinterpret_cast<const unsigned *>(&input23);
            const unsigned * a2 = reinterpret_cast<const unsigned *>(&input45);
            const unsigned * a3 = reinterpret_cast<const unsigned *>(&input67);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][0], a2[0], a2[1], b[8], b[9]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][1 % NAcc], a2[2], a2[3], b[10], b[11]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][2 % NAcc], a3[0], a3[1], b[12], b[13]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][3 % NAcc], a3[2], a3[3], b[14], b[15]);
        }
    }

#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
        for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                accum[row_tile][0][i] += accum[row_tile][chain][i];
            }
        }
    }

    if constexpr (M1Only) {
        if ((lane & 17) == 0) {
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
                for (int offset = 0; offset < 2; ++offset) {
                    const int i = pair * 4 + offset;
                    const int output_col =
                        offset | (((lane >> 1) & 1) << 1) | (pair << 2);
                    partials[warp][quadpair * 8 + output_col] = accum[0][0][i];
                }
            }
        }
    } else {
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int output_row =
                    row_tile * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
                const int output_col =
                    (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
                partials[warp][output_row * 32 + quadpair * 8 + output_col] =
                    accum[row_tile][0][i];
            }
        }
    }
    __syncthreads();

    // fixed reduction order over the split-K warps keeps the result reproducible
    constexpr int kOutputElements = M1Only ? 32 : RowTiles * 256;
    for (int element = threadIdx.x; element < kOutputElements;
         element += blockDim.x) {
        float value = 0.0f;
#pragma unroll
        for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
            value += partials[k_warp][element];
        }
        if constexpr (M1Only) {
            output[tile * 32 + element] = value;
        } else {
            const int output_row = element >> 5;
            const int output_col = element & 31;
            if (output_row < m) {
                output[(size_t) output_row * n + tile * 32 + output_col] = value;
            }
        }
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(codes, input, output, n, k, m);
#endif
}

// ---- NTiles variant of the skinny GEMM kernel (custom, v100fast) ----
// The stock kernel takes one tile of 32 weight rows per CTA, so every CTA streams the whole input (m rows x K
// halves) to process one tile: at m = 16 the input bytes per CTA exceed the weight bytes and the input loads
// are the largest cost (measured: 150 us -> 90 us on q4_K 17408x5120 when they are removed). Here a CTA takes
// NTiles tiles and loads each sub-block's input fragments once for all of them.
template <ggml_type T, int SplitK, int NAcc, int RowTiles, int NTiles, int MinBlocks = (512 / (32 * SplitK))>
__launch_bounds__(32 * SplitK, MinBlocks)
__global__ void q4k_skinny_nt_kernel(
    const uint8_t * __restrict__ codes, const half * __restrict__ input,
    float * __restrict__ output, int n, int k, int m) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    using codec = qskinny_codec<T>;
    __shared__ float partials[SplitK][RowTiles * 256];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tile0 = blockIdx.x * NTiles;
    const int quadpair = (lane >> 2) & 3;
    const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_k16 = k >> 4;
    const int sub_blocks = k >> 5;
    const int sub_blocks_per_warp = sub_blocks / SplitK;
    const int sub_block_begin = warp * sub_blocks_per_warp;
    const size_t tile_bytes = (size_t) groups_k16 * 32 * codec::record_bytes;
    const uint8_t * code_base = codes + (size_t) tile0 * tile_bytes;
    const uint8_t * meta_ptr = codes + codec::codes_bytes(n, k) +
                               (size_t) codec::meta_bytes * ((size_t) tile0 * 32 + lane);

    float accum[NTiles][RowTiles][NAcc][8];
#pragma unroll
    for (int t = 0; t < NTiles; ++t) {
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
            for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    accum[t][row_tile][chain][i] = 0.0f;
                }
            }
        }
    }
    int loaded_meta_kb = -1;
    typename codec::meta_t meta[NTiles] = {};

    for (int sub_block = sub_block_begin; sub_block < sub_block_begin + sub_blocks_per_warp; ++sub_block) {
        const int kb = sub_block >> 3;
        if (kb != loaded_meta_kb) {
#pragma unroll
            for (int t = 0; t < NTiles; ++t) {
                meta[t] = codec::load_meta(meta_ptr + ((size_t) kb * n + (size_t) t * 32) * codec::meta_bytes);
            }
            loaded_meta_kb = kb;
        }
        const int group = sub_block << 1;

        // the input fragments of this sub-block, loaded once for all tiles
        uint4 in[RowTiles][4];
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
            const int input_row_idx = row_tile * 8 + row;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                in[row_tile][j] = make_uint4(0, 0, 0, 0);
            }
            if (input_row_idx < m) {
                const half * input_row = input + (size_t) input_row_idx * k + group * 16;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    in[row_tile][j] = *reinterpret_cast<const uint4 *>(input_row + 8 * j);
                }
            }
        }

        // Cross-tile HBM2 load overlap for codecs with ample register headroom (Q4_K/Q5_K/Q6_K)
        qskinny_code pf_records[NTiles][2];
        if constexpr (T == GGML_TYPE_Q4_K || T == GGML_TYPE_Q5_K || T == GGML_TYPE_Q6_K || T == GGML_TYPE_IQ4_XS || T == GGML_TYPE_IQ4_NL) {
#pragma unroll
            for (int t = 0; t < NTiles; ++t) {
                pf_records[t][0] = qskinny_load_group_record<T>(code_base + (size_t) t * tile_bytes + (size_t) (group + 0) * 32 * codec::record_bytes, lane);
                pf_records[t][1] = qskinny_load_group_record<T>(code_base + (size_t) t * tile_bytes + (size_t) (group + 1) * 32 * codec::record_bytes, lane);
            }
        }
#pragma unroll
        for (int t = 0; t < NTiles; ++t) {
            qskinny_code records[2];
            if constexpr (T == GGML_TYPE_Q4_K || T == GGML_TYPE_Q5_K || T == GGML_TYPE_Q6_K || T == GGML_TYPE_IQ4_XS || T == GGML_TYPE_IQ4_NL) {
                records[0] = pf_records[t][0];
                records[1] = pf_records[t][1];
            } else {
                records[0] = qskinny_load_group_record<T>(code_base + (size_t) t * tile_bytes + (size_t) (group + 0) * 32 * codec::record_bytes, lane);
                records[1] = qskinny_load_group_record<T>(code_base + (size_t) t * tile_bytes + (size_t) (group + 1) * 32 * codec::record_bytes, lane);
            }
            half2 weights[codec::values_per_sub_block / 2];
            codec::decode(records, meta[t], sub_block & 7, weights);
            const unsigned * b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
            for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
                const unsigned * a0 = reinterpret_cast<const unsigned *>(&in[row_tile][0]);
                const unsigned * a1 = reinterpret_cast<const unsigned *>(&in[row_tile][1]);
                const unsigned * a2 = reinterpret_cast<const unsigned *>(&in[row_tile][2]);
                const unsigned * a3 = reinterpret_cast<const unsigned *>(&in[row_tile][3]);
                Q4K_SKINNY_MMA_8N8K4(accum[t][row_tile][0], a0[0], a0[1], b[0], b[1]);
                Q4K_SKINNY_MMA_8N8K4(accum[t][row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
                Q4K_SKINNY_MMA_8N8K4(accum[t][row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
                Q4K_SKINNY_MMA_8N8K4(accum[t][row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
                Q4K_SKINNY_MMA_8N8K4(accum[t][row_tile][0], a2[0], a2[1], b[8], b[9]);
                Q4K_SKINNY_MMA_8N8K4(accum[t][row_tile][1 % NAcc], a2[2], a2[3], b[10], b[11]);
                Q4K_SKINNY_MMA_8N8K4(accum[t][row_tile][2 % NAcc], a3[0], a3[1], b[12], b[13]);
                Q4K_SKINNY_MMA_8N8K4(accum[t][row_tile][3 % NAcc], a3[2], a3[3], b[14], b[15]);
            }
        }
    }

    // one tile at a time through the split-K reduction buffer, in a fixed order
    for (int t = 0; t < NTiles; ++t) {
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float v = accum[t][row_tile][0][i];
#pragma unroll
                for (int chain = 1; chain < NAcc; ++chain) {
                    v += accum[t][row_tile][chain][i];
                }
                const int output_row = row_tile * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
                const int output_col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
                partials[warp][output_row * 32 + quadpair * 8 + output_col] = v;
            }
        }
        __syncthreads();
        for (int element = threadIdx.x; element < RowTiles * 256; element += blockDim.x) {
            float value = 0.0f;
#pragma unroll
            for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
                value += partials[k_warp][element];
            }
            const int output_row = element >> 5;
            const int output_col = element & 31;
            if (output_row < m) {
                output[(size_t) output_row * n + (size_t) (tile0 + t) * 32 + output_col] = value;
            }
        }
        __syncthreads();
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(codes, input, output, n, k, m);
#endif
}

// ---- multiple projections of one input in one launch ----

// One segment is a repacked weight (N is a multiple of 32) plus its output tile range.
// first_tile is the running sum of n / 32 over the previous segments, so a CTA finds its
// segment from blockIdx.x with a short scan.
struct q4k_skinny_seg {
    const uint8_t * codes;
    int             n;
    float *         dst;
    int             first_tile;
};

// Passed by value. One CTA per 32-column tile, the segments cover the grid in order.
struct q4k_skinny_multi_params {
    q4k_skinny_seg seg[4];
    int n_seg;
    int n_tiles;
    const half * input;
    int k;
    int m;
};

// Same execution as q4k_skinny_kernel, but each CTA picks its segment from blockIdx.x and
// writes to that segment's dst with its own row length. launch_bounds keeps the register
// count at the single-kernel level so a row tile fits two CTAs per SM.
template <ggml_type T, int SplitK, int NAcc, bool M1Only = false, int RowTiles = 1>
__global__ void __launch_bounds__(32 * SplitK, RowTiles == 1 ? 2 : 1)
q4k_skinny_multi_kernel(const q4k_skinny_multi_params p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    using codec = qskinny_codec<T>;
    static_assert(RowTiles == 1 || RowTiles == 2,
                  "Q4_K skinny multi supports one or two 8-row tiles");
    static_assert(!M1Only || RowTiles == 1,
                  "Q4_K skinny multi M=1 specialization uses one row tile");
    __shared__ float partials[SplitK][M1Only ? 32 : RowTiles * 256];

    int seg = 0;
    while (seg + 1 < p.n_seg && (int) blockIdx.x >= p.seg[seg + 1].first_tile) {
        ++seg;
    }
    const int n = p.seg[seg].n;
    const int k = p.k;
    const int m = p.m;
    const int tile = (int) blockIdx.x - p.seg[seg].first_tile;
    const uint8_t * codes = p.seg[seg].codes;
    const half * input = p.input;
    float * output = p.seg[seg].dst;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int quadpair = (lane >> 2) & 3;
    const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_k16 = k >> 4;
    const int sub_blocks = k >> 5;
    const int sub_blocks_per_warp = sub_blocks / SplitK;
    const int sub_block_begin = warp * sub_blocks_per_warp;
    const uint8_t * code_base = codes + (size_t) tile * groups_k16 * 32 * codec::record_bytes;
    // meta stride and offset are per codec type
    const uint8_t * meta_ptr = codes + codec::codes_bytes(n, k) +
                               (size_t) codec::meta_bytes * (tile * 32 + lane);

    float accum[RowTiles][NAcc][8];
#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
        for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                accum[row_tile][chain][i] = 0.0f;
            }
        }
    }
    int loaded_meta_kb = -1;
    typename codec::meta_t meta = {};

#pragma unroll 2
    for (int sub_block = sub_block_begin; sub_block < sub_block_begin + sub_blocks_per_warp; ++sub_block) {
        const int kb = sub_block >> 3;
        if (kb != loaded_meta_kb) {
            meta = codec::load_meta(meta_ptr + (size_t) kb * n * codec::meta_bytes);
            loaded_meta_kb = kb;
        }

        const int group = sub_block << 1;
        const qskinny_code records[2] = { // record load per codec type
            qskinny_load_group_record<T>(code_base + (size_t) (group + 0) * 32 * codec::record_bytes, lane),
            qskinny_load_group_record<T>(code_base + (size_t) (group + 1) * 32 * codec::record_bytes, lane),
        };
        half2 weights[codec::values_per_sub_block / 2];
        codec::decode(records, meta, sub_block & 7, weights);

        const unsigned * b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
            uint4 input01 = make_uint4(0, 0, 0, 0);
            uint4 input23 = make_uint4(0, 0, 0, 0);
            uint4 input45 = make_uint4(0, 0, 0, 0);
            uint4 input67 = make_uint4(0, 0, 0, 0);
            const int input_row_idx = row_tile * 8 + row;
            if (input_row_idx < m) {
                const half * input_row = input + (size_t) input_row_idx * k;
                input01 = *reinterpret_cast<const uint4 *>(input_row + group * 16);
                input23 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 8);
                input45 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 16);
                input67 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 24);
            }

            const unsigned * a0 = reinterpret_cast<const unsigned *>(&input01);
            const unsigned * a1 = reinterpret_cast<const unsigned *>(&input23);
            const unsigned * a2 = reinterpret_cast<const unsigned *>(&input45);
            const unsigned * a3 = reinterpret_cast<const unsigned *>(&input67);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][0], a2[0], a2[1], b[8], b[9]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][1 % NAcc], a2[2], a2[3], b[10], b[11]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][2 % NAcc], a3[0], a3[1], b[12], b[13]);
            Q4K_SKINNY_MMA_8N8K4(accum[row_tile][3 % NAcc], a3[2], a3[3], b[14], b[15]);
        }
    }

#pragma unroll
    for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
        for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                accum[row_tile][0][i] += accum[row_tile][chain][i];
            }
        }
    }

    if constexpr (M1Only) {
        if ((lane & 17) == 0) {
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
                for (int offset = 0; offset < 2; ++offset) {
                    const int i = pair * 4 + offset;
                    const int output_col =
                        offset | (((lane >> 1) & 1) << 1) | (pair << 2);
                    partials[warp][quadpair * 8 + output_col] = accum[0][0][i];
                }
            }
        }
    } else {
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int output_row =
                    row_tile * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
                const int output_col =
                    (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
                partials[warp][output_row * 32 + quadpair * 8 + output_col] =
                    accum[row_tile][0][i];
            }
        }
    }
    __syncthreads();

    // fixed reduction order over the split-K warps keeps the result reproducible
    constexpr int kOutputElements = M1Only ? 32 : RowTiles * 256;
    for (int element = threadIdx.x; element < kOutputElements;
         element += blockDim.x) {
        float value = 0.0f;
#pragma unroll
        for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
            value += partials[k_warp][element];
        }
        if constexpr (M1Only) {
            output[tile * 32 + element] = value;
        } else {
            const int output_row = element >> 5;
            const int output_col = element & 31;
            if (output_row < m) {
                output[(size_t) output_row * n + tile * 32 + output_col] = value;
            }
        }
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED(p);
#endif
}

// ---- gated pair: the gate and up projections of one SwiGLU in one launch ----

// gate and up are two separate tensors: the block computes the two projections in two
// phases with the same split-K, accumulator chains and reduction order as two
// q4k_skinny_kernel launches, so each projection is bit-exact with its single-weight
// launch. Phase 0 keeps its reduced gate in registers; the phase 1 epilogue applies
// silu(gate) * up with the ggml_cuda_op_silu_single formula and writes float.
// launch_bounds keeps the register count at the single-kernel level so a row tile fits
// two CTAs per SM; the phase loop is not unrolled to keep the body shared.
template <ggml_type T, int SplitK, int NAcc, bool M1Only = false, int RowTiles = 1>
__global__ void __launch_bounds__(32 * SplitK, RowTiles == 1 ? 2 : 1)
q4k_skinny_gated_kernel(
    const uint8_t * __restrict__ gate_codes, const uint8_t * __restrict__ up_codes,
    const half * __restrict__ input, float * __restrict__ output, half * __restrict__ f16_out,
    int n, int k, int m) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    using codec = qskinny_codec<T>;
    static_assert(RowTiles == 1 || RowTiles == 2,
                  "Q4_K skinny gated pair supports one or two 8-row tiles");
    static_assert(!M1Only || RowTiles == 1,
                  "Q4_K skinny gated M=1 specialization uses one row tile");
    constexpr int kOutputElements = M1Only ? 32 : RowTiles * 256;
    __shared__ float partials[SplitK][kOutputElements];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tile = blockIdx.x;
    const int quadpair = (lane >> 2) & 3;
    const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_k16 = k >> 4;
    const int sub_blocks = k >> 5;
    const int sub_blocks_per_warp = sub_blocks / SplitK;
    const int sub_block_begin = warp * sub_blocks_per_warp;
    // the reduced gate of phase 0 stays in registers: each thread keeps the values of its
    // own strided loop, so the shared footprint matches the single-weight kernel
    float gate_red[(kOutputElements + 32 * SplitK - 1) / (32 * SplitK)];

#pragma unroll 1
    for (int projection = 0; projection < 2; ++projection) {
        const uint8_t * codes = projection ? up_codes : gate_codes;
        const uint8_t * code_base = codes + (size_t) tile * groups_k16 * 32 * codec::record_bytes;
        // meta stride and offset are per codec type
        const uint8_t * meta_ptr = codes + codec::codes_bytes(n, k) +
                                   (size_t) codec::meta_bytes * (tile * 32 + lane);

        float accum[RowTiles][NAcc][8];
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
            for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    accum[row_tile][chain][i] = 0.0f;
                }
            }
        }
        int loaded_meta_kb = -1;
        typename codec::meta_t meta = {};

#pragma unroll 2
        for (int sub_block = sub_block_begin; sub_block < sub_block_begin + sub_blocks_per_warp; ++sub_block) {
            const int kb = sub_block >> 3;
            if (kb != loaded_meta_kb) {
                meta = codec::load_meta(meta_ptr + (size_t) kb * n * codec::meta_bytes);
                loaded_meta_kb = kb;
            }

            const int group = sub_block << 1;
            const qskinny_code records[2] = { // record load per codec type
                qskinny_load_group_record<T>(code_base + (size_t) (group + 0) * 32 * codec::record_bytes, lane),
                qskinny_load_group_record<T>(code_base + (size_t) (group + 1) * 32 * codec::record_bytes, lane),
            };
            half2 weights[codec::values_per_sub_block / 2];
            codec::decode(records, meta, sub_block & 7, weights);

            const unsigned * b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
            for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
                uint4 input01 = make_uint4(0, 0, 0, 0);
                uint4 input23 = make_uint4(0, 0, 0, 0);
                uint4 input45 = make_uint4(0, 0, 0, 0);
                uint4 input67 = make_uint4(0, 0, 0, 0);
                const int input_row_idx = row_tile * 8 + row;
                if (input_row_idx < m) {
                    const half * input_row = input + (size_t) input_row_idx * k;
                    input01 = *reinterpret_cast<const uint4 *>(input_row + group * 16);
                    input23 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 8);
                    input45 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 16);
                    input67 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 24);
                }

                const unsigned * a0 = reinterpret_cast<const unsigned *>(&input01);
                const unsigned * a1 = reinterpret_cast<const unsigned *>(&input23);
                const unsigned * a2 = reinterpret_cast<const unsigned *>(&input45);
                const unsigned * a3 = reinterpret_cast<const unsigned *>(&input67);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][0], a2[0], a2[1], b[8], b[9]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][1 % NAcc], a2[2], a2[3], b[10], b[11]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][2 % NAcc], a3[0], a3[1], b[12], b[13]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][3 % NAcc], a3[2], a3[3], b[14], b[15]);
            }
        }

#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
            for (int chain = 1; chain < NAcc; ++chain) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    accum[row_tile][0][i] += accum[row_tile][chain][i];
                }
            }
        }

        if constexpr (M1Only) {
            if ((lane & 17) == 0) {
#pragma unroll
                for (int pair = 0; pair < 2; ++pair) {
#pragma unroll
                    for (int offset = 0; offset < 2; ++offset) {
                        const int i = pair * 4 + offset;
                        const int output_col =
                            offset | (((lane >> 1) & 1) << 1) | (pair << 2);
                        partials[warp][quadpair * 8 + output_col] = accum[0][0][i];
                    }
                }
            }
        } else {
#pragma unroll
            for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int output_row =
                        row_tile * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
                    const int output_col =
                        (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
                    partials[warp][output_row * 32 + quadpair * 8 + output_col] =
                        accum[row_tile][0][i];
                }
            }
        }
        __syncthreads();

        for (int element = threadIdx.x, slot = 0; element < kOutputElements;
             element += blockDim.x, ++slot) {
            float value = 0.0f;
#pragma unroll
            for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
                value += partials[k_warp][element];
            }
            if (projection == 0) {
                gate_red[slot] = value;
            } else {
                const float gate = gate_red[slot];
                const float silu = gate / (1.0f + expf(-gate));
                if constexpr (M1Only) {
                    const float res = silu * value;
                    const size_t oi = (size_t) tile * 32 + element;
                    output[oi] = res;
                    if (f16_out) { f16_out[oi] = __float2half(res); }
                } else {
                    const int output_row = element >> 5;
                    const int output_col = element & 31;
                    if (output_row < m) {
                        const float res = silu * value;
                        const size_t oi = (size_t) output_row * n + tile * 32 + output_col;
                        output[oi] = res;
                        if (f16_out) { f16_out[oi] = __float2half(res); }
                    }
                }
            }
        }
        __syncthreads();
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(gate_codes, up_codes, input, output, n, k, m);
#endif
}

// Split-K 16 for the long K weights, 8 for the smallest ones (k = 256 has only 8 sub-blocks
// per tile). can_repack guarantees k % 256 == 0, so one of the two always divides the
// sub-block count. The long K weights use two accumulator chains, as in the Q8_0 path.
struct q4k_skinny_config {
    int  split_k;
    int  n_acc;
};

static q4k_skinny_config q4k_skinny_config_for(const int64_t k) {
    return { k % 512 == 0 ? 16 : 8, k >= 4096 ? 2 : 1 };
}

template <ggml_type T, int SplitK, int NAcc, bool M1Only, int RowTiles>
static void q4k_skinny_launch(const uint8_t * codes, const half * input, float * output,
                              int n, int k, int m, cudaStream_t stream) {
    q4k_skinny_kernel<T, SplitK, NAcc, M1Only, RowTiles><<<n / 32, 32 * SplitK, 0, stream>>>(
        codes, input, output, n, k, m);
}

#define Q4K_SKINNY_LAUNCH(NAcc, M1Only, RowTiles)                                               \
    do {                                                                                        \
        switch (config.split_k) {                                                               \
            case 16: q4k_skinny_launch<T, 16, NAcc, M1Only, RowTiles>(codes,                     \
                         input, output, n, k, m, stream); break;                                \
            default: q4k_skinny_launch<T, 8, NAcc, M1Only, RowTiles>(codes,                      \
                         input, output, n, k, m, stream); break;                                \
        }                                                                                       \
    } while (0)

// the launch table is instantiated per codec type
template <ggml_type T>
static void q4k_skinny_mul_mat_launch_t(const uint8_t * codes, const half * input, float * output,
                                        int n, int k, int m, cudaStream_t stream) {
    const q4k_skinny_config config = q4k_skinny_config_for(k);
    // NTiles (v100fast): at 9 to 16 input rows a CTA takes two weight tiles, so each sub-block's input
    // fragments are loaded once for both. Measured on a V100 at m = 16 against the one-tile kernel:
    // q4_K 5120x5120 49.1 -> 34.4 us, 17408x5120 150.4 -> 105.9 us, 5120x17408 138.4 -> 91.7 us, same output.
    // 16 split-K warps while the grid still fills the 80 SMs with them, 8 for the tall weights.
    static const bool nt_on = [] {
        const char * e = getenv("LLAMA_SM70_SKINNY_NT");
        return e == nullptr || e[0] != '0';
    }();
    static const bool nt4_on = [] {
        const char * e = getenv("LLAMA_SM70_SKINNY_NT4");
        return e && e[0] == '1';
    }();
    static const bool cta3_on = [] {
        const char * e = getenv("LLAMA_SM70_SKINNY_3CTA");
        return e && e[0] == '1';
    }();
    static const bool sk8_all = [] {
        const char * e = getenv("LLAMA_SM70_SKINNY_SK8_M8");
        return e && e[0] == '1';
    }();
    static const bool wavefix_on = [] {
        const char * e = getenv("LLAMA_SM70_SKINNY_WAVEFIX");
        return e && e[0] == '1';
    }();
    if (nt_on && m >= 2 && n % 64 == 0) {
        if (m <= 8) {
            if (wavefix_on && (T == GGML_TYPE_Q4_K || T == GGML_TYPE_Q5_K || T == GGML_TYPE_Q6_K || T == GGML_TYPE_IQ4_XS)) {
                if (n <= 2048 && config.split_k == 16) {
                    // K/V projections (n=1024): 32 CTAs instead of 16 -> 2x active SMs!
                    q4k_skinny_nt_kernel<T, 16, 1, 1, 1, 1><<<n / 32, 32 * 16, 0, stream>>>(codes, input, output, n, k, m);
                    CUDA_CHECK(cudaGetLastError());
                    return;
                }
                if (n / 64 > 80 && n / 64 < 160) {
                    // ssm_in (n=6144): 192 CTAs @ 3 CTAs/SM -> 1 wave instead of 2 waves!
                    q4k_skinny_nt_kernel<T, 8, 1, 1, 1, 3><<<n / 32, 32 * 8, 0, stream>>>(codes, input, output, n, k, m);
                    CUDA_CHECK(cudaGetLastError());
                    return;
                }
                static const bool n5120_2cta = [] {
                    const char * e = getenv("LLAMA_SM70_SKINNY_N5120_2CTA");
                    return e && e[0] == '1';
                }();
                if (n5120_2cta && n / 64 == 80 && config.split_k == 16) {
                    // ffn_down / attn_out / ssm_out (n=5120): 160 CTAs @ 2 CTAs/SM (32 warps/SM, <=64 regs!)
                    q4k_skinny_nt_kernel<T, 16, 1, 1, 1, 2><<<n / 32, 32 * 16, 0, stream>>>(codes, input, output, n, k, m);
                    CUDA_CHECK(cudaGetLastError());
                    return;
                }
            }
            if (nt4_on && n % 128 == 0 && n / 128 >= 80 &&
                (T == GGML_TYPE_Q4_K || T == GGML_TYPE_Q5_K || T == GGML_TYPE_Q6_K || T == GGML_TYPE_IQ4_XS)) {
                q4k_skinny_nt_kernel<T, 8, 1, 1, 4, 2><<<n / 128, 32 * 8, 0, stream>>>(codes, input, output, n, k, m);
            } else if (config.split_k == 16 && n / 64 < 160 && !sk8_all) {
                q4k_skinny_nt_kernel<T, 16, 1, 1, 2><<<n / 64, 32 * 16, 0, stream>>>(codes, input, output, n, k, m);
            } else if (cta3_on && (T == GGML_TYPE_Q4_K || T == GGML_TYPE_Q5_K || T == GGML_TYPE_Q6_K || T == GGML_TYPE_IQ4_XS)) {
                q4k_skinny_nt_kernel<T, 8, 1, 1, 2, 3><<<n / 64, 32 * 8, 0, stream>>>(codes, input, output, n, k, m);
            } else {
                q4k_skinny_nt_kernel<T, 8, 1, 1, 2><<<n / 64, 32 * 8, 0, stream>>>(codes, input, output, n, k, m);
            }
        } else {
            if (config.split_k == 16 && n / 64 < 160) {
                q4k_skinny_nt_kernel<T, 16, 1, 2, 2><<<n / 64, 32 * 16, 0, stream>>>(codes, input, output, n, k, m);
            } else {
                q4k_skinny_nt_kernel<T, 8, 1, 2, 2><<<n / 64, 32 * 8, 0, stream>>>(codes, input, output, n, k, m);
            }
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (config.n_acc == 2) {
        if (m == 1) { Q4K_SKINNY_LAUNCH(2, true, 1); } else if (m <= 8) { Q4K_SKINNY_LAUNCH(2, false, 1); }
        else { Q4K_SKINNY_LAUNCH(2, false, 2); }
    } else {
        if (m == 1) { Q4K_SKINNY_LAUNCH(1, true, 1); } else if (m <= 8) { Q4K_SKINNY_LAUNCH(1, false, 1); }
        else { Q4K_SKINNY_LAUNCH(1, false, 2); }
    }
    CUDA_CHECK(cudaGetLastError());
}

#undef Q4K_SKINNY_LAUNCH
// Same epilogue as q4k_skinny_gated_kernel, applied after both projections of an M > 16 run.
__global__ void q4k_skinny_swiglu_kernel(float * __restrict__ gate, const float * __restrict__ up,
                                         int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        const float g = gate[i];
        gate[i] = g / (1.0f + expf(-g)) * up[i];
    }
}
template <ggml_type T, int SplitK, int NAcc, bool M1Only, int RowTiles>
static void q4k_skinny_multi_launch(const q4k_skinny_multi_params & p, cudaStream_t stream) {
    q4k_skinny_multi_kernel<T, SplitK, NAcc, M1Only, RowTiles>
        <<<p.n_tiles, 32 * SplitK, 0, stream>>>(p);
}

#define Q4K_SKINNY_MULTI_LAUNCH(NAcc, M1Only, RowTiles)                                          \
    do {                                                                                         \
        switch (config.split_k) {                                                                \
            case 16: q4k_skinny_multi_launch<T, 16, NAcc, M1Only, RowTiles>(p, stream); break;    \
            default: q4k_skinny_multi_launch<T, 8, NAcc, M1Only, RowTiles>(p, stream); break;     \
        }                                                                                        \
    } while (0)

// the launch table is instantiated per codec type
template <ggml_type T>
static void q4k_skinny_multi_mul_mat_launch_t(const q4k_skinny_multi_params & p, cudaStream_t stream) {
    const q4k_skinny_config config = q4k_skinny_config_for(p.k);
    if (config.n_acc == 2) {
        if (p.m == 1) { Q4K_SKINNY_MULTI_LAUNCH(2, true, 1); } else if (p.m <= 8) { Q4K_SKINNY_MULTI_LAUNCH(2, false, 1); }
        else { Q4K_SKINNY_MULTI_LAUNCH(2, false, 2); }
    } else {
        if (p.m == 1) { Q4K_SKINNY_MULTI_LAUNCH(1, true, 1); } else if (p.m <= 8) { Q4K_SKINNY_MULTI_LAUNCH(1, false, 1); }
        else { Q4K_SKINNY_MULTI_LAUNCH(1, false, 2); }
    }
}

#undef Q4K_SKINNY_MULTI_LAUNCH

static void q4k_skinny_multi_mul_mat_launch(ggml_type type, const q4k_skinny_multi_params & p,
                                            cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_Q2_K: q4k_skinny_multi_mul_mat_launch_t<GGML_TYPE_Q2_K>(p, stream); break;
        case GGML_TYPE_Q3_K: q4k_skinny_multi_mul_mat_launch_t<GGML_TYPE_Q3_K>(p, stream); break;
        case GGML_TYPE_Q4_K: q4k_skinny_multi_mul_mat_launch_t<GGML_TYPE_Q4_K>(p, stream); break;
        case GGML_TYPE_Q5_K: q4k_skinny_multi_mul_mat_launch_t<GGML_TYPE_Q5_K>(p, stream); break;
        case GGML_TYPE_Q6_K: q4k_skinny_multi_mul_mat_launch_t<GGML_TYPE_Q6_K>(p, stream); break;
        case GGML_TYPE_IQ4_XS: q4k_skinny_multi_mul_mat_launch_t<GGML_TYPE_IQ4_XS>(p, stream); break;
        case GGML_TYPE_IQ4_NL: q4k_skinny_multi_mul_mat_launch_t<GGML_TYPE_IQ4_NL>(p, stream); break;
        default: GGML_ABORT("q4k skinny: unknown type");
    }
    CUDA_CHECK(cudaGetLastError());
}

// ---- gated pair, one pass (custom, v100fast) ----
// The stock gated kernel runs the gate and the up projection as two phases, each streaming the whole input.
// Here both weights of a tile go through one K loop: a sub-block's input fragments are loaded once and used for
// the gate and the up records. Same split-K reduction order per projection, then silu(gate) * up.
template <ggml_type T, int SplitK, int NAcc, int RowTiles, int MinBlocks = (512 / (32 * SplitK))>
__launch_bounds__(32 * SplitK, MinBlocks)
__global__ void q4k_skinny_gated_nt_kernel(
    const uint8_t * __restrict__ gate_codes, const uint8_t * __restrict__ up_codes,
    const half * __restrict__ input, float * __restrict__ output, half * __restrict__ f16_out, int n, int k, int m) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    using codec = qskinny_codec<T>;
    constexpr int kOutputElements = RowTiles * 256;
    __shared__ float partials[SplitK][kOutputElements];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tile = blockIdx.x;
    const int quadpair = (lane >> 2) & 3;
    const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_k16 = k >> 4;
    const int sub_blocks = k >> 5;
    const int sub_blocks_per_warp = sub_blocks / SplitK;
    const int sub_block_begin = warp * sub_blocks_per_warp;
    const size_t code_off = (size_t) tile * groups_k16 * 32 * codec::record_bytes;
    const size_t meta_off = codec::codes_bytes(n, k) + (size_t) codec::meta_bytes * ((size_t) tile * 32 + lane);
    const uint8_t * code_base[2] = { gate_codes + code_off, up_codes + code_off };
    const uint8_t * meta_ptr[2] = { gate_codes + meta_off, up_codes + meta_off };

    float accum[2][RowTiles][NAcc][8];
#pragma unroll
    for (int t = 0; t < 2; ++t) {
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
            for (int chain = 0; chain < NAcc; ++chain) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    accum[t][row_tile][chain][i] = 0.0f;
                }
            }
        }
    }
    int loaded_meta_kb = -1;
    typename codec::meta_t meta[2] = {};

    for (int sub_block = sub_block_begin; sub_block < sub_block_begin + sub_blocks_per_warp; ++sub_block) {
        const int kb = sub_block >> 3;
        if (kb != loaded_meta_kb) {
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                meta[t] = codec::load_meta(meta_ptr[t] + (size_t) kb * n * codec::meta_bytes);
            }
            loaded_meta_kb = kb;
        }
        const int group = sub_block << 1;

        // the input fragments of this sub-block, loaded once for the gate and the up weights
        uint4 in[RowTiles][4];
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
            const int input_row_idx = row_tile * 8 + row;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                in[row_tile][j] = make_uint4(0, 0, 0, 0);
            }
            if (input_row_idx < m) {
                const half * input_row = input + (size_t) input_row_idx * k + group * 16;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    in[row_tile][j] = *reinterpret_cast<const uint4 *>(input_row + 8 * j);
                }
            }
        }

        // Cross-projection HBM2 prefetch: issue Gate + Up loads together so Up hides behind Gate decode+HMMA!
        qskinny_code pf_gnt[2][2];
        if constexpr (T == GGML_TYPE_Q4_K || T == GGML_TYPE_Q5_K || T == GGML_TYPE_Q6_K || T == GGML_TYPE_IQ4_XS || T == GGML_TYPE_IQ4_NL) {
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                pf_gnt[t][0] = qskinny_load_group_record<T>(code_base[t] + (size_t) (group + 0) * 32 * codec::record_bytes, lane);
                pf_gnt[t][1] = qskinny_load_group_record<T>(code_base[t] + (size_t) (group + 1) * 32 * codec::record_bytes, lane);
            }
        }
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            qskinny_code records[2];
            if constexpr (T == GGML_TYPE_Q4_K || T == GGML_TYPE_Q5_K || T == GGML_TYPE_Q6_K || T == GGML_TYPE_IQ4_XS || T == GGML_TYPE_IQ4_NL) {
                records[0] = pf_gnt[t][0];
                records[1] = pf_gnt[t][1];
            } else {
                records[0] = qskinny_load_group_record<T>(code_base[t] + (size_t) (group + 0) * 32 * codec::record_bytes, lane);
                records[1] = qskinny_load_group_record<T>(code_base[t] + (size_t) (group + 1) * 32 * codec::record_bytes, lane);
            }
            half2 weights[codec::values_per_sub_block / 2];
            codec::decode(records, meta[t], sub_block & 7, weights);
            const unsigned * b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
            for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
                const unsigned * a0 = reinterpret_cast<const unsigned *>(&in[row_tile][0]);
                const unsigned * a1 = reinterpret_cast<const unsigned *>(&in[row_tile][1]);
                const unsigned * a2 = reinterpret_cast<const unsigned *>(&in[row_tile][2]);
                const unsigned * a3 = reinterpret_cast<const unsigned *>(&in[row_tile][3]);
                Q4K_SKINNY_MMA_8N8K4(accum[t][row_tile][0], a0[0], a0[1], b[0], b[1]);
                Q4K_SKINNY_MMA_8N8K4(accum[t][row_tile][1 % NAcc], a0[2], a0[3], b[2], b[3]);
                Q4K_SKINNY_MMA_8N8K4(accum[t][row_tile][2 % NAcc], a1[0], a1[1], b[4], b[5]);
                Q4K_SKINNY_MMA_8N8K4(accum[t][row_tile][3 % NAcc], a1[2], a1[3], b[6], b[7]);
                Q4K_SKINNY_MMA_8N8K4(accum[t][row_tile][0], a2[0], a2[1], b[8], b[9]);
                Q4K_SKINNY_MMA_8N8K4(accum[t][row_tile][1 % NAcc], a2[2], a2[3], b[10], b[11]);
                Q4K_SKINNY_MMA_8N8K4(accum[t][row_tile][2 % NAcc], a3[0], a3[1], b[12], b[13]);
                Q4K_SKINNY_MMA_8N8K4(accum[t][row_tile][3 % NAcc], a3[2], a3[3], b[14], b[15]);
            }
        }
    }

    // gate first (kept in registers), then up: silu(gate) * up
    float gate_red[(kOutputElements + 32 * SplitK - 1) / (32 * SplitK)];
    for (int t = 0; t < 2; ++t) {
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float v = accum[t][row_tile][0][i];
#pragma unroll
                for (int chain = 1; chain < NAcc; ++chain) {
                    v += accum[t][row_tile][chain][i];
                }
                const int output_row = row_tile * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
                const int output_col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
                partials[warp][output_row * 32 + quadpair * 8 + output_col] = v;
            }
        }
        __syncthreads();
        constexpr int kSlotsPerThread = (kOutputElements + 32 * SplitK - 1) / (32 * SplitK);
#pragma unroll
        for (int slot = 0; slot < kSlotsPerThread; ++slot) {
            const int element = threadIdx.x + slot * (32 * SplitK);
            if (element >= kOutputElements) { break; }
            float value = 0.0f;
#pragma unroll
            for (int k_warp = 0; k_warp < SplitK; ++k_warp) {
                value += partials[k_warp][element];
            }
            if (t == 0) {
                gate_red[slot] = value;
            } else {
                const float gate = gate_red[slot];
                const float silu = gate / (1.0f + expf(-gate));
                const int output_row = element >> 5;
                const int output_col = element & 31;
                if (output_row < m) {
                    const float res = silu * value;
                    const size_t oi = (size_t) output_row * n + tile * 32 + output_col;
                    output[oi] = res;
                    if (f16_out) { f16_out[oi] = __float2half(res); }
                }
            }
        }
        __syncthreads();
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(gate_codes, up_codes, input, output, n, k, m);
#endif
}

// Same split-K/accumulator-chain table as the single-weight kernel, so a projection is
// bit-exact with its own q4k_skinny_kernel launch.
template <ggml_type T, int SplitK, int NAcc, bool M1Only, int RowTiles>
static void q4k_skinny_gated_launch(const uint8_t * gate_codes, const uint8_t * up_codes,
                                    const half * input, float * output, half * f16_out, int n, int k, int m,
                                    cudaStream_t stream) {
    q4k_skinny_gated_kernel<T, SplitK, NAcc, M1Only, RowTiles>
        <<<n / 32, 32 * SplitK, 0, stream>>>(gate_codes, up_codes, input, output, f16_out, n, k, m);
}

#define Q4K_SKINNY_GATED_LAUNCH(NAcc, M1Only, RowTiles)                                        \
    do {                                                                                       \
        switch (config.split_k) {                                                              \
            case 16: q4k_skinny_gated_launch<T, 16, NAcc, M1Only, RowTiles>(                    \
                         gate_codes, up_codes, input, output, f16_out, n, k, m, stream); break;\
            default: q4k_skinny_gated_launch<T, 8, NAcc, M1Only, RowTiles>(                     \
                         gate_codes, up_codes, input, output, f16_out, n, k, m, stream); break;\
        }                                                                                      \
    } while (0)

// the launch table is instantiated per codec type
template <ggml_type T>
static void q4k_skinny_gated_mul_mat_launch_t(const uint8_t * gate_codes, const uint8_t * up_codes,
                                              const half * input, float * output, half * f16_out, int n, int k, int m,
                                              cudaStream_t stream) {
    const q4k_skinny_config config = q4k_skinny_config_for(k);
    // one pass (v100fast): at 9 to 16 input rows the gate and the up weights share each sub-block's input
    // fragments. Measured on a V100 at m = 16, 17408x5120, against the two-phase kernel, same output:
    // q4_K 306 -> 198 us, q5_K 305 -> 241 us, q6_K 299 -> 243 us, iq4_xs 333 -> 249 us.
    static const bool nt_on = [] {
        const char * e = getenv("LLAMA_SM70_SKINNY_NT");
        return e == nullptr || e[0] != '0';
    }();
    static const bool gnt_sk8 = [] {
        const char * e = getenv("LLAMA_SM70_SKINNY_GNT_SK8");
        return e && e[0] == '1';
    }();
    static const bool gnt_3cta = [] {
        const char * e = getenv("LLAMA_SM70_SKINNY_GNT_3CTA");
        return e && e[0] == '1';
    }();
    if (nt_on && m >= 2) {
        if (m <= 8) {
            if (gnt_3cta && n / 32 >= 160 &&
                (T == GGML_TYPE_Q4_K || T == GGML_TYPE_Q5_K || T == GGML_TYPE_Q6_K || T == GGML_TYPE_IQ4_XS)) {
                q4k_skinny_gated_nt_kernel<T, 8, 1, 1, 3><<<n / 32, 32 * 8, 0, stream>>>(gate_codes, up_codes, input, output, f16_out, n, k, m);
            } else if (config.split_k == 16 && !(gnt_sk8 && n / 32 >= 160)) {
                q4k_skinny_gated_nt_kernel<T, 16, 1, 1><<<n / 32, 32 * 16, 0, stream>>>(gate_codes, up_codes, input, output, f16_out, n, k, m);
            } else {
                q4k_skinny_gated_nt_kernel<T, 8, 1, 1><<<n / 32, 32 * 8, 0, stream>>>(gate_codes, up_codes, input, output, f16_out, n, k, m);
            }
        } else {
            if (config.split_k == 16 && !(gnt_sk8 && n / 32 >= 160)) {
                q4k_skinny_gated_nt_kernel<T, 16, 1, 2><<<n / 32, 32 * 16, 0, stream>>>(gate_codes, up_codes, input, output, f16_out, n, k, m);
            } else {
                q4k_skinny_gated_nt_kernel<T, 8, 1, 2><<<n / 32, 32 * 8, 0, stream>>>(gate_codes, up_codes, input, output, f16_out, n, k, m);
            }
        }
        return;
    }
    if (config.n_acc == 2) {
        if (m == 1) { Q4K_SKINNY_GATED_LAUNCH(2, true, 1); } else if (m <= 8) { Q4K_SKINNY_GATED_LAUNCH(2, false, 1); }
        else { Q4K_SKINNY_GATED_LAUNCH(2, false, 2); }
    } else {
        if (m == 1) { Q4K_SKINNY_GATED_LAUNCH(1, true, 1); } else if (m <= 8) { Q4K_SKINNY_GATED_LAUNCH(1, false, 1); }
        else { Q4K_SKINNY_GATED_LAUNCH(1, false, 2); }
    }
}

#undef Q4K_SKINNY_GATED_LAUNCH

static void q4k_skinny_gated_mul_mat_launch(ggml_type type, const uint8_t * gate_codes,
                                            const uint8_t * up_codes, const half * input,
                                            float * output, half * f16_out, int n, int k, int m,
                                            cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_Q2_K: q4k_skinny_gated_mul_mat_launch_t<GGML_TYPE_Q2_K>(gate_codes, up_codes, input, output, f16_out, n, k, m, stream); break;
        case GGML_TYPE_Q3_K: q4k_skinny_gated_mul_mat_launch_t<GGML_TYPE_Q3_K>(gate_codes, up_codes, input, output, f16_out, n, k, m, stream); break;
        case GGML_TYPE_Q4_K: q4k_skinny_gated_mul_mat_launch_t<GGML_TYPE_Q4_K>(gate_codes, up_codes, input, output, f16_out, n, k, m, stream); break;
        case GGML_TYPE_Q5_K: q4k_skinny_gated_mul_mat_launch_t<GGML_TYPE_Q5_K>(gate_codes, up_codes, input, output, f16_out, n, k, m, stream); break;
        case GGML_TYPE_Q6_K: q4k_skinny_gated_mul_mat_launch_t<GGML_TYPE_Q6_K>(gate_codes, up_codes, input, output, f16_out, n, k, m, stream); break;
        case GGML_TYPE_IQ4_XS: q4k_skinny_gated_mul_mat_launch_t<GGML_TYPE_IQ4_XS>(gate_codes, up_codes, input, output, f16_out, n, k, m, stream); break;
        case GGML_TYPE_IQ4_NL: q4k_skinny_gated_mul_mat_launch_t<GGML_TYPE_IQ4_NL>(gate_codes, up_codes, input, output, f16_out, n, k, m, stream); break;
        default: GGML_ABORT("q4k skinny: unknown type");
    }
    CUDA_CHECK(cudaGetLastError());
}

// ---- mixed-quant gated pair, one pass (Stage 5, v100fast) ----
// Fuses the 24 mixed-quant (gate_w->type != up_w->type) FFN layers of UD-Q4_K_XL in a single K loop.
template <ggml_type TG, ggml_type TU, int SplitK, int RowTiles, int MinBlocks = (512 / (32 * SplitK))>
__launch_bounds__(32 * SplitK, MinBlocks)
__global__ void q4k_skinny_gated_nt_mixed_kernel(
    const uint8_t * __restrict__ gate_codes, const uint8_t * __restrict__ up_codes,
    const half * __restrict__ input, float * __restrict__ output, half * __restrict__ f16_out, int n, int k, int m) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    using codec_g = qskinny_codec<TG>;
    using codec_u = qskinny_codec<TU>;
    constexpr int kOutputElements = RowTiles * 256;
    __shared__ float partials[SplitK][kOutputElements];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tile = blockIdx.x;
    const int quadpair = (lane >> 2) & 3;
    const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_k16 = k >> 4;
    const int sub_blocks = k >> 5;
    const int sub_blocks_per_warp = sub_blocks / SplitK;
    const int sub_block_begin = warp * sub_blocks_per_warp;

    const uint8_t * code_base_g = gate_codes + (size_t) tile * groups_k16 * 32 * codec_g::record_bytes;
    const uint8_t * code_base_u = up_codes   + (size_t) tile * groups_k16 * 32 * codec_u::record_bytes;
    const uint8_t * meta_ptr_g  = gate_codes + codec_g::codes_bytes(n, k) + (size_t) codec_g::meta_bytes * ((size_t) tile * 32 + lane);
    const uint8_t * meta_ptr_u  = up_codes   + codec_u::codes_bytes(n, k) + (size_t) codec_u::meta_bytes * ((size_t) tile * 32 + lane);

    float accum[2][RowTiles][8];
#pragma unroll
    for (int t = 0; t < 2; ++t) {
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                accum[t][row_tile][i] = 0.0f;
            }
        }
    }
    int loaded_meta_kb = -1;
    typename codec_g::meta_t meta_g = {};
    typename codec_u::meta_t meta_u = {};

    for (int sub_block = sub_block_begin; sub_block < sub_block_begin + sub_blocks_per_warp; ++sub_block) {
        const int kb = sub_block >> 3;
        if (kb != loaded_meta_kb) {
            meta_g = codec_g::load_meta(meta_ptr_g + (size_t) kb * n * codec_g::meta_bytes);
            meta_u = codec_u::load_meta(meta_ptr_u + (size_t) kb * n * codec_u::meta_bytes);
            loaded_meta_kb = kb;
        }
        const int group = sub_block << 1;

        uint4 in[RowTiles][4];
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
            const int input_row_idx = row_tile * 8 + row;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                in[row_tile][j] = make_uint4(0, 0, 0, 0);
            }
            if (input_row_idx < m) {
                const half * input_row = input + (size_t) input_row_idx * k + group * 16;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    in[row_tile][j] = *reinterpret_cast<const uint4 *>(input_row + 8 * j);
                }
            }
        }

        // Issue Gate + Up weight loads together so Up HBM2 latency hides behind Gate decode+HMMA!
        const qskinny_code records_g[2] = {
            qskinny_load_group_record<TG>(code_base_g + (size_t) (group + 0) * 32 * codec_g::record_bytes, lane),
            qskinny_load_group_record<TG>(code_base_g + (size_t) (group + 1) * 32 * codec_g::record_bytes, lane),
        };
        const qskinny_code records_u[2] = {
            qskinny_load_group_record<TU>(code_base_u + (size_t) (group + 0) * 32 * codec_u::record_bytes, lane),
            qskinny_load_group_record<TU>(code_base_u + (size_t) (group + 1) * 32 * codec_u::record_bytes, lane),
        };
        // Gate projection (t = 0)
        {
            const qskinny_code (&records)[2] = records_g;
            half2 weights[16];
            codec_g::decode(records, meta_g, sub_block & 7, weights);
            const unsigned * b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
            for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
                const unsigned * a0 = reinterpret_cast<const unsigned *>(&in[row_tile][0]);
                const unsigned * a1 = reinterpret_cast<const unsigned *>(&in[row_tile][1]);
                const unsigned * a2 = reinterpret_cast<const unsigned *>(&in[row_tile][2]);
                const unsigned * a3 = reinterpret_cast<const unsigned *>(&in[row_tile][3]);
                Q4K_SKINNY_MMA_8N8K4(accum[0][row_tile], a0[0], a0[1], b[0],  b[1]);
                Q4K_SKINNY_MMA_8N8K4(accum[0][row_tile], a0[2], a0[3], b[2],  b[3]);
                Q4K_SKINNY_MMA_8N8K4(accum[0][row_tile], a1[0], a1[1], b[4],  b[5]);
                Q4K_SKINNY_MMA_8N8K4(accum[0][row_tile], a1[2], a1[3], b[6],  b[7]);
                Q4K_SKINNY_MMA_8N8K4(accum[0][row_tile], a2[0], a2[1], b[8],  b[9]);
                Q4K_SKINNY_MMA_8N8K4(accum[0][row_tile], a2[2], a2[3], b[10], b[11]);
                Q4K_SKINNY_MMA_8N8K4(accum[0][row_tile], a3[0], a3[1], b[12], b[13]);
                Q4K_SKINNY_MMA_8N8K4(accum[0][row_tile], a3[2], a3[3], b[14], b[15]);
            }
        }
        // Up projection (t = 1)
        {
            const qskinny_code (&records)[2] = records_u;
            half2 weights[16];
            codec_u::decode(records, meta_u, sub_block & 7, weights);
            const unsigned * b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
            for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
                const unsigned * a0 = reinterpret_cast<const unsigned *>(&in[row_tile][0]);
                const unsigned * a1 = reinterpret_cast<const unsigned *>(&in[row_tile][1]);
                const unsigned * a2 = reinterpret_cast<const unsigned *>(&in[row_tile][2]);
                const unsigned * a3 = reinterpret_cast<const unsigned *>(&in[row_tile][3]);
                Q4K_SKINNY_MMA_8N8K4(accum[1][row_tile], a0[0], a0[1], b[0],  b[1]);
                Q4K_SKINNY_MMA_8N8K4(accum[1][row_tile], a0[2], a0[3], b[2],  b[3]);
                Q4K_SKINNY_MMA_8N8K4(accum[1][row_tile], a1[0], a1[1], b[4],  b[5]);
                Q4K_SKINNY_MMA_8N8K4(accum[1][row_tile], a1[2], a1[3], b[6],  b[7]);
                Q4K_SKINNY_MMA_8N8K4(accum[1][row_tile], a2[0], a2[1], b[8],  b[9]);
                Q4K_SKINNY_MMA_8N8K4(accum[1][row_tile], a2[2], a2[3], b[10], b[11]);
                Q4K_SKINNY_MMA_8N8K4(accum[1][row_tile], a3[0], a3[1], b[12], b[13]);
                Q4K_SKINNY_MMA_8N8K4(accum[1][row_tile], a3[2], a3[3], b[14], b[15]);
            }
        }
    }

    float gate_red[(kOutputElements + 32 * SplitK - 1) / (32 * SplitK)];
#pragma unroll
    for (int t = 0; t < 2; ++t) {
#pragma unroll
        for (int row_tile = 0; row_tile < RowTiles; ++row_tile) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int output_row = row_tile * 8 + (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
                const int output_col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
                partials[warp][output_row * 32 + quadpair * 8 + output_col] = accum[t][row_tile][i];
            }
        }
        __syncthreads();
        constexpr int kSlotsPerThread = (kOutputElements + 32 * SplitK - 1) / (32 * SplitK);
#pragma unroll
        for (int idx = 0; idx < kSlotsPerThread; ++idx) {
            const int element = threadIdx.x + idx * (32 * SplitK);
            if (element >= kOutputElements) { break; }
            float value = partials[0][element];
#pragma unroll
            for (int w = 1; w < SplitK; ++w) {
                value += partials[w][element];
            }
            if (t == 0) {
                gate_red[idx] = value;
            } else {
                const int output_row = element >> 5;
                const int output_col = element & 31;
                if (output_row < m) {
                    const float g = gate_red[idx];
                    const float silu = g / (1.0f + expf(-g));
                    const float res = silu * value;
                    const size_t oi = (size_t) output_row * n + tile * 32 + output_col;
                    output[oi] = res;
                    if (f16_out) { f16_out[oi] = __float2half(res); }
                }
            }
        }
        __syncthreads();
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(gate_codes, up_codes, input, output, n, k, m);
#endif
}

template <ggml_type TG, ggml_type TU>
static void q4k_skinny_gated_mixed_launch_t(const uint8_t * gate_codes, const uint8_t * up_codes,
                                            const half * input, float * output, half * f16_out, int n, int k, int m,
                                            cudaStream_t stream) {
    const q4k_skinny_config config = q4k_skinny_config_for(k);
    static const bool gnt_sk8 = [] {
        const char * e = getenv("LLAMA_SM70_SKINNY_GNT_SK8");
        return e && e[0] == '1';
    }();
    static const bool gnt_3cta = [] {
        const char * e = getenv("LLAMA_SM70_SKINNY_GNT_3CTA");
        return e && e[0] == '1';
    }();
    if (m <= 8) {
        if (gnt_3cta && n / 32 >= 160) {
            q4k_skinny_gated_nt_mixed_kernel<TG, TU, 8, 1, 3><<<n / 32, 32 * 8, 0, stream>>>(gate_codes, up_codes, input, output, f16_out, n, k, m);
        } else if (config.split_k == 16 && !(gnt_sk8 && n / 32 >= 160)) {
            q4k_skinny_gated_nt_mixed_kernel<TG, TU, 16, 1><<<n / 32, 32 * 16, 0, stream>>>(gate_codes, up_codes, input, output, f16_out, n, k, m);
        } else {
            q4k_skinny_gated_nt_mixed_kernel<TG, TU, 8, 1><<<n / 32, 32 * 8, 0, stream>>>(gate_codes, up_codes, input, output, f16_out, n, k, m);
        }
    } else {
        if (config.split_k == 16 && !(gnt_sk8 && n / 32 >= 160)) {
            q4k_skinny_gated_nt_mixed_kernel<TG, TU, 16, 2><<<n / 32, 32 * 16, 0, stream>>>(gate_codes, up_codes, input, output, f16_out, n, k, m);
        } else {
            q4k_skinny_gated_nt_mixed_kernel<TG, TU, 8, 2><<<n / 32, 32 * 8, 0, stream>>>(gate_codes, up_codes, input, output, f16_out, n, k, m);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

static bool q4k_skinny_gated_mixed_supported(ggml_type tg, ggml_type tu) {
    auto key = [](ggml_type a, ggml_type b) { return ((int) a << 8) | (int) b; };
    switch (key(tg, tu)) {
        case ((int) GGML_TYPE_Q4_K   << 8) | (int) GGML_TYPE_Q5_K:
        case ((int) GGML_TYPE_Q5_K   << 8) | (int) GGML_TYPE_Q4_K:
        case ((int) GGML_TYPE_IQ4_XS << 8) | (int) GGML_TYPE_Q5_K:
        case ((int) GGML_TYPE_Q5_K   << 8) | (int) GGML_TYPE_IQ4_XS:
        case ((int) GGML_TYPE_IQ4_XS << 8) | (int) GGML_TYPE_Q4_K:
        case ((int) GGML_TYPE_Q4_K   << 8) | (int) GGML_TYPE_IQ4_XS:
        case ((int) GGML_TYPE_Q6_K   << 8) | (int) GGML_TYPE_Q5_K:
        case ((int) GGML_TYPE_Q5_K   << 8) | (int) GGML_TYPE_Q6_K:
        case ((int) GGML_TYPE_IQ4_XS << 8) | (int) GGML_TYPE_Q3_K:
        case ((int) GGML_TYPE_Q3_K   << 8) | (int) GGML_TYPE_IQ4_XS:
        case ((int) GGML_TYPE_IQ4_NL << 8) | (int) GGML_TYPE_IQ4_XS:
        case ((int) GGML_TYPE_IQ4_XS << 8) | (int) GGML_TYPE_IQ4_NL:
        case ((int) GGML_TYPE_IQ4_NL << 8) | (int) GGML_TYPE_Q5_K:
        case ((int) GGML_TYPE_Q5_K   << 8) | (int) GGML_TYPE_IQ4_NL:
            return true;
        default:
            return false;
    }
}

static bool q4k_skinny_gated_mixed_launch(ggml_type tg, ggml_type tu, const uint8_t * gate_codes,
                                          const uint8_t * up_codes, const half * input,
                                          float * output, half * f16_out, int n, int k, int m, cudaStream_t stream) {
    const int key = ((int) tg << 8) | (int) tu;
    switch (key) {
        case ((int) GGML_TYPE_Q4_K   << 8) | (int) GGML_TYPE_Q5_K:   q4k_skinny_gated_mixed_launch_t<GGML_TYPE_Q4_K,   GGML_TYPE_Q5_K  >(gate_codes, up_codes, input, output, f16_out, n, k, m, stream); return true;
        case ((int) GGML_TYPE_Q5_K   << 8) | (int) GGML_TYPE_Q4_K:   q4k_skinny_gated_mixed_launch_t<GGML_TYPE_Q5_K,   GGML_TYPE_Q4_K  >(gate_codes, up_codes, input, output, f16_out, n, k, m, stream); return true;
        case ((int) GGML_TYPE_IQ4_XS << 8) | (int) GGML_TYPE_Q5_K:   q4k_skinny_gated_mixed_launch_t<GGML_TYPE_IQ4_XS, GGML_TYPE_Q5_K  >(gate_codes, up_codes, input, output, f16_out, n, k, m, stream); return true;
        case ((int) GGML_TYPE_Q5_K   << 8) | (int) GGML_TYPE_IQ4_XS: q4k_skinny_gated_mixed_launch_t<GGML_TYPE_Q5_K,   GGML_TYPE_IQ4_XS>(gate_codes, up_codes, input, output, f16_out, n, k, m, stream); return true;
        case ((int) GGML_TYPE_IQ4_XS << 8) | (int) GGML_TYPE_Q4_K:   q4k_skinny_gated_mixed_launch_t<GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K  >(gate_codes, up_codes, input, output, f16_out, n, k, m, stream); return true;
        case ((int) GGML_TYPE_Q4_K   << 8) | (int) GGML_TYPE_IQ4_XS: q4k_skinny_gated_mixed_launch_t<GGML_TYPE_Q4_K,   GGML_TYPE_IQ4_XS>(gate_codes, up_codes, input, output, f16_out, n, k, m, stream); return true;
        case ((int) GGML_TYPE_Q6_K   << 8) | (int) GGML_TYPE_Q5_K:   q4k_skinny_gated_mixed_launch_t<GGML_TYPE_Q6_K,   GGML_TYPE_Q5_K  >(gate_codes, up_codes, input, output, f16_out, n, k, m, stream); return true;
        case ((int) GGML_TYPE_Q5_K   << 8) | (int) GGML_TYPE_Q6_K:   q4k_skinny_gated_mixed_launch_t<GGML_TYPE_Q5_K,   GGML_TYPE_Q6_K  >(gate_codes, up_codes, input, output, f16_out, n, k, m, stream); return true;
        case ((int) GGML_TYPE_IQ4_XS << 8) | (int) GGML_TYPE_Q3_K:   q4k_skinny_gated_mixed_launch_t<GGML_TYPE_IQ4_XS, GGML_TYPE_Q3_K  >(gate_codes, up_codes, input, output, f16_out, n, k, m, stream); return true;
        case ((int) GGML_TYPE_Q3_K   << 8) | (int) GGML_TYPE_IQ4_XS: q4k_skinny_gated_mixed_launch_t<GGML_TYPE_Q3_K,   GGML_TYPE_IQ4_XS>(gate_codes, up_codes, input, output, f16_out, n, k, m, stream); return true;
        case ((int) GGML_TYPE_IQ4_NL << 8) | (int) GGML_TYPE_IQ4_XS: q4k_skinny_gated_mixed_launch_t<GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS>(gate_codes, up_codes, input, output, f16_out, n, k, m, stream); return true;
        case ((int) GGML_TYPE_IQ4_XS << 8) | (int) GGML_TYPE_IQ4_NL: q4k_skinny_gated_mixed_launch_t<GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL>(gate_codes, up_codes, input, output, f16_out, n, k, m, stream); return true;
        case ((int) GGML_TYPE_IQ4_NL << 8) | (int) GGML_TYPE_Q5_K:   q4k_skinny_gated_mixed_launch_t<GGML_TYPE_IQ4_NL, GGML_TYPE_Q5_K  >(gate_codes, up_codes, input, output, f16_out, n, k, m, stream); return true;
        case ((int) GGML_TYPE_Q5_K   << 8) | (int) GGML_TYPE_IQ4_NL: q4k_skinny_gated_mixed_launch_t<GGML_TYPE_Q5_K,   GGML_TYPE_IQ4_NL>(gate_codes, up_codes, input, output, f16_out, n, k, m, stream); return true;
        default: return false;
    }
}

// ---- M=32 two-phase kernel (1Cat fp8_qpn8_sm70.cu:445-573, adapted to the codec) ----

// M=32 needs four 8-row accumulator tiles. Keeping all logical split-K warps resident would
// need 64 KiB of reduction storage for split-16, so half of the physical warps run the
// original logical warp ranges in two ordered phases. The compact first-half sum keeps the
// final reduction in the p0 + ... + p(SplitK-1) order at 36 KiB of shared memory.
template <ggml_type T, int SplitK>
__global__ void q4k_skinny_m32_kernel(
    const uint8_t * __restrict__ codes, const half * __restrict__ input,
    float * __restrict__ output, int n, int k, int m) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    using codec = qskinny_codec<T>;
    static_assert(SplitK == 12 || SplitK == 16,
                  "Q4_K skinny M=32 two-phase kernel supports split-12 or split-16");
    constexpr int kPhysicalWarps = SplitK / 2;
    constexpr int kRowTiles = 4;
    constexpr int kOutputElements = kRowTiles * 256;
    __shared__ float reduction_storage[kPhysicalWarps + 1][kOutputElements];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tile = blockIdx.x;
    const int quadpair = (lane >> 2) & 3;
    const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int groups_k16 = k >> 4;
    const int sub_blocks = k >> 5;
    const int sub_blocks_per_warp = sub_blocks / SplitK;
    const uint8_t * code_base = codes + (size_t) tile * groups_k16 * 32 * codec::record_bytes;
    // meta stride and offset are per codec type
    const uint8_t * meta_ptr = codes + codec::codes_bytes(n, k) +
                               (size_t) codec::meta_bytes * (tile * 32 + lane);

#pragma unroll
    for (int phase = 0; phase < 2; ++phase) {
        float accum[kRowTiles][2][8];
#pragma unroll
        for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
#pragma unroll
            for (int chain = 0; chain < 2; ++chain) {
#pragma unroll
                for (int index = 0; index < 8; ++index) {
                    accum[row_tile][chain][index] = 0.0f;
                }
            }
        }

        const int logical_warp = warp + phase * kPhysicalWarps;
        const int sub_block_begin = logical_warp * sub_blocks_per_warp;
        int loaded_meta_kb = -1;
        typename codec::meta_t meta = {};
#pragma unroll 2
        for (int sub_block = sub_block_begin; sub_block < sub_block_begin + sub_blocks_per_warp; ++sub_block) {
            const int kb = sub_block >> 3;
            if (kb != loaded_meta_kb) {
                meta = codec::load_meta(meta_ptr + (size_t) kb * n * codec::meta_bytes);
                loaded_meta_kb = kb;
            }

            const int group = sub_block << 1;
            const qskinny_code records[2] = { // record load per codec type
                qskinny_load_group_record<T>(code_base + (size_t) (group + 0) * 32 * codec::record_bytes, lane),
                qskinny_load_group_record<T>(code_base + (size_t) (group + 1) * 32 * codec::record_bytes, lane),
            };
            half2 weights[codec::values_per_sub_block / 2];
            codec::decode(records, meta, sub_block & 7, weights);

            const unsigned * b = reinterpret_cast<const unsigned *>(weights);
#pragma unroll
            for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
                uint4 input01 = make_uint4(0, 0, 0, 0);
                uint4 input23 = make_uint4(0, 0, 0, 0);
                uint4 input45 = make_uint4(0, 0, 0, 0);
                uint4 input67 = make_uint4(0, 0, 0, 0);
                const int input_row_idx = row_tile * 8 + row;
                if (input_row_idx < m) {
                    const half * input_row = input + (size_t) input_row_idx * k;
                    input01 = *reinterpret_cast<const uint4 *>(input_row + group * 16);
                    input23 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 8);
                    input45 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 16);
                    input67 = *reinterpret_cast<const uint4 *>(input_row + group * 16 + 24);
                }

                const unsigned * a0 = reinterpret_cast<const unsigned *>(&input01);
                const unsigned * a1 = reinterpret_cast<const unsigned *>(&input23);
                const unsigned * a2 = reinterpret_cast<const unsigned *>(&input45);
                const unsigned * a3 = reinterpret_cast<const unsigned *>(&input67);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][0], a0[0], a0[1], b[0], b[1]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][1], a0[2], a0[3], b[2], b[3]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][0], a1[0], a1[1], b[4], b[5]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][1], a1[2], a1[3], b[6], b[7]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][0], a2[0], a2[1], b[8], b[9]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][1], a2[2], a2[3], b[10], b[11]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][0], a3[0], a3[1], b[12], b[13]);
                Q4K_SKINNY_MMA_8N8K4(accum[row_tile][1], a3[2], a3[3], b[14], b[15]);
            }
        }

#pragma unroll
        for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
#pragma unroll
            for (int index = 0; index < 8; ++index) {
                accum[row_tile][0][index] += accum[row_tile][1][index];
                const int output_row =
                    row_tile * 8 + (index & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
                const int output_col =
                    (index & 1) | (((lane >> 1) & 1) << 1) | ((index >> 2) << 2);
                reduction_storage[warp][output_row * 32 + quadpair * 8 + output_col] =
                    accum[row_tile][0][index];
            }
        }
        __syncthreads();

        for (int element = threadIdx.x; element < kOutputElements;
             element += blockDim.x) {
            float value = phase == 0 ? 0.0f : reduction_storage[kPhysicalWarps][element];
#pragma unroll
            for (int k_warp = 0; k_warp < kPhysicalWarps; ++k_warp) {
                value += reduction_storage[k_warp][element];
            }
            if (phase == 0) {
                reduction_storage[kPhysicalWarps][element] = value;
            } else {
                const int output_row = element >> 5;
                const int output_col = element & 31;
                if (output_row < m) {
                    output[(size_t) output_row * n + tile * 32 + output_col] = value;
                }
            }
        }
        __syncthreads();
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(codes, input, output, n, k, m);
#endif
}

#undef Q4K_SKINNY_MMA_8N8K4

// Only split-16 is enabled in the launcher below: the shapes that reach M > 16 all select
// split 16, and the kernel keeps the 1Cat split-12 variant unused.
template <ggml_type T>
static void q4k_skinny_m32_mul_mat_launch(const uint8_t * codes, const half * input,
                                          float * output, int n, int k, int m,
                                          cudaStream_t stream) {
    constexpr int kSplitK = 16;
    q4k_skinny_m32_kernel<T, kSplitK><<<n / 32, 32 * (kSplitK / 2), 0, stream>>>(
        codes, input, output, n, k, m);
    CUDA_CHECK(cudaGetLastError());
}

// Dispatches an already F16-converted input: M <= 16 keeps the single-stage kernel, M = 17..32
// uses the two-phase M=32 kernel and M = 33..64 runs it on two row blocks. Shapes whose
// split-K is not 16 have no M > 16 kernel and return false so the caller can fall back.
template <ggml_type T>
static bool q4k_skinny_mul_mat_dispatch_t(const uint8_t * codes, const half * input, float * output,
                                          int n, int k, int m, cudaStream_t stream) {
    GGML_ASSERT(m >= 1 && m <= 64);
    if (m <= 16) {
        q4k_skinny_mul_mat_launch_t<T>(codes, input, output, n, k, m, stream);
        return true;
    }
    if (q4k_skinny_config_for(k).split_k != 16) {
        return false;
    }
    if (m <= 32) {
        q4k_skinny_m32_mul_mat_launch<T>(codes, input, output, n, k, m, stream);
    } else {
        q4k_skinny_m32_mul_mat_launch<T>(codes, input, output, n, k, 32, stream);
        q4k_skinny_m32_mul_mat_launch<T>(codes, input + (size_t) 32 * k,
                                        output + (size_t) 32 * n, n, k, m - 32, stream);
    }
    return true;
}

// the dispatch is instantiated per codec type
static bool q4k_skinny_mul_mat_dispatch(ggml_type type, const uint8_t * codes, const half * input,
                                        float * output, int n, int k, int m, cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_Q2_K: return q4k_skinny_mul_mat_dispatch_t<GGML_TYPE_Q2_K>(codes, input, output, n, k, m, stream);
        case GGML_TYPE_Q3_K: return q4k_skinny_mul_mat_dispatch_t<GGML_TYPE_Q3_K>(codes, input, output, n, k, m, stream);
        case GGML_TYPE_Q4_K: return q4k_skinny_mul_mat_dispatch_t<GGML_TYPE_Q4_K>(codes, input, output, n, k, m, stream);
        case GGML_TYPE_Q5_K: return q4k_skinny_mul_mat_dispatch_t<GGML_TYPE_Q5_K>(codes, input, output, n, k, m, stream);
        case GGML_TYPE_Q6_K: return q4k_skinny_mul_mat_dispatch_t<GGML_TYPE_Q6_K>(codes, input, output, n, k, m, stream);
        case GGML_TYPE_IQ4_XS: return q4k_skinny_mul_mat_dispatch_t<GGML_TYPE_IQ4_XS>(codes, input, output, n, k, m, stream);
        case GGML_TYPE_IQ4_NL: return q4k_skinny_mul_mat_dispatch_t<GGML_TYPE_IQ4_NL>(codes, input, output, n, k, m, stream);
        // IQ codec family (out-of-place, side buffers - T2)
        case GGML_TYPE_IQ3_S:   return q4k_skinny_mul_mat_dispatch_t<GGML_TYPE_IQ3_S  >(codes, input, output, n, k, m, stream);
        case GGML_TYPE_IQ3_XXS: return q4k_skinny_mul_mat_dispatch_t<GGML_TYPE_IQ3_XXS>(codes, input, output, n, k, m, stream);
        case GGML_TYPE_IQ2_XS:  return q4k_skinny_mul_mat_dispatch_t<GGML_TYPE_IQ2_XS >(codes, input, output, n, k, m, stream);
        case GGML_TYPE_IQ2_S:   return q4k_skinny_mul_mat_dispatch_t<GGML_TYPE_IQ2_S  >(codes, input, output, n, k, m, stream);
        default: GGML_ABORT("q4k skinny: unknown type"); return false;
    }
}

static half * qskinny_get_f16_input(ggml_backend_cuda_context & ctx, const ggml_tensor * src1,
                                    int64_t ne, ggml_cuda_pool_alloc<half> & pool_alloc) {
    const size_t f16_bytes = (size_t) ne * sizeof(half);
    const size_t half_cap = ctx.f16_act_cap >> 1;
    if (ctx.curr_stream_no == 0 && ctx.f16_act_buf != nullptr && f16_bytes <= half_cap) {
        if (ctx.f16_act_src == src1->data && (ctx.f16_act_ne == ne || ctx.f16_act_ne == -ne)) {
            return (ctx.f16_act_ne > 0)
                ? (half *) ctx.f16_act_buf
                : (half *) ((uint8_t *) ctx.f16_act_buf + half_cap);
        }
        half * ptr = (half *) ctx.f16_act_buf;
        const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
        GGML_ASSERT(to_fp16 != nullptr);
        to_fp16(src1->data, ptr, ne, ctx.stream());
        ctx.f16_act_src = src1->data;
        ctx.f16_act_src_bytes = (size_t) ne * sizeof(float);
        ctx.f16_act_ne = ne;
        return ptr;
    }
    pool_alloc.alloc(ne);
    const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
    GGML_ASSERT(to_fp16 != nullptr);
    to_fp16(src1->data, pool_alloc.get(), ne, ctx.stream());
    return pool_alloc.get();
}

bool ggml_cuda_q4k_skinny_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                  const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_q4k_skinny_is_repacked(src0));
    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = src1->ne[1];
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst) ||
            src1->ne[2] != 1 || src1->ne[3] != 1 || m < 1 || m > 64) {
        return false;
    }

    ggml_cuda_pool_alloc<half> input_pool(ctx.pool());
    half * input_ptr = qskinny_get_f16_input(ctx, src1, m * k, input_pool);

    return q4k_skinny_mul_mat_dispatch(src0->type, (const uint8_t *) src0->data, input_ptr,
                                       (float *) dst->data, (int) n, (int) k, (int) m,
                                       ctx.stream());
}
bool ggml_cuda_q4k_skinny_mul_mat_gated(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w,
                                        const ggml_tensor * up_w, const ggml_tensor * src1,
                                        ggml_tensor * dst) {
    if (!ggml_cuda_q4k_skinny_is_repacked(gate_w) || !ggml_cuda_q4k_skinny_is_repacked(up_w)) {
        return false;
    }
    const bool mixed = (gate_w->type != up_w->type);
    if (mixed) {
        const int64_t m_chk = src1->ne[1];
        if (m_chk < 1 || m_chk > 16 || !q4k_skinny_gated_mixed_supported(gate_w->type, up_w->type)) {
            return false;
        }
    }
    const int64_t k = gate_w->ne[0];
    const int64_t n = gate_w->ne[1];
    const int64_t m = src1->ne[1];
    if (!ggml_are_same_shape(gate_w, up_w) ||
            src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst) ||
            src1->ne[2] != 1 || src1->ne[3] != 1 ||
            dst->ne[0] != n || dst->ne[1] != m || dst->ne[2] != 1 || dst->ne[3] != 1 ||
            m < 1 || m > 64) { // M=32 kernel
        return false;
    }
    // M = 17..64 uses the two-phase M=32 kernel, which only exists for split-16 shapes;
    // return false before any allocation so the caller can fall back
    if (m > 16 && q4k_skinny_config_for(k).split_k != 16) {
        return false;
    }

    ggml_cuda_pool_alloc<half> input_pool(ctx.pool());
    half * input_ptr = qskinny_get_f16_input(ctx, src1, m * k, input_pool);

    if (m <= 16) {
        static const bool gated_f16_out = [] {
            const char * e = getenv("LLAMA_SM70_GATED_F16_OUT");
            return e == nullptr || e[0] != '0';
        }();
        const size_t half_cap = ctx.f16_act_cap >> 1;
        const size_t out_f16_bytes = (size_t) (n * m) * sizeof(half);
        half * base0 = (half *) ctx.f16_act_buf;
        half * base1 = (half *) ((uint8_t *) ctx.f16_act_buf + half_cap);
        const bool can_f16 = gated_f16_out && ctx.curr_stream_no == 0 && ctx.f16_act_buf != nullptr && out_f16_bytes <= half_cap;
        const bool use_slot1 = (input_ptr == base0);
        half * f16_out = can_f16 ? (use_slot1 ? base1 : base0) : nullptr;

        bool ok = true;
        if (mixed) {
            ok = q4k_skinny_gated_mixed_launch(gate_w->type, up_w->type, (const uint8_t *) gate_w->data,
                                               (const uint8_t *) up_w->data, input_ptr, (float *) dst->data,
                                               f16_out, (int) n, (int) k, (int) m, ctx.stream());
        } else {
            q4k_skinny_gated_mul_mat_launch(gate_w->type, (const uint8_t *) gate_w->data,
                                            (const uint8_t *) up_w->data, input_ptr, (float *) dst->data,
                                            f16_out, (int) n, (int) k, (int) m, ctx.stream());
        }
        if (ok && f16_out != nullptr) {
            ctx.f16_act_src       = dst->data;
            ctx.f16_act_src_bytes = (size_t) (n * m) * sizeof(float);
            ctx.f16_act_ne        = use_slot1 ? -(n * m) : (n * m);
        }
        return ok;
    }

    // M = 17..64 runs both projections with the M=32 kernel, then the SwiGLU epilogue, with
    // the same silu formula as the fused kernel
    float * gate = (float *) dst->data;
    ggml_cuda_pool_alloc<float> up(ctx.pool(), n * m);
    GGML_ASSERT(q4k_skinny_mul_mat_dispatch(gate_w->type, (const uint8_t *) gate_w->data, input_ptr, gate,
                                            (int) n, (int) k, (int) m, ctx.stream()));
    GGML_ASSERT(q4k_skinny_mul_mat_dispatch(up_w->type, (const uint8_t *) up_w->data, input_ptr, up.get(),
                                            (int) n, (int) k, (int) m, ctx.stream()));

    const int count = (int) (n * m);
    q4k_skinny_swiglu_kernel<<<(count + 255) / 256, 256, 0, ctx.stream()>>>(gate, up.get(), count);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_q4k_skinny_mul_mat_multi(ggml_backend_cuda_context & ctx,
                                        const ggml_tensor * const src0s[4], ggml_tensor * const dsts[4],
                                        int n_nodes, const ggml_tensor * src1) {
    if (n_nodes < 2 || n_nodes > 4) {
        return false;
    }
    const int64_t k = src0s[0]->ne[0];
    const int64_t m = src1->ne[1];
    if (src1->type != GGML_TYPE_F32 || !ggml_is_contiguous(src1) ||
            src1->ne[2] != 1 || src1->ne[3] != 1 || m < 1 || m > 64) { // M=32 kernel
        return false;
    }
    // M = 17..64 uses the two-phase M=32 kernel, which only exists for split-16 shapes;
    // return false before any allocation so the caller can fall back
    if (m > 16 && q4k_skinny_config_for(k).split_k != 16) {
        return false;
    }

    q4k_skinny_multi_params p = {};
    p.k = (int) k;
    p.m = (int) m;
    p.n_seg = n_nodes;
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * w = src0s[i];
        ggml_tensor * dst = dsts[i];
        const int64_t n = w->ne[1];
        if (!ggml_cuda_q4k_skinny_is_repacked(w) ||
                // the multi kernel runs one codec type for all segments
                w->type != src0s[0]->type ||
                w->ne[0] != k || w->ne[2] != 1 || w->ne[3] != 1 || !ggml_is_contiguous(w) ||
                dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
                dst->ne[0] != n || dst->ne[1] != m || dst->ne[2] != 1 || dst->ne[3] != 1) {
            return false;
        }
        q4k_skinny_seg & seg = p.seg[i];
        seg.codes = (const uint8_t *) w->data;
        seg.n = (int) n;
        seg.dst = (float *) dst->data;
    }
    for (int s = 0; s < p.n_seg; ++s) {
        p.seg[s].first_tile = p.n_tiles;
        p.n_tiles += p.seg[s].n / 32;
    }

    ggml_cuda_pool_alloc<half> input_pool(ctx.pool());
    half * input_ptr = qskinny_get_f16_input(ctx, src1, m * k, input_pool);
    p.input = input_ptr;

    if (m <= 16) {
        q4k_skinny_multi_mul_mat_launch(src0s[0]->type, p, ctx.stream());
        return true;
    }

    // M = 17..64 has no segment dispatch in the M=32 kernel, so each repacked weight gets
    // its own launch from the one converted input (the group is homogeneous)
    for (int s = 0; s < p.n_seg; ++s) {
        const q4k_skinny_seg & seg = p.seg[s];
        GGML_ASSERT(q4k_skinny_mul_mat_dispatch(src0s[s]->type, seg.codes, p.input, seg.dst, seg.n,
                                                p.k, (int) m, ctx.stream()));
    }
    return true;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

bool ggml_cuda_q4k_skinny_is_repacked(const ggml_tensor * t) {
    return t->extra == (const void *) &q4k_skinny_marker;
}

bool ggml_cuda_skinny_env_disabled() {
    return ggml_cuda_skinny_mode() == 0;
}

bool ggml_cuda_q4k_skinny_can_repack(const ggml_tensor * t) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(t);
    return false;
#else
    if (ggml_cuda_skinny_env_disabled()) {
        return false;
    }
    // the codec family serves Q2_K/Q3_K/Q4_K/Q5_K/Q6_K (see qskinny_codec) and all
    // five are repacked at runtime. Their compute moves from the MMVQ q8_1 path to F16, which
    // changes greedy tokens at ulp level; that is the intended cost of covering the mixed types.
    switch (t->type) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            break;
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL: {
            // LLAMA_SM70_SKINNY_IQ4=0 keeps IQ4_XS on the stock path
            static const bool iq4_on = [] {
                const char * e = getenv("LLAMA_SM70_SKINNY_IQ4");
                return e == nullptr || e[0] != '0';
            }();
            if (!iq4_on) {
                return false;
            }
        } break;
        default:
            return false;
    }
    if (strcmp(t->name, "token_embd.weight") == 0) { return false; }
    if (strcmp(t->name, "output.weight") == 0) {
        // the DSpine drafter reads rows of the head with GET_ROWS, served by ggml_cuda_q4k_skinny_get_rows.
        // LLAMA_SM70_SKINNY_HEAD=0 keeps the head on the stock path.
        static const bool head_on = [] {
            const char * e = getenv("LLAMA_SM70_SKINNY_HEAD");
            return e == nullptr || e[0] != '0';
        }();
        if (!head_on) {
            return false;
        }
    }
    if (t->view_src != nullptr || t->op != GGML_OP_NONE || t->extra != nullptr) {
        return false;
    }
    if (t->buffer == nullptr || ggml_backend_buffer_get_type(t->buffer) != ggml_backend_cuda_buffer_type(ggml_cuda_get_device())) {
        return false;
    }
    if (ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return false;
    }
    if (ggml_cuda_info().devices[ggml_cuda_get_device()].cc != GGML_CUDA_CC_VOLTA) {
        return false;
    }
    if (!ggml_is_contiguous(t) || t->ne[2] != 1 || t->ne[3] != 1) {
        return false;
    }
    const int64_t k = t->ne[0];
    const int64_t n = t->ne[1];
    // a whole number of 256-value super-blocks (k % 256 == 0) and whole 32-row tiles; k >= 256
    // keeps the split-K, the M=1 specialization and the to_f16 tile loop non-empty. Under
    // tensor split a device can hold an empty K slice, such a weight is never multiplied.
    return k >= 256 && k % 256 == 0 && n % 32 == 0 && n >= 32;
#endif
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// the repack kernel is instantiated per codec type
template <ggml_type T>
static void q4k_skinny_repack_tiles(const void * staging, uint8_t * codes, uint8_t * meta,
                                    int n, int k, int t0, int t1, cudaStream_t stream) {
    qskinny_repack_kernel<T><<<(unsigned) (t1 - t0), 256, 0, stream>>>(
        (const uint8_t *) staging, codes, meta, n, k, t0);
}

static void q4k_skinny_repack_tiles_dispatch(ggml_type type, const void * staging, uint8_t * codes,
                                             uint8_t * meta, int n, int k, int t0, int t1,
                                             cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_Q2_K: q4k_skinny_repack_tiles<GGML_TYPE_Q2_K>(staging, codes, meta, n, k, t0, t1, stream); break;
        case GGML_TYPE_Q3_K: q4k_skinny_repack_tiles<GGML_TYPE_Q3_K>(staging, codes, meta, n, k, t0, t1, stream); break;
        case GGML_TYPE_Q4_K: q4k_skinny_repack_tiles<GGML_TYPE_Q4_K>(staging, codes, meta, n, k, t0, t1, stream); break;
        case GGML_TYPE_Q5_K: q4k_skinny_repack_tiles<GGML_TYPE_Q5_K>(staging, codes, meta, n, k, t0, t1, stream); break;
        case GGML_TYPE_Q6_K: q4k_skinny_repack_tiles<GGML_TYPE_Q6_K>(staging, codes, meta, n, k, t0, t1, stream); break;
        case GGML_TYPE_IQ4_XS: q4k_skinny_repack_tiles<GGML_TYPE_IQ4_XS>(staging, codes, meta, n, k, t0, t1, stream); break;
        case GGML_TYPE_IQ4_NL: q4k_skinny_repack_tiles<GGML_TYPE_IQ4_NL>(staging, codes, meta, n, k, t0, t1, stream); break;
        default: GGML_ABORT("q4k skinny: unknown type");
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_q4k_skinny_repack_inplace(ggml_backend_cuda_context & ctx, ggml_tensor * t) {
    const int64_t k = t->ne[0];
    const int64_t n = t->ne[1];
    const int64_t ntiles = n / 32;
    GGML_ASSERT(k >= 256 && ntiles >= 1);
    // one codec record per 16 K values and row, one meta per 256 K values and row;
    // 16*record_bytes + meta_bytes is the block size, so codes + meta fill the tensor
    const size_t record_bytes = t->type == GGML_TYPE_Q2_K ? 4  :
                                t->type == GGML_TYPE_Q3_K ? 6  :
                                t->type == GGML_TYPE_Q5_K ? 10 :
                                t->type == GGML_TYPE_Q6_K ? 12 : 8;
    const size_t meta_bytes = t->type == GGML_TYPE_Q2_K ? 20 :
                              t->type == GGML_TYPE_Q3_K ? 14 :
                              t->type == GGML_TYPE_Q6_K ? 18 :
                              t->type == GGML_TYPE_IQ4_XS ? 8 : 16;
    const size_t tile_src_bytes = (size_t) k * (16 * record_bytes + meta_bytes) / 8;
    const size_t tile_dst_bytes = 2 * record_bytes * (size_t) k;
    const size_t meta_total = (size_t) n * k * meta_bytes / 256;
    const size_t codes_total = (size_t) n * k * record_bytes / 16;

    constexpr size_t staging_max = 32ull * 1024 * 1024;
    size_t tiles_per_chunk = staging_max / tile_src_bytes;
    if (tiles_per_chunk < 1) {
        tiles_per_chunk = 1;
    }
    if (tiles_per_chunk > (size_t) ntiles) {
        tiles_per_chunk = (size_t) ntiles;
    }

    cudaStream_t stream = ctx.stream();
    uint8_t * data = (uint8_t *) t->data;

    void * staging = nullptr;
    void * meta_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&staging, tiles_per_chunk * tile_src_bytes));
    CUDA_CHECK(cudaMalloc(&meta_tmp, meta_total));

    // Chunk [t0, t1) writes codes to [t0*tile_dst, t1*tile_dst), which ends before the source
    // [t1*tile_src, ...) of the remaining tiles (tile_dst <= tile_src), and its own source was
    // copied to staging first. The meta region [codes_total, codes_total + meta_total) overlaps
    // the sources of the last tiles, so it is written only after all codes are done.
    for (int64_t t0 = 0; t0 < ntiles; t0 += (int64_t) tiles_per_chunk) {
        const int64_t t1 = std::min(t0 + (int64_t) tiles_per_chunk, ntiles);
        CUDA_CHECK(cudaMemcpyAsync(staging, data + t0 * tile_src_bytes, (t1 - t0) * tile_src_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
        q4k_skinny_repack_tiles_dispatch(t->type, staging, data + t0 * tile_dst_bytes, (uint8_t *) meta_tmp,
                                         (int) n, (int) k, (int) t0, (int) t1, stream);
    }
    CUDA_CHECK(cudaMemcpyAsync(data + codes_total, meta_tmp, meta_total,
                               cudaMemcpyDeviceToDevice, stream));

    // no pool: the staging must not stay resident after the one-time repack
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(staging));
    CUDA_CHECK(cudaFree(meta_tmp));

    t->extra = &q4k_skinny_marker;
}

void ggml_cuda_q4k_skinny_prepass(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph) {
    // The idle counter is shared with the Q8_0 prepass, which controls the scan rate for both:
    // once it reaches 1024, neither prepass finds anything new, so the graph walk is skipped.
    // This prepass never modifies the counter so that the Q8_0 behaviour is unchanged.
    if (ctx.q8_skinny_idle_scans >= 1024) {
        return;
    }

    // candidates: Q4_K weights used as MUL_MAT src0; none in steady state
    std::vector<ggml_tensor *> candidates;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT) {
            continue;
        }
        ggml_tensor * src0 = node->src[0];
        if (!ggml_cuda_q4k_skinny_can_repack(src0) || ggml_cuda_q4k_skinny_is_repacked(src0)) {
            continue;
        }
        if (std::find(candidates.begin(), candidates.end(), src0) == candidates.end()) {
            candidates.push_back(src0);
        }
    }
    if (candidates.empty()) {
        return;
    }

    // repacking cannot be recorded into a CUDA graph; the capture is checked only here so that
    // the steady state costs no CUDA API call
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &capture_status));
    if (capture_status != cudaStreamCaptureStatusNone) {
        return;
    }

    // Repacked data can only be read by the new kernel, so drop candidates that are read
    // anywhere else (non-MUL_MAT src, MUL_MAT src1) or viewed by another tensor.
    for (int i = 0; i < cgraph->n_nodes && !candidates.empty(); ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node->view_src != nullptr) {
            candidates.erase(std::remove(candidates.begin(), candidates.end(), node->view_src),
                             candidates.end());
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * src = node->src[j];
            if (src == nullptr) {
                continue;
            }
            if (src->view_src != nullptr) {
                candidates.erase(std::remove(candidates.begin(), candidates.end(), src->view_src),
                                 candidates.end());
            }
            if (!((node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_GET_ROWS) && j == 0)) {
                candidates.erase(std::remove(candidates.begin(), candidates.end(), src),
                                 candidates.end());
            }
        }
    }

    for (ggml_tensor * t : candidates) {
        if (ggml_cuda_skinny_mode() == 2) {
            // side buffer: the original bytes survive, so the decode can fall back to
            // the stock path (see ggml_cuda_mul_mat_iq_skinny)
            ggml_cuda_iq_skinny_repack(ctx, t);
        } else {
            ggml_cuda_q4k_skinny_repack_inplace(ctx, t);
        }
    }
}

void ggml_cuda_q4k_skinny_to_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream) {
    const int n = (int) src0->ne[1];
    const int k = (int) src0->ne[0];
    const uint8_t * data = (const uint8_t *) src0->data;
    const dim3 grid((unsigned) (k / 256), (unsigned) (n / 32), 1);
    // the dequantize kernel is instantiated per codec type
    switch (src0->type) {
        case GGML_TYPE_Q2_K: qskinny_to_f16_kernel<GGML_TYPE_Q2_K><<<grid, 256, 0, stream>>>(dst, data, n, k); break;
        case GGML_TYPE_Q3_K: qskinny_to_f16_kernel<GGML_TYPE_Q3_K><<<grid, 256, 0, stream>>>(dst, data, n, k); break;
        case GGML_TYPE_Q4_K: qskinny_to_f16_kernel<GGML_TYPE_Q4_K><<<grid, 256, 0, stream>>>(dst, data, n, k); break;
        case GGML_TYPE_Q5_K: qskinny_to_f16_kernel<GGML_TYPE_Q5_K><<<grid, 256, 0, stream>>>(dst, data, n, k); break;
        case GGML_TYPE_Q6_K: qskinny_to_f16_kernel<GGML_TYPE_Q6_K><<<grid, 256, 0, stream>>>(dst, data, n, k); break;
        case GGML_TYPE_IQ4_XS: qskinny_to_f16_kernel<GGML_TYPE_IQ4_XS><<<grid, 256, 0, stream>>>(dst, data, n, k); break;
        case GGML_TYPE_IQ4_NL: qskinny_to_f16_kernel<GGML_TYPE_IQ4_NL><<<grid, 256, 0, stream>>>(dst, data, n, k); break;
        default: GGML_ABORT("q4k skinny to_f16: unknown type");
    }
    CUDA_CHECK(cudaGetLastError());
}

// ---- GET_ROWS on a repacked weight (v100fast): decode the selected rows back to float ----

// One CTA per selected row, one thread per 32-value sub-block. decode() returns the sub-block's values in
// logical K order (pair i = values 2i, 2i+1), as qskinny_to_f16_kernel relies on.
template <ggml_type T>
__global__ void qskinny_get_rows_kernel(float * __restrict__ output, const uint8_t * __restrict__ codes,
                                        const int32_t * __restrict__ ids, int n, int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    using codec = qskinny_codec<T>;
    const int row    = ids[blockIdx.x];
    const int tile   = row >> 5;
    const int lane   = qpn8_lane_from_col(row & 31);
    const int groups = k >> 4;
    const uint8_t * code_base = codes + (size_t) tile * groups * 32 * codec::record_bytes;
    const uint8_t * meta_ptr = codes + codec::codes_bytes(n, k) +
                               (size_t) codec::meta_bytes * ((size_t) tile * 32 + lane);
    float * out_row = output + (size_t) blockIdx.x * k;
    const int n_sb = k >> 5;
    const int n_sb_pad = (n_sb + 31) & ~31;
    for (int sb_i = threadIdx.x; sb_i < n_sb_pad; sb_i += blockDim.x) {
        const int sub_block = sb_i < n_sb ? sb_i : 0;
        const int kb = sub_block >> 3;
        const typename codec::meta_t meta = codec::load_meta(meta_ptr + (size_t) kb * n * codec::meta_bytes);
        const int group = sub_block << 1;
        const qskinny_code records[2] = {
            qskinny_load_group_record<T>(code_base + (size_t) (group + 0) * 32 * codec::record_bytes, lane),
            qskinny_load_group_record<T>(code_base + (size_t) (group + 1) * 32 * codec::record_bytes, lane),
        };
        half2 weights[codec::values_per_sub_block / 2];
        codec::decode(records, meta, sub_block & 7, weights);
        if (sb_i < n_sb) {
#pragma unroll
            for (int i = 0; i < codec::values_per_sub_block / 2; ++i) {
                const float2 v = __half22float2(weights[i]);
                out_row[sub_block * 32 + 2 * i + 0] = v.x;
                out_row[sub_block * 32 + 2 * i + 1] = v.y;
            }
        }
    }
#else
    NO_DEVICE_CODE;
    GGML_UNUSED_VARS(output, codes, ids, n, k);
#endif
}

bool ggml_cuda_q4k_skinny_get_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                   const ggml_tensor * src1, ggml_tensor * dst) {
    if (!ggml_cuda_q4k_skinny_is_repacked(src0)) {
        return false;
    }
    const int     k     = (int) src0->ne[0];
    const int     n     = (int) src0->ne[1];
    const int64_t n_ids = ggml_nelements(src1);
    GGML_ASSERT(src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src1) && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_nelements(dst) == n_ids * k);
    const uint8_t * codes  = (const uint8_t *) src0->data;
    const int32_t * ids    = (const int32_t *) src1->data;
    float         * out    = (float *) dst->data;
    cudaStream_t    stream = ctx.stream();
    const dim3 grid((unsigned) n_ids, 1, 1);
    switch (src0->type) {
        case GGML_TYPE_Q2_K:   qskinny_get_rows_kernel<GGML_TYPE_Q2_K  ><<<grid, 64, 0, stream>>>(out, codes, ids, n, k); break;
        case GGML_TYPE_Q3_K:   qskinny_get_rows_kernel<GGML_TYPE_Q3_K  ><<<grid, 64, 0, stream>>>(out, codes, ids, n, k); break;
        case GGML_TYPE_Q4_K:   qskinny_get_rows_kernel<GGML_TYPE_Q4_K  ><<<grid, 64, 0, stream>>>(out, codes, ids, n, k); break;
        case GGML_TYPE_Q5_K:   qskinny_get_rows_kernel<GGML_TYPE_Q5_K  ><<<grid, 64, 0, stream>>>(out, codes, ids, n, k); break;
        case GGML_TYPE_Q6_K:   qskinny_get_rows_kernel<GGML_TYPE_Q6_K  ><<<grid, 64, 0, stream>>>(out, codes, ids, n, k); break;
        case GGML_TYPE_IQ4_XS: qskinny_get_rows_kernel<GGML_TYPE_IQ4_XS><<<grid, 64, 0, stream>>>(out, codes, ids, n, k); break;
        case GGML_TYPE_IQ4_NL: qskinny_get_rows_kernel<GGML_TYPE_IQ4_NL><<<grid, 64, 0, stream>>>(out, codes, ids, n, k); break;
        default: GGML_ABORT("q4k skinny get_rows: unknown type");
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// ---- out-of-place entry points for the IQ codec family (T2) ----

bool ggml_cuda_q4k_skinny_type_is_iq(ggml_type type) {
    return type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ3_XXS ||
           type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S;
}

// codes/meta byte counts for every side-buffer-supported type (the codecs of both
// families are visible in this TU)
size_t ggml_cuda_skinny_codes_bytes(ggml_type type, int n, int k) {
    switch (type) {
        case GGML_TYPE_IQ3_S:   return qskinny_codec<GGML_TYPE_IQ3_S  >::codes_bytes(n, k);
        case GGML_TYPE_IQ3_XXS: return qskinny_codec<GGML_TYPE_IQ3_XXS>::codes_bytes(n, k);
        case GGML_TYPE_IQ2_XS:  return qskinny_codec<GGML_TYPE_IQ2_XS >::codes_bytes(n, k);
        case GGML_TYPE_IQ2_S:   return qskinny_codec<GGML_TYPE_IQ2_S  >::codes_bytes(n, k);
        case GGML_TYPE_Q2_K:    return qskinny_codec<GGML_TYPE_Q2_K   >::codes_bytes(n, k);
        case GGML_TYPE_Q3_K:    return qskinny_codec<GGML_TYPE_Q3_K   >::codes_bytes(n, k);
        case GGML_TYPE_Q4_K:    return qskinny_codec<GGML_TYPE_Q4_K   >::codes_bytes(n, k);
        case GGML_TYPE_Q5_K:    return qskinny_codec<GGML_TYPE_Q5_K   >::codes_bytes(n, k);
        case GGML_TYPE_Q6_K:    return qskinny_codec<GGML_TYPE_Q6_K   >::codes_bytes(n, k);
        case GGML_TYPE_IQ4_XS:    return qskinny_codec<GGML_TYPE_IQ4_XS >::codes_bytes(n, k);
        case GGML_TYPE_IQ4_NL:    return qskinny_codec<GGML_TYPE_IQ4_NL >::codes_bytes(n, k);
        default:                GGML_ABORT("skinny: unknown type");
    }
}

size_t ggml_cuda_skinny_meta_bytes(ggml_type type, int n, int k) {
    const int mb = type == GGML_TYPE_IQ3_S   ? qskinny_codec<GGML_TYPE_IQ3_S  >::meta_bytes :
                   type == GGML_TYPE_IQ3_XXS ? qskinny_codec<GGML_TYPE_IQ3_XXS>::meta_bytes :
                   type == GGML_TYPE_IQ2_XS  ? qskinny_codec<GGML_TYPE_IQ2_XS >::meta_bytes :
                   type == GGML_TYPE_IQ2_S   ? qskinny_codec<GGML_TYPE_IQ2_S  >::meta_bytes :
                   type == GGML_TYPE_Q2_K    ? qskinny_codec<GGML_TYPE_Q2_K   >::meta_bytes :
                   type == GGML_TYPE_Q3_K    ? qskinny_codec<GGML_TYPE_Q3_K   >::meta_bytes :
                   type == GGML_TYPE_Q4_K    ? qskinny_codec<GGML_TYPE_Q4_K   >::meta_bytes :
                   type == GGML_TYPE_Q5_K    ? qskinny_codec<GGML_TYPE_Q5_K   >::meta_bytes :
                   type == GGML_TYPE_IQ4_XS  ? qskinny_codec<GGML_TYPE_IQ4_XS >::meta_bytes :
                   type == GGML_TYPE_IQ4_NL  ? qskinny_codec<GGML_TYPE_IQ4_NL >::meta_bytes :
                                               qskinny_codec<GGML_TYPE_Q6_K   >::meta_bytes;
    // meta layout: [k/256][n/32][32][meta_bytes]
    return (size_t) (k / 256) * n * mb;
}

// every type the out-of-place (side buffer) route supports
bool ggml_cuda_q4k_skinny_side_supported(ggml_type type) {
    return ggml_cuda_q4k_skinny_type_is_iq(type) ||
           type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K ||
           type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K;
}

// Mode selection for the k-quant family (env LLAMA_SM70_SKINNY):
//   0 = off (stock paths), 1 = legacy in-place repack (skinny everywhere),
//   2 = side-buffer repack (default): skinny for M > 64 (f16 expansion, the
//       measured +10 % prefill) and the stock path for M <= 64 (the measured
//       -8..-36 % decode of the in-place route is why).
int ggml_cuda_skinny_mode() {
    static const int mode = [] {
        const char * e = getenv("LLAMA_SM70_SKINNY");
        if (e == nullptr) {
            // default 1 (v100fast): in-place repack for DSpine m=16 Volta Tensor Cores
            return 1;
        }
        const int m = atoi(e);
        return (m == 0 || m == 1) ? m : 2;
    }();
    return mode;
}

bool ggml_cuda_q4k_skinny_mul_mat_codes(ggml_type type, const uint8_t * codes, const half * input,
                                        float * output, int n, int k, int m, cudaStream_t stream) {
    if (!ggml_cuda_q4k_skinny_side_supported(type)) {
        return false;
    }
    return q4k_skinny_mul_mat_dispatch(type, codes, input, output, n, k, m, stream);
}

void ggml_cuda_q4k_skinny_to_f16_codes(ggml_type type, const uint8_t * codes, half * dst,
                                       int n, int k, cudaStream_t stream) {
    // design-B decode reads this TU's LUT; initialize it before the first IQ launch
    // (see the LUT comment in iq-skinny.cuh)
    // the LUT is per-device memory, so the flag has to be per device too
    static bool lut_ready[GGML_CUDA_MAX_DEVICES] = {};
    const int lut_dev = ggml_cuda_get_device();
    if (!lut_ready[lut_dev] && ggml_cuda_q4k_skinny_type_is_iq(type)) {
        iq_skinny_init_lut(stream);
        lut_ready[lut_dev] = true;
    }
    const dim3 grid((unsigned) (k / 256), (unsigned) (n / 32), 1);
    switch (type) {
        case GGML_TYPE_Q2_K:   qskinny_to_f16_kernel<GGML_TYPE_Q2_K  ><<<grid, 256, 0, stream>>>(dst, codes, n, k); break;
        case GGML_TYPE_Q3_K:   qskinny_to_f16_kernel<GGML_TYPE_Q3_K  ><<<grid, 256, 0, stream>>>(dst, codes, n, k); break;
        case GGML_TYPE_Q4_K:   qskinny_to_f16_kernel<GGML_TYPE_Q4_K  ><<<grid, 256, 0, stream>>>(dst, codes, n, k); break;
        case GGML_TYPE_Q5_K:   qskinny_to_f16_kernel<GGML_TYPE_Q5_K  ><<<grid, 256, 0, stream>>>(dst, codes, n, k); break;
        case GGML_TYPE_Q6_K:   qskinny_to_f16_kernel<GGML_TYPE_Q6_K  ><<<grid, 256, 0, stream>>>(dst, codes, n, k); break;
        case GGML_TYPE_IQ4_XS:   qskinny_to_f16_kernel<GGML_TYPE_IQ4_XS><<<grid, 256, 0, stream>>>(dst, codes, n, k); break;
        case GGML_TYPE_IQ4_NL:   qskinny_to_f16_kernel<GGML_TYPE_IQ4_NL><<<grid, 256, 0, stream>>>(dst, codes, n, k); break;
        case GGML_TYPE_IQ3_S:   qskinny_to_f16_kernel<GGML_TYPE_IQ3_S  ><<<grid, 256, 0, stream>>>(dst, codes, n, k); break;
        case GGML_TYPE_IQ3_XXS: qskinny_to_f16_kernel<GGML_TYPE_IQ3_XXS><<<grid, 256, 0, stream>>>(dst, codes, n, k); break;
        case GGML_TYPE_IQ2_XS:  qskinny_to_f16_kernel<GGML_TYPE_IQ2_XS ><<<grid, 256, 0, stream>>>(dst, codes, n, k); break;
        case GGML_TYPE_IQ2_S:   qskinny_to_f16_kernel<GGML_TYPE_IQ2_S  ><<<grid, 256, 0, stream>>>(dst, codes, n, k); break;
        default: GGML_ABORT("iq skinny to_f16: unknown type");
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_q4k_skinny_repack_side(ggml_type type, const void * src, uint8_t * codes,
                                      uint8_t * meta, int n, int k, cudaStream_t stream) {
    // the source tensor stays intact, so no staging copy is needed (unlike the in-place path)
    switch (type) {
        case GGML_TYPE_Q2_K:   q4k_skinny_repack_tiles<GGML_TYPE_Q2_K  >(src, codes, meta, n, k, 0, n / 32, stream); break;
        case GGML_TYPE_Q3_K:   q4k_skinny_repack_tiles<GGML_TYPE_Q3_K  >(src, codes, meta, n, k, 0, n / 32, stream); break;
        case GGML_TYPE_Q4_K:   q4k_skinny_repack_tiles<GGML_TYPE_Q4_K  >(src, codes, meta, n, k, 0, n / 32, stream); break;
        case GGML_TYPE_Q5_K:   q4k_skinny_repack_tiles<GGML_TYPE_Q5_K  >(src, codes, meta, n, k, 0, n / 32, stream); break;
        case GGML_TYPE_Q6_K:   q4k_skinny_repack_tiles<GGML_TYPE_Q6_K  >(src, codes, meta, n, k, 0, n / 32, stream); break;
        case GGML_TYPE_IQ4_XS:   q4k_skinny_repack_tiles<GGML_TYPE_IQ4_XS>(src, codes, meta, n, k, 0, n / 32, stream); break;
        case GGML_TYPE_IQ4_NL:   q4k_skinny_repack_tiles<GGML_TYPE_IQ4_NL>(src, codes, meta, n, k, 0, n / 32, stream); break;
        case GGML_TYPE_IQ3_S:   q4k_skinny_repack_tiles<GGML_TYPE_IQ3_S  >(src, codes, meta, n, k, 0, n / 32, stream); break;
        case GGML_TYPE_IQ3_XXS: q4k_skinny_repack_tiles<GGML_TYPE_IQ3_XXS>(src, codes, meta, n, k, 0, n / 32, stream); break;
        case GGML_TYPE_IQ2_XS:  q4k_skinny_repack_tiles<GGML_TYPE_IQ2_XS >(src, codes, meta, n, k, 0, n / 32, stream); break;
        case GGML_TYPE_IQ2_S:   q4k_skinny_repack_tiles<GGML_TYPE_IQ2_S  >(src, codes, meta, n, k, 0, n / 32, stream); break;
        default: GGML_ABORT("iq skinny repack: unknown type");
    }
    CUDA_CHECK(cudaGetLastError());
}


#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_q4k_skinny_repack_inplace(ggml_backend_cuda_context & ctx, ggml_tensor * t) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(t);
}

void ggml_cuda_q4k_skinny_prepass(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(cgraph);
}

bool ggml_cuda_q4k_skinny_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                  const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    return false;
}

void ggml_cuda_q4k_skinny_to_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(dst);
    GGML_UNUSED(stream);
}

bool ggml_cuda_q4k_skinny_get_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                   const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
