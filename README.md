# v100fast-qwen38-27b

**Make Qwen3.8-27B + DSpine fly on a single NVIDIA Tesla V100 (`sm_70`, 32 GB) — `128–130 tok/s` chat, `175–182 tok/s` code, `174–206 tok/s` at 12K agent context, `105–133 tok/s` at 50K context, `92–94 tok/s` (`110.9` peak) at 100K context, and `1,102 / 976 / 826 / 759 tok/s` prompt prefill at `12K / 50K / 100K / 128K`, with zero quality loss (`UD-Q4_K_XL` + `q8_0` target KV cache).**

All numbers below were measured on a single **Tesla V100-PCIE-32GB** capped at **220 W** (`ECC ON`, `0` uncorrected ECC) with `Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf` + `Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf` (`-c 229376 -ub 2048 -ctk q8_0 -ctv q8_0 --spec-draft-type-k f16 --spec-draft-type-v f16 --backend-sampling`).

---

## Measured Benchmarks (Single V100-PCIE-32GB @ 220 W)

| Workload | Baseline (`dspine` upstream) | Stage 6 (`s6_v2`) | Stage 7 (`s7v2`, cooled) | Stage 8 (`s8i`, hot 56–59 °C) | **Stage 9 (`s9_final`, hot 56–60 °C)** | Speedup vs Baseline |
|---|---:|---:|---:|---:|---:|---:|
| **Chat (short)** | 57.9 tok/s | 113.9 tok/s | 115.5 tok/s | 119.9 tok/s | **129.9 tok/s** *(+8.3% vs s8)* | **2.24× (+124.4%)** |
| **Code (short)** | 87.3 tok/s | 168.8 tok/s | 171.4 tok/s | 179.6 tok/s | **174.5 tok/s** *(182.3 on C++ LRU)* | **2.00× (+99.9%)** |
| **Agent (12K ctx, tools)** | 83.9 tok/s | 179.9 tok/s | 184.3 tok/s | 198.0 tok/s *(203.6 `s8g`)* | **206.1 tok/s** *(+4.1% vs s8)* | **2.46× (+145.6%)** |
| **All 12K Suite Avg** | — | 147.9 tok/s | 150.3 tok/s | 157.3 tok/s | **160.8 tok/s** *(+2.2% vs s8)* | — |
| **Agent (50K ctx, tools)** | 55.1 tok/s | 86.5 *(pk 91.6)* | 89.0 *(pk 95.2)* | 97.9 *(pk 104.7)* | **104.9 tok/s** *(pk **115.7**, +7.2%)* | **1.90× (+90.4%)** |
| **Agent (100K ctx, tools)** | 42.2 tok/s | 78.2 *(pk 92.4)* | 79.9 *(pk 94.8)* | 85.9 *(pk 103.1)* | **91.9 tok/s** *(pk **105.3**, +7.0%)* | **2.18× (+117.8%)** |
| **Deep-Think (`4K` / `64K` / `128K`)** | — | — | — | 62.6 / 55.4 / 43.4 tok/s | **82.6 / 63.5–70.9 / 52.7 tok/s** | **+32% / +15–28% / +21.5% vs s8** |
| **64K `nsys` Decode** | — | — | 59.8 tok/s | 63.9 tok/s | **75.9 tok/s** *(+18.8% vs s8)* | — |
| **64K Verify Step (GPU / wall)** | ~125 / ~130 ms | 60.98 / 66.03 ms *(92%)* | 60.61 / 64.73 ms *(94%)* | 57.54 / 61.63 ms *(93%)* | **47.96 / 49.21 ms** *(**97.5%** busy; 47.1 ms w/ Wave 5)* | **2.76× (−63.8% wall)** |
| **Prompt Read (12K / 50K / 100K)** | 898 / 709 / — tok/s | 1,080 / 874 / 628 tok/s | 1,083 / 875 / 628 tok/s | 1,071 / 872 / 627 tok/s | **1,102 / 976 / 826 tok/s** *(Stage 9.1 cold full-prompt)* | **+22.7% / +37.6% / +31.8% vs s8** |
| **Prefill Iso-Thermal (`12K / 64K / 100K / 128K`)** | — | — | — | 1,098 / 862 / 743 / 671 tok/s *(Stage 9 base)* | **1,134 / 944 / 830 / 759 tok/s** *(−22.7 s @ 128K)* | **+3.3% / +9.6% / +11.8% / +13.1%** |
| **155K Live Coding Session** | — | 55.2 – 66.6 tok/s | — | — | — | — |
| **196K Stress Test** | — | 550.3 pp / 37.5 tg (`31,058 MiB`) | — | — | — | — |
| **Verbatim Code Copy** *(opt-in `LLAMA_SM70_CTX_LOOKUP=1`)* | — | 311.9 tok/s *(DSpine only)* | 329.2 tok/s *(bit-exact)* | — | — | **+5.5% over DSpine** |
| **Deterministic `temp=0` Parity** | 61.0 tok/s | 115.55 tok/s | head=15/15, 4/4, 4/4 | head=15/15, 4/4, 4/4 | **100% SHA parity (`easy_code`, `4K`, `64K`, `128K`)** | **100% parity** |

*(Note: Stage 9 `s9_final` ran back-to-back without cooldown sleeps at `56–60 °C`, `220 W`, `0` uncorrected ECC errors. Every Stage 9 optimization is 100% strictly lossless with respect to `UD-Q4_K_XL` target weights and `q8_0` KV cache and can be toggled off independently via env vars.)*

### 64K `nsys` Kernel Verification (`s7e2` → `s8i` → `s9`, 220 W)

| Kernel / Family | Stage 7 (`s7e2`) | Stage 8 (`s8i`) | **Stage 9 (`s9`, `m=8` fast path)** | Delta (`s8 → s9`) |
|---|---:|---:|---:|---:|
| **Total Wall / GPU per check** | 64.73 / 60.61 ms | 61.63 / 57.54 ms | **49.21 / 47.96 ms** *(47.13 ms w/ Wave 5)* | **−14.50 ms wall (−23.5%)** |
| **GPU Idle Gap / check** (`ASYNC_SET=1`) | 4.12 ms (`94.0%` busy) | 4.09 ms (`93.4%` busy) | **1.25 ms (`97.5%` busy)** | **−69% idle** |
| **`sm70_fa96_decode_kernel`** (`FA96_LDG=1`) | 11.26 ms/chk | 11.18 ms/chk (`698 us/l`) | **7.96 ms/chk (`498 us/l`)** | **−28.8%** |
| **`q4k_skinny` (`nt` + `gated_nt` + `mixed`)** (`3CTA + WAVEFIX + N5120_2CTA`) | ~35.0 ms/chk | ~34.6 ms/chk | **30.99 ms/chk** *(~29.8 ms w/ Wave 5)* | **−13.9%** |
| **DSpine `BF16` GEMMs (`BF16_CACHE=1` TC bypass)** | ~1.08 ms/chk | ~0.30 ms/chk *(convert)* | **~0.18 ms/chk** *(cached FP16 TC, `m>=4`)* | **−40% / zero `mul_mat_vec_f`** |
| **GDN (`replay` + `save` + `ssm_conv` + `q8_small_dual`)** | 3.55 ms/chk | 2.77 ms/chk | **2.28 ms/chk** | **−17.7%** |

---

## Quick Start (One-Line `curl` Install)

Requires Linux with an NVIDIA Tesla V100 (`sm_70`), CUDA Toolkit (`nvcc` 12.x), `cmake`, `g++`, `git`, and `curl`.

```bash
curl -fsSL https://raw.githubusercontent.com/hac01/v100fast-qwen38-27b/main/install.sh | bash
```

What this does automatically:
1. Clones pinned [`Ajay9o9/llama.cpp`](https://github.com/Ajay9o9/llama.cpp) (`4703b7d`, DSpine base).
2. Applies `patches/v100fast-stage9.patch` and copies the custom Volta `sm_70` CUDA kernels from `overlay/`.
3. Compiles `llama-server` and `test-backend-ops` specifically for `sm_70` (`CMAKE_CUDA_ARCHITECTURES=70`).
4. Downloads `Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf` (17.4 GB) and `Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf` (1.2 GB) with resumable `curl`.
5. Writes your ready-to-run config (`.v100fast.env`) sized automatically for your GPU VRAM and system RAM.

### Common Installer Options

```bash
# Use official Unsloth Qwen3.8-27B UD-Q4_K_XL instead of Huihui:
curl -fsSL https://raw.githubusercontent.com/hac01/v100fast-qwen38-27b/main/install.sh | bash -s -- --model qwen

# Build, run all 12 CUDA test-backend-ops verification gates, and start immediately:
curl -fsSL https://raw.githubusercontent.com/hac01/v100fast-qwen38-27b/main/install.sh | bash -s -- --verify --start

# Skip model download if you already have GGUF files locally:
curl -fsSL https://raw.githubusercontent.com/hac01/v100fast-qwen38-27b/main/install.sh | bash -s -- --model none --drafter none
```

### Start & Connect

```bash
# Foreground
~/v100fast-qwen38-27b/scripts/serve.sh

# Background (waits for /health)
~/v100fast-qwen38-27b/scripts/serve.sh --background

# Expose on LAN with an API key (keep 127.0.0.1 unless you need LAN access)
~/v100fast-qwen38-27b/scripts/serve.sh --host 0.0.0.0 --port 8080 --api-key-file ~/api-key
```

OpenAI-compatible endpoint: `http://127.0.0.1:8080/v1` (model alias `qwen3.8-27b`). Works out of the box with OpenCode, Hermes, Cline, Continue, and standard OpenAI SDKs — with full GPU `--backend-sampling` active even during tool calling and JSON-schema grammar turns.

---

## Hardware & Memory Footprint

| Component | Footprint (V100-32GB, `-c 229376`) |
|---|---:|
| Target model (`UD-Q4_K_XL`) | ~17.4 GB |
| DSpine-b16 drafter (`Q4_K_M`) | ~1.2 GB |
| `q8_0` target KV cache (`229,376` ctx, 16 full-attn layers) + `f16` 5-layer DSpine KV + GDN states + buffers | ~12.6 GB |
| **Total Peak VRAM** (`-c 229376` w/ `f16` draft KV) | **31,264 / 32,768 MiB** *(1.50 GiB free)* |
| Host RAM prompt cache (`--cache-ram 15360`) | Up to 15 GB *(auto-scaled if host has < 20 GB RAM)* |

> **Note for passively cooled V100-PCIE cards**: Keeping the power cap at **220 W** (`sudo nvidia-smi -pm 1 -pl 220`) prevents thermal throttling while preserving ~95%+ of peak throughput. `scripts/serve.sh` applies this automatically if passwordless `sudo nvidia-smi` is available (override with `--power-limit 0` to skip).

---

## What Makes It Fast on Volta (`sm_70`)?

Every optimization is bit-exact or passes full `test-backend-ops` tolerances and `temp=0` parity. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full SASS, `nsys`, and design deep dive.

1. **`sm70_fa96` v6 Direct `q8_0` & `k8v4` (`q8_0/q4_0`) Decode FlashAttention (`LLAMA_SM70_FA96=1`)**
   - Single-pass `Br=96, Bc=32` 12-warp Volta HMMA kernel reading `q8_0` (and async `v5d` `q4_0` V for `-ctk q8_0 -ctv q4_0` ~3 GiB VRAM savings at 300K+ ctx) directly with spill-free softmax, double-buffered `k_s`/`v_s` smem (`4 → 3` `__syncthreads()`/tile), pre-scaled `Q` (`scale * Log2E`), 2-pointer KV stepping, and branchless 4B-aligned `LDG.E.32` loads (**zero `SHFL`**, avoiding Volta MIO-pipe contention).
   - Cuts 16K / 64K / 128K `q8_0` attention to **`241.7 / 703.5 / 1,354.9 us`** (**36.7–38.3 TFLOPS**; vs `1,768 / 3,425 us` upstream).
2. **`IQ4_XS` / `IQ4_NL` 4-Register `__byte_perm` LUT + 2 MiB Ping-Pong FP16 Activation Cache (`q4k-skinny.cu`, `q8-skinny.cu`, `norm.cu`, `ggml-cuda.cu`)**
   - Eliminates `64× CALL` + `70× SHFL` per tile via a bit-exact 4-register `PRMT` LUT (`65,536/65,536` `errs=0`), adds `Q5_K`/`Q6_K` SoA coalescing, cross-tile & Gate+Up cross-projection prefetch, mixed-quant FFN fusion (`14` pairs), and `RowTiles=1` FP16 paths for `m = 2..8`.
   - Stage 8D + 8E-1 pre-allocates a 2 MiB ping-pong `f16_act_buf`: both fused `ADD + RMS_NORM + MUL` and `q4k_skinny_gated` (`silu(gate)*up`) write FP16 directly to the alternate slot for the next projection (`Q/K/V`, `ssm_in/z`, `down_proj`), cutting `convert<f32,f16>` from **`1.49 → 0.297 ms/check`** (**−80%**, −11,017 launches vs Stage 7).
3. **Fused Dual `float4` Tiled `q8-small.cu` (`LLAMA_SM70_Q8_SMALL_DUAL=1`, Stage 7E-v2 + 8E-2)**
   - Replaces the 2-kernel `q8_small_partial`/`reduce` (`1.41 ms/check`, 192 launches/check across 96 GDN `ssm_beta`/`ssm_alpha` calls) with `q8_small_dual_tiled_kernel<3,4,8>`, computing both projections from a single `float4` activation load across all 48 GDN layers → **`0.617 ms/check`** (**−56% vs Stage 6, −30% vs Stage 7**).
4. **Fused Direct `ssm_conv_direct_d4_f32_kernel` (`LLAMA_SM70_DIRECT_SSM_CONV=1`, Stage 8E-3)**
   - Fuses `concat_dim0_src1_transposed` + `k_set_rows_chan_coalesced` + `ssm_conv_f32` + `silu` into one 80-CTA coalesced register-window kernel across 47/48 GDN layers, eliminating **109 MB/check** of intermediate `conv_input` HBM2 traffic and 7,144 launches (**`0.927 → 0.415 ms/check`, −55%**).
5. **GDN Direct-State + Exact FP32 Rollback Replay (`LLAMA_SM70_GDN_REPLAY=1`) & Chunked Prefill (`LLAMA_SM70_GDN_CHUNK=1`)**
   - Replaying `0 .. n_accepted - 1` in registers from `slot 1` (`s_start`) + a `24.9 MB` `(k, v, exp(g), beta)` ring buffer cuts GDN verification traffic **5.6×** (`4.18 → 1.77 ms/step`, `REG: 82, STACK: 0`) with 100% bit-exact FP32 states and automatic `--cache-ram` checkpoint materialization hooks.
6. **2-Stage Parallel `float4` `argmax` + GPU Grammar-Rejection `--backend-sampling` (`LLAMA_SM70_LAZY_BS=1`, Stage 8C-v2)**
   - Replaces 1-CTA `argmax_f32` (`98.1 us × 16 = 1.57 ms/step`) with an 80-CTA `float4` 2-stage reduction (`~5 us`).
   - Keeps GPU `--backend-sampling` active **through tool calls and grammar-constrained generation** (zero `backend sampling is not compatible` disables, zero `9.7–15.9 MB` logits D2H, zero discarded draft tokens). Each GPU-sampled token is checked against the active grammar on CPU (~1 µs); only on the rare (~0.1%) violation does `llama_get_logits_ith` lazily fetch that single row (`0.6 MB`) from GPU `t_logits` and resample on CPU.
7. **Multi-Slot Shape-Keyed CUDA Graph Cache (`LLAMA_SM70_MULTI_GRAPH=1`, Stage 8A) + Adaptive KV Padding (Stage 7D-1)**
   - 8-slot LRU CUDA graph cache keyed by topology + batch size (without `n_kv`, preserving fast `cudaGraphExecUpdate`), cutting un-graphed `cudaLaunchKernel` overhead **−50%**.
   - Adaptive KV cell padding (`256 / 1024 / 4096` at `> 4K / > 16K`) cuts mid-decode `52 ms` graph-resize stalls by 16× at long context.
8. **Volta `BF16 → FP16` Tensor Core GEMM (`LLAMA_SM70_BF16_AS_F16=1`, Stage 8B) + DSpine `f16` Draft KV**
   - Routes DSpine `BF16` projections through Volta FP16 Tensor Cores (`CUBLAS_COMPUTE_32F` FP32 accumulation, `239/239` `MUL_MAT bf16` PASS), eliminating all 3 FP32 SGEMMs (`~0.70 ms/chk`).
   - Setting `--spec-draft-type-k f16 --spec-draft-type-v f16` on the 5-layer drafter eliminates `dequantize_block_q8_0_f16` (`0.378 → 0.000 ms/chk`) for only `+78 MiB` VRAM.
9. **Hoisted `s_slot_rev` + Warp `Q8_0`/`Q4_0` KV `set-rows` + Draft Ubatch Cap (`<= 512`)**
   - Removes 141 redundant `arange+scale+cast` launches/step, accelerates KV cache writes (`32` lanes/block), and caps draft scratchpad allocation (`-ub 2048` on target while saving ~1.5 GB VRAM).
10. **Stage 9 — Dynamic `7 ↔ 15` DSpine Block Hysteresis + 3-CTA/SM & Wave-Matched Skinny GEMMs + Async H2D + Cached `BF16` TC + `__ldg` `FA96`**
    - **`LLAMA_SM70_DYN_BLOCK=1`** (`common/speculative.cpp`): Dynamically sizes DSpine's block before its forward pass (`n_draft = 7` / `m = 8` on reasoning & long-context steps, promoting to `15` / `m = 16` when all `7/7` draft tokens accept and demoting when `< 6` accept). Cuts both DSpine and target work in half on hard turns while preserving `206 tok/s` agent/tool-call bursts. Both `m = 8` and `m = 16` remain simultaneously live in Stage 8A's multi-slot CUDA graph cache with zero re-captures.
    - **`LLAMA_SM70_SKINNY_3CTA=1` + `LLAMA_SM70_SKINNY_GNT_3CTA=1`** (`q4k-skinny.cu`): Tightens `__launch_bounds__(256, 3)` on `m <= 8` `Q4_K`/`Q5_K`/`Q6_K`/`IQ4_XS` projections (`lm_head` + all 64 `ffn_gate/up` gated & mixed-quant layers), dropping `Q5_K`/`Q6_K` from `86 → 76–79` regs with **`STACK:0`** to cross Volta's `80`-reg cliff (**3 CTAs/SM = `24` warps/SM, `+50%` occupancy**).
    - **`LLAMA_SM70_SKINNY_WAVEFIX=1` + `LLAMA_SM70_SKINNY_N5120_2CTA=1`** (`q4k-skinny.cu`): Eliminates the `n = 6144` (`ssm_in`) 96-CTA tail wave (`NTiles=1, SplitK=8, 3CTA` → `REG:56–60, STACK:0`, 192 CTAs in **1 single wave**), doubles active SMs on `n = 1024` (`K/V`, `32` vs `16` CTAs), and doubles occupancy on `n = 5120` (`ffn_down`/`attn_out`/`ssm_out`, `NTiles=1, SplitK=16, MinBlocks=2` → `REG:56–60, STACK:0`, **`32` warps/SM**).
    - **`LLAMA_SM70_BF16_CACHE=1`** (`ggml-cuda.cu`): Permanently caches immutable `BF16 → F16` weights once at load and bypasses scalar `mul_mat_vec_f` for `BF16` weights at `m >= 4` so DSpine's `25` `BF16` GEMMs always hit Volta FP16 Tensor Cores (`228/228` `MUL_MAT bf16` PASS).
    - **`LLAMA_SM70_ASYNC_SET=1`** (`ggml-cuda.cu`): 4 MiB pinned-ring async H2D `set_tensor` ordered on the active compute stream (with clean stream-destroy flush), cutting GPU idle gaps from **`4.09 → 1.25 ms/check`** (`97.5%` GPU busy).
    - **`LLAMA_SM70_FA96_LDG=1`** (`fattn-sm70-q8-96.cu`): L1-coalesced `__ldg` loads for 34-byte `block_q8_0` KV tiles (`REG:165, STACK:0`, **`−0.60 ms/chk`** at 64K).
11. **Stage 9.1 — Lossless Volta Prefill Engine (`+9.6%` @ 64K, `+11.8%` @ 100K, `+13.1%` / `−22.7 s` @ 128K + Causal-Mask Bug Fix)**
    - **`LLAMA_SM70_PREFILL_BC128=2`** (`fattn-sm70-prefill-96.cu/.cuh`, `fattn-sm70-d256.cu`): New **`sm70_prefill_br128_bc64_kernel` v4** (`Br=128, Bc=64, D=256`, `REG:246, STACK:0`, `97,536 B` SMEM, fallback to `Br=64, Bc=128` for `256 <= q_len < 512`). Constant-stride padded SMEM (`0` load bank conflicts vs `31.7M`), `2×` K/V reuse (`−45%` SMEM load wavefronts), and dead-region double-buffering (`16 → 11` barriers/tile) cut `ncu` kernel duration **`7.00 → 5.55 ms` (−20.7%)** and lift `HMMA` active cycles **`37.5% → 50.8%`** (`mio_throttle`: `24.3% → 7.4%`), while skipping `216 MiB` of unused `Qs/Os/SplitKV3` scratch. **Also fixes an upstream `lo - q_pad` causal-mask shift bug** in `sm70_d256_splitd_dense_kernel` whenever a final prefill chunk is not a multiple of 64 tokens (restoring exact parity with stock `llama.cpp` `FLASH_ATTN_EXT`).
    - **`LLAMA_SM70_GDN_H48=1`** (`gdn-chunk-sm70.cu`): Compile-time `<48, 16>` TileLang GDN chunk specialization (**−11.1%**, `1,428 → 1,269 us`).
    - **`LLAMA_SM70_DIRECT_SSM_PREFILL=1`** (`ssm-conv.cu`): Extends direct `ssm_conv` to `n_t <= 8192` (in-place `BLOCK_C=32` `320`-CTA coalesced + 2D-tiled out-of-place).
    - **`LLAMA_SM70_PF_ACT_CACHE=1` + `LLAMA_SM70_SWIGLU_VEC4=1`** (`common.cuh`, `ggml-cuda.cu`, `norm.cu`, `unary.cu`): Lazy generation-guarded prefill FP16 activation cache + `float4` SwiGLU direct FP16 output, cutting `convert_unary_cont_vec4` **−87%** (`49.5 → 6.5 ms/chunk`).
    - **`LLAMA_SM70_FAST_D2H=1`** (`batch-copy.cu`): Bypasses `cudaHostRegister` (`60–127 ms` page-pinning stall/checkpoint) on `>16 MiB` D2H copies while keeping `--ctx-checkpoints 8` 100% intact.

---

## Environment Variables & Toggles

| Variable | Default | Purpose |
|---|---|---|
| `LLAMA_SM70_FA96` | `0` *(set `1` in `serve.sh`)* | Volta `sm70_fa96` v6 direct `q8_0` / `k8v4` FlashAttention decode |
| `LLAMA_SM70_FA96_LDG` | `0` *(set `1` in `serve.sh`)* | L1-coalesced `__ldg` `Q8_0` KV loads in `sm70_fa96` (Stage 9) |
| `LLAMA_SM70_GDN_CHUNK` | `0` *(set `1` in `serve.sh`)* | Volta HMMA chunked GDN prefill |
| `LLAMA_SM70_GDN_REPLAY` | `0` *(set `1` in `serve.sh`)* | GDN direct-state read + exact FP32 rollback replay |
| `LLAMA_SM70_BF16_CACHE` | `0` *(set `1` in `serve.sh`)* | Immutable `BF16→F16` weight cache + TC bypass of `mul_mat_vec_f` for `m>=4` (Stage 9) |
| `LLAMA_SM70_ASYNC_SET` | `0` *(set `1` in `serve.sh`)* | 4 MiB pinned-ring async H2D `set_tensor` on active compute stream (Stage 9) |
| `LLAMA_SM70_SKINNY_3CTA` | `0` *(set `1` in `serve.sh`)* | 3-CTA/SM (`24` warps/SM, `STACK:0`) `q4k_skinny_nt` for `m <= 8` (Stage 9) |
| `LLAMA_SM70_SKINNY_GNT_3CTA` | `0` *(set `1` in `serve.sh`)* | 3-CTA/SM (`24` warps/SM, `STACK:0`) across all 64 `ffn_gate/up` layers for `m <= 8` (Stage 9) |
| `LLAMA_SM70_SKINNY_WAVEFIX` | `0` *(set `1` in `serve.sh`)* | Single-wave 4-CTA/SM `n=6144` (`ssm_in`) + 2× SMs on `n=1024` (`K/V`) (Stage 9) |
| `LLAMA_SM70_SKINNY_N5120_2CTA` | `0` *(set `1` in `serve.sh`)* | 2-CTA/SM (`32` warps/SM, `STACK:0`) on `n=5120` (`ffn_down`/`attn_out`/`ssm_out`) (Stage 9) |
| `LLAMA_SM70_DYN_BLOCK` | `0` *(set `1` in `serve.sh`)* | Lossless dynamic `7 ↔ 15` DSpine block-size hysteresis (`DYN_UP=7, DYN_DOWN=6, DYN_STREAK=1`) (Stage 9) |
| `LLAMA_SM70_PREFILL_BC128` | `0` *(set `2` in `serve.sh`)* | Volta `Br=128, Bc=64, D=256` padded-SMEM double-buffered prefill FA + upstream causal-mask fix (Stage 9.1) |
| `LLAMA_SM70_GDN_H48` | `0` *(set `1` in `serve.sh`)* | Compile-time `<48, 16>` TileLang GDN chunk specialization (Stage 9.1) |
| `LLAMA_SM70_DIRECT_SSM_PREFILL` | `0` *(set `1` in `serve.sh`)* | Direct `ssm_conv` prefill up to `n_t <= 8192` (in-place `BLOCK_C=32` + 2D-tiled out-of-place, Stage 9.1) |
| `LLAMA_SM70_PF_ACT_CACHE` | `0` *(set `1` in `serve.sh`)* | Lazy generation-guarded prefill FP16 activation cache (`−87%` `convert_unary_cont_vec4`, Stage 9.1) |
| `LLAMA_SM70_SWIGLU_VEC4` | `0` *(set `1` in `serve.sh`)* | `float4` SwiGLU writing FP16 directly to `f16_act_buf` (Stage 9.1) |
| `LLAMA_SM70_FAST_D2H` | `0` *(set `1` in `serve.sh`)* | Bypass `>16 MiB` `cudaHostRegister` page-pinning stalls on `--ctx-checkpoints` D2H copies (Stage 9.1) |
| `LLAMA_SM70_MULTI_GRAPH` | `1` (ON) | 8-slot shape-keyed CUDA graph cache (Stage 8A) |
| `LLAMA_SM70_BF16_AS_F16` | `1` (ON) | Route Volta `BF16` GEMM through FP16 Tensor Cores w/ FP32 acc (Stage 8B) |
| `LLAMA_SM70_LAZY_BS` | `1` (ON) | GPU grammar-rejection `--backend-sampling` w/ on-demand 1-row fetch (Stage 8C-v2) |
| `LLAMA_SM70_FUSED_ADD_RMS` | `1` (ON) | Fused dual-output `ADD + RMS_NORM + MUL` → `f16_act_buf` (Stage 8D) |
| `LLAMA_SM70_GATED_F16_OUT` | `1` (ON) | Ping-pong `q4k_skinny_gated` direct FP16 output → `down_proj` (Stage 8E-1) |
| `LLAMA_SM70_Q8_SMALL_DUAL` | `1` (ON) | Fused dual `q8_small` (`ssm_beta + ssm_alpha`, Stage 8E-2) |
| `LLAMA_SM70_DIRECT_SSM_CONV` | `1` (ON) | Fused direct `ssm_conv_direct_d4_f32` + `silu` + conv state store (Stage 8E-3) |
| `LLAMA_SM70_CTX_LOOKUP` | `0` (OFF) | Optional lossless 16-token prompt-suffix lookup (`+5.5%` on verbatim code copy) |

Set any Stage 8 or Stage 9 variable to `0` to isolate or disable that optimization without rebuilding (with all Stage 9 flags `0`, execution is byte-identical to Stage 8 `s8i`).

---

## Verification & Benchmarking

```bash
# Run all 12 CUDA test-backend-ops verification gates
./scripts/verify.sh

# Run reproducible speed + temp=0 SHA benchmark
python3 ./scripts/bench.py --url http://127.0.0.1:8080 --long-ctx 12000 50000 100000
```

Verified passing on Tesla V100-PCIE-32GB (CUDA 12.8):
- `MUL_MAT` (`q4_K|q5_K|q6_K|iq4_xs|iq4_nl`): **479/479 PASS** | `MUL_MAT` (`q8_0`): **258/258 PASS** | `MUL_MAT` (`bf16`): **239/239 PASS**
- `RMS_NORM`: **121/121 PASS** | `ADD`: **167/167 PASS** | `MUL`: **PASS**
- `GET_ROWS`: **228/228 PASS**
- `FLASH_ATTN_EXT` `hsk=256` (`FA96=1`, incl. `Q8/Q4` & `Q4/Q4`): **249/249 PASS** on both `TEST_FA_CAUSAL=1` and `TEST_FA_CAUSAL=2`
- `GATED_DELTA_NET` (`REPLAY=1, CHUNK=0`): **63/63 PASS**
- `ARGMAX`: **7/7 PASS** | `SET_ROWS`: **191/191 PASS** | `CONCAT`: **221/221 PASS** | `GLU` + `SSM_CONV` (`Stage 9.1`): **75/75 PASS**
- Ground-truth prefill parity vs stock `llama.cpp` `FLASH_ATTN_EXT` (`LLAMA_SM70_D256=0`): `exact_text_match = True`, top-1 logprob diff `0.001–0.005` (`P12K`, `P_EXACT` `q_len=263`), exact `94` tok / `74/181` draft match on 12K agent turn 0; `0.000000` logprob diff between `PREFILL_BC128=2` alone and all 6 Stage 9.1 flags ON.

---

## Repository Layout

```text
v100fast-qwen38-27b/
├── install.sh                    # One-line curl installer (build + download + config)
├── patches/
│   └── v100fast-stage9.patch     # Clean git diff against pinned Ajay9o9/llama.cpp@4703b7d
├── overlay/
│   └── ggml/src/ggml-cuda/       # Custom Volta sm_70 kernels + vendored CuTe/FA2 headers
├── scripts/
│   ├── build.sh                  # Standalone clone + patch + sm_70 CMake build
│   ├── serve.sh                  # Production server launcher with Stage 9 flags
│   ├── verify.sh                 # Full CUDA test-backend-ops gate
│   └── bench.py                  # Reproducible short & long-context benchmark
├── docs/
│   └── ARCHITECTURE.md           # Technical deep dive (SASS, nsys, wins & reverted dead ends)
├── THIRD_PARTY.md                # Full third-party attributions & licenses
└── LICENSE                       # MIT License
```

---

## Credits & Licenses

This repository is MIT-licensed ([LICENSE](LICENSE)) and builds on the work of the open-source ecosystem — see [THIRD_PARTY.md](THIRD_PARTY.md) for complete attributions and licenses (`llama.cpp`, `Ajay9o9/llama.cpp` DSpine support, `v100-skinny`, `1CatAI`, `NVIDIA CUTLASS/CuTe`, `flash-attention-v100`, `Qwen`, `Tencent DSpine`, `Unsloth`, and `huihui-ai`).
