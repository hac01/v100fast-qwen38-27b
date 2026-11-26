# Architecture & Volta (`sm_70`) Optimization Deep Dive

This document explains why **Qwen3.8-27B + DSpine** runs **~2.0–2.5× faster** on a single Tesla V100-PCIE-32GB (`sm_70`, 220 W, `q8_0` target KV + `f16` draft KV) with `v100fast` Stage 9, including the `cuobjdump -sass` and `nsys` findings that drove every optimization — and the experiments we tested and reverted or left opt-in OFF.

---

## 1. Hardware Reality on Tesla V100 (`sm_70`)

- **Compute**: 80 SMs, 64 KB registers/SM, 96 KB shared memory/SM, 6 MB L2 cache, Volta FP16 Tensor Cores (`mma.sync.aligned.m8n8k4` / HMMA, ~95–100 TFLOPS practical @ 220 W; **no BF16 hardware**, no `cp.async`, no Ampere `ldmatrix`).
- **Memory**: 32 GB HBM2, ~750–800 GB/s practical streaming bandwidth @ 220 W.
- **Model Shape (Qwen3.8-27B)**: 64 layers = **48 Gated DeltaNet (GDN)** linear-attention layers (`S_v = 128, H_v = 48, H_k = 16, conv_kernel = 4`) + **16 full-attention** layers (`head_dim = 256, n_kv_heads = 4, n_q_heads = 24, GQA = 6:1`), `n_embd = 5120`, `n_vocab = 248,320`.
- **Speculative Decoding (DSpine-b16)**: Drafts `15` tokens in parallel (`16`-token target verification batch, `m = 16`), turning decode from pure GEMV (`m = 1`) into skinny Tensor Core / HBM-bound verification (`m = 16`).

---

## 2. Where Every Millisecond Went (64K Verify Step Profile)

Measured via `nsys --cuda-graph-trace=node` at 64K context (`n_draft = 15`, 16-token verify):

| Stage | GPU Time / Check | Wall Time / Check | GPU Busy % | Key Bottleneck Removed |
|---|---:|---:|---:|---|
| Baseline (`dspine`) | ~125 ms | ~130 ms | — | Stock `llama.cpp` FA + MMQ + non-fused FFNs |
| Stage 4–5b | 75.51 ms | 83.94 ms | ~89% | `sm70_fa96` v4, `Q5_K`/`Q6_K` SoA, mixed FFN fusion, GDN chunk prefill, draft `ub <= 512` |
| Stage 6 (`s6c_clean`) | 65.71 ms | 72.51 ms | 90.6% | `IQ4_XS` 4-reg PRMT LUT, Gate+Up prefetch, `m=2..8` FP16 skinny, **`sm70_fa96` v5d** |
| Stage 6 (`s6e_replay`) | 62.68 ms | 69.73 ms | 90.0% | **GDN direct-state + exact FP32 rollback replay** (`4.18 → 1.81 ms/check`) |
| Stage 6 Final (`s6_v2`) | 60.98 ms | 66.03 ms | 92.4% | Tiled transpose-concat + warp `Q8_0` & coalesced conv `set-rows` + hoisted `s_slot_rev` + parallel `argmax` + zero-D2H `--backend-sampling` |
| Stage 7 Final (`s7v2`) | 60.61 ms | 64.73 ms | 94.0% | Per-graph FP16 activation cache (`convert` −47%) + `float4` tiled multi-row `q8-small` (−37%) + `sm70_fa96 v6` double-buffered smem + adaptive KV padding + `k8v4` `Q4_0` V |
| Stage 8 Final (`s8i`) | 57.54 ms | 61.63 ms | 93.4% | Multi-slot CUDA graph cache + Volta `BF16→FP16` Tensor Core GEMM + GPU grammar-rejection `--backend-sampling` + fused `ADD+RMS_NORM+MUL→f16` + ping-pong `q4k_skinny_gated→f16` + dual `q8_small` + direct `ssm_conv_direct_d4_f32` + `f16` DSpine draft KV |
| **Stage 9 Final (`s9_final`)** | **47.96 ms** *(~46.0 w/ Wave 5)* | **49.21 ms** *(47.13 w/ Wave 5)* | **97.5%** | **Dynamic `7↔15` DSpine block hysteresis + 3-CTA/SM `q4k-skinny` & gated FFN + `WAVEFIX` (`n=6144` single-wave, `n=1024` 2× SMs) + `N5120_2CTA` (`32` warps/SM) + `BF16_CACHE` TC bypass (`m>=4`) + pinned-ring `ASYNC_SET` + `__ldg` `FA96`** |

---

## 3. Detailed Breakdown of Stage 6 Optimizations

### 3.1 `IQ4_XS` & `IQ4_NL` 4-Register `__byte_perm` LUT (`q4k-skinny.cu`)
- **Problem**: `cuobjdump -sass` showed `IQ4_XS` (`type23`, ~20% of `UD-Q4_K_XL` weights) emitted **`1,280` instructions, `70 SHFL.IDX`, and `64 CALL.REL.NOINC`** per tile (~500 GB/s vs ~710 GB/s on `Q4_K`/`Q6_K`).
- **Fix**: Replaced shuffle/table LUT with a 4-register `__byte_perm` (`PRMT`) int8 LUT (`decode_4nibbles`) encoding `kvalues_iq4nl[16] ^ 0x80` directly into 4 unsigned 32-bit immediates (`0x3F2D1801`, `0x766A5D4F`, `0xA6998D81`, `0xF1D9C5B5`), followed by branchless `PRMT + HADD2(-1152)` conversion to `half2`.
- **Verification**: Exhaustively verified `65,536 / 65,536` 16-bit nibble quads bit-exact (`errs = 0`); enabled cross-tile weight prefetch (`pf_records`) for `IQ4_XS`/`IQ4_NL`; fixed a Stage 5b `GNT_SK8` flag parsing typo (`e[0] == '1'`) and eliminated `STACK:8 → 0` spills in `SplitK=8` gated epilogues.

### 3.2 Gate+Up Cross-Projection Prefetch & `m = 2..8` FP16 Skinny Paths (`q4k-skinny.cu`)
- Added cross-projection weight prefetch in `q4k_skinny_gated_nt_kernel` (`214.3 → 206.2 us`) and `q4k_skinny_gated_nt_mixed_kernel` (`217.0 → 207.4 us`).
- Extended the fast FP16 `RowTiles=1` skinny paths (`nt`, `gated_nt`, `gated_nt_mixed`) down to `m = 2..8` (and `m = 1..8` mixed), accelerating DSpine context-injection batches while keeping `REG: 77–121, STACK: 0`.

### 3.3 `sm70_fa96` `v5d` Direct `q8_0` Decode FlashAttention (`fattn-sm70-q8-96.cu`)
- **Problem**: Stage 5b's `sm70_fa96_decode_kernel<6,2,2,0>` had `2,072` instructions, `405 IMAD` (> `320 HMMA`), `97 BRA`, `89 ISETP`, scalar `LDG.E.U16`, `REG: 168`, `STACK: 32` (`888 us` @ 64K, `1,691 us` @ 128K).
- **Fix (`v5d`)**:
  - Eliminated dynamic-index `p[swap]` stack spills in `softmax_to_half2` (`STACK: 32 → 0`).
  - Switched to 2-pointer incremental KV stepping (slashing `IMAD` address arithmetic).
  - Unconditional branchless `commit_q8_row` using 4B-aligned `LDG.E.32` loads with **zero `__shfl_sync`**.
  - Warp-uniform fast unmasked `QK^T` loop for full tiles.
- **Key Volta Microarchitecture Lesson**: We also tested 7 other variants (`v5c`–`v5i`), including coalesced `LDG.E.64` + `SHFL` scale broadcasts (`v5e`/`v5f`/`v5i`). Every variant using `SHFL` inside `commit_q8_row` was ~35 us/layer **slower** than `v5d` because `SHFL` shares Volta's MIO pipe with `LDS`/`STS`!
- **Result**: `64K q8_0` dropped **`887.6 → 708.3 us` (36.4 TFLOPS)**, `128K q8_0` dropped **`1,691.1 → 1,352.6 us` (38.1 TFLOPS)** — **−20.2%** latency, `249/249` PASS on `CAUSAL=1 & 2`.

### 3.4 GDN Direct-State + Exact FP32 Rollback Replay (`gated_delta_net.cu`, `ggml-cuda.cu`)
- **Problem**: During 16-token speculative verification (`K = 16`), each of Qwen3.8-27B's 48 GDN layers ran:
  1. `k_get_rows_float_vec` copying the `3.15 MB` input state from `s_l[il]` (`302 MB/step`, `0.58 ms/step`).
  2. `gated_delta_net_cuda` writing all **16** FP32 rollback snapshots (`16 × 48 × 3.15 MB = 2,416 MB/step`, `3.60 ms/step`).
- **Fix (`LLAMA_SM70_GDN_REPLAY=1`, opt-in)**:
  - **Coalesced `float4` I/O**: Switched `S_v = 128, !KDA, lanes_per_col = 8` to `4× LDG.E.128` / `4× STG.E.128` with streaming `__stcs` on non-live slots.
  - **Direct-state read + Replay-from-`s0`**: In `ggml-cuda.cu`, `ggml_cuda_gdn_replay_build_plan` matches `GET_ROWS(s_l[il], rs_s_copy) → GATED_DELTA_NET → CPY(s_l[il])`, skips `GET_ROWS` completely, and passes `ssm_base` + device `d_s_copy` directly to `gated_delta_net_replay_cuda` (100% CUDA-graph compatible since `d_s_copy` is updated in-place on device!).
  - On decode (`n_tokens <= 16, K > 1`): writes ONLY **`slot 0` (`s_final`)** and **`slot 1` (`s_start`)**, plus a tiny post-GDN `gdn_save_replay_cuda` kernel saving `(k, v, exp(g), beta)` (`518 KB/layer` = `24.9 MB` total — note **`q` does not affect the recurrent state `S` and doesn't need to be saved or replayed!**).
  - On rollback (`rs_idx > 0`), each CTA loads its independent column of `slot 1` into registers, replays `t = 0 .. prev_n - rs_idx - 1` using the exact same FP32 recurrence (bit-for-bit identical!), stores the new `s_start` to `slot 1`, and proceeds to the new batch.
  - **Checkpoint / Cache Safety**: Buffer hooks on `get_tensor` / `set_tensor` / `clear` / `free_buffer` (and async equivalents) automatically materialize any requested rollback slot `rs_idx >= 1` before host reads (e.g., `--cache-ram` checkpoints) and invalidate replay metadata on overwrites/clears.
- **Result**: `REG: 82, STACK: 0` (2 CTAs/SM), GDN + state gather dropped from **`4.18 → 1.75 ms/check`**, **100% bit-exact** across multi-turn rollbacks and RAM prompt-cache save/restore.

### 3.5 Small-Op Cleanup (`concat.cu`, `set-rows.cu`, `llama-graph.cpp`)
- **`concat_dim0_src1_transposed` (`concat.cu`)**: Stock `concat_non_cont` launched `10,240 × 256 = 2.62M` threads where 92.6% were idle (`ne0 = 19`) and the rest did 40 KB-strided uncoalesced reads (`19.4 us × 48 = 0.93 ms/check`). A 32×32 shared-memory tiled transpose-concat cut this to `6.9 us` (`−0.60 ms/check`, `221/221` PASS).
- **`k_set_rows_q8_0_warp` & `k_set_rows_chan_coalesced` (`set-rows.cu`)**:
  - Stock `k_set_rows_quant` launched only `1–2` CTAs on an 80-SM V100 for 16-token KV writes (`16.6 us × 53 = 0.88 ms/check`). Switching `Q8_0` to 1 warp (32 lanes) per block saturates all SMs with coalesced 128B reads and bit-exact `fmaxf` warp reduction.
  - Stock `k_set_rows` scattered `[3, 16, 10240]` conv rollback windows with `122,880 B` stride between adjacent threads (`11.6 us × 48 = 0.58 ms/check`). Channel-major iteration yields 100% coalesced 128B writes.
- **Hoisted `s_slot_rev` (`llama-graph.h/.cpp`)**: Caches `{K-1, ..., 0}` once per buffer type per graph instead of launching `arange + scale + cast` 48 times (`−141` kernel launches, `~−0.30 ms/check`).

### 3.6 Parallel `argmax` + Zero-D2H `--backend-sampling` (`argmax.cu`, `common/sampling.cpp`, `llama-context.cpp`)
- **Problem**: `nsys` revealed two massive host/PCIe stalls during speculative verification:
  1. `llama-context.cpp` copied `16 × 248,320 × 4 B = 15.89 MB` (`1.28 ms` D2H) of logits per verify step — **even with `--backend-sampling` enabled**, because `t_sampled_logits` was still copied unconditionally!
  2. Default `penalties`/`dry`/`xtc`/`typical_p`/`top_n_sigma` no-op samplers blocked `llama_sampler_chain_backend_init` at index 0, AND stock `argmax_f32` launched **1 CTA** on 248,320 floats (`98.1 us × 16 = 1.57 ms/check`).
- **Fix**:
  - `argmax.cu`: 2-stage parallel `float4` reduction (`80` CTAs → `1` CTA, deterministic lowest-index tie-break) cutting `argmax_f32` `98.1 → ~5 us` (`7/7` PASS).
  - `common/sampling.cpp`: When `--backend-sampling` is enabled, elide mathematically inactive no-op samplers (and pre-greedy filter no-ops on pure `temp=0`), and return `llama_get_sampled_token_ith` before building the 248k CPU candidate vector when `n_probs == 0 && sampled_probs == nullptr`.
  - `llama-context.cpp`: Skip D2H copy of `t_sampled_logits` / `t_candidates` on rows where `t_sampled[i] != nullptr && t_sampled_probs[i] == nullptr`.
- **Result**: D2H logits transfer dropped from **`15.89 MB → 0 bytes`**, lifting `GPU busy` to **92.4%** and jumping Chat/Code/12K by **+7–10%** while tying `temp=0.7` and preserving automatic CPU fallback for active penalties, grammars, JSON schemas, reasoning budgets, and `logprobs`.

### 3.7 Stage 7 Optimizations (`q8-small.cu`, `ggml-cuda.cu`, `fattn-sm70-q8-96.cu`, `llama-kv-cache.cpp`, `set-rows.cu`)
- **Stage 7G — Per-Graph FP16 Activation Cache (`common.cuh`, `ggml-cuda.cu`, `q4k-skinny.cu`, `q8-skinny.cu`)**: Pre-allocates a 1 MiB `f16_act_buf` before CUDA graph capture and caches the FP16 conversion of `src1` across consecutive projections sharing the same input (`Q/K/V`, `ssm_in/ssm_z`, unfused `gate/up`), invalidating on any overlapping write. Cut `convert_unary_cont_vec4<float, __half>` from `1.49 ms/check` (`418` calls) to **`0.79 ms/check`** (`287` calls, **−47% time, −131 launches/check**).
- **Stage 7E-v2 — Single-Pass `float4` Multi-Row Tiled `q8-small.cu`**: Replaced the 2-kernel `q8_small_partial`/`reduce` (`1.41 ms/check`, 192 launches/check across 96 GDN `ssm_alpha`/`ssm_beta` projections) with `q8_small_tiled_kernel<3,4,8>` (`64` CTAs = 1 wave on 80 SMs, 8 lanes per `block_q8_0`, `float4` activation loads reused across 3 weight rows in registers). Cut `q8_small` to **`0.88 ms/check`** (**−37%**, `258/258 MUL_MAT q8_0` PASS).
- **Stage 7F — `sm70_fa96 v6` Double-Buffered Smem + Pre-Scaled `Q` (`fattn-sm70-q8-96.cuh/.cu`)**: Reclaimed 9 KB of unused `kScoresBytes` for double-buffered `k_s`/`v_s` (`96,768 B < 96 KB`), cutting `4 → 3` `__syncthreads()` per KV tile, and pre-multiplied `Q` by `scale * Log2E` (converting `partial_m` back via `* Ln2`). Improved all three lengths (`16K: 245.8 → 241.7 us`, `64K: 708.6 → 703.5 us`, `128K: 1,361.4 → 1,354.9 us`).
- **Stage 7D-1 — Adaptive KV Padding (`src/llama-kv-cache.cpp`)**: `llama_kv_cache::get_n_kv()` scales padding from `256` to `1024` (`> 4K`) and `4096` (`> 16K`), cutting `52 ms` mid-decode `cudaFreeHost + cudaMallocHost + cudaGraphInstantiate` graph-resize freezes by 16× at long context (`0` mid-decode host allocs in 64K `nsys`).
- **Stage 7A — `k8v4` (`-ctk q8_0 -ctv q4_0`) Async `v5d` `Q4_0` + Warp `Q4_0` `set-rows`**: Added `case 7` (`Q8_0` K + `Q4_0` V) with branchless `PRMT+HADD2(0xE408E408)+HMUL2` `commit_q4_row` and bit-exact `k_set_rows_q4_0_warp`. Saves ~3 GiB VRAM (for 300K+ ctx) at `q8_0/q8_0` decode speed (`1,347` vs `1,355 us` @ 128K; `sm70_fa96` is Volta FP16 Tensor Core / 220 W power-bound at ~210 GB/s).

### 3.8 Stage 8 Optimizations (`s8i`, −3.10 ms/check at 64K)
- **Stage 8A — Multi-Slot Shape-Keyed CUDA Graph Cache (`ggml-cuda.cu`, `LLAMA_SM70_MULTI_GRAPH=1`)**: Hashes `(nodes[0], n_nodes, first/last shapes, first MUL_MAT src1 shape, first FA Q batch size)` — intentionally omitting `n_kv` so KV growth still hits fast `cudaGraphExecUpdate` (0.05 ms) — with an 8-slot LRU cache. Cut un-graphed `cudaLaunchKernel` overhead **−50%** (`1.92 → 0.96 ms/chk`).
- **Stage 8B — Volta `BF16 → FP16` Tensor Core GEMM (`ggml-cuda.cu`, `LLAMA_SM70_BF16_AS_F16=1`)**: Routes `GGML_TYPE_BF16` `MUL_MAT` on `sm_70` (`prec != GGML_PREC_F32`) to `GGML_TYPE_F16` (`CUBLAS_COMPUTE_32F` FP32 accumulation + direct F32 output), completely eliminating all 3 DSpine FP32 SGEMMs (`volta_sgemm_64x32`, `gemmSN_TN<float>`, `volta_sgemm_32x32` ≈ `0.70 ms/chk`) and cutting `convert<bf16,*>` by **−28%** (`239/239` `MUL_MAT bf16` PASS).
- **Stage 8C-v2 — GPU Grammar-Rejection `--backend-sampling` (`llama-context.h/.cpp`, `common/sampling.cpp`, `LLAMA_SM70_LAZY_BS=1`)**: Keeps GPU `--backend-sampling` active **through tool calls and grammar-constrained turns** (zero `backend sampling is not compatible` disables, zero `9.7–15.9 MB` logits D2H, zero discarded draft tokens). Each GPU-sampled token is checked against the active grammar on CPU (~1 µs); only on the rare (~0.1%) violation does `llama_get_logits_ith` lazily fetch that single row (`0.6 MB`) from GPU `t_logits` and resample on CPU. Drove the massive agent jumps (`12K: 184.3 → 198–204 tok/s`, `50K: 89.0 → 97.9 tok/s`, `100K: 79.9 → 85.9 tok/s`).
- **Stage 8D — Dual-Output `ADD + RMS_NORM + MUL` → Direct `f16_act_buf` (`norm.cuh/.cu`, `ggml-cuda.cu`, `LLAMA_SM70_FUSED_ADD_RMS=1`)**: Fuses `{ADD, RMS_NORM, MUL}` (preserving the fanout-2 residual stream and supporting exact-pointer `ggml_tallocr` buffer reuse) into `add_rms_norm_mul_f32_f16_kernel`, firing on all 127 target + 10 DSpine residual/norm layers (`10,336×` in 64K `nsys`). Cut `k_bin_bcast<&op_add>` **−70%** and `rms_norm_f32<1024,4>` **−96%**.
- **Stage 8E-1 — Ping-Pong 2 MiB `f16_act_buf` Direct Write from `q4k_skinny_gated` (`q4k-skinny.cu`, `q8-skinny.cu`, `LLAMA_SM70_GATED_F16_OUT=1`)**: Splits `f16_act_buf` into two 1 MiB ping-pong slots so `q4k_skinny_gated_nt_kernel` and `q4k_skinny_gated_nt_mixed_kernel` write FP16 directly to the alternate slot for `down_proj`, eliminating all 64/check `[17408, 16]` F32→F16 conversions (`convert<f32,f16>` dropped `0.792 → 0.297 ms/chk`, **−62%**).
- **Stage 8E-2 — Fused Dual `q8_small` (`q8-small.cuh/.cu`, `qwen35.cpp`, `ggml-cuda.cu`, `LLAMA_SM70_Q8_SMALL_DUAL=1`)**: Fuses `ssm_beta` + `ssm_alpha` sharing `cur` into `q8_small_dual_tiled_kernel<3,4,8>`, halving launches (`96 → 48/chk`) and activation reads across all 48 GDN layers (**−30%**, `0.891 → 0.617 ms/chk`).
- **Stage 8E-3 — Fused Direct `ssm_conv_direct_d4_f32_kernel` (`ssm-conv.cuh/.cu`, `qwen35.cpp`, `ggml-cuda.cu`, `LLAMA_SM70_DIRECT_SSM_CONV=1`)**: Replaces `concat_dim0_src1_transposed` + `k_set_rows_chan_coalesced` + `ssm_conv_f32` across 47/48 GDN layers with a single 80-CTA coalesced register-window kernel — eliminating `109 MB/check` of intermediate `conv_input` HBM2 traffic and `7,144` launches (**−55%**, `0.927 → 0.415 ms/chk`).
- **DSpine `f16` Draft KV (`--spec-draft-type-k f16 --spec-draft-type-v f16`)**: Eliminates `dequantize_block_q8_0_f16` (`0.378 → 0.000 ms/chk`) on the 5-layer drafter for `+78 MiB` VRAM.

### 3.9 Stage 9 Optimizations (`s9_final`, −14.5 ms/check wall at 64K, +7–32% across Chat/Agent/Think/128K)
- **Stage 9A — Dynamic `7 ↔ 15` DSpine Block Hysteresis (`common/speculative.cpp`, `LLAMA_SM70_DYN_BLOCK=1`)**: Tracks exact previous-step accepted count (`(pos0 - last_pos0) - 1`) per sequence before launching DSpine's parallel block forward pass. Starts at `n_draft = 7` (`m = 8`), promotes to `15` (`m = 16`) when all `7/7` draft tokens accept (`DYN_UP=7`), and demotes back to `7` when `< 6` accept (`DYN_DOWN=6, DYN_STREAK=1`). Shrinks both DSpine (`5` layers + shared `lm_head`) and target verify (`q4k-skinny` `RowTiles=1`, half-active `sm70_fa96`) on reasoning/long-context turns while riding `m = 16` bursts on tool calls & code (`Agent 12K: 198.0 → 206.1 tok/s`, `Agent 50K: 97.9 → 104.9 (pk 115.7)`, `Agent 100K: 85.9 → 91.9 (pk 105.3)`, `128K greedy: 43.38 → 52.70 tok/s`, `4K think: 62.55 → 82.56 tok/s`).
- **Stage 9B — Zero-Spill 3-CTA/SM `q4k-skinny` & Gated FFN (`q4k-skinny.cu`, `LLAMA_SM70_SKINNY_3CTA=1`, `LLAMA_SM70_SKINNY_GNT_3CTA=1`)**: `cuobjdump` showed `Q5_K` and `Q6_K` (`RowTiles=1`, `SplitK=8`) sitting at `84–86` regs — just 4–6 regs above Volta's `80`-reg `3`-CTA/SM cliff. Adding `__launch_bounds__(256, 3)` dropped `Q4_K`/`Q5_K`/`Q6_K`/`IQ4_XS` to `REG: 68–79, STACK: 0` across `lm_head` and all 64 `ffn_gate/up` gated & mixed-quant layers (**24 warps/SM, `+50%` occupancy**).
- **Stage 9C — Tail-Wave & Occupancy Fixes (`q4k-skinny.cu`, `LLAMA_SM70_SKINNY_WAVEFIX=1`, `LLAMA_SM70_SKINNY_N5120_2CTA=1`)**: `nsys` per-shape breakdown at `m = 8` revealed `ssm_in` (`n = 6144`, `96` CTAs @ `1` CTA/SM = 2 waves with 64 SMs idle on wave 2), `K/V` (`n = 1024`, only `16` CTAs = 20% of SMs), and `ffn_down`/`attn_out`/`ssm_out` (`n = 5120`, `8.72 ms/chk` at 16 warps/SM). Switching `n = 6144` to `NTiles=1, SplitK=8, MinBlocks=3` (`REG: 56–60, STACK: 0` = 4 CTAs/SM, 192 CTAs in **1 single wave**), `n = 1024` to `NTiles=1, SplitK=16` (`32` active SMs), and `n = 5120` to `NTiles=1, SplitK=16, MinBlocks=2` (`REG: 56–60, STACK: 0` = **32 warps/SM**) cut `64k_greedy` by another **−0.64 ms/check** (`47.69 → 47.13 ms/chk`, `70.27 tok/s`).
- **Stage 9D — Cached `BF16 → F16` Weights + Tensor-Core Bypass of `mul_mat_vec_f` (`ggml-cuda.cu`, `LLAMA_SM70_BF16_CACHE=1`)**: Eliminates redundant per-step `convert<bf16, f16>` on DSpine's immutable `BF16` weights AND bypasses `ggml_cuda_should_use_mmvf` for `BF16` weights at `m >= 4` (where `MMVF_MAX_BATCH_SIZE = 8` had sent all 25 DSpine `BF16` GEMMs to scalar FP32 `mul_mat_vec_f` at `0.80 ms/chk`), saving **−0.64 ms/check** (`228/228` `MUL_MAT bf16` PASS).
- **Stage 9E — Pinned-Ring Async H2D `set_tensor` (`ggml-cuda.cu`, `LLAMA_SM70_ASYNC_SET=1`)**: Stages `<= 256 KiB` non-weight H2D copies into a 4 MiB pinned host ring on the active compute stream (strictly stream-ordered with flush on reads/stream destruction), cutting GPU idle from **`4.09 → 1.25 ms/check`** (`97.5%` GPU busy).
- **Stage 9F — L1-Coalesced `__ldg` in `sm70_fa96` (`fattn-sm70-q8-96.cu`, `LLAMA_SM70_FA96_LDG=1`)**: Replaces L1-bypassing `__ldcs` with `__ldg` on 34-byte `block_q8_0` KV loads so all 4 threads reading the shared 2-byte scale and overlapping unaligned words coalesce in Volta's 128 KB L1 (`REG: 165, STACK: 0`, **−0.60 ms/check** at 64K).

### 3.10 Stage 9.1 Prefill Optimizations (`br128_v4 + W1`, +9.6% @ 64K, +11.8% @ 100K, +13.1% @ 128K / −22.7 s, plus Causal-Mask Bug Fix)
- **Root Cause (`ncu` + `nsys` at 12K / 64K)**:
  - At 12K, cuBLAS `H884` FP16 GEMMs are `66.2%` of GPU time at `81.3 TFLOPS` (~86% of V100's 220 W ceiling) and `qskinny_to_f16_kernel` (`4.9%`) saturates HBM2 at `805 GB/s`. At 64K–128K, FlashAttention dominates (`41% → 62%` of total time).
  - `ncu` on `sm70_d256_splitd_dense_kernel` (`kv=6K`) revealed DRAM was only `11.95%` (`L2 hit = 90.1%`); the bottleneck was **`L1/TEX = 72.6%` (`mio_throttle = 24.33%`)** from `383.7M` SMEM load wavefronts (`8×32` warp tiles), `51.4M` SMEM bank conflicts (`31.7M` load + `19.8M` store), and `12` `__syncthreads()` per `Bc=32` tile — capping HMMA active cycles at `37.50%`.
  - **Upstream Accuracy Bug Found**: `sm70_d256_splitd_dense_kernel` received `q_pad` (multiple of 64) as `query_len` and computed `kv_offset = lo - query_len`. Whenever a prompt's final chunk was not a multiple of 64 tokens (`q_len % 64 != 0`), it shifted the causal mask by `q_pad - q_len` tokens (e.g., `0.79–1.12` logprob corruption on `P_EXACT` `q_len=263` and a 48-token shift on 12K agent `prompt_read=12176`).
- **Stage 9.1A — `sm70_prefill_br128_bc64_kernel` v4 (`fattn-sm70-prefill-96.cu/.cuh`, `fattn-sm70-d256.cu`, `LLAMA_SM70_PREFILL_BC128=2`)**:
  - `Br=128, Bc=64, D=256`, 8 warps (256 threads), **`REG: 246, STACK: 0`**, `97,536 B <= 98,304 B` dynamic SMEM (`cudaFuncAttributePreferredSharedMemoryCarveout=100`), with fallback to `sm70_prefill_bc128_kernel` (`Br=64, Bc=128`, `REG: 200, STACK: 0`) for `256 <= q_len < 512`.
  - Each warp covers **`64` Q rows** (`2×32` row-halves) × `16` K cols (`QK^T`) / `64` D cols (`PV`), reusing every K/V SMEM load twice (**−45% SMEM load wavefronts**: `383.7M → 211.0M`).
  - Constant-stride padded SMEM (`sQ[128][264]`, `sK[64][72]`, `sV[16][264]`) achieves **`0` SMEM load bank conflicts** (down from `31.7M`), `~0` store conflicts (`3.4K` vs `19.8M`), and eliminates runtime swizzle ops (`LOP3: 514 → 61`, `IMAD: 743 → 362`).
  - Dead-region double-buffering (`sk_buf1` aliases `p_s` during `QK^T`; `sv_buf1` aliases `stat_s + sum_s + pad + p_s[sw=0]` during `PV`) cuts barriers from `16 → 11` per `Bc=64` tile, paired with software-pipelined `k0/k1` `HMMA.884` interleaving and double-buffered `LDS` in `PV`.
  - Passes true `q_len` (fixing the upstream causal-mask bug; `exact_text_match = True`, top-1 logprob diff `0.001–0.005` vs stock `llama.cpp` `FLASH_ATTN_EXT`, and exact `94` tok / `74/181` draft match on 12K agent turn 0).
  - Skips `216 MiB` of unused `Qs/Os/SplitKV3` scratch in `ggml_cuda_sm70_d256_alloc_size()` (net peak VRAM stays `31,314–31,338 MiB` at `-c 229376`).
  - **`ncu` (`kv=6K` launch)**: Duration **`7.00 → 5.55 ms` (−20.7%)**, `HMMA` active **`37.50% → 50.79%` (+35.4%)**, `mio_throttle` **`24.33% → 7.35%`**, `barrier` **`7.63% → 6.04%`**.
- **Stage 9.1B — 5 Supporting Lossless Prefill Optimizations (`0.000000` logprob diff vs `br128_v4` alone)**:
  - `LLAMA_SM70_GDN_H48=1` (`gdn-chunk-sm70.cu`): Compile-time `<48, 16>` TileLang GDN chunk specialization (`1,428 → 1,269 us`, **−11.1%**).
  - `LLAMA_SM70_DIRECT_SSM_PREFILL=1` (`ssm-conv.cu`): Extends direct `ssm_conv` to `n_t <= 8192` using in-place `BLOCK_C=32` (`320` CTAs, coalesced 128 B lines) for `ggml-alloc`-aliased tensors + 2D-tiled `ssm_conv_direct_d4_prefill_f32_kernel<128, 128>` for out-of-place tensors (`75/75` `SSM_CONV` + `GLU` PASS).
  - `LLAMA_SM70_PF_ACT_CACHE=1` (`common.cuh`, `ggml-cuda.cu`, `norm.cu`, `unary.cu`): Lazy `try_ensure_f16_act_buf(160 MiB)` (with `48 MiB → 2 MiB` graceful fallback, never OOMs) + `f16_act_gen` generation-guarded invalidation; cuts `convert_unary_cont_vec4` **`49.5 → 6.5 ms/chunk` (−87%)**.
  - `LLAMA_SM70_SWIGLU_VEC4=1` (`unary.cu`): `float4` vectorized `unary_gated_op_vec4_f32_kernel<op, WRITE_F16>` eliminating 64-bit `i/n` div/mod and writing FP16 directly to `f16_act_buf` (`45.2 → 36.7 ms/chunk` + zero `down_proj` F16 convert).
  - `LLAMA_SM70_FAST_D2H=1` (`batch-copy.cu`): Bypasses `cudaHostRegister` (`60–127 ms` kernel page-pinning stall per checkpoint) on `>16 MiB` D2H copies, keeping `--ctx-checkpoints 8` 100% functional with `0` `cudaHostRegister` calls in `nsys`.

---

## 4. What We Tried & Reverted / Left Opt-In OFF (Lessons Learned)

| Experiment | Measured Result | Why It Lost / Stayed Opt-In on Volta `sm_70` |
|---|---|---|
| `sm70_fa48_decode_kernel` (`LLAMA_SM70_FA48=1`, `Br=48`, 2 CTAs/SM, `48.4 KB` smem) | `48.60 ms/chk` vs `48.42 ms/chk` (`fa96_ldg`) at 64K | Shared `kv_s` (`<= 48 KB` for 2 CTAs/SM) still hit `STACK:16` at `REG:128` and lost double-buffered prefetch; `fa96` already skips inactive row-group 2 (`if (rows_here > 0)`) at `m=8` with `STACK:0`. Left opt-in OFF. |
| `q4k_skinny_nt` `NTiles=4` (`LLAMA_SM70_SKINNY_NT4=1`, `m <= 8`) | `39.88` vs `39.33 ms/chk` (`3CTA`) | `REG:110–120` dropped occupancy from 3 → 2 CTAs/SM, outweighing 2× lower activation reads. Left opt-in OFF. |
| `q4k_skinny_nt` `SplitK=8` on `n = 5120` (`LLAMA_SM70_SKINNY_SK8_M8=1`) | `40.81` vs `39.14 ms/chk` (`+1.67 ms` slower) | `n = 5120` with `NTiles=2` only has `80` CTAs (1/SM), so `SplitK=8` halved SM occupancy to 8 warps/SM; `N5120_2CTA` (`NTiles=1, SplitK=16`) instead doubled occupancy to 32 warps/SM. |
| `FA96` `splits = 40` (`LLAMA_SM70_FA96_SPLITS=40`) | `57.28` vs `59.30 tok/s` at 64K | Stage 8's 20 splits (`80` CTAs = 1 full wave for 1-CTA/SM `fa96`) is already optimal. |
| `DFlash2-Q4_0` & `DSpine-b16-Q8_0` drafters + `q8_0` draft KV | `141.7` / `150.0` vs `159.2 tok/s` (`DSpine-b16-Q4_K_M` + `f16` KV) | `DSpine-b16-Q4_K_M` + `f16` draft KV had higher acceptance (`6.24` vs `5.63`) and lower latency than all three alternatives. |
| Naive `LLAMA_SM70_LAZY_BS=1` (Stage 7B) | `+3.4%` long prose, `-10.3%` short 12K tool turns | Detaching backend sampler mid-turn triggered `sched_need_reserve` (~50 ms) or discarded drafted tokens on `<tool_call>`. **Solved in Stage 8C-v2** via 1-token CPU grammar rejection + on-demand single-row VRAM fetch (**+7.5–10.5% on agent turns**). |
| `LLAMA_SM70_CTX_LOOKUP=1` (Stage 7C) | `+5.5%` verbatim code copy (`311.9 → 329.2`, bit-exact), tie on novel prose | Lossless 16-token prompt-suffix lookup (`n_want <= 14`, skips stochastic `draft_q`). Left opt-in OFF by default. |
| `DSpine-b16-Q8_0` drafter | `121.6` vs `125.9 tok/s`, `+900 MiB` VRAM | Extra draft bandwidth cost outweighed tiny acceptance gain. |
| Direct F32 activation input in `q4k-skinny.cu` | `100.1` vs `125.9 tok/s` (−20.5%) | 80–544 CTAs redundantly read 2× wider F32 activations from L2 and ran 32 `F2F` ops/sub-block vs converting `src1 → F16` once. |
| Fused `gdn_prologue` (`l2_norm` inside GDN) | `75.0 → 86.3 us`, `130.9 → 129.7 tok/s` | Raised register pressure `89 → 110`, halving CTA occupancy from 2 to 1 CTA/SM. |
| `sm70_prefill96` (`Bc=32` prefill FA) | `27.72 s` vs `22.05 s` @ 64K prefill | `Bc=32` doubled softmax/`exp2f`/barrier frequency vs `splitd_dense` (`Bc=64`) and redundantly dequantized `Q8_0` per Q-head. Left opt-in OFF (`LLAMA_SM70_PREFILL_V2=0`). |
| `FA96` coalesced `LDG.E.64` + `SHFL` (`v5e/f/i`) | `743–764 us` vs `708 us` (`v5d`) @ 64K | `SHFL` in `commit_q8_row` contended on Volta's MIO pipe with `LDS`/`STS`. |
| `FA_L2SWIZZLE` grid reorder on `splitd_dense` | `−0.6%` at 64K | `splitd_dense` already had `90.1%` L2 hit rate and was SMEM/MIO-bound (`L1/TEX 72.6%`), not DRAM-bound (`11.95%`). Reverted cleanly. |
| `Br=128, Bc=128` prefill FA draft | `STACK: 216` spills | Holding `acc_f[2][8][8]` (128 regs) + `acc_s[2][4][8]` (64 regs) + `p_my[2][4][4]` (32 regs) spilled 54 regs. Replaced by spill-free `Br=128, Bc=64` (`REG: 246, STACK: 0`). |
| `-ub 4096` at `-c 229376` | OOM (`4,322 MiB` compute buffer vs `~1.5 GiB` free VRAM) | Keep `-b 2048 -ub 2048` at `-c 229376` on 32 GB V100. |
