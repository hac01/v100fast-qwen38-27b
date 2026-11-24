# Architecture & Volta (`sm_70`) Optimization Deep Dive

This document explains why **Qwen3.8-27B + DSpine** runs **~2.0–2.4× faster** on a single Tesla V100-PCIE-32GB (`sm_70`, 220 W, `q8_0` target KV + `f16` draft KV) with `v100fast` Stage 8, including the `cuobjdump -sass` and `nsys` findings that drove every optimization — and the experiments we tested and reverted or left opt-in OFF.

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
| **Stage 8 Final (`s8i`)** | **57.54 ms** | **61.63 ms** | **93.4%** | **Multi-slot CUDA graph cache + Volta `BF16→FP16` Tensor Core GEMM + GPU grammar-rejection `--backend-sampling` + fused `ADD+RMS_NORM+MUL→f16` + ping-pong `q4k_skinny_gated→f16` + dual `q8_small` + direct `ssm_conv_direct_d4_f32` + `f16` DSpine draft KV** |

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

---

## 4. What We Tried & Reverted / Left Opt-In OFF (Lessons Learned)

| Experiment | Measured Result | Why It Lost / Stayed Opt-In on Volta `sm_70` |
|---|---|---|
| Naive `LLAMA_SM70_LAZY_BS=1` (Stage 7B) | `+3.4%` long prose, `-10.3%` short 12K tool turns | Detaching backend sampler mid-turn triggered `sched_need_reserve` (~50 ms) or discarded drafted tokens on `<tool_call>`. **Solved in Stage 8C-v2** via 1-token CPU grammar rejection + on-demand single-row VRAM fetch (**+7.5–10.5% on agent turns**). |
| `LLAMA_SM70_CTX_LOOKUP=1` (Stage 7C) | `+5.5%` verbatim code copy (`311.9 → 329.2`, bit-exact), tie on novel prose | Lossless 16-token prompt-suffix lookup (`n_want <= 14`, skips stochastic `draft_q`). Left opt-in OFF by default. |
| `DSpine-b16-Q8_0` drafter | `121.6` vs `125.9 tok/s`, `+900 MiB` VRAM | Extra draft bandwidth cost outweighed tiny acceptance gain. |
| Direct F32 activation input in `q4k-skinny.cu` | `100.1` vs `125.9 tok/s` (−20.5%) | 80–544 CTAs redundantly read 2× wider F32 activations from L2 and ran 32 `F2F` ops/sub-block vs converting `src1 → F16` once. |
| Fused `gdn_prologue` (`l2_norm` inside GDN) | `75.0 → 86.3 us`, `130.9 → 129.7 tok/s` | Raised register pressure `89 → 110`, halving CTA occupancy from 2 to 1 CTA/SM. |
| `sm70_prefill96` (`Bc=32` prefill FA) | `27.72 s` vs `22.05 s` @ 64K prefill | `Bc=32` doubled softmax/`exp2f`/barrier frequency vs `splitd_dense` (`Bc=64`) and redundantly dequantized `Q8_0` per Q-head. Left opt-in OFF (`LLAMA_SM70_PREFILL_V2=0`). |
| `FA96` coalesced `LDG.E.64` + `SHFL` (`v5e/f/i`) | `743–764 us` vs `708 us` (`v5d`) @ 64K | `SHFL` in `commit_q8_row` contended on Volta's MIO pipe with `LDS`/`STS`. |
