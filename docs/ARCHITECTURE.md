# Architecture & Volta (`sm_70`) Optimization Deep Dive

This document explains why **Qwen3.8-27B + DSpine** runs **~2.0× faster** on a single Tesla V100-PCIE-32GB (`sm_70`, 220 W, `q8_0` KV) with `v100fast` Stage 6, including the `cuobjdump -sass` and `nsys` findings that drove every optimization — and the experiments we tested and reverted.

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
| **Stage 6 Final (`s6_v2`)** | **60.98 ms** | **66.03 ms** | **92.4%** | Tiled transpose-concat + warp `Q8_0` & coalesced conv `set-rows` + hoisted `s_slot_rev` + parallel `argmax` + zero-D2H `--backend-sampling` |

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

---

## 4. What We Tried & Reverted (Lessons Learned)

| Experiment | Measured Result | Why It Lost on Volta `sm_70` |
|---|---|---|
| `DSpine-b16-Q8_0` drafter | `121.6` vs `125.9 tok/s`, `+900 MiB` VRAM | Extra draft bandwidth cost outweighed tiny acceptance gain. |
| Direct F32 activation input in `q4k-skinny.cu` | `100.1` vs `125.9 tok/s` (−20.5%) | 80–544 CTAs redundantly read 2× wider F32 activations from L2 and ran 32 `F2F` ops/sub-block vs converting `src1 → F16` once. |
| Fused `gdn_prologue` (`l2_norm` inside GDN) | `75.0 → 86.3 us`, `130.9 → 129.7 tok/s` | Raised register pressure `89 → 110`, halving CTA occupancy from 2 to 1 CTA/SM. |
| `sm70_prefill96` (`Bc=32` prefill FA) | `27.72 s` vs `22.05 s` @ 64K prefill | `Bc=32` doubled softmax/`exp2f`/barrier frequency vs `splitd_dense` (`Bc=64`) and redundantly dequantized `Q8_0` per Q-head. Left opt-in OFF (`LLAMA_SM70_PREFILL_V2=0`). |
| `FA96` coalesced `LDG.E.64` + `SHFL` (`v5e/f/i`) | `743–764 us` vs `708 us` (`v5d`) @ 64K | `SHFL` in `commit_q8_row` contended on Volta's MIO pipe with `LDS`/`STS`. |
