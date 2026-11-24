# v100fast-qwen38-27b

**Make Qwen3.8-27B + DSpine fly on a single NVIDIA Tesla V100 (`sm_70`, 32 GB) — `120 tok/s` chat, `180 tok/s` code, `198–204 tok/s` at 12K agent context, `98 tok/s` (`104.7` peak) at 50K context, and `86 tok/s` (`103.1` peak) at 100K context, with zero quality loss (`UD-Q4_K_XL` + `q8_0` target KV cache).**

All numbers below were measured on a single **Tesla V100-PCIE-32GB** capped at **220 W** (`ECC ON`, `0` uncorrected ECC) with `Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf` + `Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf` (`-c 229376 -ub 2048 -ctk q8_0 -ctv q8_0 --spec-draft-type-k f16 --spec-draft-type-v f16 --backend-sampling`).

---

## Measured Benchmarks (Single V100-PCIE-32GB @ 220 W)

| Workload | Baseline (`dspine` upstream) | Stage 6 (`s6_v2`) | Stage 7 (`s7v2`, cooled) | **Stage 8 (`s8i`, hot 56–59 °C)** | Speedup vs Baseline |
|---|---:|---:|---:|---:|---:|
| **Chat (short)** | 57.9 tok/s | 113.9 tok/s | 115.5 tok/s | **119.9 tok/s** | **2.07× (+107.1%)** |
| **Code (short)** | 87.3 tok/s | 168.8 tok/s | 171.4 tok/s | **179.6 tok/s** | **2.06× (+105.7%)** |
| **Agent (12K ctx, tools)** | 83.9 tok/s | 179.9 tok/s | 184.3 tok/s | **198.0 tok/s** *(203.6 on `s8g`)* | **2.36–2.43× (+136–143%)** |
| **All 12K Suite Avg** | — | 147.9 tok/s | 150.3 tok/s | **157.3 tok/s** | — |
| **Agent (50K ctx, tools)** | 55.1 tok/s | 86.5 tok/s *(pk 91.6)* | 89.0 tok/s *(pk 95.2)* | **97.9 tok/s** *(pk **104.7**)* | **1.78× (+77.7%)** |
| **Agent (100K ctx, tools)** | 42.2 tok/s | 78.2 tok/s *(pk 92.4)* | 79.9 tok/s *(pk 94.8)* | **85.9 tok/s** *(pk **103.1**)* | **2.04× (+103.6%)** |
| **64K `nsys` Decode (300 tok)** | — | — | 59.8 tok/s | **63.9 tok/s** | **+6.9% vs Stage 7** |
| **64K Verify Step (GPU / wall)** | ~125 / ~130 ms | 60.98 / 66.03 ms *(92% busy)* | 60.61 / 64.73 ms *(94% busy)* | **57.54 / 61.63 ms** *(93% busy)* | **2.11× (−52.6% wall)** |
| **Prompt Read (12K / 50K / 100K)** | 898 / 709 / — tok/s | 1,080 / 874 / 628 tok/s | 1,083 / 875 / 628 tok/s | **1,071 / 872 / 627 tok/s** | **+19% / +23%** |
| **155K Live Coding Session** | — | 55.2 – 66.6 tok/s | — | — | — |
| **196K Stress Test** | — | 550.3 pp / 37.5 tg (`31,058 MiB`) | — | — | — |
| **Verbatim Code Copy** *(opt-in `LLAMA_SM70_CTX_LOOKUP=1`)* | — | 311.9 tok/s *(DSpine only)* | 329.2 tok/s *(bit-exact)* | — | **+5.5% over DSpine** |
| **Deterministic `temp=0` Parity** | 61.0 tok/s | 115.55 tok/s | head=15/15, 4/4, 4/4 | **head=15/15 (12K), 4/4 (50K), 4/4 (100K)** | **100% parity** |

*(Note: Stage 8 `s8i` ran back-to-back without cooldown sleeps at `56–59 °C`, whereas Stage 7 `s7v2` cooled to `<= 45 °C` before each run. On `speed_bench`, `10/10` chat+code prompts match Stage 7 bit-for-bit SHA; the 5 agent turns only differ in random tool-call IDs with identical `head` and acceptance `0.43` / `0.30`.)*

### 64K `nsys` Kernel Verification (`s7e2` → `s8i`, 75 checks, 220 W)

| Kernel / Family | Stage 7 (`s7e2`) | **Stage 8 (`s8i`)** | Delta |
|---|---:|---:|---:|
| **Total Wall / GPU per check** | 64.73 / 60.61 ms | **61.63 / 57.54 ms** | **−3.10 / −3.07 ms (−4.8% / −5.1%)** |
| **GDN conv + concat + set_rows** (8E-3) | 0.927 ms/chk (`10,944×`) | **0.415 ms/chk (`3,800×`)** | **−55% (−7,144 launches)** |
| **`q8_small` (`ssm_beta + alpha`)** (8E-2) | 0.878 ms/chk (`6,240×`) | **0.617 ms/chk (`3,648×`)** | **−30% (−3,648 launches)** |
| **`convert<f32, f16>`** (8D + 8E-1) | 0.792 ms/chk (`18,693×`) | **0.297 ms/chk (`7,676×`)** | **−62% (−11,017 launches)** |
| **`ADD + RMS_NORM`** (8D) | 1.521 ms/chk (`23,182×`) | **1.062 ms/chk (`15,504×`)** | **−30% (137.8 fused/chk)** |
| **Un-graphed `cudaLaunchKernel`** (8A) | 1.92 ms/chk (`245×/chk`) | **0.96 ms/chk (`140×/chk`)** | **−50%** |
| **DSpine FP32 SGEMMs + Q8 draft KV dequant** (8B + `f16` draft KV) | ~1.08 ms/chk | **0.000 ms/chk** | **100% eliminated** |

---

## Quick Start (One-Line `curl` Install)

Requires Linux with an NVIDIA Tesla V100 (`sm_70`), CUDA Toolkit (`nvcc` 12.x), `cmake`, `g++`, `git`, and `curl`.

```bash
curl -fsSL https://raw.githubusercontent.com/hac01/v100fast-qwen38-27b/main/install.sh | bash
```

What this does automatically:
1. Clones pinned [`Ajay9o9/llama.cpp`](https://github.com/Ajay9o9/llama.cpp) (`4703b7d`, DSpine base).
2. Applies `patches/v100fast-stage8.patch` and copies the custom Volta `sm_70` CUDA kernels from `overlay/`.
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

---

## Environment Variables & Toggles

| Variable | Default | Purpose |
|---|---|---|
| `LLAMA_SM70_FA96` | `0` *(set `1` in `serve.sh`)* | Volta `sm70_fa96` v6 direct `q8_0` / `k8v4` FlashAttention decode |
| `LLAMA_SM70_GDN_CHUNK` | `0` *(set `1` in `serve.sh`)* | Volta HMMA chunked GDN prefill |
| `LLAMA_SM70_GDN_REPLAY` | `0` *(set `1` in `serve.sh`)* | GDN direct-state read + exact FP32 rollback replay |
| `LLAMA_SM70_MULTI_GRAPH` | `1` (ON) | 8-slot shape-keyed CUDA graph cache (Stage 8A) |
| `LLAMA_SM70_BF16_AS_F16` | `1` (ON) | Route Volta `BF16` GEMM through FP16 Tensor Cores w/ FP32 acc (Stage 8B) |
| `LLAMA_SM70_LAZY_BS` | `1` (ON) | GPU grammar-rejection `--backend-sampling` w/ on-demand 1-row fetch (Stage 8C-v2) |
| `LLAMA_SM70_FUSED_ADD_RMS` | `1` (ON) | Fused dual-output `ADD + RMS_NORM + MUL` → `f16_act_buf` (Stage 8D) |
| `LLAMA_SM70_GATED_F16_OUT` | `1` (ON) | Ping-pong `q4k_skinny_gated` direct FP16 output → `down_proj` (Stage 8E-1) |
| `LLAMA_SM70_Q8_SMALL_DUAL` | `1` (ON) | Fused dual `q8_small` (`ssm_beta + ssm_alpha`, Stage 8E-2) |
| `LLAMA_SM70_DIRECT_SSM_CONV` | `1` (ON) | Fused direct `ssm_conv_direct_d4_f32` + `silu` + conv state store (Stage 8E-3) |
| `LLAMA_SM70_CTX_LOOKUP` | `0` (OFF) | Optional lossless 16-token prompt-suffix lookup (`+5.5%` on verbatim code copy) |

Set any default-ON Stage 8 variable to `0` to isolate or disable that optimization without rebuilding.

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
- `ARGMAX`: **7/7 PASS** | `SET_ROWS`: **191/191 PASS** | `CONCAT`: **221/221 PASS**
- Output parity vs Stage 7 (`s7v2`): `12K: head=15/15 (sha=10/15, 5 random tool-call UUIDs)`, `50K: head=4/4`, `100K: head=4/4`

---

## Repository Layout

```text
v100fast-qwen38-27b/
├── install.sh                    # One-line curl installer (build + download + config)
├── patches/
│   └── v100fast-stage8.patch     # Clean git diff against pinned Ajay9o9/llama.cpp@4703b7d
├── overlay/
│   └── ggml/src/ggml-cuda/       # Custom Volta sm_70 kernels + vendored CuTe/FA2 headers
├── scripts/
│   ├── build.sh                  # Standalone clone + patch + sm_70 CMake build
│   ├── serve.sh                  # Production server launcher with Stage 8 flags
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
