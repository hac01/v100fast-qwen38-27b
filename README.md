# v100fast-qwen38-27b

**Make Qwen3.8-27B + DSpine fly on a single NVIDIA Tesla V100 (`sm_70`, 32 GB) — up to `184 tok/s` at 12K context, `80–95 tok/s` at 100K context, `55–67 tok/s` at 155K context, and up to `329 tok/s` on verbatim code edits, with zero quality loss (`UD-Q4_K_XL` + `q8_0` KV cache).**

All numbers below were measured on a single **Tesla V100-PCIE-32GB** capped at **220 W** (`ECC ON`, `0` uncorrected ECC, cooled `<= 45 °C` before each timed run) with `Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf` + `Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf` (`-c 229376 -ub 2048 -ctk q8_0 -ctv q8_0`).

---

## Measured Benchmarks (Single V100-PCIE-32GB @ 220 W)

| Workload | Baseline (`dspine` upstream) | Stage 6 | **v100fast (Stage 7)** | Speedup vs Baseline |
|---|---:|---:|---:|---:|
| **Chat (short)** | 57.9 tok/s | 113.9 tok/s | **115.5 tok/s** | **1.99× (+99.5%)** |
| **Code (short)** | 87.3 tok/s | 168.8 tok/s | **171.4 tok/s** | **1.96× (+96.3%)** |
| **Agent (12K ctx)** | 83.9 tok/s | 179.9 tok/s | **184.3 tok/s** | **2.20× (+119.7%)** |
| **All 12K Suite Avg** | — | 147.9 tok/s | **150.3 tok/s** | — |
| **Agent (50K ctx)** | 55.1 tok/s | 86.5 tok/s *(pk 91.6)* | **89.0 tok/s** *(pk **95.2**)* | **1.62× (+61.5%)** |
| **Agent (100K ctx)** | 42.2 tok/s | 78.2 tok/s *(pk 92.4)* | **79.9 tok/s** *(pk **94.8**)* | **1.89× (+89.3%)** |
| **155K Live Coding Session** | — | 55.2 – 66.6 tok/s | **55.2 – 66.6+ tok/s** | — |
| **196K Stress Test** | — | 550.3 pp / 37.5 tg (`31,058 MiB`) | **550+ pp / 37.5+ tg** | — |
| **Prompt Read (12K / 50K / 100K)** | 898 / 709 / — tok/s | 1,080.4 / 874.3 / 628.1 tok/s | **1,083.0 / 874.8 / 628.3 tok/s** | **+21% / +23%** |
| **64K Verify Step (GPU / wall)** | ~125 / ~130 ms | 60.98 / 66.03 ms *(92% busy)* | **60.61 / 64.73 ms** *(94% busy)* | **2.01× (−50.2% wall)** |
| **Verbatim Code Copy** *(opt-in `LLAMA_SM70_CTX_LOOKUP=1`)* | — | 311.9 tok/s *(DSpine only)* | **329.2 tok/s** *(bit-exact)* | **+5.5% over DSpine** |
| **Deterministic `temp=0` Parity** | 61.0 tok/s | 115.55 tok/s | **head=15/15 (12K), 4/4 (50K), 4/4 (100K)** | — |

---

## Quick Start (One-Line `curl` Install)

Requires Linux with an NVIDIA Tesla V100 (`sm_70`), CUDA Toolkit (`nvcc` 12.x), `cmake`, `g++`, `git`, and `curl`.

```bash
curl -fsSL https://raw.githubusercontent.com/hac01/v100fast-qwen38-27b/main/install.sh | bash
```

What this does automatically:
1. Clones pinned [`Ajay9o9/llama.cpp`](https://github.com/Ajay9o9/llama.cpp) (`4703b7d`, DSpine base).
2. Applies `patches/v100fast-stage7.patch` and copies the custom Volta `sm_70` CUDA kernels from `overlay/`.
3. Compiles `llama-server` and `test-backend-ops` specifically for `sm_70` (`CMAKE_CUDA_ARCHITECTURES=70`).
4. Downloads `Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf` (17.4 GB) and `Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf` (1.2 GB) with resumable `curl`.
5. Writes your ready-to-run config (`.v100fast.env`) sized automatically for your GPU VRAM and system RAM.

### Common Installer Options

```bash
# Use official Unsloth Qwen3.8-27B UD-Q4_K_XL instead of Huihui:
curl -fsSL https://raw.githubusercontent.com/hac01/v100fast-qwen38-27b/main/install.sh | bash -s -- --model qwen

# Build, run all CUDA test-backend-ops verification gates, and start immediately:
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

OpenAI-compatible endpoint: `http://127.0.0.1:8080/v1` (model alias `qwen3.8-27b`). Works out of the box with OpenCode, Hermes, Cline, Continue, and standard OpenAI SDKs (including tool calling and JSON schema grammars).

---

## Hardware & Memory Footprint

| Component | Footprint (V100-32GB, `-c 229376`) |
|---|---:|
| Target model (`UD-Q4_K_XL`) | ~17.4 GB |
| DSpine-b16 drafter (`Q4_K_M`) | ~1.2 GB |
| `q8_0` KV cache (`229,376` ctx, 16 full-attn layers) + GDN states + buffers | ~12.4 GB |
| **Total Peak VRAM** (verified up to 195,237 tokens) | **31,058 / 32,768 MiB** *(1.71 GB free)* |
| Host RAM prompt cache (`--cache-ram 15360`) | Up to 15 GB *(auto-scaled if host has < 20 GB RAM)* |

> **Note for passively cooled V100-PCIE cards**: Keeping the power cap at **220 W** (`sudo nvidia-smi -pm 1 -pl 220`) prevents thermal throttling while preserving ~95%+ of peak throughput. `scripts/serve.sh` applies this automatically if passwordless `sudo nvidia-smi` is available (override with `--power-limit 0` to skip).

---

## What Makes It Fast on Volta (`sm_70`)?

Every optimization is bit-exact or passes full `test-backend-ops` tolerances and `temp=0` SHA parity. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full SASS, `nsys`, and design deep dive.

1. **`sm70_fa96` v6 Direct `q8_0` & `k8v4` (`q8_0/q4_0`) Decode FlashAttention (`LLAMA_SM70_FA96=1`)**
   - Single-pass `Br=96, Bc=32` 12-warp Volta HMMA kernel reading `q8_0` (and async `v5d` `q4_0` V for `-ctk q8_0 -ctv q4_0` ~3 GiB VRAM savings at 300K+ ctx) directly with spill-free softmax, double-buffered `k_s`/`v_s` smem (`4 → 3` `__syncthreads()`/tile), pre-scaled `Q` (`scale * Log2E`), 2-pointer KV stepping, and branchless 4B-aligned `LDG.E.32` loads (**zero `SHFL`**, avoiding Volta MIO-pipe contention).
   - Cuts 16K / 64K / 128K `q8_0` attention to **`241.7 / 703.5 / 1,354.9 us`** (**36.7–38.3 TFLOPS**; vs `1,768 / 3,425 us` upstream).
2. **`IQ4_XS` / `IQ4_NL` 4-Register `__byte_perm` LUT + Per-Graph FP16 Activation Cache (`q4k-skinny.cu`, `q8-skinny.cu`, `ggml-cuda.cu`)**
   - Eliminates `64× CALL` + `70× SHFL` per tile via a bit-exact 4-register `PRMT` LUT (`65,536/65,536` `errs=0`), adds `Q5_K`/`Q6_K` SoA coalescing, cross-tile & Gate+Up cross-projection prefetch, mixed-quant FFN fusion (`14` pairs), and `RowTiles=1` FP16 paths for `m = 2..8`.
   - Stage 7G pre-allocates a per-graph 1 MiB FP16 activation cache (`f16_act_buf`) that reuses converted FP16 `src1` across shared projections (`Q/K/V`, `ssm_in/ssm_z`, unfused `gate/up`), cutting `convert_unary_cont_vec4` from `1.49 → 0.79 ms/check` (**−47% time, −131 launches/check**).
3. **Single-Pass `float4`-Vectorized Multi-Row `q8-small.cu` (Stage 7E-v2)**
   - Replaces the 2-kernel `q8_small_partial`/`reduce` (`1.41 ms/check`, 192 launches/check across 96 GDN alpha/beta calls) with `q8_small_tiled_kernel<3,4,8>` (64 CTAs = 1 wave on 80 SMs, `float4` activation loads reused across 3 weight rows) → **`0.88 ms/check`** (**−37%**, halved launches).
4. **Adaptive KV Padding (`src/llama-kv-cache.cpp`, Stage 7D-1)**
   - Scales KV cell padding adaptively (`256 / 1024 / 4096` at `> 4K / > 16K` tokens), eliminating mid-decode `52 ms` `cudaFreeHost + cudaMallocHost + cudaGraphInstantiate` stalls by 16× at long context (`0` mid-decode host allocs in 64K `nsys`).
5. **GDN Direct-State + Exact FP32 Rollback Replay (`LLAMA_SM70_GDN_REPLAY=1`)**
   - Speculative verification (`K=16`) previously wrote 16 full FP32 GDN states per layer (`2.42 GB/step`) plus a `302 MB/step` `k_get_rows` gather (`4.18 ms/step`).
   - Replaying `0 .. n_accepted - 1` in registers from `slot 1` (`s_start`) + a `24.9 MB` `(k, v, exp(g), beta)` ring buffer cuts GDN traffic **5.6×** (`4.18 → 1.77 ms/step`, `REG: 82, STACK: 0`) with 100% bit-for-bit identical FP32 states and automatic RAM-checkpoint materialization hooks.
6. **GDN Chunked Prefill (`LLAMA_SM70_GDN_CHUNK=1`) + Draft Ubatch Cap (`<= 512`)**
   - Runs leading prefill tokens through Volta HMMA chunks and caps draft scratchpad allocation (`-ub 2048` on main model while saving ~1.5 GB VRAM).
7. **Tiled Transpose-Concat + Warp `Q8_0`/`Q4_0` & Coalesced Conv `set-rows` + Hoisted `s_slot_rev`**
   - `concat_dim0_src1_transposed`: cuts GDN `build_conv_state` concat `19.4 → 6.9 us` (**−0.60 ms/step**).
   - `k_set_rows_q8_0_warp` & `k_set_rows_q4_0_warp` (32 lanes/block, 32× higher SM occupancy) + `k_set_rows_chan_coalesced` (128B coalesced conv-state scatter): **~−1.15 ms/step**.
   - Hoisted `s_slot_rev` per buffer type: removes 141 redundant `arange+scale+cast` launches/step (**~−0.30 ms/step**).
8. **2-Stage Parallel `float4` `argmax` + Zero-D2H `--backend-sampling`**
   - Replaces 1-CTA `argmax_f32` (`98.1 us` × 16 = `1.57 ms/step`) with an 80-CTA `float4` 2-stage reduction (`~5 us`).
   - Elides inactive no-op samplers (`penalties`/`dry`/`xtc`/`typical_p`/`top_n_sigma`), skips the `15.89 MB/step` (`1.24 ms`) `t_sampled_logits` D2H PCIe transfer, and skips the 248k-element CPU candidate rebuild — while automatically falling back to CPU for active penalties, grammars, JSON schemas, reasoning budgets, and `logprobs`.
   - **Optional env flags (OFF by default)**: `LLAMA_SM70_CTX_LOOKUP=1` (lossless 16-token prompt-suffix lookup before neural drafting on greedy turns; **329.2 vs 311.9 tok/s, +5.5%** bit-exact on verbatim code copy) and `LLAMA_SM70_LAZY_BS=1` (lazy-grammar-aware `--backend-sampling`; +3.4% on long non-tool prose).

---

## Verification & Benchmarking

```bash
# Run all 9 CUDA test-backend-ops verification gates
./scripts/verify.sh

# Run reproducible speed + temp=0 SHA benchmark
python3 ./scripts/bench.py --url http://127.0.0.1:8080 --long-ctx 12000 50000 100000
```

Verified passing on Tesla V100-PCIE-32GB (CUDA 12.8):
- `MUL_MAT` (`q4_K|q5_K|q6_K|iq4_xs|iq4_nl`): **479/479 PASS** | `MUL_MAT` (`q8_0`): **258/258 PASS**
- `GET_ROWS`: **228/228 PASS**
- `FLASH_ATTN_EXT` `hsk=256` (`FA96=1`, incl. `Q8/Q4` & `Q4/Q4`): **249/249 PASS** on both `TEST_FA_CAUSAL=1` and `TEST_FA_CAUSAL=2`
- `GATED_DELTA_NET` (`REPLAY=1, CHUNK=0`): **63/63 PASS**
- `ARGMAX`: **7/7 PASS** | `SET_ROWS`: **191/191 PASS** | `CONCAT`: **221/221 PASS**
- Output parity vs Stage 6 (`s6_v2`): `12K: head=15/15 (sha=10/15, 5 random tool-call UUIDs)`, `50K: head=4/4`, `100K: head=4/4`

---

## Repository Layout

```text
v100fast-qwen38-27b/
├── install.sh                    # One-line curl installer (build + download + config)
├── patches/
│   └── v100fast-stage7.patch     # Clean git diff against pinned Ajay9o9/llama.cpp@4703b7d
├── overlay/
│   └── ggml/src/ggml-cuda/       # Custom Volta sm_70 kernels + vendored CuTe/FA2 headers
├── scripts/
│   ├── build.sh                  # Standalone clone + patch + sm_70 CMake build
│   ├── serve.sh                  # Production server launcher with Stage 7 flags
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
