# v100fast-qwen38-27b

**Make Qwen3.8-27B + DSpine fly on a single NVIDIA Tesla V100 (`sm_70`, 32 GB) — up to `180 tok/s` at 12K context, `78–92 tok/s` at 100K context, and `55–67 tok/s` at 155K context, with zero quality loss (`UD-Q4_K_XL` + `q8_0` KV cache).**

All numbers below were measured on a single **Tesla V100-PCIE-32GB** capped at **220 W** (`ECC ON`, `0` uncorrected ECC, cooled `<= 45 °C` before each timed run) with `Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf` + `Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf` (`-c 229376 -ub 2048 -ctk q8_0 -ctv q8_0`).

---

## Measured Benchmarks (Single V100-PCIE-32GB @ 220 W)

| Workload | Baseline (`dspine` upstream) | **v100fast (Stage 6)** | Speedup |
|---|---:|---:|---:|
| **Chat (short)** | 57.9 tok/s | **113.9 tok/s** | **1.97× (+96.7%)** |
| **Code (short)** | 87.3 tok/s | **168.8 tok/s** | **1.93× (+93.4%)** |
| **Agent (12K ctx)** | 83.9 tok/s | **179.9 tok/s** | **2.14× (+114.4%)** |
| **Agent (50K ctx)** | 55.1 tok/s | **86.5 tok/s** *(peak 91.6)* | **1.57× (+57.0%)** |
| **Agent (100K ctx)** | 42.2 tok/s | **78.2 tok/s** *(peak 92.4)* | **1.85× (+85.3%)** |
| **155K Live Coding Session** | — | **55.2 – 66.6 tok/s** | — |
| **196K Stress Test** | — | **550.3 tok/s pp / 37.5 tok/s tg** (`31,058 MiB` peak) | — |
| **Prompt Read (12K / 50K / 100K)** | 898 / 709 / — tok/s | **1,080.4 / 874.3 / 628.1 tok/s** | **+20% / +23%** |
| **64K Verify Step (GPU / wall)** | ~125 / ~130 ms | **60.98 / 66.03 ms** *(92% GPU busy)* | **2.0× (−49.2% wall)** |
| **Deterministic `temp=0` Parity** | 61.0 tok/s | **115.55 tok/s** *(4/4 exact SHA match)* | **1.89× (+89.4%)** |

---

## Quick Start (One-Line `curl` Install)

Requires Linux with an NVIDIA Tesla V100 (`sm_70`), CUDA Toolkit (`nvcc` 12.x), `cmake`, `g++`, `git`, and `curl`.

```bash
curl -fsSL https://raw.githubusercontent.com/hac01/v100fast-qwen38-27b/main/install.sh | bash
```

What this does automatically:
1. Clones pinned [`Ajay9o9/llama.cpp`](https://github.com/Ajay9o9/llama.cpp) (`4703b7d`, DSpine base).
2. Applies `patches/v100fast-stage6.patch` and copies the custom Volta `sm_70` CUDA kernels from `overlay/`.
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

1. **`sm70_fa96` v5d Direct `q8_0` Decode FlashAttention (`LLAMA_SM70_FA96=1`)**
   - Single-pass `Br=96, Bc=32` 12-warp Volta HMMA kernel reading `q8_0` KV directly with spill-free softmax, 2-pointer KV stepping, and branchless 4B-aligned `LDG.E.32` loads (**zero `SHFL`**, avoiding Volta MIO-pipe contention).
   - Cuts 64K / 128K `q8_0` attention from `1,768 / 3,425 us` (upstream) → **`708 / 1,353 us`** (**36.4–38.1 TFLOPS**).
2. **`IQ4_XS` / `IQ4_NL` 4-Register `__byte_perm` LUT + Skinny GEMM Overhaul (`q4k-skinny.cu`)**
   - Eliminates `64× CALL` + `70× SHFL` per tile via a bit-exact 4-register `PRMT` LUT (`65,536/65,536` `errs=0`), adds `Q5_K`/`Q6_K` SoA coalescing, cross-tile & Gate+Up cross-projection prefetch, mixed-quant FFN fusion (`14` pairs), and `RowTiles=1` FP16 paths for `m = 2..8`.
3. **GDN Direct-State + Exact FP32 Rollback Replay (`LLAMA_SM70_GDN_REPLAY=1`)**
   - Speculative verification (`K=16`) previously wrote 16 full FP32 GDN states per layer (`2.42 GB/step`) plus a `302 MB/step` `k_get_rows` gather (`4.18 ms/step`).
   - Replaying `0 .. n_accepted - 1` in registers from `slot 1` (`s_start`) + a `24.9 MB` `(k, v, exp(g), beta)` ring buffer cuts GDN traffic **5.6×** (`4.18 → 1.75 ms/step`, `REG: 82, STACK: 0`) with 100% bit-for-bit identical FP32 states and automatic RAM-checkpoint materialization hooks.
4. **GDN Chunked Prefill (`LLAMA_SM70_GDN_CHUNK=1`) + Draft Ubatch Cap (`<= 512`)**
   - Runs leading prefill tokens through Volta HMMA chunks and caps draft scratchpad allocation (`-ub 2048` on main model while saving ~1.5 GB VRAM).
5. **Tiled Transpose-Concat + Warp `Q8_0` & Coalesced Conv `set-rows` + Hoisted `s_slot_rev`**
   - `concat_dim0_src1_transposed`: cuts GDN `build_conv_state` concat `19.4 → 6.9 us` (**−0.60 ms/step**).
   - `k_set_rows_q8_0_warp` (32 lanes/block, 32× higher SM occupancy) + `k_set_rows_chan_coalesced` (128B coalesced conv-state scatter): **~−1.15 ms/step**.
   - Hoisted `s_slot_rev` per buffer type: removes 141 redundant `arange+scale+cast` launches/step (**~−0.30 ms/step**).
6. **2-Stage Parallel `float4` `argmax` + Zero-D2H `--backend-sampling`**
   - Replaces 1-CTA `argmax_f32` (`98.1 us` × 16 = `1.57 ms/step`) with an 80-CTA `float4` 2-stage reduction (`~5 us`).
   - Elides inactive no-op samplers (`penalties`/`dry`/`xtc`/`typical_p`/`top_n_sigma`), skips the `15.89 MB/step` (`1.24 ms`) `t_sampled_logits` D2H PCIe transfer, and skips the 248k-element CPU candidate rebuild — while automatically falling back to CPU for active penalties, grammars, JSON schemas, reasoning budgets, and `logprobs`.

---

## Verification & Benchmarking

```bash
# Run all 8 CUDA test-backend-ops verification gates
./scripts/verify.sh

# Run reproducible speed + temp=0 SHA benchmark
python3 ./scripts/bench.py --url http://127.0.0.1:8080 --long-ctx 12000 50000 100000
```

Verified passing on Tesla V100-PCIE-32GB (CUDA 12.8):
- `MUL_MAT` (`q4_K|q5_K|q6_K|iq4_xs|iq4_nl`): **479/479 PASS**
- `GET_ROWS`: **228/228 PASS**
- `FLASH_ATTN_EXT` `hsk=256` (`FA96=1`): **249/249 PASS** on both `TEST_FA_CAUSAL=1` and `TEST_FA_CAUSAL=2`
- `GATED_DELTA_NET` (`REPLAY=1, CHUNK=0`): **63/63 PASS**
- `ARGMAX`: **7/7 PASS** | `SET_ROWS`: **191/191 PASS** | `CONCAT`: **221/221 PASS**
- 4-turn multi-turn `temp=0` parity (730 tokens, ~160 rollbacks, RAM prompt-cache save/restore): **4/4 exact SHA match**

---

## Repository Layout

```text
v100fast-qwen38-27b/
├── install.sh                    # One-line curl installer (build + download + config)
├── patches/
│   └── v100fast-stage6.patch     # Clean git diff against pinned Ajay9o9/llama.cpp@4703b7d
├── overlay/
│   └── ggml/src/ggml-cuda/       # Custom Volta sm_70 kernels + vendored CuTe/FA2 headers
├── scripts/
│   ├── build.sh                  # Standalone clone + patch + sm_70 CMake build
│   ├── serve.sh                  # Production server launcher with Stage 6 flags
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
