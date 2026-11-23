# Third-Party Software, Kernels & Models

This repository distributes its own scripts, documentation, Stage 6 patches, and Volta (`sm_70`) CUDA kernel overlays under the MIT License, while relying on and crediting the following third-party projects (each keeping its own license):

| Component | Upstream / Source | License | Notes |
|---|---|---|---|
| **llama.cpp (DSpine base)** | https://github.com/Ajay9o9/llama.cpp (`4703b7d`) based on https://github.com/ggml-org/llama.cpp | MIT | Cloned at pinned commit by `scripts/build.sh` |
| **v100-skinny kernels** | `overlay/ggml/src/ggml-cuda/LICENSE.v100-skinny` | MIT | Base `sm_70` skinny quantized GEMM & `q8-small` infrastructure extended here |
| **NVIDIA CUTLASS / CuTe headers** | https://github.com/NVIDIA/cutlass (`62750a2b`) | Apache-2.0 | Header-only subset in `overlay/ggml/src/ggml-cuda/sm70-vendor/{cute,cutlass}` (`LICENSE-cute-cutlass`) |
| **flash-attention-v100 / 1CatAI D256** | https://github.com/zhinianqin/flash-attention-v100 (`c2eda5e6`), 1CatAI `1Cat-vLLM` | BSD-3-Clause | Header-only FA2 primitives in `overlay/ggml/src/ggml-cuda/sm70-vendor/flash` (`LICENSE-flash-attention`) and `fattn-sm70-d256*` |
| **Qwen3.8-27B** | https://huggingface.co/Qwen/Qwen3.8-27B | Apache-2.0 | Base 27B hybrid GDN + full-attention model |
| **Huihui-Qwen3.8-27B-abliterated GGUF** | https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF | Apache-2.0 | Default target (`--model huihui`), downloaded by `install.sh` |
| **Unsloth Qwen3.8-27B GGUF** | https://huggingface.co/unsloth/Qwen3.8-27B-GGUF | Apache-2.0 | Optional target (`--model qwen`), downloaded by `install.sh` |
| **ggml-org Qwen3.8-27B GGUF** | https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF | Apache-2.0 | Optional target (`--model q4km`), downloaded by `install.sh` |
| **Qwen3.8-27B-DSpine-b16 GGUF** | https://huggingface.co/aj9o9/Qwen3.8-27B-DSpine-b16-GGUF (from Tencent DSpine) | Apache-2.0 | Default speculative drafter (`--drafter dspine`), downloaded by `install.sh` |
