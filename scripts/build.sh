#!/usr/bin/env bash
# Clone pinned llama.cpp (Ajay9o9 DSpine base), apply v100fast Stage 7 patch + sm_70 overlay, and build.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENGINE_DIR="${1:-$REPO_DIR/engine}"
SRC_DIR="$ENGINE_DIR/src"
BUILD_DIR="$ENGINE_DIR/build"

BASE_REPO="https://github.com/Ajay9o9/llama.cpp.git"
BASE_COMMIT="4703b7d6ffc3a5978f9b6956ad2420052e172f8b"

if ! command -v nvcc >/dev/null 2>&1 && [ -x /usr/local/cuda/bin/nvcc ]; then
  export PATH="/usr/local/cuda/bin:$PATH"
fi

mkdir -p "$ENGINE_DIR"

if [ ! -d "$SRC_DIR/.git" ]; then
  echo "==> Fetching pinned base commit $BASE_COMMIT from $BASE_REPO"
  mkdir -p "$SRC_DIR"
  git -C "$SRC_DIR" init -q
  git -C "$SRC_DIR" remote add origin "$BASE_REPO"
  git -C "$SRC_DIR" fetch --depth 1 origin "$BASE_COMMIT"
  git -C "$SRC_DIR" checkout -q FETCH_HEAD
else
  echo "==> Resetting existing source checkout in $SRC_DIR to $BASE_COMMIT"
  git -C "$SRC_DIR" checkout -q -- .
  git -C "$SRC_DIR" clean -fdq
  if [ "$(git -C "$SRC_DIR" rev-parse HEAD)" != "$BASE_COMMIT" ]; then
    git -C "$SRC_DIR" fetch --depth 1 origin "$BASE_COMMIT"
    git -C "$SRC_DIR" checkout -q FETCH_HEAD
  fi
fi

echo "==> Applying Stage 7 tracked-file patch"
git -C "$SRC_DIR" apply --check "$REPO_DIR/patches/v100fast-stage7.patch"
git -C "$SRC_DIR" apply "$REPO_DIR/patches/v100fast-stage7.patch"

echo "==> Copying sm_70 CUDA kernel overlay"
cp -a "$REPO_DIR/overlay/." "$SRC_DIR/"

echo "==> Configuring CMake for NVIDIA Volta (sm_70)"
cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=OFF \
  -DGGML_CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;f8_e4m3-f8_e4m3" \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_MTMD=OFF \
  -DLLAMA_BUILD_WEBUI=OFF \
  -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_OPENSSL=OFF

NPROC="$(nproc 2>/dev/null || echo 4)"
echo "==> Building llama-server and test-backend-ops with -j$NPROC"
cmake --build "$BUILD_DIR" --target llama-server test-backend-ops -j"$NPROC"

echo "==> Build complete:"
ls -lh "$BUILD_DIR/bin/llama-server" "$BUILD_DIR/bin/test-backend-ops" "$BUILD_DIR/bin/libggml-cuda.so"
