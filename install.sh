#!/usr/bin/env bash
# One-command setup for Qwen3.8-27B + DSpine on NVIDIA Tesla V100 (Volta sm_70)
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/hac01/v100fast-qwen38-27b/main/install.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/hac01/v100fast-qwen38-27b/main/install.sh | bash -s -- --model qwen --start
set -euo pipefail

REPO_URL="${V100FAST_REPO_URL:-https://github.com/hac01/v100fast-qwen38-27b.git}"
REPO_BRANCH="${V100FAST_REPO_BRANCH:-main}"
INSTALL_DIR="${V100FAST_DIR:-$HOME/v100fast-qwen38-27b}"
MODEL_CHOICE="${V100FAST_MODEL:-huihui}"       # huihui | qwen | q4km | none
DRAFTER_CHOICE="${V100FAST_DRAFTER:-dspine}"   # dspine | none
CTX="${V100FAST_CTX:-229376}"
CRAM="${V100FAST_CACHE_RAM:-auto}"
POWER_LIMIT="${V100FAST_POWER_LIMIT:-220}"
HOST="${V100FAST_HOST:-127.0.0.1}"
PORT="${V100FAST_PORT:-8080}"
RUN_VERIFY=0
RUN_START=0
SKIP_BUILD=0

usage() {
  cat <<'EOF'
v100fast-qwen38-27b installer — Volta (sm_70) Qwen3.8-27B + DSpine

Options:
  --dir DIR            Install directory (default: ~/v100fast-qwen38-27b)
  --model NAME         Target model: huihui (default, UD-Q4_K_XL, 17.4 GB)
                                     qwen   (unsloth/Qwen3.8-27B UD-Q4_K_XL, 17.4 GB)
                                     q4km   (ggml-org/Qwen3.8-27B Q4_K_M, 19.0 GB)
                                     none   (skip target model download)
  --drafter NAME       Draft model:  dspine (default, DSpine-b16 Q4_K_M, 1.2 GB)
                                     none   (skip draft model download)
  --ctx N              Context size (default: 229376 for V100-32GB)
  --cache-ram MB       RAM prompt cache in MiB (default: auto from host RAM)
  --power-limit W      V100 power cap in Watts (default: 220; 0 to skip)
  --host ADDR          Bind address for serve.sh (default: 127.0.0.1)
  --port PORT          Port for serve.sh (default: 8080)
  --skip-build         Skip llama.cpp clone/patch/build (reuse existing binary)
  --verify             Run CUDA test-backend-ops verification suite after build
  --start              Start the server in the background once setup completes
  -h, --help           Show this help
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dir)         INSTALL_DIR="$2"; shift 2 ;;
    --model)       MODEL_CHOICE="$2"; shift 2 ;;
    --drafter)     DRAFTER_CHOICE="$2"; shift 2 ;;
    --ctx)         CTX="$2"; shift 2 ;;
    --cache-ram)   CRAM="$2"; shift 2 ;;
    --power-limit) POWER_LIMIT="$2"; shift 2 ;;
    --host)        HOST="$2"; shift 2 ;;
    --port)        PORT="$2"; shift 2 ;;
    --skip-build)  SKIP_BUILD=1; shift ;;
    --verify)      RUN_VERIFY=1; shift ;;
    --start)       RUN_START=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *)             echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

log()  { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; exit 1; }

check_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1. Please install it first."
}

log "Checking system prerequisites"
for c in git cmake curl; do check_cmd "$c"; done
if ! command -v g++ >/dev/null 2>&1 && ! command -v c++ >/dev/null 2>&1; then
  die "No C++ compiler found (install build-essential / g++)."
fi

# Ensure CUDA toolkit is on PATH if installed in standard location
if ! command -v nvcc >/dev/null 2>&1 && [ -x /usr/local/cuda/bin/nvcc ]; then
  export PATH="/usr/local/cuda/bin:$PATH"
fi
check_cmd nvcc
check_cmd nvidia-smi

GPU_INFO=$(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader | head -n 1)
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' ')
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
echo "Detected GPU: $GPU_INFO"
echo "CUDA toolkit: $(nvcc --version | grep -oE 'release [0-9.]+' || echo ok)"

if [ "$GPU_CC" != "7.0" ]; then
  warn "Detected compute capability $GPU_CC (this repo is tuned specifically for Volta sm_70 / Tesla V100)."
fi
if [ "$GPU_MEM_MIB" -lt 28000 ] && [ "$CTX" -gt 32768 ]; then
  warn "GPU has ${GPU_MEM_MIB} MiB VRAM (< 32 GB). Lowering default CTX from $CTX to 32768."
  CTX=32768
fi

# Auto-size RAM prompt cache from host RAM
if [ "$CRAM" = "auto" ]; then
  if [ -r /proc/meminfo ]; then
    RAM_MB=$(awk '/MemTotal:/ {printf "%d", $2/1024}' /proc/meminfo)
  else
    RAM_MB=16384
  fi
  if [ "$RAM_MB" -ge 20000 ]; then
    CRAM=15360
  elif [ "$RAM_MB" -ge 14000 ]; then
    CRAM=8192
  else
    CRAM=2048
  fi
  echo "Host RAM: ${RAM_MB} MiB -> auto --cache-ram ${CRAM} MiB"
fi

# Locate or clone this recipe repo
SCRIPT_DIR=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
  CANDIDATE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [ -f "$CANDIDATE/patches/v100fast-stage9.patch" ] && [ -d "$CANDIDATE/overlay/ggml" ]; then
    SCRIPT_DIR="$CANDIDATE"
  fi
fi

if [ -n "$SCRIPT_DIR" ]; then
  REPO_DIR="$SCRIPT_DIR"
  log "Using local recipe checkout: $REPO_DIR"
else
  REPO_DIR="$INSTALL_DIR"
  if [ -d "$REPO_DIR/.git" ]; then
    log "Updating existing recipe checkout in $REPO_DIR"
    git -C "$REPO_DIR" pull --ff-only || warn "Could not fast-forward $REPO_DIR; using existing tree."
  else
    log "Cloning $REPO_URL into $REPO_DIR"
    git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
  fi
fi

mkdir -p "$REPO_DIR/models"

# 1. Build v100fast engine
if [ "$SKIP_BUILD" -eq 0 ]; then
  log "Cloning pinned llama.cpp (DSpine base) + applying Stage 9 V100 patches & building for sm_70"
  bash "$REPO_DIR/scripts/build.sh" "$REPO_DIR/engine"
else
  log "Skipping engine build (--skip-build)"
fi

# 2. Download models (resumable curl)
download_file() {
  local url="$1"
  local dest="$2"
  if [ -f "$dest" ]; then
    echo "Resuming/verifying existing file: $dest"
  else
    echo "Downloading: $url -> $dest"
  fi
  curl -fSL -C - --retry 5 --retry-delay 3 -o "$dest" "$url"
}

TARGET_FILE=""
case "$MODEL_CHOICE" in
  huihui)
    TARGET_FILE="Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf"
    TARGET_URL="https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/resolve/main/${TARGET_FILE}"
    log "Downloading target model: $TARGET_FILE (~17.4 GB)"
    download_file "$TARGET_URL" "$REPO_DIR/models/$TARGET_FILE"
    ;;
  qwen)
    TARGET_FILE="Qwen3.8-27B-UD-Q4_K_XL.gguf"
    TARGET_URL="https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/${TARGET_FILE}"
    log "Downloading target model: $TARGET_FILE (~17.4 GB)"
    download_file "$TARGET_URL" "$REPO_DIR/models/$TARGET_FILE"
    ;;
  q4km)
    TARGET_FILE="Qwen3.8-27B-Q4_K_M.gguf"
    TARGET_URL="https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/main/${TARGET_FILE}"
    log "Downloading target model: $TARGET_FILE (~19.0 GB)"
    download_file "$TARGET_URL" "$REPO_DIR/models/$TARGET_FILE"
    ;;
  none)
    log "Skipping target model download (--model none)"
    ;;
  *)
    die "Unknown --model '$MODEL_CHOICE' (expected: huihui, qwen, q4km, none)"
    ;;
esac

DRAFT_FILE=""
case "$DRAFTER_CHOICE" in
  dspine)
    DRAFT_FILE="Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf"
    DRAFT_URL="https://huggingface.co/aj9o9/Qwen3.8-27B-DSpine-b16-GGUF/resolve/main/${DRAFT_FILE}"
    log "Downloading DSpine drafter: $DRAFT_FILE (~1.2 GB)"
    download_file "$DRAFT_URL" "$REPO_DIR/models/$DRAFT_FILE"
    ;;
  none)
    log "Skipping drafter download (--drafter none)"
    ;;
  *)
    die "Unknown --drafter '$DRAFTER_CHOICE' (expected: dspine, none)"
    ;;
esac

# 3. Write convenience config file for serve.sh
cat > "$REPO_DIR/.v100fast.env" <<EOF
MODEL_FILE="${TARGET_FILE:-Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf}"
DRAFT_FILE="${DRAFT_FILE:-Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf}"
CTX="${CTX}"
CRAM="${CRAM}"
PL="${POWER_LIMIT}"
HOST="${HOST}"
PORT="${PORT}"
EOF

# 4. Optional verification
if [ "$RUN_VERIFY" -eq 1 ]; then
  log "Running CUDA test-backend-ops verification suite"
  bash "$REPO_DIR/scripts/verify.sh" "$REPO_DIR/engine/build/bin"
fi

# 5. Optional start
if [ "$RUN_START" -eq 1 ]; then
  log "Starting v100fast server on http://${HOST}:${PORT}"
  bash "$REPO_DIR/scripts/serve.sh" --background
fi

log "Setup complete!"
cat <<EOF
Next steps:
  Start server:   $REPO_DIR/scripts/serve.sh
  Background:     $REPO_DIR/scripts/serve.sh --background
  Verify kernels: $REPO_DIR/scripts/verify.sh
  Benchmark:      python3 $REPO_DIR/scripts/bench.py --url http://${HOST}:${PORT}

OpenAI-compatible endpoint:
  Base URL: http://${HOST}:${PORT}/v1
  Model:    qwen3.8-27b
EOF
