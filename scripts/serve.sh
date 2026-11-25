#!/usr/bin/env bash
# Launch Qwen3.8-27B + DSpine on Tesla V100-PCIE-32GB with all verified Stage 9 optimizations.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$REPO_DIR/.v100fast.env" ]; then
  # shellcheck disable=SC1091
  source "$REPO_DIR/.v100fast.env"
fi

MODEL_FILE="${MODEL_FILE:-Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf}"
DRAFT_FILE="${DRAFT_FILE:-Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf}"
CTX="${CTX:-229376}"
UB="${UB:-2048}"
CRAM="${CRAM:-15360}"
CKPT="${CKPT:-8}"
NMAX="${NMAX:-15}"
PL="${PL:-220}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
ALIAS="${ALIAS:-qwen3.8-27b}"
API_KEY_FILE="${API_KEY_FILE:-}"
BACKGROUND=0
LOG_FILE="${LOG_FILE:-$REPO_DIR/v100fast-server.log}"

while [ $# -gt 0 ]; do
  case "$1" in
    --background|-d) BACKGROUND=1; shift ;;
    --model-path)    MODEL_FILE="$2"; shift 2 ;;
    --draft-path)    DRAFT_FILE="$2"; shift 2 ;;
    --ctx)           CTX="$2"; shift 2 ;;
    --cache-ram)     CRAM="$2"; shift 2 ;;
    --host)          HOST="$2"; shift 2 ;;
    --port)          PORT="$2"; shift 2 ;;
    --alias)         ALIAS="$2"; shift 2 ;;
    --api-key-file)  API_KEY_FILE="$2"; shift 2 ;;
    --power-limit)   PL="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--background] [--host 127.0.0.1] [--port 8080] [--ctx 229376] [--cache-ram 15360] [--api-key-file FILE]"
      exit 0 ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

BIN="$REPO_DIR/engine/build/bin/llama-server"
[ -x "$BIN" ] || { echo "Missing $BIN — run scripts/build.sh first." >&2; exit 1; }

MODEL_PATH="$MODEL_FILE"
[ -f "$MODEL_PATH" ] || MODEL_PATH="$REPO_DIR/models/$MODEL_FILE"
[ -f "$MODEL_PATH" ] || { echo "Target model not found: $MODEL_PATH" >&2; exit 1; }

DRAFT_ARGS=()
if [ -n "$DRAFT_FILE" ] && [ "$DRAFT_FILE" != "none" ]; then
  DRAFT_PATH="$DRAFT_FILE"
  [ -f "$DRAFT_PATH" ] || DRAFT_PATH="$REPO_DIR/models/$DRAFT_FILE"
  if [ -f "$DRAFT_PATH" ]; then
    DRAFT_ARGS=(
      -md "$DRAFT_PATH"
      --spec-type draft-dflash
      --spec-draft-n-max "$NMAX"
      --spec-draft-ngl 999
      --spec-draft-type-k f16
      --spec-draft-type-v f16
      --spec-draft-sampling probabilistic
    )
  else
    echo "[warn] Drafter not found at $DRAFT_PATH; launching without speculative drafting." >&2
  fi
fi

# Apply optional V100 power cap (recommended 220 W on PCIe passively cooled cards)
if [ "${PL:-0}" -gt 0 ] && command -v nvidia-smi >/dev/null 2>&1; then
  if sudo -n nvidia-smi -pm 1 >/dev/null 2>&1; then
    sudo -n nvidia-smi -pl "$PL" >/dev/null 2>&1 || true
  fi
fi

# Stage 9 Volta sm_70 kernel & dynamic-block flags (all strictly lossless; override any to 0 to disable)
export LLAMA_SM70_FA96="${LLAMA_SM70_FA96:-1}"
export LLAMA_SM70_FA96_LDG="${LLAMA_SM70_FA96_LDG:-1}"
export LLAMA_SM70_GDN_CHUNK="${LLAMA_SM70_GDN_CHUNK:-1}"
export LLAMA_SM70_GDN_REPLAY="${LLAMA_SM70_GDN_REPLAY:-1}"
export LLAMA_SM70_BF16_CACHE="${LLAMA_SM70_BF16_CACHE:-1}"
export LLAMA_SM70_ASYNC_SET="${LLAMA_SM70_ASYNC_SET:-1}"
export LLAMA_SM70_SKINNY_3CTA="${LLAMA_SM70_SKINNY_3CTA:-1}"
export LLAMA_SM70_SKINNY_GNT_3CTA="${LLAMA_SM70_SKINNY_GNT_3CTA:-1}"
export LLAMA_SM70_SKINNY_WAVEFIX="${LLAMA_SM70_SKINNY_WAVEFIX:-1}"
export LLAMA_SM70_SKINNY_N5120_2CTA="${LLAMA_SM70_SKINNY_N5120_2CTA:-1}"
export LLAMA_SM70_DYN_BLOCK="${LLAMA_SM70_DYN_BLOCK:-1}"

EXTRA_ARGS=()
if [ -n "$API_KEY_FILE" ]; then
  EXTRA_ARGS+=(--api-key-file "$API_KEY_FILE")
fi

CMD=(
  "$BIN"
  -m "$MODEL_PATH"
  -ngl 999 -c "$CTX" -np 1 -fa on
  -ctk q8_0 -ctv q8_0
  -b 2048 -ub "$UB"
  --jinja --metrics --fit off
  --host "$HOST" --port "$PORT"
  --alias "$ALIAS"
  --backend-sampling
  --cache-ram "$CRAM" --ctx-checkpoints "$CKPT"
  "${DRAFT_ARGS[@]}"
  "${EXTRA_ARGS[@]}"
)

if [ "$BACKGROUND" -eq 1 ]; then
  nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
  PID=$!
  echo "Started llama-server (PID=$PID, log=$LOG_FILE)"
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/health" | grep -q ok; then
      echo "Server healthy on http://${HOST}:${PORT}/v1"
      exit 0
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
      echo "Server exited unexpectedly. Last 30 lines of $LOG_FILE:" >&2
      tail -n 30 "$LOG_FILE" >&2
      exit 1
    fi
    sleep 2
  done
  echo "Timed out waiting for /health on port $PORT." >&2
  exit 1
else
  exec "${CMD[@]}"
fi
