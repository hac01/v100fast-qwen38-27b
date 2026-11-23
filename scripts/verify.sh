#!/usr/bin/env bash
# Run full Stage 6 CUDA kernel verification suite (test-backend-ops).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="${1:-$REPO_DIR/engine/build/bin}"
TBO="$BIN_DIR/test-backend-ops"

[ -x "$TBO" ] || { echo "Missing $TBO — run scripts/build.sh first." >&2; exit 1; }

echo "=== 1/8 MUL_MAT (q4_K|q5_K|q6_K|iq4_xs|iq4_nl) ==="
"$TBO" test -b CUDA0 -o MUL_MAT -p "type_a=(q4_K|q5_K|q6_K|iq4_xs|iq4_nl)" | tail -n 6

echo "=== 2/8 GET_ROWS ==="
"$TBO" test -b CUDA0 -o GET_ROWS | tail -n 6

echo "=== 3/8 FLASH_ATTN_EXT hsk=256 (FA96=1, CAUSAL=1) ==="
LLAMA_SM70_FA96=1 TEST_FA_CAUSAL=1 "$TBO" test -b CUDA0 -o FLASH_ATTN_EXT -p "hsk=256" | tail -n 6

echo "=== 4/8 FLASH_ATTN_EXT hsk=256 (FA96=1, CAUSAL=2) ==="
LLAMA_SM70_FA96=1 TEST_FA_CAUSAL=2 "$TBO" test -b CUDA0 -o FLASH_ATTN_EXT -p "hsk=256" | tail -n 6

echo "=== 5/8 GATED_DELTA_NET (REPLAY=1, CHUNK=0) ==="
LLAMA_SM70_GDN_REPLAY=1 LLAMA_SM70_GDN_CHUNK=0 "$TBO" test -b CUDA0 -o GATED_DELTA_NET | tail -n 6

echo "=== 6/8 ARGMAX ==="
"$TBO" test -b CUDA0 -o ARGMAX | tail -n 5

echo "=== 7/8 SET_ROWS ==="
"$TBO" test -b CUDA0 -o SET_ROWS | tail -n 5

echo "=== 8/8 CONCAT ==="
"$TBO" test -b CUDA0 -o CONCAT | tail -n 5

echo ""
echo "ALL STAGE 6 KERNEL VERIFICATION GATES PASSED!"
