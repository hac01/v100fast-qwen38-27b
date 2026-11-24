#!/usr/bin/env bash
# Run full Stage 8 CUDA kernel verification suite (test-backend-ops).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="${1:-$REPO_DIR/engine/build/bin}"
TBO="$BIN_DIR/test-backend-ops"

[ -x "$TBO" ] || { echo "Missing $TBO — run scripts/build.sh first." >&2; exit 1; }

echo "=== 1/12 MUL_MAT (q4_K|q5_K|q6_K|iq4_xs|iq4_nl) ==="
"$TBO" test -b CUDA0 -o MUL_MAT -p "type_a=(q4_K|q5_K|q6_K|iq4_xs|iq4_nl)" | tail -n 6

echo "=== 2/12 MUL_MAT (q8_0, Stage 7E-v2/8E-2 q8-small) ==="
"$TBO" test -b CUDA0 -o MUL_MAT -p "type_a=q8_0," | tail -n 6

echo "=== 3/12 MUL_MAT (bf16 -> Volta FP16 Tensor Core, Stage 8B) ==="
"$TBO" test -b CUDA0 -o MUL_MAT -p "type_a=bf16," | tail -n 6

echo "=== 4/12 RMS_NORM (Stage 8D fused ADD+RMS_NORM+MUL) ==="
"$TBO" test -b CUDA0 -o RMS_NORM | tail -n 6

echo "=== 5/12 ADD (Stage 8D) ==="
"$TBO" test -b CUDA0 -o ADD | tail -n 6

echo "=== 6/12 GET_ROWS ==="
"$TBO" test -b CUDA0 -o GET_ROWS | tail -n 6

echo "=== 7/12 FLASH_ATTN_EXT hsk=256 (FA96=1, CAUSAL=1) ==="
LLAMA_SM70_FA96=1 TEST_FA_CAUSAL=1 "$TBO" test -b CUDA0 -o FLASH_ATTN_EXT -p "hsk=256" | tail -n 6

echo "=== 8/12 FLASH_ATTN_EXT hsk=256 (FA96=1, CAUSAL=2) ==="
LLAMA_SM70_FA96=1 TEST_FA_CAUSAL=2 "$TBO" test -b CUDA0 -o FLASH_ATTN_EXT -p "hsk=256" | tail -n 6

echo "=== 9/12 GATED_DELTA_NET (REPLAY=1, CHUNK=0) ==="
LLAMA_SM70_GDN_REPLAY=1 LLAMA_SM70_GDN_CHUNK=0 "$TBO" test -b CUDA0 -o GATED_DELTA_NET | tail -n 6

echo "=== 10/12 ARGMAX ==="
"$TBO" test -b CUDA0 -o ARGMAX | tail -n 5

echo "=== 11/12 SET_ROWS ==="
"$TBO" test -b CUDA0 -o SET_ROWS | tail -n 5

echo "=== 12/12 CONCAT ==="
"$TBO" test -b CUDA0 -o CONCAT | tail -n 5

echo ""
echo "ALL STAGE 8 KERNEL VERIFICATION GATES PASSED!"
