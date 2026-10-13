#!/usr/bin/env bash
# Inspect-only: lower the tensor matmul to affine loops and tile them (5x5x5).
set -euo pipefail
rm -rf build
mkdir -p build

#   --convert-tensor-to-linalg                tensor ops -> linalg
#   --linalg-generalize-named-ops             linalg.matmul -> linalg.generic
#   --one-shot-bufferize=...                  tensors -> memrefs
#   --buffer-deallocation-pipeline            insert dealloc for temp buffers
#   --convert-bufferization-to-memref         finalize bufferization ops
#   --convert-linalg-to-affine-loops          linalg -> affine.for nest
#   --affine-loop-tile="tile-size=5"          break the nest into 5-wide tiles
mlir-opt matmul_tile.mlir \
  --convert-tensor-to-linalg \
  --linalg-generalize-named-ops \
  --one-shot-bufferize="bufferize-function-boundaries" \
  --buffer-deallocation-pipeline \
  --convert-bufferization-to-memref \
  --convert-linalg-to-affine-loops \
  --affine-loop-tile="tile-size=5" \
  --canonicalize \
  --cse \
  -o ./build/matmul_tiled.mlir

echo "Wrote build/matmul_tiled.mlir — the 3-loop matmul is now a 6-deep tiled nest."
echo "Look for the outer 'affine.for ... step 5' loops."
