#!/usr/bin/env bash
# Inspect-only: these files demonstrate affine maps, not a runnable program.
# We parse/verify them and save a couple of lowered/folded versions into build/.
set -euo pipefail
rm -rf build
mkdir -p build

echo "== affine_apply — parse + constant-fold =="
mlir-opt affine_apply.mlir              -o ./build/affine_apply_verified.mlir
mlir-opt affine_apply.mlir -canonicalize -o ./build/affine_apply_folded.mlir

echo "== tiled_loop — parse + lower-affine (see the scf/arith expansion) =="
mlir-opt tiled_loop.mlir               -o ./build/tiled_loop_verified.mlir
mlir-opt tiled_loop.mlir -lower-affine -o ./build/tiled_loop_lowered.mlir

echo "Done. Read the *_folded / *_lowered files in build/ to see the transforms."
