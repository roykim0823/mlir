#!/usr/bin/env bash
# Run each affine optimization pass and save the "after" IR into build/, so you
# can diff it against the original source file. Then build + run the manual
# interchange/skewing demo to prove those by-hand rewrites preserve results.
set -euo pipefail
rm -rf build
mkdir -p build

echo "== loop-invariant code motion =="
mlir-opt licm.mlir -affine-loop-invariant-code-motion -o ./build/licm_after.mlir

echo "== loop tiling (8x8) =="
mlir-opt tiling.mlir -affine-loop-tile="tile-size=8" -o ./build/tiling_after.mlir

echo "== loop unrolling (factor 4) =="
mlir-opt unroll.mlir -affine-loop-unroll="unroll-factor=4" -o ./build/unroll_after.mlir

echo "== loop fusion =="
mlir-opt fusion.mlir -affine-loop-fusion -o ./build/fusion_after.mlir

echo "== loop coalescing =="
mlir-opt coalescing.mlir -affine-loop-coalescing -o ./build/coalescing_after.mlir

echo "== loop normalization =="
mlir-opt normalize.mlir -affine-loop-normalize -o ./build/normalize_after.mlir

echo "== scalar replacement =="
mlir-opt scalrep.mlir -affine-scalrep -o ./build/scalrep_after.mlir

echo "== super-vectorization (width 8) =="
mlir-opt vectorize.mlir -affine-super-vectorize="virtual-vector-size=8" \
  -o ./build/vectorize_after.mlir

echo
echo "Compare each <name>.mlir with build/<name>_after.mlir, e.g.:"
echo "    diff licm.mlir build/licm_after.mlir"

echo
echo "== manual transforms (interchange, skewing, wavefront): compile & check =="
# Compile each hand-written file to its own shared library.
lower_manual() {  # $1 = basename (without .mlir)
  mlir-opt "$1.mlir" \
    -lower-affine -convert-scf-to-cf -convert-cf-to-llvm \
    -convert-arith-to-llvm -convert-index-to-llvm -finalize-memref-to-llvm \
    -convert-func-to-llvm -reconcile-unrealized-casts \
    -o "./build/$1_opt.mlir"
  mlir-translate "./build/$1_opt.mlir" -mlir-to-llvmir -o "./build/$1.ll"
  llc -filetype=obj --relocation-model=pic "./build/$1.ll" -o "./build/$1.o"
  clang -shared -fPIC "./build/$1.o" -o "./build/lib$1.dylib"
}
lower_manual interchange_manual
lower_manual skewing_manual
python3 aot_main.py
