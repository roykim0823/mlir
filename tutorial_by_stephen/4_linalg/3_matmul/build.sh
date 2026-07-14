#!/usr/bin/env bash
# Build the linalg.matmul kernel and run the driver.
set -euo pipefail
rm -rf build
mkdir -p build

# (inspect) the matmul written as an explicit linalg.generic, lowered TWO WAYS to
# contrast the loop-conversion passes (same nest, different dialect):
#   -convert-linalg-to-loops         -> scf.for    (generic structured loops)
#   -convert-linalg-to-affine-loops  -> affine.for (analyzable/tileable — Chapter 3's
#                                                    nest, verbatim)
mlir-opt matmul_generic.mlir -convert-linalg-to-loops        -o ./build/matmul_generic_scf.mlir
mlir-opt matmul_generic.mlir -convert-linalg-to-affine-loops -o ./build/matmul_generic_affine.mlir

# (runnable) lower to scf loops -> llvm dialect and compile.
mlir-opt matmul.mlir \
  -convert-linalg-to-loops \
  -convert-scf-to-cf \
  -convert-cf-to-llvm \
  -convert-arith-to-llvm \
  -finalize-memref-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -o ./build/matmul_opt.mlir

mlir-translate ./build/matmul_opt.mlir -mlir-to-llvmir -o ./build/matmul.ll
llc -filetype=obj --relocation-model=pic ./build/matmul.ll -o ./build/matmul.o
clang -shared -fPIC ./build/matmul.o -o ./build/libmatmul.dylib

python3 aot_main.py
