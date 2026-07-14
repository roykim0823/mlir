#!/usr/bin/env bash
# Build the affine matmul kernel into a shared library and run the driver.
set -euo pipefail
rm -rf build
mkdir -p build

# --- (inspect) show what --affine-parallelize does to the i/j loops ----------
# This output is for reading only; the runnable path below stays sequential.
mlir-opt matmul.mlir --affine-loop-normalize --affine-parallelize \
  -o ./build/matmul_parallel.mlir

# --- lower the (sequential) affine kernel all the way to the llvm dialect -----
#   -lower-affine : affine.for/load/store -> scf + arith + memref
mlir-opt matmul.mlir \
  -lower-affine \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-arith-to-llvm \
  --convert-index-to-llvm \
  --finalize-memref-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  -o ./build/matmul_opt.mlir

mlir-translate ./build/matmul_opt.mlir -mlir-to-llvmir -o ./build/matmul.ll
llc -filetype=obj --relocation-model=pic ./build/matmul.ll -o ./build/matmul.o
clang -shared -fPIC ./build/matmul.o -o ./build/libmatmul.dylib

python3 aot_main.py
