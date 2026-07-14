#!/usr/bin/env bash
# Build the affine conv2d kernel into a shared library and run the driver.
set -euo pipefail
rm -rf build
mkdir -p build

# --- (inspect) what the affine optimizer can do with this nest ---------------
# --affine-parallelize turns the independent output loops into affine.parallel.
mlir-opt conv2d.mlir --affine-loop-normalize --affine-parallelize \
  -o ./build/conv2d_parallel.mlir

# --- lower the affine kernel to the llvm dialect and compile ------------------
mlir-opt conv2d.mlir \
  --affine-loop-normalize \
  -lower-affine \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-arith-to-llvm \
  --convert-index-to-llvm \
  --finalize-memref-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  -o ./build/conv2d_opt.mlir

mlir-translate ./build/conv2d_opt.mlir -mlir-to-llvmir -o ./build/conv2d.ll
llc -filetype=obj --relocation-model=pic ./build/conv2d.ll -o ./build/conv2d.o
clang -shared -fPIC ./build/conv2d.o -o ./build/libconv2d.dylib

python3 aot_main.py
