#!/usr/bin/env bash
# Build the MLIR dense+ReLU layer into a shared library and run the driver.
set -euo pipefail
rm -rf build
mkdir -p build

mlir-opt dense_relu.mlir \
  -convert-linalg-to-loops \
  -convert-scf-to-cf \
  -convert-cf-to-llvm \
  -convert-arith-to-llvm \
  -convert-index-to-llvm \
  -finalize-memref-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -o ./build/dense_relu_opt.mlir

mlir-translate ./build/dense_relu_opt.mlir -mlir-to-llvmir -o ./build/dense_relu.ll
llc -filetype=obj --relocation-model=pic ./build/dense_relu.ll -o ./build/dense_relu.o
clang -shared -fPIC ./build/dense_relu.o -o ./build/libdense.dylib

python3 aot_main.py
