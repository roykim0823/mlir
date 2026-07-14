#!/usr/bin/env bash
# Build the MLIR softmax kernel into a shared library and run the driver.
set -euo pipefail
rm -rf build
mkdir -p build

mlir-opt softmax.mlir \
  -convert-scf-to-cf \
  -convert-cf-to-llvm \
  -convert-math-to-llvm \
  -convert-arith-to-llvm \
  -convert-index-to-llvm \
  -finalize-memref-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -o ./build/softmax_opt.mlir

mlir-translate ./build/softmax_opt.mlir -mlir-to-llvmir -o ./build/softmax.ll
llc -filetype=obj --relocation-model=pic ./build/softmax.ll -o ./build/softmax.o
clang -shared -fPIC ./build/softmax.o -o ./build/libsoftmax.dylib   # libm auto-linked

python3 aot_main.py
