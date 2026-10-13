#!/usr/bin/env bash
# Build the broadcast example and run the driver.
set -euo pipefail
rm -rf build
mkdir -p build

mlir-opt add_vec_to_mat.mlir \
  -convert-linalg-to-loops \
  -convert-scf-to-cf \
  -convert-cf-to-llvm \
  -convert-arith-to-llvm \
  -finalize-memref-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -o ./build/add_vec_to_mat_opt.mlir

mlir-translate ./build/add_vec_to_mat_opt.mlir -mlir-to-llvmir -o ./build/add_vec_to_mat.ll
llc -filetype=obj --relocation-model=pic ./build/add_vec_to_mat.ll -o ./build/add_vec_to_mat.o
clang -shared -fPIC ./build/add_vec_to_mat.o -o ./build/libbcast.dylib

python3 aot_main.py
