#!/usr/bin/env bash
rm -rf build
mkdir -p build

# Build the vectorized array_add kernel into a shared library that
# Python can dlopen and call through ctypes.
set -euo pipefail

mlir-opt add_vector_to_matrix.mlir \
  --convert-vector-to-llvm \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --convert-index-to-llvm \
  --finalize-memref-to-llvm \
  --reconcile-unrealized-casts \
  -o ./build/add_vector_to_matrix_opt.mlir

mlir-translate ./build/add_vector_to_matrix_opt.mlir \
  -mlir-to-llvmir \
  -o ./build/add_vector_to_matrix_opt.ll

llc -filetype=obj --relocation-model=pic ./build/add_vector_to_matrix_opt.ll \
  -o ./build/add_vector_to_matrix_opt.o
clang -shared -fPIC ./build/add_vector_to_matrix_opt.o -o ./build/add_vector_to_matrix_opt.so

python3 aot_main.py
