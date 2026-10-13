#!/usr/bin/env bash
# Build (and run) ONLY the vectorized array_add kernel: lower it to a shared
# library Python can dlopen and call through ctypes, then run the driver.
# To build every example in this directory at once, use ./build.sh instead.
set -euo pipefail
mkdir -p build   # no `rm -rf build` here, so build.sh can compose this step

mlir-opt array_add_vec.mlir \
  --convert-vector-to-llvm \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --convert-index-to-llvm \
  --finalize-memref-to-llvm \
  --reconcile-unrealized-casts \
  -o ./build/array_add_vec_opt.mlir

mlir-translate ./build/array_add_vec_opt.mlir \
  -mlir-to-llvmir \
  -o ./build/array_add_vec.ll

llc -filetype=obj --relocation-model=pic ./build/array_add_vec.ll \
  -o ./build/array_add_vec.o
clang -shared -fPIC ./build/array_add_vec.o -o ./build/libarray_add_vec.dylib

python3 aot_main.py
