#!/usr/bin/env bash
set -euo pipefail
mkdir -p build

# 1. Lower the memref kernel to the LLVM dialect, then 2. translate to LLVM IR,
#    3. compile to an object, 4. link a shared library Python can dlopen.
mlir-opt array_add.mlir \
--convert-tensor-to-linalg \
--convert-linalg-to-loops \
--convert-scf-to-cf \
--convert-cf-to-llvm \
--convert-math-to-llvm \
--convert-arith-to-llvm \
--convert-func-to-llvm \
--convert-index-to-llvm \
--finalize-memref-to-llvm \
--reconcile-unrealized-casts \
-o ./build/array_add_opt.mlir

mlir-translate ./build/array_add_opt.mlir \
-mlir-to-llvmir \
-o ./build/array_add.ll

llc -filetype=obj ./build/array_add.ll -o ./build/array_add.o
clang -shared -fPIC ./build/array_add.o -o ./build/libarray_add.dylib

echo "python aot_main.py"
python aot_main.py

# 2. JIT execution, no use any intermediate files in build directory.
echo "python jit_main.py"
python jit_main.py