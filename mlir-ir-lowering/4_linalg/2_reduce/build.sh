#!/usr/bin/env bash
# Build the named linalg.reduce kernel and run it; also save the lowered form of the
# linalg.generic version so you can see the parallel+reduction loop nest it implies.
set -euo pipefail
rm -rf build
mkdir -p build

# (inspect) what the generic reduction lowers to — an outer parallel loop over rows
# and an inner reduction loop that accumulates into %out. Tensors, so bufferize first.
mlir-opt reduce_generic.mlir \
  -one-shot-bufferize="bufferize-function-boundaries" \
  -convert-linalg-to-loops \
  -o ./build/reduce_generic_loops.mlir

# (runnable) lower the named linalg.reduce to the llvm dialect and compile.
mlir-opt reduce.mlir \
  -convert-linalg-to-loops \
  -convert-scf-to-cf \
  -convert-cf-to-llvm \
  -convert-arith-to-llvm \
  -finalize-memref-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -o ./build/reduce_opt.mlir

mlir-translate ./build/reduce_opt.mlir -mlir-to-llvmir -o ./build/reduce.ll
llc -filetype=obj --relocation-model=pic ./build/reduce.ll -o ./build/reduce.o
clang -shared -fPIC ./build/reduce.o -o ./build/libreduce.dylib

python3 aot_main.py
