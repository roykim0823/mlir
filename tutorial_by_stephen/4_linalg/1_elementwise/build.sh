#!/usr/bin/env bash
# Build the named-op elementwise add and run it; also save the lowered form of the
# linalg.generic version so you can see what the named op expands to.
set -euo pipefail
rm -rf build
mkdir -p build

# (inspect) what linalg.generic lowers to — a plain scf loop nest.
# NOTE: -convert-linalg-to-loops only lowers linalg ops on memrefs, not tensors, so
# we must bufferize first. Without the bufferize step this pass is a no-op (no loops).
mlir-opt generic_add.mlir \
  -one-shot-bufferize="bufferize-function-boundaries" \
  -convert-linalg-to-loops \
  -o ./build/generic_add_loops.mlir

# (inspect) the linalg.map shorthand lowers to the *identical* loop nest — same
# bufferize-first requirement, since it too works on tensors.
mlir-opt map_add.mlir \
  -one-shot-bufferize="bufferize-function-boundaries" \
  -convert-linalg-to-loops \
  -o ./build/map_add_loops.mlir

# (runnable) lower the named linalg.add to the llvm dialect and compile.
mlir-opt add.mlir \
  -convert-linalg-to-loops \
  -convert-scf-to-cf \
  -convert-cf-to-llvm \
  -convert-arith-to-llvm \
  -finalize-memref-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -o ./build/add_opt.mlir

mlir-translate ./build/add_opt.mlir -mlir-to-llvmir -o ./build/add.ll
llc -filetype=obj --relocation-model=pic ./build/add.ll -o ./build/add.o
clang -shared -fPIC ./build/add.o -o ./build/libadd.dylib

python3 aot_main.py
