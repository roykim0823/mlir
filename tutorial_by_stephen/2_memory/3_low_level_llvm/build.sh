#!/usr/bin/env bash
# Build EVERY example in this directory into ./build/.
#
# The four concept files are inspect-only: we lower them with mlir-opt (and, where
# the result is already in the llvm dialect, translate to textual LLVM IR) and
# save the output so you can read how the IR changed. Only array_add_vec is
# compiled all the way to a runnable shared library + executed — that step is
# delegated to ./build_array_add_vec.sh.
set -euo pipefail
rm -rf build
mkdir -p build

echo "== structs_arrays — llvm-dialect structs & arrays =="
mlir-opt structs_arrays.mlir -reconcile-unrealized-casts \
  -o build/structs_arrays_opt.mlir
# already in the llvm dialect, so it translates straight to LLVM IR
mlir-translate structs_arrays.mlir --mlir-to-llvmir \
  -o build/structs_arrays.ll

echo "== vectors — SIMD vector dialect =="
mlir-opt vectors.mlir \
  -convert-vector-to-llvm \
  -convert-arith-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -o build/vectors_opt.mlir
mlir-translate build/vectors_opt.mlir --mlir-to-llvmir \
  -o build/vectors.ll

echo "== bufferization — tensor -> memref (one-shot) =="
# stays at the memref level on purpose; this shows the tensor->memref rewrite
mlir-opt bufferization.mlir \
  -one-shot-bufferize="bufferize-function-boundaries" \
  -o build/bufferization_opt.mlir

echo "== c_interface — emit_c_interface wrapper =="
mlir-opt c_interface.mlir \
  -finalize-memref-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -o build/c_interface_opt.mlir
mlir-translate build/c_interface_opt.mlir --mlir-to-llvmir \
  -o build/c_interface.ll

echo "== array_add_vec — runnable SIMD kernel =="
bash build_array_add_vec.sh
