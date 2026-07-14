#!/usr/bin/env bash
set -euo pipefail
mkdir -p build

# The same lowering pipeline is used for every kernel here. The key pass is
# one-shot-bufferize with bufferize-function-boundaries=true, which turns the
# tensor signature into a memref signature so C/Python can call it.
lower() {  # $1 = basename (without .mlir)
  mlir-opt "$1.mlir" \
    -one-shot-bufferize="bufferize-function-boundaries=true" \
    -convert-linalg-to-loops \
    -convert-bufferization-to-memref \
    -finalize-memref-to-llvm \
    -convert-scf-to-cf \
    -convert-cf-to-llvm \
    -convert-arith-to-llvm \
    -convert-func-to-llvm \
    -reconcile-unrealized-casts \
    -o "./build/$1_opt.mlir"
  mlir-translate "./build/$1_opt.mlir" -mlir-to-llvmir -o "./build/$1_opt.ll"
  llc -filetype=obj --relocation-model=pic "./build/$1_opt.ll" -o "./build/$1_opt.o"
  clang -shared -fPIC "./build/$1_opt.o" -o "./build/$1.so"
}

# Lower & compile each identity kernel to a shared library (build/<name>.so):
#   identity_fill   — caller passes the output buffer; kernel fills it in place
#   identity_return — kernel allocates and returns the result; caller frees it
lower identity_fill
lower identity_return

# One driver runs both: identity_return uses the "callee allocates" convention,
# identity_fill uses destination-passing. See aot_main.py.
python aot_main.py
