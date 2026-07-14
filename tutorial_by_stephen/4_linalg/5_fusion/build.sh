#!/usr/bin/env bash
# Fuse the two elementwise ops into one, save the fused IR, then compile & run it.
set -euo pipefail
rm -rf build
mkdir -p build

# (inspect) fuse the two named ops into a single linalg.generic.
#   --canonicalize                 general cleanup
#   --linalg-fuse-elementwise-ops  the actual fusion
#   --cse                          drop redundant ops
#   --linalg-generalize-named-ops  named ops (add/mul) -> linalg.generic
mlir-opt separate_ops.mlir \
  --canonicalize \
  --linalg-fuse-elementwise-ops \
  --cse \
  --linalg-generalize-named-ops \
  --linalg-fuse-elementwise-ops \
  -o ./build/fused_ops.mlir

echo "Wrote build/fused_ops.mlir — the two ops are now one linalg.generic."
echo "Compare:  diff separate_ops.mlir build/fused_ops.mlir"

# (runnable) the fused op is on tensors, so bufferize it, turn the returned tensor
# into an in-place out-param, then lower to loops -> llvm and compile.
mlir-opt ./build/fused_ops.mlir \
  -one-shot-bufferize="bufferize-function-boundaries" \
  -buffer-results-to-out-params \
  -convert-linalg-to-loops \
  -convert-scf-to-cf \
  -convert-cf-to-llvm \
  -convert-arith-to-llvm \
  -finalize-memref-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -o ./build/addmul_opt.mlir

mlir-translate ./build/addmul_opt.mlir -mlir-to-llvmir -o ./build/addmul.ll
llc -filetype=obj --relocation-model=pic ./build/addmul.ll -o ./build/addmul.o
clang -shared -fPIC ./build/addmul.o -o ./build/libfused.dylib

python3 aot_main.py
