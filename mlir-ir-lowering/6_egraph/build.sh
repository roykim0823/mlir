#!/usr/bin/env bash
# Chapter 6: e-graph optimization, then emit MLIR and run it.
set -euo pipefail
rm -rf build
mkdir -p build

echo "=== concept demos (pure-Python e-graph) ==="
python3 optimize_demo.py

echo
echo "=== capstone: optimize an expression, emit MLIR ==="
python3 capstone.py          # writes build/orig.mlir and build/opt.mlir

# Compile both the original and the e-graph-optimized MLIR to shared libraries.
for name in orig opt; do
  mlir-opt "build/${name}.mlir" \
    -convert-arith-to-llvm \
    -convert-func-to-llvm \
    -reconcile-unrealized-casts \
    -o "build/${name}_opt.mlir"
  mlir-translate "build/${name}_opt.mlir" -mlir-to-llvmir -o "build/${name}.ll"
  llc -filetype=obj --relocation-model=pic "build/${name}.ll" -o "build/${name}.o"
  clang -shared -fPIC "build/${name}.o" -o "build/lib${name}.dylib"
done

echo
echo "=== run both compiled functions and compare ==="
python3 aot_main.py
