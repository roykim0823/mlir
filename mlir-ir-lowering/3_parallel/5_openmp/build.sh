#!/usr/bin/env bash
# Build the OpenMP example into a native executable and run it.
#
# OpenMP needs a runtime. llvm@20 does not bundle one, so we link Homebrew's
# libomp (brew install libomp). The -Wl,-rpath bakes its location into the binary
# so it loads at runtime without setting DYLD_LIBRARY_PATH.
set -euo pipefail
rm -rf build
mkdir -p build

OMP_PREFIX="$(brew --prefix libomp)"

# --- C reference: the same doubling kernel with OpenMP pragmas ----------------
# `clang -fopenmp` parallelizes the loop; we point it at Homebrew's libomp.
echo "=== C version (clang -fopenmp) ==="
clang -fopenmp -I"${OMP_PREFIX}/include" \
  -L"${OMP_PREFIX}/lib" -Wl,-rpath,"${OMP_PREFIX}/lib" \
  omp_double.c -o ./build/omp_double_c
./build/omp_double_c; echo

# --- (inspect) scf.parallel -> omp.parallel/omp.wsloop -----------------------
mlir-opt scf_parallel.mlir -convert-scf-to-openmp -o ./build/scf_parallel_omp.mlir

# --- lower the omp dialect program all the way to the llvm dialect ------------
#   -convert-openmp-to-llvm turns omp.* into calls into the OpenMP runtime.
mlir-opt omp_double.mlir \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-openmp-to-llvm \
  --convert-arith-to-llvm \
  --convert-index-to-llvm \
  --finalize-memref-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  -o ./build/omp_double_opt.mlir

mlir-translate ./build/omp_double_opt.mlir -mlir-to-llvmir -o ./build/omp_double.ll
llc -filetype=obj --relocation-model=pic ./build/omp_double.ll -o ./build/omp_double.o

# Link against libomp and bake in its rpath, then run.
clang ./build/omp_double.o -o ./build/omp_double \
  -L"${OMP_PREFIX}/lib" -lomp -Wl,-rpath,"${OMP_PREFIX}/lib"

echo "=== running (input doubled, one value per line) ==="
./build/omp_double
