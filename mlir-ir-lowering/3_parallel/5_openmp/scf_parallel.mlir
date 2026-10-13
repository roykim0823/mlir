// scf.parallel -> OpenMP (INSPECT-ONLY).
//
// You rarely write the omp dialect by hand. The usual route is to express
// parallelism with `scf.parallel` (or let --affine-parallelize produce it) and
// let `-convert-scf-to-openmp` rewrite it into omp.parallel + omp.wsloop.
//
//   mlir-opt scf_parallel.mlir -convert-scf-to-openmp
//
// In the output, the scf.parallel becomes an omp.parallel region wrapping an
// omp.wsloop, with the bounds/step preserved. Follow with
// -convert-openmp-to-llvm to continue lowering toward the OpenMP runtime.
func.func @add(%A: memref<100xf32>, %B: memref<100xf32>) {
  %c0   = arith.constant 0   : index
  %c1   = arith.constant 1   : index
  %c100 = arith.constant 100 : index
  scf.parallel (%i) = (%c0) to (%c100) step (%c1) {
    %v = memref.load %A[%i] : memref<100xf32>
    %r = arith.addf %v, %v : f32
    memref.store %r, %B[%i] : memref<100xf32>
    scf.reduce
  }
  return
}
