// Matrix multiply with the AFFINE dialect — C += A · B.
//
// Unlike scf.for (Chapter 1), affine.for is a "first-class loop": its bounds and
// the memory accesses inside it are restricted to *affine* expressions of the
// loop indices, which is exactly what lets MLIR analyze dependencies and apply
// polyhedral transformations (parallelize, tile, interchange, fuse, ...).
//
// affine.load / affine.store are the affine-aware memory ops; they carry the same
// access-pattern restriction so the dependence analysis can reason about them.
//
// The %i and %j loops are independent across iterations (each writes a distinct
// C[i, j]), so `--affine-parallelize` can turn them into an affine.parallel band
// — build.sh emits that transformed IR into build/ for you to inspect.
//
// Exported as `_mlir_ciface_matmul` (see aot_main.py). The caller passes a
// zero-initialized C, since the kernel accumulates into it.
func.func @matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>)
    attributes {llvm.emit_c_interface} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = memref.dim %A, %c0 : memref<?x?xf32>   // rows of A (and C)
  %N = memref.dim %B, %c1 : memref<?x?xf32>   // cols of B (and C)
  %K = memref.dim %A, %c1 : memref<?x?xf32>   // shared/contraction dimension

  affine.for %i = 0 to %M {
    affine.for %j = 0 to %N {
      affine.for %k = 0 to %K {
        %a = affine.load %A[%i, %k] : memref<?x?xf32>
        %b = affine.load %B[%k, %j] : memref<?x?xf32>
        %c = affine.load %C[%i, %j] : memref<?x?xf32>
        %prod = arith.mulf %a, %b : f32
        %sum  = arith.addf %c, %prod : f32
        affine.store %sum, %C[%i, %j] : memref<?x?xf32>
      }
    }
  }
  return
}
