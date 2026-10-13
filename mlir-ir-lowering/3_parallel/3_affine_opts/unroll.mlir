// Loop unrolling. INSPECT-ONLY.
//
// Unrolling replicates the loop body to cut loop overhead and expose
// instruction-level parallelism.
//
//   mlir-opt unroll.mlir -affine-loop-unroll="unroll-factor=4"
//
// After the pass the body appears 4 times per iteration (4x as many
// affine.load/store ops), with the trip count divided accordingly.
func.func @unroll(%A: memref<8xf32>) {
  affine.for %i = 0 to 8 {
    %v = affine.load %A[%i] : memref<8xf32>
    %t = arith.addf %v, %v : f32
    affine.store %t, %A[%i] : memref<8xf32>
  }
  return
}
