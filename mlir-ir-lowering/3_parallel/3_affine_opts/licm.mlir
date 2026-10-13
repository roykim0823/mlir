// Loop-invariant code motion (LICM). INSPECT-ONLY.
//
// `%x` doesn't depend on the loop index, so it's recomputed pointlessly every
// iteration. The -affine-loop-invariant-code-motion pass hoists it out.
//
//   mlir-opt licm.mlir -affine-loop-invariant-code-motion
//
// After the pass, the `arith.constant 42.0` appears BEFORE the affine.for.
func.func @licm(%A: memref<10xf32>, %B: memref<10xf32>) {
  affine.for %i = 0 to 10 {
    %x = arith.constant 42.0 : f32           // loop-invariant — gets hoisted
    %v = affine.load %A[%i] : memref<10xf32>
    %s = arith.addf %v, %x : f32
    affine.store %s, %B[%i] : memref<10xf32>
  }
  return
}
