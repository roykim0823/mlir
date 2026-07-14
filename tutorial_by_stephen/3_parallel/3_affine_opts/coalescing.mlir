// Loop coalescing. INSPECT-ONLY.
//
// Coalescing flattens a perfectly-nested loop into a SINGLE loop, recovering the
// original indices with div/mod. Useful when you want one flat iteration space
// (e.g. to then map onto a 1-D thread grid).
//
//   mlir-opt coalescing.mlir -affine-loop-coalescing
//
// After: one `affine.for ... to 32` (4*8), with %i = idx floordiv 8 and
// %j = idx mod 8 rebuilt inside.
func.func @coalescing(%A: memref<4x8xf32>) {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 8 {
      %v = affine.load %A[%i, %j] : memref<4x8xf32>
      %t = arith.addf %v, %v : f32
      affine.store %t, %A[%i, %j] : memref<4x8xf32>
    }
  }
  return
}
