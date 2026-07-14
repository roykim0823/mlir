// Loop tiling. INSPECT-ONLY.
//
// Tiling splits a loop into chunks ("tiles") so the working set fits in cache.
// A 32x32 nest becomes a 4-deep nest of (tile-loop, point-loop) pairs.
//
//   mlir-opt tiling.mlir -affine-loop-tile="tile-size=8"
//
// After the pass you'll see outer loops stepping by 8 and inner loops walking
// the 8 elements of each tile.
func.func @tiling(%A: memref<32x32xf32>) {
  affine.for %i = 0 to 32 {
    affine.for %j = 0 to 32 {
      %v = affine.load %A[%i, %j] : memref<32x32xf32>
      %t = arith.addf %v, %v : f32
      affine.store %t, %A[%i, %j] : memref<32x32xf32>
    }
  }
  return
}
