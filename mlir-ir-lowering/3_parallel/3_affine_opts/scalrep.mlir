// Scalar replacement (store-to-load forwarding). INSPECT-ONLY.
//
// When a value is stored and then immediately loaded from the same location, the
// load is redundant — affine analysis can forward the stored value directly and
// delete the load (and dead stores).
//
//   mlir-opt scalrep.mlir -affine-scalrep
//
// The `affine.load %A[%i]` right after the store disappears; the `arith.addf`
// uses the stored constant directly, and the first store is dropped as dead.
func.func @scalrep(%A: memref<10xf32>) {
  affine.for %i = 0 to 10 {
    %c = arith.constant 1.0 : f32
    affine.store %c, %A[%i] : memref<10xf32>          // store ...
    %v = affine.load %A[%i] : memref<10xf32>          // ... then load it right back
    %t = arith.addf %v, %v : f32
    affine.store %t, %A[%i] : memref<10xf32>
  }
  return
}
