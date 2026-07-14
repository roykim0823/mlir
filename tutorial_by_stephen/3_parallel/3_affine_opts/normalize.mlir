// Loop normalization. INSPECT-ONLY.
//
// Normalizing rewrites a loop to start at 0 and step 1, folding the original
// bounds/step into the index. Many other passes assume normalized loops, so this
// is a common pre-pass.
//
//   mlir-opt normalize.mlir -affine-loop-normalize
//
// `10 to 100 step 5` becomes `0 to 18` (because (100-10)/5 = 18), with the real
// index recovered as `idx * 5 + 10`.
func.func @normalize(%A: memref<100xf32>) {
  affine.for %i = 10 to 100 step 5 {
    %v = affine.load %A[%i] : memref<100xf32>
    %t = arith.addf %v, %v : f32
    affine.store %t, %A[%i] : memref<100xf32>
  }
  return
}
