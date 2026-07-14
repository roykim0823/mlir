// Loop fusion. INSPECT-ONLY.
//
// Two adjacent loops over the same range — the first writes B from A, the second
// reads B to write C. Fusing them into one loop improves locality (B[i] is still
// hot when the second body uses it) and removes a loop's overhead.
//
//   mlir-opt fusion.mlir -affine-loop-fusion
//
// After the pass the two affine.for loops collapse into a single loop.
func.func @fusion(%A: memref<10xf32>, %B: memref<10xf32>, %C: memref<10xf32>) {
  affine.for %i = 0 to 10 {
    %v1 = affine.load %A[%i] : memref<10xf32>
    %v2 = arith.mulf %v1, %v1 : f32
    affine.store %v2, %B[%i] : memref<10xf32>
  }
  affine.for %i = 0 to 10 {
    %v3 = affine.load %B[%i] : memref<10xf32>
    %v4 = arith.addf %v3, %v3 : f32
    affine.store %v4, %C[%i] : memref<10xf32>
  }
  return
}
