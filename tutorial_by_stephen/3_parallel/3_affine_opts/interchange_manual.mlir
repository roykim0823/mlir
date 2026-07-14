// Interchange — swap the loop nesting order. Done BY HAND (no one-flag affine
// pass; see the README for the Transform-dialect / C++-utility routes).
//
// Both functions copy B = A. Visiting (i, j) in a different order can't change the
// result — every element is written exactly once and the iterations are
// independent — but row-major (i outer) is far cache-friendlier than column-major.
// aot_main.py runs both and confirms they produce the same B (== A).
// RUNNABLE: carries llvm.emit_c_interface.

// before: j outer, i inner (column-major walk)
func.func @copy_ji(%A: memref<4x5xf32>, %B: memref<4x5xf32>)
    attributes {llvm.emit_c_interface} {
  affine.for %j = 0 to 5 {
    affine.for %i = 0 to 4 {
      %v = affine.load %A[%i, %j] : memref<4x5xf32>
      affine.store %v, %B[%i, %j] : memref<4x5xf32>
    }
  }
  return
}

// after: i outer, j inner (interchanged — row-major, better locality)
func.func @copy_ij(%A: memref<4x5xf32>, %B: memref<4x5xf32>)
    attributes {llvm.emit_c_interface} {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 5 {
      %v = affine.load %A[%i, %j] : memref<4x5xf32>
      affine.store %v, %B[%i, %j] : memref<4x5xf32>
    }
  }
  return
}
