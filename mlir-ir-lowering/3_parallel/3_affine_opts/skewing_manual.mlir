// Skewing — shear the iteration space. Done BY HAND (no one-flag affine pass).
//
// Stencil A[i,j] = A[i-1,j] + A[i,j-1]: each cell needs its left and upper
// neighbours, so BOTH loops are sequential. This file shows three versions that
// all compute the same array (aot_main.py checks it):
//   @stencil           — the plain sequential nest (reference)
//   @stencil_skewed    — the skew reindexing ALONE (still sequential; no speedup)
//   @stencil_wavefront — skew + interchange (inner loop genuinely parallel)
// RUNNABLE: carries llvm.emit_c_interface.

// Reference: the plain doubly-sequential stencil.
func.func @stencil(%A: memref<8x8xf32>)
    attributes {llvm.emit_c_interface} {
  affine.for %i = 1 to 8 {
    affine.for %j = 1 to 8 {
      %a = affine.load %A[%i - 1, %j] : memref<8x8xf32>
      %b = affine.load %A[%i, %j - 1] : memref<8x8xf32>
      %s = arith.addf %a, %b : f32
      affine.store %s, %A[%i, %j] : memref<8x8xf32>
    }
  }
  return
}

// Skew ALONE: remap the inner index (j_new = j - i, bounds shifted by i). A pure
// reindexing — visits the same (i, j_new) cells with the same update. With i still
// the outer loop, the inner loop still reads A[i, j_new-1] from the previous step,
// so it stays SEQUENTIAL: no parallelism is gained yet.
func.func @stencil_skewed(%A: memref<8x8xf32>)
    attributes {llvm.emit_c_interface} {
  affine.for %i = 1 to 8 {
    affine.for %j = affine_map<(d) -> (d + 1)>(%i) to affine_map<(d) -> (d + 8)>(%i) {
      %jn = affine.apply affine_map<(d0, d1) -> (d1 - d0)>(%i, %j)   // j_new = j - i
      %a = affine.load %A[%i - 1, %jn] : memref<8x8xf32>
      %b = affine.load %A[%i, %jn - 1] : memref<8x8xf32>
      %s = arith.addf %a, %b : f32
      affine.store %s, %A[%i, %jn] : memref<8x8xf32>
    }
  }
  return
}

// Skew + INTERCHANGE = the wavefront (where the benefit is). Iterate the OUTER
// loop over anti-diagonals t = i + j, the INNER loop over the cells on that
// diagonal. Every cell (i, t-i) depends only on diagonal t-1, so no two cells of
// the same diagonal depend on each other — the inner loop is genuinely
// independent, written `affine.parallel`. (Lowering here still serializes it,
// exactly like Step 1's matmul; real parallel execution needs the OpenMP/GPU
// route. The point is the loop is now legally parallel.)
func.func @stencil_wavefront(%A: memref<8x8xf32>)
    attributes {llvm.emit_c_interface} {
  affine.for %t = 2 to 15 {                                  // outer: over diagonals t = i + j
    // cells on diagonal t: i in [max(1, t-7), min(7, t-1)] so j = t-i stays in [1,7]
    affine.parallel (%i) = (max(1, %t - 7)) to (min(8, %t)) {   // inner: PARALLEL
      %j = affine.apply affine_map<(d0, d1) -> (d1 - d0)>(%i, %t)   // j = t - i
      %a = affine.load %A[%i - 1, %j] : memref<8x8xf32>
      %b = affine.load %A[%i, %j - 1] : memref<8x8xf32>
      %s = arith.addf %a, %b : f32
      affine.store %s, %A[%i, %j] : memref<8x8xf32>
    }
  }
  return
}
