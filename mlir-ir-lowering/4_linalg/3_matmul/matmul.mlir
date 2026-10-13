// linalg.matmul — a whole matrix multiply as ONE op.
//
//   C[i,j] = sum_k A[i,k] * B[k,j]
//
// Compare this to Chapter 3's hand-written affine nest: there we spelled out the
// three loops; here the single named op carries all of it. Under the hood it is a
// linalg.generic with indexing maps
//     (i,j,k)->(i,k),  (i,j,k)->(k,j),  (i,j,k)->(i,j)
// and iterator types ["parallel", "parallel", "reduction"] — the two free indices
// i, j are parallel, the contraction index k is the reduction. (See the README
// for that explicit generic form.)
//
// `-convert-linalg-to-loops` lowers it to an scf loop nest; the caller passes a
// zero-initialized C since the op accumulates into `outs`. Exported as
// `_mlir_ciface_matmul` (see aot_main.py).
func.func @matmul(%A: memref<8x10xf32>, %B: memref<10x16xf32>, %C: memref<8x16xf32>)
    attributes {llvm.emit_c_interface} {
  linalg.matmul ins(%A, %B : memref<8x10xf32>, memref<10x16xf32>)
                outs(%C : memref<8x16xf32>)
  return
}
