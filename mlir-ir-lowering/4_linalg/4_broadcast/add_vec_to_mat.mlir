// Broadcasting — add a length-4 vector to every row of a 3x4 matrix.
//
// Just like NumPy's `matrix + vector`, Linalg can implicitly stretch a smaller
// tensor to match a larger one. `linalg.broadcast` does the stretch:
//   dimensions = [0] means "the input has no dimension 0; replicate it along the
//   new leading dimension" — so the length-4 vector fills all 3 rows.
//
// We do it in two named ops, writing into the caller's %out buffer in place
// (destination-passing, no allocation):
//   1. broadcast %vector -> %out   (every row becomes [v0, v1, v2, v3])
//   2. %out = %matrix + %out       (elementwise add)
//
// Equivalent NumPy:  np.ones((3,4)) + np.array([1,2,3,4])
// Exported as `_mlir_ciface_add_vec_to_mat` (see aot_main.py).
func.func @add_vec_to_mat(%matrix: memref<3x4xf32>, %vector: memref<4xf32>, %out: memref<3x4xf32>)
    attributes {llvm.emit_c_interface} {
  // 1. broadcast the vector across the 3 rows of %out
  linalg.broadcast ins(%vector : memref<4xf32>)
                   outs(%out : memref<3x4xf32>)
                   dimensions = [0]
  // 2. add the matrix elementwise into %out (in place)
  linalg.add ins(%matrix, %out : memref<3x4xf32>, memref<3x4xf32>)
             outs(%out : memref<3x4xf32>)
  return
}
