// C-compatible wrapper demo (from the PDF's "C-compatible wrappers" section).
//
// The body is intentionally trivial — it just returns its input memref, i.e.
// the identity function. The point is NOT the computation but the calling
// convention: because the function takes AND returns a memref<3xf32>, lowering
// it with `-finalize-memref-to-llvm` unrolls each memref descriptor into five
// scalar fields (alloc ptr, aligned ptr, offset, sizes[], strides[]).
//
// `llvm.emit_c_interface` additionally emits a wrapper named
// `_mlir_ciface_add_vector_to_matrix` that takes pointers to descriptor
// structs instead of the five unrolled scalars — which is what aot_main.py
// calls via ctypes. Inspect the generated pair with:
//     mlir-opt add_vector_to_matrix.mlir -finalize-memref-to-llvm \
//       -convert-func-to-llvm -reconcile-unrealized-casts
// (see README.md for the full expanded output and explanation).
module {
  func.func @add_vector_to_matrix(%A: memref<3xf32>)
    -> memref<3xf32> attributes {llvm.emit_c_interface} {
    return %A : memref<3xf32>   // identity: hand the input buffer straight back
  }
}
