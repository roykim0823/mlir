// Demonstrates the `llvm.emit_c_interface` attribute.
//
// Without it, lowering a function that takes a memref unrolls the
// memref struct into five separate scalar arguments
//     (allocated_ptr, aligned_ptr, offset, sizes[], strides[])
// which is painful to call from C.
//
// With it, mlir-opt also emits a wrapper named `_mlir_ciface_<name>`
// that takes pointers to memref descriptor structs instead. That is
// the entry point Python / C code actually calls.
//
// Lower & inspect with:
//   mlir-opt c_interface.mlir \
//     -finalize-memref-to-llvm \
//     -convert-func-to-llvm \
//     -reconcile-unrealized-casts
//
// You should see two functions in the output: `identity` (unrolled)
// and `_mlir_ciface_identity` (struct-pointer wrapper).
module {
  func.func @identity(%A: memref<3xf32>) -> memref<3xf32>
      attributes {llvm.emit_c_interface} {
    return %A : memref<3xf32>
  }
}
