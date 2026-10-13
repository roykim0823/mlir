// Bufferization: converting tensor (value) semantics into memref
// (buffer / pointer) semantics.
//
// Try each variant below:
//
//   # 1. Explicit to_memref / to_tensor:
//   mlir-opt bufferization.mlir -canonicalize
//
//   # 2. One-shot bufferize, including function boundaries
//   #    (so memref<...> appears in the signature, not tensor<...>):
//   mlir-opt bufferization.mlir \
//     -one-shot-bufferize="bufferize-function-boundaries"
module {
  // --- 1. Manual bufferization primitives -------------------------
  //
  // bufferization.to_memref  : tensor -> memref (same data, no copy
  //                            when the analysis can prove it safe)
  // bufferization.to_tensor  : memref -> tensor (read-only view)
  func.func @manual(%t: tensor<10xf32>) -> tensor<10xf32> {
    %m = bufferization.to_memref %t : tensor<10xf32> to memref<10xf32>
    // `restrict` tells the analyzer no other tensor aliases this
    // buffer; required when this file is fed to one-shot-bufferize.
    %r = bufferization.to_tensor %m restrict
       : memref<10xf32> to tensor<10xf32>
    return %r : tensor<10xf32>
  }

  // --- 2. Tensor-level kernel for the one-shot pass ----------------
  //
  // Before bufferization the signature uses `tensor<8xf32>`; after the
  // pass (with `bufferize-function-boundaries`) it becomes
  // `memref<8xf32>` and the tensor.insert becomes a memref.store
  // performed in place.
  func.func @insert_scalar(%arg0: tensor<8xf32>, %idx: index)
      -> tensor<8xf32> {
    %f = arith.constant 5.000000e+00 : f32
    %r = tensor.insert %f into %arg0[%idx] : tensor<8xf32>
    return %r : tensor<8xf32>
  }

  // --- 3. materialize_in_destination ------------------------------
  //
  // Used at the very end of a kernel to write a tensor result into a
  // pre-existing externally-owned buffer (e.g. one provided by the
  // caller). It fuses the to_memref + copy into a single op.
  func.func @materialize(%in: tensor<10xf32>,
                         %out: memref<10xf32>) {
    bufferization.materialize_in_destination
        %in in restrict writable %out
      : (tensor<10xf32>, memref<10xf32>) -> ()
    return
  }
}
