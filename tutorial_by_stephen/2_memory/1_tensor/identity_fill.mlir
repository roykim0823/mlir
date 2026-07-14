// Identity-matrix kernel — the RUNNABLE counterpart to tensor_ops.mlir.
//
// The compute is pure tensor (`tensor.generate`), but the signature takes a
// memref<?x?xi32> output buffer. That memref is NOT incidental: a pure tensor
// has no memory, so to execute and return data we must bufferize into a real
// buffer. Here we use the destination-passing style — the caller owns the
// buffer and we write the generated matrix straight into it via
// `bufferization.materialize_in_destination` (no extra allocation, no copy).
// See tensor_ops.mlir for the pure-tensor ops, and ../3_low_level_llvm for the
// bufferization deep dive.
//
// Exported to C/Python as `_mlir_ciface_identity` (see aot_main.py, run_fill).
module {
  func.func @identity(%out: memref<?x?xi32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = memref.dim %out, %c0 : memref<?x?xi32>
    %n = memref.dim %out, %c1 : memref<?x?xi32>

    %t = tensor.generate %m, %n {
      ^bb0(%i: index, %j: index):
        %ni = arith.index_cast %i : index to i32
        %nj = arith.index_cast %j : index to i32
        %eq = arith.cmpi eq, %ni, %nj : i32
        %v  = arith.extui %eq : i1 to i32
        tensor.yield %v : i32
    } : tensor<?x?xi32>

    bufferization.materialize_in_destination %t in writable %out
      : (tensor<?x?xi32>, memref<?x?xi32>) -> ()
    return
  }
}
