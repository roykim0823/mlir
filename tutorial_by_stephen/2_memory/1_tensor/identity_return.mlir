// identity (returns-a-tensor variant) — closest to the PDF's original snippet.
//
// Difference from identity_fill.mlir: this version *returns* an m×n tensor
// instead of writing into a caller-provided memref. After bufferization the
// callee allocates the result buffer, so the C caller must free it afterwards
// (see aot_main.py, run_return, for the "callee allocates, caller frees" driver).
//
// `tensor.generate` builds a tensor by running its region once per index
// (%i, %j); whatever each invocation yields becomes that element. The result
// shape tensor<?x?xi32> is dynamic, with the actual m, n supplied at the call.
func.func @identity(%m : index, %n : index) -> tensor<?x?xi32> attributes {llvm.emit_c_interface} {
  %out = tensor.generate %m, %n {
  ^bb0(%i : index, %j : index):                  // region runs for every (i, j)
    %ni = arith.index_cast %i : index to i32      // index -> i32 so we can compare
    %nj = arith.index_cast %j : index to i32
    %eq = arith.cmpi eq, %ni, %nj : i32           // i == j ?  -> i1 (1 on the diagonal)
    %v  = arith.extui %eq : i1 to i32             // widen i1 to i32 (1 or 0)
    tensor.yield %v : i32                          // value of this (i, j) element
  } : tensor<?x?xi32>
  return %out : tensor<?x?xi32>
}