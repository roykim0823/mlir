// linalg.reduce — the named reduction op (RUNNABLE).
//
// This is reduce_generic.mlir collapsed to one line. linalg.reduce takes:
//   { arith.addf }   — the combine op; its two operands are (running accumulator,
//                      input element), and the result becomes the new accumulator.
//   dimensions = [1] — the axis (or axes) to collapse. Here dim 1 (the columns), so
//                      a 3x4 input reduces to a length-3 output: one sum per row.
//
// Because it writes the caller-provided memref %init in place (accumulating into it),
// %init must arrive PRE-ZEROED — the driver passes np.zeros. No bufferization needed
// (already memrefs), so it lowers straight to loops. Computes NumPy's a.sum(axis=1).
func.func @row_sum(%input: memref<3x4xf32>, %init: memref<3xf32>)
    attributes {llvm.emit_c_interface} {
  linalg.reduce { arith.addf }
      ins(%input : memref<3x4xf32>)
      outs(%init : memref<3xf32>)
      dimensions = [1]
  return
}
