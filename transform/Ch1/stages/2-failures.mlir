// Original function to optimize.
func.func @fc_relu(%lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
                   %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
                   -> tensor<512x512xf32> {
  // Matrix-matrix multiplication.
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
                          outs(%output: tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise addition.
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
    ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise max with 0 (ReLU).
  %c0f = arith.constant 0.0 : f32
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
    ins(%biased, %c0f : tensor<512x512xf32>, f32)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>
  func.return %relued : tensor<512x512xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op,
      %arg1: !transform.op<"linalg.matmul">,
      %arg2: !transform.op<"linalg.elemwise_binary">) {
    // "suppress": the failing split_handle stops this sequence, the remark
    // after it is skipped, but the enclosing script carries on.
    transform.sequence %arg0 : !transform.any_op failures(suppress) {
    ^bb0(%root: !transform.any_op):
      // %arg2 holds two payload ops, so splitting it into three fails.
      %a, %b, %c = transform.split_handle %arg2
          : (!transform.op<"linalg.elemwise_binary">)
          -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.debug.emit_remark_at %a, "not reached (suppress)" : !transform.any_op
      transform.yield
    }
    transform.debug.emit_remark_at %arg1, "after the suppressed sequence"
        : !transform.op<"linalg.matmul">

    // "propagate": the same failure makes the whole script fail.
    transform.sequence %arg0 : !transform.any_op failures(propagate) {
    ^bb0(%root: !transform.any_op):
      %a, %b, %c = transform.split_handle %arg2
          : (!transform.op<"linalg.elemwise_binary">)
          -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.debug.emit_remark_at %a, "not reached (propagate)" : !transform.any_op
      transform.yield
    }
    transform.debug.emit_remark_at %arg1, "not reached either"
        : !transform.op<"linalg.matmul">
    transform.yield
  }
}
