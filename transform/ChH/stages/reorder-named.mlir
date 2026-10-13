// Interchange reorders the implicit loops of a linalg.generic only: applied
// to the named linalg.broadcast, it fails.
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %bias = transform.structured.match ops{["linalg.broadcast"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    //                                             [n  y  x  c] -> [c  n  y  x]
    transform.structured.interchange %bias iterator_interchange = [3, 0, 1, 2]
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
