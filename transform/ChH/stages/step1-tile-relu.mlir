// Stage 1: materialize the co loop, then the n, y, xo loops, around the ReLU.
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %bias = transform.structured.match ops{["linalg.broadcast"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %generics = transform.structured.match ops{["linalg.generic"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %conv, %relu = transform.split_handle %generics
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    //                                                             [n  y  x  c]
    %relu2, %co = transform.structured.tile_using_forall %relu
                                                        tile_sizes [0, 0, 0, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %relu3, %n_y_xo = transform.structured.tile_using_forall %relu2
                                                        tile_sizes [1, 1, 5, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
