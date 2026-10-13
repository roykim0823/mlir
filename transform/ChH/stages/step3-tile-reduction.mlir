// Stage 3: stage 2 + materialize the rz, ry, rx reduction loops as scf.for.
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

    %conv2, %co2 = transform.structured.fuse_into_containing_op %conv into %co
      : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op)
    %conv3, %n_y_xo2 = transform.structured.fuse_into_containing_op %conv2
      into %n_y_xo
      : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op)

    %bias2, %co3 = transform.structured.fuse_into_containing_op %bias into %co2
      : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op)
    %bias3, %n_y_xo3 = transform.structured.fuse_into_containing_op %bias2
      into %n_y_xo2
      : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op)

    // Fusion clones the producer into the loop; an empty pattern sweep runs
    // dead-code elimination on the now-unused originals.
    %f00 = transform.structured.match ops{["func.func"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f00 {
    } : !transform.any_op

    //            n  y  x  c  rz ry rx
    %red_fill, %conv4, %comb, %rz_ry_rx
      = transform.structured.tile_reduction_using_for %conv3
        by tile_sizes=[0, 0, 0, 0, 1, 1, 1]
      : (!transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op,
          !transform.any_op)
    transform.yield
  }
}
