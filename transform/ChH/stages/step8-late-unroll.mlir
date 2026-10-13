// Stage 8: stage 7 with the xi/ci loops materialized as scf.forall, converted
// to scf.for and unrolled only after bufferization.
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

    // Materialize xi (size 1) and ci (size 16 = the vector length) as one
    // two-dimensional scf.forall around each of the four computations.
    //                                                                n  y  xi ci
    %conv5, %conv_xi_ci = transform.structured.tile_using_forall %conv4
                                                           tile_sizes [0, 0, 1, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %bias4, %bias_xi_ci = transform.structured.tile_using_forall %bias3
                                                           tile_sizes [0, 0, 1, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %relu4, %relu_xi_ci = transform.structured.tile_using_forall %relu3
                                                           tile_sizes [0, 0, 1, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %comb2, %comb_xi_ci = transform.structured.tile_using_forall %comb
                                                           tile_sizes [0, 0, 1, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Simplify: canonicalize, CSE, hoist loop invariants.
    transform.apply_patterns to %f00 {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.linalg.tiling_canonicalization
    } : !transform.any_op
    transform.apply_cse to %f00 : !transform.any_op
    %all_loops = transform.structured.match interface{LoopLikeInterface}
      in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %all_loops : !transform.any_op

    // Fold the unit dimensions left by tiling-by-one, then vectorize.
    transform.apply_patterns to %f00 {
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
    } : !transform.any_op
    %fv = transform.structured.vectorize_children_and_apply_patterns %f00
      : (!transform.any_op) -> !transform.any_op

    // Simplify again after vectorization.
    transform.apply_patterns to %fv {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.tensor.fold_tensor_subset_ops_into_vector_transfers
    } : !transform.any_op
    transform.apply_cse to %fv : !transform.any_op
    transform.structured.hoist_redundant_vector_transfers %fv
      : (!transform.any_op) -> !transform.any_op

    // Bufferize, then lower the vector operations (same as full.mlir).
    %arg1 = transform.bufferization.one_shot_bufferize %arg0 {
      bufferize_function_boundaries = true,
      function_boundary_type_conversion = 1 : i32 }
      : (!transform.any_op) -> !transform.any_op
    %f = transform.structured.match ops{["func.func"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "buffer-deallocation-pipeline" to %f
      : (!transform.any_op) -> !transform.any_op

    // Apply general canonicalization and CSE to each function after
    // bufferization as new simplification opportunities may have appeared.
    %fb = transform.structured.match ops{["func.func"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %fb {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %fb : !transform.any_op

    // Lower complex, multidimensional vector operations into simpler
    // primitives. This particular selection of the pattern groups corresponds
    // to vector dialect operations present in the payload IR at this stage.
    // Many of these groups can be parameterized to use different strategies or
    // lower-level primitives offering performance trade-offs. In this case, we
    // are selecting the simplest strategies.
    transform.apply_patterns to %fb {
      transform.apply_patterns.vector.lower_contraction
        lowering_strategy = parallelarith
      transform.apply_patterns.vector.lower_transfer
        max_transfer_rank = 1
      transform.apply_patterns.vector.lower_transpose
        lowering_strategy = eltwise
      transform.apply_patterns.vector.lower_shape_cast
    } : !transform.any_op

    // These patterns apply in a separate sweep to avoid transfer-to-scf
    // patterns overlap with lower-transfer patterns as they apply to the same
    // kind of operations. These patterns may produce local allocations to act
    // as temporary caches deep inside loops, which could lead to catastrophic
    // performance. Such allocations are moved onto the stack and hoisted from
    // all the surrounding loops.
    transform.apply_patterns to %fb {
      transform.apply_patterns.vector.transfer_to_scf
      transform.apply_patterns.memref.alloc_to_alloca
      } : !transform.any_op
    transform.bufferization.buffer_loop_hoisting %fb : !transform.any_op

    // A final round of cleanups additionally includes patterns to simplify
    // buffer aliasing operations that may have been introduced during
    // bufferization and could result in excessively complex address
    // computation.
    transform.apply_patterns to %fb {
      transform.apply_patterns.memref.fold_memref_alias_ops
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %fb : !transform.any_op

    // Bufferization invalidated every handle: match the loops again. Matching
    // walks the IR in post-order, so inner loops come before the loops that
    // contain them: bias, conv, combiner, ReLU, then (n, y, xo) and co.
    %foralls = transform.structured.match ops{["scf.forall"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %xi_ci_bias, %xi_ci_conv, %xi_ci_comb, %xi_ci_relu, %n_y_xo_l, %co_l
      = transform.split_handle %foralls
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
          !transform.any_op, !transform.any_op, !transform.any_op,
          !transform.any_op)
    %xi_bias, %ci_bias = transform.loop.forall_to_for %xi_ci_bias
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %xi_conv, %ci_conv = transform.loop.forall_to_for %xi_ci_conv
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %xi_comb, %ci_comb = transform.loop.forall_to_for %xi_ci_comb
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %xi_relu, %ci_relu = transform.loop.forall_to_for %xi_ci_relu
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Unrolling now happens on buffers, after all the tensor-level rewrites.
    transform.loop.unroll %ci_bias {factor = 4} : !transform.any_op
    transform.loop.unroll %xi_bias {factor = 5} : !transform.any_op
    transform.loop.unroll %ci_conv {factor = 4} : !transform.any_op
    transform.loop.unroll %xi_conv {factor = 5} : !transform.any_op
    transform.loop.unroll %ci_comb {factor = 4} : !transform.any_op
    transform.loop.unroll %xi_comb {factor = 5} : !transform.any_op
    transform.loop.unroll %ci_relu {factor = 4} : !transform.any_op
    transform.loop.unroll %xi_relu {factor = 5} : !transform.any_op
    transform.yield
  }
}
