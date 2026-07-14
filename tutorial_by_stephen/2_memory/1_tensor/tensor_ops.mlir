// Pure-tensor catalog (INSPECT-ONLY — there is nothing to execute here).
//
// A `tensor` is an immutable, side-effect-free *value*. It has a shape and an
// element type but NO memory address — you can't take a pointer to it, and you
// can't run it directly. That's exactly why this file is a snippet: to make a
// tensor program *run*, you must first bufferize it into memrefs (see the
// runnable identity kernels — identity_fill.mlir / identity_return.mlir — and the
// deep dive in ../3_low_level_llvm/bufferization.mlir).
//
// This file is a catalog of the pure tensor-producing and tensor-accessing ops
// (@make, @access). The identity-matrix example (@identity) that used to live
// here has been extracted into its own runnable file, identity_return.mlir.
//
// Round-trip / inspect it with:
//     mlir-opt tensor_ops.mlir                 # parse + verify, print back
//     mlir-opt tensor_ops.mlir -canonicalize   # see folding/cleanup

// ---- Tensor TYPES -------------------------------------------------------
// Shapes use `x` as the delimiter between dimensions and the element type.
// `?` marks a DYNAMIC dimension (size known only at runtime); a number marks
// a STATIC dimension (size fixed at compile time). Type aliases (`!Name`)
// keep long types readable.
!TensorStatic = tensor<10x10x10x10xf32>   // fully static 4-D: 10·10·10·10 f32
!TensorDyn    = tensor<?x?x10x10xf32>      // first two dims dynamic, last two static

module {
  // ---- Tensor-PRODUCING ops --------------------------------------------
  // Each op returns a NEW tensor value; nothing is mutated in place.
  func.func @make(%N: index, %M: index, %X: f32) -> tensor<3xf32> {
    // tensor.empty: allocate an uninitialized tensor. Dynamic dims are passed
    // as operands, one per `?`, in order. Static dims take no operand.
    %A = tensor.empty(%N, %M) : !TensorDyn                 // ? -> %N, ? -> %M
    %B = tensor.empty()       : !TensorStatic              // no dynamic dims

    // tensor.from_elements: build a small tensor literal from SSA values.
    %C = tensor.from_elements %X, %X, %X : tensor<3xf32>

    // Note: %A and %B are never used below. Because tensors are pure values
    // with no side effects, an unused tensor op is just dead code — run
    // `mlir-opt tensor_ops.mlir -canonicalize` and watch %A/%B disappear.
    return %C : tensor<3xf32>
  }

  // ---- Reading / "updating" elements -----------------------------------
  func.func @access(%A: !TensorDyn, %N: index, %M: index, %X: f32)
      -> (f32, !TensorDyn) {
    // tensor.extract: read one element (needs rank-many indices). Pure read.
    %e = tensor.extract %A[%N, %M, %N, %M] : !TensorDyn

    // tensor.insert: produce a NEW tensor equal to %A but with one element
    // replaced. %A itself is unchanged — this is value semantics, not a store.
    %A2 = tensor.insert %X into %A[%N, %M, %N, %M] : !TensorDyn

    return %e, %A2 : f32, !TensorDyn
  }

  // ---- Building a tensor element-by-element with a lambda ---------------
  // tensor.generate runs its region once per index tuple and uses the yielded
  // value as that element. For a complete, RUNNABLE example of this op (the m×n
  // identity matrix, 1 on the diagonal and 0 elsewhere) see identity_return.mlir.
}
