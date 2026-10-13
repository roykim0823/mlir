// linalg.generic — the most fundamental Linalg op (INSPECT-ONLY).
//
// Every other Linalg op is a specialization of linalg.generic. It describes a
// loop nest *declaratively*: instead of writing the loops, you give three things
// and let the compiler generate the nest.
//
//   indexing_maps  — how each tensor is indexed by the loop variables. Here all
//                    three use (d0) -> (d0): element i of each tensor maps to the
//                    same loop index i (a plain elementwise access).
//   iterator_types — one per loop dimension: "parallel" (independent iterations),
//                    "reduction" (accumulate into one value), or "window".
//   the region     — the scalar computation per element. There is one block arg
//                    per operand: one for each ins() value AND one for each outs()
//                    value. The result is written by linalg.yield (NOT by naming a
//                    value %out — see below). The yielded value is stored into the
//                    output element.
//
// About %out: it is the *current* value of the output element. For a plain
// elementwise write we don't read it, so it looks unused — but the block arity must
// still match the operand count, so it must be declared. You would read %out for
// accumulation (e.g. a reduction: %acc = arith.addf %out, %prod). You cannot write
// `%out = arith.addf %a, %b`: %out is already a defined SSA value (a block arg), and
// SSA values are immutable — that is a "redefinition of SSA value" error. Compute a
// fresh value and yield it instead.
//
// This computes the elementwise sum of two tensors — NumPy's `a + b`.
//
// Inspect / lower to loops with:
//   mlir-opt generic_add.mlir                      # inspect (round-trip, no change)
//   mlir-opt generic_add.mlir \
//     -one-shot-bufferize="bufferize-function-boundaries" \
//     -convert-linalg-to-loops                     # actually produces the scf loop nest
//
// NOTE: -convert-linalg-to-loops only lowers linalg ops on *memrefs* (buffers), not
// on tensors. This op works on tensor<10xf32>, so linalg-to-loops alone is a no-op —
// you must bufferize first to give the loops memory to load/store.
#map_1d = affine_map<(d0) -> (d0)>

func.func @add_tensors(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  %result = tensor.empty() : tensor<10xf32>
  %0 = linalg.generic {
      indexing_maps  = [#map_1d, #map_1d, #map_1d],   // arg0, arg1, result
      iterator_types = ["parallel"]
    }
    ins(%arg0, %arg1 : tensor<10xf32>, tensor<10xf32>)
    outs(%result : tensor<10xf32>) {
      ^bb0(%a: f32, %b: f32, %out: f32):
      %1 = arith.addf %a, %b : f32
      linalg.yield %1 : f32
    } -> tensor<10xf32>
  return %0 : tensor<10xf32>
}
