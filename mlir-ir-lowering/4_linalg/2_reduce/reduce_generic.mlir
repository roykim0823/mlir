// linalg.generic reduction — the "reduction" iterator, spelled out (INSPECT-ONLY).
//
// Step 1 was all "parallel": every iteration independent, %out never read. A
// REDUCTION is the opposite — many input elements collapse into one output, so each
// iteration must read the running total in %out, combine it with the input, and
// write it back. This is exactly the accumulation the unused %out in Step 1 hinted
// at.
//
// Here we sum each row of a 3x4 matrix into a length-3 vector (NumPy a.sum(axis=1)):
//
//   indexing_maps  — input is indexed (i, j) -> (i, j); output (i, j) -> (i). The
//                    output map drops j, which is what marks j as the collapsed axis.
//   iterator_types — ["parallel", "reduction"]: i is independent (one output per
//                    row), j is summed away.
//   the region     — reads %out (the running sum), adds the input element, yields it.
//                    NOTE %out is READ here, unlike the elementwise add in Step 1.
//
// The named linalg.reduce in reduce.mlir is this whole thing as one line.
//
// Inspect / lower to loops with:
//   mlir-opt reduce_generic.mlir \
//     -one-shot-bufferize="bufferize-function-boundaries" \
//     -convert-linalg-to-loops     # -> outer parallel loop, inner reduction loop
#map_in  = affine_map<(i, j) -> (i, j)>
#map_out = affine_map<(i, j) -> (i)>

func.func @row_sum(%input: tensor<3x4xf32>, %init: tensor<3xf32>) -> tensor<3xf32> {
  %0 = linalg.generic {
      indexing_maps  = [#map_in, #map_out],
      iterator_types = ["parallel", "reduction"]
    }
    ins(%input : tensor<3x4xf32>)
    outs(%init : tensor<3xf32>) {
      ^bb0(%in: f32, %out: f32):
      %s = arith.addf %out, %in : f32    // read %out (running sum), accumulate
      linalg.yield %s : f32
    } -> tensor<3xf32>
  return %0 : tensor<3xf32>
}
