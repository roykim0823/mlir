// linalg.map — the elementwise shorthand for linalg.generic (INSPECT-ONLY).
//
// generic_add.mlir spelled out the whole structure: identity indexing_maps, a
// "parallel" iterator, and a region. For a *pure elementwise* op that boilerplate
// is fully implied — every operand is indexed (d0)->(d0) and every dimension is
// parallel — so linalg.map lets you drop it and give only the scalar payload.
//
//   linalg.map { arith.addf }  — the payload op applied per element. Its operands
//                               are the ins() elements, in order; the single result
//                               is stored to the outs() element. No indexing_maps,
//                               no iterator_types, no explicit region needed.
//
// This is the exact same computation as generic_add.mlir — NumPy's `a + b` — just
// written at the next rung of sugar. It still returns a tensor, so lowering to
// loops needs a bufferize step first (see build.sh).
//
// Inspect / lower to loops with:
//   mlir-opt map_add.mlir                          # inspect (round-trip)
//   mlir-opt map_add.mlir \
//     -one-shot-bufferize="bufferize-function-boundaries" \
//     -convert-linalg-to-loops                     # -> the same scf loop nest
func.func @map_add(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  %init = tensor.empty() : tensor<10xf32>
  %mapped = linalg.map { arith.addf }
      ins(%arg0, %arg1 : tensor<10xf32>, tensor<10xf32>)
      outs(%init : tensor<10xf32>)
  return %mapped : tensor<10xf32>
}
