// linalg.generic matmul — the contraction, spelled out (INSPECT-ONLY).
//
// matmul.mlir is one line; this is the linalg.generic it desugars to. A matmul is a
// CONTRACTION: C[i,j] = sum_k A[i,k] * B[k,j] — two free (parallel) indices i, j and
// one summed (reduction) index k. It is Step 1's "parallel" and Step 2's "reduction"
// in a single op.
//
//   indexing_maps  — (i,j,k)->(i,k) for A, (i,j,k)->(k,j) for B, (i,j,k)->(i,j) for
//                    C. Note k is absent from C's map: that's what makes it reduced.
//   iterator_types — ["parallel", "parallel", "reduction"]: i, j independent; k summed.
//   the region     — multiply the two inputs, add into the accumulator %acc (the
//                    current C element, READ here — like the reduction in Step 2),
//                    yield the running dot product.
//
// Memref-based (like matmul.mlir), so no bufferization is needed to lower to loops.
//
// Inspect / lower to loops — two ways, same nest, different dialect:
//   mlir-opt matmul_generic.mlir -convert-linalg-to-loops          # -> scf.for
//   mlir-opt matmul_generic.mlir -convert-linalg-to-affine-loops   # -> affine.for
#map_A = affine_map<(i, j, k) -> (i, k)>
#map_B = affine_map<(i, j, k) -> (k, j)>
#map_C = affine_map<(i, j, k) -> (i, j)>

func.func @matmul(%A: memref<8x10xf32>, %B: memref<10x16xf32>, %C: memref<8x16xf32>)
    attributes {llvm.emit_c_interface} {
  linalg.generic {
      indexing_maps  = [#map_A, #map_B, #map_C],
      iterator_types = ["parallel", "parallel", "reduction"]
    }
    ins(%A, %B : memref<8x10xf32>, memref<10x16xf32>)
    outs(%C : memref<8x16xf32>) {
      ^bb0(%a: f32, %b: f32, %acc: f32):
      %p = arith.mulf %a, %b : f32
      %s = arith.addf %acc, %p : f32    // read %acc (current C[i,j]), accumulate
      linalg.yield %s : f32
    }
  return
}
