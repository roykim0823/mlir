// A neural-network DENSE (fully-connected) layer's forward pass, in MLIR.
//
//   out = relu(X @ W + b)
//
// where X is a batch of inputs (N rows x K features), W is the weight matrix
// (K x M), b is the per-output bias (M), and out is N x M. This is the matrix
// form of the layer equation from the chapter:  A = sigma(W·A_prev + b).
//
// It's built entirely from Chapter 4's Linalg pieces:
//   * linalg.matmul             — the X @ W contraction (out is pre-zeroed)
//   * a fused linalg.generic    — adds the broadcast bias AND applies ReLU in one
//                                 pass over `out`, reading b[j] via the (i,j)->(j)
//                                 indexing map (broadcasting the bias across rows)
//
// Fusing the bias-add and the activation into a single generic is exactly the
// kernel-fusion idea from Chapter 4 — one loop over the data instead of three.
// ReLU is just max(z, 0) via arith.maximumf.
//
// Exported as `_mlir_ciface_dense_relu` (see aot_main.py).
func.func @dense_relu(%X: memref<?x?xf32>, %W: memref<?x?xf32>,
                      %b: memref<?xf32>, %out: memref<?x?xf32>)
    attributes {llvm.emit_c_interface} {
  // out = X @ W   (accumulates into the caller's zeroed buffer)
  linalg.matmul ins(%X, %W : memref<?x?xf32>, memref<?x?xf32>)
                outs(%out : memref<?x?xf32>)

  // out[i,j] = relu(out[i,j] + b[j])   — bias broadcast + activation, fused
  linalg.generic {
    indexing_maps  = [affine_map<(i, j) -> (j)>,        // b: depends only on column j
                      affine_map<(i, j) -> (i, j)>],    // out: full 2-D
    iterator_types = ["parallel", "parallel"]
  } ins(%b : memref<?xf32>) outs(%out : memref<?x?xf32>) {
  ^bb0(%bias: f32, %acc: f32):
    %z    = arith.addf %acc, %bias : f32
    %zero = arith.constant 0.0 : f32
    %relu = arith.maximumf %z, %zero : f32              // ReLU = max(z, 0)
    linalg.yield %relu : f32
  }
  return
}
