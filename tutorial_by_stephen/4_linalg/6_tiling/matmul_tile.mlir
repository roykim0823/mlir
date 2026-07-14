// Tiling a linalg.matmul (INSPECT-ONLY).
//
// A whole matmul as one declarative op. Because Linalg keeps the operation
// abstract (it hasn't committed to any loop order yet), the compiler is free to
// *tile* it — break the 10x10x10 iteration space into 5x5x5 blocks that fit in
// cache, turning 3 loops into a 6-deep nest of (tile-loop, point-loop) pairs.
//
// Tiling is the optimization that matters most for memory-bound kernels (and is
// essential on GPUs, where you tile to shared-memory size — Chapter 8).
//
// build.sh runs the full tensor -> bufferize -> affine-loops -> tile pipeline and
// writes build/matmul_tiled.mlir; look for the outer loops stepping by 5.
module {
  func.func @matmul(%a: tensor<10x10xf32>, %b: tensor<10x10xf32>,
                    %c: tensor<10x10xf32>) -> tensor<10x10xf32> {
    %0 = linalg.matmul ins(%a, %b : tensor<10x10xf32>, tensor<10x10xf32>)
                       outs(%c : tensor<10x10xf32>) -> tensor<10x10xf32>
    return %0 : tensor<10x10xf32>
  }
}
