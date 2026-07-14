// affine.apply in a real loop nest: tiled traversal of a 256-element array.
// (INSPECT-ONLY.)
//
// The outer loop steps by the tile size (32); the inner loop walks the 32
// elements of each tile. affine.apply turns the (tile, within-tile) pair into a
// flat array index `%i + %j`.
//
// Inspect, and watch the affine ops expand into scf + arith:
//   mlir-opt tiled_loop.mlir
//   mlir-opt tiled_loop.mlir -lower-affine

// Example showing how affine.apply can be used in a practical loop context
func.func @tiled_loop(%arg0: memref<256xf32>) {
  affine.for %i = 0 to 256 step 32 {       // outer: one iteration per tile
    affine.for %j = 0 to 32 {              // inner: elements within the tile
      %idx = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%i, %j)
      %val = memref.load %arg0[%idx] : memref<256xf32>
      // ... process %val ...
      memref.store %val, %arg0[%idx] : memref<256xf32>
    }
  }
  return
}
