// Super-vectorization. INSPECT-ONLY.
//
// Turns a scalar affine loop into a vector one, replacing per-element loads/adds
// with `vector.transfer_read` / `arith.addf : vector<Nxf32>` / `transfer_write`.
// This is the affine dialect's own path to SIMD (Chapter 2 met the vector dialect
// directly).
//
//   mlir-opt vectorize.mlir -affine-super-vectorize="virtual-vector-size=8"
//
// After: the loop steps by 8 and the body works on `vector<8xf32>` at a time.
func.func @vectorize(%A: memref<256xf32>, %B: memref<256xf32>) {
  affine.for %i = 0 to 256 {
    %a = affine.load %A[%i] : memref<256xf32>
    %b = affine.load %B[%i] : memref<256xf32>
    %s = arith.addf %a, %b : f32
    affine.store %s, %A[%i] : memref<256xf32>
  }
  return
}
