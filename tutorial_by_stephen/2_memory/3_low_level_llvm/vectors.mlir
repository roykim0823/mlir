// MLIR vector dialect — SIMD-style operations.
//
// Vector types are independent of the target ISA; LLVM lowers them to
// AVX / NEON / etc., splits them when too wide, or scalarizes if the
// hardware has no SIMD at all.
//
// Lower with:
//   mlir-opt vectors.mlir \
//     -convert-vector-to-llvm \
//     -convert-arith-to-llvm \
//     -convert-func-to-llvm \
//     -reconcile-unrealized-casts
module {
  // Elementwise add of two 4-wide f32 vectors. Note the operation is
  // `arith.addf` applied to a vector type, not a special `vector.add`.
  func.func @vadd(%a: vector<4xf32>, %b: vector<4xf32>) -> vector<4xf32> {
    %c = arith.addf %a, %b : vector<4xf32>
    return %c : vector<4xf32>
  }

  // Elementwise multiply.
  func.func @vmul(%a: vector<4xf32>, %b: vector<4xf32>) -> vector<4xf32> {
    %c = arith.mulf %a, %b : vector<4xf32>
    return %c : vector<4xf32>
  }

  // Pull a single lane out of a vector.
  func.func @vextract(%v: vector<4xf32>) -> f32 {
    %0 = vector.extract %v[0] : f32 from vector<4xf32>
    return %0 : f32
  }

  // Shuffle: produce a new 4-vector by picking lanes from two inputs.
  // Indices 0..3 refer to %a, 4..7 refer to %b. Here we take a[0],
  // a[1], b[0], b[1].
  func.func @vshuffle(%a: vector<4xf32>, %b: vector<4xf32>)
      -> vector<4xf32> {
    %0 = vector.shuffle %a, %b [0, 1, 4, 5]
       : vector<4xf32>, vector<4xf32>
    return %0 : vector<4xf32>
  }

  // Horizontal reduction: collapse a vector to a scalar by summing
  // every lane.
  func.func @vreduce_sum(%v: vector<4xf32>) -> f32 {
    %s = vector.reduction <add>, %v : vector<4xf32> into f32
    return %s : f32
  }

  // Broadcast a scalar into every lane.
  func.func @vsplat(%x: f32) -> vector<4xf32> {
    %0 = vector.broadcast %x : f32 to vector<4xf32>
    return %0 : vector<4xf32>
  }
}
