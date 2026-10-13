// Affine maps and affine.apply (INSPECT-ONLY).
//
// An affine map is a multi-dimensional quasi-linear function from dimensions and
// symbols to results. Notation: (d0, d1)[s0] -> (...) has dimension operands
// d0, d1 and symbol operand s0. Maps may use only +, -, *constant, floordiv /
// ceildiv / mod by a constant; the result type is always `index`.
//
// `affine.apply` evaluates a map on concrete operands — handy for computing
// tile indices, offsets, and array addresses.
//
// Inspect / fold with:
//   mlir-opt affine_apply.mlir
//   mlir-opt affine_apply.mlir -canonicalize   # constant maps fold away

// Reusable named maps (the `#name` attribute-alias sigil from Chapter 1):
// Define some resuable affine maps
#tile_map    = affine_map<(d0) -> (d0 floordiv 32)>          // which 32-wide tile?
#offset_map  = affine_map<(d0)[s0] -> (d0 + s0)>            // shift by a symbol
#complex_map = affine_map<(d0, d1)[s0] -> (d0 * 2 + d1 floordiv 4 + s0)>

func.func @affine_apply_examples() {
  // create some test indices
  %c0   = arith.constant 0   : index
  %c42  = arith.constant 42  : index
  %c128 = arith.constant 128 : index

  // Example 1. Simple Tiling Calculation: 42 floordiv 32 = 1 (element 42 lives in tile #1).
  %tile = affine.apply #tile_map(%c42)

  // Example 2. An inline affine (anonymous) map for an offset: 42 + 10 = 52.
  %off  = affine.apply affine_map<(i) -> (i + 10)>(%c42)

  // Example 3. Using a symbol parameter: 42 + 128 = 170.  Dims in (), symbols in [].
  %shift = affine.apply #offset_map(%c42)[%c128]

  // Example 4. Multiple dims and a symbol: 42*2 + 128 floordiv 4 + 0 = 116.
  %cplx = affine.apply #complex_map(%c42, %c128)[%c0]

  // Example 5. Composition of affine maps — feed one apply's result into the next.
  %t     = affine.apply affine_map<(i) -> (i * 2)>(%c42)   // 84
  %final = affine.apply affine_map<(i) -> (i + 5)>(%t)     // 89

  return
}
