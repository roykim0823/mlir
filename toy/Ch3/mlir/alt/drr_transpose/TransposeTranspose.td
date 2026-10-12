//===- TransposeTranspose.td - SimplifyRedundantTranspose in DRR ----------===//
//
// The transpose pattern of ToyCombine.cpp, written in DRR instead of C++.
// Not part of the build: README section 3.7 appends it to a copy of
// ToyCombine.td and registers TransposeTransposeOptPattern instead.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/PatternBase.td"
include "toy/Ops.td"

// Transpose(Transpose(x)) = x
def TransposeTransposeOptPattern : Pat<(TransposeOp(TransposeOp $arg)),
                                       (replaceWithValue $arg)>;
