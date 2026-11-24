// A small dialect that uses something from each of the ten files under
// OpBase.td. It is only read by the commands in appendix/OpBase.md; the build
// never reads it.

include "mlir/IR/OpBase.td"

// DialectBase.td: the Dialect class.
def Demo_Dialect : Dialect {
  let name = "demo";
  let cppNamespace = "::demo";
}

// Constraints.td (And, CPred) + CommonTypeConstraints.td (Type,
// IsTensorTypePred): a type constraint of our own.
def SmallTensor : Type<
    And<[IsTensorTypePred,
         CPred<"::llvm::cast<::mlir::ShapedType>($_self).getNumElements() <= 8">]>,
    "tensor of at most 8 elements">;

// AttrTypeBase.td: a type of our own, `!demo.pair<first, second>`.
def PairType : TypeDef<Demo_Dialect, "Pair"> {
  let mnemonic = "pair";
  let parameters = (ins "::mlir::Type":$first, "::mlir::Type":$second);
  let assemblyFormat = "`<` $first `,` $second `>`";
}

// Interfaces.td: an op interface with one method.
def CostInterface : OpInterface<"CostInterface"> {
  let cppNamespace = "::demo";
  let methods = [
    InterfaceMethod<"Returns the cost of this op.", "unsigned", "getCost">
  ];
}

// OpBase.td: the Op class, OpBuilder and the Terminator trait.
def StopOp : Op<Demo_Dialect, "stop",
    [Terminator, DeclareOpInterfaceMethods<CostInterface>]> {
  // Utils.td: ins. CommonAttrConstraints.td: I64Attr. Properties.td: I32Prop.
  let arguments = (ins SmallTensor:$input, I64Attr:$code, I32Prop:$count);

  // Utils.td: CArg.
  let builders = [
    OpBuilder<(ins "::mlir::Value":$input, CArg<"int32_t", "1">:$count)>
  ];
}
