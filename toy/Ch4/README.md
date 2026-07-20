# Chapter 4: Enabling Generic Transformation with Interfaces

> **Goal:** Teach *core* MLIR passes (the inliner) and a *custom* pass (shape inference) to operate on the Toy dialect without either side hard-coding knowledge of the other — using **interfaces**.
> Official doc: <https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/>

**Chapter code in this repo:** `Ch4/` — one chapter of the out-of-tree **superbuild** described in the top-level [README](../README.md#repository-layout).

| | |
|---|---|
| Build | `cd toy && ./build.sh ch4` |
| Binary | `build/bin/toyc-ch4` |
| Run | `./run.sh ch4` → `./build/bin/toyc-ch4 ../test_Example/Toy/Ch4/codegen.toy -emit=mlir -opt` |
| Test inputs | `test_Example/Toy/Ch4/` (`codegen.toy`, `shape_inference.mlir`, ...) |

---

## 1. Overview

### The problem: generic functions and unranked tensors

Toy is intentionally "dumb" at the source level: user functions are **generic**. A function like

```toy
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}
```

says nothing about the shapes of `a` and `b`. When Chapter 2's MLIRGen lowers this to the Toy dialect, every argument and intermediate value gets the *unranked* tensor type `tensor<*xf64>` — "an f64 tensor of unknown rank and shape". Only `main`, where literals like `var a<2, 3> = ...` appear, has concrete `tensor<2x3xf64>` values. So after codegen we have a module where:

- `multiply_transpose` computes entirely on `tensor<*xf64>`;
- `main` calls it via `toy.generic_call` with statically shaped arguments but gets back a `tensor<*xf64>`.

We cannot generate efficient code (or, in Chapter 5, lower to affine loops) without knowing the actual shapes. Two things must happen:

1. **Inlining** — pull the bodies of the generic functions into `main`, so that shape information from the call sites can flow into the callee's operations. (The alternative — cloning/specializing each function per call-site signature — is what a production compiler might do; the tutorial takes the simpler intraprocedural route.)
2. **Shape inference** — propagate the known static shapes through the now-flat sequence of operations, replacing every `tensor<*xf64>` with a ranked type.

### Why interfaces?

The naive way to get inlining would be to write a Toy-specific inlining pass. But MLIR is designed around *many* coexisting dialects, and every dialect writing its own inliner (and constant folder, and CSE, and DCE...) would be an O(dialects × transformations) explosion. MLIR's answer is **interfaces**: a transformation is written once, generically, against an abstract interface; each dialect/operation *opts in* by implementing that interface. The pass never needs to know the dialect exists, and the dialect never needs to know how the pass works internally.

MLIR has two granularities of interface, and Chapter 4 uses both:

| Kind | Attached to | Used here for |
|---|---|---|
| **Dialect interface** | the whole dialect (`DialectInlinerInterface`) | answering dialect-wide questions: "may ops from this dialect be inlined?", "how do I handle your terminator?", "how do I materialize a type conversion?" |
| **Operation interface** | individual ops (`CallOpInterface`, `CallableOpInterface`, `CastOpInterface`, our own `ShapeInferenceOpInterface`) | letting a pass query/manipulate a specific op opaquely: "what do you call?", "where is your body?", "infer your result shape" |

The chapter demonstrates both directions:

- **Consuming an existing interface**: hooking Toy into MLIR's built-in **inliner** pass via `DialectInlinerInterface` + `CallOpInterface`/`CallableOpInterface` + `CastOpInterface`.
- **Defining a new interface**: declaring `ShapeInferenceOpInterface` in ODS/TableGen and writing a generic `ShapeInferencePass` that works on *any* op implementing it — Toy ops today, anyone else's ops tomorrow.

---

## 2. Inlining

MLIR ships a general-purpose inliner pass (`mlir::createInlinerPass()`). It knows nothing about Toy. To make it work on our IR we must answer four questions through interfaces:

1. *Policy*: which Toy ops/regions are legal to inline? → `DialectInlinerInterface::isLegalToInline`
2. *Mechanics*: what happens to `toy.return` when a body is spliced into the caller? → `handleTerminator`
3. *Discovery*: which ops are calls, and which ops are callable? → `CallOpInterface` on `toy.generic_call`, `CallableOpInterface` (via `FunctionOpInterface`) on `toy.func`
4. *Type mismatches*: call sites pass `tensor<2x3xf64>` but the callee's block arguments are `tensor<*xf64>` — who bridges that? → `materializeCallConversion` + a new `toy.cast` op

### 2.1 `ToyInlinerInterface` — the dialect interface

***mlir/Dialect.cpp***

```cpp
#include "mlir/Transforms/InliningUtils.h"

/// This class defines the interface for handling inlining with Toy operations.
struct ToyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// All call operations within toy can be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// All operations within toy can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  // All functions within toy can be inlined.
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator(toy.return) by replacing it with a new
  /// operation as necessary.
  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    // Only "toy.return" needs to be handled here.
    auto returnOp = cast<ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }

  /// Attempts to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. ...
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    return builder.create<CastOp>(conversionLoc, resultType, input);
  }
};
```

What each hook means:

- **Three `isLegalToInline` overloads** — the inliner asks progressively finer-grained questions:
  - *(call, callable, wouldBeCloned)*: may this particular call to this particular callable be inlined at all? `wouldBeCloned` is `true` if the callee body would be *copied* (other uses remain) rather than moved.
  - *(Operation, Region, ...)*: may this specific operation be moved into the destination region? The `IRMapping` maps callee values to their caller-side replacements.
  - *(Region, Region, ...)*: may the source region as a whole be inlined into the destination region?

  Toy has no side-effectful, region-sensitive, or otherwise inlining-hostile operations, so all three return `true` unconditionally. A real dialect would inspect the operands/attributes here (e.g. refuse to inline ops that depend on function-scoped state).

- **`handleTerminator`** — after the callee's blocks are spliced into the caller, the callee's terminator (`toy.return %2`) must disappear; whatever it returned must become the value(s) that the original `toy.generic_call` produced. `valuesToRepl` are exactly those call results, so we RAUW each one with the corresponding return operand. The inliner then erases the terminator.

- **`materializeCallConversion`** — covered in §2.3.

The interface is registered on the dialect in `ToyDialect::initialize()` (also `Dialect.cpp`):

```cpp
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
      >();
  addInterfaces<ToyInlinerInterface>();
}
```

### 2.2 Marking calls and callables — `CallOpInterface` / `CallableOpInterface`

The inliner also needs to *find* the call graph. It does this via two op interfaces from `mlir/Interfaces/CallInterfaces.td`:

- `CallableOpInterface` — "I am a thing that can be called; here is my region and my signature."
- `CallOpInterface` — "I am a call; here is who I call and with what arguments."

In `include/toy/Ops.td` the interfaces are attached declaratively:

***include/toy/Ops.td***

```tablegen
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"

def FuncOp : Toy_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  ...
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);
  ...
}

def GenericCallOp : Toy_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  ...
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<F64Tensor>:$inputs,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let results = (outs F64Tensor);
  ...
}
```

Notes on the ODS side:

- `toy.func` gets `CallableOpInterface` *transitively* through `FunctionOpInterface` (which implies it). The `arg_attrs` / `res_attrs` optional attributes are required by the call interfaces in modern MLIR (they carry per-argument/result attribute dictionaries across inlining) — both `FuncOp` and `GenericCallOp` declare them.
- `DeclareOpInterfaceMethods<CallOpInterface>` is the key ODS construct: it attaches the interface **and** declares its non-defaulted methods on the generated C++ op class, leaving their bodies for us to write in `Dialect.cpp`. (Contrast with just listing the interface, which would expect default implementations to suffice.)

The C++ implementations in `Dialect.cpp`:

***include/toy/Ops.td***

```cpp
// FuncOp — the CallableOpInterface side (in Ops.td's extraClassDeclaration):
ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
ArrayRef<Type> getResultTypes()   { return getFunctionType().getResults(); }
Region *getCallableRegion()       { return &getBody(); }
```

`getCallableRegion()` returns the region the inliner should clone from — the function body. (It is defined inline in `Ops.td`'s `extraClassDeclaration` block for `FuncOp`.)

***mlir/Dialect.cpp***

```cpp
// GenericCallOp — the CallOpInterface side (Dialect.cpp):

/// Return the callee of the generic call operation.
CallInterfaceCallable GenericCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// Set the callee for the generic call operation.
void GenericCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
}

/// Get the argument operands to the called function.
Operation::operand_range GenericCallOp::getArgOperands() { return getInputs(); }

/// Get the argument operands as a mutable range.
MutableOperandRange GenericCallOp::getArgOperandsMutable() {
  return getInputsMutable();
}
```

- `getCallableForCallee()` returns a `CallInterfaceCallable` — either an SSA value (indirect call) or, as here, a `SymbolRefAttr` naming the callee. The inliner resolves the symbol to the `toy.func` in the module's symbol table.
- `getArgOperands()` / `getArgOperandsMutable()` tell the inliner which operands map to the callee's block arguments.

One more prerequisite lives in `mlir/MLIRGen.cpp`: every function except `main` is marked **private**, so that once all its call sites are inlined away, the symbol-DCE built into the inliner can delete the dead function body:

***mlir/MLIRGen.cpp***

```cpp
// If this function isn't main, then set the visibility to private.
if (funcAST.getProto()->getName() != "main")
  function.setPrivate();
```

This is why the "without `-opt`" dump in §6 prints `toy.func private @multiply_transpose(...)`.

### 2.3 `toy.cast` — bridging the type mismatch

There is a subtlety: at the call site the arguments are `tensor<2x3xf64>`, but the callee's block arguments are `tensor<*xf64>`. If the inliner blindly wired caller SSA values into the callee body, operand types would silently change — the IR might not even verify. The inliner therefore asks the dialect to **materialize an explicit conversion** for every mismatched argument/result, via the `materializeCallConversion` hook shown in §2.1:

***mlir/Dialect.cpp***

```cpp
Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                     Type resultType,
                                     Location conversionLoc) const final {
  return builder.create<CastOp>(conversionLoc, resultType, input);
}
```

If this hook returned `nullptr` for some pair of types, the inliner would simply refuse to inline that call. We support it by introducing a dedicated op, `toy.cast`, in `Ops.td`:

***include/toy/Ops.td***

```tablegen
def CastOp : Toy_Op<"cast", [
     DeclareOpInterfaceMethods<CastOpInterface>,
     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
     Pure,
     SameOperandsAndResultShape
  ]> {
  let summary = "shape cast operation";
  let description = [{
    The "cast" operation converts a tensor from one type to an equivalent type
    without changing any data elements. The source and destination types must
    both be tensor types with the same element type. If both are ranked, then
    shape is required to match. The operation is invalid if converting to a
    mismatching constant dimension.
  }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}
```

Trait/interface breakdown:

- **`CastOpInterface`** marks the op as a pure type cast; generic utilities (verification, folding of redundant casts by the canonicalizer, `foldCastOp` helpers) can then reason about it. Its one required method, `areCastCompatible`, is implemented in `Dialect.cpp`:

  ***mlir/Dialect.cpp***

  ```cpp
  bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
    if (inputs.size() != 1 || outputs.size() != 1)
      return false;
    // The inputs must be Tensors with the same element type.
    TensorType input = llvm::dyn_cast<TensorType>(inputs.front());
    TensorType output = llvm::dyn_cast<TensorType>(outputs.front());
    if (!input || !output || input.getElementType() != output.getElementType())
      return false;
    // The shape is required to match if both types are ranked.
    return !input.hasRank() || !output.hasRank() || input == output;
  }
  ```

  So `tensor<2x3xf64> → tensor<*xf64>` (rank erasure) and the reverse (rank refinement) are legal; `tensor<2x3xf64> → tensor<3x2xf64>` is not.

- **`Pure`** — no side effects; dead casts can be removed.
- **`SameOperandsAndResultShape`** — if both sides *are* ranked, shapes must agree (this is what lets shape inference turn a cast into the identity, which the canonicalizer then folds away).
- It also declares **`ShapeInferenceOpInterface`** — after inlining, the casts are exactly the ops through which static shapes must flow (§3.3).

### 2.4 Registering the inliner pass

With policy, discovery, and conversion in place, enabling inlining is one line in `toyc.cpp`:

***toyc.cpp***

```cpp
// Inline all functions into main and then delete them.
pm.addPass(mlir::createInlinerPass());
```

The inliner is a *module*-level pass: it builds the call graph from `CallOpInterface`/`CallableOpInterface`, inlines bottom-up, runs canonicalization on the intermediate results, and erases now-unreferenced private functions.

**Actual IR right after the inliner** — reproduce with the pass-manager debug flag and read the first dump block (note the two materialized `toy.cast` ops and that `@multiply_transpose` is gone):

```bash
cd toy
./build/bin/toyc-ch4 ../test_Example/Toy/Ch4/codegen.toy -emit=mlir -opt --mlir-print-ir-after-all 2>&1
```

```mlir
// -----// IR Dump After Inliner (inline) //----- //
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %2 = toy.cast %1 : tensor<2x3xf64> to tensor<*xf64>
    %3 = toy.cast %0 : tensor<2x3xf64> to tensor<*xf64>
    %4 = toy.transpose(%2 : tensor<*xf64>) to tensor<*xf64>
    %5 = toy.transpose(%3 : tensor<*xf64>) to tensor<*xf64>
    %6 = toy.mul %4, %5 : tensor<*xf64>
    toy.print %6 : tensor<*xf64>
    toy.return
  }
}
```

(Only one of the two original calls survives to this point: the inliner's built-in canonicalization already removed the unused `var c = multiply_transpose(a, b)` — `toy.generic_call` results were dead and the ops involved are `Pure`. The `toy.reshape`s were folded by the Chapter 3 canonicalization patterns, too.)

---

## 3. Shape Inference with a Custom Op Interface

After inlining we have a single flat `main`, but the body still computes on `tensor<*xf64>`. Now we propagate shapes. Again we resist writing "a pass that switches over Toy op names"; instead we define a **new operation interface** so the pass stays generic and other dialects could plug into it.

### 3.1 Declaring the interface in ODS

***include/toy/ShapeInferenceInterface.td***

```tablegen
#ifndef SHAPE_INFERENCE_INTERFACE
#define SHAPE_INFERENCE_INTERFACE

include "mlir/IR/OpBase.td"

def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "void", "inferShapes">
  ];
}

#endif // SHAPE_INFERENCE_INTERFACE
```

Anatomy:

- `OpInterface<"ShapeInference">` — the TableGen def name (`ShapeInferenceOpInterface`) is what you reference in other `.td` files; the string `"ShapeInference"` is the name of the **generated C++ class** (`mlir::toy::ShapeInference`) that passes will `dyn_cast` to.
- `InterfaceMethod<description, returnType, methodName>` — declares one virtual-like method, `void inferShapes()`. `InterfaceMethod` can also take an argument list, a default implementation body, and a static marker; here the simplest form suffices.

`mlir-tblgen` turns this into two generated files (see §5): `ShapeInferenceOpInterfaces.h.inc` (the `ShapeInference` class declaration) and `ShapeInferenceOpInterfaces.cpp.inc` (its concept/model machinery). They are pulled into the build by two hand-written files:

***include/toy/ShapeInferenceInterface.h***

```cpp
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace toy {

/// Include the auto-generated declarations.
#include "toy/ShapeInferenceOpInterfaces.h.inc"

} // namespace toy
} // namespace mlir
```

and the `.cpp.inc` is included once in `ShapeInferencePass.cpp` (§3.3).

Internally MLIR uses a *concept-based polymorphism* model (like `llvm::Any`/type-erasure): the generated `ShapeInference` class wraps an `Operation*` plus a vtable-like "model" per registered op class. This is why an interface can be attached to ops from unrelated dialects without a common C++ base class, and why `dyn_cast<ShapeInference>(op)` works on a raw `Operation*`.

### 3.2 Attaching the interface to ops — `DeclareOpInterfaceMethods`

In `Ops.td`, every op whose result shape can be computed from its operand shapes opts in:

***include/toy/Ops.td***

```tablegen
include "toy/ShapeInferenceInterface.td"

def AddOp : Toy_Op<"add",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> { ... }

def MulOp : Toy_Op<"mul",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> { ... }

def TransposeOp : Toy_Op<"transpose",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> { ... }

def CastOp : Toy_Op<"cast", [
     DeclareOpInterfaceMethods<CastOpInterface>,
     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
     Pure, SameOperandsAndResultShape]> { ... }
```

`DeclareOpInterfaceMethods<ShapeInferenceOpInterface>` adds a `void inferShapes();` declaration to each generated op class; we provide the definitions in `Dialect.cpp`. Ops that don't need inference don't participate: `toy.constant` and `toy.reshape` already produce statically shaped results by construction (`StaticShapeTensorOf<[F64]>` for reshape), `toy.print`/`toy.return` have no results, and `toy.generic_call` no longer exists after inlining.

### 3.3 Implementing `inferShapes()` per op

All four implementations are small; each *refines the result type in place* from the (already-ranked) operand types:

***mlir/Dialect.cpp***

```cpp
/// AddOp: element-wise — output shape equals input shape.
void AddOp::inferShapes() { getResult().setType(getLhs().getType()); }

/// MulOp: element-wise — output shape equals input shape.
void MulOp::inferShapes() { getResult().setType(getLhs().getType()); }

/// CastOp: shape passes through unchanged (only the "ranked-ness" changes).
void CastOp::inferShapes() { getResult().setType(getInput().getType()); }

/// TransposeOp: reverse the input dimensions.
void TransposeOp::inferShapes() {
  auto arrayTy = llvm::cast<RankedTensorType>(getOperand().getType());
  SmallVector<int64_t, 2> dims(llvm::reverse(arrayTy.getShape()));
  getResult().setType(RankedTensorType::get(dims, arrayTy.getElementType()));
}
```

Points worth noticing:

- `CastOp::inferShapes()` sets the result type equal to the input type — turning `toy.cast %1 : tensor<2x3xf64> to tensor<*xf64>` into the identity cast `tensor<2x3xf64> to tensor<2x3xf64>`. That is what makes the cast *foldable* later: the canonicalizer's cast-folding logic (enabled by `CastOpInterface`) removes same-type casts.
- `TransposeOp::inferShapes()` may safely `cast<RankedTensorType>` because the pass only calls `inferShapes()` once **all** operands are ranked (see the `allOperandsInferred` gate below).
- Mutating result types in place is fine *within* a function here because every consumer of these values is itself either shape-inference-capable or shape-agnostic (`toy.print` accepts any `F64Tensor`); `ReturnOp::verify()` in `Dialect.cpp` also deliberately tolerates unranked/ranked mismatches against the function signature.

### 3.4 The `ShapeInferencePass` — a generic worklist algorithm

First, the generated interface definitions are linked in:

***mlir/ShapeInferencePass.cpp***

```cpp
#include "toy/ShapeInferenceInterface.h"
...
/// Include the auto-generated definitions for the shape inference interfaces.
#include "toy/ShapeInferenceOpInterfaces.cpp.inc"
```

The pass itself is a function-level pass over `toy::FuncOp`:

```cpp
/// The ShapeInferencePass is a pass that performs intra-procedural
/// shape inference.
///
///    Algorithm:
///
///   1) Build a worklist containing all the operations that return a
///      dynamically shaped tensor: these are the operations that need shape
///      inference.
///   2) Iterate on the worklist:
///     a) find an operation to process: the next ready operation in the
///        worklist has all of its arguments non-generic,
///     b) if no operation is found, break out of the loop,
///     c) remove the operation from the worklist,
///     d) infer the shape of its output from the argument types.
///   3) If the worklist is empty, the algorithm succeeded.
///
struct ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass, OperationPass<toy::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)
  StringRef getArgument() const override { return "toy-shape-inference"; }

  void runOnOperation() override {
    auto f = getOperation();

    // Populate the worklist with the operations that need shape inference:
    // these are operations that return a dynamic shape.
    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    f.walk([&](mlir::Operation *op) {
      if (returnsDynamicShape(op))
        opWorklist.insert(op);
    });

    // Iterate on the operations in the worklist until all operations have been
    // inferred or no change happened (fix point).
    while (!opWorklist.empty()) {
      // Find the next operation ready for inference, that is an operation
      // with all operands already resolved (non-generic).
      auto nextop = llvm::find_if(opWorklist, allOperandsInferred);
      if (nextop == opWorklist.end())
        break;

      Operation *op = *nextop;
      opWorklist.erase(op);

      // Ask the operation to infer its output shapes.
      LLVM_DEBUG(llvm::dbgs() << "Inferring shape for: " << *op << "\n");
      if (auto shapeOp = dyn_cast<ShapeInference>(op)) {
        shapeOp.inferShapes();
      } else {
        op->emitError("unable to infer shape of operation without shape "
                      "inference interface");
        return signalPassFailure();
      }
    }

    // If the operation worklist isn't empty, this indicates a failure.
    if (!opWorklist.empty()) {
      f.emitError("Shape inference failed, ")
          << opWorklist.size() << " operations couldn't be inferred\n";
      signalPassFailure();
    }
  }

  /// A utility method that returns if the given operation has all of its
  /// operands inferred.
  static bool allOperandsInferred(Operation *op) {
    return llvm::all_of(op->getOperandTypes(), [](Type operandType) {
      return llvm::isa<RankedTensorType>(operandType);
    });
  }

  /// A utility method that returns if the given operation has a dynamically
  /// shaped result.
  static bool returnsDynamicShape(Operation *op) {
    return llvm::any_of(op->getResultTypes(), [](Type resultType) {
      return !llvm::isa<RankedTensorType>(resultType);
    });
  }
};
```

Walk through the design:

- **Worklist seeding** (`returnsDynamicShape`): any op with at least one non-`RankedTensorType` result needs inference. After inlining, that's the two `toy.cast`s, the two `toy.transpose`s, and the `toy.mul`.
- **Readiness test** (`allOperandsInferred`): an op can only be inferred once *all* of its operands are ranked. Initially only the casts qualify (their inputs are `toy.constant` results, already `tensor<2x3xf64>`). Inferring the casts makes the transposes ready; inferring those makes the mul ready. Shapes thus flow forward through use-def chains — a classic dataflow fixpoint, here implemented with a simple "find any ready op" scan since Toy IR is small and acyclic.
- **The interface dispatch** is the whole point of the chapter: `dyn_cast<ShapeInference>(op)` asks "does this op — whatever dialect it belongs to — implement the interface?" The pass never mentions `AddOp`, `MulOp`, etc. If an op needs inference but doesn't implement the interface, that's a hard error.
- **Failure detection**: if the loop stalls (no ready op) with work remaining, shapes couldn't be fully resolved — e.g. a `toy.generic_call` survived because inlining didn't run first. The pass reports and fails. You can watch the per-op inference with `-debug-only=shape-inference` (the `DEBUG_TYPE` at the top of the file) in a debug build.

The pass is exposed through a factory declared in `include/toy/Passes.h`:

***include/toy/Passes.h***

```cpp
namespace mlir {
class Pass;
namespace toy {
std::unique_ptr<Pass> createShapeInferencePass();
} // namespace toy
} // namespace mlir
```

***mlir/ShapeInferencePass.cpp***

```cpp
/// Create a Shape Inference pass.  (ShapeInferencePass.cpp)
std::unique_ptr<mlir::Pass> mlir::toy::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}
```

**Actual IR right after shape inference** (from this repo). Every type is now ranked, and both casts have become identities:

```mlir
// -----// IR Dump After (anonymous namespace)::ShapeInferencePass (toy-shape-inference) //----- //
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.cast %1 : tensor<2x3xf64> to tensor<2x3xf64>
  %3 = toy.cast %0 : tensor<2x3xf64> to tensor<2x3xf64>
  %4 = toy.transpose(%2 : tensor<2x3xf64>) to tensor<3x2xf64>
  %5 = toy.transpose(%3 : tensor<2x3xf64>) to tensor<3x2xf64>
  %6 = toy.mul %4, %5 : tensor<3x2xf64>
  toy.print %6 : tensor<3x2xf64>
  toy.return
}
```

---

## 4. The Pass Pipeline

The `-opt` pipeline in `toyc.cpp` (`dumpMLIR()`):

***toyc.cpp***

```cpp
if (enableOpt) {
  mlir::PassManager pm(module.get()->getName());
  // Apply any generic pass manager command line options and run the pipeline.
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;

  // Inline all functions into main and then delete them.
  pm.addPass(mlir::createInlinerPass());

  // Now that there is only one function, we can infer the shapes of each of
  // the operations.
  mlir::OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
  optPM.addPass(mlir::toy::createShapeInferencePass());
  optPM.addPass(mlir::createCanonicalizerPass());
  optPM.addPass(mlir::createCSEPass());

  if (mlir::failed(pm.run(*module)))
    return 4;
}
```

Order matters, and each step enables the next:

1. **`createInlinerPass()`** (module scope) — must run first. Shape inference is *intraprocedural*; it cannot see through `toy.generic_call`. Inlining brings the callee bodies to where the concrete shapes live and inserts `toy.cast` bridges. It also DCEs the now-unused `private` functions.
2. **`pm.nest<toy::FuncOp>()`** — the remaining passes are *function*-level, so they are nested to run on every `toy.func` inside the module (in parallel, in principle). Note the nesting is on **`toy::FuncOp`**, not the builtin `func::FuncOp` — the pass manager anchors on the exact op type.
3. **`createShapeInferencePass()`** — the worklist algorithm from §3.4 replaces every `tensor<*xf64>` with a ranked type and degrades the casts to identities.
4. **`createCanonicalizerPass()`** — now the Chapter 3 patterns plus the `CastOpInterface`-driven folding clean up: identity `toy.cast`s vanish, and any transpose/reshape simplifications fire *with full shape knowledge*. Running it before shape inference would miss the cast folds (a cast to `tensor<*xf64>` is not redundant yet).
5. **`createCSEPass()`** — after canonicalization the two constants are structurally identical and the two transposes take the same operand; CSE merges them, shrinking `main` to one constant, one transpose, one mul.

You can watch the whole cascade yourself:

```bash
cd toy
./build/bin/toyc-ch4 ../test_Example/Toy/Ch4/codegen.toy -emit=mlir -opt --mlir-print-ir-after-all
```

(This works because `main()` calls `mlir::registerPassManagerCLOptions()` and the pipeline applies them via `applyPassManagerCLOptions(pm)`.)

---

## 5. Building

The shared machinery (superbuild, presets, dual-mode guard, `build.sh`) is documented in the top-level [README, "The build system"](../README.md#the-build-system); the op/dialect TableGen pattern is in [Ch2 §6.1](../Ch2/README.md) and the DRR rewriter step in [Ch3 §5.1](../Ch3/README.md). Build this chapter with `cd toy && ./build.sh ch4` → `./build/bin/toyc-ch4`.

### 5.1 What Chapter 4 adds to the build: interface TableGen

Relative to Chapter 3, [`Ch4/CMakeLists.txt`](CMakeLists.txt) adds:

- **`mlir/ShapeInferencePass.cpp`** in the source list.
- A third TableGen flavor and its dependency: **`ToyCh4ShapeInferenceInterfaceIncGen`** must run before compiling, since `ShapeInferenceInterface.h` includes a generated `.h.inc`:

  ***CMakeLists.txt***

  ```cmake
  add_dependencies(toyc-ch4 ToyCh4OpsIncGen)
  add_dependencies(toyc-ch4 ToyCh4ShapeInferenceInterfaceIncGen)
  add_dependencies(toyc-ch4 ToyCh4CombineIncGen)
  ```

The interface generation itself lives in [`include/toy/CMakeLists.txt`](include/toy/CMakeLists.txt), next to the Ch2-style op generation:

***include/toy/CMakeLists.txt***

```cmake
# Most dialects should use add_mlir_interfaces().
set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)
mlir_tablegen(ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ToyCh4ShapeInferenceInterfaceIncGen)
```

- **`-gen-op-interface-decls`** → `ShapeInferenceOpInterfaces.h.inc` — the `ShapeInference` interface class (included by `ShapeInferenceInterface.h`);
- **`-gen-op-interface-defs`** → `ShapeInferenceOpInterfaces.cpp.inc` — the interface's registration/model definitions (included by `ShapeInferencePass.cpp`).

(As the file's comments note, upstream projects would typically wrap all of this in the `add_mlir_dialect()` / `add_mlir_interfaces()` convenience macros — this repo spells the steps out, which is more instructive.) In the superbuild everything generates into `toy/build/Ch4/include/toy/`.

Linking is still just the monolithic `MLIR` + `LLVM` dylibs — the upstream chapter would add `MLIRCallInterfaces`, `MLIRCastInterfaces`, `MLIRFunctionInterfaces`, `MLIRTransforms`, etc. as individual components.

---

## 6. Running and Testing

### 6.1 The input program

***test_Example/Toy/Ch4/codegen.toy***

```toy
# User defined generic function that operates on unknown shaped arguments
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  var d = multiply_transpose(b, a);
  print(d);
}
```

Note: `a` and `b` hold the *same six values* shaped `<2,3>` (one from a nested literal, one reshaped from a flat literal), `c` is computed but **never used**, and only `d` is printed. Each of these details is visible in what the optimizer does below.

### 6.2 Without `-opt` — the raw codegen

```bash
cd toy
./build/bin/toyc-ch4 ../test_Example/Toy/Ch4/codegen.toy -emit=mlir 2>&1
```

Actual output:

```mlir
module {
  toy.func private @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
    %2 = toy.mul %0, %1 : tensor<*xf64>
    toy.return %2 : tensor<*xf64>
  }
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
    %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
    %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64>
    %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    toy.print %5 : tensor<*xf64>
    toy.return
  }
}
```

What to observe:

- `@multiply_transpose` is **`private`** (the `MLIRGen.cpp` `setPrivate()` from §2.2) and fully **generic**: every type inside it is `tensor<*xf64>`.
- Both `toy.generic_call`s pass ranked `tensor<2x3xf64>` arguments but produce **unranked** `tensor<*xf64>` results — the shape information dies at the call boundary.
- The dead `%4` (variable `c`) and the redundant reshapes are still present — no optimization has run.

### 6.3 With `-opt` — inlined, shape-inferred, canonicalized, CSE'd

```bash
cd toy
./build/bin/toyc-ch4 ../test_Example/Toy/Ch4/codegen.toy -emit=mlir -opt 2>&1
# or via the wrapper: ./run.sh ch4
```

Actual output:

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
    %2 = toy.mul %1, %1 : tensor<3x2xf64>
    toy.print %2 : tensor<3x2xf64>
    toy.return
  }
}
```

Everything this chapter built is visible in the diff:

| Before (`-emit=mlir`) | After (`-emit=mlir -opt`) | Which mechanism |
|---|---|---|
| `toy.func private @multiply_transpose` exists | **gone** — dead after inlining | inliner + private visibility (§2.1, §2.2) |
| two `toy.generic_call` ops | **gone** — bodies spliced into `main` | `CallOpInterface`/`CallableOpInterface` (§2.2) |
| `tensor<*xf64>` everywhere in the callee | every type ranked: `tensor<2x3xf64>`, `tensor<3x2xf64>` | `ShapeInferenceOpInterface` + pass (§3) |
| (transiently) `toy.cast ... to tensor<*xf64>` after inlining | **gone** — inferred to identity, folded | `materializeCallConversion` + `CastOpInterface` + canonicalizer (§2.3, §3.3) |
| unused call `%4` (variable `c`), redundant reshapes | **gone** | canonicalizer / DCE on `Pure` ops |
| two identical constants, two identical transposes | **one** constant, **one** transpose, `toy.mul %1, %1` | CSE (§4) |

### 6.4 Watching the intermediate stages

The full pass-by-pass story (this is real output from this repo, elided):

```bash
./build/bin/toyc-ch4 ../test_Example/Toy/Ch4/codegen.toy -emit=mlir -opt --mlir-print-ir-after-all 2>&1
```

```mlir
// -----// IR Dump After Inliner (inline) //----- //
//   casts materialized, callee body inlined, @multiply_transpose deleted:
%2 = toy.cast %1 : tensor<2x3xf64> to tensor<*xf64>
%3 = toy.cast %0 : tensor<2x3xf64> to tensor<*xf64>
%4 = toy.transpose(%2 : tensor<*xf64>) to tensor<*xf64>
...

// -----// IR Dump After ShapeInferencePass (toy-shape-inference) //----- //
//   all types ranked; casts are now identities:
%2 = toy.cast %1 : tensor<2x3xf64> to tensor<2x3xf64>
%4 = toy.transpose(%2 : tensor<2x3xf64>) to tensor<3x2xf64>
%6 = toy.mul %4, %5 : tensor<3x2xf64>
...

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
//   identity casts folded away:
%2 = toy.transpose(%1 : tensor<2x3xf64>) to tensor<3x2xf64>
%3 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
%4 = toy.mul %2, %3 : tensor<3x2xf64>

// -----// IR Dump After CSE (cse) //----- //
//   duplicate constant and transpose merged:
%1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
%2 = toy.mul %1, %1 : tensor<3x2xf64>
```

**The ecosystem view.** Look at the parenthesized names in those dump headers: `(inline)`, `(canonicalize)`, `(cse)` are the *registered pass names*, and three of the four passes in this chapter's pipeline are stock MLIR passes — the same ones any `mlir-opt`-family tool exposes as `--inline`, `--canonicalize`, `--cse`. Only `(toy-shape-inference)` is chapter-local. `toyc-ch4 -opt` is therefore just a hardcoded rendering of the pipeline `mlir-opt --pass-pipeline='builtin.module(inline, toy.func(toy-shape-inference, canonicalize, cse))'` inside a binary that happens to link the Toy dialect and its one custom pass — which is precisely how real MLIR projects structure their `foo-opt` tools.

### 6.5 Other test inputs

`test_Example/Toy/Ch4/` also contains `shape_inference.mlir` (a pre-written `.mlir` module exercising the pipeline directly — feed it with `-x mlir`... or just by its `.mlir` extension, which `loadMLIR()` in `toyc.cpp` detects), plus `ast.toy`, `scalar.toy`, `trivial_reshape.toy`, `transpose_transpose.toy`, and `invalid.mlir` from earlier chapters:

```bash
./build/bin/toyc-ch4 ../test_Example/Toy/Ch4/shape_inference.mlir -emit=mlir -opt
```

---

## 7. Key Takeaways & Pitfalls

**Takeaways**

1. **Interfaces invert the dependency.** The inliner never learned about Toy; Toy taught itself to the inliner. This is the core scaling trick of MLIR: transformations are O(1) per new dialect, not O(dialects).
2. **Dialect interfaces vs. op interfaces.** Use a *dialect* interface for blanket policy (`ToyInlinerInterface`: "everything is inlinable", terminator handling, cast materialization); use *op* interfaces for per-op capabilities (`CallOpInterface`, `ShapeInferenceOpInterface`).
3. **`DeclareOpInterfaceMethods<...>` is the ODS glue.** It both attaches the interface and declares the methods on the generated op class so you implement them in your `.cpp` — forget it and you get default-only behavior or link errors.
4. **Casts make type refinement safe.** Rather than mutating types across a call boundary, the inliner asks the dialect to materialize explicit `toy.cast`s; shape inference later proves them trivial and the canonicalizer deletes them. Explicit-then-fold is a recurring MLIR idiom.
5. **A custom interface is ~15 lines of TableGen.** `OpInterface` + `InterfaceMethod` + two `mlir_tablegen` invocations gives you a `dyn_cast`-able C++ class usable across dialects.
6. **Pipeline order encodes the reasoning**: inline (bring shapes to the code) → infer (propagate them) → canonicalize (exploit them) → CSE (deduplicate what canonicalization exposed).

**Pitfalls**

- **Forgetting `addInterfaces<ToyInlinerInterface>()`** in `ToyDialect::initialize()` — the inliner silently declines to inline anything from your dialect (all `isLegalToInline` queries default to "no").
- **Forgetting `setPrivate()`** on non-`main` functions — inlining still happens, but the dead `toy.func`s are never deleted (public symbols are presumed externally visible).
- **Missing `arg_attrs`/`res_attrs`** on `FuncOp`/`GenericCallOp` in MLIR 18+ — the call interfaces expect these optional attributes; older tutorial code without them fails tablegen/verification against modern MLIR.
- **Nesting the function passes on the wrong op**: it must be `pm.nest<mlir::toy::FuncOp>()`. Nesting on builtin `func::FuncOp` runs the shape-inference pass on zero functions and the IR stays unranked — an easy, silent mistake.
- **Running shape inference without inlining first**: `toy.generic_call` produces `tensor<*xf64>` and does not implement `ShapeInference`, so the pass hard-fails ("unable to infer shape of operation without shape inference interface").
- **`inferShapes()` ordering assumptions**: implementations like `TransposeOp::inferShapes()` `cast<RankedTensorType>` unconditionally — safe only because the pass gates on `allOperandsInferred`. Reusing an `inferShapes` outside that discipline will assert.
- **Out-of-tree specifics**: this project links monolithic `MLIR`/`LLVM` dylibs; if you instead copy upstream's per-component `target_link_libraries`, be sure the Homebrew build actually ships those static libs. Also remember `add_dependencies(toyc-ch4 ToyCh4ShapeInferenceInterfaceIncGen)` — without it, parallel Ninja builds race to compile `ShapeInferencePass.cpp` before its `.inc` files exist.

---

## Links

- Official doc: [Toy Tutorial Chapter 4 — Enabling Generic Transformation with Interfaces](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/)
- Previous: [Chapter 3: High-level Language-Specific Analysis and Transformation](../Ch3/README.md)
- Next: [Chapter 5: Partial Lowering to Lower-Level Dialects for Optimization](../Ch5/README.md)
- Back to [README](../README.md)
- Related MLIR docs: [Interfaces](https://mlir.llvm.org/docs/Interfaces/), [Operation Definition Specification (ODS)](https://mlir.llvm.org/docs/DefiningDialects/Operations/), [Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/)
