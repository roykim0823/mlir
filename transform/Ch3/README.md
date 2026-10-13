# Chapter 3: More than Simple Transform Operations

> **Goal:** Shrink a transform operation to its essence with **type constraints** and the **`TransformEachOpTrait`**, define a **new transform type** (a handle to anything implementing `CallOpInterface`), write an operation that **consumes** its operand and returns a new handle, and replace hand-written memory effects with **memory-effect traits** — based on the official tutorial [Transform Ch3: More than Simple Transform Operations](https://mlir.llvm.org/docs/Tutorials/transform/Ch3/).

---

## 1. Overview

Chapter 2 added `transform.my.change_call_target`, whose hand-written `apply` method fetched the payload operations of its handle, looped over them, checked by hand that each one was a `func.call`, and rewrote it. Every transform operation that works "one payload operation at a time" repeats that loop and that check. This chapter removes the repetition and extends the extension:

1. **Type constraints and `TransformEachOpTrait`** (section 2): the operand type `!transform.op<"func.call">` states the precondition, and the trait supplies the loop, so the operation only implements `applyToOne` for a single `func::CallOp`.
2. **A transform type of our own** (section 3): `!transform.my.call_op_interface`, a handle whose payload operations must implement `CallOpInterface`, defined in ODS (Operation Definition Specification) with `TypeDef` and checked by a C++ `checkPayload` method.
3. **Operand consumption** (section 4): `transform.my.call_to_op` replaces each call with an (unregistered) `my.mm4` operation, so it *consumes* its operand handle and *produces* a new one.
4. **Memory-effect traits** (section 5): `FunctionalStyleTransformOpTrait` and `NavigationTransformOpTrait` declare the common effect patterns, so `call_to_op` needs no `getEffects` of its own.

The chapter's code lives in `include/MyExtension.td` (both operations), the new `include/MyExtensionTypes.td` (the type), `lib/MyExtension.cpp` (the extension, `applyToOne` bodies, `checkPayload`) and `transform-opt/transform-opt.cpp` (the `transform-opt-ch3` driver, unchanged apart from its name). Inputs are the chapter's tests in `test_Example/transform/Ch3/`.

Sections 2–5 follow the chapter's path; section 6 builds `transform-opt-ch3` and runs the complete examples; section 7 shows how to derive this chapter's operations, type and build wiring yourself from the installed headers and tools. The appendices hold the documentation ODS generates for the extension (A), what `qualified(...)` in the assembly format changes (B), the build wiring (C), and the script that runs every command in this README (D).

All commands below use the chapter's own tool, `./build/transform-opt-ch3`; build it first (section 6.1).

---

## 2. Type Constraints and the ApplyEach Trait

A transform operation that applies to each payload operation individually and requires it to be of a specific kind is a repeated pattern. Transform dialect **types** can express the precondition: instead of the wide `TransformHandleTypeInterface` (any handle), the operand of `change_call_target` now has the narrow type `Transform_ConcreteOpType<"func.call">`, i.e. `!transform.op<"func.call">`. And the `TransformEachOpTrait` trait provides the skeleton of the `apply` method — verification of each payload operation's kind, iteration over the payload, and concatenation of results — so the operation only says what to do with *one* payload operation.

### 2.1 The improved ODS definition

***include/MyExtension.td***
```tablegen
// Define the new operation. By convention, prefix its name with the name of the dialect 
// extension, "my.". The full operation name will be further prefixed with "transform.".
def ChangeCallTargetOp : Op<Transform_Dialect, "my.change_call_target",
    // Indicate that the operation implements the required TransformOpInterface and
    // MemoryEffectsOpInterface. Use the TransformEach trait to provide the 
    // implementation for TransformOpInterface.
    [TransformOpInterface, TransformEachOpTrait,                                          // (1)
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {                              // (2)
  // Provide a brief and a full description. It is recommended that the latter describes 
  // the effects on the operands and how the operation processes various failure modes.
  let summary = "Changes the callee of a call operation to the specified one";
  let description = [{
    ...
  }];

  // The arguments include the handle to the payload operations and the attribute that 
  // specifies the new callee. The handle must implement TransformHandleTypeInterface.   
  // We use a string attribute as the symbol may not exist in the transform IR so the 
  // verification may fail. 
  let arguments = (ins
    // Specify the type constraint on the input accepting only `func.call` payload
    // operations.
    Transform_ConcreteOpType<"func.call">:$call,                                          // (3)
    StrAttr:$new_target);

  // The results are empty as the transformation does not produce any new payload.
  let results = (outs);

  // Provide nice syntax.
  let assemblyFormat = "$call `,` $new_target attr-dict `:` qualified(type($call))";      // (4)

  // Declare the function implementing the interface for a single payload operation.
  let extraClassDeclaration = [{                                                          // (5)
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::func::CallOp call,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}
```

Compared with Chapter 2's definition:

- **(1)** `TransformOpInterface` is now listed plainly instead of as `DeclareOpInterfaceMethods<TransformOpInterface>`: the op no longer declares its own `apply`, because `TransformEachOpTrait` implements it.
- **(2)** The memory effects are still declared by hand (`getEffects`, unchanged from Chapter 2: the handle is only read, the payload is modified). Section 5 shows the trait-based alternative.
- **(3)** The operand type is `Transform_ConcreteOpType<"func.call">` — a constraint on the *transform* type `!transform.op<"...">` whose operation-name parameter must be `func.call`.
- **(4)** `qualified(type($call))` prints the operand type with its dialect prefix, `!transform.op<"func.call">` (appendix B shows what happens without it).
- **(5)** `applyToOne` is declared by hand in `extraClassDeclaration`; its second parameter is the *payload* operation class, `func::CallOp`, not `Operation *`.

`Transform_ConcreteOpType` turns into an ordinary ODS type constraint, checked by the operation verifier. The generated check is:

```bash
cd /Users/roy/study/mlir/transform/Ch3
mlir-tblgen -gen-op-defs -I include -I /opt/homebrew/opt/llvm@20/include include/MyExtension.td | awk '/^static ::llvm::LogicalResult __mlir_ods_local_type_constraint_MyExtension3\(/,/^}/'
```

Real output **[1]**:

```text
static ::llvm::LogicalResult __mlir_ods_local_type_constraint_MyExtension3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::transform::OperationType>(type))) && ((::llvm::cast<::mlir::transform::OperationType>(type).getOperationName() == "func.call")))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be Transform IR handle to func.call operations, but got " << type;
  }
  return ::mlir::success();
}
```

So a transform script that passes a wider handle to the operation is rejected before anything runs:

```bash
cd /Users/roy/study/mlir/transform/Ch3
./build/transform-opt-ch3 --transform-interpreter <<'EOF'
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %call = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.my.change_call_target %call, "updated" : !transform.any_op
    transform.yield
  }
}
EOF
```

Real output **[2]** (exit status 1):

```text
<stdin>:4:5: error: 'transform.my.change_call_target' op operand #0 must be Transform IR handle to func.call operations, but got '!transform.any_op'
    transform.my.change_call_target %call, "updated" : !transform.any_op
    ^
<stdin>:4:5: note: see current operation: "transform.my.change_call_target"(%0) <{new_target = "updated"}> : (!transform.any_op) -> ()
```

This is a static error in the transform IR, found by the verifier while the input is loaded; the payload IR is never touched.

### 2.2 Who guarantees the payload really is a `func.call`?

A typed handle is a promise about its payload, and the promise is checked whenever payload operations are *associated* with a handle of that type: the type's `checkPayload` method runs (section 3.3 implements one). For `!transform.op<"func.call">` this check is built in. Here `transform.structured.match` finds a `func.return` and tries to return it in a `func.call`-typed handle:

```bash
cd /Users/roy/study/mlir/transform/Ch3
./build/transform-opt-ch3 --transform-interpreter <<'EOF'
func.func @test() {
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %ret = transform.structured.match ops{["func.return"]} in %arg0 : (!transform.any_op) -> !transform.op<"func.call">
    transform.yield
  }
}
EOF
```

Real output **[3]** (exit status 1):

```text
<stdin>:6:12: error: incompatible payload operation name expected func.call vs func.return -> "func.return"() : () -> ()
    %ret = transform.structured.match ops{["func.return"]} in %arg0 : (!transform.any_op) -> !transform.op<"func.call">
           ^
<stdin>:2:3: note: payload operation
  return
  ^
```

The error points at the `match` that *produced* the handle, not at a later consumer. Every operation receiving a `!transform.op<"func.call">` handle can therefore rely on its payload being `func.call` operations. **A handle type moves a precondition from every consumer to the single place where the handle is created**, so it is checked once, and the error points at the operation that broke it.

### 2.3 `applyToOne`: the transformation for one payload operation

Instead of defining the `apply` method with a loop, the operation defines a function that applies to an individual payload operation, and the trait takes care of the rest:

***lib/MyExtension.cpp***
```cpp
::mlir::DiagnosedSilenceableFailure
mlir::transform::ChangeCallTargetOp::applyToOne(
    // The rewriter that should be used when modifying IR.
    ::mlir::transform::TransformRewriter &rewriter,
    // The single payload operation to which the transformation is applied.
    ::mlir::func::CallOp call,
    // The payload IR entities that will be appended to lists associated with
    // the results of this transform operation. This list contains one entry per
    // result.
    ::mlir::transform::ApplyToEachResultList &results,
    // The transform application state. This object can be used to query the
    // current associations between transform IR values and payload IR entities.
    // It can also carry additional user-defined state.
    ::mlir::transform::TransformState &state) {

  // Dispatch to the actual transformation.
  updateCallee(call, getNewTarget());

  // If everything went well, return success.
  return DiagnosedSilenceableFailure::success();
}
```

`updateCallee` is the same one-liner as in Chapter 2 (`call.setCallee(newTarget)`). The function returns the tri-state `DiagnosedSilenceableFailure`: *success*; *definite failure* (the payload may be left in an invalid state, a diagnostic has been emitted, nothing further should run); or *silenceable failure* (a precondition did not hold, the payload is untouched, and the enclosing sequence may choose to suppress the diagnostic).

`TransformEachOpTrait::apply` (an inline template in the installed `TransformInterfaces.h`, quoted in section 7.4) does, for the payload operations of the single operand:

1. if the handle is empty, set every result to an empty list and succeed (section 4.4 shows this);
2. otherwise, for each payload operation, `dyn_cast` it to the class named in `applyToOne`'s second parameter; if that fails, record the silenceable error "transform applied to the wrong op kind" and continue with the next one;
3. call `applyToOne`; a definite failure stops everything, a silenceable failure is recorded and iteration continues;
4. check that `applyToOne` pushed exactly one entry per result (section 4.3 shows the error otherwise), and append the entries to the results' lists.

Applied to two calls, the operation runs once per call, and the handle is still usable afterwards because `getEffects` says it is only read:

```bash
cd /Users/roy/study/mlir/transform/Ch3
./build/transform-opt-ch3 --transform-interpreter -o /dev/null <<'EOF'
func.func private @orig()
func.func private @microkernel()
func.func @test() {
  call @orig() : () -> ()
  call @orig() : () -> ()
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %call = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.op<"func.call">
    transform.my.change_call_target %call, "microkernel" : !transform.op<"func.call">
    transform.debug.emit_remark_at %call, "still valid after change_call_target" : !transform.op<"func.call">
    transform.yield
  }
}
EOF
```

Real output **[4]**:

```text
<stdin>:4:3: remark: still valid after change_call_target
  call @orig() : () -> ()
  ^
<stdin>:4:3: note: see current operation: func.call @microkernel() : () -> ()
<stdin>:5:3: remark: still valid after change_call_target
  call @orig() : () -> ()
  ^
<stdin>:5:3: note: see current operation: func.call @microkernel() : () -> ()
```

`transform.debug.emit_remark_at` emits a remark at each payload operation of the handle; the "see current operation" notes show both calls now target `@microkernel` (the source lines still show the original text).

### 2.4 What the trait checks when the type doesn't

With the `func.call` type constraint in place, the trait's own kind check can't fire: the handle can't hold anything else. To see it, a copy of the chapter is built with the Chapter 2 operand type, `TransformHandleTypeInterface`, and the operation is applied to a handle holding a `func.call` *and* a `func.return`. The second input wraps the same operation in a `transform.sequence ... failures(suppress)`, which drops silenceable failures of its body:

```bash
cd /Users/roy/study/mlir/transform/Ch3
T=$(mktemp -d)
cp -R CMakeLists.txt include lib transform-opt "$T"/
sed -i '' 's/Transform_ConcreteOpType<"func.call">:\$call,/TransformHandleTypeInterface:$call,/' "$T"/include/MyExtension.td
grep -n 'TransformHandleTypeInterface:\$call' "$T"/include/MyExtension.td
cmake -S "$T" -B "$T"/build -G Ninja > /dev/null && cmake --build "$T"/build > /dev/null 2>&1
cat > "$T"/payload.mlir <<'EOF'
func.func private @orig()
func.func private @updated()
func.func @test() {
  call @orig() : () -> ()
  return
}
EOF
cat "$T"/payload.mlir - > "$T"/plain.mlir <<'EOF'
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %ops = transform.structured.match ops{["func.call", "func.return"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.my.change_call_target %ops, "updated" : !transform.any_op
    transform.yield
  }
}
EOF
cat "$T"/payload.mlir - > "$T"/suppressed.mlir <<'EOF'
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %ops = transform.structured.match ops{["func.call", "func.return"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.sequence %arg0 : !transform.any_op failures(suppress) {
    ^bb0(%root: !transform.any_op):
      transform.my.change_call_target %ops, "updated" : !transform.any_op
    }
    transform.yield
  }
}
EOF
"$T"/build/transform-opt-ch3 --transform-interpreter < "$T"/plain.mlir; echo "transform-opt-ch3 exit status: $?"
"$T"/build/transform-opt-ch3 --transform-interpreter < "$T"/suppressed.mlir; echo "transform-opt-ch3 exit status: $?"
rm -rf "$T"
```

Real output **[5]**:

```text
51:    TransformHandleTypeInterface:$call,
<stdin>:10:5: error: transform applied to the wrong op kind
    transform.my.change_call_target %ops, "updated" : !transform.any_op
    ^
<stdin>:5:3: note: when applied to this op
  return
  ^
transform-opt-ch3 exit status: 1
module {
  func.func private @orig()
  func.func private @updated()
  func.func @test() {
    call @updated() : () -> ()
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.call", "func.return"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.sequence %arg0 : !transform.any_op failures(suppress) {
      ^bb0(%arg1: !transform.any_op):
        transform.my.change_call_target %0, "updated" : !transform.any_op
      }
      transform.yield 
    }
  }
}

transform-opt-ch3 exit status: 0
```

- The trait reports the `func.return` with the same message for any operation using it: Chapter 2's hand-written "only applies to func.call payloads" check is no longer needed.
- The failure is silenceable: under `failures(suppress)` the diagnostic disappears, and the output shows the trait **kept going** after the `func.return` — the call was still rewritten to `@updated`.

---

## 3. Defining a Transform Type

Besides operations, a Transform dialect extension can define and inject attributes and **types**. Types specify constraints on payload operations, as section 2 showed. `change_call_target` applies only to `func.call`; a more general call rewriter could accept any payload operation that implements `CallOpInterface` (`func.call`, `func.call_indirect`, `llvm.call`, ...), but the Transform dialect has no type that checks for an interface. The extension defines one.

### 3.1 The `TypeDef`

Defining a transform type is identical to defining a dialect type with ODS:

***include/MyExtensionTypes.td***
```tablegen
include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"

// Transform dialect allows additional types to be defined and injected.
def CallOpInterfaceHandle
  : TypeDef<Transform_Dialect, "CallOpInterfaceHandle",                  // (1)
      // The type must implement `TransformHandleTypeInterface`.
      [DeclareTypeInterfaceMethods<TransformHandleTypeInterface>]> {     // (2)

  // The usual components of a type such as description, mnemonic and assembly format 
  // should be provided.
  let summary = "handle to payload operations implementing CallOpInterface";
  let mnemonic = "my.call_op_interface";                                 // (3)
  let assemblyFormat = "";                                               // (4)
}
```

- **(1)** The type belongs to `Transform_Dialect` (the extension injects it, just as it injects operations); its C++ class is the name plus `Type`: `CallOpInterfaceHandleType`.
- **(2)** Every handle type implements `TransformHandleTypeInterface`; `DeclareTypeInterfaceMethods` makes ODS declare the interface method, `checkPayload`, for us to define.
- **(3)** The mnemonic, by convention prefixed with the extension name like the ops, gives the textual form `!transform.my.call_op_interface`.
- **(4)** The type has no parameters, so its assembly format is empty.

### 3.2 What ODS generates for it

The build runs `mlir-tblgen -gen-typedef-decls` and `-gen-typedef-defs` on this file (appendix C). The declaration:

```bash
cd /Users/roy/study/mlir/transform/Ch3
mlir-tblgen -gen-typedef-decls -I include -I /opt/homebrew/opt/llvm@20/include include/MyExtensionTypes.td | grep -v '^$'
```

Real output **[6]**:

```text
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/
#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES
namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace transform {
class CallOpInterfaceHandleType;
class CallOpInterfaceHandleType : public ::mlir::Type::TypeBase<CallOpInterfaceHandleType, ::mlir::Type, ::mlir::TypeStorage, ::mlir::transform::TransformHandleTypeInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "transform.my.call_op_interface";
  static constexpr ::llvm::StringLiteral dialectName = "transform";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"my.call_op_interface"};
  }
  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::DiagnosedSilenceableFailure checkPayload(::mlir::Location loc, ::mlir::ArrayRef<::mlir::Operation *> payload) const;
};
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::CallOpInterfaceHandleType)
#endif  // GET_TYPEDEF_CLASSES
```

| Member | Meaning |
| --- | --- |
| `TypeBase<..., ::mlir::TypeStorage, ...TransformHandleTypeInterface::Trait>` | A parameterless type (plain `TypeStorage`, uniqued per context) that implements the handle interface. |
| `name`, `dialectName`, `getMnemonic()` | The full name `transform.my.call_op_interface`, the owning dialect, and the keyword the parser dispatches on. |
| `parse` / `print` | Generated from the empty `assemblyFormat`. |
| `checkPayload(Location, ArrayRef<Operation *>) const` | The interface method requested by `DeclareTypeInterfaceMethods`; declared here, **defined by hand** in section 3.3. |
| `MLIR_DECLARE_EXPLICIT_TYPE_ID` | The type's `TypeID`, defined in the `.cpp.inc`. |

The definitions:

```bash
cd /Users/roy/study/mlir/transform/Ch3
mlir-tblgen -gen-typedef-defs -I include -I /opt/homebrew/opt/llvm@20/include include/MyExtensionTypes.td | grep -v '^$'
```

Real output **[7]**:

```text
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/
#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST
::mlir::transform::CallOpInterfaceHandleType
#endif  // GET_TYPEDEF_LIST
#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES
static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::transform::CallOpInterfaceHandleType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::transform::CallOpInterfaceHandleType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}
static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)    .Case<::mlir::transform::CallOpInterfaceHandleType>([&](auto t) {
      printer << ::mlir::transform::CallOpInterfaceHandleType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}
namespace mlir {
namespace transform {
::mlir::Type CallOpInterfaceHandleType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  return CallOpInterfaceHandleType::get(odsParser.getContext());
}
void CallOpInterfaceHandleType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
}
} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::CallOpInterfaceHandleType)
#endif  // GET_TYPEDEF_CLASSES
```

- `GET_TYPEDEF_LIST` expands to the comma-separated list of type classes — the extension's registration (section 3.4) includes the file with it as a template argument list.
- `generatedTypeParser` / `generatedTypePrinter` are the helpers a dialect's `parseType`/`printType` would call. The Transform dialect parses and prints extension types through its own mechanism (each type registered by `registerTypes` is looked up by its mnemonic), so the extension never calls them; section 3.4 shows the `(void)&...` lines that silence the "unused function" warning.
- `parse` and `print` are trivial: there is nothing between the mnemonic and the end of the type.

### 3.3 `checkPayload`: the constraint itself

To finalize the type, the extension implements the interface method:

***lib/MyExtension.cpp***
```cpp
// The interface declares this method to verify constraints this type has on
// payload operations. It returns the now familiar tri-state result.
mlir::DiagnosedSilenceableFailure
mlir::transform::CallOpInterfaceHandleType::checkPayload(
    // Location at which diagnostics should be emitted.
    mlir::Location loc,
    // List of payload operations that are about to be associated with the
    // handle that has this type.
    llvm::ArrayRef<mlir::Operation *> payload) const {

  // All payload operations are expected to implement CallOpInterface, check
  // this.
  for (Operation *op : payload) {
    if (llvm::isa<mlir::CallOpInterface>(op))
      continue;

    // By convention, these verifiers always emit a silenceable failure since
    // they are checking a precondition.
    DiagnosedSilenceableFailure diag =
        emitSilenceableError(loc)
        << "expected the payload operation to implement CallOpInterface";
    diag.attachNote(op->getLoc()) << "offending operation";
    return diag;
  }

  // If everything is okay, return success.
  return DiagnosedSilenceableFailure::success();
}
```

`loc` is the location of the transform value being associated (an operation result or a block argument); `emitSilenceableError` is a helper every handle type inherits from the interface. The chapter's negative test uses the type for the entry point's argument. The interpreter binds that argument to the payload root — the top-level `module` — which is not a call:

***test_Example/transform/Ch3/invalid.mlir***
```mlir
// RUN: transform-opt-ch3 %s --transform-interpreter --split-input-file --verify-diagnostics

// expected-note @below {{offending operation}}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
  // expected-error @below {{expected the payload operation to implement CallOpInterface}}
  %arg0: !transform.my.call_op_interface) {
    transform.yield
  }
}
```

```bash
cd /Users/roy/study/mlir/transform/Ch3
./build/transform-opt-ch3 ../../test_Example/transform/Ch3/invalid.mlir --transform-interpreter
```

Real output **[8]** (exit status 1):

```text
../../test_Example/transform/Ch3/invalid.mlir:7:3: error: expected the payload operation to implement CallOpInterface
  %arg0: !transform.my.call_op_interface) {
  ^
../../test_Example/transform/Ch3/invalid.mlir:4:1: note: offending operation
module attributes {transform.with_named_sequence} {
^
```

The error sits at the block argument (`loc`), the note at the module (`op->getLoc()`). With `--verify-diagnostics`, as in the test's `RUN` line, the tool matches each diagnostic against the `expected-error`/`expected-note` comments instead of printing it, and succeeds when they all match:

```bash
cd /Users/roy/study/mlir/transform/Ch3
./build/transform-opt-ch3 ../../test_Example/transform/Ch3/invalid.mlir --transform-interpreter --split-input-file --verify-diagnostics
```

Real output **[9]** (exit status 0): none.

### 3.4 Registering the type

Additional attributes and types are registered in the extension, next to operations:

***lib/MyExtension.cpp***
```cpp
void MyExtension::init() {
  ...
  declareGeneratedDialect<::mlir::scf::SCFDialect>();
  declareGeneratedDialect<::mlir::func::FuncDialect>();

  // Register the additional transform dialect types with the dialect. List all
  // types generated from ODS.
  registerTypes<
#define GET_TYPEDEF_LIST
#include "MyExtensionTypes.cpp.inc"
      >();

  // ODS generates these helpers for type printing and parsing, but the
  // Transform dialect provides its own support for types supplied by the
  // extension. Reference these functions to avoid a compiler warning.
  (void)&generatedTypeParser;
  (void)&generatedTypePrinter;

  // Finally, we register the additional transform operations with the dialect.
  // List all operations generated from ODS. This call will perform additional
  // checks that the operations implement the transform and memory effect
  // interfaces required by the dialect interpreter and assert if they do not.
  registerTransformOps<
#define GET_OP_LIST
#include "MyExtension.cpp.inc"
      >();
}
```

The `.cpp.inc` files are included twice in `lib/MyExtension.cpp`: once at the top with `GET_TYPEDEF_CLASSES` / `GET_OP_CLASSES` for the class definitions, and here with `GET_TYPEDEF_LIST` / `GET_OP_LIST` for the class lists. The header `include/MyExtension.h` includes `MyExtensionTypes.h.inc` *before* `MyExtension.h.inc`, because the operation classes use the type class (`TypedValue<CallOpInterfaceHandleType>`).

### 3.5 Using the type

The type is now directly available in the Transform dialect and can be used in operations. The usual way to get such a handle is `transform.cast`, which re-associates the payload of one handle with a handle of another type — and runs the target type's `checkPayload`. The tutorial's two-line example casts a handle and passes it to `change_call_target`:

```mlir
  // Cast to our new type.
  %casted = transform.cast %call : !transform.any_op to !transform.my.call_op_interface
  // Using our new operation.
  transform.my.change_call_target %casted, "microkernel" : !transform.my.call_op_interface
```

Completed into a full input and run against this chapter's operations, the second line is rejected:

```bash
cd /Users/roy/study/mlir/transform/Ch3
./build/transform-opt-ch3 --transform-interpreter <<'EOF'
func.func private @orig()
func.func private @microkernel()
func.func @test() {
  call @orig() : () -> ()
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %call = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // Cast to our new type.
    %casted = transform.cast %call : !transform.any_op to !transform.my.call_op_interface
    // Using our new operation.
    transform.my.change_call_target %casted, "microkernel" : !transform.my.call_op_interface
    transform.yield
  }
}
EOF
```

Real output **[10]** (exit status 1):

```text
<stdin>:13:5: error: 'transform.my.change_call_target' op operand #0 must be Transform IR handle to func.call operations, but got '!transform.my.call_op_interface'
    transform.my.change_call_target %casted, "microkernel" : !transform.my.call_op_interface
    ^
<stdin>:13:5: note: see current operation: "transform.my.change_call_target"(%1) <{new_target = "microkernel"}> : (!transform.my.call_op_interface) -> ()
```

`change_call_target`'s `applyToOne` takes a `func::CallOp`, and its operand type says so; a handle to "anything implementing `CallOpInterface`" is wider than that, so the verifier refuses it. `CallOpInterface` has no `setCallee(StringRef)` — not every call has a symbol callee — which is why the operation stays specific to `func.call`. The operation that does take the new type is `call_to_op` (section 4). Used correctly — cast to `!transform.op<"func.call">` for `change_call_target`, and to `!transform.my.call_op_interface` for `call_to_op`:

```bash
cd /Users/roy/study/mlir/transform/Ch3
./build/transform-opt-ch3 --transform-interpreter --allow-unregistered-dialect <<'EOF'
func.func private @orig()
func.func private @microkernel()
func.func @test() {
  call @orig() : () -> ()
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %call = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %as_call = transform.cast %call : !transform.any_op to !transform.op<"func.call">
    transform.my.change_call_target %as_call, "microkernel" : !transform.op<"func.call">
    transform.debug.emit_remark_at %as_call, "retargeted" : !transform.op<"func.call">
    %casted = transform.cast %call : !transform.any_op to !transform.my.call_op_interface
    %mm4 = transform.my.call_to_op %casted : (!transform.my.call_op_interface) -> !transform.any_op
    transform.yield
  }
}
EOF
```

Real output **[11]**:

```text
<stdin>:4:3: remark: retargeted
  call @orig() : () -> ()
  ^
<stdin>:4:3: note: see current operation: func.call @microkernel() : () -> ()
module {
  func.func private @orig()
  func.func private @microkernel()
  func.func @test() {
    "my.mm4"() : () -> ()
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.cast %0 : !transform.any_op to !transform.op<"func.call">
      transform.my.change_call_target %1, "microkernel" : !transform.op<"func.call">
      transform.debug.emit_remark_at %1, "retargeted" : !transform.op<"func.call">
      %2 = transform.cast %0 : !transform.any_op to !transform.my.call_op_interface
      %3 = transform.my.call_to_op %2 : (!transform.my.call_op_interface) -> !transform.any_op
      transform.yield 
    }
  }
}
```

The call is first retargeted (the remark), then replaced by `"my.mm4"()`, an operation of the unregistered `my` dialect — hence `--allow-unregistered-dialect`. The new type round-trips through the printer as `!transform.my.call_op_interface`.

A cast whose payload doesn't satisfy the type fails in `checkPayload`:

```bash
cd /Users/roy/study/mlir/transform/Ch3
./build/transform-opt-ch3 --transform-interpreter <<'EOF'
func.func @test() {
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %ret = transform.structured.match ops{["func.return"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %casted = transform.cast %ret : !transform.any_op to !transform.my.call_op_interface
    transform.yield
  }
}
EOF
```

Real output **[12]** (exit status 1):

```text
<stdin>:7:15: error: expected the payload operation to implement CallOpInterface
    %casted = transform.cast %ret : !transform.any_op to !transform.my.call_op_interface
              ^
<stdin>:2:3: note: offending operation
  return
  ^
```

`checkPayload` returns a *silenceable* failure "by convention", yet the interpreter does not let a sequence suppress it. The check runs while the interpreter records the results of the cast, after the operation has already run, and a failed result check there is reported and turned into a definite failure:

```bash
cd /Users/roy/study/mlir/transform/Ch3
./build/transform-opt-ch3 --transform-interpreter <<'EOF'
func.func @test() {
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.sequence %arg0 : !transform.any_op failures(suppress) {
    ^bb0(%root: !transform.any_op):
      %ret = transform.structured.match ops{["func.return"]} in %root : (!transform.any_op) -> !transform.any_op
      %casted = transform.cast %ret : !transform.any_op to !transform.my.call_op_interface
      transform.debug.emit_remark_at %ret, "not reached" : !transform.any_op
    }
    transform.yield
  }
}
EOF
```

Real output **[13]** (exit status 1):

```text
<stdin>:9:17: error: expected the payload operation to implement CallOpInterface
      %casted = transform.cast %ret : !transform.any_op to !transform.my.call_op_interface
                ^
<stdin>:2:3: note: offending operation
  return
  ^
```

Compare section 2.4, where the trait's silenceable "wrong op kind" error *was* suppressed. **A handle's type is an invariant of the transform IR, not a recoverable precondition of one transformation**: once a handle exists, everything after it relies on the type, so a violation stops the script.

---

## 4. Operand Consumption

### 4.1 `call_to_op`: replacing the payload operation

As an exercise, the extension adds an operation that rewrites the call into a custom operation `my.mm4`. The original call is erased, so the handle pointing to it would dangle: the operation must **consume** its operand handle, and it returns a **new** handle to the newly produced payload operations. The tutorial first writes this with hand-declared memory effects (not this repo's code):

```tablegen
// In MyExtension.td.

// Define another transform operation.
def CallToOp : Op<Transform_Dialect, "my.call_to_op",
     // Indicate that the operation implements the required TransformOpInterface and
     // MemoryEffectsOpInterface. Use the TransformEach trait to provide the
     // implementation for TransformOpInterface.
    [TransformOpInterface, TransformEachOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  // Summary and description omitted for brevity.

  // The argument is the handle to the payload operations.
  let arguments = (ins CallOpInterfaceHandle:$call);

  // The result is the handle to the payload operations produced during the
  // transformation.
  let results = (outs TransformHandleTypeInterface:$transformed);

  // Provide nice syntax.
  let assemblyFormat = "$call attr-dict `:` functional-type(inputs, outputs)";

  // Declare the function implementing the interface for a single payload operation.
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::CallOpInterface call,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}
```

The repo's definition (section 5.1) differs only in its traits — the effects come from `FunctionalStyleTransformOpTrait` — and in writing the assembly format as `functional-type(operands, results)`. The operand is the new `CallOpInterfaceHandle` type, and `applyToOne` takes a `CallOpInterface`: the trait's `dyn_cast` (section 2.3) works for interfaces as well as op classes. The result is a plain `TransformHandleTypeInterface` (in practice `!transform.any_op`): `my.mm4` is unregistered, so no narrower type could describe it.

### 4.2 The implementation

The tutorial's version of the interface methods (not this repo's code):

```c++
// In MyExtension.cpp.

::mlir::DiagnosedSilenceableFailure CallToOp::applyToOne(
    ::mlir::transform::TransformRewriter &rewriter,
    ::mlir::CallOpInterface call,
    ::mlir::transform::ApplyToEachResultList &results,
    ::mlir::transform::TransformState &state) {
  // Call the actual rewrite.
  Operation *rewritten = rewriteToOp(call);

  // Report an error if the rewriter produced a null pointer. Note that it may have
  // irreversibly modified the payload IR, so we produce a definite failure.
  if (!rewritten) {
    return emitDefiniteError() << "failed to rewrite call to operation";
  }

  // On success, push the resulting operation into the result list. The list is expected
  // to contain exactly one entity per result and per application. The handles will be
  // associated with lists of the respective values produced by each application.
  results.push_back(rewritten);

  // If everything is fine, return success.
  return DiagnosedSilenceableFailure::success();
}

void CallToOp::getEffects(
    ::llvm::SmallVectorImpl<::mlir::MemoryEffects::EffectInstance> &effects) {
  // Indicate using side effects that the operand handle is consumed, and the
  // result handle is produced.
  consumesHandle(getCall(), effects);
  producesHandle(getTransformed(), effects);

  // Indicate that the payload IR is modified.
  modifiesPayload(effects);
}
```

The repo's version cannot fail, so it has no definite-error branch, and it has no `getEffects` (section 5):

***lib/MyExtension.cpp***
```cpp
static mlir::Operation *replaceCallWithOp(mlir::RewriterBase &rewriter,
                                          mlir::CallOpInterface call) {
  // Construct an operation from an unregistered dialect. This is discouraged
  // and is only used here for brevity of the overall example.
  mlir::OperationState state(call.getLoc(), "my.mm4");
  state.types.assign(call->result_type_begin(), call->result_type_end());
  state.operands.assign(call->operand_begin(), call->operand_end());

  mlir::Operation *replacement = rewriter.create(state);
  rewriter.replaceOp(call, replacement->getResults());
  return replacement;
}

// See above for the signature description.
mlir::DiagnosedSilenceableFailure mlir::transform::CallToOp::applyToOne(
    mlir::transform::TransformRewriter &rewriter, mlir::CallOpInterface call,
    mlir::transform::ApplyToEachResultList &results,
    mlir::transform::TransformState &state) {

  // Dispatch to the actual transformation.
  Operation *replacement = replaceCallWithOp(rewriter, call);

  // Associate the payload operation produced by the rewrite with the result
  // handle of this transform operation.
  results.push_back(replacement);

  // If everything went well, return success.
  return DiagnosedSilenceableFailure::success();
}
```

`my.mm4` takes the call's operands and result types, and is created through the `TransformRewriter` passed in (the trait set its insertion point right before the call). `rewriter.replaceOp` redirects the call's uses and erases it; going through the rewriter lets the interpreter's listener notice the replacement.

Consuming the handle has a visible consequence: once `call_to_op` has run, any later use of the consumed handle is an error, because the payload it pointed to has been erased:

```bash
cd /Users/roy/study/mlir/transform/Ch3
./build/transform-opt-ch3 --transform-interpreter --allow-unregistered-dialect <<'EOF'
func.func private @orig()
func.func @test() {
  call @orig() : () -> ()
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %call = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.my.call_op_interface
    %mm4 = transform.my.call_to_op %call : (!transform.my.call_op_interface) -> !transform.any_op
    transform.debug.emit_remark_at %call, "still there?" : !transform.my.call_op_interface
    transform.yield
  }
}
EOF
```

Real output **[14]** (exit status 1):

```text
<stdin>:10:5: error: op uses a handle invalidated by a previously executed transform op
    transform.debug.emit_remark_at %call, "still there?" : !transform.my.call_op_interface
    ^
<stdin>:8:13: note: handle to invalidated ops
    %call = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.my.call_op_interface
            ^
<stdin>:9:12: note: invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them
    %mm4 = transform.my.call_to_op %call : (!transform.my.call_op_interface) -> !transform.any_op
           ^
<stdin>:3:3: note: ancestor payload op
  call @orig() : () -> ()
  ^
<stdin>:3:3: note: nested payload op
```

The interpreter knows `%call` was consumed only from the operation's declared memory effects. Compare section 2.3, where the same pattern after `change_call_target` (which only reads its handle) was fine. **The effects are a contract, not decoration**: declaring "only reads" for an operation that erases its payload would let later operations touch freed memory.

### 4.3 Every result needs an entry, on every application

The application specifies the payload to associate with each handle it produces. Operations are required to specify the entities for *all* results on success, even if the list is empty. In case of failure, the interpreter automatically associates all results that are not yet defined with empty lists. With `applyToOne`, "all results" means exactly one entry per result per payload operation. To see the check, a copy of the chapter is built whose `applyToOne` forgets the `push_back`:

```bash
cd /Users/roy/study/mlir/transform/Ch3
T=$(mktemp -d)
cp -R CMakeLists.txt include lib transform-opt "$T"/
sed -i '' 's/^  results.push_back(replacement);$/  (void)replacement;/' "$T"/lib/MyExtension.cpp
grep -n '(void)replacement;' "$T"/lib/MyExtension.cpp
cmake -S "$T" -B "$T"/build -G Ninja > /dev/null && cmake --build "$T"/build > /dev/null 2>&1
"$T"/build/transform-opt-ch3 ../../test_Example/transform/Ch3/ops.mlir --transform-interpreter --allow-unregistered-dialect --split-input-file -o /dev/null; echo "transform-opt-ch3 exit status: $?"
rm -rf "$T"
```

Real output **[15]**:

```text
179:  (void)replacement;
within split at ../../test_Example/transform/Ch3/ops.mlir:30 offset :16:5: error: application of transform.my.call_to_op expected to produce 1 results (actually produced 0).
    transform.my.call_to_op %call : (!transform.my.call_op_interface) -> !transform.any_op
    ^
within split at ../../test_Example/transform/Ch3/ops.mlir:30 offset :8:3: note: when applied to this op
  call @orig() : () -> ()
  ^
within split at ../../test_Example/transform/Ch3/ops.mlir:30 offset :16:5: note: if you need variadic results, consider a generic `apply` instead of the specialized `applyToOne`.
    transform.my.call_to_op %call : (!transform.my.call_op_interface) -> !transform.any_op
    ^
transform-opt-ch3 exit status: 1
```

The trait's check (step 4 in section 2.3) turns the mistake into a definite failure with a diagnostic; the note names the alternative. Since `applyToOne` always expects one payload entity per result handle per application, it cannot return handles associated with empty lists for non-empty operand handles. An operation that needs that implements `apply` directly (not this repo's code):

```c++
::mlir::DiagnosedSilenceableFailure SomeOtherOp::apply(
    ::mlir::transform::TransformRewriter &rewriter,
    ::mlir::transform::TransformResults &results,
    ::mlir::transform::TransformState &state) {
  // ...

  // Associate the result `transformed` with an empty list of payload operations.
  results.set(cast<OpResult>(getTransformed()), {});
  return DiagnosedSilenceableFailure::success();
}
```

### 4.4 An empty operand gives empty results

The one way an `applyToOne` operation produces empty result lists is an empty operand: the trait then never calls `applyToOne` and sets every result to an empty list (step 1 in section 2.3):

```bash
cd /Users/roy/study/mlir/transform/Ch3
./build/transform-opt-ch3 --transform-interpreter -o /dev/null <<'EOF'
func.func @test() {
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %call = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.my.call_op_interface
    %mm4 = transform.my.call_to_op %call : (!transform.my.call_op_interface) -> !transform.any_op
    %n = transform.num_associations %mm4 : (!transform.any_op) -> !transform.param<i64>
    transform.debug.emit_param_as_remark %n, "payload ops in %mm4:" : !transform.param<i64>
    transform.yield
  }
}
EOF
```

Real output **[16]**:

```text
<stdin>:9:5: remark: payload ops in %mm4: 0 : i64
    transform.debug.emit_param_as_remark %n, "payload ops in %mm4:" : !transform.param<i64>
    ^
<stdin>:9:5: note: see current operation: transform.debug.emit_param_as_remark %2, "payload ops in %mm4:" : !transform.param<i64>
```

`transform.num_associations` returns the number of payload operations of a handle as a parameter (`!transform.param<i64>`), and `emit_param_as_remark` prints it. No call, no error, and an empty `%mm4`: a script written for "all calls" quietly does nothing on a module without calls.

---

## 5. Memory Effects Traits

Some common memory-effect patterns are available as traits, to minimize the boilerplate:

- `FunctionalStyleTransformOpTrait` indicates that all handle-typed operands are consumed, all results are produced, and the payload IR is modified.
- `NavigationTransformOpTrait` indicates that all handle-typed operands are only read, all results are produced, and the payload IR is only read.

Using these traits removes the need to declare or define the methods of the `MemoryEffectsOpInterface`.

### 5.1 `call_to_op` with `FunctionalStyleTransformOpTrait`

This is the repo's definition:

***include/MyExtension.td***
```tablegen
// Define another transform operation.
def CallToOp : Op<Transform_Dialect, "my.call_to_op",
     // Indicate that the operation implements the required TransformOpInterface.
     // Use the TransformEach trait to provide implementation of this interface.
    [TransformOpInterface, TransformEachOpTrait,
     // Indicate that the operation implements the required MemoryEffectsOpInterface.
     // Use the FunctionalStyle trait to provide the implementation for this interface.
     MemoryEffectsOpInterface, FunctionalStyleTransformOpTrait]> {
  // Summary and description omitted for brevity.

  // The argument is the handle to the payload operations.
  let arguments = (ins CallOpInterfaceHandle:$call);

  // The result is the handle to the payload operations produced during the 
  // transformation.
  let results = (outs TransformHandleTypeInterface:$transformed);

  // Provide nice syntax.
  let assemblyFormat = "$call attr-dict `:` functional-type(operands, results)";

  // Declare the function implementing the interface for a single payload operation.
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::CallOpInterface call,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}
```

`MemoryEffectsOpInterface` is listed plainly (the op implements the interface), and `FunctionalStyleTransformOpTrait` supplies `getEffects`. The difference between the two operations is visible in the generated declarations — the trait lists, and which class declares a `getEffects` to be written by hand:

```bash
cd /Users/roy/study/mlir/transform/Ch3
mlir-tblgen -gen-op-decls -I include -I /opt/homebrew/opt/llvm@20/include include/MyExtension.td | grep -E '^class (ChangeCallTargetOp|CallToOp) |getEffects\(|applyToOne\(' | sed 's/, ::mlir::/,\n    ::mlir::/g'
```

Real output **[17]**:

```text
class CallToOp : public ::mlir::Op<CallToOp,
    ::mlir::OpTrait::ZeroRegions,
    ::mlir::OpTrait::OneResult,
    ::mlir::OpTrait::OneTypedResult<::mlir::transform::TransformHandleTypeInterface>::Impl,
    ::mlir::OpTrait::ZeroSuccessors,
    ::mlir::OpTrait::OneOperand,
    ::mlir::OpTrait::OpInvariants,
    ::mlir::transform::TransformOpInterface::Trait,
    ::mlir::transform::TransformEachOpTrait,
    ::mlir::MemoryEffectOpInterface::Trait,
    ::mlir::transform::FunctionalStyleTransformOpTrait> {
  ::mlir::DiagnosedSilenceableFailure applyToOne(
class ChangeCallTargetOp : public ::mlir::Op<ChangeCallTargetOp,
    ::mlir::OpTrait::ZeroRegions,
    ::mlir::OpTrait::ZeroResults,
    ::mlir::OpTrait::ZeroSuccessors,
    ::mlir::OpTrait::OneOperand,
    ::mlir::OpTrait::OpInvariants,
    ::mlir::BytecodeOpInterface::Trait,
    ::mlir::transform::TransformOpInterface::Trait,
    ::mlir::transform::TransformEachOpTrait,
    ::mlir::MemoryEffectOpInterface::Trait> {
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  ::mlir::DiagnosedSilenceableFailure applyToOne(
```

(The `sed` only breaks the long base-class lists into one trait per line.)

- Both classes carry `TransformOpInterface::Trait` and `TransformEachOpTrait`; neither declares `apply` — the trait's template provides it.
- `ChangeCallTargetOp` declares `getEffects` (from `DeclareOpInterfaceMethods<MemoryEffectsOpInterface>`), defined in `lib/MyExtension.cpp` with `onlyReadsHandle` + `modifiesPayload`.
- `CallToOp` declares no `getEffects`: `FunctionalStyleTransformOpTrait` inherits one into the class (section 7.5 quotes it: `consumesHandle` on all operands, `producesHandle` on all results, `modifiesPayload`). Section 4.2's run showed those effects in action.
- `ChangeCallTargetOp` also has `BytecodeOpInterface::Trait` because it has an attribute stored as a property (`new_target`); `CallToOp` has none.

`change_call_target` could not use either trait: it modifies the payload (not navigation) but only reads its handle (not functional style), so it keeps its hand-written `getEffects`. **Pick a trait when the effects follow from the operation's shape; write `getEffects` when one operand behaves differently.**

---

## 6. Build and Run

The top-level README of the repository covers the shared toolchain; this chapter builds standalone from its own directory.

### 6.1 Building

```bash
cd /Users/roy/study/mlir/transform/Ch3
cmake -S . -B build -G Ninja
cmake --build build          # → ./build/transform-opt-ch3
```

No `-DMLIR_DIR` or compiler flags are needed: the shell has `/opt/homebrew/opt/llvm@20/bin` on `PATH` and `CXX`/`CC` set to Homebrew llvm@20 clang, so `find_package` finds `llvm@20/lib/cmake/{mlir,llvm}`. If your environment differs, pass `-DMLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir -DLLVM_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/llvm -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++`. Appendix C explains the build wiring.

`transform-opt-ch3` is an `mlir-opt`-like driver: `MlirOptMain` with all upstream dialects and extensions, `registerMyExtension`, the `transform-interpreter` pass, and the `canonicalize`, `cse` and `symbol-dce` cleanup passes.

### 6.2 Running the operations test

`ops.mlir` holds two independent inputs separated by `// -----` (hence `--split-input-file`): `change_call_target` on a `!transform.op<"func.call">` handle, and `call_to_op` on a `!transform.my.call_op_interface` handle.

***test_Example/transform/Ch3/ops.mlir***
```mlir
// RUN: transform-opt-ch3 %s --transform-interpreter \
// RUN:   --allow-unregistered-dialect --split-input-file | FileCheck %s
...
func.func private @orig()
func.func private @updated()

// CHECK-LABEL: func @test1
func.func @test1() {
  // CHECK: call @updated
  call @orig() : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %call = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.op<"func.call">
    // CHECK: transform.my.change_call_target %{{.*}}, "updated" : !transform.op<"func.call">
    transform.my.change_call_target %call, "updated" : !transform.op<"func.call">
    transform.yield
  }
}

// -----

func.func private @orig()

// CHECK-LABEL: func @test2
func.func @test2() {
  // CHECK: "my.mm4"
  call @orig() : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %call = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.my.call_op_interface
    // CHECK: transform.my.call_to_op %{{.*}} : (!transform.my.call_op_interface) -> !transform.any_op
    transform.my.call_to_op %call : (!transform.my.call_op_interface) -> !transform.any_op
    transform.yield
  }
}
```

```bash
cd /Users/roy/study/mlir/transform/Ch3
./build/transform-opt-ch3 ../../test_Example/transform/Ch3/ops.mlir --transform-interpreter --allow-unregistered-dialect --split-input-file
```

Real output **[18]**:

```text
module {
  func.func private @orig()
  func.func private @updated()
  func.func @test1() {
    call @updated() : () -> ()
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.op<"func.call">
      transform.my.change_call_target %0, "updated" : !transform.op<"func.call">
      transform.yield 
    }
  }
}

// -----
module {
  func.func private @orig()
  func.func @test2() {
    "my.mm4"() : () -> ()
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.my.call_op_interface
      %1 = transform.my.call_to_op %0 : (!transform.my.call_op_interface) -> !transform.any_op
      transform.yield 
    }
  }
}

```

- In `@test1` the call now targets `@updated`; in `@test2` it has become `"my.mm4"()`.
- The transform IR prints back with both kinds of handle types, `!transform.op<"func.call">` (thanks to `qualified`) and `!transform.my.call_op_interface` — the round trip the `CHECK` lines on the transform operations test.

Piping the same command into `FileCheck` checks every `CHECK` line:

```bash
cd /Users/roy/study/mlir/transform/Ch3
./build/transform-opt-ch3 ../../test_Example/transform/Ch3/ops.mlir --transform-interpreter --allow-unregistered-dialect --split-input-file | FileCheck ../../test_Example/transform/Ch3/ops.mlir
```

Real output **[19]** (exit status 0): none.

### 6.3 The complete example: tiling, fusion, outlining, and a microkernel call

`sequence.mlir` is the schedule built in Chapter 1 — tile the ReLU, fuse the bias addition and the matmul into the loop, tile again, outline the inner loop into a function — finished with this chapter's `change_call_target`, which redirects the call of the outlined function to a hand-written `@microkernel`. Its transform operation now uses the typed handle `!transform.op<"func.call">` that `transform.loop.outline` returns.

***test_Example/transform/Ch3/sequence.mlir***
```mlir
// RUN: transform-opt-ch3 %s \
// RUN:   --pass-pipeline="builtin.module(transform-interpreter{ \
// RUN:        debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary},\
// RUN:        canonicalize,cse,symbol-dce)" |\
// RUN: FileCheck %s
...
// Original function to optimize.
func.func @fc_relu(%lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
                   %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
                   -> tensor<512x512xf32> {
  // Matrix-matrix multiplication.
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
                          outs(%output: tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise addition.
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
    ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise max with 0 (ReLU).
  %c0f = arith.constant 0.0 : f32
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
    ins(%biased, %c0f : tensor<512x512xf32>, f32)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>
  func.return %relued : tensor<512x512xf32>
}
...
// Declaration of the "microkernel" function that we will be targeting.
func.func private @microkernel(
    %lhs: tensor<4x512xf32>,
    %rhs: tensor<512x4xf32>,
    %bias: tensor<4x4xf32>,
    %init: tensor<4x4xf32>,
    %output: tensor<4x4xf32>) -> tensor<4x4xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
       %arg0: !transform.any_op,
       %arg1: !transform.op<"linalg.matmul">,
       %arg2: !transform.op<"linalg.elemwise_binary">) {
    ...
    // Since outlining is currently only implemented for region-holding operations
    // such as loops, use tiling to size 1 to materialize the outer loop that is
    // going to be outlined.
    %_0, %loop_third = transform.structured.tile_using_forall %tiled_second tile_sizes [1]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %_1, %outline_target = transform.structured.fuse_into_containing_op %matmul_fused_2 into %loop_third
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %func, %call = transform.loop.outline %outline_target {func_name = "outlined"}
        : (!transform.any_op) -> (!transform.any_op, !transform.op<"func.call">)
  
    // Rewrite the call target.
    transform.my.change_call_target %call, "microkernel" : !transform.op<"func.call">
  
    transform.yield
  }
}
```

The abridged parts are the `CHECK` lines and the split/tile/fuse steps explained in Chapter 1. `debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary` binds the entry point's trailing arguments `%arg1` and `%arg2` to all payload operations with those names (a debugging convenience instead of matching them in the script); the cleanup passes after the interpreter simplify the result.

```bash
cd /Users/roy/study/mlir/transform/Ch3
./build/transform-opt-ch3 ../../test_Example/transform/Ch3/sequence.mlir --pass-pipeline="builtin.module(transform-interpreter{debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary},canonicalize,cse,symbol-dce)"
```

Real output **[20]** (abridged: the transform module at the end is left out):

```text
#map = affine_map<(d0) -> (d0 * 8)>
#map1 = affine_map<(d0) -> (d0 * 32)>
#map2 = affine_map<(d0) -> (d0 * 4)>
module {
  func.func @outlined(%arg0: tensor<4x512xf32>, %arg1: tensor<512x4xf32>, %arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>, %arg4: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = scf.forall (%arg5) in (4) shared_outs(%arg6 = %arg4) -> (tensor<4x4xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg5, 0] [1, 512] [1, 1] : tensor<4x512xf32> to tensor<1x512xf32>
      %extracted_slice_0 = tensor.extract_slice %arg2[%arg5, 0] [1, 4] [1, 1] : tensor<4x4xf32> to tensor<1x4xf32>
      %1 = linalg.matmul ins(%extracted_slice, %arg1 : tensor<1x512xf32>, tensor<512x4xf32>) outs(%extracted_slice_0 : tensor<1x4xf32>) -> tensor<1x4xf32>
      %extracted_slice_1 = tensor.extract_slice %arg3[%arg5, 0] [1, 4] [1, 1] : tensor<4x4xf32> to tensor<1x4xf32>
      %extracted_slice_2 = tensor.extract_slice %arg6[%arg5, 0] [1, 4] [1, 1] : tensor<4x4xf32> to tensor<1x4xf32>
      %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %extracted_slice_1 : tensor<1x4xf32>, tensor<1x4xf32>) outs(%extracted_slice_2 : tensor<1x4xf32>) -> tensor<1x4xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %2 into %arg6[%arg5, 0] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<4x4xf32>
      }
    }
    return %0 : tensor<4x4xf32>
  }
  func.func @fc_relu(%arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>, %arg2: tensor<512x512xf32>, %arg3: tensor<512x512xf32>) -> tensor<512x512xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = scf.forall (%arg4, %arg5) in (64, 16) shared_outs(%arg6 = %arg3) -> (tensor<512x512xf32>) {
      %1 = affine.apply #map(%arg4)
      %2 = affine.apply #map1(%arg5)
      %extracted_slice = tensor.extract_slice %arg0[%1, 0] [8, 512] [1, 1] : tensor<512x512xf32> to tensor<8x512xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %2] [512, 32] [1, 1] : tensor<512x512xf32> to tensor<512x32xf32>
      %extracted_slice_1 = tensor.extract_slice %arg3[%1, %2] [8, 32] [1, 1] : tensor<512x512xf32> to tensor<8x32xf32>
      %extracted_slice_2 = tensor.extract_slice %arg2[%1, %2] [8, 32] [1, 1] : tensor<512x512xf32> to tensor<8x32xf32>
      %3 = scf.forall (%arg7, %arg8) in (2, 8) shared_outs(%arg9 = %extracted_slice_1) -> (tensor<8x32xf32>) {
        %5 = affine.apply #map2(%arg7)
        %6 = affine.apply #map2(%arg8)
        %extracted_slice_4 = tensor.extract_slice %extracted_slice[%5, 0] [4, 512] [1, 1] : tensor<8x512xf32> to tensor<4x512xf32>
        %extracted_slice_5 = tensor.extract_slice %extracted_slice_0[0, %6] [512, 4] [1, 1] : tensor<512x32xf32> to tensor<512x4xf32>
        %extracted_slice_6 = tensor.extract_slice %extracted_slice_1[%5, %6] [4, 4] [1, 1] : tensor<8x32xf32> to tensor<4x4xf32>
        %extracted_slice_7 = tensor.extract_slice %extracted_slice_2[%5, %6] [4, 4] [1, 1] : tensor<8x32xf32> to tensor<4x4xf32>
        %extracted_slice_8 = tensor.extract_slice %arg9[%5, %6] [4, 4] [1, 1] : tensor<8x32xf32> to tensor<4x4xf32>
        %7 = func.call @microkernel(%extracted_slice_4, %extracted_slice_5, %extracted_slice_6, %extracted_slice_7, %extracted_slice_8) : (tensor<4x512xf32>, tensor<512x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %7 into %arg9[%5, %6] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<8x32xf32>
        }
      }
      %extracted_slice_3 = tensor.extract_slice %arg6[%1, %2] [8, 32] [1, 1] : tensor<512x512xf32> to tensor<8x32xf32>
      %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%3, %cst : tensor<8x32xf32>, f32) outs(%extracted_slice_3 : tensor<8x32xf32>) -> tensor<8x32xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %4 into %arg6[%1, %2] [8, 32] [1, 1] : tensor<8x32xf32> into tensor<512x512xf32>
      }
    }
    return %0 : tensor<512x512xf32>
  }
  func.func private @microkernel(tensor<4x512xf32>, tensor<512x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  module attributes {transform.with_named_sequence} {
...
  }
}
```

- The outer `scf.forall` iterates over 64×16 tiles of 8×32 (the first `tile_sizes [8, 32]`); the ReLU (`max_signed`) stays at that level.
- The inner `scf.forall` over 2×8 tiles of 4×4 holds what was the fused matmul + bias addition: it is now one `func.call @microkernel` on five 4×4-shaped slices — exactly the signature of the declared `@microkernel`, which is why `change_call_target` can swap the callee without touching the operands.
- `@outlined` is still in the module: it is public, so `symbol-dce` keeps it even though nothing calls it any more.

The same command through `FileCheck`:

```bash
cd /Users/roy/study/mlir/transform/Ch3
./build/transform-opt-ch3 ../../test_Example/transform/Ch3/sequence.mlir --pass-pipeline="builtin.module(transform-interpreter{debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary},canonicalize,cse,symbol-dce)" | FileCheck ../../test_Example/transform/Ch3/sequence.mlir
```

Real output **[21]** (exit status 0): none.

---

## 7. Deriving the Chapter 3 operations, type and build yourself

Sections 2–5 explain this chapter's code as it is. This section shows how to arrive at it starting from Chapter 2, with only the installed MLIR 20 headers and tools. The new file is `include/MyExtensionTypes.td`; the rest are edits to Chapter 2's files:

| Part | Lines | Where it comes from |
| --- | --- | --- |
| `include/MyExtension.td`: `TransformEachOpTrait` and the `applyToOne` declarations | 30, 61–67, 74, 91–97 | The trait's doc comment in `TransformInterfaces.h` (section 7.4) |
| `include/MyExtension.td`: `Transform_ConcreteOpType<"func.call">` and `qualified(...)` | 51, 58 | The handle types in `TransformTypes.td` (section 7.2); `qualified` in appendix B |
| `include/MyExtension.td`: `CallToOp` and its `FunctionalStyleTransformOpTrait` | 71–98 | The effect traits in `TransformInterfaces.h` (section 7.5) |
| `include/MyExtensionTypes.td`: the `CallOpInterfaceHandle` `TypeDef` | 22–32 | No stock type checks an interface (section 7.2); the shape of `Transform_AnyOpType` |
| `lib/MyExtension.cpp`: `checkPayload` | 193–216 | `TransformTypeInterfaceBase` in `TransformInterfaces.td` (section 7.3), or the generated declaration (section 3.2) |
| `lib/MyExtension.cpp`: the two `applyToOne` bodies | 104–138, 169–183 | Chapter 2's `apply` minus the loop and the `dyn_cast` (section 7.4) |
| `lib/MyExtension.cpp`: `replaceCallWithOp` | 155–166 | `OperationState`, `OpBuilder::create` and `RewriterBase::replaceOp` in the IR headers (section 7.8) |
| `lib/MyExtension.cpp`: `registerTypes<GET_TYPEDEF_LIST>` and the `(void)&generated...` lines | 73–84 | `TransformDialectExtension::registerTypes` (section 7.8) and an unused-function warning (section 7.7) |
| `lib/MyExtension.cpp`: the new `#include`s and the `.cpp.inc` includes at the top | 18–27 | The generated type definitions (sections 3.2, 7.7) |
| `include/MyExtension.h`: forward declarations and the `MyExtensionTypes.h.inc` include | 18–29 | The compiler errors of section 7.7 |
| `include/CMakeLists.txt`: the typedef rules | 8–13 | `mlir-tblgen --help` and a build without them (section 7.6) |

The other changes rename `ch2` to `ch3` in the CMake files and in the driver, which also drops an unused forward declaration (`test::registerTestTransformDialectExtension`).

### 7.1 Step 1: see what changes from Chapter 2

```bash
cd /Users/roy/study/mlir/transform/Ch3
diff ../Ch2/include/MyExtension.td include/MyExtension.td
```

Real output **[22]** (exit status 1):

```text
10c10
< // Chapter 2 of the Transform dialect tutorial.
---
> // Chapter 3 of the Transform dialect tutorial.
16a17
> include "MyExtensionTypes.td"
18a20
> include "mlir/Dialect/Transform/IR/TransformTypes.td"
26,27c28,30
<     // MemoryEffectsOpInterface.
<     [DeclareOpInterfaceMethods<TransformOpInterface>,
---
>     // MemoryEffectsOpInterface. Use the TransformEach trait to provide the 
>     // implementation for TransformOpInterface.
>     [TransformOpInterface, TransformEachOpTrait,
46c49,51
<     TransformHandleTypeInterface:$call,
---
>     // Specify the type constraint on the input accepting only `func.call` payload
>     // operations.
>     Transform_ConcreteOpType<"func.call">:$call,
53c58,67
<   let assemblyFormat = "$call `,` $new_target attr-dict `:` type($call)";
---
>   let assemblyFormat = "$call `,` $new_target attr-dict `:` qualified(type($call))";
> 
>   // Declare the function implementing the interface for a single payload operation.
>   let extraClassDeclaration = [{
>     ::mlir::DiagnosedSilenceableFailure applyToOne(
>         ::mlir::transform::TransformRewriter &rewriter,
>         ::mlir::func::CallOp call,
>         ::mlir::transform::ApplyToEachResultList &results,
>         ::mlir::transform::TransformState &state);
>   }];
55a70,99
> // Define another transform operation.
> def CallToOp : Op<Transform_Dialect, "my.call_to_op",
>      // Indicate that the operation implements the required TransformOpInterface.
>      // Use the TransformEach trait to provide implementation of this interface.
>     [TransformOpInterface, TransformEachOpTrait,
>      // Indicate that the operation implements the required MemoryEffectsOpInterface.
>      // Use the FunctionalStyle trait to provide the implementation for this interface.
>      MemoryEffectsOpInterface, FunctionalStyleTransformOpTrait]> {
>   // Summary and description omitted for brevity.
> 
>   // The argument is the handle to the payload operations.
>   let arguments = (ins CallOpInterfaceHandle:$call);
> 
>   // The result is the handle to the payload operations produced during the 
>   // transformation.
>   let results = (outs TransformHandleTypeInterface:$transformed);
> 
>   // Provide nice syntax.
>   let assemblyFormat = "$call attr-dict `:` functional-type(operands, results)";
> 
>   // Declare the function implementing the interface for a single payload operation.
>   let extraClassDeclaration = [{
>     ::mlir::DiagnosedSilenceableFailure applyToOne(
>         ::mlir::transform::TransformRewriter &rewriter,
>         ::mlir::CallOpInterface call,
>         ::mlir::transform::ApplyToEachResultList &results,
>         ::mlir::transform::TransformState &state);
>   }];
> }
> 
```

Three edits to `change_call_target` (trait, operand type, `applyToOne` declaration — plus `qualified`) and one new operation. The C++ side follows: Chapter 2's `apply` loop becomes `applyToOne`, and the type needs `checkPayload` and registration.

### 7.2 Step 2: which handle types exist?

```bash
cd /Users/roy/study/mlir/transform/Ch3
grep -n '^def \|^class ' /opt/homebrew/opt/llvm@20/include/mlir/Dialect/Transform/IR/TransformTypes.td
```

Real output **[23]**:

```text
16:def Transform_AffineMapParamType : TypeDef<Transform_Dialect, "AffineMapParam",
26:def Transform_AnyOpType : TypeDef<Transform_Dialect, "AnyOp",
36:def Transform_AnyValue : TypeDef<Transform_Dialect, "AnyValue",
45:def Transform_OperationType : TypeDef<Transform_Dialect, "Operation",
58:def Transform_AnyParamType : TypeDef<Transform_Dialect, "AnyParam",
68:def Transform_ParamType : TypeDef<Transform_Dialect, "Param",
84:def Transform_TypeParamType : TypeDef<Transform_Dialect, "TypeParam",
94:class Transform_ConcreteOpType<string opname>
101:def TransformAnyHandle : Type<
```

The operation handles are `!transform.any_op` (`Transform_AnyOpType`) and `!transform.op<"name">` (`Transform_OperationType`), and `Transform_ConcreteOpType<opname>` is the ODS constraint that fixes the name — what `change_call_target`'s operand needs. Nothing checks for an interface, hence the new type; `Transform_AnyOpType` (`TypeDef` + `DeclareTypeInterfaceMethods<TransformHandleTypeInterface>` + `mnemonic` + empty `assemblyFormat`) is the template `CallOpInterfaceHandle` follows.

### 7.3 Step 3: what a handle type must implement

```bash
cd /Users/roy/study/mlir/transform/Ch3
sed -n '120,136p' /opt/homebrew/opt/llvm@20/include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.td
```

Real output **[24]**:

```text
class TransformTypeInterfaceBase<string cppClass, string cppObjectType>
    : TypeInterface<cppClass> {
  let cppNamespace = "::mlir::transform";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Checks if the given associated objects (Payload IR operations or attributes)
        satisfy the conditions defined by this type. If not, produces a silenceable
        error at the specified location.
      }],
      /*returnType=*/"::mlir::DiagnosedSilenceableFailure",
      /*name=*/"checkPayload",
      /*arguments=*/(ins "::mlir::Location":$loc,
                         "::mlir::ArrayRef<" # cppObjectType # ">":$payload)
    >
  ];
```

One method, `checkPayload(Location, ArrayRef<Operation *>)` for `TransformHandleTypeInterface` (where `cppObjectType` is `::mlir::Operation *`) — exactly what section 3.2's generated declaration asks for.

### 7.4 Step 4: the `applyToOne` contract

```bash
cd /Users/roy/study/mlir/transform/Ch3
sed -n '1185,1204p' /opt/homebrew/opt/llvm@20/include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.h
```

Real output **[25]**:

```text
/// Trait implementing the TransformOpInterface for operations applying a
/// transformation to a single operation handle and producing an arbitrary
/// number of handles and parameter values.
/// The op must implement a method with the following signature:
///   - DiagnosedSilenceableFailure applyToOne(OpTy,
///       ApplyToEachResultList &results, TransformState &state)
/// to perform a transformation that is applied in turn to all payload IR
/// operations that correspond to the handle of the transform IR operation.
/// In `applyToOne`, OpTy is either Operation* or a concrete payload IR Op class
/// that the transformation is applied to (and NOT the class of the transform IR
/// op).
/// The `applyToOne` method takes an empty `results` vector that it fills with
/// zero, one or multiple operations depending on the number of results expected
/// by the transform op.
/// The number of results must match the number of results of the transform op.
/// `applyToOne` is allowed to fill the `results` with all null elements to
/// signify that the transformation did not apply to the payload IR operations.
/// Such null elements are filtered out from results before return.
///
/// The transform op having this trait is expected to have a single operand.
```

The comment leaves out the `TransformRewriter &` first parameter, which the trait's implementation passes (`transformOp.applyToOne(rewriter, specificOp, partialResults, state)` in the same header); the full signature is the one in the `extraClassDeclaration`. Choosing `OpTy` = `func::CallOp` or `CallOpInterface` is what makes the trait check the payload kind (section 2.4). The last line explains why the trait fits these single-operand operations. Null entries in `results` are another way to say "did not apply" without a failure.

### 7.5 Step 5: the effect traits

```bash
cd /Users/roy/study/mlir/transform/Ch3
sed -n '1294,1307p;1321,1338p' /opt/homebrew/opt/llvm@20/include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.h
```

Real output **[26]**:

```text
/// Trait implementing the MemoryEffectOpInterface for operations that "consume"
/// their operands and produce new results.
template <typename OpTy>
class FunctionalStyleTransformOpTrait
    : public OpTrait::TraitBase<OpTy, FunctionalStyleTransformOpTrait> {
public:
  /// This op "consumes" the operands by reading and freeing then, "produces"
  /// the results by allocating and writing it and reads/writes the payload IR
  /// in the process.
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
    consumesHandle(this->getOperation()->getOpOperands(), effects);
    producesHandle(this->getOperation()->getOpResults(), effects);
    modifiesPayload(effects);
  }
/// operands without consuming and without modifying the Payload IR to
/// potentially produce new handles.
template <typename OpTy>
class NavigationTransformOpTrait
    : public OpTrait::TraitBase<OpTy, NavigationTransformOpTrait> {
public:
  /// This op produces handles to the Payload IR without consuming the original
  /// handles and without modifying the IR itself.
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
    onlyReadsHandle(this->getOperation()->getOpOperands(), effects);
    producesHandle(this->getOperation()->getOpResults(), effects);
    if (llvm::any_of(this->getOperation()->getOperandTypes(), [](Type t) {
          return isa<TransformHandleTypeInterface,
                     TransformValueHandleTypeInterface>(t);
        })) {
      onlyReadsPayload(effects);
    }
  }
```

The two traits are the two `getEffects` bodies a hand-written version would contain. `call_to_op` replaces its payload, so it matches `FunctionalStyleTransformOpTrait` line for line — the tutorial's hand-written `getEffects` in section 4.2 is the same three calls. `change_call_target` matches neither (section 5.1).

### 7.6 Step 6: generate the type's C++ in the build

The type lives in its own `.td` file, so the build needs a TableGen rule for it. `mlir-tblgen` lists the generators for a `TypeDef`; a copy of the chapter whose `include/CMakeLists.txt` lacks the type lines (deleted by `sed`) shows what breaks without them:

```bash
cd /Users/roy/study/mlir/transform/Ch3
mlir-tblgen --help | grep -E -- '-gen-typedef-(decls|defs) '
T=$(mktemp -d)
cp -R CMakeLists.txt include lib transform-opt "$T"/
sed -i '' '/MyExtensionTypes/d' "$T"/include/CMakeLists.txt
cmake -S "$T" -B "$T"/build -G Ninja > /dev/null
cmake --build "$T"/build 2>&1 | grep -E 'error:' | sed "s|$T|\$T|" | sort -u
rm -rf "$T"
```

Real output **[27]**:

```text
      --gen-typedef-decls                               - Generate TypeDef declarations
      --gen-typedef-defs                                - Generate TypeDef definitions
$T/include/MyExtension.h:29:10: fatal error: 'MyExtensionTypes.h.inc' file not found
```

- The two generators are the ones section 3.2 runs by hand. `include/CMakeLists.txt` points `LLVM_TARGET_DEFINITIONS` at `MyExtensionTypes.td` and adds one `mlir_tablegen` call per generator, before `add_public_tablegen_target`, so the existing target also covers the type files (appendix C.1).
- Without the rule, nothing writes `MyExtensionTypes.h.inc`, and the first file to include `MyExtension.h` stops there.

### 7.7 Step 7: let the compiler list what the header and the source need

The generated operation classes now name C++ types that Chapter 2's header never needed. A copy of the chapter without `include/MyExtension.h`'s forward declarations, and a second copy without the two `(void)&generated...` lines in `lib/MyExtension.cpp`, show why each is there:

```bash
cd /Users/roy/study/mlir/transform/Ch3
T=$(mktemp -d)
cp -R CMakeLists.txt include lib transform-opt "$T"/
sed -i '' '/^namespace mlir {$/,/^} \/\/ namespace mlir$/d' "$T"/include/MyExtension.h
cmake -S "$T" -B "$T"/build -G Ninja > /dev/null
cmake --build "$T"/build 2>&1 | grep -E 'error:' | sed "s|$T|\$T|" | sort -u
rm -rf "$T"
T=$(mktemp -d)
cp -R CMakeLists.txt include lib transform-opt "$T"/
sed -i '' '/(void)&generatedType/d' "$T"/lib/MyExtension.cpp
cmake -S "$T" -B "$T"/build -G Ninja > /dev/null
cmake --build "$T"/build 2>&1 | grep -E 'warning:' | grep -v Wdangling | sed "s|$T|\$T|" | sort -u
rm -rf "$T"
```

Real output **[28]**:

```text
$T/build/include/MyExtension.h.inc:155:15: error: no template named 'CallOpInterface' in namespace 'mlir'; did you mean 'OpInterface'?
$T/build/include/MyExtension.h.inc:155:15: error: use of class template '::mlir::OpInterface' requires template arguments
$T/build/include/MyExtension.h.inc:305:41: error: no member named 'OperationType' in namespace 'mlir::transform'
$T/build/include/MyExtension.h.inc:306:63: error: no member named 'OperationType' in namespace 'mlir::transform'
$T/build/include/MyExtension.h.inc:369:15: error: no member named 'func' in namespace 'mlir'
$T/lib/MyExtension.cpp:119:38: error: out-of-line definition of 'applyToOne' does not match any declaration in 'mlir::transform::ChangeCallTargetOp'
$T/lib/MyExtension.cpp:169:62: error: out-of-line definition of 'applyToOne' does not match any declaration in 'mlir::transform::CallToOp'
$T/build/include/MyExtensionTypes.cpp.inc:19:36: warning: unused function 'generatedTypeParser' [-Wunused-function]
$T/build/include/MyExtensionTypes.cpp.inc:31:30: warning: unused function 'generatedTypePrinter' [-Wunused-function]
```

- **The forward declarations.** Each error names one of the three classes `include/MyExtension.h` declares: `CallOpInterface` from `CallToOp`'s `applyToOne` (line 155 of the generated header), `transform::OperationType` from the `getCall()` accessor that `Transform_ConcreteOpType` gives `change_call_target` (lines 305–306), and `func::CallOp` from its `applyToOne` (line 369). The header only has to make the names known; `lib/MyExtension.cpp`, which defines the methods, includes the full classes first (`FuncOps.h`, `TransformTypes.h`, `CallInterfaces.h`). With the declarations missing, the two `applyToOne` definitions then match nothing.
- **The `(void)&...` lines.** `generatedTypeParser` and `generatedTypePrinter` are `static` functions in the generated definitions (section 3.2) that the extension never calls, so the compiler warns about them; taking their address is the cheapest way to use them.
- **The other new includes** come from the same generated definitions: `generatedTypePrinter` uses `llvm::TypeSwitch` (`llvm/ADT/TypeSwitch.h`), and `generatedTypeParser` uses `AsmParser` (`mlir/IR/DialectImplementation.h`).

**Generated code has its own include list: when ODS starts naming new C++ types, the compiler errors in the `.inc` file say exactly which declarations the header owes.**

### 7.8 Step 8: find the registration and rewrite APIs

Two C++ calls remain: registering the type with the Transform dialect, and building the replacement `my.mm4` operation in `call_to_op`. The installed headers declare both:

```bash
cd /Users/roy/study/mlir/transform/Ch3
I=/opt/homebrew/opt/llvm@20/include
grep -n -B6 'void registerTypes()' $I/mlir/Dialect/Transform/IR/TransformDialect.h
grep -n '  OperationState(Location location, StringRef name);' $I/mlir/IR/OperationSupport.h
grep -n '  Operation \*create(const OperationState &state);' $I/mlir/IR/Builders.h
grep -n -B3 'virtual void replaceOp(Operation \*op, ValueRange newValues);' $I/mlir/IR/PatternMatch.h
```

Real output **[29]**:

```text
198-  /// Injects the types into the Transform dialect. The types must implement
199-  /// the TransformHandleTypeInterface and the implementation must be already
200-  /// available when the type is injected. Furthermore, the types must provide
201-  /// a `getMnemonic` static method returning an object convertible to
202-  /// `StringRef` that is unique across all injected types.
203-  template <typename... TypeTys>
204:  void registerTypes() {
976:  OperationState(Location location, StringRef name);
475:  Operation *create(const OperationState &state);
528-  /// Replace the results of the given (original) operation with the specified
529-  /// list of values (replacements). The result types of the given op and the
530-  /// replacements must match. The original op is erased.
531:  virtual void replaceOp(Operation *op, ValueRange newValues);
```

- **`registerTypes<TypeTys...>()`** sits next to `registerTransformOps` in `TransformDialectExtension`, and like it takes the class list, which `GET_TYPEDEF_LIST` supplies (section 3.4). Its comment states the two requirements the `TypeDef` already meets: `TransformHandleTypeInterface` and a `getMnemonic` (both in section 3.2's generated class).
- **`replaceCallWithOp`** is these three calls in order (section 4.2): an `OperationState` for an operation by name, which is how an operation of an unregistered dialect is built; `create` on the rewriter, which is an `OpBuilder`; and `replaceOp` with the new results, which erases the call. Going through the rewriter instead of `Operation` methods lets the interpreter's listener notice the replacement (section 4.2).

---

## 8. Key Takeaways & Pitfalls

- **Typed handles state preconditions once.** `!transform.op<"func.call">` is checked by the verifier when a script uses it (section 2.1) and by `checkPayload` when payload is associated (section 2.2); consumers can then rely on it.
- **`TransformEachOpTrait` writes the loop.** An operation with one handle operand implements `applyToOne(rewriter, PayloadOpTy, results, state)`; the trait iterates, `dyn_cast`s each payload operation (silenceable "wrong op kind" error, iteration continues), and collects results (sections 2.3, 2.4).
- **A transform type is a `TypeDef` on `Transform_Dialect` plus `checkPayload`**, registered with `registerTypes<GET_TYPEDEF_LIST>` next to `registerTransformOps` (section 3).
- **A failed type check is fatal.** `checkPayload` returns a silenceable error, but the interpreter turns a failed result check into a definite failure that `failures(suppress)` cannot hide (section 3.5).
- **A wider handle doesn't fit a narrower operand.** `!transform.my.call_op_interface` cannot be passed to `change_call_target`, which needs `func.call`; cast to the type the operation declares (section 3.5).
- **Consuming is about the payload's fate.** An operation that erases or replaces its payload consumes its handle and returns a new one; later uses of the old handle are reported as invalidated (section 4.2).
- **`applyToOne` must push one entry per result per application**, or the interpreter fails definitely; use `apply` for variadic or empty results (section 4.3). An empty operand quietly yields empty results (section 4.4).
- **Effect traits are shortcuts for the two common shapes:** functional style (consume all, produce all, modify payload) and navigation (read all, produce all, read payload). Mixed behavior needs a hand-written `getEffects` (section 5).
- **Everything new is found in the installed headers and tools**: handle types in `TransformTypes.td`, the type interface in `TransformInterfaces.td`, the trait contracts in `TransformInterfaces.h`, `registerTypes` in `TransformDialect.h`; the build rule and the header's forward declarations follow from `mlir-tblgen --help` and the compiler's errors on the generated code (section 7).
- **Pitfall — unregistered results:** `my.mm4` is an unregistered operation, so every run that produces it needs `--allow-unregistered-dialect`; a real extension would emit operations of a registered dialect and declare it with `declareGeneratedDialect`.

---

## Appendix A: Autogenerated Documentation

The build's `add_mlir_doc` rule (appendix C) generates Markdown reference documentation from the `.td` files. The same output, printed directly:

```bash
cd /Users/roy/study/mlir/transform/Ch3
mlir-tblgen -gen-op-doc -I include -I /opt/homebrew/opt/llvm@20/include include/MyExtension.td
```

Real output **[30]**:

~~~markdown
<!-- Autogenerated by mlir-tblgen; don't manually edit -->
### `transform.my.call_to_op` (transform::CallToOp)

Syntax:

```
operation ::= `transform.my.call_to_op` $call attr-dict `:` functional-type(operands, results)
```


Traits: `FunctionalStyleTransformOpTrait`, `TransformEachOpTrait`

Interfaces: `MemoryEffectsOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `call` | handle to payload operations implementing CallOpInterface

#### Results:

| Result | Description |
| :----: | ----------- |
| `transformed` | TransformHandleTypeInterface instance

### `transform.my.change_call_target` (transform::ChangeCallTargetOp)

_Changes the callee of a call operation to the specified one_


Syntax:

```
operation ::= `transform.my.change_call_target` $call `,` $new_target attr-dict `:` qualified(type($call))
```

For each `func.call` payload operation associated with the handle, changes its 
callee to be the symbol whose name is provided as an attribute to this operation.

Generates a silenceable failure if the operand is associated with payload operations 
that are not `func.call`.
Only reads the operand.

Traits: `TransformEachOpTrait`

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>new_target</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `call` | Transform IR handle to func.call operations

~~~

- `call_to_op` has no summary or description (its ODS comment says "omitted for brevity"); `change_call_target` shows both.
- The operand descriptions are the type constraints' descriptions: "handle to payload operations implementing CallOpInterface" is the `summary` of the new type, "Transform IR handle to func.call operations" comes from `Transform_ConcreteOpType`.
- The description of `change_call_target` still says it "generates a silenceable failure" for non-`func.call` payloads; with the typed operand, that case is ruled out before the operation runs (sections 2.1, 2.2).

The type's documentation comes from `-gen-typedef-doc` (the build does not generate it):

```bash
cd /Users/roy/study/mlir/transform/Ch3
mlir-tblgen -gen-typedef-doc -I include -I /opt/homebrew/opt/llvm@20/include include/MyExtensionTypes.td
```

Real output **[31]**:

```text
<!-- Autogenerated by mlir-tblgen; don't manually edit -->
### CallOpInterfaceHandleType

handle to payload operations implementing CallOpInterface

Syntax: `!transform.my.call_op_interface`



```

---

## Appendix B: `qualified(type($call))` in the assembly format

`change_call_target`'s format prints its operand type with `qualified(...)`. A copy of the chapter built with plain `type($call)` (Chapter 2's spelling) prints the operation of `ops.mlir` like this:

```bash
cd /Users/roy/study/mlir/transform/Ch3
T=$(mktemp -d)
cp -R CMakeLists.txt include lib transform-opt "$T"/
sed -i '' 's/qualified(type(\$call))/type($call)/' "$T"/include/MyExtension.td
grep -n 'assemblyFormat = "\$call `,`' "$T"/include/MyExtension.td
cmake -S "$T" -B "$T"/build -G Ninja > /dev/null && cmake --build "$T"/build > /dev/null 2>&1
"$T"/build/transform-opt-ch3 ../../test_Example/transform/Ch3/ops.mlir --split-input-file --allow-unregistered-dialect | grep change_call_target
rm -rf "$T"
```

Real output **[32]**:

```text
58:  let assemblyFormat = "$call `,` $new_target attr-dict `:` type($call)";
      transform.my.change_call_target %0, "updated" : <"func.call">
```

For an operand whose constraint has a concrete C++ type — `Transform_ConcreteOpType` names `::mlir::transform::OperationType` — the generated printer prints the type in its *stripped* form, without the `!transform.op` prefix, and the parser expects it that way; `!transform.op<"func.call">` shrinks to `<"func.call">`, which is hard to read. Chapter 2's operand constraint was the interface `TransformHandleTypeInterface`, which has no concrete type to strip, so the full type was printed. `qualified` keeps the full spelling, and the `CHECK` line `: !transform.op<"func.call">` in `ops.mlir` (section 6.2) relies on it.

---

## Appendix C: The Build

### C.1 What Chapter 3 adds to the build

The chapter's `CMakeLists.txt` is Chapter 2's with the names changed: a standalone-mode guard (it finds MLIR/LLVM, defaults to a Release build because `libMLIR.dylib` is built with `NDEBUG`, and turns on `LLVM_BUILD_EXAMPLES` so `add_llvm_example` builds by default), then the chapter targets:

***CMakeLists.txt***
```cmake
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)

add_dependencies(TransformExample transform-opt-ch3)
add_llvm_example(transform-opt-ch3
  transform-opt/transform-opt.cpp)

target_link_libraries(transform-opt-ch3
  PRIVATE
  MLIR                      # libMLIR.dylib (all dialects, passes, extensions)
  MyExtensionCh3
)
```

The new part is in `include/CMakeLists.txt`: a second `LLVM_TARGET_DEFINITIONS` for the type file, with the typedef generators:

***include/CMakeLists.txt***
```cmake
# Tell Tablegen to use MyExtension.td as input.
set(LLVM_TARGET_DEFINITIONS MyExtension.td)

# Ask Tablegen to generate op declarations and definitions from ODS.
mlir_tablegen(MyExtension.h.inc -gen-op-decls)                          # (1)
mlir_tablegen(MyExtension.cpp.inc -gen-op-defs)

# Tell Tablegen to use MyExtensionTypes.td as input.
set(LLVM_TARGET_DEFINITIONS MyExtensionTypes.td)                        # (2)

# Ask Tablegen to generate type declarations and definitions from ODS.
mlir_tablegen(MyExtensionTypes.h.inc -gen-typedef-decls)                # (3)
mlir_tablegen(MyExtensionTypes.cpp.inc -gen-typedef-defs)

# Add a CMakeTarget we can depend on to ensure the generation happens before the compilation.
add_public_tablegen_target(MyExtensionCh3IncGen)                        # (4)

# Don't forget to generate the documentation, this will produce a
# MyExtensionCh3.md under Tutorials/transform
add_mlir_doc(MyExtension MyExtensionCh3 Tutorials/transform/ -gen-op-doc)  # (5)
```

- **(1)** The operation declarations and definitions, as in Chapter 2.
- **(2)** Each `mlir_tablegen` call reads the current value of `LLVM_TARGET_DEFINITIONS`, so resetting it switches the input file for the calls that follow.
- **(3)** The typedef generators shown in section 3.2.
- **(4)** One target collects all four outputs; the library depends on it.
- **(5)** The op documentation of appendix A; it is added to the `mlir-doc` target, not built by default.

The library, `lib/CMakeLists.txt`, links only `libMLIR.dylib`:

***lib/CMakeLists.txt***
```cmake
# Outside examples, this should be `add_mlir_library`.
add_mlir_example_library(
  # Library called MyExtension.
  MyExtensionCh3

  # Built from the following source files.
  MyExtension.cpp

  # Make includes visible without top-level path.
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/examples/transform/Ch3/include

  # Make sure ODS declaration and definitions are generated before compiling this.
  DEPENDS
  MyExtensionCh3IncGen

  # Link in the transform dialect, an all generated dialects.
  LINK_LIBS PRIVATE
  MLIR                      # libMLIR.dylib
)
```

Linking the shared `libMLIR.dylib` only, and none of the static `MLIR*` archives, keeps a single copy of every MLIR `TypeID` in the process.

### C.2 The generated files

After a build, the four generated files sit in `build/include` (with their dependency files):

```bash
cd /Users/roy/study/mlir/transform/Ch3
ls build/include
```

Real output **[33]**:

```text
CMakeFiles
MyExtension.cpp.inc
MyExtension.cpp.inc.d
MyExtension.h.inc
MyExtension.h.inc.d
MyExtensionTypes.cpp.inc
MyExtensionTypes.cpp.inc.d
MyExtensionTypes.h.inc
MyExtensionTypes.h.inc.d
cmake_install.cmake
```

---

## Appendix D: Running this README's commands: `examples.sh`

[`examples.sh`](examples.sh), a symlink to the shared [`../examples.sh`](../examples.sh), runs the commands of this README and shows each one with its output and exit status. Build the chapter first (section 6.1), then:

- `./examples.sh` runs every command, in README order;
- `./examples.sh 7` runs command [7] only (several numbers are fine);
- `./examples.sh -l` lists the numbered commands without running them:

```bash
cd /Users/roy/study/mlir/transform/Ch3
./examples.sh -l
```

Real output **[34]** (abridged):

```text
[ 1] 2.1   mlir-tblgen -gen-op-defs -I include -I /opt/homebrew/opt/llvm@20/include include/MyExtension.td | awk '/^static ::llvm::LogicalResult __mlir_ods_local_type_constraint_MyExtension3\(/,/^}/'
[ 2] 2.1   ./build/transform-opt-ch3 --transform-interpreter <<'EOF'
...
[22] 7.1   diff ../Ch2/include/MyExtension.td include/MyExtension.td
...
[33] C.2   ls build/include
[34] D     ./examples.sh -l
```

The script reads the commands from `README.md` on every run. A command is every `bash` code block in file order, except the build block (its first command after `cd` is `cmake`) and file excerpts (a `***path***` label above the fence). Number *N* is the *N*-th such block, and the README marks that block's output "Real output **[N]**". The patched-copy commands (5, 15, 27, 28, 32) each configure and build a temporary copy of the chapter, so they take a while.

---

## Links

- [Transform Ch3: More than Simple Transform Operations](https://mlir.llvm.org/docs/Tutorials/transform/Ch3/)
- [Transform dialect reference](https://mlir.llvm.org/docs/Dialects/Transform/)
- [Defining Dialect Attributes and Types (ODS `TypeDef`)](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/)
- [Side Effects and Speculation](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/)
