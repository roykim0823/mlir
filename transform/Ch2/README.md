# Chapter 2: Adding a Simple New Transformation Operation

> **Goal:** Define a new Transform dialect operation, `transform.my.change_call_target`, in a dialect *extension* (ODS (Operation Definition Specification) for the op, C++ for its `apply` and its memory effects), register it in a small `mlir-opt`-like tool, and use it at the end of a tiling/fusion/outlining schedule — based on the official tutorial [Transform Ch2: Adding a Simple New Transformation Operation](https://mlir.llvm.org/docs/Tutorials/transform/Ch2/).

---

## 1. Overview

Chapter 1 composed transform operations that ship with MLIR. Sooner or later a schedule needs a transformation nobody has written yet — typically one tied to a project's own dialects or conventions. This chapter adds one, the smallest useful kind:

1. Choose **where the new op lives**: not in the Transform dialect itself, but in a **dialect extension** that injects ops into it when it is loaded (section 2).
2. **Define the op** in ODS and implement the two interfaces every transform op needs: `TransformOpInterface` (`apply`, the transformation) and `MemoryEffectsOpInterface` (`getEffects`, what happens to handles and payload) (section 3).
3. **Register** the extension in the tool's `main` and use the op from a transform script — first on a toy payload, then as the last step of Chapter 1's matmul + bias + ReLU (rectified linear unit) schedule, where it retargets the call to an outlined loop to a "microkernel" (section 4).

The op, `transform.my.change_call_target %h, "name"`, takes a handle to `func.call` payload ops and changes each call's callee to `@name`. The chapter's code is four files: `include/MyExtension.td` (the op), `include/MyExtension.h` and `lib/MyExtension.cpp` (the extension and the interface methods), and `transform-opt/transform-opt.cpp` (the tool, `transform-opt-ch2`). The inputs are `test_Example/transform/Ch2/{ops,invalid,sequence}.mlir`.

Sections 2–4 follow the chapter's essential path; section 5 builds the tool and runs the three tests, and section 6 shows how to arrive at the chapter's code, its tool and its build yourself. The appendices hold the op's generated documentation (A), the build wiring, including what the standalone build needs to link against Homebrew's `libMLIR.dylib` (B), and the script that runs every command in this README (C).

---

## 2. Setting Up to Add New Transformations

### 2.1 Where the implementation lives: a dialect extension

Before defining a new transform operation, we need to choose where its implementation should be located. While MLIR encourages upstream contributions, it is not always possible or even desirable to modify the main Transform dialect — for example, if the transformation is specific to some out-of-tree dialect that is not itself available upstream.

The Transform dialect uses the **dialect extension** mechanism to allow additional operations to be injected without modifying the dialect itself. Dialect extensions are registered with the context (through a `DialectRegistry`) and loaded when the dialect itself is loaded. The extension is a class deriving from `TransformDialectExtension<Derived>` — the CRTP (curiously recurring template pattern) idiom, where the class passes itself as the template argument, gives each extension its own identity:

***lib/MyExtension.cpp***
```cpp
// Define a new transform dialect extension. This uses the CRTP idiom to
// identify extensions.
class MyExtension
    : public ::mlir::transform::TransformDialectExtension<MyExtension> {
public:
  // The TypeID of this extension.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyExtension)

  // The extension must derive the base constructor.
  using Base::Base;

  // This function initializes the extension, similarly to `initialize` in
  // dialect definitions. List individual operations and dependent dialects
  // here.
  void init();
};
```

- `using Base::Base;` inherits the base constructor (`Base` is `TransformDialectExtension<MyExtension>`).
- `MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID` gives the class a `TypeID` defined right here. A `DialectRegistry` keys extensions by `TypeID`, so registering the same extension twice is harmless; defining the ID explicitly keeps it stable no matter how the class is linked (static library, `libMLIR.dylib`, ...).
- `init()` is the hook the base class calls when the extension is constructed; it records what to do when the Transform dialect gets loaded.

### 2.2 Dependent dialects vs. generated dialects

`init()` declares the dialects the extension needs and the ops it adds:

***lib/MyExtension.cpp***
```cpp
void MyExtension::init() {
  // Similarly to dialects, an extension can declare a dependent dialect. This
  // dialect will be loaded along with the extension and, therefore, along with
  // the Transform dialect. Only declare as dependent the dialects that contain
  // the attributes or types used by transform operations. Do NOT declare as
  // dependent the dialects produced during the transformation.
  // declareDependentDialect<MyDialect>();

  // When transformations are applied, they may produce new operations from
  // previously unloaded dialects. Typically, a pass would need to declare
  // itself dependent on the dialects containing such new operations. To avoid
  // confusion with the dialects the extension itself depends on, the Transform
  // dialects differentiates between:
  //   - dependent dialects, which are used by the transform operations, and
  //   - generated dialects, which contain the entities (attributes, operations,
  //     types) that may be produced by applying the transformation even when
  //     not present in the original payload IR.
  // In the following chapter, we will be add operations that generate function
  // calls and structured control flow operations, so let's declare the
  // corresponding dialects as generated.
  declareGeneratedDialect<::mlir::scf::SCFDialect>();
  declareGeneratedDialect<::mlir::func::FuncDialect>();

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

The two kinds of dialect exist because MLIR does not allow loading a dialect while a pass is running on IR (multithreaded passes would race on the context):

| Declared with | Meaning | When it is loaded | Example |
|---|---|---|---|
| `declareDependentDialect<D>()` | the *transform ops themselves* use attributes or types of `D` (in their operands, attributes, canonicalization) | together with the Transform dialect, i.e. when the transform script is parsed | an op taking a `#my.tile_config` attribute |
| `declareGeneratedDialect<D>()` | *applying* the ops may create ops/attributes/types of `D` in the payload, even if the payload had none | before the interpreter pass runs (like a pass's `getDependentDialects`) | `scf` (loops) and `func` (calls), which Chapter 3's ops produce |

`registerTransformOps<...>()` doesn't add the ops right away either: it queues an initializer that runs `addOperationsChecked<...>()` when the Transform dialect is loaded. The template argument list comes from ODS, as section 2.4 shows.

### 2.3 The op's ODS skeleton

The operations themselves are defined using ODS, exactly the same way as regular operations in a dialect. Every transform op definition starts from the same prelude: the Transform dialect's own `.td` (for `Transform_Dialect`), the transform interfaces (for `TransformOpInterface` and the handle type constraints), `OpBase.td`, and the side-effect interfaces (for `MemoryEffectsOpInterface`):

***include/MyExtension.td***
```tablegen
#ifndef MY_EXTENSION
#define MY_EXTENSION

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

...

#endif // MY_EXTENSION
```

Each op is then a record of the form below (not this repo's code; section 3.1 fills it in):

```tablegen
def MyOp : Op<Transform_Dialect, "my.op", [
    // TODO: interfaces and traits here.
   ]> {
  let summary = "my transform op";
  // TODO: define the operation properties.
}
```

The op is defined in `Transform_Dialect`, so its full name is `transform.` + the mnemonic. The extension does not get a dialect namespace of its own — the "`my.`" prefix is only a naming convention that keeps extension ops from colliding.

### 2.4 Generating C++ from ODS

Similarly to dialects, TableGen generates the header and implementation of these operations. CMake is told to do it in `include/CMakeLists.txt` (the full file, with `add_mlir_doc`, is in appendix B.2):

***include/CMakeLists.txt***
```cmake
# Tell Tablegen to use MyExtension.td as input.
set(LLVM_TARGET_DEFINITIONS MyExtension.td)

# Ask Tablegen to generate op declarations and definitions from ODS.
mlir_tablegen(MyExtension.h.inc -gen-op-decls)
mlir_tablegen(MyExtension.cpp.inc -gen-op-defs)

# Add a CMakeTarget we can depend on to ensure the generation happens before the compilation.
add_public_tablegen_target(MyExtensionCh2IncGen)
```

and the library that compiles `MyExtension.cpp` depends on the `MyExtensionCh2IncGen` target so the generated files exist first (appendix B.3). This generates two files, `MyExtension.h.inc` and `MyExtension.cpp.inc`, that are included into the declaration and definition of the transform operations, respectively. The header includes the declarations with `GET_OP_CLASSES` defined, next to the Transform dialect headers they need:

***include/MyExtension.h***
```cpp
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

#define GET_OP_CLASSES
#include "MyExtension.h.inc"

// Registers our Transform dialect extension.
void registerMyExtension(::mlir::DialectRegistry &registry);
```

`MyExtension.cpp` includes `MyExtension.cpp.inc` twice, with different macros: once with `GET_OP_LIST` inside the `registerTransformOps<...>` template argument list (section 2.2), and once with `GET_OP_CLASSES` for the op definitions:

***lib/MyExtension.cpp***
```cpp
#define GET_OP_CLASSES
#include "MyExtension.cpp.inc"
```

What `GET_OP_LIST` expands to is visible by running the generator by hand on the `.td` file (`-I` points at the installed MLIR headers for the `include`s):

```bash
cd /Users/roy/study/mlir/transform/Ch2
mlir-tblgen -gen-op-defs -I /opt/homebrew/opt/llvm@20/include include/MyExtension.td \
  | sed -n '/^#ifdef GET_OP_LIST/,/^#endif/p'
```

Real output **[1]**:

```text
#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::transform::ChangeCallTargetOp
#endif  // GET_OP_LIST
```

It is a comma-separated list of the op classes in the file (here just one), so `registerTransformOps<` ... `>()` becomes `registerTransformOps<::mlir::transform::ChangeCallTargetOp>()`. Adding a second `def` to the `.td` file registers it with no C++ change: **the `.td` file is the single list of the extension's ops, and the C++ only ever refers to it through the macro.**

The `GET_OP_CLASSES` part of the declarations is the op class itself; section 3.1 shows it.

---

## 3. Defining a Transform Operation

### 3.1 The op in ODS

With this setup, we are ready to define the new transform operation to rewrite the function call. This is identical to defining a regular operation in a dialect. The Transform dialect requires operations to implement the `TransformOpInterface` as well as `MemoryEffectsOpInterface` to indicate whether the operands are consumed or only read:

***include/MyExtension.td***
```tablegen
// Define the new operation. By convention, prefix its name with the name of the dialect 
// extension, "my.". The full operation name will be further prefixed with "transform.".
def ChangeCallTargetOp : Op<Transform_Dialect, "my.change_call_target",               // (1)
    // Indicate that the operation implements the required TransformOpInterface and
    // MemoryEffectsOpInterface.
    [DeclareOpInterfaceMethods<TransformOpInterface>,                                  // (2)
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
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
    TransformHandleTypeInterface:$call,                                                // (3)
    StrAttr:$new_target);                                                              // (4)

  // The results are empty as the transformation does not produce any new payload.
  let results = (outs);                                                                // (5)

  // Provide nice syntax.
  let assemblyFormat = "$call `,` $new_target attr-dict `:` type($call)";              // (6)
}
```

The description (elided above) reads: *"For each `func.call` payload operation associated with the handle, changes its callee to be the symbol whose name is provided as an attribute to this operation. Generates a silenceable failure if the operand is associated with payload operations that are not `func.call`. Only reads the operand."* — it states the effect on the operand and the failure mode, as every transform op's description should.

- **(1)** The full name is `transform.my.change_call_target`; the class is `mlir::transform::ChangeCallTargetOp` (the dialect's C++ namespace).
- **(2)** `DeclareOpInterfaceMethods<I>` attaches interface `I` *and* declares its methods in the generated class, leaving their definitions to us: `apply` for `TransformOpInterface`, `getEffects` for `MemoryEffectsOpInterface`.
- **(3)** The operand is a **handle**: a transform IR value that stands for a list of payload ops. `TransformHandleTypeInterface` accepts any handle type (`!transform.any_op`, `!transform.op<"func.call">`, ...) and rejects non-handle types such as parameters (`!transform.param<i64>`).
- **(4)** The new callee is a plain `StrAttr`, not a `SymbolRefAttr`: the symbol lives in the payload, not in the transform IR, so a symbol reference would fail symbol verification of the transform module.
- **(5)** No results: the op modifies existing payload ops in place and creates nothing a later transform op would need a handle to.
- **(6)** The custom syntax, `transform.my.change_call_target %h, "name" : !transform.any_op`.

The generated op class shows what each field turned into. The full class is about a hundred lines; this command keeps the lines that matter here:

```bash
cd /Users/roy/study/mlir/transform/Ch2
mlir-tblgen -gen-op-decls -I /opt/homebrew/opt/llvm@20/include include/MyExtension.td \
  | sed -n '/^class ChangeCallTargetOp :/,/^};/p' \
  | grep -E '^class|"transform\.my|getCall\(\) \{|getCallMutable\(\) \{|getNewTarget\(\);|build\(.*, ::mlir::Value call, ::llvm::StringRef|apply\(|getEffects\(|verifyInvariants\(\);|parse\(|void print\('
```

Real output **[2]** (abridged by the `grep`):

```text
class ChangeCallTargetOp : public ::mlir::Op<ChangeCallTargetOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::transform::TransformOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
    return ::llvm::StringLiteral("transform.my.change_call_target");
  ::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface> getCall() {
  ::mlir::OpOperand &getCallMutable() {
  ::llvm::StringRef getNewTarget();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value call, ::llvm::StringRef new_target);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value call, ::llvm::StringRef new_target);
  ::llvm::LogicalResult verifyInvariants();
  ::mlir::DiagnosedSilenceableFailure apply(::mlir::transform::TransformRewriter &rewriter, ::mlir::transform::TransformResults &transformResults, ::mlir::transform::TransformState &state);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
```

| Generated member | Comes from |
|---|---|
| `ZeroResults`, `OneOperand`, `ZeroRegions`, `ZeroSuccessors` traits | `results = (outs)` and the single operand |
| `TransformOpInterface::Trait`, `MemoryEffectOpInterface::Trait` | the two `DeclareOpInterfaceMethods` |
| `getOperationName()` returning `"transform.my.change_call_target"` | `Transform_Dialect` + the mnemonic |
| `getCall()` (typed as a handle) and `getCallMutable()` (the `OpOperand`) | `TransformHandleTypeInterface:$call` |
| `getNewTarget()` returning a `StringRef` | `StrAttr:$new_target` |
| `build(..., Value call, StringRef new_target)` | the arguments (for creating the op from C++) |
| `verifyInvariants()` | the operand type and attribute constraints (section 3.2) |
| `apply(...)`, `getEffects(...)` — **declared, not defined** | `DeclareOpInterfaceMethods`; we define them in sections 3.3 and 3.4 |
| `parse`, `print` | `assemblyFormat` |

### 3.2 What ODS verifies for free

The generated `verifyInvariants()` and parser enforce the argument constraints before any of our C++ runs. A non-handle operand, written in the generic form so that it gets past the parser:

```bash
cd /Users/roy/study/mlir/transform/Ch2
./build/transform-opt-ch2 --transform-interpreter <<'EOF'
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %p = transform.param.constant 42 : i64 -> !transform.param<i64>
    "transform.my.change_call_target"(%p) <{new_target = "updated"}> : (!transform.param<i64>) -> ()
    transform.yield
  }
}
EOF
```

Real output **[3]** (exit status 1):

```text
<stdin>:4:5: error: 'transform.my.change_call_target' op operand #0 must be TransformHandleTypeInterface instance, but got '!transform.param<i64>'
    "transform.my.change_call_target"(%p) <{new_target = "updated"}> : (!transform.param<i64>) -> ()
    ^
<stdin>:4:5: note: see current operation: "transform.my.change_call_target"(%0) <{new_target = "updated"}> : (!transform.param<i64>) -> ()
```

The same op in the custom syntax fails earlier, in the generated parser, which parses `type($call)` as a `TransformHandleTypeInterface` directly:

```bash
cd /Users/roy/study/mlir/transform/Ch2
./build/transform-opt-ch2 --transform-interpreter <<'EOF'
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %p = transform.param.constant 42 : i64 -> !transform.param<i64>
    transform.my.change_call_target %p, "updated" : !transform.param<i64>
    transform.yield
  }
}
EOF
```

Real output **[4]** (exit status 1):

```text
<stdin>:4:53: error: custom op 'transform.my.change_call_target' invalid kind of type specified
    transform.my.change_call_target %p, "updated" : !transform.param<i64>
                                                    ^
```

And the attribute is mandatory:

```bash
cd /Users/roy/study/mlir/transform/Ch2
./build/transform-opt-ch2 --transform-interpreter <<'EOF'
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    "transform.my.change_call_target"(%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
}
EOF
```

Real output **[5]** (exit status 1):

```text
<stdin>:3:5: error: 'transform.my.change_call_target' op requires attribute 'new_target'
    "transform.my.change_call_target"(%arg0) : (!transform.any_op) -> ()
    ^
<stdin>:3:5: note: see current operation: "transform.my.change_call_target"(%arg0) : (!transform.any_op) -> ()
```

All three are rejected while the transform script is parsed and verified, before the interpreter touches the payload: **constraints written in ODS turn mistakes in a schedule into compile-time errors on the schedule, not half-applied transformations.**

### 3.3 `apply`: the transformation and its tri-state result

To finalize the definition we implement the interface methods. `TransformOpInterface` requires only one method, `apply`, that performs the actual transformation. It is good practice to limit the body of `apply` to manipulating Transform dialect constructs (handles, diagnostics) and keep the actual transformation in a standalone function so it can be used from other places in the code. Like in rewrite patterns, all IR must be modified through the provided rewriter.

The standalone function is a one-liner — `func.call`'s generated `setCallee(StringRef)` replaces its `callee` attribute:

***lib/MyExtension.cpp***
```cpp
static void updateCallee(mlir::func::CallOp call, llvm::StringRef newTarget) {
  call.setCallee(newTarget);
}
```

and `apply` wraps it:

***lib/MyExtension.cpp***
```cpp
// Implementation of our transform dialect operation.
// This operation returns a tri-state result that can be one of:
// - success when the transformation succeeded;
// - definite failure when the transformation failed in such a way that
//   following transformations are impossible or undesirable, typically it could
//   have left payload IR in an invalid state; it is expected that a diagnostic
//   is emitted immediately before returning the definite error;
// - silenceable failure when the transformation failed but following
//   transformations are still applicable, typically this means a precondition
//   for the transformation is not satisfied and the payload IR has not been
//   modified. The silenceable failure additionally carries a Diagnostic that
//   can be emitted to the user.
::mlir::DiagnosedSilenceableFailure mlir::transform::ChangeCallTargetOp::apply(
    // The rewriter that should be used when modifying IR.
    ::mlir::transform::TransformRewriter &rewriter,
    // The list of payload IR entities that will be associated with the
    // transform IR values defined by this transform operation. In this case, it
    // can remain empty as there are no results.
    ::mlir::transform::TransformResults &results,
    // The transform application state. This object can be used to query the
    // current associations between transform IR values and payload IR entities.
    // It can also carry additional user-defined state.
    ::mlir::transform::TransformState &state) {

  // First, we need to obtain the list of payload operations that are associated
  // with the operand handle.
  auto payload = state.getPayloadOps(getCall());                               // (1)

  // Then, we iterate over the list of operands and call the actual IR-mutating
  // function. We also check the preconditions here.
  for (Operation *payloadOp : payload) {
    auto call = dyn_cast<::mlir::func::CallOp>(payloadOp);
    if (!call) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError() << "only applies to func.call payloads";      // (2)
      diag.attachNote(payloadOp->getLoc()) << "offending payload";             // (3)
      return diag;
    }

    updateCallee(call, getNewTarget());                                        // (4)
  }

  // If everything went well, return success.
  return DiagnosedSilenceableFailure::success();                               // (5)
}
```

- **(1)** `state.getPayloadOps(handle)` maps the transform IR value to the payload ops currently associated with it — the core lookup of the interpreter.
- **(2)** A wrong payload op is a **silenceable** failure: the precondition is checked before anything is modified, so the payload is still valid and an enclosing construct may decide to carry on. `emitSilenceableError()` creates the diagnostic attached to this transform op.
- **(3)** The note points at the payload op that caused the failure, so the user sees both sides: the transform op and the payload.
- **(4)** The transformation itself, through the standalone function. `setCallee` only changes an attribute; a transformation that replaces or erases ops would go through `rewriter` so the interpreter can keep handles up to date.
- **(5)** Success. The third outcome, a **definite** failure (`DiagnosedSilenceableFailure::definiteFailure()`, after emitting an error), is for when the payload may have been left half-transformed; this op never needs it.

`invalid.mlir` hands the op a handle to the whole payload module (`%arg0`), which is not a `func.call`. Its `expected-*` comments are for `--verify-diagnostics` (the second run below):

***test_Example/transform/Ch2/invalid.mlir***
```mlir
// expected-note @below {{offending payload}}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{only applies to func.call payloads}}
    transform.my.change_call_target %arg0, "updated" : !transform.any_op
    transform.yield
  }
}
```

```bash
cd /Users/roy/study/mlir/transform/Ch2
./build/transform-opt-ch2 ../../test_Example/transform/Ch2/invalid.mlir --transform-interpreter
```

Real output **[6]** (exit status 1):

```text
../../test_Example/transform/Ch2/invalid.mlir:8:5: error: only applies to func.call payloads
    transform.my.change_call_target %arg0, "updated" : !transform.any_op
    ^
../../test_Example/transform/Ch2/invalid.mlir:5:1: note: offending payload
module attributes {transform.with_named_sequence} {
^
```

The file holds both the payload and the script (the module *is* the payload root, which is why the note points at line 5). At the top level of a named sequence nothing silences the failure, so the interpreter reports it as an error and the pass fails. The test's own `RUN` line checks exactly these two diagnostics against the `expected-*` comments, and succeeds when they match:

```bash
cd /Users/roy/study/mlir/transform/Ch2
./build/transform-opt-ch2 ../../test_Example/transform/Ch2/invalid.mlir --transform-interpreter \
  --split-input-file --verify-diagnostics; echo "exit status: $?"
```

Real output **[7]**:

```text
exit status: 0
```

What makes the failure *silenceable* is that a surrounding `transform.sequence ... failures(suppress)` can swallow it. Here the op is first given a handle to the `func.func` ops (wrong, suppressed), then a correct handle to the call:

```bash
cd /Users/roy/study/mlir/transform/Ch2
./build/transform-opt-ch2 --transform-interpreter <<'EOF'
func.func private @orig()
func.func private @updated()
func.func @test() {
  call @orig() : () -> ()
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.sequence %arg0 : !transform.any_op failures(suppress) {
    ^bb0(%root: !transform.any_op):
      transform.my.change_call_target %funcs, "updated" : !transform.any_op
    }
    %call = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.my.change_call_target %call, "updated" : !transform.any_op
    transform.yield
  }
}
EOF
```

Real output **[8]**:

```mlir
module {
  func.func private @orig()
  func.func private @updated()
  func.func @test() {
    call @updated() : () -> ()
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.sequence %arg0 : !transform.any_op failures(suppress) {
      ^bb0(%arg1: !transform.any_op):
        transform.my.change_call_target %0, "updated" : !transform.any_op
      }
      %1 = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.my.change_call_target %1, "updated" : !transform.any_op
      transform.yield 
    }
  }
}
```

No diagnostic, exit status 0, and the call was retargeted by the second op. The first op failed on the first `func.func` before modifying anything, so suppressing it was safe — which is exactly the contract a silenceable failure promises. **Checking every precondition before the first mutation is what lets an op report a silenceable failure honestly; a failure after a partial rewrite must be definite.**

The op does *not* check that the new target exists: the attribute is a string the transform IR knows nothing about (section 3.1, (4)). The payload verifier catches a dangling callee after the interpreter pass:

```bash
cd /Users/roy/study/mlir/transform/Ch2
./build/transform-opt-ch2 --transform-interpreter <<'EOF'
func.func private @orig()
func.func @test() {
  call @orig() : () -> ()
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %call = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.my.change_call_target %call, "nonexistent" : !transform.any_op
    transform.yield
  }
}
EOF
```

Real output **[9]** (exit status 1):

```text
<stdin>:3:3: error: 'func.call' op 'nonexistent' does not reference a valid function
  call @orig() : () -> ()
  ^
<stdin>:3:3: note: see current operation: "func.call"() <{callee = @nonexistent}> : () -> ()
```

The error is reported against the *payload* `func.call`, by `func.call`'s own symbol verification, and the transform op that caused it isn't mentioned. A production version of this op would look the symbol up (`SymbolTable::lookupNearestSymbolFrom`) in `apply` and return a silenceable failure instead.

### 3.4 `getEffects`: what the op does to handles and payload

The implementation of `MemoryEffectsOpInterface` must specify the effects this operation has on its operands (consumed or only read) and on the payload IR (modified or only read). The Transform dialect's checks rely on these effects being present (in builds with assertions enabled, registering an op whose effects are missing asserts).

***lib/MyExtension.cpp***
```cpp
void mlir::transform::ChangeCallTargetOp::getEffects(
    ::llvm::SmallVectorImpl<::mlir::MemoryEffects::EffectInstance> &effects) {
  // Indicate that the `call` handle is only read by this operation because the
  // associated operation is not erased but rather modified in-place, so the
  // reference to it remains valid.
  onlyReadsHandle(getCallMutable(), effects);

  // Indicate that the payload is modified by this operation.
  modifiesPayload(effects);
}
```

The helpers take the operand as an `OpOperand` (`getCallMutable()`, section 3.1's table), and translate handle semantics into ordinary memory effects on special resources: *consume* = read + free, *produce* = allocate + write, *only read* = read; `modifiesPayload` writes the `transform.payload_ir` resource. The choice matters to the interpreter: a handle that an op **consumes** is invalidated, together with every other handle pointing to the same payload ops or ops nested in them, and any later use is an error.

`change_call_target` only edits an attribute, so the `func.call` ops survive and the handle stays valid. Using it twice in a row works:

```bash
cd /Users/roy/study/mlir/transform/Ch2
./build/transform-opt-ch2 --transform-interpreter <<'EOF'
func.func private @orig()
func.func private @updated()
func.func private @final()
func.func @test() {
  call @orig() : () -> ()
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %call = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.my.change_call_target %call, "updated" : !transform.any_op
    transform.my.change_call_target %call, "final" : !transform.any_op
    transform.yield
  }
}
EOF
```

Real output **[10]**:

```mlir
module {
  func.func private @orig()
  func.func private @updated()
  func.func private @final()
  func.func @test() {
    call @final() : () -> ()
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.my.change_call_target %0, "updated" : !transform.any_op
      transform.my.change_call_target %0, "final" : !transform.any_op
      transform.yield 
    }
  }
}
```

The same script against a copy of the chapter in which `getEffects` says `consumesHandle` instead (built in a temporary directory, then removed):

```bash
cd /Users/roy/study/mlir/transform/Ch2
T=$(mktemp -d) && cp -R CMakeLists.txt include lib transform-opt "$T" \
&& sed -i '' 's/onlyReadsHandle(getCallMutable(), effects);/consumesHandle(getCallMutable(), effects);/' "$T/lib/MyExtension.cpp" \
&& cmake -S "$T" -B "$T/build" -G Ninja > /dev/null && cmake --build "$T/build" > /dev/null \
&& "$T/build/transform-opt-ch2" --transform-interpreter <<'EOF'; echo "transform-opt-ch2 exit status: $?"; rm -rf "$T"
func.func private @orig()
func.func private @updated()
func.func private @final()
func.func @test() {
  call @orig() : () -> ()
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %call = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.my.change_call_target %call, "updated" : !transform.any_op
    transform.my.change_call_target %call, "final" : !transform.any_op
    transform.yield
  }
}
EOF
```

Real output **[11]**:

```text
<stdin>:12:5: error: op uses a handle invalidated by a previously executed transform op
    transform.my.change_call_target %call, "final" : !transform.any_op
    ^
<stdin>:10:13: note: handle to invalidated ops
    %call = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.any_op
            ^
<stdin>:11:5: note: invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them
    transform.my.change_call_target %call, "updated" : !transform.any_op
    ^
<stdin>:5:3: note: ancestor payload op
  call @orig() : () -> ()
  ^
<stdin>:5:3: note: nested payload op
transform-opt-ch2 exit status: 1
```

The C++ of `apply` is identical in both builds; only the declared effect changed, and the interpreter's use-after-consume check (one of its "expensive checks", on by default — `--transform-interpreter=disable-expensive-checks` turns them off) rejected the second use. **The effects are the op's contract with the interpreter: declare "consumes" whenever `apply` erases or replaces the payload, so stale handles become errors instead of dangling pointers — and "only reads" when it doesn't, so schedules can keep using the handle.**

---

## 4. Registration and Usage

### 4.1 The registration hook and the tool

This is enough to define transform operations. The only remaining piece is the extension registration hook that the project's `main` calls. It is declared in `include/MyExtension.h` (section 2.4) and defined at the end of `lib/MyExtension.cpp`:

***lib/MyExtension.cpp***
```cpp
void registerMyExtension(::mlir::DialectRegistry &registry) {
  registry.addExtensions<MyExtension>();
}
```

`addExtensions` doesn't load anything; it records "when the Transform dialect is loaded into a context built from this registry, run `MyExtension`". The tool is `mlir-opt`'s driver, `MlirOptMain`, plus that registration:

***transform-opt/transform-opt.cpp***
```cpp
int main(int argc, char **argv) {
  // Register all "core" dialects and our transform dialect extension.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);           // (1)
  registerMyExtension(registry);                   // (2)

  // Register transform interpreter pass.
  mlir::transform::registerInterpreterPass();      // (3)

  // Register a handful of cleanup passes that we can run to make the output IR
  // look nicer.
  mlir::registerCanonicalizerPass();               // (4)
  mlir::registerCSEPass();
  mlir::registerSymbolDCEPass();

  // Delegate to the MLIR utility for parsing and pass management.
  return mlir::MlirOptMain(argc, argv, "transform-opt-ch2", registry)
                 .succeeded()
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
```

- **(1)** All upstream dialect extensions, including the ones that add `transform.structured.*` (Linalg) and `transform.loop.*` (SCF) ops to the Transform dialect — the same mechanism as ours.
- **(2)** Our extension.
- **(3)** `--transform-interpreter`, the pass that finds the entry point `@__transform_main` and applies it to the payload.
- **(4)** Only these three passes (plus the interpreter) are registered: `canonicalize`, `cse` (common subexpression elimination), and `symbol-dce` (dead code elimination of unused symbols). The tool isn't a full `mlir-opt`.

(The file also declares `test::registerTestTransformDialectExtension`, which `main` never calls.)

Without (2) the op is unknown — which is exactly the situation of stock `mlir-opt`:

```bash
cd /Users/roy/study/mlir/transform/Ch2
mlir-opt ../../test_Example/transform/Ch2/ops.mlir --transform-interpreter
```

Real output **[12]** (exit status 1):

```text
../../test_Example/transform/Ch2/ops.mlir:24:5: error: custom op 'transform.my.change_call_target' is unknown
    transform.my.change_call_target %call, "updated" : !transform.any_op
    ^
```

The error comes from the parser: the op's name lives in the Transform dialect's namespace, but only a registry carrying the extension teaches the dialect about it.

### 4.2 A minimal use: `ops.mlir`

`ops.mlir` holds a payload with one call to `@orig` and a script that matches every `func.call` and retargets it to `@updated`:

***test_Example/transform/Ch2/ops.mlir***
```mlir
func.func private @orig()
func.func private @updated()

// CHECK-LABEL: func @test
func.func @test() {
  // CHECK: call @updated
  call @orig() : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %call = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.my.change_call_target %{{.*}}, "updated" : !transform.any_op
    transform.my.change_call_target %call, "updated" : !transform.any_op
    transform.yield
  }
}
```

```bash
cd /Users/roy/study/mlir/transform/Ch2
./build/transform-opt-ch2 ../../test_Example/transform/Ch2/ops.mlir --transform-interpreter
```

Real output **[13]**:

```mlir
module {
  func.func private @orig()
  func.func private @updated()
  func.func @test() {
    call @updated() : () -> ()
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      %0 = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.my.change_call_target %0, "updated" : !transform.any_op
      transform.yield 
    }
  }
}
```

- `call @orig()` became `call @updated()`: `structured.match` collected the one `func.call` into `%call`, and our `apply` rewrote its callee.
- The script stays in the output — the interpreter applies it but doesn't erase it — and it round-trips in the op's custom syntax, printed by the generated `print`.

### 4.3 Using the op in a full schedule: `sequence.mlir`

After registering the extension, the new op can be used in any Transform dialect script, right next to the upstream ones. `sequence.mlir` is Chapter 1's fully connected layer with ReLU — a `linalg.matmul`, a bias `linalg.elemwise_binary` add and a `linalg.elemwise_binary` max with 0 — together with a declaration of the hand-written `@microkernel` the schedule will call:

***test_Example/transform/Ch2/sequence.mlir***
```mlir
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
```

The schedule tiles the ReLU by 8×32, fuses the add and the matmul into that loop, tiles the add again by 4×4 and fuses the matmul into the inner loop, wraps the inner body in a loop of size 1 so there is a region to outline, outlines it into a function `@outlined`, and finally uses our op to make the resulting call target `@microkernel` instead:

***test_Example/transform/Ch2/sequence.mlir***
```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op,
      %arg1: !transform.op<"linalg.matmul">,
      %arg2: !transform.op<"linalg.elemwise_binary">) {
    // Since the %arg2 handle is associated with both elementwise operations,
    // we need to split it into two handles so we can target only the second
    // elementwise operation.
    %add, %max = transform.split_handle %arg2 : (!transform.op<"linalg.elemwise_binary">)
        -> (!transform.any_op, !transform.any_op)

    // The actual tiling transformation takes tile sizes as attributes. It produces a
    // handle to the loop generated during tiling.
    %tiled, %loop = transform.structured.tile_using_forall %max tile_sizes [8, 32]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // We can now fuse the other operations into the loop. Here, we fuse
    // operations one-by-one. This requires the operation that is being fused
    // to define the value used within the loop, so the order of such fusions
    // is important. We could also use "transform.merge_handles" to obtain
    // a single handle to all operations and give it to `fuse_into_containing_op`
    // that would take care of the ordering in this case.
    %add_fused, %loop2 = transform.structured.fuse_into_containing_op %add into %loop
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %matmul_fused, %loop3 = transform.structured.fuse_into_containing_op %arg1 into %loop2
        : (!transform.op<"linalg.matmul">, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Tile again to get the desired size. Note that this time this tiles the
    // "add" operation and fuses matmul into the loop, but doesn't affect the
    // "max" operation. This illustrates the precise targeting with the transform
    // dialect. Otherwise, it is difficult to differentiate "add" and "max", both
    // of which having the same kind.
    %tiled_second, %loop_second = transform.structured.tile_using_forall %add_fused tile_sizes [4, 4]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %matmul_fused_2, %loop_second_2 =
        transform.structured.fuse_into_containing_op %matmul_fused into %loop_second
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Since outlining is currently only implemented for region-holding operations
    // such as loops, use tiling to size 1 to materialize the outer loop that is
    // going to be outlined.
    %_0, %loop_third = transform.structured.tile_using_forall %tiled_second tile_sizes [1]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %_1, %outline_target = transform.structured.fuse_into_containing_op %matmul_fused_2 into %loop_third
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %func, %call = transform.loop.outline %outline_target {func_name = "outlined"}
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Rewrite the call target.
    transform.my.change_call_target %call, "microkernel" : !transform.any_op

    transform.yield
  }
}
```

Points to note in MLIR 20's syntax:

- `transform.structured.tile_using_forall` returns **(tiled op, loop)**, in that order, and `transform.structured.fuse_into_containing_op` returns **(fused op, new containing loop)**. Fusion replaces the loop, so each fusion's second result is the handle to use for the next one (`%loop` → `%loop2` → `%loop3`).
- `transform.loop.outline` returns a handle to the new function and one to the `func.call` it left in place of the loop. That `%call` handle is what our op receives — it composes with upstream ops because all of them speak the same handle language.

The entry point takes two extra arguments beyond the payload root, `%arg1` and `%arg2`. The interpreter has to be told what to bind them to; without that it refuses to start:

```bash
cd /Users/roy/study/mlir/transform/Ch2
./build/transform-opt-ch2 ../../test_Example/transform/Ch2/sequence.mlir --transform-interpreter
```

Real output **[14]** (exit status 1):

```text
../../test_Example/transform/Ch2/sequence.mlir:59:3: error: operation expects 2 extra value bindings, but 0 were provided to the interpreter
  transform.named_sequence @__transform_main(
  ^
```

The interpreter pass's options, from the tool's `--help`:

```bash
cd /Users/roy/study/mlir/transform/Ch2
./build/transform-opt-ch2 --help | grep -A4 -- '--transform-interpreter '
```

Real output **[15]**:

```text
      --transform-interpreter                                -   transform dialect interpreter
        --debug-bind-trailing-args=<string>                  - Binds trailing arguments of the entry point to the payload operations with specified names.
        --debug-payload-root-tag=<string>                    - Select the operation with 'transform.target_tag' attribute having the given value as payload IR root. If empty select the pass anchor operation as the payload IR root.
        --disable-expensive-checks                           - Disable expensive checks in the interpreter for a faster run.
        --entry-point=<string>                               - Entry point of the pass pipeline.
```

`debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary` binds `%arg1` to all `linalg.matmul` ops in the payload and `%arg2` to all `linalg.elemwise_binary` ops (both of them, hence the `split_handle`). It is a debugging aid; a real schedule would find the ops itself with `transform.structured.match`. The full pipeline, as in the test's `RUN` line, runs the interpreter and then the cleanup passes:

```bash
cd /Users/roy/study/mlir/transform/Ch2
./build/transform-opt-ch2 ../../test_Example/transform/Ch2/sequence.mlir \
  --pass-pipeline="builtin.module(transform-interpreter{debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary},canonicalize,cse,symbol-dce)"
```

Real output **[16]**:

```mlir
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
    transform.named_sequence @__transform_main(%arg0: !transform.any_op, %arg1: !transform.op<"linalg.matmul">, %arg2: !transform.op<"linalg.elemwise_binary">) {
      %0:2 = transform.split_handle %arg2 : (!transform.op<"linalg.elemwise_binary">) -> (!transform.any_op, !transform.any_op)
      %tiled_op, %forall_op = transform.structured.tile_using_forall %0#1 tile_sizes [8, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %0#0 into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %fused_op_0, %new_containing_op_1 = transform.structured.fuse_into_containing_op %arg1 into %new_containing_op : (!transform.op<"linalg.matmul">, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %tiled_op_2, %forall_op_3 = transform.structured.tile_using_forall %fused_op tile_sizes [4, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      %fused_op_4, %new_containing_op_5 = transform.structured.fuse_into_containing_op %fused_op_0 into %forall_op_3 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %tiled_op_6, %forall_op_7 = transform.structured.tile_using_forall %tiled_op_2 tile_sizes [1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      %fused_op_8, %new_containing_op_9 = transform.structured.fuse_into_containing_op %fused_op_4 into %forall_op_7 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %function, %call = transform.loop.outline %new_containing_op_9 {func_name = "outlined"} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.my.change_call_target %call, "microkernel" : !transform.any_op
      transform.yield 
    }
  }
}
```

Reading it from the outside in:

- `@fc_relu`'s outer `scf.forall ... in (64, 16)` is the 8×32 tiling of the 512×512 ReLU (512/8 = 64, 512/32 = 16). The ReLU (`max_signed`) stays at that level, on an 8×32 slice.
- Inside it, the `scf.forall ... in (2, 8)` is the 4×4 tiling of the fused add (8/4 = 2, 32/4 = 8). Its body takes five slices — 4×512 of `lhs`, 512×4 of `rhs`, and three 4×4 slices (output-as-init for the matmul, bias, and the add's output) — and passes them to **`func.call @microkernel`**: the call `transform.loop.outline` created pointed at `@outlined`, and our op retargeted it. The five arguments match `@microkernel`'s declared signature.
- `@outlined` still exists: it holds the outlined size-1 loop (4 iterations of a 1×512 · 512×4 matmul + add). `transform.loop.outline` creates it as a public function, so `symbol-dce` keeps it although nothing calls it any more.

The cleanup passes matter for readability, not correctness. Counting a few kinds of ops with and without them:

```bash
cd /Users/roy/study/mlir/transform/Ch2
for p in "" ",canonicalize,cse,symbol-dce"; do
  echo "pipeline: transform-interpreter$p"
  ./build/transform-opt-ch2 ../../test_Example/transform/Ch2/sequence.mlir \
    --pass-pipeline="builtin.module(transform-interpreter{debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary}$p)" \
    | grep -oE 'affine.apply|tensor.extract_slice|func.func @[a-z_]+|func.call @[a-z]+' | sort | uniq -c
done
```

Real output **[17]**:

```text
pipeline: transform-interpreter
  18 affine.apply
   1 func.call @microkernel
   1 func.func @fc_relu
   1 func.func @outlined
  16 tensor.extract_slice
pipeline: transform-interpreter,canonicalize,cse,symbol-dce
   4 affine.apply
   1 func.call @microkernel
   1 func.func @fc_relu
   1 func.func @outlined
  14 tensor.extract_slice
```

Tiling and fusion emit index computations (`affine.apply`) and slices per step, with many duplicates; canonicalization and CSE fold them from 18 to 4. That's why the tool registers those passes (section 4.1, (4)): **transform ops can stay simple and emit straightforward IR, because cleanup is a separate, reusable pass that runs after the schedule.**

---

## 5. Build and Run

The top-level README of this repository describes the shared toolchain (Homebrew LLVM/MLIR 20). This chapter builds standalone, from its own directory.

### 5.1 Building

```bash
cd /Users/roy/study/mlir/transform/Ch2
cmake -S . -B build -G Ninja
cmake --build build          # → ./build/transform-opt-ch2
```

No `-DMLIR_DIR` or compiler flags are needed: the shell has `CXX`/`CC` set to Homebrew llvm@20's clang and `/opt/homebrew/opt/llvm@20/bin` on `PATH`, so `find_package(MLIR)` finds `llvm@20/lib/cmake/mlir`. If your environment differs, pass them explicitly:

```text
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++
```

The build runs `mlir-tblgen` twice (`build/include/MyExtension.{h,cpp}.inc`), compiles the extension into `build/lib/libMyExtensionCh2.a`, and links the tool against it and `libMLIR.dylib`. Appendix B explains the wiring, including why the standalone build defaults to Release.

### 5.2 Running the chapter's tests

The three inputs are the chapter's tests; their `RUN` lines pipe the output into `FileCheck` (an LLVM tool that matches the `// CHECK:` comments in a file against text) or use `--verify-diagnostics`. Running all three exactly as their `RUN` lines do:

```bash
cd /Users/roy/study/mlir/transform/Ch2
T=../../test_Example/transform/Ch2
./build/transform-opt-ch2 $T/ops.mlir --transform-interpreter | FileCheck $T/ops.mlir && echo "ops.mlir: PASS"
./build/transform-opt-ch2 $T/invalid.mlir --transform-interpreter --split-input-file --verify-diagnostics && echo "invalid.mlir: PASS"
./build/transform-opt-ch2 $T/sequence.mlir \
  --pass-pipeline="builtin.module(transform-interpreter{debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary},canonicalize,cse,symbol-dce)" \
  | FileCheck $T/sequence.mlir && echo "sequence.mlir: PASS"
```

Real output **[18]**:

```text
ops.mlir: PASS
invalid.mlir: PASS
sequence.mlir: PASS
```

The outputs being checked are the ones explained in sections 4.2 (`ops.mlir`), 3.3 (`invalid.mlir`) and 4.3 (`sequence.mlir`). `sequence.mlir`'s `CHECK` lines encode section 4.3's reading: two nested `scf.forall`, five `tensor.extract_slice` feeding `func.call @microkernel`, no `linalg.matmul` or add left in the inner loop, and the `max_signed` ReLU in the outer loop.

---

## 6. Deriving the extension, the tool and the build yourself

Sections 2–4 explain the code and appendix B the build; this section shows how to arrive at them without having them, using only the installed headers and CMake modules, `mlir-tblgen`, and errors from the compiler and the linker. Every file of the chapter is new:

| Part | Lines | Where it comes from |
|---|---|---|
| `include/MyExtension.td`: the prelude `include`s and the op record with its two `DeclareOpInterfaceMethods` | 17–28 | What the op must implement; `mlir-tblgen` lists the methods you owe (section 6.3) |
| `include/MyExtension.td`: `arguments`, `results`, `assemblyFormat` | 45–53 | The handle and the new callee name; a `StrAttr`, because the callee is a payload symbol (section 6.2) |
| `lib/MyExtension.cpp`: `class MyExtension`, `init()` with `declareGeneratedDialect` and `registerTransformOps` | 29–75 | The extension API in `mlir/Dialect/Transform/IR/TransformDialect.h` (section 6.1) |
| `lib/MyExtension.cpp`: `updateCallee` → `call.setCallee(newTarget)` | 80–82 | The generated `func.call` class, `FuncOps.h.inc` (section 6.2) |
| `lib/MyExtension.cpp`: `getEffects` with `onlyReadsHandle`, `modifiesPayload` | 130–139 | The effect helpers in `TransformInterfaces.h` (section 6.4) |
| `lib/MyExtension.cpp`: the bodies of `apply` and `getEffects` | 96–139 | The linker's list of undefined symbols (section 6.5) |
| `include/MyExtension.h`, `registerMyExtension` (`lib/MyExtension.cpp`), and `transform-opt/transform-opt.cpp` | 14–22, 141–143, 13–49 | `mlir-opt`'s own building blocks in the installed headers, and the parser's "unknown op" errors (section 6.6) |
| `include/CMakeLists.txt`, `lib/CMakeLists.txt`, and the targets in `CMakeLists.txt` | 1–13, 1–20, 39–48 | The `mlir-tblgen` generators and the CMake functions that wrap them (section 6.7) |
| The link lines in `CMakeLists.txt` and `lib/CMakeLists.txt` | 50–54, 18–19 | The linker again, on a copy without `libMLIR.dylib` (section 6.8) |
| The standalone guard in `CMakeLists.txt` | 7–37 | Appendix B.1, including the Debug build that fails to link |

### 6.1 Step 1: find the extension API

```bash
cd /Users/roy/study/mlir/transform/Ch2
grep -n 'void registerTransformOps\|void registerTypes\|void declareDependentDialect\|void declareGeneratedDialect\|^class TransformDialectExtension' \
  /opt/homebrew/opt/llvm@20/include/mlir/Dialect/Transform/IR/TransformDialect.h
```

Real output **[19]**:

```text
117:class TransformDialectExtension
192:  void registerTransformOps() {
204:  void registerTypes() {
219:  void declareDependentDialect() {
231:  void declareGeneratedDialect() {
```

That is the whole vocabulary of `init()`: register ops (and, in Chapter 3, types), declare dependent and generated dialects. The comments above each function in the header are the source of section 2.2's table. The payload this op works on (`func.call`) and the payload the schedule creates (`scf.forall` loops, `func.func` via outlining) say which dialects to declare as generated.

### 6.2 Step 2: find the payload API

The transformation itself — "change the callee" — is a setter on `func.call`, whose class is generated from its ODS definition into an installed header:

```bash
cd /Users/roy/study/mlir/transform/Ch2
grep -n 'void setCallee(::llvm::StringRef' /opt/homebrew/opt/llvm@20/include/mlir/Dialect/Func/IR/FuncOps.h.inc
```

Real output **[20]**:

```text
385:  void setCallee(::llvm::StringRef attrValue);
```

`callee` is a `FlatSymbolRefAttr` in `func.call`'s ODS, and ODS generates a `StringRef` setter for it — so the `StrAttr` of our op can be passed straight through (`getNewTarget()` returns a `StringRef`, section 3.1). The result `apply` must produce is the payload with `call @updated() : () -> ()` in place of the old call; since `@updated` is checked only by the payload's verifier, the payload has to declare it (section 3.3 shows the verifier rejecting a callee that doesn't exist).

### 6.3 Step 3: let ODS list the methods you owe

Write the `.td` record with the two `DeclareOpInterfaceMethods` and run `mlir-tblgen -gen-op-decls` on it: section 3.1's abridged class (command [2]) shows `apply(...)` and `getEffects(...)` declared with their exact signatures, which you copy into `MyExtension.cpp`. The prelude `include`s are the files that define the names the record uses: `Transform_Dialect` (`TransformDialect.td`), `TransformOpInterface` and `TransformHandleTypeInterface` (`TransformInterfaces.td`), `MemoryEffectsOpInterface` (`SideEffectInterfaces.td`).

### 6.4 Step 4: find the effect helpers

```bash
cd /Users/roy/study/mlir/transform/Ch2
grep -n '^void \(consumesHandle\|producesHandle\|onlyReadsHandle\|modifiesPayload\|onlyReadsPayload\)' \
  /opt/homebrew/opt/llvm@20/include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.h
```

Real output **[21]**:

```text
1267:void consumesHandle(MutableArrayRef<OpOperand> handles,
1269:void producesHandle(ResultRange handles,
1271:void producesHandle(MutableArrayRef<BlockArgument> handles,
1273:void onlyReadsHandle(MutableArrayRef<OpOperand> handles,
1281:void modifiesPayload(SmallVectorImpl<MemoryEffects::EffectInstance> &effects);
1282:void onlyReadsPayload(SmallVectorImpl<MemoryEffects::EffectInstance> &effects);
```

`getEffects` makes one call per operand, one per result (`producesHandle`, for ops that have results), and one for the payload. The handle helpers take `OpOperand`s, which is why the code passes `getCallMutable()` rather than `getCall()`. Choosing between consume and only-read is the question "does `apply` erase or replace the payload ops behind this handle?" (section 3.4).

### 6.5 Step 5: let the linker list the C++ you still owe

With the `.td` file and the extension written but `apply` and `getEffects` not yet defined, the build fails at link time and names exactly what's missing. Demonstrated on a copy of the chapter with both definitions cut out:

```bash
cd /Users/roy/study/mlir/transform/Ch2
T=$(mktemp -d) && cp -R CMakeLists.txt include lib transform-opt "$T" \
&& sed -i '' '/^::mlir::DiagnosedSilenceableFailure mlir::transform::ChangeCallTargetOp::apply(/,$d' "$T/lib/MyExtension.cpp" \
&& printf 'void registerMyExtension(::mlir::DialectRegistry &registry) {\n  registry.addExtensions<MyExtension>();\n}\n' >> "$T/lib/MyExtension.cpp" \
&& cmake -S "$T" -B "$T/build" -G Ninja > /dev/null && cmake --build "$T/build" 2>&1 | grep -A6 '^Undefined symbols'; rm -rf "$T"
```

Real output **[22]**:

```text
Undefined symbols for architecture arm64:
  "mlir::transform::ChangeCallTargetOp::getEffects(llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>&)", referenced from:
      mlir::detail::MemoryEffectOpInterfaceInterfaceTraits::Model<mlir::transform::ChangeCallTargetOp>::getEffects(mlir::detail::MemoryEffectOpInterfaceInterfaceTraits::Concept const*, mlir::Operation*, llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>&) in libMyExtensionCh2.a[2](MyExtension.cpp.o)
  "mlir::transform::ChangeCallTargetOp::apply(mlir::transform::TransformRewriter&, mlir::transform::TransformResults&, mlir::transform::TransformState&)", referenced from:
      mlir::transform::detail::TransformOpInterfaceInterfaceTraits::Model<mlir::transform::ChangeCallTargetOp>::apply(mlir::transform::detail::TransformOpInterfaceInterfaceTraits::Concept const*, mlir::Operation*, mlir::transform::TransformRewriter&, mlir::transform::TransformResults&, mlir::transform::TransformState&) in libMyExtensionCh2.a[2](MyExtension.cpp.o)
ld: symbol(s) not found for architecture arm64
clang++: error: linker command failed with exit code 1 (use -v to see invocation)
```

Each missing symbol is referenced from an interface `Model<ChangeCallTargetOp>` — the glue that `DeclareOpInterfaceMethods` generated to dispatch interface calls to the op's methods. Fill in the two bodies (sections 3.3 and 3.4) and the tool links.

### 6.6 Step 6: build the tool from `mlir-opt`'s parts

The tool only has to do what `mlir-opt` does, plus load the extension. `mlir-opt`'s building blocks are installed: its driver, the registries of all dialects and extensions, and the registration functions of the passes the tests need:

```bash
cd /Users/roy/study/mlir/transform/Ch2
I=/opt/homebrew/opt/llvm@20/include
grep -n 'LogicalResult MlirOptMain(int argc, char \*\*argv, llvm::StringRef toolName' $I/mlir/Tools/mlir-opt/MlirOptMain.h
grep -n 'inline void registerAllDialects(DialectRegistry' $I/mlir/InitAllDialects.h
grep -n 'inline void registerAllExtensions(DialectRegistry' $I/mlir/InitAllExtensions.h
grep -n 'inline void registerInterpreterPass()' $I/mlir/Dialect/Transform/Transforms/Passes.h.inc
grep -n 'inline void register\(Canonicalizer\|CSE\|SymbolDCE\)Pass()' $I/mlir/Transforms/Passes.h.inc
```

Real output **[23]**:

```text
324:LogicalResult MlirOptMain(int argc, char **argv, llvm::StringRef toolName,
107:inline void registerAllDialects(DialectRegistry &registry) {
62:inline void registerAllExtensions(DialectRegistry &registry) {
404:inline void registerInterpreterPass() {
1510:inline void registerCSEPass() {
1527:inline void registerCanonicalizerPass() {
1782:inline void registerSymbolDCEPass() {
```

- Each function is one line of `main` (section 4.1), and each header is one of `transform-opt.cpp`'s `#include`s; the two `Passes.h.inc` files are generated and included through `mlir/Dialect/Transform/Transforms/Passes.h` and `mlir/Transforms/Passes.h`.
- The `MlirOptMain` overload takes the `DialectRegistry`, which is where `registerMyExtension` adds the extension. So the extension needs a function that the tool can call with that registry: `registerMyExtension`, declared in `include/MyExtension.h` next to the op classes (`GET_OP_CLASSES`) and defined in `lib/MyExtension.cpp` (section 4.1).

Leaving the extension out makes `transform.my.change_call_target` unknown (section 4.1's run with stock `mlir-opt`). `registerAllExtensions` is needed just as much, because the schedules also use ops that other extensions add to the Transform dialect. A scratch copy without that line (built by `sed` in a temporary directory) can't parse `ops.mlir`:

```bash
cd /Users/roy/study/mlir/transform/Ch2
T=$(mktemp -d) && cp -R CMakeLists.txt include lib transform-opt "$T" \
&& sed -i '' '/mlir::registerAllExtensions(registry);/d' "$T/transform-opt/transform-opt.cpp" \
&& cmake -S "$T" -B "$T/build" -G Ninja > /dev/null && cmake --build "$T/build" > /dev/null \
&& "$T/build/transform-opt-ch2" ../../test_Example/transform/Ch2/ops.mlir --transform-interpreter; echo "transform-opt-ch2 exit status: $?"; rm -rf "$T"
```

Real output **[24]**:

```text
../../test_Example/transform/Ch2/ops.mlir:22:13: error: custom op 'transform.structured.match' is unknown
    %call = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.any_op
            ^
transform-opt-ch2 exit status: 1
```

`transform.structured.match` is not in the Transform dialect itself either: the Linalg dialect's extension adds it, the same way `MyExtension` adds `transform.my.change_call_target`. **A transform op is available only if the tool's registry carries the extension that defines it, whether the extension is ours or MLIR's own.**

### 6.7 Step 7: find the TableGen generators and the CMake functions

The C++ that `MyExtension.h` and `MyExtension.cpp` include comes from `mlir-tblgen`, and the CMake rules that run it are functions in the installed CMake modules:

```bash
cd /Users/roy/study/mlir/transform/Ch2
mlir-tblgen --help | grep -E -- '--gen-op-(decls|defs|doc) '
C=/opt/homebrew/opt/llvm@20/lib/cmake
grep -nE '^(function|macro)\((mlir_tablegen|add_mlir_doc|add_mlir_example_library|add_public_tablegen_target|add_llvm_example) ' \
  $C/mlir/AddMLIR.cmake $C/llvm/TableGen.cmake $C/llvm/AddLLVM.cmake | sed "s|$C/||"
```

Real output **[25]**:

```text
      --gen-op-decls                                    - Generate op declarations
      --gen-op-defs                                     - Generate op definitions
      --gen-op-doc                                      - Generate dialect documentation
mlir/AddMLIR.cmake:4:function(mlir_tablegen ofn)
mlir/AddMLIR.cmake:201:function(add_mlir_doc doc_filename output_file output_directory command)
mlir/AddMLIR.cmake:270:function(add_mlir_example_library name)
llvm/TableGen.cmake:161:function(add_public_tablegen_target target)
llvm/AddLLVM.cmake:1521:macro(add_llvm_example name)
```

- **`include/CMakeLists.txt`:** one `mlir_tablegen` rule per generator, `-gen-op-decls` for `MyExtension.h.inc` and `-gen-op-defs` for `MyExtension.cpp.inc` (sections 2.4 and 3.1 run them by hand), grouped by `add_public_tablegen_target` into a target the library can depend on; `-gen-op-doc` is what `add_mlir_doc` runs (appendix A).
- **`lib/CMakeLists.txt`:** `add_mlir_example_library` builds `MyExtension.cpp` into `MyExtensionCh2`, with `DEPENDS` on that target so the `.inc` files exist before it compiles.
- **`CMakeLists.txt`:** `add_llvm_example` builds the tool from `transform-opt.cpp`, and the build directory goes on the include path because the `.inc` files are written there (appendix B.1, markers (5) and (6)).
- `CMakeLists.txt` includes the modules that define these functions (`TableGen`, `AddLLVM`, `AddMLIR`) in its standalone guard (appendix B.1, marker (3)).

### 6.8 Step 8: let the linker show which line links MLIR

Both the library and the tool name `MLIR`, i.e. Homebrew's `libMLIR.dylib`, which has everything the code calls: the Transform dialect, `func.call`, `scf`, and `mlir-opt`'s driver. Two scratch copies, one without either `MLIR` line and one without the tool's only, show which line the link needs:

```bash
cd /Users/roy/study/mlir/transform/Ch2
for files in "CMakeLists.txt lib/CMakeLists.txt" "CMakeLists.txt"; do
  T=$(mktemp -d) && cp -R CMakeLists.txt include lib transform-opt "$T"
  for f in $files; do sed -i '' '/^  MLIR  .*libMLIR.dylib/d' "$T/$f"; done
  cmake -S "$T" -B "$T/build" -G Ninja > /dev/null
  echo "MLIR removed from $files: $(cmake --build "$T/build" 2>&1 | grep -c '^  "') undefined symbols"
  rm -rf "$T"
done
```

Real output **[26]**:

```text
MLIR removed from CMakeLists.txt lib/CMakeLists.txt: 221 undefined symbols
MLIR removed from CMakeLists.txt: 0 undefined symbols
```

- **Without either line**, the link fails with 221 undefined symbols: `MyExtension.cpp` and `transform-opt.cpp` call into MLIR, and nothing provides it.
- **Without the tool's line only**, the tool still links. `MLIR` is in the library's `LINK_LIBS`, and `MyExtensionCh2` is a static archive, so CMake passes the archive's own dependencies on to whatever links it, `PRIVATE` or not. The tool's `MLIR` line is still worth keeping, because `main` calls `MlirOptMain` and the registries directly rather than through the library.
- `libMLIR.dylib` is the whole of MLIR, so these two lines replace the list of static MLIR libraries (`MLIRTransformDialect`, `MLIRFuncDialect`, ...) a build inside the LLVM tree would name; mixing the two would duplicate MLIR's `TypeID` statics (appendix B.1, marker (7)).

**Linking a static library carries its dependencies along, so the library's `LINK_LIBS` is the line that must name `libMLIR.dylib`; the tool's own line only states what `main` uses.**

---

## 7. Key Takeaways & Pitfalls

**Takeaways**

- **New transform ops live in a dialect extension**, not in the Transform dialect: `TransformDialectExtension<T>::init()` registers ops and declares *dependent* dialects (used by the transform ops) separately from *generated* dialects (produced in the payload) (section 2.2).
- **The `.td` file is the single list of the extension's ops**: `GET_OP_LIST` feeds `registerTransformOps<...>` (section 2.4).
- **A transform op is a regular ODS op plus two interfaces**: `apply` does the transformation, `getEffects` declares what happens to handles and payload (section 3).
- **ODS constraints catch schedule mistakes before any payload is touched** (section 3.2).
- **Silenceable vs. definite failure is a promise about the payload**: silenceable only if nothing was modified, so it can be suppressed safely (section 3.3).
- **The declared effects are the op's contract with the interpreter**: "consumes" invalidates every alias of the handle, "only reads" keeps it usable (section 3.4).
- **Custom ops compose with upstream ones through handles**: `transform.loop.outline`'s call handle feeds straight into `transform.my.change_call_target` (section 4.3).
- **Cleanup is a separate pass**, so transform ops can emit straightforward IR (section 4.3).
- **Deriving the code**: the extension API and effect helpers are in two installed headers, `mlir-tblgen` lists the methods to write, and the linker confirms what's still missing; the tool is `mlir-opt`'s installed building blocks plus one registration call, and the build is the generators `mlir-tblgen --help` lists wrapped in the installed CMake functions (section 6).

**Pitfalls**

- **Release vs. Debug against Homebrew MLIR.** `registerTransformOps` calls `checkImplementsTransformOpInterface` only when `NDEBUG` is *not* defined, and Homebrew's `libMLIR.dylib` (a Release build) doesn't contain it. A Debug build of the chapter fails to link (appendix B.1); the standalone guard defaults to Release for this reason. The flip side: the "does the op implement both interfaces" assertion never runs here, so a wrong `getEffects` is not caught at registration.
- **Forgetting `registerMyExtension(registry)`** gives "custom op 'transform.my.change_call_target' is unknown" (section 4.1), the same error stock `mlir-opt` gives.
- **Extra entry-point arguments need bindings**: `@__transform_main` with more than the root argument fails with "operation expects 2 extra value bindings" unless `debug-bind-trailing-args` (or a caller) provides them (section 4.3).
- **`StrAttr` targets are unchecked**: a misspelled callee surfaces later as a `func.call` verifier error on the payload (section 3.3).
- **Result order.** `tile_using_forall` returns (tiled op, loop) and `fuse_into_containing_op` returns (fused op, new loop); keep using the *new* loop handle after each fusion (section 4.3).
- **Don't build the `mlir-doc` target standalone.** `add_mlir_doc` writes under `${MLIR_BINARY_DIR}/docs`, which is empty outside an LLVM build tree, so its output path becomes `/docs/Tutorials/transform/MyExtensionCh2.md`; use `mlir-tblgen -gen-op-doc` instead (appendix A).

---

## Appendix A: Autogenerated Documentation

ODS generates the op's reference documentation from the same record. In an LLVM build tree, `add_mlir_doc` (appendix B.2) does this as part of the `mlir-doc` target; standalone, run the generator directly:

```bash
cd /Users/roy/study/mlir/transform/Ch2
mlir-tblgen -gen-op-doc -I /opt/homebrew/opt/llvm@20/include include/MyExtension.td
```

Real output **[27]**:

~~~text
<!-- Autogenerated by mlir-tblgen; don't manually edit -->
### `transform.my.change_call_target` (transform::ChangeCallTargetOp)

_Changes the callee of a call operation to the specified one_


Syntax:

```
operation ::= `transform.my.change_call_target` $call `,` $new_target attr-dict `:` type($call)
```

For each `func.call` payload operation associated with the handle, changes its 
callee to be the symbol whose name is provided as an attribute to this operation.

Generates a silenceable failure if the operand is associated with payload operations 
that are not `func.call`.
Only reads the operand.

Interfaces: `MemoryEffectOpInterface`, `TransformOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>new_target</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `call` | TransformHandleTypeInterface instance

~~~

Every part maps back to a field of section 3.1: `summary` (italic line), `assemblyFormat` (Syntax), `description` (verbatim), the two interfaces, `StrAttr:$new_target` (Attributes) and `TransformHandleTypeInterface:$call` (Operands). This is why the description should state the op's effect on its operands and its failure modes: it becomes the reference page users read.

---

## Appendix B: What Chapter 2 adds to the build

### B.1 `CMakeLists.txt`: standalone guard and the tool

***CMakeLists.txt***
```cmake
cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Standalone-mode boilerplate.
# Runs only when this chapter is configured directly (cmake -S Ch2).
# ---------------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)                           # (1)
  if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "macOS Deployment Target" FORCE)
  endif()
  # libMLIR.dylib is a Release (NDEBUG) build; MLIR headers declare debug-only
  # helpers under #ifndef NDEBUG, so the chapter must define NDEBUG too.
  if(NOT CMAKE_BUILD_TYPE)                                                       # (2)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  endif()
  project(transform-ch2)

  find_package(MLIR REQUIRED CONFIG)                                             # (3)
  find_package(LLVM REQUIRED CONFIG)

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  include_directories(${MLIR_INCLUDE_DIRS}
                      ${LLVM_INCLUDE_DIRS})

  # add_llvm_example() leaves examples out of `all` unless this is ON.
  set(LLVM_BUILD_EXAMPLES ON)                                                    # (4)
  # The umbrella target the chapter adds itself to (defined by the parent
  # CMakeLists.txt in the superbuild).
  add_custom_target(TransformExample)
endif()

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)                         # (5)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)

add_dependencies(TransformExample transform-opt-ch2)
add_llvm_example(transform-opt-ch2                                               # (6)
  transform-opt/transform-opt.cpp)

target_link_libraries(transform-opt-ch2
  PRIVATE
  MLIR                      # libMLIR.dylib (all dialects, passes, extensions)  # (7)
  MyExtensionCh2
)
```

- **(1)** The guard runs only when the chapter directory is the top of the configure (`cmake -S .` from `Ch2/`). It does what an enclosing LLVM build would otherwise have done.
- **(2)** Default to Release. MLIR's headers compile some calls only `#ifndef NDEBUG`; `TransformDialect::addOperationIfNotRegistered` (used by `registerTransformOps`) calls `detail::checkImplementsTransformOpInterface`, which a Release `libMLIR.dylib` doesn't export. The run below shows what a Debug build does.
- **(3)** Finds Homebrew's MLIR/LLVM CMake packages and their helper modules (`mlir_tablegen`, `add_mlir_library`, `add_llvm_example`, ...).
- **(4)** `add_llvm_example` marks the target `EXCLUDE_FROM_ALL` unless `LLVM_BUILD_EXAMPLES` is on; `TransformExample` is the umbrella target the chapter adds itself to.
- **(5)** The generated `MyExtension.{h,cpp}.inc` land in `build/include`, which is on the include path so `#include "MyExtension.h.inc"` works.
- **(6)** The tool, from `transform-opt/transform-opt.cpp`. The standalone binary is `./build/transform-opt-ch2`.
- **(7)** Link only `libMLIR.dylib` (every dialect, pass and extension `registerAllDialects`/`registerAllExtensions` need) plus the chapter's static library. Mixing `libMLIR.dylib` with MLIR's static `.a` archives would put two copies of MLIR's `TypeID` statics into the process.

A Debug configure of the same sources, in a throw-away build directory:

```bash
cd /Users/roy/study/mlir/transform/Ch2
T=$(mktemp -d) && cmake -S . -B "$T" -G Ninja -DCMAKE_BUILD_TYPE=Debug > /dev/null \
&& cmake --build "$T" 2>&1 | grep -A3 '^Undefined symbols'; rm -rf "$T"
```

Real output **[28]**:

```text
Undefined symbols for architecture arm64:
  "mlir::transform::detail::checkImplementsTransformOpInterface(llvm::StringRef, mlir::MLIRContext*)", referenced from:
      void mlir::transform::TransformDialect::addOperationIfNotRegistered<mlir::transform::ChangeCallTargetOp>() in libMyExtensionCh2.a[2](MyExtension.cpp.o)
ld: symbol(s) not found for architecture arm64
```

**The build type of an out-of-tree project has to match the `NDEBUG` setting of the MLIR it links against**, because MLIR's headers and its library must agree on which debug-only functions exist.

### B.2 `include/CMakeLists.txt`: TableGen rules

***include/CMakeLists.txt***
```cmake
# Tell Tablegen to use MyExtension.td as input.
set(LLVM_TARGET_DEFINITIONS MyExtension.td)

# Ask Tablegen to generate op declarations and definitions from ODS.
mlir_tablegen(MyExtension.h.inc -gen-op-decls)
mlir_tablegen(MyExtension.cpp.inc -gen-op-defs)

# Add a CMakeTarget we can depend on to ensure the generation happens before the compilation.
add_public_tablegen_target(MyExtensionCh2IncGen)

# Don't forget to generate the documentation, this will produce a
# MyExtensionCh2.md under Tutorials/transform
add_mlir_doc(MyExtension MyExtensionCh2 Tutorials/transform/ -gen-op-doc)
```

`mlir_tablegen` adds a rule running `mlir-tblgen <flag>` on `LLVM_TARGET_DEFINITIONS`; `add_public_tablegen_target` groups the rules into `MyExtensionCh2IncGen`. `add_mlir_doc` adds `MyExtensionCh2DocGen` to the `mlir-doc` target — not part of the default build, and not useful standalone (section 7, Pitfalls; appendix A has the direct command).

### B.3 `lib/CMakeLists.txt`: the extension library

***lib/CMakeLists.txt***
```cmake
# Outside examples, this should be `add_mlir_library`.
add_mlir_example_library(
  # Library called MyExtension.
  MyExtensionCh2

  # Built from the following source files.
  MyExtension.cpp

  # Make includes visible without top-level path.
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/examples/transform/Ch2/include

  # Make sure ODS declaration and definitions are generated before compiling this.
  DEPENDS
  MyExtensionCh2IncGen

  # Link in the transform dialect, an all generated dialects.
  LINK_LIBS PRIVATE
  MLIR                      # libMLIR.dylib
)
```

`add_mlir_example_library` is `add_mlir_library` for examples (kept out of `libMLIR` itself). `DEPENDS MyExtensionCh2IncGen` orders TableGen before compiling `MyExtension.cpp`. The `LINK_LIBS` would list `MLIRTransformDialect`, `MLIRFuncDialect` and `MLIRSCFDialect` — the Transform dialect plus the generated dialects of section 2.2 — when linking MLIR's static libraries; against Homebrew's MLIR all three are inside `libMLIR.dylib`. (`ADDITIONAL_HEADER_DIRS` names a path that only exists in an LLVM source tree; it only affects IDE header lists.)

---

## Appendix C: `examples.sh`

`examples.sh` runs the commands of this README straight from the file: every `bash` block in order, except the `cmake` build block of section 5.1 and labeled file excerpts. Each block's output label carries the block's number, e.g. "Real output **[16]**". Build the chapter first (section 5.1).

```text
./examples.sh          # run every command, in README order
./examples.sh 16       # run command [16] only (several numbers are fine)
./examples.sh -l       # list the numbered commands without running them
```

`./examples.sh -l`:

```text
[ 1] 2.4   mlir-tblgen -gen-op-defs -I /opt/homebrew/opt/llvm@20/include include/MyExtension.td \
[ 2] 3.1   mlir-tblgen -gen-op-decls -I /opt/homebrew/opt/llvm@20/include include/MyExtension.td \
[ 3] 3.2   ./build/transform-opt-ch2 --transform-interpreter <<'EOF'
[ 4] 3.2   ./build/transform-opt-ch2 --transform-interpreter <<'EOF'
[ 5] 3.2   ./build/transform-opt-ch2 --transform-interpreter <<'EOF'
[ 6] 3.3   ./build/transform-opt-ch2 ../../test_Example/transform/Ch2/invalid.mlir --transform-interpreter
[ 7] 3.3   ./build/transform-opt-ch2 ../../test_Example/transform/Ch2/invalid.mlir --transform-interpreter \
[ 8] 3.3   ./build/transform-opt-ch2 --transform-interpreter <<'EOF'
[ 9] 3.3   ./build/transform-opt-ch2 --transform-interpreter <<'EOF'
[10] 3.4   ./build/transform-opt-ch2 --transform-interpreter <<'EOF'
[11] 3.4   T=$(mktemp -d) && cp -R CMakeLists.txt include lib transform-opt "$T" \
[12] 4.1   mlir-opt ../../test_Example/transform/Ch2/ops.mlir --transform-interpreter
[13] 4.2   ./build/transform-opt-ch2 ../../test_Example/transform/Ch2/ops.mlir --transform-interpreter
[14] 4.3   ./build/transform-opt-ch2 ../../test_Example/transform/Ch2/sequence.mlir --transform-interpreter
[15] 4.3   ./build/transform-opt-ch2 --help | grep -A4 -- '--transform-interpreter '
[16] 4.3   ./build/transform-opt-ch2 ../../test_Example/transform/Ch2/sequence.mlir \
[17] 4.3   for p in "" ",canonicalize,cse,symbol-dce"; do
[18] 5.2   T=../../test_Example/transform/Ch2
[19] 6.1   grep -n 'void registerTransformOps\|void registerTypes\|void declareDependentDialect\|void declareGeneratedDialect\|^class TransformDialectExtension' \
[20] 6.2   grep -n 'void setCallee(::llvm::StringRef' /opt/homebrew/opt/llvm@20/include/mlir/Dialect/Func/IR/FuncOps.h.inc
[21] 6.4   grep -n '^void \(consumesHandle\|producesHandle\|onlyReadsHandle\|modifiesPayload\|onlyReadsPayload\)' \
[22] 6.5   T=$(mktemp -d) && cp -R CMakeLists.txt include lib transform-opt "$T" \
[23] 6.6   I=/opt/homebrew/opt/llvm@20/include
[24] 6.6   T=$(mktemp -d) && cp -R CMakeLists.txt include lib transform-opt "$T" \
[25] 6.7   mlir-tblgen --help | grep -E -- '--gen-op-(decls|defs|doc) '
[26] 6.8   for files in "CMakeLists.txt lib/CMakeLists.txt" "CMakeLists.txt"; do
[27] A     mlir-tblgen -gen-op-doc -I /opt/homebrew/opt/llvm@20/include include/MyExtension.td
[28] B.1   T=$(mktemp -d) && cmake -S . -B "$T" -G Ninja -DCMAKE_BUILD_TYPE=Debug > /dev/null \
```

Commands [11], [22], [24], [26] and [28] configure and build a temporary copy, so they take a minute each.

---

## Links

- [Transform Ch2: Adding a Simple New Transformation Operation](https://mlir.llvm.org/docs/Tutorials/transform/Ch2/)
- [Transform dialect reference](https://mlir.llvm.org/docs/Dialects/Transform/)
- [Operation Definition Specification (ODS)](https://mlir.llvm.org/docs/DefiningDialects/Operations/)
- [Side effects and the `MemoryEffectsOpInterface`](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/)
