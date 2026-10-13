# Chapter 4: Enabling Generic Transformation with Interfaces

> **Goal:** Teach *core* MLIR passes (the inliner) and a *custom* pass (shape inference) to operate on the Toy dialect without either side hard-coding knowledge of the other — using **interfaces**. Official doc: [Toy Tutorial Chapter 4 — Enabling Generic Transformation with Interfaces](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/)

---

## 1. Overview

Chapters 2 and 3 produced Toy IR in which every user function is generic (`tensor<*xf64>` everywhere) and cleaned it up with Toy-specific rewrite patterns. This chapter makes the IR *shape-specialized* by reusing MLIR's generic machinery rather than writing Toy-only passes. We will:

1. Hook Toy into MLIR's built-in **inliner** through the `DialectInlinerInterface` and the `CallOpInterface`/`CallableOpInterface` op interfaces, adding a `toy.cast` op (with `CastOpInterface`) to bridge type mismatches at call boundaries.
2. **Define a new op interface**, `ShapeInferenceOpInterface`, in TableGen, implement it on the Toy ops that need it, and write a generic `ShapeInferencePass` against it.
3. Assemble the `-opt` pipeline: inline → shape inference → canonicalize → CSE (common subexpression elimination: merge ops that compute the same value).
4. Show how to derive this code and its build wiring yourself from MLIR's own headers and tools (section 6).

The lexer, parser, AST, and the Chapter 3 DRR (declarative rewrite rule) and C++ rewrite patterns (`mlir/ToyCombine.td`, `mlir/ToyCombine.cpp`) are carried over. New or changed: `mlir/MLIRGen.cpp` (private functions, section 4.1); `mlir/Dialect.cpp` (the inliner interface, `toy.cast`, the call conversion and `inferShapes()`: sections 4.1, 4.4, 4.5, 5.2); `include/toy/Ops.td` (sections 4.2, 4.4, 5.2); `include/toy/ShapeInferenceInterface.td` and `.h` (section 5.1); `mlir/ShapeInferencePass.cpp` and `include/toy/Passes.h` (section 5.3); `toyc.cpp` (the pipeline, sections 4.3 and 5.4).

The appendices hold the extras: the `-opt` pipeline pass by pass (A), more inputs (B), what Chapter 4 adds to the build (C), and the script that runs every command in this README (D).

Building the chapter is covered once in the top-level [README](../README.md#build--run). Every command here is numbered by its output (**[1]**–**[32]**); after a standalone build, `./examples.sh` runs them all, `./examples.sh 2` runs only command [2], and `./examples.sh -l` lists them (appendix D).

---

## 2. Background: Grappling with an Extensible IR

Through dialects, MLIR can represent many different levels of abstraction; the Toy dialect is one of them. Different as those abstractions are, there is a common set of transformations and analyses we want to run on all of them. Implementing each transformation separately for each dialect would duplicate large amounts of code, because the internal algorithms are generally the same. What we want is for transformations to hook *opaquely* into a dialect like Toy to get the information they need.

MLIR does have some always-available hooks for core transformations — Chapter 3 registered canonicalization patterns through the `getCanonicalizationPatterns` hook on our ops (`let hasCanonicalizer = 1;`). But a hook built into every op only covers transformations that core MLIR anticipated. Anything else would mean each dialect writing its own inliner, constant folder, CSE, DCE (dead code elimination) and so on: an O(dialects × transformations) explosion. MLIR's more general answer is **interfaces**: a transformation is written once, generically, against an abstract interface; each dialect/operation *opts in* by implementing that interface. The pass never needs to know the dialect exists, and the dialect never needs to know how the pass works internally.

MLIR has two granularities of interface, and Chapter 4 uses both:

| Kind | Attached to | Used here for |
|---|---|---|
| **Dialect interface** | the whole dialect (`DialectInlinerInterface`) | answering dialect-wide questions: "may ops from this dialect be inlined?", "how do I handle your terminator?", "how do I materialize a type conversion?" |
| **Operation interface** | individual ops (`CallOpInterface`, `CallableOpInterface`, `CastOpInterface`, our own `ShapeInferenceOpInterface`) | letting a pass query/manipulate a specific op opaquely: "what do you call?", "where is your body?", "infer your result shape" |

The chapter demonstrates both directions:

- **Consuming existing interfaces**: hooking Toy into MLIR's built-in **inliner** pass via `DialectInlinerInterface` + `CallOpInterface`/`CallableOpInterface` + `CastOpInterface` (section 4).
- **Defining a new interface**: declaring `ShapeInferenceOpInterface` in ODS (Operation Definition Specification) and TableGen, and writing a generic `ShapeInferencePass` that works on *any* op implementing it — Toy ops today, anyone else's ops tomorrow (section 5).

---

## 3. Shape Inference: Preparing for Code Generation

**Shape inference** works out the concrete shape of every tensor value from the shapes of its operands, and rewrites its type accordingly. For example, transposing a `tensor<2x3xf64>` gives a `tensor<3x2xf64>`, and multiplying two `tensor<3x2xf64>` values element-wise gives a `tensor<3x2xf64>`. Starting from the shapes that are known, the constants, it fills in the rest, until no value is left with the unknown shape `tensor<*xf64>`. Code generation needs this: a loop over a tensor needs to know its size (Chapter 5). Section 5 implements it; this section explains what stands in the way.

Toy is intentionally "dumb" at the source level: user functions are **generic**. The chapter's working example, `codegen.toy` (abridged: its `# CHECK:` lines are elided):

***test_Example/Toy/Ch4/codegen.toy***
```text
# RUN: toyc-ch4 %s -emit=mlir 2>&1 | FileCheck %s

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
...
```

`multiply_transpose` says nothing about the shapes of `a` and `b`. When Chapter 2's MLIRGen lowers this to the Toy dialect, every argument and intermediate value gets the *unranked* tensor type `tensor<*xf64>` — "an f64 tensor of unknown rank and shape". Only `main`, where literals like `var a<2, 3> = ...` appear, has concrete `tensor<2x3xf64>` values; outside of constant initialization we don't know any shapes.

**What to watch for.** `a` and `b` hold the *same six values* shaped `<2,3>` (one from a nested literal, one reshaped from a flat literal), `c` is computed but **never used**, and only `d` is printed. Each of these details shows up in what the optimizer does to this file (section 5.4).

The driver's arguments, used throughout this README:

- `../../test_Example/Toy/Ch4/codegen.toy` — the positional input file (shown above), relative to `Ch4/`. `-` reads from stdin (sections 4.1, 4.4 and appendix B).
- `-emit=mlir` — dump the module after MLIRGen (and after the pipeline, if `-opt` is given). `-emit=ast` from Chapter 1 still works.
- `-opt` — run the pass pipeline: inliner, then shape inference, canonicalizer and CSE (common subexpression elimination) on each `toy.func` (section 5.4). Without it you get the raw codegen output.
- `--mlir-print-ir-after-all` — a generic pass-manager flag that dumps the IR after every pass (section 4.1, appendix A). It exists because `main()` calls `mlir::registerPassManagerCLOptions()` and the pipeline applies it via `applyPassManagerCLOptions(pm)`.
- An input ending in `.mlir` (or any input with `-x mlir`) is parsed as MLIR instead of Toy source; `shape_inference.mlir` in section 4.1 and appendix B uses this.
- `2>&1` — `module->dump()` and the pass-manager dumps write to **stderr**.

The raw codegen, without `-opt`:

```bash
cd /Users/roy/study/mlir/toy/Ch4
./build/toyc-ch4 ../../test_Example/Toy/Ch4/codegen.toy -emit=mlir 2>&1
```

Real output **[1]**:

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

- `@multiply_transpose` is fully **generic**: every type inside it is `tensor<*xf64>`, while both `toy.generic_call`s pass `tensor<2x3xf64>` operands.
- The calls return **unranked** `tensor<*xf64>`: the shape information stops at the call boundary.
- The callee is **`private`**, which matters for inlining (section 4.1).
- No pass has run: the dead `%4` (variable `c`) and the redundant reshapes are still there.

These unknown shapes complicate optimization, and we cannot generate efficient code (or, in Chapter 5, lower to affine loops) without knowing the actual shapes. Within one function, shapes can simply be propagated through the computation until all of them are known. The hard part is calls to user-defined generic functions: every call site may deduce different shapes. There are three options:

1. **Symbolic inference** based on the argument types — hard to generalize once the language gains more control flow.
2. **Function specialization** — clone the callee for every call site with new argument shapes and specialize the clone (what a production compiler might do).
3. **Inline everything, then propagate intraprocedurally** — the approach Toy takes:
   - **Inlining** (section 4) pulls the bodies of the generic functions into `main`, so that shape information from the call sites can flow into the callee's operations;
   - **intraprocedural shape inference** (section 5) propagates the known static shapes through the now-flat sequence of operations, replacing every `tensor<*xf64>` with a ranked type.

---

## 4. Inlining

We could write an inliner specifically for Toy, but even disregarding cost modeling, the pure structural transformation is complex to implement from scratch. MLIR instead ships a generic inliner pass (`mlir::createInlinerPass()`) that dialects plug into. It knows nothing about Toy; to make it work on our IR we answer four questions through interfaces (section 6 shows how to find them in MLIR's own headers):

1. *Policy*: which Toy ops/regions are legal to inline? → `DialectInlinerInterface::isLegalToInline` (section 4.1)
2. *Mechanics*: what happens to `toy.return` when a body is spliced into the caller? → `handleTerminator` (section 4.1)
3. *Discovery*: which ops are calls, and which ops are callable? → `CallOpInterface` on `toy.generic_call`, `CallableOpInterface` (via `FunctionOpInterface`) on `toy.func` (section 4.2)
4. *Type mismatches*: call sites pass `tensor<2x3xf64>` but the callee's block arguments are `tensor<*xf64>` — who bridges that? → a new `toy.cast` op and the `materializeCallConversion` hook (sections 4.3–4.5)

### 4.1 `ToyInlinerInterface` — the dialect interface

The constraints on inlining Toy operations are provided through a **dialect interface**: a class with a set of virtual hooks the dialect can override, here `DialectInlinerInterface` (from `mlir/Transforms/InliningUtils.h`). The analysis and terminator hooks at the top of `Dialect.cpp` (abridged; the conversion hook at the end of the struct is shown in section 4.5):

***mlir/Dialect.cpp***
```cpp
#include "mlir/Transforms/InliningUtils.h"
...
/// This class defines the interface for handling inlining with Toy
/// operations.
struct ToyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// All call operations within toy can be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,                    // (1)
                       bool wouldBeCloned) const final {
    return true;
  }

  /// All operations within toy can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {  // (2)
    return true;
  }

  // All functions within toy can be inlined.
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {     // (3)
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator(toy.return) by replacing it with a new
  /// operation as necessary.
  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {   // (4)
    // Only "toy.return" needs to be handled here.
    auto returnOp = cast<ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());                  // (5)
  }
  ...
};
```

What each hook means:

- **Three `isLegalToInline` overloads**, one per level the inliner asks about (the numbers match the `// (N)` markers in the code):
  - **(1)** *(call, callable, wouldBeCloned)*: may this particular call to this particular callable be inlined at all? `wouldBeCloned` is `true` if the callee body would be *copied* (other uses remain) rather than moved.
  - **(2)** *(Operation, Region, ...)*: may this specific operation be moved into the destination region? The `IRMapping` maps callee values to their caller-side replacements.
  - **(3)** *(Region, Region, ...)*: may the source region as a whole (the callee's body) be inlined into the destination region?

  Splicing a Toy function body into its caller keeps every op's meaning and order, side effects included (`toy.print`), so all three return `true` unconditionally. A real dialect would inspect the operands/attributes here (e.g. refuse to inline ops that depend on function-scoped state).

- **(4), (5) `handleTerminator`** — after the callee's block is spliced into the caller, the callee's terminator (such as `toy.return %2` in the example below) must disappear; whatever it returned must become the value(s) that the original `toy.generic_call` produced. `valuesToRepl` (a `ValueRange`) are exactly those call results, so each one has all its uses replaced with the corresponding return operand (RAUW, replace-all-uses-with). The inliner then erases the terminator. (This is the single-block overload; a second overload taking a `Block *newDest` exists for callees with several blocks, which Toy never produces.)

**Private visibility for non-`main` functions.** The inliner only discards unused function definitions whose symbol visibility is **private**; a public symbol might be referenced from outside the module. So `mlir/MLIRGen.cpp` (the only change to that file relative to Chapter 3) marks every function except `main` private:

***mlir/MLIRGen.cpp***
```cpp
    // If this function isn't main, then set the visibility to private.
    if (funcAST.getProto()->getName() != "main")
      function.setPrivate();
```

This is why the raw codegen prints `toy.func private @multiply_transpose(...)` (section 3). The test input `shape_inference.mlir` is that raw codegen of `codegen.toy`, written out as MLIR:

***test_Example/Toy/Ch4/shape_inference.mlir***
```mlir
...
toy.func private @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
  %2 = toy.mul %0, %1 : tensor<*xf64>
  toy.return %2 : tensor<*xf64>
}
toy.func @main() {
  ...
}
...
```

Run the `-opt` pipeline on it twice, as generated and with `private` removed by `sed`, and look at the IR right after the inliner. `-x mlir` makes `toyc-ch4` parse its input as MLIR (`-` is standard input), and `--mlir-print-ir-after-all` prints the IR after each pass:

```bash
cd /Users/roy/study/mlir/toy/Ch4
echo "== as generated: multiply_transpose is private"
./build/toyc-ch4 ../../test_Example/Toy/Ch4/shape_inference.mlir -x mlir -emit=mlir -opt --mlir-print-ir-after-all 2>&1 \
  | sed -n '/IR Dump After Inliner/,/^}/p' | grep -E 'toy.func|generic_call'
echo "== with private removed"
sed 's/toy.func private/toy.func/' ../../test_Example/Toy/Ch4/shape_inference.mlir \
  | ./build/toyc-ch4 - -x mlir -emit=mlir -opt --mlir-print-ir-after-all 2>&1 \
  | sed -n '/IR Dump After Inliner/,/^}/p' | grep -E 'toy.func|generic_call'
```

Real output **[2]**:

```text
== as generated: multiply_transpose is private
  toy.func @main() {
== with private removed
  toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
  toy.func @main() {
```

- **Both calls were inlined either way**: no `toy.generic_call` is left in `main`.
- **Only the private function was deleted.** The public `@multiply_transpose` stays, although nothing calls it any more: the inliner can't rule out a caller outside the module. Being still generic, it later makes shape inference fail (section 5.3).

Generic passes act only on what the IR states: **visibility, not the absence of callers, decides whether a function may be deleted.**

**Registering the interface on the dialect.** The dialect interface is registered in `ToyDialect::initialize()`, next to the ops, right below the struct in `Dialect.cpp`:

***mlir/Dialect.cpp***
```cpp
/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
      >();
  addInterfaces<ToyInlinerInterface>();
}
```

`addInterfaces<ToyInlinerInterface>()` is what attaches the interface to the dialect. Without it the inliner finds no inliner interface for Toy ops, so it gets no permission to inline them and leaves every call alone, as when the interface's answers are missing in section 4.3.

### 4.2 Marking calls and callables — `CallOpInterface` / `CallableOpInterface`

The inliner also needs to know that `toy.generic_call` represents a call and `toy.func` represents a function, so it can build the call graph. That information is specific to single operations, so it comes from two **operation interfaces** in `mlir/Interfaces/CallInterfaces.td`:

- `CallableOpInterface` — "I am a thing that can be called; here is my region and my signature."
- `CallOpInterface` — "I am a call; here is who I call and with what arguments."

In `include/toy/Ops.td` the interface definitions are included and the interfaces attached declaratively (abridged; the `description` bodies are elided):

***include/toy/Ops.td***
```tablegen
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"             // (1)
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "toy/ShapeInferenceInterface.td"
...
def FuncOp : Toy_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove              // (2)
  ]> {
  ...
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,             // (3)
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);
  ...
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }  // (4)
  }];
  ...
}
...
def GenericCallOp : Toy_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {     // (5)
  ...
  // The generic call operation takes a symbol reference attribute as the
  // callee, and inputs for the call.
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<F64Tensor>:$inputs,
    OptionalAttr<DictArrayAttr>:$arg_attrs,             // (6)
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  // The generic call operation returns a single value of TensorType.
  let results = (outs F64Tensor);
  ...
}
```

Notes on the ODS side; the numbers match the `// (N)` markers:

- **(1) `include "mlir/Interfaces/CallInterfaces.td"`** brings in the ODS definitions of both `CallOpInterface` and `CallableOpInterface`. An interface has to be included before an op's trait list can name it; the other includes do the same for the function, symbol, cast, side-effect and shape-inference interfaces.
- **(2), (4) `toy.func` gets `CallableOpInterface` through `FunctionOpInterface`**, which implies it together with `Symbol` and brings the function parse/print helpers `FuncOp` already uses; its callable methods are written inline in `FuncOp`'s `extraClassDeclaration`: `getCallableRegion()` returns the region the inliner should clone from — the function body — and `getArgumentTypes()`/`getResultTypes()` expose the signature.
- **(3), (6) `arg_attrs` / `res_attrs`** are optional per-argument and per-result attributes. `FunctionOpInterface` expects them on `FuncOp`; on `GenericCallOp` nothing in this chapter reads them.
- **(5) `DeclareOpInterfaceMethods<CallOpInterface>`** is the key ODS construct: it attaches the interface **and** declares its methods that have no default implementation on the generated C++ op class, leaving their bodies for us to write in `Dialect.cpp`. (Listing the bare interface instead would attach it without declaring those methods.)

The `CallOpInterface` implementations for `GenericCallOp` in `Dialect.cpp`:

***mlir/Dialect.cpp***
```cpp
/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable GenericCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// Set the callee for the generic call operation, this is required by the call
/// interface.
void GenericCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range GenericCallOp::getArgOperands() { return getInputs(); }

/// Get the argument operands to the called function as a mutable range, this is
/// required by the call interface.
MutableOperandRange GenericCallOp::getArgOperandsMutable() {
  return getInputsMutable();
}
```

- `getCallableForCallee()` returns a `CallInterfaceCallable` — either an SSA (static single assignment) value (indirect call) or, as here, a `SymbolRefAttr` naming the callee. The inliner resolves the symbol to the `toy.func` in the module's symbol table (the interface's default `resolveCallable` does this).
- `setCalleeFromCallable()` lets a transformation retarget the call, for example a pass that replaces the callee with a specialized copy.
- `getArgOperands()` / `getArgOperandsMutable()` tell the inliner which operands map to the callee's block arguments.

`getArgOperandsMutable()` has no default implementation in `CallOpInterface`, so it must be defined; the op class doesn't provide `getAttrOfType`, so `getCallableForCallee()` goes through the `Operation *` (`(*this)->`).

### 4.3 Adding the inliner pass — and the hidden type conversion

With policy, discovery, and registration in place, enabling inlining is one line in `toyc.cpp`:

***toyc.cpp***
```cpp
    // Inline all functions into main and then delete them.
    pm.addPass(mlir::createInlinerPass());
```

The inliner is a *module*-level pass: it builds the call graph from `CallOpInterface`/`CallableOpInterface`, inlines bottom-up, runs a simplification pipeline on the functions it visits, and erases now-unreferenced private functions. Section 5.4 shows the rest of the pipeline around it.

**The working example.** The running example is `codegen.toy`; section 3 shows the file and its raw codegen. Its `main` calls `multiply_transpose` twice and never uses `c`; the calls pass `tensor<2x3xf64>` operands to a callee whose arguments are `tensor<*xf64>`, and the callee is `private` (section 4.1).

**The hidden type conversion.** With only the pieces so far in place, the calls are not inlined. The missing piece is a hidden type conversion on the edge of the call. The `toy.generic_call` operands are `tensor<2x3xf64>`, while the callee's arguments expect `tensor<*xf64>`. If the inliner blindly wired caller SSA values into the callee body, operand types would silently change and the IR might not even verify. Instead the inliner expects the dialect to materialize an **explicit cast** for each mismatched argument or result (the `materializeCallConversion` hook, section 4.5); if it can't get one, it declines to inline that call.

To see it, build a scratch copy of the chapter in which the hook can't make a cast: the command copies the sources to a temporary directory, uses `sed` to change the hook's `return builder.create<CastOp>(...)` line (section 4.5 shows the hook) into `return nullptr;`, builds the copy (about 10 seconds), and runs it with `-opt`. It prints the exit status, the IR right after the inliner, and the errors:

```bash
cd /Users/roy/study/mlir/toy/Ch4
S=$(mktemp -d)
cp -R CMakeLists.txt include mlir parser toyc.cpp "$S"/
sed -i '' 's/return builder.create<CastOp>(conversionLoc, resultType, input);/return nullptr;/' "$S"/mlir/Dialect.cpp
cmake -S "$S" -B "$S"/build -G Ninja >/dev/null && cmake --build "$S"/build >/dev/null
"$S"/build/toyc-ch4 ../../test_Example/Toy/Ch4/codegen.toy -emit=mlir -opt --mlir-print-ir-after-all > "$S"/out.txt 2>&1; echo "toyc-ch4 exit status: $?"
sed -n '/After Inliner/,/^}/p' "$S"/out.txt
grep 'error:' "$S"/out.txt
rm -rf "$S"
```

Real output **[3]**:

```text
toyc-ch4 exit status: 4
// -----// IR Dump After Inliner (inline) //----- //
module {
  toy.func private @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
    %2 = toy.mul %0, %1 : tensor<*xf64>
    toy.return %2 : tensor<*xf64>
  }
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %2 = toy.generic_call @multiply_transpose(%0, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    %3 = toy.generic_call @multiply_transpose(%1, %0) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    toy.print %3 : tensor<*xf64>
    toy.return
  }
}
../../test_Example/Toy/Ch4/codegen.toy:4:1: error: Shape inference failed, 3 operations couldn't be inferred
../../test_Example/Toy/Ch4/codegen.toy:11:11: error: unable to infer shape of operation without shape inference interface
```

- Both `toy.generic_call`s and the callee `@multiply_transpose` are still there: without a cast, the inliner left every call alone.
- The reshapes are gone even so. The inliner runs the canonicalizer on every function it visits, as part of its *simplification pipeline*, whether or not it inlines anything; the Chapter 3 reshape patterns removed them.
- The pipeline then fails in the shape-inference pass (exit status 4) with the two errors at the end: the generic function's three ops never get shapes, and a `toy.generic_call` has no shape-inference rule. Section 5.3 explains both.

### 4.4 `toy.cast` and `CastOpInterface`

To represent such casts between two different shapes, the chapter adds a new operation, `toy.cast`, in `Ops.td`:

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

What each trait and interface gives `toy.cast`:

- **`CastOpInterface`** marks the op as a cast-like op and provides generic utilities for it: a verifier (`impl::verifyCastInterfaceOp`) and a fold hook that removes identity casts (`impl::foldCastInterfaceOp`), both wired in by `mlir/Interfaces/CastInterfaces.td` (included at the top of `Ops.td`, section 4.2). We hook into the interface by defining its one required method, `areCastCompatible` (below).
- **`Pure`** — no side effects; dead casts can be removed.
- **`SameOperandsAndResultShape`** — if both sides *are* ranked, shapes must agree.
- It also declares **`ShapeInferenceOpInterface`**: after inlining, the casts are exactly the ops through which static shapes must flow (section 5.2).

#### `areCastCompatible` decides which casts verify

The one method `CastOpInterface` requires, in `Dialect.cpp`:

***mlir/Dialect.cpp***
```cpp
/// Returns true if the given set of input and result types are compatible with
/// this cast operation. This is required by the `CastOpInterface` to verify
/// this operation and provide other additional utilities.
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

`toy.cast` can be written by hand and fed to `toyc-ch4` on stdin (`-` plus `-x mlir`). Without `-opt` the tool only parses, verifies, and prints. Rank erasure (`tensor<2x3xf64>` to `tensor<*xf64>`) and rank refinement (the reverse) both verify:

```bash
cd /Users/roy/study/mlir/toy/Ch4
cat <<'EOF' | ./build/toyc-ch4 - -x mlir -emit=mlir 2>&1
toy.func @main() {
  %0 = toy.constant dense<1.0> : tensor<2x3xf64>
  %1 = toy.cast %0 : tensor<2x3xf64> to tensor<*xf64>
  %2 = toy.cast %1 : tensor<*xf64> to tensor<2x3xf64>
  toy.print %2 : tensor<2x3xf64>
  toy.return
}
EOF
```

Real output **[4]**:

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<1.000000e+00> : tensor<2x3xf64>
    %1 = toy.cast %0 : tensor<2x3xf64> to tensor<*xf64>
    %2 = toy.cast %1 : tensor<*xf64> to tensor<2x3xf64>
    toy.print %2 : tensor<2x3xf64>
    toy.return
  }
}
```

A cast between two different ranked shapes fails the last line of `areCastCompatible` (`input == output`), so the verifier rejects it while parsing:

```bash
cd /Users/roy/study/mlir/toy/Ch4
cat <<'EOF' | ./build/toyc-ch4 - -x mlir -emit=mlir 2>&1
toy.func @main() {
  %0 = toy.constant dense<1.0> : tensor<2x3xf64>
  %1 = toy.cast %0 : tensor<2x3xf64> to tensor<3x2xf64>
  toy.print %1 : tensor<3x2xf64>
  toy.return
}
EOF
```

Real output **[5]** (exit status 3):

```text
<stdin>:3:8: error: 'toy.cast' op operand type 'tensor<2x3xf64>' and result type 'tensor<3x2xf64>' are cast incompatible
  %1 = toy.cast %0 : tensor<2x3xf64> to tensor<3x2xf64>
       ^
<stdin>:3:8: note: see current operation: %1 = "toy.cast"(%0) : (tensor<2x3xf64>) -> tensor<3x2xf64>
Error can't load file -
```

- In the first run, one side of each cast is unranked, so `!input.hasRank() || !output.hasRank()` is true and both casts verify. These are the two directions the inliner and shape inference need.
- The error text comes from the interface's generic verifier (`impl::verifyCastInterfaceOp`), not from code in this repo; `areCastCompatible` only answers yes or no.
- Exit status 3 is `loadMLIR()` failing to parse the input, before any pass runs.

### 4.5 `materializeCallConversion` — inserting the casts

With a proper cast op, the last `ToyInlinerInterface` hook inserts it when necessary (the end of the struct from section 4.1, abridged):

***mlir/Dialect.cpp***
```cpp
struct ToyInlinerInterface : public DialectInlinerInterface {
  ...
  /// Attempts to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    return builder.create<CastOp>(conversionLoc, resultType, input);
  }
};
```

The hook must build an op that takes `input` as its only operand and produces one result of `resultType`, or return `nullptr` if no conversion is possible (the default in `DialectInlinerInterface` returns `nullptr`, which is the "nothing changes" case of section 4.3).

With the hook in place, both calls are inlined. `-opt` runs the whole pipeline, so to see the IR right after the inliner, print the IR after every pass (`--mlir-print-ir-after-all`, appendix A) and keep only the `Inliner (inline)` dump with `sed`:

```bash
cd /Users/roy/study/mlir/toy/Ch4
./build/toyc-ch4 ../../test_Example/Toy/Ch4/codegen.toy -emit=mlir -opt --mlir-print-ir-after-all 2>&1 \
  | sed -n '/After Inliner/,/^}/p'
```

Real output **[6]**:

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

- **One `toy.cast` per inlined argument**: `%2` and `%3` are the casts `materializeCallConversion` built, from the caller's `tensor<2x3xf64>` to the callee's `tensor<*xf64>`. The inlined body (`%4`–`%6`) still computes on `tensor<*xf64>`; making it ranked is section 5's job.
- **`@multiply_transpose` is gone**: it was private and has no uses left (section 4.1).
- **The inliner also simplifies**, so the result is cleaner than a literal splice: its simplification pipeline (section 4.3) is set by its `default-pipeline` option, `"canonicalize"` by default (`mlir/Transforms/Passes.td`). That is why the two reshapes are already gone, `b`'s flat constant has become a second `tensor<2x3xf64>` constant (`%1`), and the ops of the unused call `c = multiply_transpose(a, b)` have been erased. Appendix A shows those nested canonicalizer runs.

Compare this run with section 4.3's scratch build: the same generic inliner, given a cast here and refused one there. **A generic pass never guesses a dialect's semantics: it asks through an interface, and if the interface can't answer, it leaves the IR alone.** That is how interfaces make the infrastructure as extensible as the representation.

---

## 5. Intraprocedural Shape Inference

After inlining we have a single `main` containing a mix of statically and dynamically shaped operations. Now we propagate shapes within that one function. We could write a pass that directly encodes the constraints of each Toy op, but a good rule of thumb is to express a transformation as generically as possible, so other dialects can reuse it. At its core, the problem is: given statically known input types, tell me the output types. That property belongs to each individual operation, so we define a **new operation interface** that ops needing shape inference implement, and the pass stays generic.

### 5.1 Defining the interface in ODS

Like operations, operation interfaces can be defined with ODS. The interface inherits from `OpInterface`, which takes the name of the generated C++ class as its template argument; it gets a description and a list of methods:

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

Its parts:

- `OpInterface<"ShapeInference">` — the TableGen def name (`ShapeInferenceOpInterface`) is what you reference in other `.td` files; the string `"ShapeInference"` is the name of the **generated C++ class** (`mlir::toy::ShapeInference`) that passes will `dyn_cast` to.
- `InterfaceMethod<description, returnType, methodName>` — declares one method, `void inferShapes()`, from a description, a C++ return type as a string, and a method name as a string. `InterfaceMethod` can also take an argument list, a method body, and a default implementation; here the simplest form suffices.

`mlir-tblgen` turns this into two generated files (appendix C has the build rules). The declarations, `-gen-op-interface-decls` (`ShapeInferenceOpInterfaces.h.inc`):

```bash
cd /Users/roy/study/mlir/toy/Ch4
mlir-tblgen -gen-op-interface-decls include/toy/ShapeInferenceInterface.td -I include -I /opt/homebrew/opt/llvm@20/include | sed -n '10,$p'
```

Real output **[7]** (with `// (N)` markers added for the list below):

```cpp
namespace detail {
struct ShapeInferenceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    void (*inferShapes)(const Concept *impl, ::mlir::Operation *);                                          // (1)
  };
  template<typename ConcreteOp>
  class Model : public Concept {                                                                            // (2)
  public:
    using Interface = ShapeInference;
    Model() : Concept{inferShapes} {}

    static inline void inferShapes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {                                                                    // (3)
  public:
    using Interface = ShapeInference;
    FallbackModel() : Concept{inferShapes} {}

    static inline void inferShapes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct ShapeInferenceTrait;

} // namespace detail
class ShapeInference : public ::mlir::OpInterface<ShapeInference, detail::ShapeInferenceInterfaceTraits> {  // (4)
public:
  using ::mlir::OpInterface<ShapeInference, detail::ShapeInferenceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ShapeInferenceTrait<ConcreteOp> {};
  /// Infer and set the output shape for the current operation.
  void inferShapes();                                                                                       // (5)
};
namespace detail {
  template <typename ConcreteOp>
  struct ShapeInferenceTrait : public ::mlir::OpInterface<ShapeInference, detail::ShapeInferenceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
template<typename ConcreteOp>
void detail::ShapeInferenceInterfaceTraits::Model<ConcreteOp>::inferShapes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).inferShapes();                                       // (6)
}
template<typename ConcreteOp>
void detail::ShapeInferenceInterfaceTraits::FallbackModel<ConcreteOp>::inferShapes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->inferShapes(tablegen_opaque_val);
}
```

The interface is *type-erased*: a pass holds a `ShapeInference`, never a `TransposeOp`, yet the call reaches `TransposeOp::inferShapes()`. The pieces, by marker:

- **(1) `Concept`** is a table of function pointers, one per interface method; here a single `inferShapes` entry taking the op as a plain `Operation *`.
- **(2) `Model<ConcreteOp>`** fills that table for one op class. MLIR builds one per op class that declares the interface (section 5.2) and stores it with the op's registration.
- **(6)** Each `Model` entry casts the `Operation *` back to the concrete op and calls its `inferShapes()`: the code each op writes in section 5.2.
- **(3) `FallbackModel`** and `ExternalModel` are for attaching the interface to an op from outside its definition, for example to an op of another dialect; Toy doesn't use them.
- **(4) `ShapeInference`** is the class passes use. It wraps an `Operation *`, so `dyn_cast<ShapeInference>(op)` works on any op and returns null when the op's class has no model.
- **(5) `inferShapes()`** is declared here and defined in the second generated file, `-gen-op-interface-defs` (`ShapeInferenceOpInterfaces.cpp.inc`):

```bash
cd /Users/roy/study/mlir/toy/Ch4
mlir-tblgen -gen-op-interface-defs include/toy/ShapeInferenceInterface.td -I include -I /opt/homebrew/opt/llvm@20/include | sed -n '10,$p'
```

Real output **[8]**:

```cpp
void ShapeInference::inferShapes() {
      return getImpl()->inferShapes(getImpl(), getOperation());
  }
```

It looks up the op's model (`getImpl()`) and calls the model's function pointer (1). This one dispatch is why an interface can be attached to ops from unrelated dialects without a common C++ base class.

The two generated files are pulled into the build by two hand-written files. The declarations go into a small header, which `Dialect.h` includes:

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

and the `.cpp.inc` is included once in `ShapeInferencePass.cpp` (section 5.3).

### 5.2 Attaching the interface and implementing `inferShapes()`

The interface is added to ops the same way `CallOpInterface` was added to `GenericCallOp`, with `DeclareOpInterfaceMethods` (section 4.2). In `Ops.td`, every op whose result shape can be computed from its operand shapes opts in (abridged; the interface `.td` is pulled in by the `include "toy/ShapeInferenceInterface.td"` shown in section 4.2):

***include/toy/Ops.td***
```tablegen
def AddOp : Toy_Op<"add",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  ...
}
...
def CastOp : Toy_Op<"cast", [
     DeclareOpInterfaceMethods<CastOpInterface>,
     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
     Pure,
     SameOperandsAndResultShape
  ]> {
  ...
}
...
def MulOp : Toy_Op<"mul",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  ...
}
...
def TransposeOp : Toy_Op<"transpose",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  ...
}
```

`DeclareOpInterfaceMethods<ShapeInferenceOpInterface>` adds a `void inferShapes();` declaration to each generated op class that lists it. The generated classes that have one:

```bash
cd /Users/roy/study/mlir/toy/Ch4
mlir-tblgen -gen-op-decls include/toy/Ops.td -I include -I /opt/homebrew/opt/llvm@20/include \
  | grep -E '^class [A-Za-z]+ : |void inferShapes\(\);' | grep -B1 'void inferShapes' | grep -o '^class [A-Za-z]*'
```

Real output **[9]**:

```text
class AddOp
class CastOp
class MulOp
class TransposeOp
```

The first `grep` keeps each class header and each `inferShapes` declaration, and the second keeps the header right before each declaration. We provide the four definitions in `Dialect.cpp`. Ops that don't need inference don't participate: `toy.constant` and `toy.reshape` already produce statically shaped results by construction (`StaticShapeTensorOf<[F64]>` for reshape), `toy.print`/`toy.return` have no results, and `toy.generic_call` no longer exists after inlining.

**Implementing `inferShapes()` per op.** Each of these ops now needs a definition of `inferShapes()`. All four are small, and each *refines the result type in place* from the (already-ranked) operand types. They are spread over the per-op sections of `Dialect.cpp` (abridged):

***mlir/Dialect.cpp***
```cpp
/// Infer the output shape of the AddOp, this is required by the shape inference
/// interface.
void AddOp::inferShapes() { getResult().setType(getLhs().getType()); }
...
/// Infer the output shape of the CastOp, this is required by the shape
/// inference interface.
void CastOp::inferShapes() { getResult().setType(getInput().getType()); }
...
/// Infer the output shape of the MulOp, this is required by the shape inference
/// interface.
void MulOp::inferShapes() { getResult().setType(getLhs().getType()); }
...
void TransposeOp::inferShapes() {
  auto arrayTy = llvm::cast<RankedTensorType>(getOperand().getType());
  SmallVector<int64_t, 2> dims(llvm::reverse(arrayTy.getShape()));
  getResult().setType(RankedTensorType::get(dims, arrayTy.getElementType()));
}
```

How the two pieces fit together:

- `AddOp` and `MulOp` are element-wise, so the result takes the (left) operand's type; `TransposeOp` reverses the input dimensions.
- `CastOp::inferShapes()` sets the result type equal to the input type — turning `toy.cast %1 : tensor<2x3xf64> to tensor<*xf64>` into the identity cast `tensor<2x3xf64> to tensor<2x3xf64>`. That is what makes the cast *foldable* later: the `CastOpInterface` fold hook (section 4.4) removes same-type casts when the canonicalizer runs. Section 5.3 shows the identity casts after the pass, and appendix A the `Canonicalizer` dump that follows, with the casts gone.
- `TransposeOp::inferShapes()` may safely `cast<RankedTensorType>` because the pass only calls `inferShapes()` once **all** operands are ranked (see the `allOperandsInferred` gate in section 5.3).
- **Changing result types in place is safe here.** Every user of these values either infers its own shape or accepts any `F64Tensor` (`toy.print`).
- **`toy.return` tolerates the change too.** `ReturnOp::verify()` in `Dialect.cpp` accepts an operand that is ranked where the function's result type is unranked, or the other way round.

### 5.3 The `ShapeInferencePass` — a generic worklist algorithm

The pass operates on functions: it runs on each `toy.func` in isolation. MLIR also supports general `OperationPass`es that run on any isolated operation, but our module only contains functions, so there is no need to generalize. The pass is therefore a class deriving from `PassWrapper<..., OperationPass<toy::FuncOp>>` that overrides `runOnOperation()`. `ShapeInferencePass.cpp` includes the generated interface definitions and defines the pass plus its factory (abridged at the top):

***mlir/ShapeInferencePass.cpp***
```cpp
#include "toy/ShapeInferenceInterface.h"
...
#define DEBUG_TYPE "shape-inference"

using namespace mlir;
using namespace toy;

/// Include the auto-generated definitions for the shape inference interfaces.
#include "toy/ShapeInferenceOpInterfaces.cpp.inc"

namespace {
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
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)   // (1)
  StringRef getArgument() const override { return "toy-shape-inference"; }

  void runOnOperation() override {
    auto f = getOperation();

    // Populate the worklist with the operations that need shape inference:
    // these are operations that return a dynamic shape.
    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    f.walk([&](mlir::Operation *op) {
      if (returnsDynamicShape(op))                                   // (2)
        opWorklist.insert(op);
    });

    // Iterate on the operations in the worklist until all operations have been
    // inferred or no change happened (fix point).
    while (!opWorklist.empty()) {
      // Find the next operation ready for inference, that is an operation
      // with all operands already resolved (non-generic).
      auto nextop = llvm::find_if(opWorklist, allOperandsInferred);  // (3)
      if (nextop == opWorklist.end())
        break;

      Operation *op = *nextop;
      opWorklist.erase(op);

      // Ask the operation to infer its output shapes.
      LLVM_DEBUG(llvm::dbgs() << "Inferring shape for: " << *op << "\n");
      if (auto shapeOp = dyn_cast<ShapeInference>(op)) {             // (4)
        shapeOp.inferShapes();
      } else {
        op->emitError("unable to infer shape of operation without shape "
                      "inference interface");
        return signalPassFailure();
      }
    }

    // If the operation worklist isn't empty, this indicates a failure.
    if (!opWorklist.empty()) {
      f.emitError("Shape inference failed, ")                        // (5)
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
} // namespace

/// Create a Shape Inference pass.
std::unique_ptr<mlir::Pass> mlir::toy::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}
```

How the code implements the algorithm in its comment; the numbers match the `// (N)` markers:

- **(1) `MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID`** gives the pass an explicit `TypeID`. `mlir/Support/TypeID.h` says the implicit, name-based fallback must not be used for classes in anonymous namespaces.
- **(2) Worklist seeding** (`returnsDynamicShape`): any op with at least one non-`RankedTensorType` result needs inference. After inlining, that's the two `toy.cast`s, the two `toy.transpose`s, and the `toy.mul`.
- **(3) Readiness test** (`allOperandsInferred`): an op can only be inferred once *all* of its operands are ranked. Initially only the casts qualify (their inputs are `toy.constant` results, already `tensor<2x3xf64>`). Inferring the casts makes the transposes ready; inferring those makes the mul ready. Shapes thus flow forward through use-def chains Each round takes any ready op, in no particular order (the worklist is a `SmallPtrSet`), until none is left.
- **(4) The interface dispatch** is the whole point of the chapter: `dyn_cast<ShapeInference>(op)` asks "does this op — whatever dialect it belongs to — implement the interface?" The pass never mentions `AddOp`, `MulOp`, etc. If an op needs inference but doesn't implement the interface, that's a hard error.
- **(5) Failure detection**: if the loop stalls (no ready op) with work remaining, shapes couldn't be fully resolved — for example in the body of a function that wasn't inlined, whose arguments stay unranked. The pass reports and fails (run below).

The factory is declared in `include/toy/Passes.h`, which `toyc.cpp` includes; it lives in namespace `mlir::toy`:

***include/toy/Passes.h***
```cpp
namespace mlir {
class Pass;

namespace toy {
std::unique_ptr<Pass> createShapeInferencePass();
} // namespace toy
} // namespace mlir
```

#### The pass on the working example

The input to the pass is `main` right after inlining (section 4.5). Printing the IR after every pass and keeping only the shape-inference dump shows its result:

```bash
cd /Users/roy/study/mlir/toy/Ch4
./build/toyc-ch4 ../../test_Example/Toy/Ch4/codegen.toy -emit=mlir -opt --mlir-print-ir-after-all 2>&1 \
  | sed -n '/ShapeInferencePass/,/^}/p'
```

Real output **[10]**:

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

- The five seeded ops now have ranked results. The casts (`CastOp::inferShapes()`, section 5.2) became identities, the transposes reversed `2x3` to `3x2`, and the mul took its left operand's type.
- The pass changed only result types. It removed nothing: the identity casts and the duplicate constant are left for the canonicalizer and CSE (section 5.4).
- The header names the pass `(anonymous namespace)::ShapeInferencePass`, because the struct is in an anonymous namespace, and `(toy-shape-inference)`, the registered name from `getArgument()`. The dump shows a `toy.func`, not the module, because the pass runs nested on each function.

#### When inference fails

A call the inliner leaves in place defeats the pass. A self-recursive function is one: the inliner skips calls to a function that calls itself, so both the call in `main` and the one inside `@f` survive. The input is not a repo file, so it goes on stdin:

```bash
cd /Users/roy/study/mlir/toy/Ch4
cat <<'EOF' | ./build/toyc-ch4 - -x mlir -emit=mlir -opt 2>&1
toy.func private @f(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.generic_call @f(%arg0) : (tensor<*xf64>) -> tensor<*xf64>
  toy.return %0 : tensor<*xf64>
}
toy.func @main() {
  %0 = toy.constant dense<1.0> : tensor<2x3xf64>
  %1 = toy.generic_call @f(%0) : (tensor<2x3xf64>) -> tensor<*xf64>
  toy.print %1 : tensor<*xf64>
  toy.return
}
EOF
```

Real output **[11]** (exit status 4):

```text
<stdin>:1:1: error: Shape inference failed, 1 operations couldn't be inferred

toy.func private @f(%arg0: tensor<*xf64>) -> tensor<*xf64> {
^
<stdin>:1:1: note: see current operation: 
"toy.func"() <{function_type = (tensor<*xf64>) -> tensor<*xf64>, sym_name = "f"}> ({
^bb0(%arg0: tensor<*xf64>):
  %0 = "toy.generic_call"(%arg0) <{callee = @f}> : (tensor<*xf64>) -> tensor<*xf64>
  "toy.return"(%0) : (tensor<*xf64>) -> ()
}) {sym_visibility = "private"} : () -> ()
<stdin>:7:8: error: unable to infer shape of operation without shape inference interface
  %1 = toy.generic_call @f(%0) : (tensor<2x3xf64>) -> tensor<*xf64>
       ^
<stdin>:7:8: note: see current operation: %1 = "toy.generic_call"(%0) <{callee = @f}> : (tensor<2x3xf64>) -> tensor<*xf64>
```

The two errors are the pass's two failure paths:

- **In `@f`, the worklist stalls.** The call's only operand is the unranked `%arg0`, so `allOperandsInferred` never holds. The loop breaks with the op still in the worklist, and the pass reports `f.emitError("Shape inference failed, ") << opWorklist.size() << ...` on the function.
- **In `main`, the dispatch fails.** The call's operand `%0` is ranked, so the call is ready, but `dyn_cast<ShapeInference>(op)` returns null: `toy.generic_call` doesn't implement the interface. The pass reports `op->emitError("unable to infer shape ...")` on the op.
- Section 4.3's scratch build, whose `materializeCallConversion` returns `nullptr`, shows the same two errors on `codegen.toy`.

#### Debug output needs an assertions-enabled LLVM

`DEBUG_TYPE "shape-inference"` tags the `LLVM_DEBUG` line in `runOnOperation()` (both in the excerpt above), so a debug (assertions-enabled) LLVM build shows each inference step with `-debug-only=shape-inference`. The Homebrew LLVM 20 used here is a release build, which doesn't have that option:

```bash
cd /Users/roy/study/mlir/toy/Ch4
./build/toyc-ch4 ../../test_Example/Toy/Ch4/codegen.toy -emit=mlir -opt -debug-only=shape-inference
```

Real output **[12]** (exit status 1):

```text
toyc-ch4: Unknown command line argument '-debug-only=shape-inference'.  Try: './build/toyc-ch4 --help'
toyc-ch4: Did you mean '--debug-pass=shape-inference'?
```

The `--debug-pass` option it suggests is an unrelated LLVM option, not a replacement.

### 5.4 Adding the pass: the complete `-opt` pipeline

The pass is added to the pass manager in this repo's `toyc.cpp` (`dumpMLIR()`), as part of the `-opt` pipeline:

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

1. **`createInlinerPass()`** (module scope) — must run first. Shape inference is *intraprocedural*; it cannot see through `toy.generic_call`. Inlining brings the callee bodies to where the concrete shapes live and inserts `toy.cast` bridges (section 4.5).
2. **`pm.nest<toy::FuncOp>()`** — the remaining passes are *function*-level, so they are nested to run on every `toy.func` inside the module (in parallel, in principle). The nesting must be on **`toy::FuncOp`**: the pass manager anchors on the exact op type, and the shape-inference pass is restricted to `toy.func` (run below).
3. **`createShapeInferencePass()`** — the worklist algorithm from section 5.3 replaces every `tensor<*xf64>` with a ranked type and turns the casts into identities.
4. **`createCanonicalizerPass()`** — now the Chapter 3 patterns plus the `CastOpInterface` fold clean up: identity `toy.cast`s vanish. Before shape inference the casts aren't identities yet (`tensor<2x3xf64>` to `tensor<*xf64>`), so the fold, which only removes same-type casts (section 4.4), couldn't touch them.
5. **`createCSEPass()`** — common subexpression elimination. When two ops have the same name, operands, attributes and result types, CSE keeps the first and replaces every use of the second with it. It merges only ops whose side effects allow it — for Toy, the `Pure` ones; two `toy.print`s are never merged. The canonicalizer can't do this: its patterns rewrite one op from its operands, while CSE compares independent ops. Here, after canonicalization the two constants are structurally identical; CSE merges them, which makes the two transposes identical as well, shrinking `main` to one constant, one transpose, one mul.

**The pass must be nested.** Adding it to the module-level pass manager instead (`pm.addPass(...)` in place of `optPM.addPass(...)`) aborts while the pipeline is being built. A scratch copy with that one-line change shows it:

```bash
cd /Users/roy/study/mlir/toy/Ch4
S=$(mktemp -d)
cp -R CMakeLists.txt include mlir parser toyc.cpp "$S"/
sed -i '' 's/optPM.addPass(mlir::toy::createShapeInferencePass());/pm.addPass(mlir::toy::createShapeInferencePass());/' "$S"/toyc.cpp
cmake -S "$S" -B "$S"/build -G Ninja >/dev/null && cmake --build "$S"/build >/dev/null
"$S"/build/toyc-ch4 ../../test_Example/Toy/Ch4/codegen.toy -emit=mlir -opt 2>&1; echo "toyc-ch4 exit status: $?"
rm -rf "$S"
```

Real output **[13]** (abridged — the shell's own `Abort trap: 6` report, which carries a process ID, is left out):

```text
LLVM ERROR: Can't add pass '(anonymous namespace)::ShapeInferencePass' restricted to 'toy.func' on a PassManager intended to run on 'builtin.module', did you intend to nest?
toyc-ch4 exit status: 134
```

`toyc-ch4` aborts (`SIGABRT`, exit status 134) before any pass runs.

Rerunning the original example with the whole pipeline:

```bash
cd /Users/roy/study/mlir/toy/Ch4
./build/toyc-ch4 ../../test_Example/Toy/Ch4/codegen.toy -emit=mlir -opt 2>&1
```

Real output **[14]**:

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

- Compared with the shape-inference output in section 5.3, step 4 removed the identity casts and step 5 merged the constants and then the transposes: `toy.mul %1, %1` multiplies one transpose by itself.
- Appendix A shows the IR after each of these steps.

Everything this chapter built is visible in the difference from the raw codegen in section 3:

| Before (`-emit=mlir`, section 3) | After (`-emit=mlir -opt`) | Which mechanism |
|---|---|---|
| `toy.func private @multiply_transpose` exists | **gone** — dead after inlining | inliner + private visibility (section 4.1) |
| two `toy.generic_call` ops | **gone** — bodies spliced into `main` | `CallOpInterface`/`CallableOpInterface` (section 4.2) |
| `tensor<*xf64>` everywhere in the callee | every type ranked: `tensor<2x3xf64>`, `tensor<3x2xf64>` | `ShapeInferenceOpInterface` + pass (section 5) |
| (transiently) `toy.cast ... to tensor<*xf64>` after inlining | **gone** — inferred to identity, folded | `toy.cast` + `materializeCallConversion` + canonicalizer (sections 4.4, 4.5, 5.2) |
| unused call `%4` (variable `c`), redundant reshapes | **gone** | the inliner's canonicalizer: Chapter 3 reshape patterns, and dead `Pure` ops erased once the call is inlined (sections 4.3, 4.5) |
| two identical constants, two identical transposes | **one** constant, **one** transpose, `toy.mul %1, %1` | CSE (step 5 above) |

Each pass leaves the IR valid and simpler for the next: **the inliner makes the type change explicit as a cast, shape inference turns the cast into an identity, and the canonicalizer folds it away.**

---

## 6. Deriving the Chapter 4 interfaces, passes and build yourself

Sections 4 and 5 explain the chapter's code as it is. This section shows how to arrive at it without having it. Most of it is dictated by MLIR's own headers and `.td` files: which hooks to override, which methods to write, what an interface and a pass look like. Only the bodies (what Toy answers, how shapes propagate) are designed:

| Part of the code | Lines | Where it comes from |
|---|---|---|
| `ToyInlinerInterface` and `addInterfaces<ToyInlinerInterface>()` in `mlir/Dialect.cpp` | 47–96, 109 | The hooks of `DialectInlinerInterface` whose defaults refuse or abort (6.1) |
| `DeclareOpInterfaceMethods<...>` on `GenericCallOp`, `CastOp` and the four inferring ops in `include/toy/Ops.td`; `FuncOp`'s inline callable methods | — | The interfaces' method lists: which methods have no default (6.2) |
| `GenericCallOp`'s four call methods, `CastOp::areCastCompatible` and the four `inferShapes()` in `mlir/Dialect.cpp` | 329–347; 266–276; 253, 261, 368, 417–421 | The declarations ODS generates from those lists (6.2); a missing body fails at link time (6.3) |
| `include/toy/ShapeInferenceInterface.td` and `.h` | 1–30, 1–28 | `InterfaceMethod`'s parameters and a stock interface such as `CastOpInterface` (6.4) |
| `mlir/ShapeInferencePass.cpp` and `include/toy/Passes.h` | 1–123, 1–26 | `PassWrapper`/`OperationPass` in `mlir/Pass/Pass.h` and the stock factory declarations (6.5) |
| The `-opt` pipeline in `toyc.cpp` | 128–137 | The same pipeline written for stock `mlir-opt` first (6.6) |
| `function.setPrivate()` for every function but `main` in `mlir/MLIRGen.cpp` | 177–179 | The stock inliner keeping a public callee (6.6) |
| `toy.cast`'s record in `include/toy/Ops.td` | 118–137 | A stock cast op, `tensor.cast` (6.7) |
| The interface's TableGen rule in `include/toy/CMakeLists.txt`, its `add_dependencies` in `CMakeLists.txt`, and the two new interface `#include`s in `include/toy/Dialect.h` | 10–13; 53; 22, 25 | The traits the generated op classes name, and the build error without the rule (6.8) |

What is left to design is small: the answers `ToyInlinerInterface` gives (section 4.1), the shape rules in each `inferShapes()` (section 5.2), and the worklist algorithm in `runOnOperation()` (section 5.3). The other differences from Chapter 3 need no step: `GenericCallOp`'s optional `arg_attrs`/`res_attrs`, which nothing in this chapter reads (section 4.2), and a comment line in `mlir/ToyCombine.td`.

### 6.1 Step 1: find the inliner hooks Toy must override

The inliner reaches a dialect through `DialectInlinerInterface` (`mlir/Transforms/InliningUtils.h`, section 4.1). Its hooks are virtual methods with default bodies, and the defaults tell you which ones a dialect must override. The second `awk` joins each hook, from `virtual` to the end of its default body, onto one line:

```bash
cd /Users/roy/study/mlir/toy/Ch4
awk '/^class DialectInlinerInterface/,/^};/' /opt/homebrew/opt/llvm@20/include/mlir/Transforms/InliningUtils.h \
  | awk '/virtual/ { h = 1; s = "" } h { sub(/^ +/, ""); s = s (s == "" ? "" : " ") $0 } h && /}$/ { print s; h = 0 }'
```

Real output **[15]**:

```text
virtual bool isLegalToInline(Operation *call, Operation *callable, bool wouldBeCloned) const { return false; }
virtual bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned, IRMapping &valueMapping) const { return false; }
virtual bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned, IRMapping &valueMapping) const { return false; }
virtual bool shouldAnalyzeRecursively(Operation *op) const { return true; }
virtual void handleTerminator(Operation *op, Block *newDest) const { llvm_unreachable("must implement handleTerminator in the case of multiple " "inlined blocks"); }
virtual void handleTerminator(Operation *op, ValueRange valuesToReplace) const { llvm_unreachable( "must implement handleTerminator in the case of one inlined block"); }
virtual Operation *materializeCallConversion(OpBuilder &builder, Value input, Type resultType, Location conversionLoc) const { return nullptr; }
virtual Value handleArgument(OpBuilder &builder, Operation *call, Operation *callable, Value argument, DictionaryAttr argumentAttrs) const { return argument; }
virtual Value handleResult(OpBuilder &builder, Operation *call, Operation *callable, Value result, DictionaryAttr resultAttrs) const { return result; }
virtual void processInlinedCallBlocks( Operation *call, iterator_range<Region::iterator> inlinedBlocks) const {}
virtual bool allowSingleBlockOptimization( iterator_range<Region::iterator> inlinedBlocks) const { return true; }
```

- **The three `isLegalToInline` overloads return `false`**: a dialect that doesn't override them is never inlined. Toy overrides all three to return `true` (section 4.1).
- **The single-block `handleTerminator` aborts** (`llvm_unreachable`), so any dialect whose callees have one block must implement it. Toy's functions always have one block, which is why `ToyInlinerInterface` implements this overload and not the `Block *newDest` one.
- **`materializeCallConversion` returns `nullptr`**: no cast, so a call whose operand types differ from the callee's isn't inlined (section 4.3's scratch build). Toy overrides it to build a `toy.cast` (section 4.5).
- The other hooks have defaults that are already right for Toy (analyze nested regions, pass arguments and results through unchanged), so the struct doesn't mention them.

### 6.2 Step 2: find which interface methods need a body

An op interface lists its methods in ODS; a method with a `defaultImplementation` is provided by the interface's trait, and every other method has to be written by the op. The generated declarations show both: the `Concept` struct lists every method (section 5.1), the `...Trait` struct defines the ones that have a default. The `awk` script collects the first and marks the second:

```bash
cd /Users/roy/study/mlir/toy/Ch4
I=/opt/homebrew/opt/llvm@20/include
for f in CallInterfaces CastInterfaces; do
  mlir-tblgen -gen-op-interface-decls -I $I $I/mlir/Interfaces/$f.td \
    | awk '/^struct [A-Za-z]+InterfaceTraits/ { i = $2; sub(/InterfaceInterfaceTraits$/, "Interface", i) }
           /^    [^ ].*\(\*[A-Za-z]+\)\(/ { match($0, /\(\*[A-Za-z]+\)/); m = substr($0, RSTART + 2, RLENGTH - 3); n++; iface[n] = i; name[n] = m }
           /^  struct [A-Za-z]+Trait : / { t = 1; next }
           t && /^  };/ { t = 0 }
           t && /^    [^ \/].*[A-Za-z]+\(.*\{$/ { s = $0; sub(/\(.*/, "", s); sub(/.*[ *&]/, "", s); dflt[s] = 1 }
           END { for (k = 1; k <= n; k++) printf "%-20s %-24s %s\n", iface[k], name[k], (name[k] in dflt) ? "default" : "MUST WRITE" }'
done
```

Real output **[16]**:

```text
CallOpInterface      getCallableForCallee     MUST WRITE
CallOpInterface      setCalleeFromCallable    MUST WRITE
CallOpInterface      getArgOperands           MUST WRITE
CallOpInterface      getArgOperandsMutable    MUST WRITE
CallOpInterface      resolveCallableInTable   default
CallOpInterface      resolveCallable          default
CallableOpInterface  getCallableRegion        MUST WRITE
CallableOpInterface  getArgumentTypes         MUST WRITE
CallableOpInterface  getResultTypes           MUST WRITE
CallableOpInterface  getArgAttrsAttr          default
CallableOpInterface  getResAttrsAttr          default
CallableOpInterface  setArgAttrsAttr          default
CallableOpInterface  setResAttrsAttr          default
CallableOpInterface  removeArgAttrsAttr       default
CallableOpInterface  removeResAttrsAttr       default
CastOpInterface      areCastCompatible        MUST WRITE
```

- **`CallOpInterface`: four methods to write**, all about where the callee and its arguments are. Resolving the callee symbol has a default, which is why `GenericCallOp` never looks up `@multiply_transpose` itself (section 4.2).
- **`CallableOpInterface`: three methods to write** (region, argument types, result types). The attribute methods default to "no attributes".
- **`CastOpInterface`: one method**, `areCastCompatible` (section 4.4). The verifier and the fold hook come with the trait, not as methods to write.

`DeclareOpInterfaceMethods<...>` in `Ops.td` turns exactly the "MUST WRITE" methods into declarations on the generated op class (section 4.2). In the repo's generated classes:

```bash
cd /Users/roy/study/mlir/toy/Ch4
mlir-tblgen -gen-op-decls include/toy/Ops.td -I include -I /opt/homebrew/opt/llvm@20/include \
  | awk '/^class [A-Za-z]+ : / { c = $2 }
         /[ *](getCallableForCallee|setCalleeFromCallable|getArgOperands|getArgOperandsMutable|getCallableRegion|getArgumentTypes|getResultTypes|areCastCompatible)\(/ { sub(/^ +/, ""); print c ": " $0 }'
```

Real output **[17]**:

```text
CastOp: static bool areCastCompatible(::mlir::TypeRange inputs, ::mlir::TypeRange outputs);
FuncOp: ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
FuncOp: ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
FuncOp: Region *getCallableRegion() { return &getBody(); }
GenericCallOp: ::mlir::CallInterfaceCallable getCallableForCallee();
GenericCallOp: void setCalleeFromCallable(::mlir::CallInterfaceCallable callee);
GenericCallOp: ::mlir::Operation::operand_range getArgOperands();
GenericCallOp: ::mlir::MutableOperandRange getArgOperandsMutable();
```

- `GenericCallOp`'s four and `CastOp`'s one are declarations without a body: the bodies are this chapter's additions to `Dialect.cpp` (sections 4.2 and 4.4). Section 5.2 runs the same check for `inferShapes()`.
- `FuncOp`'s three come with bodies, because `toy.func` gets `CallableOpInterface` through `FunctionOpInterface` and writes the methods inline in its `extraClassDeclaration` (section 4.2) instead of declaring them with `DeclareOpInterfaceMethods`.

### 6.3 Step 3: let the linker find a missing body

A declared method without a body compiles, because each op's interface model only calls it; the linker reports it. A scratch copy of the chapter without `GenericCallOp::getArgOperandsMutable()` (deleted by `sed` in a temporary directory; the build takes about 15 seconds) shows the error you get when a "MUST WRITE" method of section 6.2 is forgotten:

```bash
cd /Users/roy/study/mlir/toy/Ch4
S=$(mktemp -d)
cp -R CMakeLists.txt include mlir parser toyc.cpp "$S"/
sed -i '' '/^MutableOperandRange GenericCallOp::getArgOperandsMutable() {$/,/^}$/d' "$S"/mlir/Dialect.cpp
cmake -S "$S" -B "$S"/build -G Ninja >/dev/null
cmake --build "$S"/build > "$S"/build.log 2>&1; echo "cmake --build exit status: $?"
grep -A3 'Undefined symbols' "$S"/build.log
rm -rf "$S"
```

Real output **[18]**:

```text
cmake --build exit status: 1
Undefined symbols for architecture arm64:
  "mlir::toy::GenericCallOp::getArgOperandsMutable()", referenced from:
      mlir::detail::CallOpInterfaceInterfaceTraits::Model<mlir::toy::GenericCallOp>::getArgOperandsMutable(mlir::detail::CallOpInterfaceInterfaceTraits::Concept const*, mlir::Operation*) in Dialect.cpp.o
ld: symbol(s) not found for architecture arm64
```

- The missing symbol is referenced from `CallOpInterfaceInterfaceTraits::Model<mlir::toy::GenericCallOp>`, the generated model that fills the interface's function-pointer table for this op (section 5.1).
- So the error names both the method and the interface that needs it; the fix is to write the body in `Dialect.cpp`.

### 6.4 Step 4: define the new interface on a stock one's pattern

`ShapeInferenceOpInterface` is a new interface, so nothing generates its list; you write it. The ODS classes for it are in `mlir/IR/Interfaces.td`, and `CastOpInterface` is a short stock interface to copy the shape from:

```bash
cd /Users/roy/study/mlir/toy/Ch4
I=/opt/homebrew/opt/llvm@20/include
grep -n -A2 '^class InterfaceMethod<\|^class OpInterface<' $I/mlir/IR/Interfaces.td
sed -n '/^def CastOpInterface/,/^  \];/p' $I/mlir/Interfaces/CastInterfaces.td
```

Real output **[19]**:

```text
59:class InterfaceMethod<string desc, string retTy, string methodName,
60-                      dag args = (ins), code methodBody = [{}],
61-                      code defaultImplementation = [{}]> {
--
138:class OpInterface<string name, list<Interface> baseInterfaces = []>
139-  : Interface<name, baseInterfaces>, OpInterfaceTrait<name>;
140-
def CastOpInterface : OpInterface<"CastOpInterface"> {
  let description = [{
    A cast-like operation is one that converts from a set of input types to a
    set of output types. The arity of the inputs may be from 0-N, whereas the
    arity of the outputs may be anything from 1-N. Cast-like operations are
    trivially removable in cases where they produce an No-op, i.e when the
    input types and output types match 1-1.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    StaticInterfaceMethod<[{
        Returns true if the given set of input and result types are compatible
        to cast using this cast operation.
      }],
      "bool", "areCastCompatible",
      (ins "::mlir::TypeRange":$inputs, "::mlir::TypeRange":$outputs)
    >,
  ];
```

- `InterfaceMethod<desc, retTy, methodName, args, methodBody, defaultImplementation>`: only the first three are required, and leaving out the default is what makes the method one each op must write (6.2). `ShapeInferenceInterface.td` uses exactly the three: a description, `"void"`, `"inferShapes"` (section 5.1).
- `CastOpInterface` has the same layout: `OpInterface<"ClassName">`, a `description`, a `methods` list. Its method is a `StaticInterfaceMethod` (called without an op, `CastOp::areCastCompatible(...)`); `inferShapes` changes one op's result, so it is a plain `InterfaceMethod`.
- One line differs: `let cppNamespace = "::mlir";`. The Toy interface doesn't set it, which decides how its header is written.

The header follows from that one line. A stock header includes its generated declarations at file scope, because the generated code opens its own `namespace mlir`; without `cppNamespace` the generated code opens none, so `ShapeInferenceInterface.h` wraps the `#include` in `namespace mlir { namespace toy {` (section 5.1 shows the header). Count the namespaces each generated file opens:

```bash
cd /Users/roy/study/mlir/toy/Ch4
I=/opt/homebrew/opt/llvm@20/include
grep -n 'include.*\.h\.inc' $I/mlir/Interfaces/CastInterfaces.h include/toy/ShapeInferenceInterface.h
mlir-tblgen -gen-op-interface-decls -I $I $I/mlir/Interfaces/CastInterfaces.td | grep -c '^namespace mlir {'
mlir-tblgen -gen-op-interface-decls include/toy/ShapeInferenceInterface.td -I include -I $I | grep -c '^namespace mlir {'
```

Real output **[20]** (exit status 1 — `grep -c` exits 1 when it counts no match):

```text
/opt/homebrew/opt/llvm@20/include/mlir/Interfaces/CastInterfaces.h:38:#include "mlir/Interfaces/CastInterfaces.h.inc"
include/toy/ShapeInferenceInterface.h:23:#include "toy/ShapeInferenceOpInterfaces.h.inc"
2
0
```

- The stock header includes `CastInterfaces.h.inc` on its own line, outside any namespace, and the generated code opens `namespace mlir` twice by itself.
- The Toy header's `#include` (line 23) sits inside the namespaces, because its generated code opens none: without that wrapping `ShapeInference` would land in the global namespace instead of `mlir::toy`.

### 6.5 Step 5: take the pass skeleton from `mlir/Pass/Pass.h`

A pass is a class the pass manager can run on one kind of op. `mlir/Pass/Pass.h` has the base classes, and `mlir/Transforms/Passes.h` shows how stock passes publish their factory:

```bash
cd /Users/roy/study/mlir/toy/Ch4
I=/opt/homebrew/opt/llvm@20/include/mlir
grep -n -B1 '^class OperationPass : \|^class PassWrapper : ' $I/Pass/Pass.h
grep -n 'virtual .* = 0;\|virtual StringRef getArgument() const' $I/Pass/Pass.h
grep -n 'std::unique_ptr<Pass> create\(Canonicalizer\|CSE\|Inliner\)Pass();' $I/Transforms/Passes.h
```

Real output **[21]**:

```text
354-template <typename OpT = void>
355:class OperationPass : public Pass {
--
456-template <typename PassT, typename BaseT>
457:class PassWrapper : public BaseT {
64:  virtual StringRef getName() const = 0;
75:  virtual StringRef getArgument() const { return ""; }
179:  virtual void runOnOperation() = 0;
194:  virtual bool canScheduleOn(RegisteredOperationName opName) const = 0;
294:  virtual std::unique_ptr<Pass> clonePass() const = 0;
53:std::unique_ptr<Pass> createCanonicalizerPass();
72:std::unique_ptr<Pass> createCSEPass();
100:std::unique_ptr<Pass> createInlinerPass();
```

- **Four pure virtual methods** make a `Pass` abstract. The two base classes fill three of them, so a pass deriving from `PassWrapper<..., OperationPass<OpT>>` writes only `runOnOperation()`:
  - **`OperationPass<OpT>`** defines `canScheduleOn`: the pass runs only on `OpT`. With `OpT = toy::FuncOp`, shape inference runs on each `toy.func` on its own (section 5.3), and the pass manager refuses it at module level (section 5.4).
  - **`PassWrapper<PassT, BaseT>`** defines `getName` and `clonePass` from the derived class `PassT` it is given, and its constructor passes `TypeID::get<PassT>()` to the base: that is why the struct needs the `MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID` line (section 5.3).
- **`getArgument()`** is optional; its default `""` leaves the pass without a command-line name, so the pass overrides it with `toy-shape-inference`, the name in the `--mlir-print-ir-after-all` headers.
- **The factory** follows the stock declarations: `std::unique_ptr<Pass> createXPass();`. `include/toy/Passes.h` declares `createShapeInferencePass()` the same way in `mlir::toy`, with a forward declaration of `mlir::Pass` instead of including `Pass.h` (section 5.3), and the struct itself stays in an anonymous namespace in the `.cpp`.

### 6.6 Step 6: write the pipeline for stock `mlir-opt` first

The three other passes of the `-opt` pipeline are stock passes, so `mlir-opt` has them:

```bash
cd /Users/roy/study/mlir/toy/Ch4
mlir-opt --help | grep -E -- '--(inline|canonicalize|cse) '
```

Real output **[22]**:

```text
      --canonicalize                                         -   Canonicalize operations
      --cse                                                  -   Eliminate common sub-expressions
      --inline                                               -   Inline function calls
```

The `create...Pass()` factories found in section 6.5 are the C++ side of these options. Before writing `toyc.cpp`, try the pipeline's structure on a stock-dialect stand-in for `codegen.toy` (the input is not a repo file, so it goes on stdin): a private function called twice, with the inliner at module level and the per-function passes nested under `func.func`. The same input runs a second time with `private` removed by `sed`:

```bash
cd /Users/roy/study/mlir/toy/Ch4
IN=$(cat <<'EOF'
func.func private @square(%arg0: tensor<2x3xf64>) -> tensor<2x3xf64> {
  %0 = arith.mulf %arg0, %arg0 : tensor<2x3xf64>
  return %0 : tensor<2x3xf64>
}
func.func @main(%arg0: tensor<2x3xf64>) -> (tensor<2x3xf64>, tensor<2x3xf64>) {
  %0 = call @square(%arg0) : (tensor<2x3xf64>) -> tensor<2x3xf64>
  %1 = call @square(%arg0) : (tensor<2x3xf64>) -> tensor<2x3xf64>
  return %0, %1 : tensor<2x3xf64>, tensor<2x3xf64>
}
EOF
)
echo "$IN" | mlir-opt --pass-pipeline='builtin.module(inline,func.func(canonicalize,cse))'
echo "== @square public"
echo "$IN" | sed 's/func.func private/func.func/' | mlir-opt --pass-pipeline='builtin.module(inline,func.func(canonicalize,cse))'
```

Real output **[23]**:

```mlir
module {
  func.func @main(%arg0: tensor<2x3xf64>) -> (tensor<2x3xf64>, tensor<2x3xf64>) {
    %0 = arith.mulf %arg0, %arg0 : tensor<2x3xf64>
    return %0, %0 : tensor<2x3xf64>, tensor<2x3xf64>
  }
}

== @square public
module {
  func.func @square(%arg0: tensor<2x3xf64>) -> tensor<2x3xf64> {
    %0 = arith.mulf %arg0, %arg0 : tensor<2x3xf64>
    return %0 : tensor<2x3xf64>
  }
  func.func @main(%arg0: tensor<2x3xf64>) -> (tensor<2x3xf64>, tensor<2x3xf64>) {
    %0 = arith.mulf %arg0, %arg0 : tensor<2x3xf64>
    return %0, %0 : tensor<2x3xf64>, tensor<2x3xf64>
  }
}
```

- The calls are inlined, the private `@square` is erased, and the two identical `arith.mulf`s are merged into one: the stock passes do for `func` what Toy wants for `toy.func`.
- **Public, `@square` survives:** the calls are still inlined, but the inliner erases only a callee that nothing outside the module can reference. That is the reason for the one change to `mlir/MLIRGen.cpp`: `function.setPrivate()` for every function except `main` (section 4.1, which runs the same experiment on `toyc-ch4`).
- The pipeline string is the C++ pipeline of section 5.4 written as text: `builtin.module(inline, ...)` is `pm.addPass(mlir::createInlinerPass())`, and `func.func(canonicalize,cse)` is `pm.nest<...>()` with `createCanonicalizerPass()` and `createCSEPass()` added to the nested manager. For Toy the nesting op is `toy::FuncOp` and the nested list starts with `createShapeInferencePass()`, the one pass `mlir-opt` doesn't have.

**Write the pipeline as an `mlir-opt` string first: its nesting is the `pm.nest<...>()` structure the C++ needs, and the stock passes check it before a line of the driver exists.**

### 6.7 Step 7: take `toy.cast` from a stock cast op

`materializeCallConversion` has to build *some* op that converts one tensor type to another (section 6.1), and Toy has none yet. MLIR's own dialects have cast ops declared with `CastOpInterface`; `tensor.cast` is the closest to Toy's tensors. Its record, with the description elided:

```bash
cd /Users/roy/study/mlir/toy/Ch4
awk '/^def Tensor_CastOp /,/^}/' /opt/homebrew/opt/llvm@20/include/mlir/Dialect/Tensor/IR/TensorOps.td \
  | awk '/let description/ { d = 1 } d && /\}\];/ { d = 0; print "  let description = [{ ... }];"; next } !d && NF'
```

Real output **[24]**:

```text
def Tensor_CastOp : Tensor_Op<"cast", [
    DeclareOpInterfaceMethods<CastOpInterface>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    Pure
  ]> {
  let summary = "tensor cast operation";
  let description = [{ ... }];
  let arguments = (ins AnyTensor:$source);
  let results = (outs AnyTensor:$dest);
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";
  let hasCanonicalizer = 1;
}
```

Compared with `toy.cast`'s record (section 4.4):

- **Kept:** `DeclareOpInterfaceMethods<CastOpInterface>`, which leaves `areCastCompatible` to write (section 6.2), `Pure`, one tensor in and one out, and the assembly format `$input attr-dict : type($input) to type($output)`, with Toy's operand names and `F64Tensor` for `AnyTensor`.
- **Dropped:** `OpAsmOpInterface`, which only names the result in printed IR (`%cast`), and the canonicalizer.
- **Added:** `DeclareOpInterfaceMethods<ShapeInferenceOpInterface>`, because shapes must flow through the casts the inliner inserts (section 5.2), and `SameOperandsAndResultShape` (section 4.4).

### 6.8 Step 8: generate the interface's C++ in the build

The op records now name two interfaces whose C++ the build has to provide. `mlir-tblgen` lists the backends that generate an op interface, and the repo's generated op classes show which interface traits they derive from, i.e. which headers `Dialect.h` must include before `Ops.h.inc`. A scratch copy without the interface's TableGen rule (removed by `sed`; the build takes about 15 seconds) shows what breaks without it:

```bash
cd /Users/roy/study/mlir/toy/Ch4
I=/opt/homebrew/opt/llvm@20/include
mlir-tblgen --help | grep -E -- '-gen-op-interface-(decls|defs) '
mlir-tblgen -gen-op-decls -I include -I $I include/toy/Ops.td | grep -oE '[A-Za-z:]*(CastOpInterface|ShapeInference)::Trait' | sort -u
S=$(mktemp -d)
cp -R CMakeLists.txt include mlir parser toyc.cpp "$S"/
sed -E -i '' '/ShapeInference(Interface\.td|OpInterfaces|InterfaceIncGen)/d' "$S"/include/toy/CMakeLists.txt "$S"/CMakeLists.txt
cmake -S "$S" -B "$S"/build -G Ninja >/dev/null
cmake --build "$S"/build > "$S"/build.log 2>&1; echo "cmake --build exit status: $?"
grep -m1 -A2 'fatal error' "$S"/build.log | sed "s|$S/||"
rm -rf "$S"
```

Real output **[25]**:

```text
      --gen-op-interface-decls                          - Generate op interface declarations
      --gen-op-interface-defs                           - Generate op interface definitions
::mlir::CastOpInterface::Trait
ShapeInference::Trait
cmake --build exit status: 1
include/toy/ShapeInferenceInterface.h:23:10: fatal error: 'toy/ShapeInferenceOpInterfaces.h.inc' file not found
   23 | #include "toy/ShapeInferenceOpInterfaces.h.inc"
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

- **The two backends** are the ones sections 6.2 and 6.4 already ran by hand. `include/toy/CMakeLists.txt` runs them as one `mlir_tablegen` line each, writing `ShapeInferenceOpInterfaces.h.inc` and `.cpp.inc`, and `add_public_tablegen_target` names the step `ToyCh4ShapeInferenceInterfaceIncGen`, the same pattern as the `Ops.td` rule above it (appendix C.2).
- **The two traits** are why `Dialect.h` gains two includes: `mlir/Interfaces/CastInterfaces.h` for `::mlir::CastOpInterface::Trait`, and `toy/ShapeInferenceInterface.h` for `ShapeInference::Trait`, which `Ops.h.inc` uses inside `mlir::toy` (section 6.4).
- **Without the rule**, nothing writes the `.h.inc` that `ShapeInferenceInterface.h` includes, and the first file that includes `Dialect.h` fails to compile. `add_dependencies(toyc-ch4 ToyCh4ShapeInferenceInterfaceIncGen)` in `CMakeLists.txt` makes sure the files exist before any source that needs them is compiled (appendix C.1).

---

## 7. Key Takeaways & Pitfalls

**Takeaways**

1. **Interfaces invert the dependency.** The inliner never learned about Toy; Toy taught itself to the inliner. This is MLIR's scaling trick: each pass is written once and each dialect implements a few interfaces, so the work grows as dialects + transformations, not dialects × transformations (section 2).
2. **Dialect interfaces vs. op interfaces.** Use a *dialect* interface for blanket policy (`ToyInlinerInterface`: "everything is inlinable", terminator handling, cast materialization); use *op* interfaces for per-op capabilities (`CallOpInterface`, `ShapeInferenceOpInterface`).
3. **`DeclareOpInterfaceMethods<...>` is the ODS glue.** It both attaches the interface and declares the methods on the generated op class so you implement them in your `.cpp` — declare them and forget a body, and you get a link error (section 4.2).
4. **Casts make type refinement safe.** Rather than mutating types across a call boundary, the inliner asks the dialect to materialize explicit `toy.cast`s; shape inference later proves them trivial and the canonicalizer deletes them. Explicit-then-fold is a recurring MLIR idiom.
5. **A custom interface is ~15 lines of TableGen.** `OpInterface` + `InterfaceMethod` + two `mlir_tablegen` invocations gives you a `dyn_cast`-able C++ class usable across dialects.
6. **MLIR's headers say what a dialect must write** (section 6): a hook whose default refuses or aborts, or an interface method without a `defaultImplementation`, is one to provide, and the linker names a forgotten one.
7. **Pipeline order encodes the reasoning**: inline (bring shapes to the code) → infer (propagate them) → canonicalize (exploit them) → CSE (deduplicate what canonicalization exposed); each pass leaves the IR ready for the next (section 5.4).

**Pitfalls**

- **Forgetting `addInterfaces<ToyInlinerInterface>()`** in `ToyDialect::initialize()`, or returning `nullptr` from `materializeCallConversion` — the inliner quietly leaves every `toy.generic_call` in place; the first visible symptom is the shape-inference pass failing, as in section 4.3's scratch build (section 5.3 explains the two errors).
- **Forgetting `setPrivate()`** on non-`main` functions — inlining still happens, but the dead public `@multiply_transpose` is kept, and since it is still generic, `-opt` fails with `Shape inference failed, 3 operations couldn't be inferred` on it (run in section 4.1).
- **`pm.addPass(createShapeInferencePass())` at module level** — the pass is restricted to `toy.func`, so adding it to the module pass manager aborts with `LLVM ERROR: Can't add pass ... restricted to 'toy.func' ...` before anything runs (section 5.4).
- **Running shape inference without inlining first**: `toy.generic_call` produces `tensor<*xf64>` and does not implement `ShapeInference`, so the pass hard-fails ("unable to infer shape of operation without shape inference interface"; section 5.3).
- **`inferShapes()` ordering assumptions**: implementations like `TransposeOp::inferShapes()` `cast<RankedTensorType>` unconditionally — safe only because the pass gates on `allOperandsInferred`. Called on an op whose operand is still unranked, the `cast` would be wrong; this release build has no assertions to catch it.
- **Reading `--mlir-print-ir-after-all` output**: the first dumps are the inliner's *nested* canonicalization runs, not the inliner itself; the `Inliner (inline)` dump comes after them (appendix A).
- **Dropping `add_dependencies(toyc-ch4 ToyCh4ShapeInferenceInterfaceIncGen)`** can make the build compile `Dialect.cpp` or `ShapeInferencePass.cpp` before the interface `.inc` files they include exist (appendix C.1).

---

## Appendix A: The `-opt` pipeline pass by pass

This extends section 5.4: the IR after every pass of the `-opt` pipeline, including the canonicalizer runs nested inside the inliner (section 4.3). Sections 4.5 and 5.3 showed the inliner and shape-inference dumps on their own.

```bash
cd /Users/roy/study/mlir/toy/Ch4
./build/toyc-ch4 ../../test_Example/Toy/Ch4/codegen.toy -emit=mlir -opt --mlir-print-ir-after-all 2>&1
```

Real output **[26]** (abridged: the final `module { ... }` dump, identical to section 5.4, is omitted):

```mlir
// -----// IR Dump After Canonicalizer (canonicalize) //----- //
toy.func private @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
  %2 = toy.mul %0, %1 : tensor<*xf64>
  toy.return %2 : tensor<*xf64>
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.generic_call @multiply_transpose(%0, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  %3 = toy.generic_call @multiply_transpose(%1, %0) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  toy.print %3 : tensor<*xf64>
  toy.return
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
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

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.transpose(%1 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %4 = toy.mul %2, %3 : tensor<3x2xf64>
  toy.print %4 : tensor<3x2xf64>
  toy.return
}

// -----// IR Dump After CSE (cse) //----- //
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %2 = toy.mul %1, %1 : tensor<3x2xf64>
  toy.print %2 : tensor<3x2xf64>
  toy.return
}
```

Reading the dumps in order:

1. **The first three `Canonicalizer` dumps happen *inside* the inliner.** The inliner runs a simplification pipeline (canonicalization by default, section 4.5) on each function while it walks the call graph, and `--mlir-print-ir-after-all` prints those nested runs too. The first one leaves the callee unchanged. The second one, on `main` *before* any call is inlined, already applies the Chapter 3 patterns: both reshapes are gone and `b`'s flat constant has been folded into a second `tensor<2x3xf64>` constant. Both `toy.generic_call`s are still there — a call is not `Pure`, so the canonicalizer can't delete the unused one.
2. The third `Canonicalizer` dump is `main` after both calls were inlined. Each inlined argument got a `toy.cast` to `tensor<*xf64>` (materialized by the hook in section 4.5). The inlined body of the unused call `c = multiply_transpose(a, b)` consisted only of `Pure` transposes and a mul with no users, so this canonicalization erased it; only the ops for `d` remain.
3. **`Inliner (inline)`** — the module after the inliner finishes: `@multiply_transpose` has been deleted (it was private and has no remaining uses), and `main` is the IR from step 2.
4. **`ShapeInferencePass (toy-shape-inference)`** — every type is now ranked, and both casts have become identities (`tensor<2x3xf64> to tensor<2x3xf64>`). The pass is explained in section 5.3.
5. **`Canonicalizer`** — the identity casts are folded away (sections 4.4, 5.2).
6. **`CSE`** — the duplicate constant is merged, which makes the two transposes identical, so they are merged too, leaving `toy.mul %1, %1`.

**The ecosystem view.** The parenthesized names in the dump headers — `(inline)`, `(canonicalize)`, `(cse)` — are the *registered pass names*, and three of the four passes in this chapter's pipeline are stock MLIR passes, the same ones any `mlir-opt`-family tool exposes as `--inline`, `--canonicalize`, `--cse`. Only `(toy-shape-inference)` is chapter-local. `toyc-ch4 -opt` is therefore a hardcoded version of the pipeline that `--pass-pipeline` syntax would write as `builtin.module(inline, toy.func(toy-shape-inference, canonicalize, cse))` inside a binary that links the Toy dialect and its one custom pass.

---

## Appendix B: More inputs

`test_Example/Toy/Ch4/` also contains `shape_inference.mlir` — the raw codegen module of section 3 written out as MLIR, so it exercises the pipeline without the Toy frontend. The `.mlir` extension alone selects the MLIR parser path in `loadMLIR()`:

```bash
cd /Users/roy/study/mlir/toy/Ch4
./build/toyc-ch4 ../../test_Example/Toy/Ch4/shape_inference.mlir -emit=mlir -opt 2>&1
```

Real output **[27]** (the same module as section 5.4):

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

In `transpose_transpose.toy` the `transpose(transpose(x))` sits inside the generic function `transpose_transpose`. Once it is inlined into `main`, the Chapter 3 `SimplifyRedundantTranspose` pattern removes both transposes, and the function disappears:

```bash
cd /Users/roy/study/mlir/toy/Ch4
./build/toyc-ch4 ../../test_Example/Toy/Ch4/transpose_transpose.toy -emit=mlir -opt 2>&1
```

Real output **[28]**:

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    toy.print %0 : tensor<2x3xf64>
    toy.return
  }
}
```

`trivial_reshape.toy` has no function to inline; the Chapter 3 reshape patterns fold its reshapes into one constant, as before:

```bash
cd /Users/roy/study/mlir/toy/Ch4
./build/toyc-ch4 ../../test_Example/Toy/Ch4/trivial_reshape.toy -emit=mlir -opt 2>&1
```

Real output **[29]**:

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>
    toy.print %0 : tensor<2x1xf64>
    toy.return
  }
}
```


The negative test `invalid.mlir` is rejected by the verifier while parsing:

```bash
cd /Users/roy/study/mlir/toy/Ch4
./build/toyc-ch4 ../../test_Example/Toy/Ch4/invalid.mlir -emit=mlir
```

Real output **[30]** (exit status 3):

```text
../../test_Example/Toy/Ch4/invalid.mlir:8:8: error: 'toy.print' op requires zero results
  %0 = "toy.print"()  : () -> tensor<2x3xf64>
       ^
../../test_Example/Toy/Ch4/invalid.mlir:8:8: note: see current operation: %0 = "toy.print"() : () -> tensor<2x3xf64>
Error can't load file ../../test_Example/Toy/Ch4/invalid.mlir
```

`ast.toy`, `empty.toy` and `scalar.toy` are carried over unchanged from earlier chapters.

**Poking at `toy.cast` directly.** The hand-written casts of section 4.4 can go through `-opt` too. A cast to `tensor<*xf64>` in front of a generic op goes through the whole story of sections 4.4–5.4: shape inference makes the cast an identity and the canonicalizer folds it away.

```bash
cd /Users/roy/study/mlir/toy/Ch4
cat <<'EOF' | ./build/toyc-ch4 - -x mlir -emit=mlir -opt 2>&1
toy.func @main() {
  %0 = toy.constant dense<1.0> : tensor<2x3xf64>
  %1 = toy.cast %0 : tensor<2x3xf64> to tensor<*xf64>
  %2 = toy.transpose(%1 : tensor<*xf64>) to tensor<*xf64>
  toy.print %2 : tensor<*xf64>
  toy.return
}
EOF
```

Real output **[31]**:

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<1.000000e+00> : tensor<2x3xf64>
    %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
    toy.print %1 : tensor<3x2xf64>
    toy.return
  }
}
```

---

## Appendix C: What Chapter 4 adds to the build

The op/dialect TableGen pattern is explained in Ch2 appendix E and the DRR rewriter step in Ch3 appendix D. Chapter 4 adds a third TableGen flavor: op-interface generation. So three TableGen targets now run before the C++ compiles: the op/dialect `.inc` files, the DRR rewriters (`build/ToyCombine.inc`), and the shape-inference interface (`build/include/toy/ShapeInferenceOpInterfaces.{h,cpp}.inc`).

### C.1 The chapter targets

Below the standalone guard:

***CMakeLists.txt***
```cmake
include_directories(include)
add_subdirectory(include)

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCh4CombineIncGen)

add_executable(toyc-ch4
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/ShapeInferencePass.cpp
  mlir/ToyCombine.cpp
  )

add_dependencies(toyc-ch4 ToyCh4OpsIncGen)
add_dependencies(toyc-ch4 ToyCh4ShapeInferenceInterfaceIncGen)
add_dependencies(toyc-ch4 ToyCh4CombineIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

target_link_libraries(toyc-ch4
  PRIVATE
    MLIR                        # libMLIR.dylib (all dialects, passes, conversions)
    LLVM                        # libLLVM.dylib (all targets, all components)
    )
```

Relative to Chapter 3:

- **`mlir/ShapeInferencePass.cpp`** joins the source list.
- **`add_dependencies(toyc-ch4 ToyCh4ShapeInferenceInterfaceIncGen)`** makes the build generate the interface files before compiling the sources that include them, since `ShapeInferenceInterface.h` (included from `Dialect.h`) includes a generated `.h.inc`, and `ShapeInferencePass.cpp` includes the generated `.cpp.inc`.
- Linking is still just the monolithic `MLIR` + `LLVM` dylibs, which already contain the call, cast and function interfaces and the transforms used here.

### C.2 Interface TableGen

The interface generation lives in `include/toy/CMakeLists.txt`, next to the Chapter 2-style op generation:

***include/toy/CMakeLists.txt***
```cmake
# Most dialects should use add_mlir_dialect().  See examples/standalone.
set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(ToyCh4OpsIncGen)

# Most dialects should use add_mlir_interfaces().
set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)
mlir_tablegen(ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ToyCh4ShapeInferenceInterfaceIncGen)
```

- **`-gen-op-interface-decls`** → `ShapeInferenceOpInterfaces.h.inc` — the `ShapeInference` interface class (included by `ShapeInferenceInterface.h`);
- **`-gen-op-interface-defs`** → `ShapeInferenceOpInterfaces.cpp.inc` — the out-of-line `ShapeInference::inferShapes()`, which dispatches to the op's model (section 5.1; included by `ShapeInferencePass.cpp`).

(As the file's comments note, `add_mlir_dialect()` / `add_mlir_interfaces()` would wrap all of this; this repo spells the steps out, which is more instructive.) In a standalone build the files land in `Ch4/build/include/toy/`; in the superbuild, in `toy/build/Ch4/include/toy/`. `Ops.td` also includes `ShapeInferenceInterface.td`, so the op generator sees the interface definition when it expands `DeclareOpInterfaceMethods<ShapeInferenceOpInterface>`.

---

## Appendix D: Running this README's commands: `examples.sh`

[`examples.sh`](examples.sh) — a symlink to the shared [`../examples.sh`](../examples.sh) — runs the commands of this README and shows each one with its output and exit status:

```bash
cd /Users/roy/study/mlir/toy/Ch4
./examples.sh -l
```

Real output **[32]** (abridged):

```text
[ 1] 3     ./build/toyc-ch4 ../../test_Example/Toy/Ch4/codegen.toy -emit=mlir 2>&1
[ 2] 4.1   echo "== as generated: multiply_transpose is private"
...
[31] B     cat <<'EOF' | ./build/toyc-ch4 - -x mlir -emit=mlir -opt 2>&1
[32] D     ./examples.sh -l
```

The script reads the commands from `README.md` on every run. A command is every `bash` code block in file order, except file excerpts (a `***path***` label above the fence). Number *N* is the *N*-th such block, and the README marks that block's output "Real output **[N]**". So adding, removing or reordering a command block renumbers the ones after it, and their **[N]** labels must be updated to match.

---

## Links

- Official doc: [Toy Tutorial Chapter 4 — Enabling Generic Transformation with Interfaces](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/)
- Related MLIR docs: [Interfaces](https://mlir.llvm.org/docs/Interfaces/) · [Operation Definition Specification (ODS)](https://mlir.llvm.org/docs/DefiningDialects/Operations/) · [Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/)
- Previous: [Chapter 3 — High-level Language-Specific Analysis and Transformation](../Ch3/README.md)
- Next: [Chapter 5 — Partial Lowering to Lower-Level Dialects for Optimization](../Ch5/README.md)
- Back to [README](../README.md)
