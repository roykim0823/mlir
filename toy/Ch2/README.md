# Chapter 2: Emitting Basic MLIR

> **Goal:** Define the *Toy dialect* and its operations in MLIR (mostly declaratively with ODS/TableGen), then walk the Chapter 1 AST and emit real MLIR from it — based on the official tutorial [Toy Ch-2: Emitting Basic MLIR](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/).

---

## 1. Overview

In Chapter 1 we built a classic frontend: a lexer, a parser, and an AST for the Toy language. In this chapter the compiler finally meets MLIR. We will:

1. Learn the **core MLIR concepts**: operations, values, attributes, regions/blocks, and dialects.
2. Define the **Toy dialect** — a namespace that groups all Toy-specific abstractions.
3. Define the **nine Toy operations** (`toy.constant`, `toy.add`, `toy.func`, `toy.generic_call`, `toy.mul`, `toy.print`, `toy.reshape`, `toy.return`, `toy.transpose`) using the **Operation Definition Specification (ODS)** framework, i.e. TableGen.
4. Implement **MLIRGen**, a module that walks the Toy AST and emits those operations.
5. Build and run the `toyc-ch2` binary and verify that the emitted MLIR **round-trips** (print → parse → print) through our dialect's custom parsers and printers.

Why bother? Unlike LLVM IR's fixed, low-level instruction set, MLIR lets us define a *high-level*, tensor-based Toy IR that preserves the language's semantics, so later chapters can optimize at that level (section 2.1).

The lexer, parser, and AST (`Lexer.h`, `Parser.h`, `AST.h`, `parser/AST.cpp`) are carried over unchanged from Chapter 1; everything new — the ODS definitions in `include/toy/Ops.td`, the dialect implementation in `mlir/Dialect.cpp`, the emitter in `mlir/MLIRGen.cpp`, the extended driver, is introduced in the section that explains it; the TableGen build wiring is in the appendix. (For the repo-wide layout and build setup, see the top-level [README](../README.md#repository-layout).)

### How this README maps to the upstream chapter

Every topic of the official [Ch-2](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/) is covered here, in the same order, with each op feature shown through this repo's real code and generated output:

| Upstream section | Here |
|---|---|
| Introduction: Multi-Level Intermediate Representation | 2.1 |
| Interfacing with MLIR (dialects, operation anatomy, core concepts, locations) | 2.2, with real output in 2.3 |
| Opaque API | 2.4 |
| Defining a Toy Dialect | 3.1–3.3 |
| Defining Toy Operations (hand-written `ConstantOp`) | 4.1 |
| Op vs Operation: Using MLIR Operations | 4.2 |
| Using the ODS Framework | 4.3 |
| Defining Arguments and Results · Adding Documentation · Verifying Operation Semantics · Attaching `build` Methods | 4.4, steps 1–4 |
| Specifying a Custom Assembly Format (`toy.print`, directives/literals/variables) | 4.5 |
| Complete Toy Example (emit, round trip, `mlir-tblgen`) | 6.2–6.5, appendix A.3 |

Beyond upstream: all nine ops at a glance (4.6) and the other eight in detail (4.7–4.13), the generated C++ (4.14), the MLIRGen walkthrough (section 5), and the build wiring (appendix).

---

## 2. MLIR Core Concepts

### 2.1 Why a multi-level IR

Compilers like LLVM (see the [Kaleidoscope tutorial](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html)) offer a *fixed* set of predefined types and (usually low-level, RISC-like) instructions. Everything language-specific — type checking, analysis, transformation — must happen in the frontend *before* emitting LLVM IR. Clang, for example, uses its AST not only for static analysis but also for transformations such as C++ template instantiation (by cloning and rewriting AST subtrees). Languages with higher-level constructs than C/C++ may need a non-trivial lowering from their AST to LLVM IR.

The consequence: every frontend reimplements significant infrastructure for these analyses and transformations. And once `transpose(a) * transpose(b)` has been lowered to LLVM IR, the fact that these were tensor transposes is gone — an optimization like `transpose(transpose(x)) = x` can no longer be written.

MLIR addresses this by being designed for **extensibility**: it has very few predefined instructions (*operations*, in MLIR terminology) or types. You define your own operations, types, and attributes at whatever level of abstraction fits the problem, and reuse shared infrastructure for analyses, transformations, location tracking, and multithreaded compilation. Toy's dialect is a *high-level*, tensor-based IR that preserves language semantics, so the next chapters can do meaningful optimizations on it.

### 2.2 Interfacing with MLIR: dialects, operations, locations

A **dialect** groups operations, attributes, and types under a unique namespace — think of it as a C++ namespace plus a registration point for parsing/printing/verification hooks. MLIR ships with many (e.g. `builtin`, `func`, `arith`, `tensor`, `llvm`), and they coexist in one module: that is the "multi-level" in Multi-Level IR — you lower *gradually* from `toy` to lower-level dialects rather than jumping straight to LLVM IR. Section 3 defines the `toy` dialect.

In MLIR there is no closed set of attributes (think: constant metadata), operations, or types, and no hard-coded notion of "instruction", "function", or "module". The core unit of abstraction and computation is the **operation** — similar in many ways to an LLVM instruction, but it can have application-specific semantics and can represent *all* of LLVM's core IR structures: instructions, globals (like functions), modules, etc. Functions are operations, modules are operations, `if` statements are operations, and every generic piece of infrastructure (printing, parsing, verification, pass management, location tracking) works on all of them.

Dissect a Toy operation in its *generic* textual form:

```mlir
%t_tensor = "toy.transpose"(%tensor) {inplace = true} : (tensor<2x3xf64>) -> tensor<3x2xf64> loc("example/file/path":12:1)
```

| Piece | Meaning |
|---|---|
| `%t_tensor` | The name of the result defined by this operation. The `%` is a sigil that keeps value names from colliding with other identifiers. An operation defines zero or more results, which are SSA **values** (Toy limits itself to single-result ops). An op with several results names them together (`%res:2 = ...`) and uses them individually as `%res#0`, `%res#1`. The name is used during parsing but is **not persistent**: it is not tracked in the in-memory representation (section 2.4 shows `mlir-opt` renaming it). |
| `"toy.transpose"` | The **operation name**: a unique string, the **dialect** namespace followed by `.` and the mnemonic — read it as "the `transpose` operation in the `toy` dialect". |
| `(%tensor)` | A list of zero or more **operands** (arguments): SSA values defined by other operations or referring to block arguments. |
| `{inplace = true}` | A dictionary of zero or more **attributes**: special operands that are always *constant*. Here, a boolean attribute named `inplace` with the value `true`. Attributes are how MLIR attaches data where a runtime variable is never allowed. |
| `(tensor<2x3xf64>) -> tensor<3x2xf64>` | The type of the operation in functional form: the argument types in parentheses, then the result types. |
| `loc("example/file/path":12:1)` | The **source location** this operation originated from. |

Operations are thus modeled by a small, fixed set of concepts, which is what lets MLIR reason about and manipulate *any* operation generically:

1. A **name**.
2. A list of SSA **operand** values.
3. A list of [**attributes**](https://mlir.llvm.org/docs/LangRef/#attributes).
4. A list of [**types**](https://mlir.llvm.org/docs/LangRef/#type-system) for the result values.
5. A [**source location**](https://mlir.llvm.org/docs/Diagnostics/#source-locations) for debugging.
6. A list of **successor** [blocks](https://mlir.llvm.org/docs/LangRef/#blocks) — targets for branch-like terminators (Toy doesn't need these yet).
7. A list of [**regions**](https://mlir.llvm.org/docs/LangRef/#regions) — attached bodies for structural operations like functions. A region contains **blocks**; each block holds an ordered list of operations and may have **block arguments** (MLIR's functional-style replacement for PHI nodes). Our `toy.func` has one region: the function body.

So the structural recursion is **operation → regions → blocks → operations → …**, and that single recursion expresses modules, functions, loops, and straight-line code alike.

**Locations are mandatory.** In LLVM IR, debug locations are metadata that passes may drop. In MLIR, **every operation has a mandatory source location**; it is a core requirement that APIs depend on and manipulate. Dropping a location is therefore an explicit choice that cannot happen by mistake: if a transformation replaces an operation with another, the new operation must still get a location, so you can always track where it came from. When there is no meaningful one, you must say so explicitly (`loc(unknown)` / `builder.getUnknownLoc()`).

Tools don't *print* locations by default (locations are always stored); `-mlir-print-debuginfo` turns them on — both in `mlir-opt` (see `mlir-opt --help` for more printing options) and in `toyc-ch2`. Section 2.3 shows them in real output; they even survive a print → parse round trip (section 6.5).

### 2.3 Dialects, operations and locations in real output

Three runs of `toyc-ch2` make the concepts above concrete. Each shows the code that produces the output, the exact command, and the real output. The build is in section 6.1.

#### Several dialects in one module, every op with a location

The input is this repo's `codegen.toy` (section 6.3 shows it whole):

***codegen.toy***
```text
# User defined generic function that operates on unknown shaped arguments.
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}
```

MLIRGen creates the outer module op (with no source location) and converts every Toy AST location into an MLIR `FileLineColLoc` (section 5.1):

***mlir/MLIRGen.cpp***
```cpp
mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
  // We create an empty MLIR module and codegen functions one at a time and
  // add them to the module.
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  ...
/// Helper conversion for a Toy AST location to an MLIR location.
mlir::Location loc(const Location &loc) {
  return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file), loc.line,
                                   loc.col);
}
```

Print the result in generic form, with locations:

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 codegen.toy -emit=mlir -mlir-print-op-generic -mlir-print-debuginfo 2>&1
```

Real output (abridged; section 4.5 shows all of it):

```mlir
"builtin.module"() ({
  "toy.func"() <{function_type = (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>, sym_name = "multiply_transpose"}> ({
  ^bb0(%arg0: tensor<*xf64> loc("codegen.toy":2:1), %arg1: tensor<*xf64> loc("codegen.toy":2:1)):
    %6 = "toy.transpose"(%arg0) : (tensor<*xf64>) -> tensor<*xf64> loc("codegen.toy":3:10)
    %7 = "toy.transpose"(%arg1) : (tensor<*xf64>) -> tensor<*xf64> loc("codegen.toy":3:25)
  ...
}) : () -> () loc(unknown)
```

- **Two dialects in one module**: the outermost op is `builtin.module` from the `builtin` dialect; it holds `toy.func` and `toy.transpose` ops from our dialect. Without `-mlir-print-op-generic` the same module prints as `module { toy.func @multiply_transpose(...) ... }` — the `builtin.` prefix is elided (section 6.4).
- **The anatomy of section 2.2**, on real ops: `%6` is the result, `"toy.transpose"` the name, `(%arg0)` the operand, `(tensor<*xf64>) -> tensor<*xf64>` the functional type. `toy.func` also shows *properties* in `<Ellipsis>` and a region `(Ellipsis)` whose entry block `^bb0` has two block arguments.
- **Locations everywhere**: `loc("codegen.toy":3:10)` is line 3, column 10 of the input — the `transpose(a)` call. Block arguments carry locations too. The module's `loc(unknown)` comes from `builder.getUnknownLoc()` above — the explicit "no location" choice.

#### A tool only understands the dialects it has loaded

`toyc-ch2` loads the `toy` dialect into its context before doing anything else (section 3.3):

***toyc.cpp***
```cpp
mlir::MLIRContext context;
// Load our Dialect in this MLIR Context.
context.getOrLoadDialect<mlir::toy::ToyDialect>();
```

Stock `mlir-opt` has no such line for `toy`, so it cannot parse our (custom-syntax) output:

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 codegen.toy -emit=mlir 2>&1 | /opt/homebrew/opt/llvm@20/bin/mlir-opt -allow-unregistered-dialect
```

Real output:

```text
<stdin>:2:3: error: Dialect `toy' not found for custom op 'toy.func'
  toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
  ^
```

`-allow-unregistered-dialect` doesn't help: it only covers the generic form, which `mlir-opt` can read without knowing the dialect (section 2.4, and section 6.7 for the generic-form round trip).

#### A registered dialect verifies its ops

The test input breaks three rules of `toy.print`:

***test_Example/Toy/Ch2/invalid.mlir***
```mlir
// RUN: not toyc-ch2 %s -emit=mlir 2>&1

// The following IR is not "valid":
// - toy.print should not return a value.
// - toy.print should take an argument.
// - There should be a block terminator.
toy.func @main() {
  %0 = "toy.print"()  : () -> tensor<2x3xf64>
}
```

`PrintOp` declares no results in `Ops.td` (section 4.10), so ODS gives the generated class the `ZeroResults` trait, whose verifier rejects any result:

***build/op-decls.inc***
```cpp
class PrintOp : public ::mlir::Op<PrintOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants> {
```

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 ../../test_Example/Toy/Ch2/invalid.mlir -emit=mlir
```

Real output (exit status 3):

```text
loc("../../test_Example/Toy/Ch2/invalid.mlir":8:8): error: 'toy.print' op requires zero results
Error can't load file ../../test_Example/Toy/Ch2/invalid.mlir
```

The diagnostic carries the location of the offending op (line 8, column 8). Verification stops at the first failure, so the other two problems are not reported. Section 2.4 feeds the same IR, with `func.func` instead of `toy.func`, to `mlir-opt`, which accepts it because `toy` is not registered there.

### 2.4 The opaque API: MLIR works even on ops it has never heard of

MLIR lets every IR element — attributes, operations, types — be customized, yet any of them can always be reduced to the fundamental concepts of section 2.2. That is what lets MLIR parse, represent, and [round-trip](https://mlir.llvm.org/getting_started/Glossary/#round-trip) IR for *any* operation, even one from a dialect nobody registered. Two runs of the stock Homebrew `mlir-opt`, which knows nothing about `toy`, show this.

#### An unregistered op round-trips

The input is the upstream tutorial's example, not a file in this repo, so the commands pass it on stdin: the `toy.transpose` from section 2.2 inside a `func.func`. By default `mlir-opt` refuses unregistered dialects:

```bash
cd /Users/roy/study/mlir/toy/Ch2
/opt/homebrew/opt/llvm@20/bin/mlir-opt <<'EOF'
func.func @toy_func(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> {
  %t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
  return %t_tensor : tensor<3x2xf64>
}
EOF
```

Real output (exit status 1):

```text
<stdin>:2:30: error: operation being parsed with an unregistered dialect. If this is intended, please use -allow-unregistered-dialect with the MLIR tool used
  %t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
                             ^
```

With `-allow-unregistered-dialect`, it parses and prints the IR back:

```bash
cd /Users/roy/study/mlir/toy/Ch2
/opt/homebrew/opt/llvm@20/bin/mlir-opt -allow-unregistered-dialect <<'EOF'
func.func @toy_func(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> {
  %t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
  return %t_tensor : tensor<3x2xf64>
}
EOF
```

Real output:

```mlir
module {
  func.func @toy_func(%arg0: tensor<2x3xf64>) -> tensor<3x2xf64> {
    %0 = "toy.transpose"(%arg0) {inplace = true} : (tensor<2x3xf64>) -> tensor<3x2xf64>
    return %0 : tensor<3x2xf64>
  }
}
```

Note that `%tensor`/`%t_tensor` came back as `%arg0`/`%0` — the value names were never part of the in-memory IR. The op itself survives, but only as structure:

For unregistered attributes, operations, and types, MLIR only enforces *structural* constraints (e.g. dominance); otherwise they are completely **opaque**. MLIR has no idea whether an unregistered op can operate on particular data types, how many operands it takes, or how many results it produces. That flexibility is handy for bootstrapping, but advised against in mature systems: transformations and analyses must treat unregistered ops conservatively, and they are much harder to construct and manipulate.

#### Invalid IR passes when the dialect is unregistered

The test input `test_Example/Toy/Ch2/invalid.mlir` (shown in section 2.3) holds IR that is *invalid* for Toy. Section 2.3 feeds it to `toyc-ch2`, which rejects it. Here the same IR goes to stock `mlir-opt`, with `toy.func` swapped for the builtin `func.func` so that only the `toy.print` op is unregistered:

```bash
cd /Users/roy/study/mlir/toy/Ch2
sed 's/toy\.func/func.func/' ../../test_Example/Toy/Ch2/invalid.mlir \
  | /opt/homebrew/opt/llvm@20/bin/mlir-opt -allow-unregistered-dialect
```

Real output (exit status 0 — no verifier error):

```mlir
module {
  func.func @main() {
    %0 = "toy.print"() : () -> tensor<2x3xf64>
  }
}
```

There are three problems here, none of which MLIR can see:

1. `toy.print` is not a terminator — yet the block has no other terminator. MLIR accepts this because, for all it knows, an unregistered op *could* be a terminator.
2. `toy.print` should take an operand.
3. `toy.print` shouldn't return any values.

**Registering** the dialect and its operations fixes all of that: the verifier learns each op's invariants (the same IR as a `toy.func` is rejected with `'toy.print' op requires zero results` — section 2.3), and you get typed accessor methods instead of stringly-typed attribute lookups, a custom (pretty) assembly syntax, and hooks for optimization. That is what the rest of this chapter builds.

---

## 3. Defining the Toy Dialect

To interface with MLIR effectively, we define a new `toy` dialect. It models the structure of the Toy language and gives us an easy avenue for high-level analysis and transformation.

### 3.1 What it would look like in raw C++

A dialect is a class deriving from `mlir::Dialect` that registers custom attributes, operations, and types; it can also override virtual methods to change general behavior (later chapters do this). Written by hand, as in the upstream tutorial:

```cpp
/// This is the definition of the Toy dialect. A dialect inherits from
/// mlir::Dialect and registers custom attributes, operations, and types. It can
/// also override virtual methods to change some general behavior, which will be
/// demonstrated in later chapters of the tutorial.
class ToyDialect : public mlir::Dialect {
public:
  explicit ToyDialect(mlir::MLIRContext *ctx);

  /// Provide a utility accessor to the dialect namespace.
  static llvm::StringRef getDialectNamespace() { return "toy"; }

  /// An initializer called from the constructor of ToyDialect that is used to
  /// register attributes, operations, types, and more within the Toy dialect.
  void initialize();
};
```

### 3.2 What this repo actually does: ODS

MLIR also supports defining dialects *declaratively* in [TableGen](https://llvm.org/docs/TableGen/ProgRef.html). That is much cleaner — it removes most of the boilerplate — and lets documentation live right next to the definition. This is the very top of the repo's file:

***include/toy/Ops.td***
```tablegen
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "::mlir::toy";
}
```

The `Dialect` record has more fields than the two used here. The upstream tutorial's version shows the full set:

| Field | Meaning | Upstream value | This repo |
|---|---|---|---|
| `name` | The namespace; corresponds 1-1 with `ToyDialect::getDialectNamespace()` | `"toy"` | `"toy"` |
| `summary` | A short one-line summary, used in generated docs | `"A high-level dialect for analyzing and optimizing the Toy language"` | not set |
| `description` | A longer description, used in generated docs | a paragraph about the tensor-based Toy language | not set |
| `cppNamespace` | The C++ namespace the generated class lives in | `"toy"` | `"::mlir::toy"` — hence `mlir::toy::ToyDialect` everywhere in this repo's C++ |

Running `mlir-tblgen -gen-dialect-decls` on this yields exactly the boilerplate class of section 3.1. Here is the **actual generated code** from `Ch2/build/dialect-decls.inc` in this repo (produced by `run_mlir-tblgen.sh`, see the [appendix](#a3-inspecting-tablegen-output-by-hand)):

***build/dialect-decls.inc***
```cpp
namespace mlir {
namespace toy {

class ToyDialect : public ::mlir::Dialect {
  explicit ToyDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~ToyDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("toy");
  }
};
} // namespace toy
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::toy::ToyDialect)
```

and `-gen-dialect-defs` produces the constructor/destructor (the odd whitespace is verbatim generator output):

***build/dialect-defs.inc***
```cpp
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::toy::ToyDialect)
namespace mlir {
namespace toy {

ToyDialect::ToyDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<ToyDialect>())

     {

  initialize();
}

ToyDialect::~ToyDialect() = default;

} // namespace toy
} // namespace mlir
```

Compared with the hand-written class, the generated one makes the constructor private and befriends `MLIRContext` (only a context may create a dialect), adds a destructor, and registers a `TypeID` for the dialect.

The only thing left for us to write by hand is `initialize()`, which registers the operations. `Dialect.cpp` first includes the generated constructor/destructor above, then defines it:

***mlir/Dialect.cpp***
```cpp
#include "toy/Dialect.cpp.inc"
...
/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
      >();
}
```

With a single hand-written op this would simply be `addOperations<ConstantOp>();` (the upstream tutorial's example). With ODS, `toy/Ops.cpp.inc` expands to the complete op list `ConstantOp, AddOp, FuncOp, ...`. `GET_OP_LIST` is a guard macro: the same generated file `Ops.cpp.inc` contains both a comma-separated list of all op classes and their full method definitions; the macro selects which slice gets included.

### 3.3 Header wiring and loading the dialect

`Dialect.h` is nothing but glue — it includes interface headers the generated code needs, then the two generated declaration files:

***include/toy/Dialect.h***
```cpp
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

/// Include the auto-generated header file containing the declaration of the toy
/// dialect.
#include "toy/Dialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// toy operations.
#define GET_OP_CLASSES
#include "toy/Ops.h.inc"
```

Finally, a dialect must be **loaded into the `MLIRContext`** before use: by default a context loads only the [builtin dialect](https://mlir.llvm.org/docs/Dialects/Builtin/) (which provides a few core IR components such as `builtin.module`), so every other dialect must be loaded explicitly. The driver does this first thing in `dumpMLIR()`:

***toyc.cpp***
```cpp
mlir::MLIRContext context;
// Load our Dialect in this MLIR Context.
context.getOrLoadDialect<mlir::toy::ToyDialect>();
```

The upstream text writes `context.loadDialect<ToyDialect>()`; the two are interchangeable here. `loadDialect<Ts...>()` loads one or more dialects and returns nothing, while `getOrLoadDialect<T>()` loads a single dialect (if not already loaded) and returns a `T *` to it. Without either line, the context can't parse `toy.*` ops — the same failure stock `mlir-opt` shows in section 6.7: ``Dialect `toy' not found for custom op 'toy.func'``.

---

## 4. Defining Toy Operations with ODS

With a dialect in place we can define operations, which is how we give the rest of the system semantic information to hook into. This section follows the upstream tutorial's path with one op, `toy.constant`: first written by hand in C++ (4.1–4.2), then declaratively with ODS, one feature at a time (4.3–4.4), then with a custom assembly format (4.5). After that, 4.6 gives all nine ops at a glance, and 4.7–4.13 cover the remaining eight one by one.

### 4.1 The C++ way: a hand-written `ConstantOp`

`toy.constant` represents a constant value in the Toy language. In generic form:

```mlir
%4 = "toy.constant"() {value = dense<1.0> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
```

It takes **zero operands**, a [dense elements](https://mlir.llvm.org/docs/Dialects/Builtin/#denseintorfpelementsattr) **attribute** named `value` holding the constant, and returns a **single result** of [RankedTensorType](https://mlir.llvm.org/docs/Dialects/Builtin/#rankedtensortype). (That line still parses today. MLIR 20 prints it back as `%0 = "toy.constant"() <{value = dense<1.000000e+00> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>`: `value` is now stored as an op *property* — an inherent attribute, printed in `<{...}>` — rather than in the discardable-attribute dictionary `{...}`.)

An operation class inherits from the [CRTP](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern) template `mlir::Op`, which also takes optional [**traits**](https://mlir.llvm.org/docs/Traits/). Traits inject extra behavior into an operation — accessors, verification, and more. A possible hand-written definition, from the upstream tutorial:

```cpp
class ConstantOp : public mlir::Op<
                     /// `mlir::Op` is a CRTP class, meaning that we provide the
                     /// derived class as a template parameter.
                     ConstantOp,
                     /// The ConstantOp takes zero input operands.
                     mlir::OpTrait::ZeroOperands,
                     /// The ConstantOp returns a single result.
                     mlir::OpTrait::OneResult,
                     /// We also provide a utility `getType` accessor that
                     /// returns the TensorType of the single result.
                     mlir::OpTrait::OneTypedResult<TensorType>::Impl> {

 public:
  /// Inherit the constructors from the base Op class.
  using Op::Op;

  /// Provide the unique name for this operation. MLIR will use this to register
  /// the operation and uniquely identify it throughout the system. The name
  /// provided here must be prefixed by the parent dialect namespace followed
  /// by a `.`.
  static llvm::StringRef getOperationName() { return "toy.constant"; }

  /// Return the value of the constant by fetching it from the attribute.
  mlir::DenseElementsAttr getValue();

  /// Operations may provide additional verification beyond what the attached
  /// traits provide.  Here we will ensure that the specific invariants of the
  /// constant operation are upheld, for example the result type must be
  /// of TensorType and matches the type of the constant `value`.
  LogicalResult verifyInvariants();

  /// Provide an interface to build this operation from a set of input values.
  /// This interface is used by the `builder` classes to allow for easily
  /// generating instances of this operation:
  ///   mlir::OpBuilder::create<ConstantOp>(...)
  /// This method populates the given `state` that MLIR uses to create
  /// operations. This state is a collection of all of the discrete elements
  /// that an operation may contain.
  /// Build a constant with the given return type and `value` attribute.
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Type result, mlir::DenseElementsAttr value);
  /// Build a constant and reuse the type from the given 'value'.
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::DenseElementsAttr value);
  /// Build a constant by broadcasting the given 'value'.
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    double value);
};
```

It would then be registered in the dialect initializer with `addOperations<ConstantOp>();` (section 3.2). The class has five ingredients — a **name**, **traits** for the operand/result structure, an **accessor**, a **verifier**, and **builders**. Section 4.4 rebuilds each of them in ODS.

### 4.2 `Operation` vs. `Op`

Now that we have an op, we want to access and transform it. MLIR has two main classes for that, and they are easy to confuse:

- `mlir::Operation` models *all* operations generically. It is **opaque**: it describes no particular operation, but provides a general API into any operation instance (`getOperands()`, `getAttrs()`, `getLoc()`, …).
- Each specific kind of operation is an `mlir::Op`-derived class (e.g. our `ConstantOp`, "zero inputs, one output, always the same value"). It is a thin **smart-pointer wrapper around an `Operation*`** that adds operation-specific accessors and type-safe properties.

So defining a Toy op means defining a clean, semantically useful *interface* for building and inspecting an `Operation`. That is why `ConstantOp` declares **no data fields**: all of its data lives in the referenced `Operation`. A consequence is that `Op` classes are always passed **by value**, not by reference or pointer — a common MLIR idiom that applies equally to attributes, types, etc.

Given a generic `Operation*`, you get the specific `Op` back with LLVM's casting infrastructure (upstream tutorial example):

```cpp
void processConstantOp(mlir::Operation *operation) {
  ConstantOp op = llvm::dyn_cast<ConstantOp>(operation);

  // This operation is not an instance of `ConstantOp`.
  if (!op)
    return;

  // Get the internal operation instance wrapped by the smart pointer.
  mlir::Operation *internalOperation = op.getOperation();
  assert(internalOperation == operation &&
         "these operation instances are the same");
}
```

### 4.3 ODS: the declarative way

Instead of specializing `mlir::Op` by hand, you can describe an operation in the [Operation Definition Specification (ODS)](https://mlir.llvm.org/docs/DefiningDialects/Operations/) framework: the facts about an op go into a TableGen record, which `mlir-tblgen` expands into the equivalent `mlir::Op` C++ specialization at build time. ODS is *the* recommended way to define ops — it is simpler, more concise, and stays stable when MLIR's C++ APIs change (the hand-written class above would drift; the generated one is always current).

Operations in ODS inherit from the `Op` class in `OpBase.td`. To keep definitions short, all Toy ops share a base class that fixes the parent dialect:

***include/toy/Ops.td***
```tablegen
// Base class for toy dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;
```

A Toy op then inherits from `Toy_Op`, giving the **mnemonic** — the op name *without* the `toy.` prefix, matching `ConstantOp::getOperationName()` above — and an optional trait list. The starting point is just the shell:

```tablegen
def ConstantOp : Toy_Op<"constant"> {
}
```

Notice that the `ZeroOperands` and `OneResult` traits from the C++ version are *missing*: ODS infers them from the `arguments` and `results` fields (section 4.14 shows the inferred trait list of the generated `TransposeOp` class). To see what TableGen generates at any point, run `mlir-tblgen` with `-gen-op-decls` (class declarations) or `-gen-op-defs` (implementations); this repo's `run_mlir-tblgen.sh` does exactly that (see the [appendix](#a3-inspecting-tablegen-output-by-hand)). Comparing the output with the hand-written class is the best way to get comfortable with TableGen.

### 4.4 `ConstantOp` step by step

Here is the complete `ConstantOp` as it stands in this repo. The rest of this section adds its fields one at a time, in the upstream tutorial's order, and shows what each one generates:

***include/toy/Ops.td***
```tablegen
def ConstantOp : Toy_Op<"constant", [Pure]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:
    ...
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<(ins "double":$value)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}
```

#### Step 1: Arguments and results

The [arguments](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-arguments) (inputs) of an op may be **attributes** or **types of SSA operand values**; the [results](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-results) are the types of the values it produces:

- `let arguments = (ins F64ElementsAttr:$value);` — an *attribute* (`F64ElementsAttr`: a 64-bit float `ElementsAttr`) sits in `ins` right where an operand would go; ODS tells them apart by the constraint kind.
- `let results = (outs F64Tensor);` — `F64Tensor` is a predefined constraint, "tensor of 64-bit float". Because it's unnamed, only generic result accessors are produced.

Naming an argument (`$value`) makes ODS generate matching accessors. The upstream text calls the accessor `ConstantOp::value()`; that is the old naming scheme — MLIR now prefixes accessors with `get`/`set`. This is the real generated code:

***build/op-decls.inc***
```cpp
  ::mlir::DenseElementsAttr getValueAttr() {
    return ::llvm::cast<::mlir::DenseElementsAttr>(getProperties().value);
  }

  ::mlir::DenseElementsAttr getValue();
  void setValueAttr(::mlir::DenseElementsAttr attr) {
    getProperties().value = attr;
  }
```

`getValueAttr()` returns the attribute stored in the op's properties, and `setValueAttr()` replaces it. `getValue()` returns the attribute's *value*; for `F64ElementsAttr` that is the attribute itself, so its generated body in `op-defs.inc` is just `auto attr = getValueAttr(); return attr;`.

#### Step 2: Documentation

Ops may provide [`summary` and `description`](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-documentation) fields describing their semantics. They are useful for dialect users and feed auto-generated Markdown. Running `mlir-tblgen -gen-op-doc` on this repo's `Ops.td` renders them; for `toy.add`, the real output starts:

```text
### `toy.add` (toy::AddOp)

_Element-wise addition operation_

The "add" operation performs element-wise addition between two tensors.
The shapes of the tensor operands are expected to match.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lhs` | tensor of 64-bit float values
| `rhs` | tensor of 64-bit float values
```

#### Step 3: Verifying operation semantics

Much like the named accessors, ODS generates most verification logic from the constraints you gave: we don't need to verify the structure of the result type, or even the input attribute `value`. Often no extra verification is needed at all. When it is, `let hasVerifier = 1;` declares a `::llvm::LogicalResult verify()` method on the op class that runs **after** the ODS-generated checks (the upstream prose mentions an older `verifier` code-blob field; `hasVerifier` replaced it). We implement it in `Dialect.cpp`; it can assume every other invariant already holds, and checks that the result tensor's shape matches the attribute's shape:

***mlir/Dialect.cpp***
```cpp
/// Verifier for the constant operation. This corresponds to the
/// `let hasVerifier = 1` in the op definition.
llvm::LogicalResult ConstantOp::verify() {
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType = llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!resultType)
    return success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
  if (attrType.getRank() != resultType.getRank()) {
    return emitOpError("return type must match the one of the attached value "
                       "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}
```

Section 4.14 shows the generated `verifyInvariants()` that chains the two in that order.

#### Step 4: Attaching `build` methods

The last piece of the C++ version is its `build` methods — what `builder.create<ConstantOp>(loc, ...)` calls to populate the `OperationState` (the collection of all the elements an operation may contain) that MLIR uses to create the op. ODS generates simple ones automatically; the [`builders`](https://mlir.llvm.org/docs/DefiningDialects/Operations/#custom-builder-methods) field adds more. Each `OpBuilder` takes a list of C++ parameters `(ins ...)` and an optional inline body:

- The first custom builder, `(ins "DenseElementsAttr":$value)`, has an **inline body** (the `[{ ... }]` blob) that calls an autogenerated `build` with the attribute's own type. Inside the blob, `$_builder` and `$_state` are placeholders for the builder and the state being filled (the upstream text's `build(builder, result, ...)` is the older spelling).
- The second, `(ins "double":$value)`, has **no body**: ODS only *declares* `ConstantOp::build(OpBuilder &, OperationState &, double)`, and we implement it in the dialect:

***mlir/Dialect.cpp***
```cpp
/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}
```

The generated class ends up with five `build` overloads — our two custom ones, plus three that ODS generated from `arguments`/`results` (explicit result type + attribute; a result-type range + attribute; and the fully generic types/operands/attributes form):

***build/op-decls.inc***
```cpp
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, DenseElementsAttr value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, double value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::DenseElementsAttr value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::DenseElementsAttr value);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
```

#### Step 5 (not in the upstream chapter): the `Pure` trait

This repo's `ConstantOp` also lists the `Pure` trait: no side effects, so an unused constant can be deleted and duplicates merged. That is what the generated `getEffects()` declaration above reports to the rest of MLIR. It pays off in Chapter 3, where canonicalization and CSE start deleting and merging ops. (`Pure` replaces what older MLIR called `NoSideEffect`.)

The final field, `hasCustomAssemblyFormat`, is the subject of the next section.

### 4.5 Custom assembly formats

#### Generic vs. custom syntax

With the ops defined so far we can already generate "Toy IR", but every op prints in the **generic** assembly format — the one dissected in section 2.2. This is the real output for `codegen.toy` when the custom formats are bypassed with `-mlir-print-op-generic` (plus `-mlir-print-debuginfo`):

```mlir
"builtin.module"() ({
  "toy.func"() <{function_type = (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>, sym_name = "multiply_transpose"}> ({
  ^bb0(%arg0: tensor<*xf64> loc("codegen.toy":2:1), %arg1: tensor<*xf64> loc("codegen.toy":2:1)):
    %6 = "toy.transpose"(%arg0) : (tensor<*xf64>) -> tensor<*xf64> loc("codegen.toy":3:10)
    %7 = "toy.transpose"(%arg1) : (tensor<*xf64>) -> tensor<*xf64> loc("codegen.toy":3:25)
    %8 = "toy.mul"(%6, %7) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64> loc("codegen.toy":3:25)
    "toy.return"(%8) : (tensor<*xf64>) -> () loc("codegen.toy":3:3)
  }) : () -> () loc("codegen.toy":2:1)
  "toy.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "toy.constant"() <{value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>}> : () -> tensor<2x3xf64> loc("codegen.toy":7:17)
    %1 = "toy.reshape"(%0) : (tensor<2x3xf64>) -> tensor<2x3xf64> loc("codegen.toy":7:3)
    %2 = "toy.constant"() <{value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>}> : () -> tensor<6xf64> loc("codegen.toy":8:17)
    %3 = "toy.reshape"(%2) : (tensor<6xf64>) -> tensor<2x3xf64> loc("codegen.toy":8:3)
    %4 = "toy.generic_call"(%1, %3) <{callee = @multiply_transpose}> : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("codegen.toy":9:11)
    %5 = "toy.generic_call"(%3, %1) <{callee = @multiply_transpose}> : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("codegen.toy":10:11)
    "toy.print"(%5) : (tensor<*xf64>) -> () loc("codegen.toy":11:3)
    "toy.return"() : () -> () loc("codegen.toy":6:1)
  }) : () -> () loc("codegen.toy":6:1)
}) : () -> () loc(unknown)
```

Two differences from the upstream chapter's (older) dump: inherent attributes print as properties in `<{...}>`, and the function signature attribute is now `function_type` rather than `type`. Compare this with the custom-format output of section 6.4 — same IR, much less fluff. MLIR lets each op define its own format, either **imperatively in C++** (`let hasCustomAssemblyFormat = 1;`, then write `parse()`/`print()`) or **declaratively** (`let assemblyFormat = "...";`), which ODS compiles into those same C++ methods internally.

#### Walkthrough: `toy.print`

The generic `"toy.print"(%5) : (tensor<*xf64>) -> ()` is verbose. Stripped to the essentials, a good format is:

```mlir
toy.print %5 : tensor<*xf64> loc(...)
```

**The C++ way.** Set `hasCustomAssemblyFormat` to divert printing and parsing to `print`/`parse` methods (upstream tutorial example — a stripped-down `PrintOp`, not this repo's):

```tablegen
/// Consider a stripped definition of `toy.print` here.
def PrintOp : Toy_Op<"print"> {
  let arguments = (ins F64Tensor:$input);

  // Divert the printer and parser to `parse` and `print` methods on our operation,
  // to be implemented in the .cpp file. More details on these methods is shown below.
  let hasCustomAssemblyFormat = 1;
}
```

```cpp
/// The 'OpAsmPrinter' class is a stream that will allows for formatting
/// strings, attributes, operands, types, etc.
void PrintOp::print(mlir::OpAsmPrinter &printer) {
  printer << "toy.print " << op.input();
  printer.printOptionalAttrDict(op.getAttrs());
  printer << " : " << op.input().getType();
}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
mlir::ParseResult PrintOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  // Parse the input operand, the attribute dictionary, and the type of the
  // input.
  mlir::OpAsmParser::UnresolvedOperand inputOperand;
  mlir::Type inputType;
  if (parser.parseOperand(inputOperand) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(inputType))
    return mlir::failure();

  // Resolve the input operand to the type we parsed in.
  if (parser.resolveOperand(inputOperand, inputType, result.operands))
    return mlir::failure();

  return mlir::success();
}
```

(This snippet is illustrative and slightly dated: it uses the old `op.input()` accessor naming, and today's printer already emits the op name itself, so a real `print()` starts with `printer << " "` — compare this repo's `ConstantOp::print` below.) The parser shows the general pattern: parse each piece in order, chaining with `||` because a `ParseResult` converts to `true` on failure; then *resolve* the parsed operand name against its type to get an SSA value.

**The declarative way.** The [declarative format](https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format) is built from three kinds of components:

| Component | What it is | In `toy.print` |
|---|---|---|
| **Directives** | Builtin functions, with optional arguments | `attr-dict`, `type($input)` |
| **Literals** | A keyword or punctuation, surrounded by backticks | `` `:` `` |
| **Variables** | An entity registered on the op itself: an argument (attribute or operand), result, successor, … | `$input` |

A direct mapping of the C++ format is the one-liner this repo actually uses (the full definition is in section 4.10):

***include/toy/Ops.td***
```tablegen
let assemblyFormat = "$input attr-dict `:` type($input)";
```

Every piece of the declarative format used anywhere in this repo's `Ops.td`:

| Piece | Kind | Meaning | Used by |
|---|---|---|---|
| `$name` | variable | print/parse that operand or attribute | all declarative ops |
| `attr-dict` | directive | the remaining (non-elided) attributes, if any; mandatory in every format | all declarative ops |
| `type($x)` | directive | the type of operand `$x` | `print`, `reshape`, `transpose`, `return` |
| `type(results)` | directive | the result type(s) | `reshape`, `transpose` |
| `functional-type($ins, results)` | directive | `(operand types) -> result types` | `generic_call` |
| `` `(` ``, `` `:` ``, `` `to` `` | literal | punctuation or a keyword, printed as-is | `generic_call`, `reshape`, `transpose` |
| `( ... )?` with `$x^` | optional group | print/parse the group only if the anchor `$x` is present | `return` |

The declarative format has many more features, so check it before writing a format in C++. Reach for C++ only when the syntax isn't regular — this repo does so twice: the binary ops' *conditional* type syntax (section 4.7) and `toy.constant`, below.

#### `toy.constant`'s C++ format

`ConstantOp` sets `hasCustomAssemblyFormat = 1` to print `%0 = toy.constant dense<...> : tensor<2x3xf64>` instead of the generic form. ODS declares `parse()`/`print()`, and `Dialect.cpp` implements them:

***mlir/Dialect.cpp***
```cpp
/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

/// The 'OpAsmPrinter' class is a stream that allows for formatting
/// strings, attributes, operands, types, etc.
void ConstantOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << getValue();
}
```

Note the trick in `parse`: the result type is not written in the pretty syntax at all — it is *recovered* from the parsed attribute's type. Custom syntax may omit anything that is reconstructible. And `print` elides `value` from the attribute dictionary because it already printed it.

### 4.6 All nine ops at a glance

| Op | Traits | Arguments (`ins`) | Results | Assembly format | Verifier | Custom builders |
|---|---|---|---|---|---|---|
| `toy.constant` (4.4–4.5) | `Pure` | `F64ElementsAttr:$value` | `F64Tensor` | C++ | yes | `(DenseElementsAttr)`, `(double)` |
| `toy.add` / `toy.mul` (4.7) | — | `F64Tensor:$lhs, F64Tensor:$rhs` | `F64Tensor` | C++ (shared) | no | `(Value, Value)` |
| `toy.func` (4.8) | `FunctionOpInterface`, `IsolatedFromAbove` | `$sym_name`, `$function_type`, optional `$arg_attrs`/`$res_attrs`; region `$body` | — | C++ (library helpers) | no | `(name, type, attrs)`; `skipDefaultBuilders` |
| `toy.generic_call` (4.9) | — | `FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs` | `F64Tensor` | declarative | no | `(callee, arguments)` |
| `toy.print` (4.10) | — | `F64Tensor:$input` | — | declarative | no | — |
| `toy.reshape` (4.11) | — | `F64Tensor:$input` | `StaticShapeTensorOf<[F64]>` | declarative | no (ODS constraint) | — |
| `toy.return` (4.12) | `Pure`, `HasParent<"FuncOp">`, `Terminator` | `Variadic<F64Tensor>:$input` | — | declarative (optional group) | yes | `()` |
| `toy.transpose` (4.13) | — | `F64Tensor:$input` | `F64Tensor` | declarative | yes | `(Value)` |

The custom builders of `add`, `mul`, `generic_call` and `transpose` always produce an **unranked** result (`tensor<*xf64>`); only `toy.constant` (typed by its literal) and `toy.reshape` (typed by the declared shape) get ranked types in this chapter. Shape inference comes in Chapter 4.

### 4.7 `AddOp` and `MulOp` — binary element-wise ops with a shared hand-written syntax

***include/toy/Ops.td***
```tablegen
def AddOp : Toy_Op<"add"> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}
```

(`MulOp` = same shape with mnemonic `"mul"`.) The declared two-`Value` builder and the `parse`/`print` hooks are implemented in `Dialect.cpp`; note that the result type is *always* an unranked `tensor<*xf64>` at this stage — shape inference is Chapter 4's job:

***mlir/Dialect.cpp***
```cpp
void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult AddOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void AddOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }
```

Both ops share one hand-written parser/printer pair, showing why you'd sometimes prefer C++ over the declarative format — *conditional* syntax. If operand and result types all match, print the type once (`toy.mul %0, %1 : tensor<*xf64>`); otherwise print a full functional type:

***mlir/Dialect.cpp***
```cpp
/// A generalized printer for binary operations. It prints in two different
/// forms depending on if all of the types match.
static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
  printer << " " << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  // If all of the types are the same, print the type directly.
  Type resultType = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(),
                   [=](Type type) { return type == resultType; })) {
    printer << resultType;
    return;
  }

  // Otherwise, print a functional type.
  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}
```

The matching `parseBinaryOp` parses exactly two operands, an optional attribute dict, then a colon-type; if that type turns out to be a `FunctionType` it distributes inputs/results accordingly, otherwise the single type is used for both operands and the result. As shown above, `AddOp::parse/print` (and likewise `MulOp`'s) are one-line dispatches to these helpers.

### 4.8 `FuncOp` — regions, interfaces, `extraClassDeclaration`

The most structurally interesting op — this *is* a Toy function:

***include/toy/Ops.td***
```tablegen
def FuncOp : Toy_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  let summary = "user defined function operation";
  let description = [{
    ...
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}
```

- **`FunctionOpInterface`**: an *interface* trait — makes the op usable through MLIR's generic function machinery (that's what the `extraClassDeclaration` methods implement). Interfaces are covered in depth in Chapter 4.
- **`IsolatedFromAbove`**: the region body may not reference SSA values defined *outside* the op. This is what lets MLIR process functions in parallel safely.
- **`arguments`**: all attributes here, no operands! `sym_name` (the `@main` symbol), the `FunctionType` stored as a type attribute, plus optional per-argument/result attribute arrays required by `FunctionOpInterface`.
- **`regions`**: one region named `$body` — the function body. This generates `getBody()`.
- **`skipDefaultBuilders = 1`**: suppress the autogenerated builders (they wouldn't create the entry block properly); our single custom builder leans on the interface helper:

***mlir/Dialect.cpp***
```cpp
void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  // FunctionOpInterface provides a convenient `build` method that will populate
  // the state of our FuncOp, and create an entry block.
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}
```

- The custom `parse`/`print` also just delegate to library helpers (`mlir::function_interface_impl::parseFunctionOp` / `printFunctionOp`), which is why `toy.func @main() { ... }` looks exactly like `func.func`.

### 4.9 `GenericCallOp` — symbol references and variadic operands

***include/toy/Ops.td***
```tablegen
def GenericCallOp : Toy_Op<"generic_call"> {
  let summary = "generic call operation";
  let description = [{
    ...
  }];

  // The generic call operation takes a symbol reference attribute as the
  // callee, and inputs for the call.
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);

  // The generic call operation returns a single value of TensorType.
  let results = (outs F64Tensor);

  // Specialize assembly printing and parsing using a declarative format.
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  // Add custom build methods for the generic call operation.
  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
  ];
}
```

- **`FlatSymbolRefAttr`**: the callee is not an SSA operand but a *symbol reference* (`@multiply_transpose`) — calls reference functions by name, resolved through MLIR's symbol tables.
- **`Variadic<F64Tensor>`**: any number of tensor operands.
- **`assemblyFormat`** (declarative this time): variables (`$callee`, `$inputs`) print/parse the corresponding argument; back-ticked literals (`` `(` ``) are punctuation; `attr-dict` is the mandatory directive for remaining attributes; `functional-type($inputs, results)` prints `(operand types) -> result types`. Result: `toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>`.
- The builder sets the result to unranked and stores the callee as an attribute:

***mlir/Dialect.cpp***
```cpp
void GenericCallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          StringRef callee, ArrayRef<mlir::Value> arguments) {
  // Generic call always returns an unranked Tensor initially.
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(arguments);
  state.addAttribute("callee",
                     mlir::SymbolRefAttr::get(builder.getContext(), callee));
}
```

### 4.10 `PrintOp` — the minimal declarative op

The upstream tutorial uses `toy.print` to introduce custom assembly formats (section 4.5). This repo uses the declarative form:

***include/toy/Ops.td***
```tablegen
def PrintOp : Toy_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  let arguments = (ins F64Tensor:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}
```

`type($input)` is the directive that prints/parses the type of `$input` — producing `toy.print %5 : tensor<*xf64>`. Section 4.5 walks through this exact format: the hand-written C++ `parse`/`print` it replaces, and the directive/literal/variable components it is made of. One TableGen line replaces ~25 lines of C++; prefer the declarative format whenever the syntax is regular.

### 4.11 `ReshapeOp` — result type constraints

***include/toy/Ops.td***
```tablegen
def ReshapeOp : Toy_Op<"reshape"> {
  let summary = "tensor reshape operation";
  let description = [{
    ...
  }];

  let arguments = (ins F64Tensor:$input);

  // We expect that the reshape operation returns a statically shaped tensor.
  let results = (outs StaticShapeTensorOf<[F64]>);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];
}
```

- **`StaticShapeTensorOf<[F64]>`**: a stronger constraint than `F64Tensor` — the result must be a *statically shaped* f64 tensor. This constraint is enforced by the ODS-generated `verifyInvariantsImpl()` for free; no hand-written verifier needed. (Try changing a reshape result to `tensor<*xf64>` in a `.mlir` file and re-parsing — the verifier rejects it.)
- The `assemblyFormat` produces `%1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>` — note the free-form keyword literal `` `to` ``.

### 4.12 `ReturnOp` — traits (`Terminator`, `HasParent`), optional operands

***include/toy/Ops.td***
```tablegen
def ReturnOp : Toy_Op<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {
  let summary = "return operation";
  let description = [{
    ...
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<F64Tensor>:$input);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>
  ];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  // Invoke a static verify method to verify this return operation.
  let hasVerifier = 1;
}
```

- **`Terminator`**: this op must be the last operation in its block (MLIR requires every block to end with a terminator).
- **`HasParent<"FuncOp">`**: structurally valid only directly inside a `toy.func`. Because of this trait, the verifier can safely do `cast<FuncOp>((*this)->getParentOp())` without checking.
- **Optional operand via `Variadic`**: ODS has no dedicated "0 or 1"-with-this-syntax mechanism here, so a variadic list is used and the verifier caps it at one.
- **Assembly format optional group**: `($input^ ...)?` prints/parses the parenthesized group only when `$input` (the anchor `^`) is non-empty. So both `toy.return` and `toy.return %2 : tensor<*xf64>` are valid.
- The **verifier** cross-checks the return against the *enclosing function's* signature — a nice example of contextual verification:

***mlir/Dialect.cpp***
```cpp
llvm::LogicalResult ReturnOp::verify() {
  // We know that the parent operation is a function, because of the 'HasParent'
  // trait attached to the operation definition.
  auto function = cast<FuncOp>((*this)->getParentOp());

  /// ReturnOps can only have a single optional operand.
  if (getNumOperands() > 1)
    return emitOpError() << "expects at most 1 return operand";

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError() << "does not return the same number of values ("
                         << getNumOperands() << ") as the enclosing function ("
                         << results.size() << ")";

  // If the operation does not have an input, we are done.
  if (!hasOperand())
    return mlir::success();

  auto inputType = *operand_type_begin();
  auto resultType = results.front();

  // Check that the result type of the function matches the operand type.
  if (inputType == resultType || llvm::isa<mlir::UnrankedTensorType>(inputType) ||
      llvm::isa<mlir::UnrankedTensorType>(resultType))
    return mlir::success();

  return emitError() << "type of return operand (" << inputType
                     << ") doesn't match function result type (" << resultType
                     << ")";
}
```

### 4.13 `TransposeOp` — the running example

***include/toy/Ops.td***
```tablegen
def TransposeOp : Toy_Op<"transpose"> {
  let summary = "transpose operation";

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];

  // Allow building a TransposeOp with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];

  // Invoke a static verify method to verify this transpose operation.
  let hasVerifier = 1;
}
```

Its builder (like the binary ops', an unranked result) and verifier are in `Dialect.cpp`. The verifier only fires when both shapes are known (ranked), in which case the result shape must be the reverse of the input shape:

***mlir/Dialect.cpp***
```cpp
void TransposeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(value);
}

llvm::LogicalResult TransposeOp::verify() {
  auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(getType());
  if (!inputType || !resultType)
    return mlir::success();

  auto inputShape = inputType.getShape();
  if (!std::equal(inputShape.begin(), inputShape.end(),
                  resultType.getShape().rbegin())) {
    return emitError()
           << "expected result shape to be a transpose of the input";
  }
  return mlir::success();
}
```

### 4.14 Demystifying TableGen: the actually-generated code

You never *have* to read the generated code, but seeing it once cures TableGen of its magic. This is the real generated `TransposeOp` class from `Ch2/build/op-decls.inc` in this repo (abridged):

***build/op-decls.inc***
```cpp
class TransposeOp : public ::mlir::Op<TransposeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TransposeOpAdaptor;
...
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("toy.transpose");
  }
...
  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }
...
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value input);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
```

Note how the trait list we saw in the hand-written C++ version (section 4.1) got derived automatically from `arguments`/`results` (`OneOperand`, `OneResult`, `OneTypedResult<TensorType>`). And in the generated definitions, verification is wired so **structural checks run before your semantic verifier**:

***build/op-defs.inc***
```cpp
::llvm::LogicalResult TransposeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}
```

Even the `assemblyFormat` string was compiled into a ~40-line `TransposeOp::parse()` that calls `parseLParen()`, `parseOperand()`, `parseColon()`, `parseKeyword("to")`, etc. — exactly what you would have written by hand. See the [appendix](#a3-inspecting-tablegen-output-by-hand) for how to regenerate/inspect these files yourself.

---

## 5. The MLIRGen Module

With the dialect in place, [`mlir/MLIRGen.cpp`](mlir/MLIRGen.cpp) walks the Chapter 1 AST and emits ops. The public API is one function:

***include/toy/MLIRGen.h***
```cpp
/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          ModuleAST &moduleAST);
```

`OwningOpRef` is an RAII handle that erases the module when it goes out of scope. Internally everything lives in a private `MLIRGenImpl` class with three key members:

***mlir/MLIRGen.cpp***
```cpp
class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}
  ...
private:
  /// A "module" matches a Toy source file: containing a list of functions.
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;
  ...
};
```

### 5.1 Locations

Every `builder.create<...>` call needs a location. A tiny helper converts Toy AST locations (file/line/col captured by the lexer) into MLIR locations:

***mlir/MLIRGen.cpp***
```cpp
/// Helper conversion for a Toy AST location to an MLIR location.
mlir::Location loc(const Location &loc) {
  return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file), loc.line,
                                   loc.col);
}
```

This is why the output in section 6.4 carries precise `loc("codegen.toy":3:10)` annotations for free. (The file string is simply the input path as passed on the command line.)

### 5.2 The module and function overloads

The top-level `mlirGen(ModuleAST&)` creates an empty `builtin.module`, codegens each function into it, and **verifies** the result — this is where all the ODS constraints and our hand-written `verify()` methods actually run:

***mlir/MLIRGen.cpp***
```cpp
/// Public API: convert the AST for a Toy module (source file) to an MLIR
/// Module operation.
mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
  // We create an empty MLIR module and codegen functions one at a time and
  // add them to the module.
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

  for (FunctionAST &f : moduleAST)
    mlirGen(f);

  // Verify the module after we have finished constructing it, this will check
  // the structural properties of the IR and invoke any specific verifiers we
  // have on the Toy operations.
  if (failed(mlir::verify(theModule))) {
    theModule.emitError("module verification error");
    return nullptr;
  }

  return theModule;
}
```

`mlirGen(PrototypeAST&)` builds the `toy.func` header. Since Toy is dynamically shaped, **every argument is typed `tensor<*xf64>`** (unranked) and the return type starts empty. `mlirGen(FunctionAST&)` then orchestrates a whole function and shows all three members working together, with `declare()` registering names in the symbol table:

***mlir/MLIRGen.cpp***
```cpp
/// Declare a variable in the current scope, return success if the variable
/// wasn't declared yet.
llvm::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
  if (symbolTable.count(var))
    return mlir::failure();
  symbolTable.insert(var, value);
  return mlir::success();
}

/// Create the prototype for an MLIR function with as many arguments as the
/// provided Toy AST prototype.
mlir::toy::FuncOp mlirGen(PrototypeAST &proto) {
  auto location = loc(proto.loc());

  // This is a generic function, the return type will be inferred later.
  // Arguments type are uniformly unranked tensors.
  llvm::SmallVector<mlir::Type, 4> argTypes(proto.getArgs().size(),
                                            getType(VarType{}));
  auto funcType = builder.getFunctionType(argTypes, {});
  return builder.create<mlir::toy::FuncOp>(location, proto.getName(),
                                           funcType);
}

/// Emit a new function and add it to the MLIR module.
mlir::toy::FuncOp mlirGen(FunctionAST &funcAST) {
  // Create a scope in the symbol table to hold variable declarations.
  ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

  // Create an MLIR function for the given prototype.
  builder.setInsertionPointToEnd(theModule.getBody());
  mlir::toy::FuncOp function = mlirGen(*funcAST.getProto());
  if (!function)
    return nullptr;

  // Let's start the body of the function now!
  mlir::Block &entryBlock = function.front();
  auto protoArgs = funcAST.getProto()->getArgs();

  // Declare all the function arguments in the symbol table.
  for (const auto nameValue :
       llvm::zip(protoArgs, entryBlock.getArguments())) {
    if (failed(declare(std::get<0>(nameValue)->getName(),
                       std::get<1>(nameValue))))
      return nullptr;
  }

  // Set the insertion point in the builder to the beginning of the function
  // body, it will be used throughout the codegen to create operations in this
  // function.
  builder.setInsertionPointToStart(&entryBlock);

  // Emit the body of the function.
  if (mlir::failed(mlirGen(*funcAST.getBody()))) {
    function.erase();
    return nullptr;
  }

  // Implicitly return void if no return statement was emitted.
  // FIXME: we may fix the parser instead to always return the last expression
  // (this would possibly help the REPL case later)
  ReturnOp returnOp;
  if (!entryBlock.empty())
    returnOp = dyn_cast<ReturnOp>(entryBlock.back());
  if (!returnOp) {
    builder.create<ReturnOp>(loc(funcAST.getProto()->loc()));
  } else if (returnOp.hasOperand()) {
    // Otherwise, if this return operation has an operand then add a result to
    // the function.
    function.setType(builder.getFunctionType(
        function.getFunctionType().getInputs(), getType(VarType{})));
  }

  return function;
}
```

Two things worth internalizing: the **insertion point** is how the builder knows *where* ops land (module end for the `FuncOp` itself, entry-block start for its body); and `declare()` (a `symbolTable.count/insert` wrapper) rejects double declarations in a scope. A missing `return` gets an implicit `toy.return`; a `return` with an operand patches the function type to return one unranked tensor.

### 5.3 Expression overloads

Codegen dispatches on the AST node kind (LLVM-style RTTI):

***mlir/MLIRGen.cpp***
```cpp
/// Dispatch codegen for the right expression subclass using RTTI.
mlir::Value mlirGen(ExprAST &expr) {
  switch (expr.getKind()) {
  case toy::ExprAST::Expr_BinOp:
    return mlirGen(cast<BinaryExprAST>(expr));
  case toy::ExprAST::Expr_Var:
    return mlirGen(cast<VariableExprAST>(expr));
  case toy::ExprAST::Expr_Literal:
    return mlirGen(cast<LiteralExprAST>(expr));
  case toy::ExprAST::Expr_Call:
    return mlirGen(cast<CallExprAST>(expr));
  case toy::ExprAST::Expr_Num:
    return mlirGen(cast<NumberExprAST>(expr));
  default:
    emitError(loc(expr.loc()))
        << "MLIR codegen encountered an unhandled expr kind '"
        << Twine(expr.getKind()) << "'";
    return nullptr;
  }
}
```

Each overload maps to one or two Toy ops:

- **`BinaryExprAST`** → `toy.add` / `toy.mul`. Recurses into LHS then RHS first (so operand ops are emitted before the op that uses them — SSA order), then switches on the operator.
- **`VariableExprAST`** → no op at all, just a **symbol-table lookup**; an unknown name produces a proper diagnostic anchored at the source location:

***mlir/MLIRGen.cpp***
```cpp
  // Derive the operation name from the binary operator. At the moment we only
  // support '+' and '*'.
  switch (binop.getOp()) {
  case '+':
    return builder.create<AddOp>(location, lhs, rhs);
  case '*':
    return builder.create<MulOp>(location, lhs, rhs);
  }

  emitError(location, "invalid binary operator '") << binop.getOp() << "'";
  return nullptr;
...
/// This is a reference to a variable in an expression. The variable is
/// expected to have been declared and so should have a value in the symbol
/// table, otherwise emit an error and return nullptr.
mlir::Value mlirGen(VariableExprAST &expr) {
  if (auto variable = symbolTable.lookup(expr.getName()))
    return variable;

  emitError(loc(expr.loc()), "error: unknown variable '")
      << expr.getName() << "'";
  return nullptr;
}
```

- **`LiteralExprAST`** → `toy.constant`. The nested array literal is flattened into a `std::vector<double>` by the recursive `collectData()` helper, wrapped in a `DenseElementsAttr` typed `tensor<2x3xf64>` (etc.), and attached to the op — this is the canonical "constant data goes into attributes" pattern:

***mlir/MLIRGen.cpp***
```cpp
mlir::Value mlirGen(LiteralExprAST &lit) {
  auto type = getType(lit.getDims());

  // The attribute is a vector with a floating point value per element
  // (number) in the array, see `collectData()` below for more details.
  std::vector<double> data;
  data.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(), 1,
                               std::multiplies<int>()));
  collectData(lit, data);

  // The type of this attribute is tensor of 64-bit floating-point with the
  // shape of the literal.
  mlir::Type elementType = builder.getF64Type();
  auto dataType = mlir::RankedTensorType::get(lit.getDims(), elementType);

  // This is the actual attribute that holds the list of values for this
  // tensor literal.
  auto dataAttribute =
      mlir::DenseElementsAttr::get(dataType, llvm::ArrayRef(data));

  // Build the MLIR op `toy.constant`. This invokes the `ConstantOp::build`
  // method.
  return builder.create<ConstantOp>(loc(lit.loc()), type, dataAttribute);
}
```

- **`NumberExprAST`** → `toy.constant` via the convenience `double` builder we declared in ODS: `builder.create<ConstantOp>(loc(num.loc()), num.getValue());`
- **`CallExprAST`** → the builtin `transpose(x)` becomes `toy.transpose` (with an arity check); *any other* callee becomes a `toy.generic_call` carrying the callee name as a symbol attribute:

***mlir/MLIRGen.cpp***
```cpp
  // Builtin calls have their custom operation, meaning this is a
  // straightforward emission.
  if (callee == "transpose") {
    if (call.getArgs().size() != 1) {
      emitError(location, "MLIR codegen encountered an error: toy.transpose "
                          "does not accept multiple arguments");
      return nullptr;
    }
    return builder.create<TransposeOp>(location, operands[0]);
  }

  // Otherwise this is a call to a user-defined function. Calls to
  // user-defined functions are mapped to a custom call that takes the callee
  // name as an attribute.
  return builder.create<GenericCallOp>(location, callee, operands);
```

- **`PrintExprAST`** → `toy.print` on the codegen'd argument; **`ReturnExprAST`** → `toy.return` with zero or one operand.
- **`VarDeclExprAST`** (`var a<2,3> = ...;`) → codegen the initializer, then, if the declaration specifies a shape, insert a **`toy.reshape`** to that ranked type, and finally `declare()` the name:

***mlir/MLIRGen.cpp***
```cpp
  mlir::Value value = mlirGen(*init);
  if (!value)
    return nullptr;

  // We have the initializer value, but in case the variable was declared
  // with specific shape, we emit a "reshape" operation. It will get
  // optimized out later as needed.
  if (!vardecl.getType().shape.empty()) {
    value = builder.create<ReshapeOp>(loc(vardecl.loc()),
                                      getType(vardecl.getType()), value);
  }

  // Register the value in the symbol table.
  if (failed(declare(vardecl.getName(), value)))
    return nullptr;
  return value;
```

This is why `var a<2, 3> = [[1, 2, 3], [4, 5, 6]]` produces a constant *plus* a (here redundant) reshape — Chapter 3 will optimize such reshapes away.

- **`ExprASTList`** (a block of statements) opens another `ScopedHashTableScope` and dispatches per statement kind.

Finally the type helper — the root of all the `tensor<*xf64>` in our output:

***mlir/MLIRGen.cpp***
```cpp
/// Build a tensor type from a list of shape dimensions.
mlir::Type getType(ArrayRef<int64_t> shape) {
  // If the shape is empty, then this type is unranked.
  if (shape.empty())
    return mlir::UnrankedTensorType::get(builder.getF64Type());

  // Otherwise, we use the given shape.
  return mlir::RankedTensorType::get(shape, builder.getF64Type());
}

/// Build an MLIR type from a Toy AST variable type (forward to the generic
/// getType above).
mlir::Type getType(const VarType &type) { return getType(type.shape); }
```

### 5.4 The driver

The driver gains a `-emit=mlir` action next to Chapter 1's `-emit=ast`, plus an input-kind switch. `dumpMLIR()` handles **two input paths** — a `.toy` source goes through the Toy frontend and `mlirGen`; a `.mlir` file (or `-x mlir`) goes through MLIR's own parser, using *our* registered dialect including the custom `ConstantOp::parse` etc. — which is what enables the round-trip test. `main()` also registers the MLIR/asm-printer command-line option categories; that is where the `-mlir-print-debuginfo` flag comes from:

***toyc.cpp***
```cpp
int dumpMLIR() {
  mlir::MLIRContext context;
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::toy::ToyDialect>();

  // Handle '.toy' input to the compiler.
  if (inputType != InputType::MLIR &&
      !llvm::StringRef(inputFilename).ends_with(".mlir")) {
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
      return 6;
    mlir::OwningOpRef<mlir::ModuleOp> module = mlirGen(context, *moduleAST);
    if (!module)
      return 1;

    module->dump();
    return 0;
  }

  // Otherwise, the input is '.mlir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }

  module->dump();
  return 0;
}
...
int main(int argc, char **argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  switch (emitAction) {
  case Action::DumpAST:
    return dumpAST();
  case Action::DumpMLIR:
    return dumpMLIR();
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }

  return 0;
}
```

---

## 6. Build and Run

The superbuild (`toy/build.sh`, `toy/run.sh`, `CMakePresets.json`) is documented once in the top-level [README](../README.md#the-build-system). This section builds and runs Chapter 2 on its own, from the chapter directory. What the chapter's `CMakeLists.txt` adds — the TableGen wiring — is in the [appendix](#appendix-what-chapter-2-adds-to-the-build).

### 6.1 Building

```bash
cd /Users/roy/study/mlir/toy/Ch2
cmake -S . -B build -G Ninja
cmake --build build          # → ./build/toyc-ch2
```

No preset applies at the chapter level, yet no toolchain flags are needed, because the shell environment already points at Homebrew LLVM 20:

- `CXX=/opt/homebrew/opt/llvm@20/bin/clang++` (and `CC`) selects the compiler.
- `/opt/homebrew/opt/llvm@20/bin` is on `PATH`, and `find_package` also searches the prefix above each `PATH` entry, so it finds `/opt/homebrew/opt/llvm@20/lib/cmake/{mlir,llvm}` by itself.

In a shell without that setup, pass them explicitly: `-DMLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++`.

This is the first chapter with a TableGen step: `mlir-tblgen` generates C++ from `Ops.td` into `build/include/toy/*.inc` before anything else compiles. A standalone build puts the binary directly in `build/`; the superbuild's `toy/build/bin/toyc-ch2` behaves identically.

### 6.2 Running

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 codegen.toy -emit=mlir -mlir-print-debuginfo 2>&1
```

- `codegen.toy` — the input program (section 6.3), relative to `Ch2/`.
- `-emit=mlir` — take the driver's `DumpMLIR` action (vs. `-emit=ast` from Chapter 1, which still works).
- `-mlir-print-debuginfo` — ask the asm printer to include `loc(...)` on every operation (locations are always *stored*, just not printed by default). This flag exists because `main()` called `mlir::registerAsmPrinterCLOptions()` (section 5.4).
- `2>&1` — `module->dump()` writes to **stderr**; redirect it if you want to pipe or save the IR.
- An input ending in `.mlir` (or any input with `-x mlir`) takes the driver's *Path B* (section 5.4): it uses MLIR's parser — and therefore our dialect's registered `parse()` methods — instead of the Toy frontend. The round trip in section 6.5 relies on this.

### 6.3 The input

***codegen.toy***
```text
# User defined generic function that operates on unknown shaped arguments.
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

### 6.4 Actual captured output, annotated

Real output on this machine (every `loc(...)` reproduces the input path exactly as given on the command line — here `codegen.toy`):

```mlir
module {
  toy.func @multiply_transpose(%arg0: tensor<*xf64> loc("codegen.toy":2:1), %arg1: tensor<*xf64> loc("codegen.toy":2:1)) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64> loc("codegen.toy":3:10)
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64> loc("codegen.toy":3:25)
    %2 = toy.mul %0, %1 : tensor<*xf64> loc("codegen.toy":3:25)
    toy.return %2 : tensor<*xf64> loc("codegen.toy":3:3)
  } loc("codegen.toy":2:1)
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64> loc("codegen.toy":7:17)
    %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64> loc("codegen.toy":7:3)
    %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64> loc("codegen.toy":8:17)
    %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64> loc("codegen.toy":8:3)
    %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("codegen.toy":9:11)
    %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("codegen.toy":10:11)
    toy.print %5 : tensor<*xf64> loc("codegen.toy":11:3)
    toy.return loc("codegen.toy":6:1)
  } loc("codegen.toy":6:1)
} loc(unknown)
```

Line-by-line mapping back to the `.toy` source (the `mlirGen` overloads named here are explained in section 5):

| MLIR | Emitted by | Toy source |
|---|---|---|
| `module { ... } loc(unknown)` | `mlirGen(ModuleAST&)` — created with `getUnknownLoc()`, hence `loc(unknown)` | the whole file |
| `toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: ...)` | `mlirGen(PrototypeAST&)` — args are unranked `tensor<*xf64>` because shapes are unknown | `def multiply_transpose(a, b)` (line 2, hence `2:1`) |
| `%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>` | `mlirGen(CallExprAST&)` builtin path → `TransposeOp` | `transpose(a)` at line 3, col 10 |
| `%1 = toy.transpose(%arg1 ...)` | same | `transpose(b)` at 3:25 |
| `%2 = toy.mul %0, %1 : tensor<*xf64>` | `mlirGen(BinaryExprAST&)` `'*'` case → `MulOp`; single type printed by `printBinaryOp` since all types match | `*` (location = RHS position 3:25) |
| `toy.return %2 : tensor<*xf64>` | `mlirGen(ReturnExprAST&)` | `return ...;` at 3:3 |
| `-> tensor<*xf64>` on the func | the `function.setType(...)` patch-up because the return had an operand | |
| `%0 = toy.constant dense<[[1.0...]]> : tensor<2x3xf64>` | `mlirGen(LiteralExprAST&)` — nested literal flattened into a `DenseElementsAttr` of type `tensor<2x3xf64>` | `[[1, 2, 3], [4, 5, 6]]` at 7:17 |
| `%1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>` | `mlirGen(VarDeclExprAST&)` — declared shape `<2, 3>` forces a reshape (redundant here; Chapter 3 removes it) | `var a<2, 3> = ...` at 7:3 |
| `%2 = toy.constant dense<[1.0, ..., 6.0]> : tensor<6xf64>` | flat 6-element literal → rank-1 tensor | `[1, 2, 3, 4, 5, 6]` at 8:17 |
| `%3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64>` | this reshape is *not* redundant: rank 1 → rank 2 | `var b<2, 3> = ...` at 8:3 |
| `%4 = toy.generic_call @multiply_transpose(%1, %3) : (...) -> tensor<*xf64>` | `mlirGen(CallExprAST&)` user-function path → `GenericCallOp`; callee is the `@...` symbol attribute; result unranked pending Ch4 shape inference | `var c = multiply_transpose(a, b);` at 9:11 |
| `%5 = toy.generic_call @multiply_transpose(%3, %1) ...` | same, swapped args (`%4`/`c` is dead — later chapters clean it up) | `var d = multiply_transpose(b, a);` at 10:11 |
| `toy.print %5 : tensor<*xf64>` | `mlirGen(PrintExprAST&)` → `PrintOp`, printed by its declarative `assemblyFormat` | `print(d);` at 11:3 |
| `toy.return` (no operand) | the *implicit* return inserted by `mlirGen(FunctionAST&)`; its location is the function prototype's (6:1) | end of `main` |

Also note the function-argument locations (`%arg0: tensor<*xf64> loc("codegen.toy":2:1)`) — block arguments carry locations too.

### 6.5 The round trip and why it matters

Re-parse the emitted MLIR and emit it again — pipe the first emission straight back in, using `-` (stdin) and `-x mlir` to force the MLIR parser path regardless of extension:

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 codegen.toy -emit=mlir 2>&1 | ./build/toyc-ch2 - -x mlir -emit=mlir 2>&1
```

Actual output (no `-mlir-print-debuginfo` this time, so no `loc(...)`):

```mlir
module {
  toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
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

The upstream tutorial does the same through a file: `./build/toyc-ch2 codegen.toy -emit=mlir -mlir-print-debuginfo 2> codegen.mlir`, then `./build/toyc-ch2 codegen.mlir -emit=mlir` (a `.mlir` suffix selects the MLIR parser, no `-x mlir` needed). That prints exactly the output above; adding `-mlir-print-debuginfo` to the second command reproduces section 6.4 byte for byte — the `loc(...)` annotations survive the round trip too.

Semantically identical to the first emission — the round trip **passes**. Why is this a meaningful test? Because it exercises *both directions* of every custom assembly definition we wrote:

- **Emit path** exercises the *printers*: `ConstantOp::print`, `printBinaryOp`, the `assemblyFormat`-generated printers for `transpose`/`reshape`/`print`/`generic_call`/`return`, and `printFunctionOp`.
- **Re-parse path** exercises the *parsers*: `ConstantOp::parse` (including reconstructing the result type from the attribute), `parseBinaryOp` (including the matched-types shorthand `: tensor<*xf64>`), the format-generated parsers (`(` … `:` … `)` … `to` …), and `parseFunctionOp`.
- Parsing also re-runs the **verifiers**, so any structurally invalid syntax we might print would be caught immediately.

A parser/printer mismatch (say, printing `to` but parsing `into`) is one of the most common dialect bugs, and a round trip catches it instantly. This is exactly how upstream MLIR lit tests work: `toyc-ch2 ... -emit=mlir | toyc-ch2 - -x mlir -emit=mlir | FileCheck`.

### 6.6 More inputs to try

`test_Example/Toy/Ch2/` has additional cases:

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 ../../test_Example/Toy/Ch2/scalar.toy -emit=mlir    # scalar constant + reshape
./build/toyc-ch2 ../../test_Example/Toy/Ch2/empty.toy  -emit=mlir    # no `def`: parse error, empty module
./build/toyc-ch2 ../../test_Example/Toy/Ch2/ast.toy    -emit=ast     # Chapter 1 action still works
./build/toyc-ch2 ../../test_Example/Toy/Ch2/invalid.mlir -emit=mlir  # exercises parser diagnostics
```

`invalid.mlir` is the negative test: it contains malformed Toy IR, and the point is to watch the *registered* dialect reject it with a precise diagnostic instead of accepting it opaquely (contrast with section 2.4).

### 6.7 The ecosystem view: feeding Toy IR to stock `mlir-opt`

`toyc-ch2` is just `mlir-opt` with the Toy dialect linked in. Two experiments against the *stock* Homebrew `mlir-opt` make that concrete. First, the custom assembly syntax is unparseable without the dialect's registered `parse()` methods — even with the escape hatch flag:

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 codegen.toy -emit=mlir 2>&1 \
  | /opt/homebrew/opt/llvm@20/bin/mlir-opt -allow-unregistered-dialect
# → error: Dialect `toy' not found for custom op 'toy.func'
```

But print the same module in **generic form** and stock `mlir-opt` round-trips it fine, treating every `toy.*` op as an opaque registered-by-nobody operation (this is section 2.4's registered-vs-opaque distinction, live):

```bash
./build/toyc-ch2 codegen.toy -emit=mlir -mlir-print-op-generic 2>&1 \
  | /opt/homebrew/opt/llvm@20/bin/mlir-opt -allow-unregistered-dialect
# → "toy.func"() <{function_type = ..., sym_name = "multiply_transpose"}> ({ ... — parses and reprints
```

The lesson: MLIR's *generic* syntax (`"toy.transpose"(%arg0) : (tensor<*xf64>) -> tensor<*xf64>`) needs no dialect at all — it's pure structure — while the *custom* syntax, verification, and semantics all come from registration. That's why every project with a dialect ships its own `foo-opt` binary, and why `toyc-chN` exists rather than everything funneling through one universal tool.

---

## 7. Key Takeaways & Pitfalls

**Takeaways**

1. **One concept scales the whole IR**: operations (with operands, results, attributes, regions, locations) model everything from modules to arithmetic. Dialects namespace them; contexts load dialects.
2. **Registered beats opaque**: MLIR will happily round-trip unknown ops, but only registration buys verification, typed accessors, pretty syntax, and a foundation for optimization.
3. **ODS is leverage**: ~30 lines of TableGen per op replace hundreds of lines of brittle C++ (compare `Ops.td` with `op-decls.inc`/`op-defs.inc`). Custom C++ remains available exactly where declarativeness runs out (conditional syntax like `printBinaryOp`, semantic verifiers, non-trivial builders).
4. **Verification is layered**: ODS constraints (`F64Tensor`, `StaticShapeTensorOf`, trait invariants) run automatically in `verifyInvariantsImpl()`; `hasVerifier = 1` appends your semantic `verify()`. `mlir::verify(module)` in MLIRGen triggers the whole stack.
5. **MLIRGen is small on purpose**: a stateful `OpBuilder` (insertion point!), a `ScopedHashTable` symbol table, a `loc()` helper, and one `mlirGen` overload per AST node. Shapes are deliberately left unranked (`tensor<*xf64>`) — inference comes in Chapter 4, reshape/transpose cleanups in Chapter 3.
6. **Round-tripping is the cheapest dialect test you'll ever write** — it validates printer and parser against each other.
7. **The generic form is the ground truth**: every op, registered or not, reduces to name/operands/attributes/result types/location/successors/regions, and `-mlir-print-op-generic` shows it. Custom formats are just sugar over it — and in current MLIR, inherent attributes are *properties*, printed as `<{...}>`.

**Pitfalls**

- **Forgetting `context.getOrLoadDialect<ToyDialect>()`** — parsing `.mlir` input then fails with an unregistered-dialect error even though the code compiled fine.
- **Out-of-tree TableGen races**: without `add_dependencies(toyc-ch2 ToyCh2OpsIncGen)`, Ninja can compile `Dialect.cpp` before `Ops.h.inc` exists. In-tree builds hide this because their helper macros add the dependency; out-of-tree builds (superbuild or standalone) must do it explicitly (this repo's CMakeLists carries a comment about exactly this fix).
- **`GET_OP_LIST` vs. `GET_OP_CLASSES`**: `Ops.cpp.inc` is multi-purpose; including it without the right guard macro gives baffling redefinition or "no ops registered" problems.
- **`dump()` writes to stderr** — hence `2>&1` in the commands above. Redirecting stdout captures nothing.
- **Locations are easy to squander**: use `loc(expr.loc())` for every `builder.create<>`; falling back to `getUnknownLoc()` everywhere destroys diagnostics quality later.
- **Verifier ordering assumption**: your `verify()` runs *after* structural checks, so you may rely on operand counts/types already being validated — but nothing more. E.g. `ReturnOp::verify` can `cast<FuncOp>` its parent only because of the `HasParent<"FuncOp">` trait.
- **Custom `parse()` must fully populate `OperationState`** — forgetting `result.addTypes(...)` (as `ConstantOp::parse` does from the attribute type) yields an op with zero results and confusing downstream errors.
- Homebrew-specific: linking granular `MLIRxxx` component libraries mixes badly with the monolithic `libMLIR.dylib`; this repo links just `MLIR` + `LLVM`.

---

## Appendix: What Chapter 2 adds to the build

This is the chapter where the build stops being a plain C++ compile — `mlir-tblgen` now generates C++ from `Ops.td` before anything else can compile.

### A.1 The chapter targets and the TableGen wiring

Below the standalone guard:

***CMakeLists.txt***
```cmake
add_subdirectory(include)

add_executable(toyc-ch2
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  )

add_dependencies(toyc-ch2 ToyCh2OpsIncGen)  # Added to fix include dependency for toy/Dialect.h.inc

include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

target_link_libraries(toyc-ch2
  PRIVATE
    MLIR                      # libMLIR.dylib (all dialects, passes, conversions)
    LLVM                      # libLLVM.dylib (all targets, all components)
    )
```

Notable differences from upstream:

- **Monolithic shared libraries**: Homebrew's LLVM ships `libMLIR.dylib` / `libLLVM.dylib`, so instead of listing fine-grained components (`MLIRAnalysis`, `MLIRIR`, `MLIRParser`, …) we link just two libraries.
- **Explicit `add_dependencies(toyc-ch2 ToyCh2OpsIncGen)`**: in-tree helper macros normally add this dependency for you. Out-of-tree, without it, Ninja may try to compile `Dialect.cpp` before TableGen has produced `toy/Ops.h.inc` — a classic build race. This line was added to fix exactly that.
- Two include roots: the *source* `include/` (for `Ops.td`, `Dialect.h`) and the chapter's *binary* include dir (`${CMAKE_CURRENT_BINARY_DIR}/include/`), where the generated `.inc` files land mirroring the source layout. In the superbuild that is `build/Ch2/include/toy/*.inc`; in a standalone chapter build it is `Ch2/build/include/toy/*.inc`.

The TableGen wiring itself lives one level down (reached via `include/CMakeLists.txt`, which is just `add_subdirectory(toy)`):

***include/toy/CMakeLists.txt***
```cmake
set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(ToyCh2OpsIncGen)
```

Line by line: `LLVM_TARGET_DEFINITIONS` names the `.td` input; each `mlir_tablegen(<output> <generator>)` adds a build rule running `mlir-tblgen <generator>` over it; `add_public_tablegen_target` bundles those four rules into the named target `ToyCh2OpsIncGen` that other targets can depend on. Four generated files, four consumers:

| Generated file | Generator | Included from |
|---|---|---|
| `Dialect.h.inc` | `-gen-dialect-decls` | `include/toy/Dialect.h` |
| `Dialect.cpp.inc` | `-gen-dialect-defs` | `mlir/Dialect.cpp` |
| `Ops.h.inc` | `-gen-op-decls` | `include/toy/Dialect.h` (under `GET_OP_CLASSES`) |
| `Ops.cpp.inc` | `-gen-op-defs` | `mlir/Dialect.cpp` (under `GET_OP_LIST` and `GET_OP_CLASSES`) |

### A.2 Every later chapter repeats this pattern

Chapters 3–7 all keep this exact structure — `include/toy/CMakeLists.txt` generating the op/dialect `.inc` files into a `ToyChNOpsIncGen` target, plus the explicit `add_dependencies` race guard — and each adds its own extra `.td` → generator pairs on top (DRR rewriters in Ch3, interfaces in Ch4). Their READMEs describe only those additions.

### A.3 Inspecting TableGen output by hand

You don't need CMake to see what ODS generates — [`run_mlir-tblgen.sh`](run_mlir-tblgen.sh) (run from inside `Ch2/`) invokes `mlir-tblgen` directly, once per generator, with the Homebrew MLIR headers on the include path (needed to resolve `include "mlir/IR/OpBase.td"` etc.):

***run_mlir-tblgen.sh***
```bash
mkdir -p ./build
mlir-tblgen -gen-dialect-decls ./include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include/ -o ./build/dialect-decls.inc
mlir-tblgen -gen-dialect-defs ./include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include/ -o ./build/dialect-defs.inc
mlir-tblgen -gen-op-decls ./include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include/ -o ./build/op-decls.inc
mlir-tblgen -gen-op-defs ./include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include/ -o ./build/op-defs.inc
```

The outputs land in `Ch2/build/` (next to, but separate from, the CMake-generated `.inc` files under `build/include/toy/`) as `dialect-decls.inc`, `dialect-defs.inc`, `op-decls.inc` (~1600 lines), and `op-defs.inc` (~1600 lines) — these are exactly the files quoted in sections 3.2, 4.4 and 4.14. Other useful generators to try: `mlir-tblgen -gen-op-doc ./include/toy/Ops.td -I ...` renders the `summary`/`description` fields as markdown documentation.

---

## Links

- Official doc: [Toy Tutorial Chapter 2 — Emitting Basic MLIR](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/)
- Related MLIR docs: [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/) · [Operation Definition Specification (ODS)](https://mlir.llvm.org/docs/DefiningDialects/Operations/) · [Declarative Assembly Format](https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format)
- Previous: [Chapter 1 — Toy Language and AST](../Ch1/README.md)
- Next: [Chapter 3 — High-level Language-Specific Analysis and Transformation](../Ch3/README.md)
- Back to [README](../README.md)
