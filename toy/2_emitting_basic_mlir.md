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

Why bother? LLVM IR has a *fixed* instruction set at a *fixed* (low) level of abstraction. By the time you have lowered `transpose(a) * transpose(b)` to LLVM IR, the fact that these were tensor transposes is gone — you cannot easily write an optimization like "transpose(transpose(x)) = x" anymore. MLIR is different: it ships with very few built-in operations and instead lets *you* define new operations, types, and attributes at whatever level of abstraction fits your problem. Multiple frontends can then share one infrastructure for analyses, transformations, tracking source locations, multithreaded compilation, and so on, instead of each reinventing its own IR. Toy's dialect is a *high-level*, tensor-based IR that preserves language semantics so the next chapters can do meaningful optimizations on it.

### Where things live in this repo

This chapter is part of an **out-of-tree CMake superbuild** at `toy/` that covers all chapters (Ch1–Ch7) — it is *not* built inside `llvm-project` like the upstream tutorial. It links against a prebuilt Homebrew LLVM/MLIR 20. Each `ChN/` also still configures as a standalone project (see section 6).

| Path | Purpose |
|---|---|
| `/Users/roy/study/mlir/toy/Ch2/toyc.cpp` | Compiler driver (`-emit=ast` / `-emit=mlir`) |
| `/Users/roy/study/mlir/toy/Ch2/include/toy/Ops.td` | ODS (TableGen) definitions: dialect + 9 ops |
| `/Users/roy/study/mlir/toy/Ch2/include/toy/Dialect.h` | Pulls in the TableGen-generated declarations |
| `/Users/roy/study/mlir/toy/Ch2/mlir/Dialect.cpp` | Dialect init + hand-written verifiers/builders/parsers/printers |
| `/Users/roy/study/mlir/toy/Ch2/mlir/MLIRGen.cpp` | AST → MLIR emission |
| `/Users/roy/study/mlir/toy/Ch2/include/toy/MLIRGen.h` | Public `mlirGen()` entry point |
| `/Users/roy/study/mlir/toy/CMakeLists.txt`, `CMakePresets.json` | Superbuild top level: finds MLIR/LLVM once, adds `Ch1`…`Ch7`; preset pins Ninja/compilers/`MLIR_DIR` |
| `/Users/roy/study/mlir/toy/Ch2/CMakeLists.txt` | Chapter build wiring (TableGen + executable); dual-mode: superbuild or standalone |
| `/Users/roy/study/mlir/toy/Ch2/codegen.toy` | The example program compiled in this chapter |
| `/Users/roy/study/mlir/toy/build.sh`, `run.sh` | Build / run scripts for all chapters (`./build.sh ch2`, `./run.sh ch2`) |
| `/Users/roy/study/mlir/toy/Ch2/run_mlir-tblgen.sh` | TableGen-inspection script |
| `/Users/roy/study/mlir/test_Example/Toy/Ch2/` | Extra test inputs (`ast.toy`, `codegen.toy`, `empty.toy`, `scalar.toy`, `invalid.mlir`) |

Lexer/Parser/AST files (`Lexer.h`, `Parser.h`, `AST.h`, `parser/AST.cpp`) are carried over unchanged from Chapter 1.

---

## 2. MLIR Core Concepts

### 2.1 Everything is an operation

In MLIR there is no hard-coded notion of "instruction", "function", or "module". There is exactly one first-class unit of computation and structure: the **operation**. Functions are operations, modules are operations, `if` statements are operations. This uniformity is what makes MLIR extensible: users can define operations that model *application-specific* semantics at any abstraction level, and every generic piece of infrastructure (printing, parsing, verification, pass management, location tracking) works on them automatically.

Dissect a Toy operation in its *generic* textual form:

```mlir
%t_tensor = "toy.transpose"(%tensor) {inplace = true}
    : (tensor<2x3xf64>) -> tensor<3x2xf64> loc("example/file/path":12:1)
```

| Piece | Meaning |
|---|---|
| `%t_tensor` | The name of the (single) **result** — an SSA **value** defined by this op. A `#` suffix can index into multiple results. This name is only a convenience of the printer; it is not part of the in-memory IR. |
| `"toy.transpose"` | The **operation name**. Always a unique string prefixed by the **dialect** namespace (`toy.`) followed by the mnemonic (`transpose`). |
| `(%tensor)` | The list of zero or more **operands** — SSA values produced by other operations or block arguments. |
| `{inplace = true}` | A dictionary of zero or more **attributes**: named, *constant* (compile-time) data. Attributes are how MLIR attaches data where a runtime variable is never allowed. |
| `(tensor<2x3xf64>) -> tensor<3x2xf64>` | A functional **type signature**: operand types → result types. |
| `loc("example/file/path":12:1)` | The **source location**. |

An operation may additionally have:

- **Regions** — a list of attached bodies. A region contains **blocks**, each block contains an ordered list of operations and may have **block arguments** (MLIR's functional-style replacement for PHI nodes). Our `toy.func` op has one region: the function body.
- **Successors** — target blocks for branch-like terminators (Toy doesn't need these yet).

So the structural recursion is: **operation → regions → blocks → operations → …** That single recursion expresses modules, functions, loops, and straight-line code alike.

### 2.2 Locations are mandatory

In LLVM IR, debug locations are optional metadata that passes routinely drop. In MLIR, **every operation has a mandatory source location**. If you don't have a meaningful one you must *explicitly* say so (`loc(unknown)` / `builder.getUnknownLoc()`). This inverts the default: transformations must consciously decide what location a new/replacement op gets, so provenance survives lowering, which is essential for debugging and for good diagnostics. You will see `loc(...)` on every line of our output in section 7 (locations are printed only when asked for, via `-mlir-print-debuginfo`).

### 2.3 Dialects

A **dialect** groups operations, attributes, and types under a unique namespace — think of it as a C++ namespace plus a registration point for parsing/printing/verification hooks. MLIR ships with many (e.g. `func`, `arith`, `tensor`, `llvm`), and they coexist in one module: that is the "multi-level" in Multi-Level IR — you lower *gradually* from `toy` to lower-level dialects rather than jumping straight to LLVM IR.

### 2.4 The opaque API: MLIR works even on ops it has never heard of

A remarkable property: MLIR can *parse, print, and round-trip* IR containing completely **unregistered** operations. Everything in the generic form above is structurally self-describing (name string, operand list, attribute dictionary, type signature), so `mlir-opt` can handle this without knowing anything about `toy`:

```mlir
func.func @toy_func(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> {
  %t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
  return %t_tensor : tensor<3x2xf64>
}
```

Pipe that through `mlir-opt` (with `-allow-unregistered-dialect`) and it comes back intact. The op is treated as **opaque**: MLIR knows its structure but nothing about its meaning.

The flip side: with unregistered ops, the verifier can only check generic structure, not semantics. This IR is nonsense — `toy.print` "produces" a tensor out of thin air and the function has no terminator semantics attached — yet it passes:

```mlir
func.func @main() {
  %0 = "toy.print"() : () -> tensor<2x3xf64>
}
```

Working opaquely is possible but discouraged. **Registering** a dialect and its operations buys you: verification of invariants (the errors above become hard failures), nice accessor methods instead of stringly-typed attribute lookups, custom (pretty) assembly syntax, and hooks for optimization. That is exactly what the rest of this chapter does.

### 2.5 `Operation` vs. `Op`

Two C++ classes are easy to confuse:

- `mlir::Operation` — the *generic*, opaque runtime object. Every op instance is an `Operation` under the hood; it exposes generic APIs (`getOperands()`, `getAttrs()`, `getLoc()`, …).
- `mlir::Op` derivatives (e.g. our `ConstantOp`) — thin, *typed* smart-pointer wrappers around an `Operation*`. They add op-specific accessors and are passed **by value**.

You move between the two with LLVM-style casts:

```cpp
void processConstantOp(mlir::Operation *operation) {
  ConstantOp op = llvm::dyn_cast<ConstantOp>(operation);
  if (!op)          // not a toy.constant
    return;
  // Get back the generic Operation* wrapped inside ConstantOp:
  mlir::Operation *internalOperation = op.getOperation();
  assert(internalOperation == operation);
}
```

---

## 3. Defining the Toy Dialect

### 3.1 What it would look like in raw C++

A dialect is modeled by a class deriving from `mlir::Dialect`. Defined by hand it would be:

```cpp
class ToyDialect : public mlir::Dialect {
public:
  explicit ToyDialect(mlir::MLIRContext *ctx);

  /// The dialect namespace: the "toy" in "toy.transpose".
  static llvm::StringRef getDialectNamespace() { return "toy"; }

  /// Called by the constructor: registers attributes, operations, types, ...
  void initialize();
};
```

### 3.2 What this repo actually does: ODS

Instead of writing that boilerplate, we declare the dialect in TableGen and let `mlir-tblgen` generate the C++. This is the very top of [`include/toy/Ops.td`](Ch2/include/toy/Ops.td):

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

(The upstream doc's version also sets `summary`/`description`, which feed generated documentation; this repo keeps it minimal.) Running the `-gen-dialect-decls` generator on this yields exactly the boilerplate class — here is the **actual generated code** from `Ch2/build/dialect-decls.inc` in this repo (produced by `run_mlir-tblgen.sh`, section 6.3):

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

and `-gen-dialect-defs` produces the constructor/destructor (`Ch2/build/dialect-defs.inc`):

```cpp
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::toy::ToyDialect)

ToyDialect::ToyDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
                      ::mlir::TypeID::get<ToyDialect>()) {
  initialize();
}

ToyDialect::~ToyDialect() = default;
```

The only thing left for us to write by hand is `initialize()`, which registers the operations. In [`mlir/Dialect.cpp`](Ch2/mlir/Dialect.cpp):

```cpp
#include "toy/Dialect.cpp.inc"   // the generated ctor/dtor above

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"       // expands to: ConstantOp, AddOp, FuncOp, ...
      >();
}
```

`GET_OP_LIST` is a guard macro: the same generated file `Ops.cpp.inc` contains both a comma-separated list of all op classes and their full method definitions; the macro selects which slice gets included.

### 3.3 Header wiring and loading the dialect

[`include/toy/Dialect.h`](Ch2/include/toy/Dialect.h) is nothing but glue — it includes interface headers the generated code needs, then the two generated declaration files:

```cpp
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

/// Auto-generated dialect declaration.
#include "toy/Dialect.h.inc"

/// Auto-generated op class declarations.
#define GET_OP_CLASSES
#include "toy/Ops.h.inc"
```

Finally, a dialect must be **loaded into the `MLIRContext`** before use — contexts only load the builtin dialect by default. The driver [`toyc.cpp`](Ch2/toyc.cpp) does this first thing in `dumpMLIR()`:

```cpp
mlir::MLIRContext context;
// Load our Dialect in this MLIR Context.
context.getOrLoadDialect<mlir::toy::ToyDialect>();
```

Without this line, parsing a `.mlir` file containing `toy.*` ops would fail with "dialect 'toy' not found".

---

## 4. Defining Toy Operations with ODS

### 4.1 The C++ alternative (what ODS saves you from)

An operation class derives from the CRTP template `mlir::Op`, parameterized by **traits** that inject verification and accessors. `toy.constant` written by hand would be:

```cpp
class ConstantOp : public mlir::Op<
                     /// `mlir::Op` is a CRTP class, meaning that we provide the
                     /// derived class as a template parameter.
                     ConstantOp,
                     /// The ConstantOp takes zero input operands.
                     mlir::OpTrait::ZeroOperands,
                     /// The ConstantOp returns a single result.
                     mlir::OpTrait::OneResult,
                     /// The result type must be a TensorType.
                     mlir::OpTrait::OneTypedResult<TensorType>::Impl> {
public:
  /// Inherit the constructors from the base Op class.
  using Op::Op;

  /// The unique name of this operation.
  static llvm::StringRef getOperationName() { return "toy.constant"; }

  /// Fetch the "value" attribute with a typed accessor.
  mlir::DenseElementsAttr getValue();

  /// Custom semantic verification, called after the trait verifiers.
  LogicalResult verifyInvariants();

  /// build() methods populate the OperationState that
  /// builder.create<ConstantOp>(...) uses to create the op.
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Type result, mlir::DenseElementsAttr value);
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::DenseElementsAttr value);
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    double value);
};
```

That is a lot of mechanical code per op — and it drifts as MLIR's APIs evolve. **ODS** (Operation Definition Specification) instead describes each op declaratively in TableGen; `mlir-tblgen` expands it into (better, always-up-to-date versions of) the above. ODS is *the* recommended way to define ops.

All Toy ops share a base class that fixes the parent dialect:

```tablegen
// Base class for toy dialect operations. Provides: the parent dialect,
// the mnemonic (op name without the "toy." prefix), and a trait list.
class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;
```

Now let's go through **every op** in [`Ops.td`](Ch2/include/toy/Ops.td). Watch for the recurring `let` fields:

- `arguments` / `results` — typed operands **and attributes** in, values out. Naming an entity (`$value`) generates accessors (`getValue()`).
- `builders` — extra convenience `build()` overloads.
- `hasVerifier` — declares a hand-written `verify()` in `Dialect.cpp`.
- `assemblyFormat` / `hasCustomAssemblyFormat` — declarative vs. hand-written pretty syntax.

### 4.2 `ConstantOp` — attributes, builders, verifier, custom parser/printer

```tablegen
def ConstantOp : Toy_Op<"constant", [Pure]> {
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. ...
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

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

Point by point:

- **Trait `Pure`**: no side effects → dead constants can be eliminated, and it enables later folding/CSE. (This subsumes what older MLIR called `NoSideEffect`.)
- **`arguments`**: note that an *attribute* (`F64ElementsAttr`, a dense array of f64) appears in `ins` right alongside where operands would go. ODS distinguishes them by the constraint kind. Naming it `$value` generates `getValue()` returning `DenseElementsAttr`.
- **`results`**: `F64Tensor` is a predefined ODS constraint = "tensor of 64-bit float". Because it's unnamed, only generic result accessors are produced.
- **`builders`**: `builder.create<ConstantOp>(loc, ...)` needs a matching `build()` overload. ODS always auto-generates one taking (result types, operands, attributes); here we add two ergonomic ones. The first has an *inline* body (the `[{ ... }]` blob — `$_builder`/`$_state` are magic substitutions) that delegates to the autogenerated overload. The second only *declares* `build(OpBuilder&, OperationState&, double)`; its body lives in `Dialect.cpp`:

```cpp
void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}
```

- **`hasVerifier = 1`**: generates a declaration `llvm::LogicalResult ConstantOp::verify()`; ODS-generated structural checks (`verifyInvariantsImpl`) run *first*, then our semantic check. The implementation in `Dialect.cpp` checks that the result tensor's shape matches the attribute's shape:

```cpp
llvm::LogicalResult ConstantOp::verify() {
  // If the return type is an unranked tensor, any attribute shape is fine.
  auto resultType = llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!resultType)
    return success();

  // Rank must match...
  auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
  if (attrType.getRank() != resultType.getRank())
    return emitOpError("return type must match the one of the attached value "
                       "attribute: ") << attrType.getRank() << " != "
                                      << resultType.getRank();

  // ...and every dimension too.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim)
    if (attrType.getShape()[dim] != resultType.getShape()[dim])
      return emitOpError("return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
  return mlir::success();
}
```

- **`hasCustomAssemblyFormat = 1`**: instead of the verbose generic form `%0 = "toy.constant"() {value = dense<...>} : () -> tensor<2x3xf64>`, we want `%0 = toy.constant dense<...> : tensor<2x3xf64>`. This flag makes ODS declare `parse()`/`print()` which we implement by hand in `Dialect.cpp`:

```cpp
/// OpAsmParser methods return ParseResult: a LogicalResult wrapper that
/// converts to `true` on FAILURE, allowing chains of `||`.
mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());  // result type = attribute type
  return success();
}

void ConstantOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << getValue();
}
```

Note the trick in `parse`: the result type is not written in the pretty syntax at all — it is *recovered* from the parsed attribute's type. Custom syntax may omit anything that is reconstructible.

### 4.3 `AddOp` and `MulOp` — binary element-wise ops with a shared hand-written syntax

```tablegen
def AddOp : Toy_Op<"add"> {
  let summary = "element-wise addition operation";
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
  let hasCustomAssemblyFormat = 1;
  let builders = [ OpBuilder<(ins "Value":$lhs, "Value":$rhs)> ];
}
```

(`MulOp` = same shape with mnemonic `"mul"`.) The declared two-`Value` builder is implemented in `Dialect.cpp`; note that the result type is *always* an unranked `tensor<*xf64>` at this stage — shape inference is Chapter 4's job:

```cpp
void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}
```

Both ops share one hand-written parser/printer pair, showing why you'd sometimes prefer C++ over the declarative format — *conditional* syntax. If operand and result types all match, print the type once (`toy.mul %0, %1 : tensor<*xf64>`); otherwise print a full functional type:

```cpp
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

The matching `parseBinaryOp` parses exactly two operands, an optional attribute dict, then a colon-type; if that type turns out to be a `FunctionType` it distributes inputs/results accordingly, otherwise the single type is used for both operands and the result. `AddOp::parse/print` and `MulOp::parse/print` are one-line dispatches to these helpers.

### 4.4 `FuncOp` — regions, interfaces, `extraClassDeclaration`

The most structurally interesting op — this *is* a Toy function:

```tablegen
def FuncOp : Toy_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  let summary = "user defined function operation";

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
- **`skipDefaultBuilders = 1`**: suppress the autogenerated builders (they wouldn't create the entry block properly); our single custom builder in `Dialect.cpp` leans on the interface helper:

```cpp
void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  // FunctionOpInterface provides a convenient `build` that populates the
  // state AND creates an entry block whose arguments mirror the inputs.
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}
```

- The custom `parse`/`print` also just delegate to library helpers (`mlir::function_interface_impl::parseFunctionOp` / `printFunctionOp`), which is why `toy.func @main() { ... }` looks exactly like `func.func`.

### 4.5 `GenericCallOp` — symbol references and variadic operands

```tablegen
def GenericCallOp : Toy_Op<"generic_call"> {
  let summary = "generic call operation";

  // The callee is a symbol reference attribute, plus variadic tensor inputs.
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);
  let results = (outs F64Tensor);

  // Declarative assembly format this time:
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
  ];
}
```

- **`FlatSymbolRefAttr`**: the callee is not an SSA operand but a *symbol reference* (`@multiply_transpose`) — calls reference functions by name, resolved through MLIR's symbol tables.
- **`Variadic<F64Tensor>`**: any number of tensor operands.
- **`assemblyFormat`** (declarative this time): variables (`$callee`, `$inputs`) print/parse the corresponding argument; back-ticked literals (`` `(` ``) are punctuation; `attr-dict` is the mandatory directive for remaining attributes; `functional-type($inputs, results)` prints `(operand types) -> result types`. Result: `toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>`.
- The builder (in `Dialect.cpp`) sets the result to unranked and stores the callee as an attribute:

```cpp
void GenericCallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          StringRef callee, ArrayRef<mlir::Value> arguments) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(arguments);
  state.addAttribute("callee",
                     mlir::SymbolRefAttr::get(builder.getContext(), callee));
}
```

### 4.6 `PrintOp` — the minimal declarative op

The official docs use `PrintOp` to contrast a hand-written parser/printer (about 20 lines of C++ manipulating `OpAsmParser`/`OpAsmPrinter`) with the declarative one-liner. This repo uses the declarative form:

```tablegen
def PrintOp : Toy_Op<"print"> {
  let summary = "print operation";
  let arguments = (ins F64Tensor:$input);
  // No results, no builders, no verifier needed.
  let assemblyFormat = "$input attr-dict `:` type($input)";
}
```

`type($input)` is the directive that prints/parses the type of `$input` — producing `toy.print %5 : tensor<*xf64>`. For comparison, the equivalent hand-written version from the docs:

```cpp
void PrintOp::print(mlir::OpAsmPrinter &printer) {
  printer << "toy.print " << op.input();
  printer.printOptionalAttrDict(op.getAttrs());
  printer << " : " << op.input().getType();
}

mlir::ParseResult PrintOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand inputOperand;
  mlir::Type inputType;
  if (parser.parseOperand(inputOperand) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColon() || parser.parseType(inputType))
    return mlir::failure();
  if (parser.resolveOperand(inputOperand, inputType, result.operands))
    return mlir::failure();
  return mlir::success();
}
```

One TableGen line replaces all of that. Prefer the declarative format whenever the syntax is regular.

### 4.7 `ReshapeOp` — result type constraints

```tablegen
def ReshapeOp : Toy_Op<"reshape"> {
  let summary = "tensor reshape operation";
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

### 4.8 `ReturnOp` — traits (`Terminator`, `HasParent`), optional operands

```tablegen
def ReturnOp : Toy_Op<"return", [Pure, HasParent<"FuncOp">, Terminator]> {
  let summary = "return operation";

  // The return operation takes an OPTIONAL input operand.
  let arguments = (ins Variadic<F64Tensor>:$input);

  // Only print the operand group if present: the `?` makes the anchored
  // group (marked by `^`) optional.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>
  ];

  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  let hasVerifier = 1;
}
```

- **`Terminator`**: this op must be the last operation in its block (MLIR requires every block to end with a terminator).
- **`HasParent<"FuncOp">`**: structurally valid only directly inside a `toy.func`. Because of this trait, the verifier can safely do `cast<FuncOp>((*this)->getParentOp())` without checking.
- **Optional operand via `Variadic`**: ODS has no dedicated "0 or 1"-with-this-syntax mechanism here, so a variadic list is used and the verifier caps it at one.
- **Assembly format optional group**: `($input^ ...)?` prints/parses the parenthesized group only when `$input` (the anchor `^`) is non-empty. So both `toy.return` and `toy.return %2 : tensor<*xf64>` are valid.
- The **verifier** cross-checks the return against the *enclosing function's* signature — a nice example of contextual verification:

```cpp
llvm::LogicalResult ReturnOp::verify() {
  auto function = cast<FuncOp>((*this)->getParentOp()); // safe: HasParent trait

  if (getNumOperands() > 1)
    return emitOpError() << "expects at most 1 return operand";

  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError() << "does not return the same number of values ("
                         << getNumOperands() << ") as the enclosing function ("
                         << results.size() << ")";

  if (!hasOperand())
    return mlir::success();

  auto inputType = *operand_type_begin();
  auto resultType = results.front();
  // Unranked tensors are compatible with anything (shapes unknown until Ch4).
  if (inputType == resultType ||
      llvm::isa<mlir::UnrankedTensorType>(inputType) ||
      llvm::isa<mlir::UnrankedTensorType>(resultType))
    return mlir::success();

  return emitError() << "type of return operand (" << inputType
                     << ") doesn't match function result type (" << resultType << ")";
}
```

### 4.9 `TransposeOp` — the running example

```tablegen
def TransposeOp : Toy_Op<"transpose"> {
  let summary = "transpose operation";

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];

  let builders = [ OpBuilder<(ins "Value":$input)> ];
  let hasVerifier = 1;
}
```

The verifier only fires when both shapes are known (ranked), in which case the result shape must be the reverse of the input shape:

```cpp
llvm::LogicalResult TransposeOp::verify() {
  auto inputType  = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(getType());
  if (!inputType || !resultType)
    return mlir::success();   // unranked: nothing to check yet

  auto inputShape = inputType.getShape();
  if (!std::equal(inputShape.begin(), inputShape.end(),
                  resultType.getShape().rbegin()))
    return emitError() << "expected result shape to be a transpose of the input";
  return mlir::success();
}
```

### 4.10 Demystifying TableGen: the actually-generated code

You never *have* to read the generated code, but seeing it once cures TableGen of its magic. This is the real generated `TransposeOp` class from `Ch2/build/op-decls.inc` in this repo (abridged):

```cpp
class TransposeOp : public ::mlir::Op<TransposeOp,
    ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult,
    ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl,
    ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand,
    ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Adaptor = TransposeOpAdaptor;

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("toy.transpose");
  }

  // Named-operand accessor generated from `$input`:
  ::mlir::TypedValue<::mlir::TensorType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(
        *getODSOperands(0).begin());
  }

  // Autogenerated + our declared builders:
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &, Value input);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &,
                    ::mlir::Type resultType0, ::mlir::Value input);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &,
                    ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                    ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});

  ::llvm::LogicalResult verifyInvariantsImpl();  // ODS constraint checks
  ::llvm::LogicalResult verifyInvariants();      // impl + our verify()
  ::llvm::LogicalResult verify();                // hasVerifier=1 -> we implement
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &, ::mlir::OperationState &);
  void print(::mlir::OpAsmPrinter &);
};
```

Note how the trait list we saw in the hand-written C++ version (section 4.1) got derived automatically from `arguments`/`results` (`OneOperand`, `OneResult`, `OneTypedResult<TensorType>`). And in `Ch2/build/op-defs.inc`, verification is wired so **structural checks run before your semantic verifier**:

```cpp
::llvm::LogicalResult TransposeOp::verifyInvariants() {
  if (::mlir::succeeded(verifyInvariantsImpl()) &&   // ODS type constraints
      ::mlir::succeeded(verify()))                   // our hand-written verify()
    return ::mlir::success();
  return ::mlir::failure();
}
```

Even the `assemblyFormat` string was compiled into a ~40-line `TransposeOp::parse()` that calls `parseLParen()`, `parseOperand()`, `parseColon()`, `parseKeyword("to")`, etc. — exactly what you would have written by hand. See section 6 for how to regenerate/inspect these files yourself.

---

## 5. The MLIRGen Module

With the dialect in place, [`mlir/MLIRGen.cpp`](Ch2/mlir/MLIRGen.cpp) walks the Chapter 1 AST and emits ops. The public API ([`include/toy/MLIRGen.h`](Ch2/include/toy/MLIRGen.h)) is one function:

```cpp
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          ModuleAST &moduleAST);
```

`OwningOpRef` is an RAII handle that erases the module when it goes out of scope. Internally everything lives in a private `MLIRGenImpl` class with three key members:

```cpp
class MLIRGenImpl {
  /// A "module" matches a Toy source file: a list of functions.
  mlir::ModuleOp theModule;

  /// Stateful helper for creating IR; crucially it keeps an "insertion
  /// point" where the next created operation will be placed.
  mlir::OpBuilder builder;

  /// Maps a variable name to its SSA value in the current scope. Toy has
  /// function-level scoping; ScopedHashTable pops all insertions made in a
  /// scope when that scope object dies.
  llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;
  ...
};
```

### 5.1 Locations

Every `builder.create<...>` call needs a location. A tiny helper converts Toy AST locations (file/line/col captured by the lexer) into MLIR locations:

```cpp
mlir::Location loc(const Location &loc) {
  return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file),
                                   loc.line, loc.col);
}
```

This is why the output in section 7 carries precise `loc("Ch2/codegen.toy":3:10)` annotations for free. (The file string is simply the input path as passed on the command line — `run.sh` runs from `toy/` and passes `Ch2/codegen.toy`.)

### 5.2 The module and function overloads

The top-level `mlirGen(ModuleAST&)` creates an empty `builtin.module`, codegens each function into it, and **verifies** the result — this is where all the ODS constraints and our hand-written `verify()` methods actually run:

```cpp
mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

  for (FunctionAST &f : moduleAST)
    mlirGen(f);

  // Check structural properties of the IR and invoke Toy op verifiers.
  if (failed(mlir::verify(theModule))) {
    theModule.emitError("module verification error");
    return nullptr;
  }
  return theModule;
}
```

`mlirGen(PrototypeAST&)` builds the `toy.func` header. Since Toy is dynamically shaped, **every argument is typed `tensor<*xf64>`** (unranked) and the return type starts empty:

```cpp
mlir::toy::FuncOp mlirGen(PrototypeAST &proto) {
  auto location = loc(proto.loc());
  llvm::SmallVector<mlir::Type, 4> argTypes(proto.getArgs().size(),
                                            getType(VarType{}));   // tensor<*xf64>
  auto funcType = builder.getFunctionType(argTypes, {});
  return builder.create<mlir::toy::FuncOp>(location, proto.getName(), funcType);
}
```

`mlirGen(FunctionAST&)` orchestrates a whole function and shows all three members working together:

```cpp
mlir::toy::FuncOp mlirGen(FunctionAST &funcAST) {
  // 1. Open a new variable scope (RAII: popped at end of function).
  ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

  // 2. Create the FuncOp at the end of the module.
  builder.setInsertionPointToEnd(theModule.getBody());
  mlir::toy::FuncOp function = mlirGen(*funcAST.getProto());
  if (!function)
    return nullptr;

  // 3. Map each Toy parameter name to the corresponding entry-block argument.
  mlir::Block &entryBlock = function.front();
  for (const auto nameValue :
       llvm::zip(funcAST.getProto()->getArgs(), entryBlock.getArguments()))
    if (failed(declare(std::get<0>(nameValue)->getName(),
                       std::get<1>(nameValue))))
      return nullptr;

  // 4. All subsequent ops go at the start of the entry block.
  builder.setInsertionPointToStart(&entryBlock);

  // 5. Emit the body; on failure erase the half-built function.
  if (mlir::failed(mlirGen(*funcAST.getBody()))) {
    function.erase();
    return nullptr;
  }

  // 6. Ensure a terminator: implicitly `toy.return` if none was written.
  ReturnOp returnOp;
  if (!entryBlock.empty())
    returnOp = dyn_cast<ReturnOp>(entryBlock.back());
  if (!returnOp) {
    builder.create<ReturnOp>(loc(funcAST.getProto()->loc()));
  } else if (returnOp.hasOperand()) {
    // The function actually returns something: patch the function type
    // to return one (unranked) tensor.
    function.setType(builder.getFunctionType(
        function.getFunctionType().getInputs(), getType(VarType{})));
  }
  return function;
}
```

Two things worth internalizing: the **insertion point** is how the builder knows *where* ops land (module end for the `FuncOp` itself, entry-block start for its body); and `declare()` (a `symbolTable.count/insert` wrapper) rejects double declarations in a scope.

### 5.3 Expression overloads

Codegen dispatches on the AST node kind (LLVM-style RTTI):

```cpp
mlir::Value mlirGen(ExprAST &expr) {
  switch (expr.getKind()) {
  case toy::ExprAST::Expr_BinOp:   return mlirGen(cast<BinaryExprAST>(expr));
  case toy::ExprAST::Expr_Var:     return mlirGen(cast<VariableExprAST>(expr));
  case toy::ExprAST::Expr_Literal: return mlirGen(cast<LiteralExprAST>(expr));
  case toy::ExprAST::Expr_Call:    return mlirGen(cast<CallExprAST>(expr));
  case toy::ExprAST::Expr_Num:     return mlirGen(cast<NumberExprAST>(expr));
  default: /* emitError: unhandled expr kind */ return nullptr;
  }
}
```

Each overload maps to one or two Toy ops:

- **`BinaryExprAST`** → `toy.add` / `toy.mul`. Recurses into LHS then RHS first (so operand ops are emitted before the op that uses them — SSA order), then:

```cpp
switch (binop.getOp()) {
case '+': return builder.create<AddOp>(location, lhs, rhs);
case '*': return builder.create<MulOp>(location, lhs, rhs);
}
```

- **`VariableExprAST`** → no op at all, just a **symbol-table lookup**; an unknown name produces a proper diagnostic anchored at the source location:

```cpp
if (auto variable = symbolTable.lookup(expr.getName()))
  return variable;
emitError(loc(expr.loc()), "error: unknown variable '") << expr.getName() << "'";
return nullptr;
```

- **`LiteralExprAST`** → `toy.constant`. The nested array literal is flattened into a `std::vector<double>` by the recursive `collectData()` helper, wrapped in a `DenseElementsAttr` typed `tensor<2x3xf64>` (etc.), and attached to the op — this is the canonical "constant data goes into attributes" pattern:

```cpp
mlir::Value mlirGen(LiteralExprAST &lit) {
  auto type = getType(lit.getDims());              // ranked tensor type

  std::vector<double> data;                        // flattened values
  data.reserve(...); collectData(lit, data);

  auto dataType = mlir::RankedTensorType::get(lit.getDims(),
                                              builder.getF64Type());
  auto dataAttribute =
      mlir::DenseElementsAttr::get(dataType, llvm::ArrayRef(data));

  // Invokes ConstantOp::build(builder, state, type, dataAttribute).
  return builder.create<ConstantOp>(loc(lit.loc()), type, dataAttribute);
}
```

- **`NumberExprAST`** → `toy.constant` via the convenience `double` builder we declared in ODS: `builder.create<ConstantOp>(loc(num.loc()), num.getValue());`
- **`CallExprAST`** → the builtin `transpose(x)` becomes `toy.transpose` (with an arity check); *any other* callee becomes a `toy.generic_call` carrying the callee name as a symbol attribute:

```cpp
if (callee == "transpose") {
  if (call.getArgs().size() != 1) { /* emitError */ return nullptr; }
  return builder.create<TransposeOp>(location, operands[0]);
}
return builder.create<GenericCallOp>(location, callee, operands);
```

- **`PrintExprAST`** → `toy.print` on the codegen'd argument; **`ReturnExprAST`** → `toy.return` with zero or one operand.
- **`VarDeclExprAST`** (`var a<2,3> = ...;`) → codegen the initializer, then, if the declaration specifies a shape, insert a **`toy.reshape`** to that ranked type, and finally `declare()` the name:

```cpp
mlir::Value value = mlirGen(*init);
if (!vardecl.getType().shape.empty())
  value = builder.create<ReshapeOp>(loc(vardecl.loc()),
                                    getType(vardecl.getType()), value);
if (failed(declare(vardecl.getName(), value)))
  return nullptr;
```

This is why `var a<2, 3> = [[1, 2, 3], [4, 5, 6]]` produces a constant *plus* a (here redundant) reshape — Chapter 3 will optimize such reshapes away.

- **`ExprASTList`** (a block of statements) opens another `ScopedHashTableScope` and dispatches per statement kind.

Finally the type helper — the root of all the `tensor<*xf64>` in our output:

```cpp
mlir::Type getType(ArrayRef<int64_t> shape) {
  if (shape.empty())   // no shape info -> unranked
    return mlir::UnrankedTensorType::get(builder.getF64Type());
  return mlir::RankedTensorType::get(shape, builder.getF64Type());
}
```

### 5.4 The driver

[`toyc.cpp`](Ch2/toyc.cpp) gains a `-emit=mlir` action next to Chapter 1's `-emit=ast`, plus an input-kind switch. `dumpMLIR()` handles **two input paths**, which is what enables the round-trip test:

```cpp
int dumpMLIR() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::toy::ToyDialect>();

  // Path A: '.toy' source -> lex/parse -> AST -> mlirGen -> dump.
  if (inputType != InputType::MLIR &&
      !llvm::StringRef(inputFilename).ends_with(".mlir")) {
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST) return 6;
    mlir::OwningOpRef<mlir::ModuleOp> module = mlirGen(context, *moduleAST);
    if (!module) return 1;
    module->dump();
    return 0;
  }

  // Path B: '.mlir' text -> MLIR parser (using OUR registered dialect,
  // including the custom ConstantOp::parse etc.) -> dump.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  ...
  module->dump();
  return 0;
}
```

`main()` also registers the MLIR/asm-printer command-line option categories — that is where the `-mlir-print-debuginfo` flag used in `run.sh` comes from:

```cpp
mlir::registerAsmPrinterCLOptions();
mlir::registerMLIRContextCLOptions();
cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");
```

---

## 6. Building

### 6.1 CMake walkthrough (this repo's out-of-tree superbuild)

Upstream builds Toy *inside* the llvm-project tree with helper macros like `add_toy_chapter`. This repo instead builds all chapters as one **standalone superbuild** against Homebrew LLVM/MLIR 20 (macOS, Ninja). The top-level [`CMakeLists.txt`](CMakeLists.txt) at `toy/` does the expensive common setup exactly once, then pulls in every chapter:

```cmake
cmake_minimum_required(VERSION 3.20)
if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "macOS Deployment Target" FORCE)
endif()
project(toy-tutorial)

# 1. Find the installed MLIR/LLVM packages (uses MLIR_DIR / LLVM_DIR).
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

# 2. Make their CMake modules discoverable...
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# 3. ...and include the macros that define mlir_tablegen() etc.
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})

# Collect every chapter binary in build/bin/ instead of build/ChN/.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(Ch1)
add_subdirectory(Ch2)
# ... Ch3 through Ch7
```

Where do `MLIR_DIR`, `LLVM_DIR`, and the compiler come from? [`CMakePresets.json`](CMakePresets.json) pins them, so `cmake --preset default` needs no `-D` flags:

```json
{
  "name": "default",
  "displayName": "Homebrew LLVM/MLIR 20 (Ninja, Release)",
  "generator": "Ninja",
  "binaryDir": "${sourceDir}/build",
  "cacheVariables": {
    "CMAKE_BUILD_TYPE": "Release",
    "CMAKE_C_COMPILER": "/opt/homebrew/opt/llvm@20/bin/clang",
    "CMAKE_CXX_COMPILER": "/opt/homebrew/opt/llvm@20/bin/clang++",
    "MLIR_DIR": "/opt/homebrew/opt/llvm@20/lib/cmake/mlir",
    "LLVM_DIR": "/opt/homebrew/opt/llvm@20/lib/cmake/llvm"
  }
}
```

The chapter's own [`Ch2/CMakeLists.txt`](Ch2/CMakeLists.txt) is **dual-mode**: the same `find_package`/`include` boilerplate as the top level is wrapped in a guard that fires only when the chapter is configured *directly* (standalone mode); in the superbuild the top level has already done it:

```cmake
# Runs only when this chapter is configured directly (cmake -S Ch2).
# In the superbuild (cmake -S toy/), ../CMakeLists.txt already did all this.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(toy-ch2)
  find_package(MLIR REQUIRED CONFIG)
  find_package(LLVM REQUIRED CONFIG)
  # ... CMAKE_MODULE_PATH, include(TableGen/AddLLVM/AddMLIR/HandleLLVMOptions),
  #     include_directories — same boilerplate as the top level
endif()
```

The chapter targets below the guard are unchanged by the restructure:

```cmake
add_subdirectory(include)          # runs TableGen (see below)

add_executable(toyc-ch2
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  )

# Ensure the generated .inc files exist before compiling anything that
# includes toy/Dialect.h.
add_dependencies(toyc-ch2 ToyCh2OpsIncGen)

include_directories(include/)                          # Ops.td, headers
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)  # generated .inc

target_link_libraries(toyc-ch2
  PRIVATE
    MLIR     # libMLIR.dylib (all dialects, passes, conversions)
    LLVM     # libLLVM.dylib (all targets, all components)
    )
```

Notable differences from upstream:

- **Monolithic shared libraries**: Homebrew's LLVM ships `libMLIR.dylib` / `libLLVM.dylib`, so instead of listing fine-grained components (`MLIRAnalysis`, `MLIRIR`, `MLIRParser`, …) we link just two libraries.
- **Explicit `add_dependencies(toyc-ch2 ToyCh2OpsIncGen)`**: in-tree helper macros normally add this dependency for you. Out-of-tree, without it, Ninja may try to compile `Dialect.cpp` before TableGen has produced `toy/Ops.h.inc` — a classic build race. This line was added to fix exactly that.
- Two include roots: the *source* `include/` (for `Ops.td`, `Dialect.h`) and the chapter's *binary* include dir (`${CMAKE_CURRENT_BINARY_DIR}/include/`), where the generated `.inc` files land mirroring the source layout. In the superbuild that is `build/Ch2/include/toy/*.inc`; in a standalone chapter build it is `Ch2/build/include/toy/*.inc`.

The TableGen wiring itself lives in [`include/toy/CMakeLists.txt`](Ch2/include/toy/CMakeLists.txt) (reached via `include/CMakeLists.txt`, which is just `add_subdirectory(toy)`):

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

### 6.2 build.sh

The top-level [`build.sh`](build.sh) drives one shared, **incremental** build tree for all chapters:

```bash
cd /Users/roy/study/mlir/toy
./build.sh ch2          # build only toyc-ch2
```

Under the hood it configures once with the preset if (and only if) `build/CMakeCache.txt` doesn't exist yet, then builds the requested target:

```bash
cmake --preset default                                # first time only
cmake --build --preset default --target toyc-ch2      # or no --target for all
```

Other invocations: `./build.sh` (everything), `./build.sh --fresh` (wipe `build/` and rebuild), `./build.sh ch2 --fresh`. There is no `rm -rf build` on the normal path — Ninja re-runs CMake automatically whenever a `CMakeLists.txt` changes, so rebuilds are incremental. The result is `build/bin/toyc-ch2` (all chapter binaries are collected in `build/bin/` via `CMAKE_RUNTIME_OUTPUT_DIRECTORY`).

Thanks to the dual-mode guard, **standalone mode still works** — configure the chapter directly, passing what the preset would have provided:

```bash
cd /Users/roy/study/mlir/toy
cmake -S Ch2 -B Ch2/build -G Ninja \
  -DMLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir \
  -DLLVM_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/llvm
cmake --build Ch2/build
```

That produces `Ch2/build/toyc-ch2`, which `run.sh` will find as a fallback when `build/bin/toyc-ch2` doesn't exist.

### 6.3 Inspecting TableGen output by hand: run_mlir-tblgen.sh

You don't need CMake to see what ODS generates — [`run_mlir-tblgen.sh`](Ch2/run_mlir-tblgen.sh) (run from inside `Ch2/`; unchanged by the superbuild restructure) invokes `mlir-tblgen` directly, once per generator, with the Homebrew MLIR headers on the include path (needed to resolve `include "mlir/IR/OpBase.td"` etc.):

```bash
mlir-tblgen -gen-dialect-decls ./include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include/ -o ./build/dialect-decls.inc
mlir-tblgen -gen-dialect-defs  ./include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include/ -o ./build/dialect-defs.inc
mlir-tblgen -gen-op-decls      ./include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include/ -o ./build/op-decls.inc
mlir-tblgen -gen-op-defs       ./include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include/ -o ./build/op-defs.inc
```

The outputs land in `Ch2/build/` (a scratch location, separate from the superbuild's `.inc` files under `build/Ch2/include/`) as `dialect-decls.inc`, `dialect-defs.inc`, `op-decls.inc` (~1600 lines), and `op-defs.inc` (~1600 lines) — these are exactly the files quoted in sections 3.2 and 4.10. Other useful generators to try: `mlir-tblgen -gen-op-doc ./include/toy/Ops.td -I ...` renders the `summary`/`description` fields as markdown documentation.

---

## 7. Running and Testing

### 7.1 run.sh

```bash
cd /Users/roy/study/mlir/toy
./run.sh ch2
```

The top-level [`run.sh`](run.sh) looks up the binary in `build/bin/` (superbuild) first, falling back to `Ch2/build/` (standalone chapter build), and its `run_ch2` function executes three commands:

```bash
# 1. Compile Ch2/codegen.toy to MLIR, printing source locations.
./build/bin/toyc-ch2 Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo

# 2+3. Round trip: save the MLIR text to a temp file, then RE-PARSE it and
#      emit again. The temp file is deleted afterwards.
tmp=$(mktemp -t toy-ch2-codegen).mlir
./build/bin/toyc-ch2 Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo 2> "$tmp"
./build/bin/toyc-ch2 "$tmp" -emit=mlir
rm -f "$tmp"
```

Flags:

- `-emit=mlir` — take the driver's `DumpMLIR` action (vs. `-emit=ast` from Chapter 1).
- `-mlir-print-debuginfo` — ask the asm printer to include `loc(...)` on every operation (locations are always *stored*, just not printed by default). This flag exists because `main()` called `mlir::registerAsmPrinterCLOptions()`.
- Note the `2>` in step 2: `module->dump()` writes to **stderr**, so stderr is what gets captured into the temp file. The round trip no longer leaves a `codegen.mlir` behind in the build tree — the intermediate file is a `mktemp` scratch file, removed once step 3 has re-parsed it.
- In step 3 the input file ends in `.mlir` (that is why a `.mlir` suffix is appended to the `mktemp` name), so the driver takes *Path B* (section 5.4): it uses MLIR's parser — and therefore our dialect's registered `parse()` methods — instead of the Toy frontend. (`-x mlir` would force this for any extension.)

### 7.2 The input

[`codegen.toy`](Ch2/codegen.toy):

```
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

### 7.3 Actual captured output (step 1), annotated

This is the real output of `./build/bin/toyc-ch2 Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo` on this machine (run from `/Users/roy/study/mlir/toy` — hence the `Ch2/` prefix in every `loc(...)`, which reproduces the input path exactly as given):

```mlir
module {
  toy.func @multiply_transpose(%arg0: tensor<*xf64> loc("Ch2/codegen.toy":2:1), %arg1: tensor<*xf64> loc("Ch2/codegen.toy":2:1)) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64> loc("Ch2/codegen.toy":3:10)
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64> loc("Ch2/codegen.toy":3:25)
    %2 = toy.mul %0, %1 : tensor<*xf64> loc("Ch2/codegen.toy":3:25)
    toy.return %2 : tensor<*xf64> loc("Ch2/codegen.toy":3:3)
  } loc("Ch2/codegen.toy":2:1)
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64> loc("Ch2/codegen.toy":7:17)
    %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64> loc("Ch2/codegen.toy":7:3)
    %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64> loc("Ch2/codegen.toy":8:17)
    %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64> loc("Ch2/codegen.toy":8:3)
    %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("Ch2/codegen.toy":9:11)
    %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("Ch2/codegen.toy":10:11)
    toy.print %5 : tensor<*xf64> loc("Ch2/codegen.toy":11:3)
    toy.return loc("Ch2/codegen.toy":6:1)
  } loc("Ch2/codegen.toy":6:1)
} loc(unknown)
```

Line-by-line mapping back to the `.toy` source:

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

Also note the function-argument locations (`%arg0: tensor<*xf64> loc("Ch2/codegen.toy":2:1)`) — block arguments carry locations too.

### 7.4 The round trip (steps 2–3) and why it matters

Step 3 re-parses the temporary `.mlir` file and prints (actual output, no `-mlir-print-debuginfo` this time so no `loc(...)`):

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

Semantically identical to the first emission — the round trip **passes**. Why is this a meaningful test? Because it exercises *both directions* of every custom assembly definition we wrote:

- **Emit path** exercises the *printers*: `ConstantOp::print`, `printBinaryOp`, the `assemblyFormat`-generated printers for `transpose`/`reshape`/`print`/`generic_call`/`return`, and `printFunctionOp`.
- **Re-parse path** exercises the *parsers*: `ConstantOp::parse` (including reconstructing the result type from the attribute), `parseBinaryOp` (including the matched-types shorthand `: tensor<*xf64>`), the format-generated parsers (`(` … `:` … `)` … `to` …), and `parseFunctionOp`.
- Parsing also re-runs the **verifiers**, so any structurally invalid syntax we might print would be caught immediately.

A parser/printer mismatch (say, printing `to` but parsing `into`) is one of the most common dialect bugs, and a round trip catches it instantly. This is exactly how upstream MLIR lit tests work: `toyc-ch2 ... -emit=mlir | toyc-ch2 - -x mlir -emit=mlir | FileCheck`.

### 7.5 More inputs to try

`/Users/roy/study/mlir/test_Example/Toy/Ch2/` has additional cases:

```bash
cd /Users/roy/study/mlir/toy
./build/bin/toyc-ch2 ../test_Example/Toy/Ch2/scalar.toy -emit=mlir   # scalar constant + reshape
./build/bin/toyc-ch2 ../test_Example/Toy/Ch2/empty.toy  -emit=mlir   # empty main -> implicit toy.return
./build/bin/toyc-ch2 ../test_Example/Toy/Ch2/ast.toy    -emit=ast    # Chapter 1 action still works
./build/bin/toyc-ch2 ../test_Example/Toy/Ch2/invalid.mlir -emit=mlir # exercises parser diagnostics
```

`invalid.mlir` is the negative test: it contains malformed Toy IR, and the point is to watch the *registered* dialect reject it with a precise diagnostic instead of accepting it opaquely (contrast with section 2.4).

---

## 8. Key Takeaways & Pitfalls

**Takeaways**

1. **One concept scales the whole IR**: operations (with operands, results, attributes, regions, locations) model everything from modules to arithmetic. Dialects namespace them; contexts load dialects.
2. **Registered beats opaque**: MLIR will happily round-trip unknown ops, but only registration buys verification, typed accessors, pretty syntax, and a foundation for optimization.
3. **ODS is leverage**: ~30 lines of TableGen per op replace hundreds of lines of brittle C++ (compare `Ops.td` with `op-decls.inc`/`op-defs.inc`). Custom C++ remains available exactly where declarativeness runs out (conditional syntax like `printBinaryOp`, semantic verifiers, non-trivial builders).
4. **Verification is layered**: ODS constraints (`F64Tensor`, `StaticShapeTensorOf`, trait invariants) run automatically in `verifyInvariantsImpl()`; `hasVerifier = 1` appends your semantic `verify()`. `mlir::verify(module)` in MLIRGen triggers the whole stack.
5. **MLIRGen is small on purpose**: a stateful `OpBuilder` (insertion point!), a `ScopedHashTable` symbol table, a `loc()` helper, and one `mlirGen` overload per AST node. Shapes are deliberately left unranked (`tensor<*xf64>`) — inference comes in Chapter 4, reshape/transpose cleanups in Chapter 3.
6. **Round-tripping is the cheapest dialect test you'll ever write** — it validates printer and parser against each other.

**Pitfalls**

- **Forgetting `context.getOrLoadDialect<ToyDialect>()`** — parsing `.mlir` input then fails with an unregistered-dialect error even though the code compiled fine.
- **Out-of-tree TableGen races**: without `add_dependencies(toyc-ch2 ToyCh2OpsIncGen)`, Ninja can compile `Dialect.cpp` before `Ops.h.inc` exists. In-tree builds hide this because their helper macros add the dependency; out-of-tree builds (superbuild or standalone) must do it explicitly (this repo's CMakeLists carries a comment about exactly this fix).
- **`GET_OP_LIST` vs. `GET_OP_CLASSES`**: `Ops.cpp.inc` is multi-purpose; including it without the right guard macro gives baffling redefinition or "no ops registered" problems.
- **`dump()` writes to stderr** — hence `2>` in `run.sh`. Redirecting stdout captures nothing.
- **Locations are easy to squander**: use `loc(expr.loc())` for every `builder.create<>`; falling back to `getUnknownLoc()` everywhere destroys diagnostics quality later.
- **Verifier ordering assumption**: your `verify()` runs *after* structural checks, so you may rely on operand counts/types already being validated — but nothing more. E.g. `ReturnOp::verify` can `cast<FuncOp>` its parent only because of the `HasParent<"FuncOp">` trait.
- **Custom `parse()` must fully populate `OperationState`** — forgetting `result.addTypes(...)` (as `ConstantOp::parse` does from the attribute type) yields an op with zero results and confusing downstream errors.
- Homebrew-specific: linking granular `MLIRxxx` component libraries mixes badly with the monolithic `libMLIR.dylib`; this repo links just `MLIR` + `LLVM`.

---

## Links

- Official doc: [Toy Tutorial Chapter 2 — Emitting Basic MLIR](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/)
- Related MLIR docs: [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/) · [Operation Definition Specification (ODS)](https://mlir.llvm.org/docs/DefiningDialects/Operations/) · [Declarative Assembly Format](https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format)
- Previous: [Chapter 1 — Toy Language and AST](1_toy_lang_ast.md)
- Next: [Chapter 3 — High-level Language-Specific Analysis and Transformation](3_high_level_transformations.md)
- Back to [README](README.md)
