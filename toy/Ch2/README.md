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

The lexer, parser, and AST (`Lexer.h`, `Parser.h`, `AST.h`, `parser/AST.cpp`) are carried over unchanged from Chapter 1. Everything new is introduced in the section that explains it: the ODS definitions in `include/toy/Ops.td` (sections 3–4), the dialect implementation in `mlir/Dialect.cpp` (sections 3–4), the emitter in `mlir/MLIRGen.cpp` and the extended driver (section 5).

Sections 1–7 follow the chapter's essential path. The appendices hold the reference material: the other eight ops (A), `Ops.td` as a whole, its generated code, and the hand-written `Dialect.cpp` (B), printing details and stock `mlir-opt` (C), more inputs (D), the build wiring (E), and the script that runs every command in this README (F).

### Where Chapter 2 lives in this repo

How this repo is organized — the out-of-tree CMake superbuild at `toy/`, the pinned Homebrew LLVM/MLIR 20 toolchain, and the `build.sh`/`run.sh` helpers — is documented once in the top-level [README](../README.md#repository-layout). This chapter can also be built as a standalone project (section 6.1). Chapter 2 specifics:

| Item | Location / value |
|---|---|
| Chapter 2 code | `/Users/roy/study/mlir/toy/Ch2/` |
| Build | `cd toy && ./build.sh ch2` → binary at `./build/bin/toyc-ch2` |
| Run | `cd toy && ./run.sh ch2` → `codegen.toy` with `-emit=mlir -mlir-print-debuginfo`, then a print → parse → print round trip |
| Test inputs | `/Users/roy/study/mlir/test_Example/Toy/Ch2/` (`codegen.toy`, `scalar.toy`, `empty.toy`, `ast.toy`, `invalid.mlir`) |

What changes relative to Chapter 1:

```text
Ch2/
├── CMakeLists.txt          # + TableGen wiring, MLIRGen.cpp/Dialect.cpp, MLIR + LLVM link line (appendix E)
├── toyc.cpp                # + -emit=mlir, .mlir input path, MLIR command-line options (section 5.6)
├── run_mlir-tblgen.sh      # NEW: runs mlir-tblgen by hand into build/*.inc (section 6.4)
├── examples.sh             # NEW: symlink to ../examples.sh, runs this README's commands (appendix F)
├── mlir/
│   ├── Dialect.cpp         # NEW: dialect initialize(), op verifiers, builders, parse/print (sections 3–4)
│   └── MLIRGen.cpp         # NEW: AST → Toy dialect (section 5)
├── parser/AST.cpp          # unchanged from Chapter 1
└── include/
    ├── CMakeLists.txt      # NEW: add_subdirectory(toy)
    └── toy/
        ├── CMakeLists.txt  # NEW: mlir_tablegen rules for Ops.td (appendix E)
        ├── Ops.td          # NEW: the Toy dialect and its nine ops in ODS (sections 3.2, 4.3–4.5)
        ├── Dialect.h       # NEW: includes the generated dialect and op declarations (section 3.3)
        ├── MLIRGen.h       # NEW: the mlirGen() entry point (section 5)
        ├── Ops/            # NEW: Ops0.td–Ops5.td, the walkthrough stages of section 4.4 (not built)
        └── ...             # AST.h, Lexer.h, Parser.h unchanged
```

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
| `{inplace = true}` | A dictionary of zero or more **attributes**: special operands that are always *constant*. Here, a boolean attribute named `inplace` with the value `true`. Attributes hold compile-time constant data; they are never SSA values. |
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

Tools don't *print* locations by default (locations are always stored); `-mlir-print-debuginfo` turns them on — both in `mlir-opt` (see `mlir-opt --help` for more printing options) and in `toyc-ch2`. Section 2.3 shows them in real output; they even survive a print → parse round trip (section 6.3).

### 2.3 Dialects, operations and locations in real output

One run of `toyc-ch2` makes the concepts above concrete: several dialects in one module, section 2.2's op anatomy on real ops, and a location on everything. The code producing this behavior is walked through in sections 3 and 5; the build is in section 6.1. The input is `test_Example/Toy/Ch2/codegen.toy` (section 4.5 shows it, abridged):

***test_Example/Toy/Ch2/codegen.toy***
```text
# RUN: toyc-ch2 %s -emit=mlir 2>&1 | FileCheck %s

# User defined generic function that operates on unknown shaped arguments
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}
...
```

MLIRGen (section 5) turns the Toy AST into MLIR. Two of its choices matter for the output below: it creates the outer module op with `builder.getUnknownLoc()` — the explicit "no location" choice — and it converts every Toy AST location into an MLIR `FileLineColLoc`; section 5.2 shows the code. Print the result in generic form, with locations:

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 ../../test_Example/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-op-generic -mlir-print-debuginfo 2>&1
```

Real output **[1]** (abridged; section 4.5 shows all of it):

```mlir
"builtin.module"() ({
  "toy.func"() <{function_type = (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>, sym_name = "multiply_transpose"}> ({
  ^bb0(%arg0: tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":4:1), %arg1: tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":4:1)):
    %6 = "toy.transpose"(%arg0) : (tensor<*xf64>) -> tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":5:10)
    %7 = "toy.transpose"(%arg1) : (tensor<*xf64>) -> tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":5:25)
  ...
}) : () -> () loc(unknown)
```

- **Two dialects in one module**: the outermost op is `builtin.module` from the `builtin` dialect; it holds `toy.func` and `toy.transpose` ops from our dialect. Without `-mlir-print-op-generic` the same module prints as `module { toy.func @multiply_transpose(...) ... }` — the `builtin.` prefix is elided because `builtin.module` has a custom format of its own (section 4.5).
- **The anatomy of section 2.2**, on real ops: `%6` is the result, `"toy.transpose"` the name, `(%arg0)` the operand, `(tensor<*xf64>) -> tensor<*xf64>` the functional type. `toy.func` also shows *properties* (inherent attributes, appendix C.1) in `<{...}>` and a region `({...})` whose entry block `^bb0` has two block arguments.
- **Locations everywhere**: `loc("../../test_Example/Toy/Ch2/codegen.toy":5:10)` is line 5, column 10 of the input — the `transpose(a)` call. Block arguments carry locations too. The module's `loc(unknown)` is the `getUnknownLoc()` mentioned above (section 5.2).

Two follow-up experiments with this module are shown where their context lives. First, a tool only understands the dialects it has **loaded**: `toyc-ch2` loads `toy` into its `MLIRContext` (section 3.3), while stock `mlir-opt` doesn't and can't even parse `toyc-ch2`'s default custom-form output (appendix C.2). Second, a **registered** dialect *verifies* its ops and rejects invalid IR with a precise diagnostic (section 2.4).

### 2.4 The opaque API: MLIR works even on ops it has never heard of

MLIR lets every IR element — attributes, operations, types — be customized, yet any of them can always be reduced to the fundamental concepts of section 2.2. That is what lets MLIR parse, represent, and [round-trip](https://mlir.llvm.org/getting_started/Glossary/#round-trip) IR for *any* operation, even one from a dialect nobody registered. The stock Homebrew `mlir-opt`, which knows nothing about `toy`, demonstrates this below; `toyc-ch2`, which registers `toy`, provides the contrast at the end.

#### An unregistered op round-trips

The input isn't a file in this repo, so the commands pass it on stdin: the `toy.transpose` from section 2.2 inside a `func.func`. By default `mlir-opt` refuses unregistered dialects:

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-opt <<'EOF'
func.func @toy_func(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> {
  %t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
  return %t_tensor : tensor<3x2xf64>
}
EOF
```

Real output **[2]** (exit status 1):

```text
<stdin>:2:30: error: operation being parsed with an unregistered dialect. If this is intended, please use -allow-unregistered-dialect with the MLIR tool used
  %t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
                             ^
```

With `-allow-unregistered-dialect`, it parses and prints the IR back:

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-opt -allow-unregistered-dialect <<'EOF'
func.func @toy_func(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> {
  %t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
  return %t_tensor : tensor<3x2xf64>
}
EOF
```

Real output **[3]**:

```mlir
module {
  func.func @toy_func(%arg0: tensor<2x3xf64>) -> tensor<3x2xf64> {
    %0 = "toy.transpose"(%arg0) {inplace = true} : (tensor<2x3xf64>) -> tensor<3x2xf64>
    return %0 : tensor<3x2xf64>
  }
}
```

Note that `%tensor`/`%t_tensor` came back as `%arg0`/`%0` — the value names were never part of the in-memory IR. The op itself survives, but only as structure. For unregistered attributes, operations, and types, MLIR only enforces *structural* constraints (e.g. dominance); otherwise they are completely **opaque**. MLIR has no idea whether an unregistered op can operate on particular data types, how many operands it takes, or how many results it produces. That flexibility is handy for bootstrapping, but advised against in mature systems: transformations and analyses must treat unregistered ops conservatively, and they are much harder to construct and manipulate.

**What the flag does — and does not — buy.** Reading an op from text needs two things: the parser must *understand its syntax*, and it must be *allowed to create* an op whose dialect isn't loaded. `-allow-unregistered-dialect` grants only the second — without it you get the `operation being parsed with an unregistered dialect` error above. The first comes for free here because the generic form is self-describing (quoted op name, operand list, attribute dictionary, functional type), so MLIR's core parser reads it with no dialect code at all. Appendix C.2 shows what happens with Toy's custom form, which is not self-describing.

#### Invalid IR: accepted when unregistered, rejected when registered

The previous runs showed that MLIR can *carry* an op it doesn't know. This part shows the cost: if MLIR doesn't know an op, it can't *check* it either. The **same** invalid IR passes when `toy` is unregistered and fails when `toy` is registered. Only registration changes between the two runs, so the error has to come from the dialect.

It takes three runs, because the first one fails for an unrelated reason:

| Run | Tool | Input | Outcome | What it shows |
|---|---|---|---|---|
| 1 | stock `mlir-opt` + flag | `invalid.mlir` as is | parse error | a detour: stock `mlir-opt` can't even *read* the file, so it never gets to check it |
| 2 | stock `mlir-opt` + flag | `toy.func` → `func.func` | **accepted** | toy unregistered: the invalid `toy.print` passes |
| 3 | `toyc-ch2` | `invalid.mlir` as is | **verifier error** | toy registered: the same `toy.print` is rejected |

The input is this repo's negative test. Its IR is *invalid* for Toy: it breaks three rules of `toy.print`:

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

The file mixes two syntaxes. Line 7, `toy.func @main() {`, is in Toy's **custom** form: bare op name, `@symbol`, no quotes. Line 8, `"toy.print"() : () -> ...`, is in the **generic** form, which is self-describing ("What the flag does" above). Its three violations are all on line 8.

**Run 1: stock `mlir-opt` on the file as is.** Feed the file to stock `mlir-opt`. It doesn't get past line 7, even with the flag:

```bash
cd /Users/roy/study/mlir/toy/Ch2
/opt/homebrew/opt/llvm@20/bin/mlir-opt -allow-unregistered-dialect ../../test_Example/Toy/Ch2/invalid.mlir
```

Real output **[4]** (exit status 1; the dialect list is abridged):

```text
../../test_Example/Toy/Ch2/invalid.mlir:7:1: error: Dialect `toy' not found for custom op 'toy.func' 
toy.func @main() {
^
../../test_Example/Toy/Ch2/invalid.mlir:7:1: note: Registered dialects: acc, affine, amdgpu, amx, arith, ... ; for more info on dialect registration see https://mlir.llvm.org/getting_started/Faq/#registered-loaded-dependent-whats-up-with-dialects-management
```

This is a **parse** error, not a verification error, and it says nothing about `toy.print`. Recall the two requirements above: the parser must understand the syntax, and it must be allowed to create the op. The flag covers only the second. A custom-form op can be read only by its dialect's own parser (appendix C.2), and stock `mlir-opt` doesn't have the toy dialect, so parsing stops at line 7. Line 8 is never reached. The IR isn't checked for validity at all, so this run can't prove anything about the claim.

**Run 2: toy unregistered, the invalid op passes.** To get past line 7, the `sed` replaces `toy.func` with the builtin `func.func`, which stock `mlir-opt` can parse. Its custom syntax `func.func @main() { ... }` is the same. Line 8 stays as it was: the invalid generic-form `"toy.print"`. Now parsing succeeds, the verifier runs, and it reports nothing:

```bash
cd /Users/roy/study/mlir/toy/Ch2
sed 's/toy\.func/func.func/' ../../test_Example/Toy/Ch2/invalid.mlir \
  | /opt/homebrew/opt/llvm@20/bin/mlir-opt -allow-unregistered-dialect
```

Real output **[5]** (exit status 0 — no verifier error):

```mlir
module {
  func.func @main() {
    %0 = "toy.print"() : () -> tensor<2x3xf64>
  }
}
```

`mlir-opt` verifies the module after parsing, and exit status 0 means verification passed. All three problems in line 8 are still there. MLIR misses them because, for an unregistered op, it knows only the name `toy.print`. It doesn't know the op's operand count, result count, types or traits (the "opaque" paragraph above). So each check gives the op the benefit of the doubt:

1. **Missing terminator.** A `func.func` body must end with a terminator op (normally `func.return`). Here the last op is `toy.print`, which is not one. The verifier finds a terminator by the op's `IsTerminator` trait, and an unregistered op has no traits it can look up. For all MLIR knows, `toy.print` *could* be a terminator, so it lets this pass.
2. **No operand.** `toy.print` should take one tensor operand. MLIR doesn't know how many operands it takes, so zero is as good as any other count.
3. **One result.** `toy.print` shouldn't return a value. MLIR doesn't know how many results it has, so `%0` is accepted.

The only checks left are structural ones, which hold for every op (e.g. each operand is defined before use), and this IR passes them. The output is the same IR printed back.

**Run 3: toy registered, the same op is rejected.** `toyc-ch2` registers the `toy` dialect, so it has toy's parser *and* `toy.print`'s definition. No `sed` is needed: give it the unmodified file, `toy.func` included:

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 ../../test_Example/Toy/Ch2/invalid.mlir -emit=mlir
```

Real output **[6]** (exit status 3):

```text
loc("../../test_Example/Toy/Ch2/invalid.mlir":8:8): error: 'toy.print' op requires zero results
Error can't load file ../../test_Example/Toy/Ch2/invalid.mlir
```

This time line 7 parses, because toy's parser reads `toy.func`. `toyc-ch2` loads the file with `mlir::parseSourceFile`, which parses and then verifies. The verifier now has `toy.print`'s ODS definition (appendix A.5): zero results, one `F64Tensor` operand. It rejects the op where run 2 accepted it:

- `error: 'toy.print' op requires zero results` is problem 3. The location `invalid.mlir:8:8` points at `"toy.print"` on line 8.
- Verification stops at the first failure an op reports, so problems 1 and 2 are not reported in this run. Fixing problem 3 would let the verifier reach the next one.
- When verification fails, `parseSourceFile` returns no module. The driver then prints `Error can't load file ...` and exits with status 3 (`toyc.cpp`). The RUN line's `not toyc-ch2` expects this non-zero exit.

Compare runs 2 and 3: the `toy.print` line is identical, but run 2 exits 0 and run 3 exits 3. The only thing that changed is whether the tool knows the `toy` dialect.

**Registering** the dialect and its operations is what makes the difference: the verifier learns each op's invariants, and you get typed accessor methods instead of stringly-typed attribute lookups, a custom (pretty) assembly syntax, and hooks for optimization. That is what the rest of this chapter builds.

---

## 3. Defining the Toy Dialect

To interface with MLIR effectively, we define a new `toy` dialect. It models the structure of the Toy language and gives us an easy avenue for high-level analysis and transformation.

### 3.1 What it would look like in raw C++

A dialect is a class deriving from `mlir::Dialect` that registers custom attributes, operations, and types; it can also override virtual methods to change general behavior (later chapters do this). Written by hand, it would look like this (not this repo's code):

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

MLIR also supports defining dialects — and their operations — *declaratively* in TableGen. That is much cleaner: it removes most of the boilerplate and lets documentation live right next to the definition. This repo defines the dialect and all nine ops in one file, `include/toy/Ops.td`, the heart of this chapter; section 4.3 explains how that file is written. The dialect is its first record:

***include/toy/Ops.td***
```tablegen
// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "::mlir::toy";
}
```

The `Dialect` record has more fields than the two used here:

| Field | Meaning | This repo |
|---|---|---|
| `name` | The namespace; corresponds 1-1 with `ToyDialect::getDialectNamespace()` | `"toy"` |
| `summary` | A short one-line summary, used in generated docs | not set |
| `description` | A longer description, used in generated docs | not set |
| `cppNamespace` | The C++ namespace the generated class lives in | `"::mlir::toy"` — hence `mlir::toy::ToyDialect` everywhere in this repo's C++ |

`mlir-tblgen -gen-dialect-decls` turns this record into exactly the boilerplate class of section 3.1:

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen -gen-dialect-decls include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include | sed -n '10,$p'
```

Real output **[7]**:

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

`-gen-dialect-defs` produces the constructor and destructor (the odd whitespace is verbatim generator output):

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen -gen-dialect-defs include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include | sed -n '10,$p'
```

Real output **[8]**:

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

(The first nine lines, cut by `sed`, are the generated-file banner. Section 6.4 writes all four generated files to disk.)

Compared with the hand-written class, the generated one makes the constructor private and befriends `MLIRContext` (only a context may create a dialect), adds a destructor, and registers a `TypeID` for the dialect.

### 3.3 Wiring the generated code into C++ and loading the dialect

Section 3.2's generated code reaches the compiler through two hand-written files: a header that includes the generated declarations, and a source file that includes the generated definitions and adds the one method ODS leaves to us.

**The header.** `Dialect.h` is nothing but glue — it includes interface headers the generated code needs, then the two generated declaration files:

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

**The source file.** The only thing left for us to write by hand is `initialize()`, which registers the operations. `Dialect.cpp` includes `Dialect.h`, then the generated constructor/destructor of section 3.2 (`Dialect.cpp.inc`), then defines `initialize()` (abridged):

***mlir/Dialect.cpp***
```cpp
#include "toy/Dialect.h"
...
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

With a single hand-written op this would simply be `addOperations<ConstantOp>();`. With ODS, `toy/Ops.cpp.inc` expands to the complete op list, in alphabetical order: `::mlir::toy::AddOp, ::mlir::toy::ConstantOp, ...`. `GET_OP_LIST` is a guard macro. The generated `Ops.cpp.inc` contains both a comma-separated list of all op classes and their full method definitions, and the macro selects which part gets included. The method definitions come in through a second include under `GET_OP_CLASSES`, at the end of `Dialect.cpp` (section 4.6).

Finally, a dialect must be **loaded into the `MLIRContext`** before use: by default a context loads only the [builtin dialect](https://mlir.llvm.org/docs/Dialects/Builtin/) (which provides a few core IR components such as `builtin.module`), so every other dialect must be loaded explicitly. The driver does this first thing in `dumpMLIR()`:

***toyc.cpp***
```cpp
mlir::MLIRContext context;
// Load our Dialect in this MLIR Context.
context.getOrLoadDialect<mlir::toy::ToyDialect>();
```

`context.loadDialect<ToyDialect>()` would work equally well here. `loadDialect<Ts...>()` loads one or more dialects and returns nothing, while `getOrLoadDialect<T>()` loads a single dialect (if not already loaded) and returns a `T *` to it. Without either line, the context can't parse or create `toy.*` ops — the same failure stock `mlir-opt` shows in appendix C.2: ``Dialect `toy' not found for custom op 'toy.func'``.

---

## 4. Defining Toy Operations with ODS

With a dialect in place we can define operations, which is how we give the rest of the system semantic information to hook into. This section follows one op, `toy.constant`: first written by hand in C++, then declaratively with ODS, one feature at a time, then with a custom assembly format. Section 4.6 sums up the C++ that `Ops.td` leaves to `Dialect.cpp`. Appendix A covers the other eight ops, and appendix B the whole of `Ops.td`, its generated C++, and `Dialect.cpp`.

### 4.1 The C++ way: a hand-written `ConstantOp`

`toy.constant` represents a constant value in the Toy language. In generic form (not this repo's output):

```mlir
%4 = "toy.constant"() {value = dense<1.0> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
```

It takes **zero operands**, a [dense elements](https://mlir.llvm.org/docs/Dialects/Builtin/#denseintorfpelementsattr) **attribute** named `value` holding the constant, and returns a **single result** of [RankedTensorType](https://mlir.llvm.org/docs/Dialects/Builtin/#rankedtensortype). (MLIR 20 prints `value` in `<{...}>` instead, as a *property*, an inherent attribute; section 4.5 shows real output and appendix C.1 explains.)

An operation class inherits from the [CRTP](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern) template `mlir::Op`, which also takes optional [**traits**](https://mlir.llvm.org/docs/Traits/). Traits inject extra behavior into an operation — accessors, verification, and more. A possible hand-written definition (not this repo's code):

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

It would then be registered in the dialect initializer (not this repo's code; the real `initialize()` in section 3.3 registers all nine ops from the generated list):

```cpp
void ToyDialect::initialize() {
  addOperations<ConstantOp>();
}
```

The class has five ingredients — a **name**, **traits** for the operand/result structure, an **accessor**, a **verifier**, and **builders**. Section 4.4 rebuilds each of them in ODS.

### 4.2 `Operation` vs. `Op`

Now that we have an op, we want to access and transform it. MLIR has two main classes for that, and they are easy to confuse:

- `mlir::Operation` models *all* operations generically. It is **opaque**: it describes no particular operation, but provides a general API into any operation instance (`getOperands()`, `getAttrs()`, `getLoc()`, …).
- Each specific kind of operation is an `mlir::Op`-derived class (e.g. our `ConstantOp`, "zero inputs, one output, always the same value"). It is a thin **smart-pointer wrapper around an `Operation*`** that adds operation-specific accessors and type-safe properties.

So defining a Toy op means defining a clean, semantically useful *interface* for building and inspecting an `Operation`. That is why `ConstantOp` declares **no data fields**: all of its data lives in the referenced `Operation`. A consequence is that `Op` classes are always passed **by value**, not by reference or pointer — a common MLIR idiom that applies equally to attributes, types, etc.

Given a generic `Operation*`, you get the specific `Op` back with LLVM's casting infrastructure:

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

Instead of specializing `mlir::Op` by hand, you describe the operation as an ODS record in `Ops.td`, and `mlir-tblgen` expands it into the equivalent `mlir::Op` C++ specialization at build time. ODS is *the* recommended way to define ops — it is simpler, more concise, and stays stable when MLIR's C++ APIs change (the hand-written class above would drift; the generated one is always current).

#### What TableGen and ODS are

Two layers are involved:

- **[TableGen](https://llvm.org/docs/TableGen/ProgRef.html)** is LLVM's small declarative language for *records*: named collections of typed fields. A `.td` file only declares records; it has no meaning of its own. A **backend** reads the records and writes code from them.
- **[ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/)** (Operation Definition Specification) is MLIR's vocabulary on top of TableGen. It has two parts:
  - TableGen classes such as `Dialect`, `Op`, `OpBuilder`, traits, and type/attribute constraints like `F64Tensor`, defined in `mlir/IR/OpBase.td` and the files it includes;
  - the `mlir-tblgen` backends that turn records derived from those classes into C++: `-gen-dialect-decls`/`-gen-dialect-defs` produce `Dialect.h.inc`/`Dialect.cpp.inc`, `-gen-op-decls`/`-gen-op-defs` produce `Ops.h.inc`/`Ops.cpp.inc`, and `-gen-op-doc` produces Markdown documentation.

  The build runs these backends for you (appendix E.1), and `run_mlir-tblgen.sh` runs them by hand (section 6.4).

So writing `Ops.td` means filling in the fields of MLIR's classes, and the generated C++ is the backends' reading of those fields.

#### The `Toy_Op` base class

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

`Toy_Op` is a `class`, so it generates nothing itself; each `def` derived from it becomes one op class. A Toy op inherits from `Toy_Op`, giving the **mnemonic** — the op name *without* the `toy.` prefix, matching `ConstantOp::getOperationName()` above — and an optional trait list.

Section 4.4 builds `ConstantOp` on this base class one field at a time (appendix B.1 lists every TableGen construct `Ops.td` uses).

### 4.4 `ConstantOp` step by step

Each step defines `ConstantOp` with a few more fields than the last and runs the result through `mlir-tblgen` to show what the new fields generate. Step *N*'s definition is the file `include/toy/Ops/OpsN.td`: standalone inputs for this walkthrough, one per stage, while the build compiles only `include/toy/Ops.td`. Each file holds the include, `Toy_Dialect` and `Toy_Op` it needs (shown in full in step 0, elided as `...` after that), plus the previous step's fields and the new ones, so `diff include/toy/Ops/Ops1.td include/toy/Ops/Ops2.td` shows exactly what step 2 adds.

#### Step 0: The empty shell

The starting point is just the shell, a `def` with a mnemonic and no fields, in its own file together with the dialect and base class it needs:

***include/toy/Ops/Ops0.td***
```tablegen
include "mlir/IR/OpBase.td"

def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "::mlir::toy";
}

class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;

def ConstantOp : Toy_Op<"constant"> {
}
```

`--print-records` shows the record after TableGen has resolved every class and template argument, before any backend runs:

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen --print-records include/toy/Ops/Ops0.td -I /opt/homebrew/opt/llvm@20/include | awk '/^def ConstantOp /,/^}/'
```

Real output **[9]**:

```text
def ConstantOp {	// Op Toy_Op
  Dialect opDialect = Toy_Dialect;
  string opName = "constant";
  string cppNamespace = "::mlir::toy";
  string summary = "";
  string description = "";
  OpDocGroup opDocGroup = ?;
  dag arguments = (ins);
  dag results = (outs);
  dag regions = (region);
  dag successors = (successor);
  list<OpBuilder> builders = ?;
  bit skipDefaultBuilders = 0;
  string assemblyFormat = ?;
  bit hasCustomAssemblyFormat = 0;
  bit hasVerifier = 0;
  bit hasRegionVerifier = 0;
  bit hasCanonicalizer = 0;
  bit hasCanonicalizeMethod = 0;
  bit hasFolder = 0;
  bit useCustomPropertiesEncoding = 0;
  list<Trait> traits = [];
  string extraClassDeclaration = ?;
  string extraClassDefinition = ?;
}
```

- **`// Op Toy_Op`** lists the classes the record derives from. A record is flat: everything inherited is copied in.
- **Template arguments are resolved.** `Toy_Op<"constant">` passed `"constant"` on as `Op`'s mnemonic, which became `opName`. `Toy_Dialect` became `opDialect`, which in turn supplied `cppNamespace`. The omitted trait list kept its default `[]`.
- **Every other field holds its default**: no arguments `(ins)`, no results `(outs)`, an empty `summary`, `hasVerifier = 0`, and `?` for "unset". These fields are all the backends read. The steps below set them one at a time; appendix B.3 lists every field `Ops.td` sets.

The op-declaration backend reads this record as an op with nothing in it, but already a complete `mlir::Op` class:

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen -gen-op-decls include/toy/Ops/Ops0.td -I /opt/homebrew/opt/llvm@20/include | awk '/^class ConstantOp : /,/^};/'
```

Real output **[10]**:

```cpp
class ConstantOp : public ::mlir::Op<ConstantOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConstantOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConstantOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("toy.constant");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
public:
};
```

Every member is already there for a reason, even with no fields:

| Generated member | What it is |
|---|---|
| `class ConstantOp : public ::mlir::Op<ConstantOp, ...>` | The CRTP base of section 4.1, with a trait list derived from the record: `ZeroRegions`, `ZeroResults`, `ZeroSuccessors` and `ZeroOperands` come from the empty `regions`, `results`, `successors` and `arguments` dags, and `OpInvariants` hooks `verifyInvariants()` (below) into MLIR's verifier |
| `using Op::Op;` | Inherits the constructor that wraps an `Operation *`, as in section 4.1 |
| `using Op::print;` | Keeps the base class's `print(raw_ostream &)` visible next to any generated `print(OpAsmPrinter &)` |
| `Adaptor`, `GenericAdaptor`, `FoldAdaptor` | Companion classes generated just above this one (`ConstantOpAdaptor`, `ConstantOpGenericAdaptor`). They offer the same named accessors over a plain list of values or attributes instead of a live op; folding and conversion patterns in later chapters use them |
| `getAttributeNames()` | The op's inherent attribute names — none yet |
| `getOperationName()` | `"toy.constant"`: the dialect's `name` plus the mnemonic |
| `getODSOperandIndexAndLength()`, `getODSOperands()` | Map an operand *group* of the `arguments` dag to a range of the op's actual operands. `{index, 1}` means "one value per group"; a `Variadic` group changes that. The named operand accessors (`getLhs()`, …) are built on these |
| `getODSResultIndexAndLength()`, `getODSResults()` | The same for result groups |
| three `build()` overloads | No arguments; explicit result types; and the fully generic "result types, operands, attributes" form every op has |
| `verifyInvariantsImpl()`, `verifyInvariants()` | Where the checks ODS derives from the fields go, and the entry point MLIR's verifier calls |

`-gen-op-defs` writes the method bodies, into the file that `Dialect.cpp` includes twice (section 3.3). Everything after the nine-line banner:

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen -gen-op-defs include/toy/Ops/Ops0.td -I /opt/homebrew/opt/llvm@20/include | sed -n '10,$p'
```

Real output **[11]**:

```cpp
#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::toy::ConstantOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace toy {
} // namespace toy
} // namespace mlir
namespace mlir {
namespace toy {

//===----------------------------------------------------------------------===//
// ::mlir::toy::ConstantOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
ConstantOpAdaptor::ConstantOpAdaptor(ConstantOp op) : ConstantOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ConstantOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstantOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult ConstantOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::llvm::LogicalResult ConstantOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace toy
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::toy::ConstantOp)


#endif  // GET_OP_CLASSES
```

- **`GET_OP_LIST`** wraps the comma-separated op list — here just `::mlir::toy::ConstantOp` — that `ToyDialect::initialize()` passes to `addOperations<...>` (section 3.3). **`GET_OP_CLASSES`** wraps everything else.
- **Local Utility Method Definitions** is where step 1's constraint-checking functions will go; for now its `namespace mlir { namespace toy {` pair is empty, as is `namespace detail`, which will hold the adaptor's base class.
- **`ConstantOpAdaptor`**'s constructor takes a live op's operands, and its `verify()` will check attributes without an op; nothing to check yet.
- **The three `build()` bodies** fill the `OperationState` from their arguments, with asserts encoding the arity ODS knows: zero results, zero operands. The no-argument one has nothing to add.
- **`verifyInvariants()`** is what MLIR calls to verify the op; it delegates to `verifyInvariantsImpl()`, where ODS puts the checks it derives from the fields — none so far.
- **`MLIR_DEFINE_EXPLICIT_TYPE_ID`** gives the op class its `TypeID`, as the dialect got one in section 3.2; `isa<ConstantOp>`/`dyn_cast<ConstantOp>` (section 4.2) compare it.

#### Step 1: Defining arguments and results

The [arguments](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-arguments) (inputs) of an op may be **attributes** or **types of SSA operand values**; the [results](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-results) are the types of the values it produces. Giving an argument a name (`$value`) makes ODS generate matching accessors:

- `let arguments = (ins F64ElementsAttr:$value);` — an *attribute* (`F64ElementsAttr`: a 64-bit float `ElementsAttr`) sits in `ins` right where an operand would go; ODS tells them apart by the constraint kind.
- `let results = (outs F64Tensor);` — `F64Tensor` is a predefined constraint, "tensor of 64-bit float". Because it's unnamed, only generic result accessors are produced.

***include/toy/Ops/Ops1.td***
```tablegen
...
def ConstantOp : Toy_Op<"constant"> {
  // The constant operation takes an attribute as the only input.
  // `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  // F64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs F64Tensor);
}
```

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen -gen-op-decls include/toy/Ops/Ops1.td -I /opt/homebrew/opt/llvm@20/include \
  | awk '/^class ConstantOp : /,/^};/' | sed -n '1p;56,63p'
```

Real output **[12]**:

```cpp
class ConstantOp : public ::mlir::Op<ConstantOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
  ::mlir::DenseElementsAttr getValueAttr() {
    return ::llvm::cast<::mlir::DenseElementsAttr>(getProperties().value);
  }

  ::mlir::DenseElementsAttr getValue();
  void setValueAttr(::mlir::DenseElementsAttr attr) {
    getProperties().value = attr;
  }
```

- **The traits changed**: the shell's `ZeroResults` is now `OneResult` plus `OneTypedResult<TensorType>`, which section 4.1 had to write by hand. `OneTypedResult<TensorType>` adds typed accessors: `getResult()` and `getType()` return a `TensorType` value and type rather than a plain `Value`/`Type`. `ZeroOperands` stays, because the attribute is not an operand.
- **`BytecodeOpInterface::Trait` is new**: the op now stores data in a property (`value`), and this interface writes and reads properties in MLIR's binary bytecode format (the `readProperties()`/`writeProperties()` methods the class declares; the `sed` cuts them, and the full class at the end of 4.4 shows them).
- **The accessors**, prefixed with `get`/`set`: `getValueAttr()` returns the attribute stored in the op's properties, and `setValueAttr()` replaces it. `getValue()` returns the attribute's *value*; for `F64ElementsAttr` that is the attribute itself.

The definitions show what the two constraints turn into:

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen -gen-op-defs include/toy/Ops/Ops1.td -I /opt/homebrew/opt/llvm@20/include \
  | awk '/^static ::llvm::LogicalResult __mlir_ods_local_/,/^}/; /^::mlir::DenseElementsAttr ConstantOp::getValue\(/,/^}/; /^::llvm::LogicalResult ConstantOp::verifyInvariantsImpl\(/,/^}/'
```

Real output **[13]**:

```cpp
static ::llvm::LogicalResult __mlir_ods_local_type_constraint_Ops11(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return (elementType.isF64()); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of 64-bit float values, but got " << type;
  }
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_Ops11(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::DenseFPElementsAttr>(attr) &&::llvm::cast<::mlir::DenseElementsAttr>(attr).getType().getElementType().isF64())))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 64-bit float elements attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_Ops11(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_Ops11(attr, attrName, [op]() {
    return op->emitOpError();
  });
}
::mlir::DenseElementsAttr ConstantOp::getValue() {
  auto attr = getValueAttr();
  return attr;
}
::llvm::LogicalResult ConstantOp::verifyInvariantsImpl() {
  auto tblgen_value = getProperties().value; (void)tblgen_value;
  if (!tblgen_value) return emitOpError("requires attribute 'value'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Ops11(*this, tblgen_value, "value")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops11(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}
```

- **Each constraint became a checking function**, local to the generated file (`static`), named after its kind plus the input file's name and a counter — `_Ops11` is constraint 1 of `Ops1.td`; the real `Ops.td` gets `_Ops1`, `_Ops2`, … — so functions generated from different `.td` files never clash: `__mlir_ods_local_type_constraint_*` tests "tensor of `f64`" for `F64Tensor`, and `__mlir_ods_local_attr_constraint_*` tests "dense `f64` elements" for `F64ElementsAttr`. Their messages ("must be tensor of 64-bit float values", "failed to satisfy constraint: 64-bit float elements attribute") are the errors a user sees.
- **The attribute check comes in two overloads**: the first reports through any `emitError` callback, so code without a live op can use it — `verifyInherentAttrs()` checks an attribute dictionary with it before the op exists; the second wraps it with `op->emitOpError()` for a live op. (The adaptor's `verify()` inlines the same test.)
- **`verifyInvariantsImpl()` now calls them**, in field order: it reads `value` from the op's properties and fails with "requires attribute 'value'" if it is missing, runs the attribute check on it, then loops over result group 0 (`getODSResults(0)`) running the type check with a running `index` — that is the `#0` in messages like "result #0 must be …". These are the "ODS-generated checks" that step 3's `verify()` runs after.
- **`getValue()`** is just `auto attr = getValueAttr(); return attr;`.

**The checks in action.** An `i32` attribute fails the attribute check before any hand-written code runs:

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 - -x mlir -emit=mlir <<'EOF'
toy.func @main() {
  %0 = "toy.constant"() {value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xf64>
  toy.return
}
EOF
```

Real output **[14]** (exit status 3):

```text
loc("<stdin>":2:8): error: 'toy.constant' op attribute 'value' failed to satisfy constraint: 64-bit float elements attribute
Error can't load file -
```

#### Step 2: Adding documentation

Ops may provide [`summary` and `description`](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-documentation) fields describing their semantics. They are useful for dialect users and feed auto-generated Markdown, which `-gen-op-doc` renders:

***include/toy/Ops/Ops2.td***
```tablegen
...
def ConstantOp : Toy_Op<"constant"> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant operation";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

      %0 = "toy.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
  }];

  // The constant operation takes an attribute as the only input.
  // `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  // F64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs F64Tensor);
}
```

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen -gen-op-doc include/toy/Ops/Ops2.td -I /opt/homebrew/opt/llvm@20/include | awk '/^### `toy.constant`/,0'
```

Real output **[15]**:

```text
### `toy.constant` (toy::ConstantOp)

_Constant operation_

Constant operation turns a literal into an SSA value. The data is attached
to the operation as an attribute. For example:

  %0 = "toy.constant"()
     { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
    : () -> tensor<2x3xf64>

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>value</code></td><td>::mlir::DenseElementsAttr</td><td>64-bit float elements attribute</td></tr>
</table>

#### Results:

| Result | Description |
| :----: | ----------- |
&laquo;unnamed&raquo; | tensor of 64-bit float values
```

The summary became the italic line, the description the paragraph, and ODS added the attribute and result tables from `arguments` and `results` by itself. The generated C++ doesn't change: compared with step 1, `-gen-op-decls` and `-gen-op-defs` differ only in names derived from the file name (`From: Ops2.td`, `_Ops21`).

#### Step 3: Verifying operation semantics

Much like the named accessors, ODS generates most verification logic from the constraints you gave: we don't need to verify the structure of the result type, or even the input attribute `value`. Often no extra verification is needed at all. When it is, `let hasVerifier = 1;` declares a `::llvm::LogicalResult verify()` method on the op class that runs **after** the ODS-generated checks:

***include/toy/Ops/Ops3.td***
```tablegen
...
def ConstantOp : Toy_Op<"constant"> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant operation";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

      %0 = "toy.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
  }];

  // The constant operation takes an attribute as the only input.
  // `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  // F64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs F64Tensor);

  // Add additional verification logic to the constant operation. Setting this bit
  // to `1` will generate a `::llvm::LogicalResult verify()` declaration on the
  // operation class that is called after ODS constructs have been verified, for
  // example the types of arguments and results. We implement additional verification
  // in the definition of this `verify` method in the C++ source file.
  let hasVerifier = 1;
}
```

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen -gen-op-decls include/toy/Ops/Ops3.td -I /opt/homebrew/opt/llvm@20/include \
  | awk '/^class ConstantOp : /,/^};/' | grep 'LogicalResult verify'
```

Real output **[16]**:

```cpp
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
```

`verifyInvariantsImpl()` holds the generated checks of step 1, and `verifyInvariants()` now runs both, in order:

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen -gen-op-defs include/toy/Ops/Ops3.td -I /opt/homebrew/opt/llvm@20/include \
  | awk '/^::llvm::LogicalResult ConstantOp::verifyInvariants\(/,/^}/'
```

Real output **[17]**:

```cpp
::llvm::LogicalResult ConstantOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}
```

Compared with the shell's `return verifyInvariantsImpl();` (step 0), `verify()` runs only if the generated checks succeeded. We implement `verify()` in `Dialect.cpp`; it can assume every other invariant already holds, and checks that the result tensor's shape matches the attribute's shape:

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

A constant whose result type matches its attribute passes, and prints back in custom form:

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 - -x mlir -emit=mlir <<'EOF'
toy.func @main() {
  %0 = "toy.constant"() {value = dense<[1.0, 2.0]> : tensor<2xf64>} : () -> tensor<2xf64>
  toy.return
}
EOF
```

Real output **[18]**:

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
    toy.return
  }
}
```

Change only the result type, and the generated checks still pass — both types are valid `F64Tensor`s and the attribute is a valid `F64ElementsAttr` — but `verify()` catches the mismatch:

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 - -x mlir -emit=mlir <<'EOF'
toy.func @main() {
  %0 = "toy.constant"() {value = dense<[1.0, 2.0]> : tensor<2xf64>} : () -> tensor<3xf64>
  toy.return
}
EOF
```

Real output **[19]** (exit status 3):

```text
loc("<stdin>":2:8): error: 'toy.constant' op return type shape mismatches its attribute at dimension 0: 2 != 3
Error can't load file -
```

#### Step 4: Attaching `build` methods

The last piece of the C++ version is its `build` methods — what `builder.create<ConstantOp>(loc, ...)` calls to populate the `OperationState` (the collection of all the elements an operation may contain) that MLIR uses to create the op. ODS generates simple ones automatically; the [`builders`](https://mlir.llvm.org/docs/DefiningDialects/Operations/#custom-builder-methods) field adds more. Each `OpBuilder` takes a list of C++ parameters `(ins ...)` and an optional inline body:

***include/toy/Ops/Ops4.td***
```tablegen
...
def ConstantOp : Toy_Op<"constant"> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant operation";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

      %0 = "toy.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
  }];

  // The constant operation takes an attribute as the only input.
  // `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  // F64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs F64Tensor);

  // Add additional verification logic to the constant operation. Setting this bit
  // to `1` will generate a `::llvm::LogicalResult verify()` declaration on the
  // operation class that is called after ODS constructs have been verified, for
  // example the types of arguments and results. We implement additional verification
  // in the definition of this `verify` method in the C++ source file.
  let hasVerifier = 1;

  // Add custom build methods for the constant operation. These methods populate
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      // Call into an autogenerated `build` method.
      build($_builder, $_state, value.getType(), value);
    }]>,

    // Build a constant with a given constant floating-point value. This builder
    // creates a declaration for `ConstantOp::build` with the given parameters.
    OpBuilder<(ins "double":$value)>
  ];
}
```

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen -gen-op-decls include/toy/Ops/Ops4.td -I /opt/homebrew/opt/llvm@20/include \
  | awk '/^class ConstantOp : /,/^};/' | grep 'static void build'
```

Real output **[20]**:

```cpp
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, DenseElementsAttr value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, double value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::DenseElementsAttr value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::DenseElementsAttr value);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
```

The class now has five `build` overloads: the first two are ours, the other three are what ODS generates from `arguments`/`results` (explicit result type + attribute; a result-type range + attribute; the fully generic types/operands/attributes form). Every overload starts with the builder and the state being filled (`odsBuilder`, `odsState`); the rest are the `(ins ...)` parameters. Ours keep the C++ types exactly as written in the `ins` strings (`DenseElementsAttr`, `double`), resolved inside namespace `mlir::toy`, while ODS spells its own fully qualified (`::mlir::DenseElementsAttr`). The generic overload leaves the builder unnamed, because it never uses it, and defaults `attributes` to `{}`.

- The first custom builder, `(ins "DenseElementsAttr":$value)`, has an **inline body** (the `[{ ... }]` blob) that calls an autogenerated `build` with the attribute's own type. Inside the blob, `$_builder` and `$_state` are placeholders for the builder and the state being filled.
- The second, `(ins "double":$value)`, has **no body**: ODS only *declares* `ConstantOp::build(OpBuilder &, OperationState &, double)`.

The definitions confirm both. Four `build()` bodies are generated — the `double` one is missing:

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen -gen-op-defs include/toy/Ops/Ops4.td -I /opt/homebrew/opt/llvm@20/include \
  | awk '/^void ConstantOp::build\(/,/^}/'
```

Real output **[21]**:

```cpp
void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, DenseElementsAttr value) {
      // Call into an autogenerated `build` method.
      build(odsBuilder, odsState, value.getType(), value);
    
}
void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::DenseElementsAttr value) {
  odsState.getOrAddProperties<Properties>().value = value;
  odsState.addTypes(resultType0);
}
void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::DenseElementsAttr value) {
  odsState.getOrAddProperties<Properties>().value = value;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}
void ConstantOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<ConstantOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}
```

- **Ours, first**: the inline body, pasted in verbatim (blank-line indentation included) with `$_builder`/`$_state` replaced by `odsBuilder`/`odsState`. It forwards to the next overload with the attribute's own type.
- **Result type + attribute**: stores `value` in the op's properties (`getOrAddProperties<Properties>().value`) and adds the result type. This is all a builder does — fill the `OperationState`; MLIR creates the op from it afterwards.
- **Result-type range + attribute**: the same, with an assert that exactly one result type was passed.
- **Generic**: asserts zero operands and one result type, then copies operands, attributes and types into the state. Because `value` is a property, any attributes passed as a dictionary are converted into properties at the end (`setOpPropertiesFromAttribute`).

**Why the `double` builder has no body.** ODS writes a builder body only when it is given one. `OpBuilder` is a TableGen class in `OpBase.td` with two template arguments: the parameter dag `p` and a `code` body `b`, which defaults to the empty string. An empty body is the documented signal to generate *only* the declaration. The second command below shows how each of our two `OpBuilder<...>` entries became an anonymous record in the `builders` list:

```bash
cd /Users/roy/study/mlir/toy/Ch2
sed -n '276,282p' /opt/homebrew/opt/llvm@20/include/mlir/IR/OpBase.td
mlir-tblgen --print-records include/toy/Ops/Ops4.td -I /opt/homebrew/opt/llvm@20/include \
  | awk '/\/\/ OpBuilder$/,/^}/'
```

Real output **[22]**:

```text
// If an empty string is passed in for `body`, then *only* the builder
// declaration will be generated; this provides a way to define complicated
// builders entirely in C++.
class OpBuilder<dag p, code b = ""> {
  dag dagParams = p;
  code body = b;
}
def anonymous_344 {	// OpBuilder
  dag dagParams = (ins "DenseElementsAttr":$value);
  code body = [{
      // Call into an autogenerated `build` method.
      build($_builder, $_state, value.getType(), value);
    }];
}
def anonymous_345 {	// OpBuilder
  dag dagParams = (ins "double":$value);
  string body = "";
}
```

- **`anonymous_344`** is `OpBuilder<(ins "DenseElementsAttr":$value), [{ ... }]>`. Its `body` holds the code block, so `-gen-op-defs` wraps it in a function (the first `build` definition above).
- **`anonymous_345`** is `OpBuilder<(ins "double":$value)>`. The second argument was omitted, so `body` is the default `""`. `-gen-op-decls` still emits the declaration from `dagParams` (the second `build` declaration above), and `-gen-op-defs` skips the definition.

Why leave a body out at all? A builder body is only C++ pasted into the generated file, so either way works. The forwarding builder is one line, so it fits inline. The `double` builder has to build a tensor type and a dense attribute from a scalar. That code is easier to write, read and debug in a real `.cpp` file than in a `[{ ... }]` string inside TableGen. The cost is that someone must define it: with the declaration and no definition, any call to `ConstantOp::build(builder, state, 1.0)` (MLIRGen's `builder.create<ConstantOp>(loc(num.loc()), num.getValue())` for a number literal is one) fails at link time with an undefined symbol (appendix B.5 shows the error).

That definition is in the dialect:

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

#### Step 5: Adding traits — `Pure`

The second template argument of `Toy_Op` is the op's trait list (section 4.3), empty so far. `Pure` declares that the op has no side effects, so an unused constant can be deleted and duplicates merged. It is defined in `SideEffectInterfaces.td`, so `Ops5.td` also includes that file:

***include/toy/Ops/Ops5.td***
```tablegen
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
...
// We define a toy operation by inheriting from our base 'Toy_Op' class above.
// Here we provide the mnemonic and a list of traits for the operation. The
// constant operation is marked as 'Pure' as it is a pure operation
// and may be removed if dead.
def ConstantOp : Toy_Op<"constant", [Pure]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant operation";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

      %0 = "toy.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
  }];

  // The constant operation takes an attribute as the only input.
  // `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  // F64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs F64Tensor);

  // Add additional verification logic to the constant operation. Setting this bit
  // to `1` will generate a `::llvm::LogicalResult verify()` declaration on the
  // operation class that is called after ODS constructs have been verified, for
  // example the types of arguments and results. We implement additional verification
  // in the definition of this `verify` method in the C++ source file.
  let hasVerifier = 1;

  // Add custom build methods for the constant operation. These methods populate
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      // Call into an autogenerated `build` method.
      build($_builder, $_state, value.getType(), value);
    }]>,

    // Build a constant with a given constant floating-point value. This builder
    // creates a declaration for `ConstantOp::build` with the given parameters.
    OpBuilder<(ins "double":$value)>
  ];
}
```

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen -gen-op-decls include/toy/Ops/Ops5.td -I /opt/homebrew/opt/llvm@20/include \
  | awk '/^class ConstantOp : /,/^};/' | grep 'class ConstantOp : \|getEffects'
```

Real output **[23]**:

```cpp
class ConstantOp : public ::mlir::Op<ConstantOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
```

`Pure` is a list of traits itself: the class gains `ConditionallySpeculatable::Trait` and `AlwaysSpeculatableImplTrait` ("safe to execute speculatively") plus `MemoryEffectOpInterface::Trait`, whose `getEffects()` method reports the op's memory effects. Its generated body adds none:

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen -gen-op-defs include/toy/Ops/Ops5.td -I /opt/homebrew/opt/llvm@20/include \
  | awk '/^void ConstantOp::getEffects\(/,/^}/'
```

Real output **[24]**:

```cpp
void ConstantOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}
```

An empty effect list is how the rest of MLIR learns the op is side-effect free. It pays off in Chapter 3, where canonicalization and CSE start deleting and merging ops.

#### The finished `ConstantOp`: `Ops.td` vs the hand-written class

This is the record in `Ops.td`:

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

It holds the five steps' fields, with slightly different comments and summary, plus one more: **`hasCustomAssemblyFormat = 1`**, a hand-written `parse()`/`print()` and the subject of section 4.5.

The real `ConstantOp` generates this class:

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen -gen-op-decls include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include | awk '/^class ConstantOp : /,/^};/'
```

Real output **[25]**:

```cpp
class ConstantOp : public ::mlir::Op<ConstantOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConstantOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConstantOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("value")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getValueAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getValueAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("toy.constant");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::DenseElementsAttr getValueAttr() {
    return ::llvm::cast<::mlir::DenseElementsAttr>(getProperties().value);
  }

  ::mlir::DenseElementsAttr getValue();
  void setValueAttr(::mlir::DenseElementsAttr attr) {
    getProperties().value = attr;
  }

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
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
```

Read side by side with the hand-written class of section 4.1, the finished generated class has all five of its ingredients. Each bullet compares the hand-written class (4.1), the empty shell (step 0) and the finished class:

- **Name**: all three return `"toy.constant"` from `getOperationName()`. In the shell and the finished class, the mnemonic alone produced it.
- **Traits**: the hand-written class lists `ZeroOperands`, `OneResult` and `OneTypedResult<TensorType>` itself; in ODS nobody writes them. The shell had `ZeroResults`; the finished class has `OneResult` and `OneTypedResult<TensorType>` because `results` became `(outs F64Tensor)` (step 1), plus the speculation and memory-effect traits from `Pure` (step 5).
- **Accessor**: the hand-written class declares `getValue()` and must define it. The finished class gets `getValue()`, `getValueAttr()` and `setValueAttr()` generated from the named argument `$value` (step 1); the shell has none. Because `value` is stored as a property, the finished class also gets the property plumbing (`setPropertiesFromAttr()`, `getInherentAttr()`, `readProperties()`, …).
- **Verifier**: the hand-written class writes `verifyInvariants()` itself. The shell and the finished class both have the generated `verifyInvariantsImpl()`/`verifyInvariants()`; only the finished class also declares `verify()`, from `hasVerifier = 1` (step 3).
- **Builders**: the hand-written class declares three `build()` overloads. The shell gets three generic ones; the finished class has five, two of them from `builders` (step 4). `parse()`/`print()` come from `hasCustomAssemblyFormat = 1` (section 4.5), and `getEffects()` from `Pure` (step 5).

Comparing `mlir-tblgen` output with the hand-written class like this is the best way to get comfortable with TableGen.

### 4.5 Custom assembly formats

#### Generic vs. custom syntax

With the ops defined so far we can already generate "Toy IR" from a Toy program (abridged: the file's `# CHECK` lines are left out):

***test_Example/Toy/Ch2/codegen.toy***
```text
# RUN: toyc-ch2 %s -emit=mlir 2>&1 | FileCheck %s

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

Without custom formats, every op prints in the **generic** assembly format — the one dissected in section 2.2. `-mlir-print-op-generic` bypasses the custom formats this repo defines, so it shows that stage (`-mlir-print-debuginfo` adds the locations):

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 ../../test_Example/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-op-generic -mlir-print-debuginfo 2>&1
```

Real output **[26]**:

```mlir
"builtin.module"() ({
  "toy.func"() <{function_type = (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>, sym_name = "multiply_transpose"}> ({
  ^bb0(%arg0: tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":4:1), %arg1: tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":4:1)):
    %6 = "toy.transpose"(%arg0) : (tensor<*xf64>) -> tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":5:10)
    %7 = "toy.transpose"(%arg1) : (tensor<*xf64>) -> tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":5:25)
    %8 = "toy.mul"(%6, %7) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":5:25)
    "toy.return"(%8) : (tensor<*xf64>) -> () loc("../../test_Example/Toy/Ch2/codegen.toy":5:3)
  }) : () -> () loc("../../test_Example/Toy/Ch2/codegen.toy":4:1)
  "toy.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "toy.constant"() <{value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>}> : () -> tensor<2x3xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":9:17)
    %1 = "toy.reshape"(%0) : (tensor<2x3xf64>) -> tensor<2x3xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":9:3)
    %2 = "toy.constant"() <{value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>}> : () -> tensor<6xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":10:17)
    %3 = "toy.reshape"(%2) : (tensor<6xf64>) -> tensor<2x3xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":10:3)
    %4 = "toy.generic_call"(%1, %3) <{callee = @multiply_transpose}> : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":11:11)
    %5 = "toy.generic_call"(%3, %1) <{callee = @multiply_transpose}> : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":12:11)
    "toy.print"(%5) : (tensor<*xf64>) -> () loc("../../test_Example/Toy/Ch2/codegen.toy":13:3)
    "toy.return"() : () -> () loc("../../test_Example/Toy/Ch2/codegen.toy":8:1)
  }) : () -> () loc("../../test_Example/Toy/Ch2/codegen.toy":8:1)
}) : () -> () loc(unknown)
```

Three things in this dump come from how MLIR 20 prints, not from Toy (appendix C.1 has the details):

- **Inherent attributes print as properties**, in `<{...}>` before the region: `<{function_type = ..., sym_name = "multiply_transpose"}>`, `<{value = dense<...>}>`, `<{callee = @multiply_transpose}>`.
- **The function signature is `function_type`**, the name `FuncOp` gives it in `arguments` (appendix A.3).
- **Values are numbered across the whole module** (`main` gets `%0`–`%5`, `multiply_transpose` `%6`–`%8`), because the generic form implies `-mlir-print-unique-ssa-ids`.

**This dump does not come from any op's `print()`.** `-mlir-print-op-generic` makes MLIR's core printer write every op itself, in the one generic layout every op shares: quoted name, operands, properties, regions, functional type. It never calls an op's own `print()` method. So `ConstantOp::print` in `mlir/Dialect.cpp`, shown later in this section, plays no part in `"toy.constant"() <{value = dense<...> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>`. The same op prints through `ConstantOp::print` only when the flag is left out, as the last run of this section shows. Both forms describe the same in-memory op; only the text differs.

MLIR lets each op define its own format, either **imperatively in C++** (`let hasCustomAssemblyFormat = 1;`, then write `parse()`/`print()`) or **declaratively** (`let assemblyFormat = "...";`), which ODS compiles into those same C++ methods internally.

#### `toy.print`

The generic `"toy.print"(%5) : (tensor<*xf64>) -> ()` is verbose. Stripped to the essentials, a good format is:

```mlir
toy.print %5 : tensor<*xf64> loc(...)
```

**The C++ way.** Set `hasCustomAssemblyFormat` to divert printing and parsing to `print`/`parse` methods. A stripped-down `PrintOp` written that way (not this repo's code: Ch2's `PrintOp` uses the declarative format below, so there is no hand-written `PrintOp::print` in `mlir/Dialect.cpp`):

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

(The snippet doesn't compile as is with MLIR 20: the accessor is `getInput()`, and the printer already emits the op name, so a real `print()` starts with `printer << " "` — compare this repo's `ConstantOp::print` below.) The parser shows the general pattern: parse each piece in order, chaining with `||` because a `ParseResult` converts to `true` on failure; then *resolve* the parsed operand name against its type to get an SSA value.

**The declarative way.** The [declarative format](https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format) is built from three kinds of components:

| Component | What it is | In `toy.print` |
|---|---|---|
| **Directives** | Builtin functions, with optional arguments | `attr-dict`, `type($input)` |
| **Literals** | A keyword or punctuation, surrounded by backticks | `` `:` `` |
| **Variables** | An entity registered on the op itself: an argument (attribute or operand), result, successor, … | `$input` |

A direct mapping of the C++ format is the one-liner this repo actually uses (the full definition is in appendix A.5):

***include/toy/Ops.td***
```tablegen
let assemblyFormat = "$input attr-dict `:` type($input)";
```

ODS turns that line into the same two C++ methods; this is the `PrintOp::parse`/`print` pair in the generated `Ops.cpp.inc`:

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen -gen-op-defs include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include \
  | awk '/^::mlir::ParseResult PrintOp::parse\(/,/^}/; /^void PrintOp::print\(/,/^}/'
```

Real output **[27]**:

```cpp
::mlir::ParseResult PrintOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOperands(&inputRawOperand, 1);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type inputRawType{};
  ::llvm::ArrayRef<::mlir::Type> inputTypes(&inputRawType, 1);

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    inputRawType = type;
  }
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}
void PrintOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInput();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getInput().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}
```

The generated parser follows the hand-written one step for step — `parseOperand`, `parseOptionalAttrDict`, `parseColon`, the type, then `resolveOperands` — and the printer emits the same pieces in the same order, starting with `' '` because the op name is already printed.

The declarative format has many more features (appendix A.9 lists every piece this repo uses), so check it before writing a format in C++. Reach for C++ only when the syntax isn't regular — this repo does so three times: the binary ops' *conditional* type syntax (appendix A.2), `toy.func`'s signature syntax (appendix A.3), and `toy.constant`, below.

#### `toy.constant`'s C++ format

`ConstantOp` sets `hasCustomAssemblyFormat = 1` to print `%0 = toy.constant dense<...> : tensor<2x3xf64>` instead of the generic form. Unlike `PrintOp`'s `assemblyFormat`, that field gives ODS no format to generate code from. ODS only *declares* `parse()`/`print()` (the `static ::mlir::ParseResult parse(...)` and `void print(::mlir::OpAsmPrinter &p)` members of the finished class at the end of section 4.4), just as it only declared step 4's bodiless `double` builder. `Dialect.cpp` defines them:

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

Compare the two forms of the first constant in `main`. The generic one is from the run at the start of this section; the custom one is from the next run:

```text
generic: %0 = "toy.constant"() <{value = dense<[[1.000000e+00, ...]]> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>
custom:  %0 = toy.constant dense<[[1.000000e+00, ...]]> : tensor<2x3xf64>
```

`print` produces the custom line piece by piece. MLIR prints `%0 = toy.constant` itself, and then calls `print`:

- `printer << " "` writes the space after the op name.
- `printOptionalAttrDict(..., {"value"})` prints any attribute other than `value` as `{...}`. `toy.constant` has none, so it prints nothing. `value` is left out because the next line prints it.
- `printer << getValue()` prints the `DenseElementsAttr`. The trailing `: tensor<2x3xf64>` is part of the attribute's own printed form: a dense attribute always prints with its type. `print` never writes a `:` or a type itself.

`parse` reads the same text back in the same order: `parseOptionalAttrDict` takes the empty dictionary, and `parseAttribute(value, "value", ...)` reads `dense<...> : tensor<2x3xf64>` as one attribute. Note the trick: the result type isn't written separately in the custom syntax. Instead, `result.addTypes(value.getType())` *recovers* it from the attribute's type. The generic form spells out `() -> tensor<2x3xf64>`, but custom syntax may leave out anything that can be rebuilt from the rest.

#### The result: Toy IR in custom form

With all nine formats in place, the same program prints much more readably (no `-mlir-print-op-generic` this time):

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 ../../test_Example/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo 2>&1
```

Real output **[28]** (abridged; section 6.2 shows all of it and maps every line back to the source):

```mlir
module {
  toy.func @multiply_transpose(%arg0: tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":4:1), %arg1: tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":4:1)) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":5:10)
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":5:25)
    %2 = toy.mul %0, %1 : tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":5:25)
    toy.return %2 : tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":5:3)
  } loc("../../test_Example/Toy/Ch2/codegen.toy":4:1)
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":9:17)
    ...
    toy.print %5 : tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":13:3)
    toy.return loc("../../test_Example/Toy/Ch2/codegen.toy":8:1)
  } loc("../../test_Example/Toy/Ch2/codegen.toy":8:1)
} loc(unknown)
```

Each line now uses its op's format: `toy.constant` its `Dialect.cpp` `print()` (above), `toy.func` and `toy.mul` their C++ `print()` (appendices A.3 and A.2), `toy.transpose`, `toy.print` and `toy.return` their `assemblyFormat` (appendices A.8, A.5 and A.7). The `builtin.module` prints as `module` because it has a custom format of its own.

### 4.6 The hand-written half: `Dialect.cpp`

ODS generates most of each op, but some `Ops.td` fields only *promise* C++ that someone has to write: a builder without a body, `hasVerifier = 1`, `hasCustomAssemblyFormat = 1`. For each of these, `mlir-tblgen` puts a declaration in `Ops.h.inc` and leaves out the definition, and `mlir/Dialect.cpp` supplies it. `include/toy/Dialect.h` (section 3.3) includes the generated declarations, so `Dialect.cpp` can define them and `MLIRGen.cpp` and `toyc.cpp` can call them.

Everything `Dialect.cpp` defines, and why:

| `Dialect.cpp` defines | Because `Ops.td` has | Called by | Explained in |
|---|---|---|---|
| `ToyDialect::initialize()` | the dialect; ODS always leaves `initialize()` to you | the generated `ToyDialect` constructor, when a context loads the dialect | 3.3 |
| `ConstantOp::build(..., double)` | a builder without a body | `builder.create<ConstantOp>(loc, 5.5)` in MLIRGen | 4.4 step 4 |
| `ConstantOp::verify()` | `hasVerifier = 1` | the generated `verifyInvariants()`, during verification | 4.4 step 3 |
| `ConstantOp::parse()`, `print()` | `hasCustomAssemblyFormat = 1` | MLIR's parser and printer | 4.5 |
| `AddOp`, `MulOp`: `build(lhs, rhs)`, `parse()`, `print()` | a builder without a body; `hasCustomAssemblyFormat = 1` | `builder.create<AddOp>(...)`; parser and printer | A.2 |
| `GenericCallOp::build(callee, arguments)` | a builder without a body | `builder.create<GenericCallOp>(...)` | A.4 |
| `FuncOp::build(name, type, attrs)`, `parse()`, `print()` | a builder without a body; `hasCustomAssemblyFormat = 1` | `builder.create<FuncOp>(...)` (section 5.3); parser and printer | A.3 |
| `ReturnOp::verify()` | `hasVerifier = 1` | the generated `verifyInvariants()` | A.7 |
| `TransposeOp::build(value)`, `verify()` | a builder without a body; `hasVerifier = 1` | `builder.create<TransposeOp>(...)`; `verifyInvariants()` | A.8 |

What the table leaves out is generated in full. `PrintOp` and `ReshapeOp` have only an `assemblyFormat` and generated builders, so they have nothing in `Dialect.cpp` (section 4.5 showed `PrintOp`'s generated parser and printer). `ReturnOp` needs `Dialect.cpp` only for `verify()`; its format and its builder, whose body is inline in `Ops.td`, are generated.

The file ends by including the generated method bodies (`GET_OP_CLASSES` from `Ops.cpp.inc`), so both halves of every op are compiled together in `Dialect.cpp`. Appendix B.5 maps the file in source order and shows the build failing when one promised definition is missing.

---

## 5. The MLIRGen Module

With the dialect in place, [`mlir/MLIRGen.cpp`](mlir/MLIRGen.cpp) turns the Chapter 1 AST into Toy ops. It is a recursive walk over the tree: there is one `mlirGen` overload per AST node kind, each creates the op(s) for its node, and each returns what its caller needs. The public API is one function:

***include/toy/MLIRGen.h***
```cpp
/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          ModuleAST &moduleAST);
```

It is defined at the end of `MLIRGen.cpp`. It creates a temporary `MLIRGenImpl`, the class that holds the actual generator, and calls its module overload:

***mlir/MLIRGen.cpp***
```cpp
// The public API for codegen.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(moduleAST);
}
```

The overload returns a plain `mlir::ModuleOp`, a non-owning handle like every `Op` class (section 4.2). Converting it to `OwningOpRef<ModuleOp>` hands over ownership: when the `OwningOpRef` is destroyed, it erases the module and everything inside it. A null `ModuleOp` becomes an empty handle, which the driver tests with `if (!module)` (section 5.6).

Every overload, by AST node:

| AST node | Toy source | Overload returns | Emits | Section |
|---|---|---|---|---|
| `ModuleAST` | the whole file | `mlir::ModuleOp` | `builtin.module` | 5.2 |
| `PrototypeAST` | `def name(a, b)` | `mlir::toy::FuncOp` | `toy.func` with an empty entry block | 5.3 |
| `FunctionAST` | `def name(a, b) { ... }` | `mlir::toy::FuncOp` | the body, plus an implicit `toy.return` if needed | 5.3 |
| `ExprASTList` | the statements in `{ ... }` | `LogicalResult` | nothing itself: dispatches each statement | 5.4 |
| `VarDeclExprAST` | `var a<2, 3> = ...;` | `mlir::Value` | the initializer, plus `toy.reshape` if a shape is given | 5.4 |
| `ReturnExprAST` | `return x;` | `LogicalResult` | `toy.return` | 5.4 |
| `PrintExprAST` | `print(x);` | `LogicalResult` | `toy.print` | 5.4 |
| `ExprAST` | any expression | `mlir::Value` | nothing itself: dispatches by kind | 5.4 |
| `BinaryExprAST` | `a + b`, `a * b` | `mlir::Value` | `toy.add`, `toy.mul` | 5.4 |
| `VariableExprAST` | `a` | `mlir::Value` | nothing: a symbol-table lookup | 5.4 |
| `LiteralExprAST` | `[[1, 2], [3, 4]]` | `mlir::Value` | `toy.constant` | 5.4 |
| `NumberExprAST` | `5.5` | `mlir::Value` | `toy.constant` | 5.4 |
| `CallExprAST` | `transpose(a)`, `f(a, b)` | `mlir::Value` | `toy.transpose`, `toy.generic_call` | 5.4 |

The return types follow one rule:
- An expression, or a `var`, returns the `mlir::Value` that holds its result. A caller uses that value as an operand of the op it creates next.
- A statement with no value returns `success()` or `failure()`.
- On error, an overload returns a null `Value`, a null op, or `failure()`. Section 5.5 follows these up to the driver.

**The calling sequence.** The overloads call each other in a fixed pattern, from the driver down to the `builder.create<...>` calls that make ops (a diagram of `MLIRGen.cpp` and `toyc.cpp`, not tool output):

```text
main()                                         toyc.cpp
└─ dumpMLIR()                                  toyc.cpp, section 5.6
   ├─ parseInputFile()                         Chapter 1 lexer and parser → ModuleAST
   ├─ toy::mlirGen(context, moduleAST)         the public API
   │  └─ MLIRGenImpl::mlirGen(ModuleAST&)      section 5.2
   │     ├─ ModuleOp::create(getUnknownLoc())
   │     ├─ for each function:
   │     │  └─ mlirGen(FunctionAST&)           section 5.3
   │     │     ├─ mlirGen(PrototypeAST&)
   │     │     │  └─ create<FuncOp> → FuncOp::build → buildWithEntryBlock
   │     │     ├─ declare(name, blockArgument)        for each parameter
   │     │     ├─ mlirGen(ExprASTList&)        section 5.4: one call per statement
   │     │     │  ├─ mlirGen(VarDeclExprAST&)
   │     │     │  │  ├─ mlirGen(ExprAST&)             the initializer
   │     │     │  │  ├─ create<ReshapeOp>             only if a shape is declared
   │     │     │  │  └─ declare(name, value)
   │     │     │  ├─ mlirGen(ReturnExprAST&)          ends the statement loop
   │     │     │  │  ├─ mlirGen(ExprAST&)             only if there is an operand
   │     │     │  │  └─ create<ReturnOp>
   │     │     │  ├─ mlirGen(PrintExprAST&)
   │     │     │  │  ├─ mlirGen(ExprAST&)
   │     │     │  │  └─ create<PrintOp>
   │     │     │  └─ mlirGen(ExprAST&)                any other statement
   │     │     └─ create<ReturnOp> if missing, or function.setType(...) if it has an operand
   │     └─ mlir::verify(theModule)
   └─ module->dump()
```

Every expression, including every operand of another expression, goes through the dispatcher `mlirGen(ExprAST&)`, which picks one of five overloads. Each of them that has sub-expressions calls the dispatcher again, so the recursion follows the shape of the expression tree:

```text
mlirGen(ExprAST&)                switch on getKind()
├─ mlirGen(BinaryExprAST&)       mlirGen(ExprAST&) for LHS, then RHS → create<AddOp> / create<MulOp>
├─ mlirGen(VariableExprAST&)     symbolTable.lookup(name), no op
├─ mlirGen(LiteralExprAST&)      collectData (recursive) → create<ConstantOp>(type, attr)
├─ mlirGen(CallExprAST&)         mlirGen(ExprAST&) per argument → create<TransposeOp> / create<GenericCallOp>
└─ mlirGen(NumberExprAST&)       create<ConstantOp>(double)
```

`var`, `return` and `print` aren't in the dispatcher: they are statements, which only `mlirGen(ExprASTList&)` handles. Because each call generates its operands before creating its own op, ops are created deepest-first, and each operand exists before the op that uses it. Section 6.2 maps every op of a real run back to the overload that emitted it.

### 5.1 The `MLIRGenImpl` class

The generator's state is three members:

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

**`builder`: where and how ops are created.** `mlir::OpBuilder` is constructed from the context. Every op in this file is made by `builder.create<OpTy>(location, args...)`, which:

1. starts an `OperationState` for `OpTy`'s op name at `location`;
2. passes it to the `OpTy::build(builder, state, args...)` overload that matches `args`. Section 4.4, step 4 shows `ConstantOp`'s overloads: `create<ConstantOp>(loc, 5.5)` picks the `double` builder, and `create<ConstantOp>(loc, type, attr)` an ODS-generated one;
3. creates the op from the filled state, inserts it at the builder's **insertion point**, and returns it as an `OpTy`.

The insertion point is a position inside a block. The builder keeps it between calls, so after a `setInsertionPoint...` call, every `create` adds its op there, one after another in call order. This file sets the insertion point in only two places (section 5.3): at the end of the module body for each `toy.func`, and at the start of the function's entry block for its body.

Most overloads end with `return builder.create<SomeOp>(...);` although they return `mlir::Value`. This works because every Toy op that produces a value has exactly one result. ODS gives such an op the `OneTypedResult<...>` trait (`OneTypedResult<::mlir::TensorType>::Impl` in the class declaration of section 4.4, step 1), and that trait converts the op to its single result implicitly:

```bash
cd /Users/roy/study/mlir/toy/Ch2
sed -n '703,705p' /opt/homebrew/opt/llvm@20/include/mlir/IR/OpDefinition.h
```

Real output **[29]**:

```cpp
    /// If the operation returns a single value, then the Op can be implicitly
    /// converted to a Value. This yields the value of the only result.
    operator mlir::TypedValue<ResultType>() { return getResult(); }
```

**`symbolTable`: variables are SSA values.** A Toy variable never becomes an op. `symbolTable` maps its name to the `mlir::Value` that holds it: the result of the op that computed it, or a function argument. Reading `a` later is a lookup that returns that value (section 5.4). `llvm::ScopedHashTable` adds scopes to a plain map:
- A scope is a `ScopedHashTableScope` object. Constructing one opens a scope; when it is destroyed at the end of its C++ block, every name inserted while it was open is removed.
- `insert` adds to the innermost open scope. `lookup` and `count` search all open scopes.
- `mlirGen(FunctionAST&)` opens a scope for the arguments, and `mlirGen(ExprASTList&)` opens one inside it for the body's `var`s. When a function's code generation returns, both are gone, so names never leak from one function into the next.

**`theModule`** is the `builtin.module` being filled, created in section 5.2.

### 5.2 Locations and the module

Every `builder.create<...>` call needs a location. A tiny helper converts a Toy AST location, the file/line/column the lexer recorded (`struct Location` in `include/toy/Lexer.h`), into an MLIR `FileLineColLoc`:

***mlir/MLIRGen.cpp***
```cpp
/// Helper conversion for a Toy AST location to an MLIR location.
mlir::Location loc(const Location &loc) {
  return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file), loc.line,
                                   loc.col);
}
```

The file name becomes a `StringAttr`, so it is stored once in the context and shared by every location in the file. It is the input path exactly as given on the command line, which is why the output in sections 2.3 and 6.2 carries `loc("../../test_Example/Toy/Ch2/codegen.toy":5:10)`, and why stdin input (section 5.5) shows `loc("-":...)`.

The top-level overload creates an empty `builtin.module`, generates each function into it, and **verifies** the result:

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

- **`ModuleOp::create(builder.getUnknownLoc())`** creates a standalone module op, with its one region and empty body block. It isn't inserted anywhere, since nothing encloses it. The source has no single location for "the whole file", so it gets the explicit unknown location, `loc(unknown)` in the output (section 2.3).
- **The loop** calls `mlirGen(FunctionAST&)` for each function in source order, so the `toy.func` ops appear in the module in that order. It ignores the returned `FuncOp`, which matters when a function fails (section 5.5).
- **`mlir::verify(theModule)`** walks every op in the module. It runs the structural checks every op gets (such as each operand being defined before use), the checks ODS generated from each op's fields (`verifyInvariants()`, section 4.4, step 0), and the hand-written `verify()` methods (section 4.4, step 3). MLIRGen never checks shapes or types itself; it relies on this pass. If verification fails, the overload emits `module verification error` at the module's location and returns `nullptr` (section 5.5 has a run).

### 5.3 Functions

A function is made in two steps: `mlirGen(PrototypeAST&)` creates the `toy.func` with its signature, and `mlirGen(FunctionAST&)` fills in the body. `declare()` puts a name in the symbol table:

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
```

- **`declare()`** fails if the name is already in the table. Because `count` searches all open scopes (section 5.1), that includes the function's arguments. On failure it emits no message (section 5.5).
- **Every argument is typed `tensor<*xf64>`**: `getType(VarType{})` with an empty shape is the unranked tensor type (section 5.4 shows `getType`). Toy functions are generic over shapes, and Chapter 4 infers the actual shapes. The result list starts empty, so a function returns nothing until a `return` with a value is found.
- **`builder.create<FuncOp>(location, name, funcType)`** uses `FuncOp`'s custom builder in `Dialect.cpp`:

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

`buildWithEntryBlock` is provided by `FunctionOpInterface` (appendix A.3). Besides setting the name and type, it creates the body region with one **entry block**, and gives that block one argument per input type. Those block arguments are the function's parameters, printed `%arg0`, `%arg1`. That is why `mlirGen(FunctionAST&)` can take `function.front()` right after creating the op:

***mlir/MLIRGen.cpp***
```cpp
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

Step by step:

1. **`varScope`** opens the symbol-table scope for the arguments. It is closed when the function returns.
2. **`setInsertionPointToEnd(theModule.getBody())`**, then `mlirGen(*funcAST.getProto())`: the new `toy.func` is appended to the module, after the functions already generated.
3. **`entryBlock = function.front()`** is the block created by `buildWithEntryBlock`. `protoArgs` is the AST's parameter list, an `ArrayRef<std::unique_ptr<VariableExprAST>>`.
4. **`llvm::zip(protoArgs, entryBlock.getArguments())`** walks the two lists in step. Each `nameValue` is a tuple: `std::get<0>` is the AST parameter (`->getName()` is `"a"`), and `std::get<1>` the matching block argument (`%arg0`). `declare` binds the name to the value, so a later `a` in the body resolves to `%arg0`. A parameter name used twice (`def f(a, a)`) fails here. Unlike step 6, this early return doesn't erase the function, and section 5.5 shows the consequence.
5. **`setInsertionPointToStart(&entryBlock)`**: from here on, every `create` in the body appends to the entry block in emission order. (The block is empty, so its start is also its end.)
6. **`mlirGen(*funcAST.getBody())`** generates the statements (section 5.4). On failure the half-built function is erased from the module.
7. **The return.** Every block must end with a terminator, and `toy.return` is the terminator `toy.func` expects (appendix A.7). If the block's last op isn't a `ReturnOp`, one without operands is created at the end, using the prototype's location. `dyn_cast<ReturnOp>` yields a null `ReturnOp` for any other op. If the source's `return` has an operand (`hasOperand()`, an `extraClassDeclaration` helper in appendix A.7), the function type is rebuilt with the same inputs and one `tensor<*xf64>` result.

A small program shows all three cases: a function returning a value, one without a `return`, and `main`. It isn't a repo file, so it is passed on stdin (`-`):

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 - -emit=mlir <<'EOF'
def id(a) {
  return a;
}

def noret(a) {
  print(a);
}

def main() {
  var x = id([1, 2]);
  noret(x);
}
EOF
```

Real output **[30]**:

```mlir
module {
  toy.func @id(%arg0: tensor<*xf64>) -> tensor<*xf64> {
    toy.return %arg0 : tensor<*xf64>
  }
  toy.func @noret(%arg0: tensor<*xf64>) {
    toy.print %arg0 : tensor<*xf64>
    toy.return
  }
  toy.func @main() {
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
    %1 = toy.generic_call @id(%0) : (tensor<2xf64>) -> tensor<*xf64>
    %2 = toy.generic_call @noret(%1) : (tensor<*xf64>) -> tensor<*xf64>
    toy.return
  }
}
```

- **`@id`**: `return a` emits no op for `a`; the lookup returns the block argument, so the return's operand is `%arg0` itself. Because that return has an operand, the type was patched to `-> tensor<*xf64>`.
- **`@noret`** has no `return`, so step 7 appended an operand-free `toy.return`, and the type keeps its empty result list (no `->`).
- **`@main`**: both functions are called through `toy.generic_call`, whose builder always gives the call one `tensor<*xf64>` result (appendix A.4). That is true even for `@noret`, which returns nothing; Chapter 2 doesn't check calls against their callee. The call passes the ranked `tensor<2xf64>` constant to a `tensor<*xf64>` parameter; Chapter 4's shape inference specializes such calls. `noret(x);` is an expression statement: its value `%2` is created and never used.

### 5.4 Statements and expressions

**Statements.** A function body is an `ExprASTList`. Its overload opens a second scope, for the body's `var`s, and handles the three kinds of statement that can only appear at the top level of a function body, never nested in an expression:

***mlir/MLIRGen.cpp***
```cpp
/// Codegen a list of expression, return failure if one of them hit an error.
llvm::LogicalResult mlirGen(ExprASTList &blockAST) {
  ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);
  for (auto &expr : blockAST) {
    // Specific handling for variable declarations, return statement, and
    // print. These can only appear in block list and not in nested
    // expressions.
    if (auto *vardecl = dyn_cast<VarDeclExprAST>(expr.get())) {
      if (!mlirGen(*vardecl))
        return mlir::failure();
      continue;
    }
    if (auto *ret = dyn_cast<ReturnExprAST>(expr.get()))
      return mlirGen(*ret);
    if (auto *print = dyn_cast<PrintExprAST>(expr.get())) {
      if (mlir::failed(mlirGen(*print)))
        return mlir::success();
      continue;
    }

    // Generic expression dispatch codegen.
    if (!mlirGen(*expr))
      return mlir::failure();
  }
  return mlir::success();
}
```

- **`var`** generates the declaration. A null result fails the whole body.
- **`return`** generates the return and *returns* from the loop, so statements after a `return` are never generated.
- **`print`** generates the print. On failure, though, this branch returns `success()`, not `failure()`: the rest of the body is skipped, but the function is kept (section 5.5 shows the effect).
- **Anything else** is an expression statement such as `noret(x);`. It goes to the expression dispatcher, and its value is dropped.

**The expression dispatcher** switches on the node's kind, LLVM-style RTTI (`getKind()` plus `cast<>`, from Chapter 1's `AST.h`), and calls the matching overload:

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

**`var` declarations** generate the initializer, add a reshape if the declaration gives a shape, and bind the name:

***mlir/MLIRGen.cpp***
```cpp
/// Handle a variable declaration, we'll codegen the expression that forms the
/// initializer and record the value in the symbol table before returning it.
/// Future expressions will be able to reference this variable through symbol
/// table lookup.
mlir::Value mlirGen(VarDeclExprAST &vardecl) {
  auto *init = vardecl.getInitVal();
  if (!init) {
    emitError(loc(vardecl.loc()),
              "missing initializer in variable declaration");
    return nullptr;
  }

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
}
```

The reshape's result type is `getType(vardecl.getType())`, the ranked type of the declared shape, such as `tensor<2x3xf64>` for `var a<2, 3>`. The name is bound to the reshape's result, not the initializer's, so later uses of `a` see the declared shape. This is why `var a<2, 3> = [[1, 2, 3], [4, 5, 6]]` produces a constant *plus* a reshape that changes nothing (section 6.2). Chapter 3 optimizes such reshapes away. Without a shape (`var x = ...`), the name is bound to the initializer's value directly.

**Binary operators** generate both operands first, then the op:

***mlir/MLIRGen.cpp***
```cpp
/// Emit a binary operation
mlir::Value mlirGen(BinaryExprAST &binop) {
  // First emit the operations for each side of the operation before emitting
  // the operation itself. For example if the expression is `a + foo(a)`
  // 1) First it will visiting the LHS, which will return a reference to the
  //    value holding `a`. This value should have been emitted at declaration
  //    time and registered in the symbol table, so nothing would be
  //    codegen'd. If the value is not in the symbol table, an error has been
  //    emitted and nullptr is returned.
  // 2) Then the RHS is visited (recursively) and a call to `foo` is emitted
  //    and the result value is returned. If an error occurs we get a nullptr
  //    and propagate.
  //
  mlir::Value lhs = mlirGen(*binop.getLHS());
  if (!lhs)
    return nullptr;
  mlir::Value rhs = mlirGen(*binop.getRHS());
  if (!rhs)
    return nullptr;
  auto location = loc(binop.loc());

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
}
```

Generating LHS, then RHS, then the op means every operand's defining op is created before the op that uses it, which is the order SSA requires. Each `create` appends at the insertion point, so the order of the C++ calls *is* the order of the ops in the block. In `transpose(a) * transpose(b)` the two `toy.transpose` ops come before `toy.mul`. The op's location is `binop.loc()`, which the parser sets at the start of the right-hand side (section 6.2's table: `*` gets `5:25`). Only `+` and `*` have ops; any other operator falls out of the `switch` into `emitError`.

**Variable references** are pure lookups:

***mlir/MLIRGen.cpp***
```cpp
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

**Array literals** become one `toy.constant` that holds all the numbers in a `DenseElementsAttr`:

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

/// Recursive helper function to accumulate the data that compose an array
/// literal. It flattens the nested structure in the supplied vector. For
/// example with this array:
///  [[1, 2], [3, 4]]
/// we will generate:
///  [ 1, 2, 3, 4 ]
/// Individual numbers are represented as doubles.
/// Attributes are the way MLIR attaches constant to operations.
void collectData(ExprAST &expr, std::vector<double> &data) {
  if (auto *lit = dyn_cast<LiteralExprAST>(&expr)) {
    for (auto &value : lit->getValues())
      collectData(*value, data);
    return;
  }

  assert(isa<NumberExprAST>(expr) && "expected literal or number expr");
  data.push_back(cast<NumberExprAST>(expr).getValue());
}
```

(The doc comment above `mlirGen(LiteralExprAST&)` is left out.) For `[[1, 2, 3], [4, 5, 6]]`, `lit.getDims()` is `{2, 3}`:

- **`type`** is the op's result type, `getType({2, 3})` = `tensor<2x3xf64>`.
- **`data`** gets room for 2 × 3 = 6 values (`std::accumulate` multiplies the dimensions). `collectData` then flattens the nested literal into `{1, 2, 3, 4, 5, 6}`, row by row: it recurses into each nested `LiteralExprAST` and appends each `NumberExprAST` as a `double`.
- **`dataType`** is the attribute's type, the same `tensor<2x3xf64>` built directly.
- **`DenseElementsAttr::get(dataType, data)`** stores the six doubles with that shape. This is the standard MLIR pattern: constant data goes into an attribute.
- **`builder.create<ConstantOp>(loc, type, dataAttribute)`** calls the ODS-generated builder that takes a result type and the attribute (section 4.4, step 4). `ConstantOp::verify` (section 4.4, step 3) later checks that the two types match.

**Single numbers** use the hand-written `double` builder of section 4.4, step 4, which wraps the value in a rank-0 `tensor<f64>` (appendix D.1 shows one):

***mlir/MLIRGen.cpp***
```cpp
/// Emit a constant for a single number (FIXME: semantic? broadcast?)
mlir::Value mlirGen(NumberExprAST &num) {
  return builder.create<ConstantOp>(loc(num.loc()), num.getValue());
}
```

**Calls** generate their arguments first, then pick an op by callee name:

***mlir/MLIRGen.cpp***
```cpp
/// Emit a call expression. It emits specific operations for the `transpose`
/// builtin. Other identifiers are assumed to be user-defined functions.
mlir::Value mlirGen(CallExprAST &call) {
  llvm::StringRef callee = call.getCallee();
  auto location = loc(call.loc());

  // Codegen the operands first.
  SmallVector<mlir::Value, 4> operands;
  for (auto &expr : call.getArgs()) {
    auto arg = mlirGen(*expr);
    if (!arg)
      return nullptr;
    operands.push_back(arg);
  }

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
}
```

`transpose` is a builtin with its own op and exactly one argument. Any other name is taken to be a user function, and becomes a `toy.generic_call` whose `callee` attribute is a symbol reference to the function (appendix A.4). Nothing checks here that the function exists or takes that many arguments.

**`print` and `return`** each generate their operand, if any, and then the op:

***mlir/MLIRGen.cpp***
```cpp
/// Emit a print expression. It emits specific operations for two builtins:
/// transpose(x) and print(x).
llvm::LogicalResult mlirGen(PrintExprAST &call) {
  auto arg = mlirGen(*call.getArg());
  if (!arg)
    return mlir::failure();

  builder.create<PrintOp>(loc(call.loc()), arg);
  return mlir::success();
}
```

***mlir/MLIRGen.cpp***
```cpp
/// Emit a return operation. This will return failure if any generation fails.
llvm::LogicalResult mlirGen(ReturnExprAST &ret) {
  auto location = loc(ret.loc());

  // 'return' takes an optional expression, handle that case here.
  mlir::Value expr = nullptr;
  if (ret.getExpr().has_value()) {
    if (!(expr = mlirGen(**ret.getExpr())))
      return mlir::failure();
  }

  // Otherwise, this return operation has zero operands.
  builder.create<ReturnOp>(location,
                           expr ? ArrayRef(expr) : ArrayRef<mlir::Value>());
  return mlir::success();
}
```

`toy.return` takes an optional operand (`Variadic` with at most one value, appendix A.7). `ArrayRef(expr)` makes a one-element list from the value, `ArrayRef<mlir::Value>()` an empty one.

**Types.** Finally the type helper, the source of every `tensor<*xf64>` in the output:

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

An empty shape (an untyped `var` or a function parameter) gives the unranked `tensor<*xf64>`; a shape gives a ranked type.

### 5.5 Error handling

MLIRGen reports an error where it finds it, then passes failure up to its caller:

- **Reporting.** `emitError(location, message)` sends a diagnostic through the context. With no handler installed, it is printed to stderr as `loc(...): error: message`. The returned diagnostic accepts more text through `<<`, as in `<< expr.getName() << "'"`.
- **Passing it up.** The failing overload returns a null `Value` or `failure()`, and each caller checks and returns the same. If the body fails, `mlirGen(FunctionAST&)` then erases the half-built function (section 5.3, step 6); the next two runs show the exceptions.
- **At the module.** `mlirGen(ModuleAST&)` ignores each function's result (section 5.2), so a failed function is simply missing from the module. The module still verifies, gets printed, and `toyc-ch2` exits 0. Only a verifier failure returns `nullptr`, which makes the driver exit 1 (section 5.6).

The three runs below show one case each: an error that leaves the function in place, a function dropped silently, and the verifier failure.

#### An unknown variable inside `print`

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 - -emit=mlir <<'EOF'
def main() {
  print(y);
}
EOF
```

Real output **[31]** (exit status 0):

```text
loc("-":2:9): error: error: unknown variable 'y'
module {
  toy.func @main() {
    toy.return
  }
}
```

- **The diagnostic** points at `y` (line 2, column 9). It says `error:` twice because the message text in `mlirGen(VariableExprAST&)` itself starts with `"error: "`.
- **`main` is kept.** `mlirGen(PrintExprAST&)` returns `failure()`, but the `print` branch of `mlirGen(ExprASTList&)` turns that into `success()` (section 5.4). The function looks complete, so the implicit `toy.return` is added and nothing is erased.
- **Exit status 0**, although an error was reported: the module verifies.

#### A name declared twice

`declare()` fails without emitting anything:

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 - -emit=mlir <<'EOF'
def f(a) {
  var a = 1;
  print(a);
}

def g() {
  var b = 1;
  var b = 2;
  print(b);
}

def main() {
  print([1, 2]);
}
EOF
```

Real output **[32]** (exit status 0):

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
    toy.print %0 : tensor<2xf64>
    toy.return
  }
}
```

- **`@g`** declares `b` twice in the same scope, so the second `declare` fails.
- **`@f`** declares a `var a` in the body scope while the argument `a` is in the enclosing scope. `ScopedHashTable` could shadow it, but `declare` checks `count`, which searches every open scope (section 5.1), so this fails too.
- In both functions `mlirGen(VarDeclExprAST&)` returns `nullptr`, the body fails, and the function is erased. With no message at all, the only sign is that `@f` and `@g` are missing from the output.

#### A repeated parameter: the verifier catches what MLIRGen left behind

Here the argument `declare` in `mlirGen(FunctionAST&)` fails (section 5.3, step 4). That early `return nullptr` comes *before* the body is generated, and unlike the body-failure path it doesn't erase the function:

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 - -emit=mlir <<'EOF'
def f(a, a) {
  print(a);
}

def main() {
  print([1, 2]);
}
EOF
```

Real output **[33]** (exit status 1):

```text
loc("-":1:1): error: empty block: expect at least a terminator
error: module verification error
```

- **An empty `toy.func` is left in the module**: the op and its entry block from section 5.3, steps 2–3, with no ops and no `toy.return`.
- **`mlir::verify` rejects it.** A block must end with a terminator, and the diagnostic carries `@f`'s location (`1:1`). This is MLIR's structural check, not one of Toy's.
- **`mlirGen(ModuleAST&)` adds its own message.** The module's location is unknown, so the message has no `loc(...)` prefix. It then returns `nullptr`: no module is printed, and the driver exits 1.

### 5.6 The driver

The driver `toyc.cpp` gains a `-emit=mlir` action next to Chapter 1's `-emit=ast`, and an `-x` option that says how to read the input:

***toyc.cpp***
```cpp
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum InputType { Toy, MLIR };
} // namespace
static cl::opt<enum InputType> inputType(
    "x", cl::init(Toy), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(Toy, "toy", "load the input file as a Toy source.")),
    cl::values(clEnumValN(MLIR, "mlir",
                          "load the input file as an MLIR file")));

namespace {
enum Action { None, DumpAST, DumpMLIR };
} // namespace
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")));
```

- **The input file** is a positional argument that defaults to `-`, standard input.
- **`-x toy`** (the default) or **`-x mlir`** chooses the input kind. A file name ending in `.mlir` selects MLIR as well (below).
- **`-emit=ast`** prints Chapter 1's AST dump (appendix D.3); **`-emit=mlir`** prints the module.

`dumpMLIR()` handles the two input paths, and `main()` registers MLIR's command-line options before parsing its own:


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

- **The `.toy` path** runs when the input kind is `toy` and the name doesn't end in `.mlir`. `parseInputFile()` (Chapter 1's lexer and parser) builds the AST, `mlirGen` builds and verifies the module (section 5.2), and `module->dump()` prints it. `module->dump()` writes to stderr, which is why the commands in this README add `2>&1` (section 6.2).
- **The `.mlir` path** runs otherwise. `getFileOrSTDIN` reads the file, or standard input for `-`. `mlir::parseSourceFile` parses it with the registered `toy` dialect, including custom parsers such as `ConstantOp::parse`, and verifies the result. This path is what makes the round trip of section 6.3 possible.
- **Exit statuses.** 0 when a module was printed, even if MLIRGen reported errors on the way (section 5.5). 6 when the `.toy` path got no AST (the file couldn't be opened, or the parser gave up). 1 when MLIRGen's verification failed. 3 when the `.mlir` path couldn't parse or verify the input (`Error can't load file`). -1 (seen by the shell as 255) when the `.mlir` path couldn't open the file.
- **`registerAsmPrinterCLOptions()` and `registerMLIRContextCLOptions()`** add MLIR's printing and context flags to the command line. That is where `-mlir-print-debuginfo` and `-mlir-print-op-generic` come from.

---

## 6. Build and Run

The superbuild (`toy/build.sh`, `toy/run.sh`, `CMakePresets.json`) is documented once in the top-level [README](../README.md#the-build-system). This section builds and runs Chapter 2 on its own, from the chapter directory. What the chapter's `CMakeLists.txt` adds — the TableGen wiring — is in appendix E.

Every command in this README is numbered by its output (**[1]**–**[59]**). After building, `./examples.sh` runs them all, printing each command followed by its output and exit status; `./examples.sh 7` runs only command [7], and `./examples.sh -l` lists them (appendix F).

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

**What the two commands do.**
- `cmake -S . -B build -G Ninja` *configures*. It reads `CMakeLists.txt`, finds the MLIR and LLVM CMake packages, and writes a Ninja build file into `build/`. Nothing is compiled yet.
- `cmake --build build` runs Ninja, which builds in dependency order:
  1. **TableGen.** This is the first chapter with a TableGen step. The `ToyCh2OpsIncGen` target runs `mlir-tblgen` four times over `include/toy/Ops.td`, writing the generated declarations and definitions (sections 3.2 and 4.4) into `build/include/toy/`.
  2. **Compile.** The four C++ sources are compiled: `toyc.cpp` (the driver, section 5.6), `parser/AST.cpp` (Chapter 1's AST dumper), `mlir/MLIRGen.cpp` (section 5) and `mlir/Dialect.cpp` (sections 3.3 and 4). `Dialect.h` and `Dialect.cpp` include the generated files, which is why step 1 must come first (appendix E.1).
  3. **Link.** The objects are linked against Homebrew's `libMLIR` and `libLLVM` into `build/toyc-ch2`.

The result:

```bash
cd /Users/roy/study/mlir/toy/Ch2
ls build/toyc-ch2 build/include/toy/
```

Real output **[34]**:

```text
build/toyc-ch2

build/include/toy/:
CMakeFiles
Dialect.cpp.inc
Dialect.cpp.inc.d
Dialect.h.inc
Dialect.h.inc.d
Ops.cpp.inc
Ops.cpp.inc.d
Ops.h.inc
Ops.h.inc.d
cmake_install.cmake
```

- **`build/toyc-ch2`** is the compiler. A standalone build puts it directly in `build/`.
- **The four `.inc` files** are the TableGen output: `Dialect.h.inc`/`Dialect.cpp.inc` for the dialect class and `Ops.h.inc`/`Ops.cpp.inc` for the nine op classes. `build/include/` mirrors the source `include/` layout, so `#include "toy/Ops.h.inc"` resolves here (appendix E.1 has the include paths).
- **Each `.inc.d`** is a dependency file: it lists every `.td` file the generator read, including MLIR's `OpBase.td`. Ninja uses it to rerun `mlir-tblgen` when any of them changes, not only `Ops.td`.
- **`CMakeFiles/` and `cmake_install.cmake`** are CMake's own bookkeeping for this subdirectory.

### 6.2 Running

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 ../../test_Example/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo 2>&1
```

- `../../test_Example/Toy/Ch2/codegen.toy` — the input program, listed in section 4.5; the path is relative to `Ch2/`.
- `-emit=mlir` — take the driver's `DumpMLIR` action (vs. `-emit=ast` from Chapter 1, which still works).
- `-mlir-print-debuginfo` — ask the asm printer to include `loc(...)` on every operation (locations are always *stored*, just not printed by default). This flag exists because `main()` called `mlir::registerAsmPrinterCLOptions()` (section 5.6).
- `2>&1` — merges stderr, where `module->dump()` writes, into stdout (why it matters: below).

Real output **[35]** on this machine (every `loc(...)` reproduces the input path exactly as given on the command line — here `../../test_Example/Toy/Ch2/codegen.toy`):

```mlir
module {
  toy.func @multiply_transpose(%arg0: tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":4:1), %arg1: tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":4:1)) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":5:10)
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":5:25)
    %2 = toy.mul %0, %1 : tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":5:25)
    toy.return %2 : tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":5:3)
  } loc("../../test_Example/Toy/Ch2/codegen.toy":4:1)
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":9:17)
    %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":9:3)
    %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":10:17)
    %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":10:3)
    %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":11:11)
    %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":12:11)
    toy.print %5 : tensor<*xf64> loc("../../test_Example/Toy/Ch2/codegen.toy":13:3)
    toy.return loc("../../test_Example/Toy/Ch2/codegen.toy":8:1)
  } loc("../../test_Example/Toy/Ch2/codegen.toy":8:1)
} loc(unknown)
```

The input is a `.toy` file, so the run takes the driver's `.toy` path (section 5.6): Chapter 1's parser builds the AST, `mlirGen` builds and verifies the module (section 5), and `module->dump()` prints it in custom form, each op through its own printer (section 4.5). The exit status is 0.

How to read the output:
- **Value names restart in each function.** `multiply_transpose` and `main` both start at `%0`. A `toy.func` is isolated from above (appendix A.3): its body can't use values defined outside it, so the printer numbers each function on its own. Compare the generic form in section 4.5, which numbers across the whole module.
- **Block arguments are the parameters.** `%arg0`, `%arg1` are the entry block's arguments (section 5.3), shown inside the signature in the custom form, each with its own location.
- **Arguments share the function's location.** `%arg0` and `%arg1` both carry `4:1`, the position of `def`, because `mlirGen(PrototypeAST&)` uses one location for the whole prototype.

Line-by-line mapping back to the `.toy` source, with the overload that emitted each op (section 5.4 explains them):

| MLIR | Emitted by | Toy source |
|---|---|---|
| `module { ... } loc(unknown)` | `mlirGen(ModuleAST&)`, with `getUnknownLoc()` | the whole file |
| `toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: ...)` | `mlirGen(PrototypeAST&)`; parameters are unranked | `def multiply_transpose(a, b)` at 4:1 |
| `%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>` | `mlirGen(CallExprAST&)`, builtin path → `TransposeOp` | `transpose(a)` at 5:10 |
| `%1 = toy.transpose(%arg1 ...)` | the same | `transpose(b)` at 5:25 |
| `%2 = toy.mul %0, %1 : tensor<*xf64>` | `mlirGen(BinaryExprAST&)`, `'*'` → `MulOp`; one type printed because all types match (appendix A.2) | `*`, located at the right-hand side, 5:25 |
| `toy.return %2 : tensor<*xf64>` | `mlirGen(ReturnExprAST&)` | `return ...;` at 5:3 |
| `-> tensor<*xf64>` on the function | `mlirGen(FunctionAST&)`'s `function.setType(...)`, because the return has an operand | the `return` with a value |
| `%0 = toy.constant dense<[[1.0...]]> : tensor<2x3xf64>` | `mlirGen(LiteralExprAST&)`: the nested literal, flattened | `[[1, 2, 3], [4, 5, 6]]` at 9:17 |
| `%1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>` | `mlirGen(VarDeclExprAST&)`: the declared shape `<2, 3>`; this reshape changes nothing (Chapter 3 removes it) | `var a<2, 3> = ...` at 9:3 |
| `%2 = toy.constant dense<[1.0, ..., 6.0]> : tensor<6xf64>` | `mlirGen(LiteralExprAST&)`: a flat literal gives a rank-1 tensor | `[1, 2, 3, 4, 5, 6]` at 10:17 |
| `%3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64>` | `mlirGen(VarDeclExprAST&)`: this reshape does change the shape, rank 1 → rank 2 | `var b<2, 3> = ...` at 10:3 |
| `%4 = toy.generic_call @multiply_transpose(%1, %3) : (...) -> tensor<*xf64>` | `mlirGen(CallExprAST&)`, user-function path → `GenericCallOp`; the result stays unranked until Chapter 4's shape inference | `multiply_transpose(a, b)` at 11:11 |
| `%5 = toy.generic_call @multiply_transpose(%3, %1) ...` | the same, arguments swapped; `%4` (`c`) is never used | `multiply_transpose(b, a)` at 12:11 |
| `toy.print %5 : tensor<*xf64>` | `mlirGen(PrintExprAST&)` → `PrintOp` | `print(d);` at 13:3 |
| `toy.return` (no operand) | the implicit return added by `mlirGen(FunctionAST&)`, at the prototype's location | the end of `main`, located at 8:1 |

**Why `2>&1`.** `module->dump()` prints to stderr, like LLVM's other `dump()` debugging helpers. Without the redirect a pipe receives nothing. Discard stderr and count the bytes that reach stdout:

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 ../../test_Example/Toy/Ch2/codegen.toy -emit=mlir 2>/dev/null | wc -c
```

Real output **[36]**:

```text
       0
```

### 6.3 The round trip and why it matters

A **round trip** prints the IR as text, parses the text back, and prints it again. If the second print matches the first, every op's printer and parser agree with each other. `toyc-ch2` can do the parsing half because of its `.mlir` path (section 5.6): with `-x mlir`, or an input name ending in `.mlir`, it skips the Toy frontend and hands the text to `mlir::parseSourceFile`. That function parses with the `toy` dialect's registered parsers and then verifies the module.

#### Through a pipe

Compare the first emission with the same emission parsed and printed again:

```bash
cd /Users/roy/study/mlir/toy/Ch2
diff <(./build/toyc-ch2 ../../test_Example/Toy/Ch2/codegen.toy -emit=mlir 2>&1) \
     <(./build/toyc-ch2 ../../test_Example/Toy/Ch2/codegen.toy -emit=mlir 2>&1 | ./build/toyc-ch2 - -x mlir -emit=mlir 2>&1) \
  && echo "identical"
```

Real output **[37]**:

```text
identical
```

- **`<(...)`** is bash process substitution: it runs a command and passes its output to `diff` as if it were a file.
- **The left side** is the normal emission, the module of section 6.2 without locations.
- **The right side** pipes that emission into a second `toyc-ch2`. The input name `-` means stdin (`getFileOrSTDIN`, section 5.6). `-x mlir` selects the MLIR path, which a name without a `.mlir` suffix needs.
- **`diff` prints nothing and exits 0 when its inputs are equal**, so `&& echo identical` runs: the re-parsed module prints exactly like the original.

#### Through a file, with locations

Locations survive the trip too. The first `toyc-ch2` writes the module with `loc(...)` annotations to a `.mlir` file in a temporary directory. The second reads it, choosing the MLIR path from the suffix alone, and prints it again with `-mlir-print-debuginfo`:

```bash
cd /Users/roy/study/mlir/toy/Ch2
tmp=$(mktemp -d)
./build/toyc-ch2 ../../test_Example/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo 2> $tmp/codegen.mlir
./build/toyc-ch2 $tmp/codegen.mlir -emit=mlir -mlir-print-debuginfo 2>&1 | diff - $tmp/codegen.mlir && echo "identical"
rm -r $tmp
```

Real output **[38]**:

```text
identical
```

The parser reads each `loc("../../test_Example/Toy/Ch2/codegen.toy":5:10)` back as a `FileLineColLoc`, and the printer writes it out unchanged. So the reprinted module points at the original `.toy` source, not at the temporary `.mlir` file, and matches section 6.2's output byte for byte.

#### Why this is a meaningful test

The two halves exercise *both directions* of every custom assembly format:

- **Emit path** exercises the *printers*: `ConstantOp::print`, `printBinaryOp`, the `assemblyFormat`-generated printers for `transpose`/`reshape`/`print`/`generic_call`/`return`, and `printFunctionOp`.
- **Re-parse path** exercises the *parsers*: `ConstantOp::parse` (including recovering the result type from the attribute), `parseBinaryOp` (including the single-type shorthand `: tensor<*xf64>`), the format-generated parsers (`(` … `:` … `)` … `to` …), and `parseFunctionOp`.
- **Parsing re-runs the verifiers**, because `parseSourceFile` verifies what it parsed. Text that parses but breaks an op's rules is rejected.

Editing the text between the two halves shows both failures.

**A printer/parser mismatch.** Suppose a printer wrote `into` where its parser expects `to`. A `sed` makes that change in the emitted text:

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 ../../test_Example/Toy/Ch2/codegen.toy -emit=mlir 2>&1 \
  | sed 's/ to tensor/ into tensor/' \
  | ./build/toyc-ch2 - -x mlir -emit=mlir
```

Real output **[39]** (exit status 3):

```text
loc("<stdin>":3:47): error: custom op 'toy.transpose' expected 'to'
Error can't load file -
```

- **The parser generated from `toy.transpose`'s `assemblyFormat`** (appendix A.8) stops at the first token that doesn't fit: line 3 is the first `toy.transpose`, and column 47 is where `into` begins.
- **The location names `<stdin>`**, MLIR's name for standard input, not `-`: this diagnostic comes from MLIR's parser, while the Toy lexer used the name as given (section 5.5).
- **The driver exits with status 3** (`Error can't load file`, section 5.6), so a round trip in a test fails loudly.

**IR that parses but doesn't verify.** `main` returns nothing, so its `toy.return` has no operand. Give it one:

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 ../../test_Example/Toy/Ch2/codegen.toy -emit=mlir 2>&1 \
  | sed 's/^    toy.return$/    toy.return %5 : tensor<*xf64>/' \
  | ./build/toyc-ch2 - -x mlir -emit=mlir
```

Real output **[40]** (exit status 3):

```text
loc("<stdin>":16:5): error: 'toy.return' op does not return the same number of values (1) as the enclosing function (0)
Error can't load file -
```

The text is well formed, so parsing succeeds. Then `ReturnOp::verify` in `Dialect.cpp` (appendix A.7) compares the operand count with the enclosing `toy.func`'s result count and rejects it. A printer that emitted IR violating an op's own rules would be caught the same way.

A dialect's tests commonly chain the two halves the same way, `toyc-ch2 ... -emit=mlir 2>&1 | toyc-ch2 - -x mlir -emit=mlir 2>&1 | FileCheck ...`, so every format is tested in both directions.

### 6.4 Inspecting the generated C++

You don't need CMake to see what ODS generates — [`run_mlir-tblgen.sh`](run_mlir-tblgen.sh) (run from inside `Ch2/`) invokes `mlir-tblgen` directly, once per generator, with the Homebrew MLIR headers on the include path (needed to resolve `include "mlir/IR/OpBase.td"` etc.):

***run_mlir-tblgen.sh***
```bash
mkdir -p ./build
mlir-tblgen -gen-dialect-decls ./include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include/ -o ./build/dialect-decls.inc
mlir-tblgen -gen-dialect-defs ./include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include/ -o ./build/dialect-defs.inc
mlir-tblgen -gen-op-decls ./include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include/ -o ./build/op-decls.inc
mlir-tblgen -gen-op-defs ./include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include/ -o ./build/op-defs.inc
```

Run it and count what it wrote:

```bash
cd /Users/roy/study/mlir/toy/Ch2
./run_mlir-tblgen.sh && wc -l build/*.inc
```

Real output **[41]**:

```text
      26 build/dialect-decls.inc
      25 build/dialect-defs.inc
    1593 build/op-decls.inc
    1582 build/op-defs.inc
    3226 total
```

- **The dialect files are small**: mainly one class, its constructor and destructor (section 3.2).
- **The op files are about 1,600 lines each** for nine ops. Every op gets a class, adaptor classes and all their method bodies.

**The same text as the build's.** The files land in `Ch2/build/`, next to but separate from the CMake-generated `.inc` files in `build/include/toy/` (section 6.1). The generators and the input are the same, and so is the output:

```bash
cd /Users/roy/study/mlir/toy/Ch2
for pair in dialect-decls:Dialect.h dialect-defs:Dialect.cpp op-decls:Ops.h op-defs:Ops.cpp; do
  diff -q build/${pair%%:*}.inc build/include/toy/${pair#*:}.inc && echo "build/${pair%%:*}.inc = build/include/toy/${pair#*:}.inc"
done
```

Real output **[42]**:

```text
build/dialect-decls.inc = build/include/toy/Dialect.h.inc
build/dialect-defs.inc = build/include/toy/Dialect.cpp.inc
build/op-decls.inc = build/include/toy/Ops.h.inc
build/op-defs.inc = build/include/toy/Ops.cpp.inc
```

`${pair%%:*}` and `${pair#*:}` split each `script-name:build-name` pair at the colon. `diff -q` is silent for equal files, so each line is printed only when the two files match. Reading `build/op-decls.inc` is therefore reading exactly what `Dialect.h` includes (section 3.3).

**Finding your way in `op-decls.inc`.** For each op, the file declares a group of classes. `AddOp`'s:

```bash
cd /Users/roy/study/mlir/toy/Ch2
grep -oE '^class AddOp[A-Za-z]*( : public [^<{]*)?' build/op-decls.inc
```

Real output **[43]**:

```text
class AddOp
class AddOpGenericAdaptorBase
class AddOpGenericAdaptor : public detail::AddOpGenericAdaptorBase 
class AddOpAdaptor : public AddOpGenericAdaptor
class AddOp : public ::mlir::Op
```

- **`class AddOp`** (the first line) is a forward declaration. The file starts with one for all nine ops, so any class can refer to any op.
- **`AddOpGenericAdaptorBase`**, in namespace `detail`, holds the parts of the adaptor that don't depend on how operands are stored: attributes, properties, and the operand-group arithmetic.
- **`AddOpGenericAdaptor`** adds the named operand accessors (`getLhs()`, `getRhs()`) over any range type. **`AddOpAdaptor`** is that template over `mlir::ValueRange`, with its own `verify()`. Section 4.4, step 0 explains what adaptors are for.
- **`AddOp : public ::mlir::Op`** is the op class itself, followed by its trait list (cut by the `grep`). Section 4.4, step 0 explains it member by member.

`op-defs.inc` has the method bodies for all these classes, wrapped in the `GET_OP_LIST` and `GET_OP_CLASSES` sections that `Dialect.cpp` selects (section 3.3). The `mlir-tblgen` commands of sections 3.2 and 4.4 print the same text in pieces; appendix B.4 reads more of it. `-gen-op-doc` (section 4.4, step 2) is the other generator worth trying.

---

## 7. Key Takeaways & Pitfalls

**Takeaways**

1. **One concept scales the whole IR**: operations (with operands, results, attributes, regions, locations) model everything from modules to arithmetic. Dialects namespace them; contexts load dialects.
2. **Registered beats opaque**: MLIR will round-trip unknown ops, but only registration buys verification, typed accessors, custom syntax, and a foundation for optimization (section 2.4).
3. **ODS is leverage**: a few lines of TableGen per op generate the whole op class (compare `Ops.td` with the generated files of section 6.4). Hand-written C++ remains available exactly where the declarative fields run out, and `Dialect.cpp` holds all of it (section 4.6).
4. **Verification is layered**: MLIR's structural checks (each block ends with a terminator, values are defined before use), each trait's checks, the checks ODS derives from the fields (`verifyInvariantsImpl()`), and, with `hasVerifier = 1`, your `verify()`, which runs only after the ODS checks pass. `mlir::verify(module)` in MLIRGen, and `parseSourceFile` on the `.mlir` path, run all of them.
5. **MLIRGen is small on purpose**: a stateful `OpBuilder` (insertion point!), a `ScopedHashTable` symbol table, a `loc()` helper, and one `mlirGen` overload per AST node (section 5). Parameters and call results are left unranked (`tensor<*xf64>`); inference comes in Chapter 4, reshape cleanups in Chapter 3.
6. **Round-tripping is the cheapest dialect test you'll ever write**: it checks printers and parsers against each other, and re-runs the verifiers (section 6.3).
7. **The generic form is the ground truth**: every op, registered or not, reduces to name, operands, attributes, result types, location, successors and regions, and `-mlir-print-op-generic` shows it. Custom formats are just sugar over it. Inherent attributes are *properties*, printed as `<{...}>` (appendix C.1).

**Pitfalls**

- **Forgetting `context.getOrLoadDialect<ToyDialect>()`**: parsing `.mlir` input then fails with ``Dialect `toy' not found for custom op`` even though the code compiled fine (section 3.3).
- **A promised method that is never defined**: a bodiless builder, `hasVerifier = 1` or `hasCustomAssemblyFormat = 1` without the matching definition in `Dialect.cpp` compiles but fails to link (appendix B.5).
- **`GET_OP_LIST` vs. `GET_OP_CLASSES`**: `Ops.cpp.inc` holds two parts, and each include must select one with its macro (section 3.3).
- **`dump()` writes to stderr**: without `2>&1`, a pipe receives nothing (section 6.2).
- **MLIRGen errors don't change the exit status**: a function that fails is dropped from the module and `toyc-ch2` still exits 0. A failing `print` keeps its function, and a redeclared name drops its function without any message (section 5.5).
- **Locations are easy to squander**: use `loc(expr.loc())` for every `builder.create<>`; falling back to `getUnknownLoc()` everywhere destroys diagnostics later.
- **Verifier ordering assumption**: your `verify()` runs *after* the ODS-generated checks, so it may rely on operand counts and types already being validated, but nothing more. `ReturnOp::verify` can `cast<FuncOp>` its parent only because of the `HasParent<"FuncOp">` trait.
- **Custom `parse()` must fully populate `OperationState`**: forgetting `result.addTypes(...)` (which `ConstantOp::parse` derives from the attribute type) yields an op with zero results and confusing errors later.
- **Out-of-tree TableGen ordering**: without `add_dependencies(toyc-ch2 ToyCh2OpsIncGen)`, Ninja can compile `Dialect.cpp` before the generated `.inc` files exist (appendix E.1).

---

## Appendix A: The other eight ops

Sections 4.1–4.5 followed `toy.constant`; this appendix covers the rest of `Ops.td` op by op, then every piece of the declarative assembly format the file uses.

### A.1 All nine ops at a glance

| Op | Traits | Arguments (`ins`) | Results | Assembly format | Verifier | Custom builders |
|---|---|---|---|---|---|---|
| `toy.constant` (4.4–4.5) | `Pure` | `F64ElementsAttr:$value` | `F64Tensor` | C++ | yes | `(DenseElementsAttr)`, `(double)` |
| `toy.add` / `toy.mul` (A.2) | — | `F64Tensor:$lhs, F64Tensor:$rhs` | `F64Tensor` | C++ (shared) | no | `(Value, Value)` |
| `toy.func` (A.3) | `FunctionOpInterface`, `IsolatedFromAbove` | `$sym_name`, `$function_type`, optional `$arg_attrs`/`$res_attrs`; region `$body` | — | C++ (library helpers) | no | `(name, type, attrs)`; `skipDefaultBuilders` |
| `toy.generic_call` (A.4) | — | `FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs` | `F64Tensor` | declarative | no | `(callee, arguments)` |
| `toy.print` (A.5) | — | `F64Tensor:$input` | — | declarative | no | — |
| `toy.reshape` (A.6) | — | `F64Tensor:$input` | `StaticShapeTensorOf<[F64]>` | declarative | no (ODS constraint) | — |
| `toy.return` (A.7) | `Pure`, `HasParent<"FuncOp">`, `Terminator` | `Variadic<F64Tensor>:$input` | — | declarative (optional group) | yes | `()` |
| `toy.transpose` (A.8) | — | `F64Tensor:$input` | `F64Tensor` | declarative | yes | `(Value)` |

The custom builders of `add`, `mul`, `generic_call` and `transpose` always produce an **unranked** result (`tensor<*xf64>`); only `toy.constant` (typed by its literal) and `toy.reshape` (typed by the declared shape) get ranked types in this chapter. Shape inference comes in Chapter 4.

### A.2 `AddOp` and `MulOp` — binary element-wise ops with a shared hand-written syntax

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

(`MulOp` = same shape with mnemonic `"mul"`.) The declared two-`Value` builder and the `parse`/`print` hooks are implemented in `Dialect.cpp`:

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

The matching `parseBinaryOp` reads the same pieces back:

1. exactly two operands;
2. an optional attribute dictionary;
3. a `:` and a type.

If that type is a `FunctionType`, its inputs type the operands and its results the result. Otherwise the single type is used for both operands and the result. A round trip shows both forms. The `toy.add` is written with a full functional type, but its types all match, so it comes back in the short form. The `toy.mul` mixes a ranked and an unranked operand, so it keeps the functional type:

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 - -x mlir -emit=mlir <<'EOF'
toy.func @f(%a: tensor<2xf64>, %b: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.add %a, %a : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
  %1 = toy.mul %0, %b : (tensor<2xf64>, tensor<*xf64>) -> tensor<*xf64>
  toy.return %1 : tensor<*xf64>
}
EOF
```

Real output **[44]**:

```mlir
module {
  toy.func @f(%arg0: tensor<2xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.add %arg0, %arg0 : tensor<2xf64>
    %1 = toy.mul %0, %arg1 : (tensor<2xf64>, tensor<*xf64>) -> tensor<*xf64>
    toy.return %1 : tensor<*xf64>
  }
}
```

### A.3 `FuncOp` — regions, interfaces, `extraClassDeclaration`

`FuncOp` is a Toy function, and the op with the most structure:

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
- **`skipDefaultBuilders = 1`**: suppress the autogenerated builders, which would create the body region without the entry block MLIRGen relies on (section 5.3); the single custom builder leans on the interface helper instead:

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

The custom `parse`/`print` also just delegate to library helpers (`mlir::function_interface_impl::parseFunctionOp` / `printFunctionOp`), which is why `toy.func @main() { ... }` looks exactly like `func.func`.

### A.4 `GenericCallOp` — symbol references and variadic operands

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
- **`assemblyFormat`**: variables (`$callee`, `$inputs`) print/parse the corresponding argument; back-ticked literals (`` `(` ``) are punctuation; `attr-dict` is the mandatory directive for remaining attributes; `functional-type($inputs, results)` prints `(operand types) -> result types`. Result: `toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>`.
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

### A.5 `PrintOp` — the minimal declarative op

Section 4.5 uses `toy.print` to introduce custom assembly formats. This repo uses the declarative form:

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

`type($input)` is the directive that prints/parses the type of `$input` — producing `toy.print %5 : tensor<*xf64>`. Section 4.5 walks through this exact format: the hand-written C++ `parse`/`print` it replaces, and the directive/literal/variable components it is made of.

### A.6 `ReshapeOp` — result type constraints

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

- **`StaticShapeTensorOf<[F64]>`**: a stronger constraint than `F64Tensor` — the result must be a *statically shaped* f64 tensor. This constraint is enforced by the ODS-generated `verifyInvariantsImpl()` for free; no hand-written verifier needed.
- The `assemblyFormat` produces `%1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>` — note the free-form keyword literal `` `to` ``.

An unranked reshape result is rejected by that generated check alone:

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 - -x mlir -emit=mlir <<'EOF'
toy.func @main() {
  %0 = toy.constant dense<[1.0, 2.0]> : tensor<2xf64>
  %1 = toy.reshape(%0 : tensor<2xf64>) to tensor<*xf64>
  toy.return
}
EOF
```

Real output **[45]** (exit status 3):

```text
loc("<stdin>":3:8): error: 'toy.reshape' op result #0 must be statically shaped tensor of 64-bit float values, but got 'tensor<*xf64>'
Error can't load file -
```

### A.7 `ReturnOp` — traits (`Terminator`, `HasParent`), optional operands

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
- The **verifier** cross-checks the return against the *enclosing function's* signature, a case of contextual verification. Section 6.3 shows it rejecting a `toy.return` with an operand in `main`, which returns nothing:

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

### A.8 `TransposeOp` — the running example

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

Both outcomes in one input: the first `toy.transpose` turns `2x3` into `3x2` and passes, the second claims `2x3` again and fails:

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 - -x mlir -emit=mlir <<'EOF'
toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
  toy.return
}
EOF
```

Real output **[46]** (exit status 3):

```text
loc("<stdin>":4:8): error: expected result shape to be a transpose of the input
Error can't load file -
```

The diagnostic points at line 4, the second transpose; the first one, on line 3, was verified before it and passed. The message has no `'toy.transpose' op` prefix because `verify()` uses `emitError()` rather than `emitOpError()`.

### A.9 Every piece of the declarative format

The [declarative format](https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format) of section 4.5 has more pieces than `toy.print` uses. Every piece of the declarative format used anywhere in this repo's `Ops.td`:

| Piece | Kind | Meaning | Used by |
|---|---|---|---|
| `$name` | variable | print/parse that operand or attribute | all declarative ops |
| `attr-dict` | directive | the remaining (non-elided) attributes, if any; mandatory in every format | all declarative ops |
| `type($x)` | directive | the type of operand `$x` | `print`, `reshape`, `transpose`, `return` |
| `type(results)` | directive | the result type(s) | `reshape`, `transpose` |
| `functional-type($ins, results)` | directive | `(operand types) -> result types` | `generic_call` |
| `` `(` ``, `` `:` ``, `` `to` `` | literal | punctuation or a keyword, printed as-is | `generic_call`, `reshape`, `transpose` |
| `( ... )?` with `$x^` | optional group | print/parse the group only if the anchor `$x` is present | `return` |

---

## Appendix B: `Ops.td` and `Dialect.cpp` as a whole

With every op explained, here is `Ops.td` in one view: its syntax, a map, every field it sets, and what the generator makes of it. B.5 maps the hand-written half, `Dialect.cpp`.

### B.1 TableGen syntax, on `Ops.td`

Every TableGen construct `Ops.td` uses, each on a real line of the file:

| Construct | In `Ops.td` | Meaning |
|---|---|---|
| `include "file.td"` | `include "mlir/IR/OpBase.td"` | Textual include, searched on the `-I` path; brings in the classes used below |
| `#ifndef` / `#define` / `#endif` | `#ifndef TOY_OPS` | Include guard (TableGen has a minimal preprocessor) |
| `class Name<params> : Parent<args>;` | `class Toy_Op<string mnemonic, list<Trait> traits = []> : Op<Toy_Dialect, mnemonic, traits>;` | A *template* for records: typed parameters (with optional defaults) passed on to the parent class. Generates nothing by itself |
| `def Name : Parent<args> { ... }` | `def PrintOp : Toy_Op<"print"> { ... }` | A concrete record. Each `def` derived from `Dialect` or `Op` becomes one C++ class with the same name (`toy::PrintOp`) |
| `let field = value;` | `let summary = "print operation";` | Overrides a field inherited from the parent class; fields you don't `let` keep their defaults |
| `"..."` / `[{ ... }]` | `"constant"` / `let description = [{ ... }];` | String / *code* literal: multi-line, no escaping — used for long docs and embedded C++ |
| `[a, b]` | `[Pure, HasParent<"FuncOp">, Terminator]` | List |
| `Class<args>` as a value | `Variadic<F64Tensor>`, `HasParent<"FuncOp">` | An anonymous record instantiated in place |
| `(op arg:$name, ...)` | `(ins F64Tensor:$lhs, F64Tensor:$rhs)` | A *dag*: an operator (`ins`, `outs`, `region`) applied to arguments. `$name` names an argument; ODS turns names into accessors (`getLhs()`) and uses them in assembly formats |
| `0` / `1` | `let hasVerifier = 1;` | Value of a `bit` field — an on/off flag |
| `$_builder`, `$_state` | `[{ build($_builder, $_state, {}); }]` | Placeholders inside code literals, substituted by `mlir-tblgen` (an ODS convention, not TableGen syntax) |

### B.2 A map of `Ops.td`

| Lines | Content | Explained in |
|---|---|---|
| 13–14, 335 | Include guard `TOY_OPS` | — |
| 16 | `include "mlir/IR/OpBase.td"` — `Dialect`, `Op`, `OpBuilder`, `CArg`, core traits (`IsolatedFromAbove`, `Terminator`, `HasParent`), `AnyRegion`, `Variadic`, and the type/attribute constraints (`F64Tensor`, `F64ElementsAttr`, `SymbolNameAttr`, ...) | 4.3 |
| 17 | `include "mlir/Interfaces/FunctionInterfaces.td"` — `FunctionOpInterface` | A.3 |
| 18 | `include "mlir/IR/SymbolInterfaces.td"` — the symbol interfaces. `FunctionInterfaces.td` already includes this file, so the line is redundant: deleting it leaves the generated code unchanged | — |
| 19 | `include "mlir/Interfaces/SideEffectInterfaces.td"` — `Pure` | 4.4, step 5 |
| 21–26 | `def Toy_Dialect` | 3.2 |
| 28–34 | `class Toy_Op`, the base class of every op | 4.3 |
| 44–86 | `ConstantOp` | 4.4, 4.5 |
| 92–109, 209–226 | `AddOp`, `MulOp` | A.2 |
| 115–164 | `FuncOp` | A.3 |
| 170–203 | `GenericCallOp` | A.4 |
| 232–243 | `PrintOp` | A.5 |
| 249–268 | `ReshapeOp` | A.6 |
| 274–310 | `ReturnOp` | A.7 |
| 316–333 | `TransposeOp` | A.8 |

A name is only visible if some include defines it. Drop the `SideEffectInterfaces.td` include and TableGen stops at the first use of `Pure`:

```bash
cd /Users/roy/study/mlir/toy/Ch2
sed '/SideEffectInterfaces.td/d' include/toy/Ops.td \
  | mlir-tblgen -gen-op-decls -I /opt/homebrew/opt/llvm@20/include
```

Real output **[47]** (exit status 1; line 47 because the `sed` removed one line above it):

```text
<stdin>:47:38: error: Variable not defined: 'Pure'
def ConstantOp : Toy_Op<"constant", [Pure]> {
                                     ^
```

### B.3 Fields used in `Ops.td`

Every field the file sets, with its type and default as `--print-records` shows them (section 4.4, step 0), and where each is explained:

| Field | Type, default | What it does | Set by | Explained in |
|---|---|---|---|---|
| traits (2nd `Toy_Op` argument) | `list<Trait>`, `[]` | Attaches traits and interfaces to the op class | `constant`, `func`, `return` | 4.4 step 5, A.3, A.7 |
| `summary` | `string`, `""` | One-line description for generated docs | all nine | 4.4 step 2 |
| `description` | `string`, `""` | Long description for generated docs | all but `transpose` | 4.4 step 2 |
| `arguments` | `dag`, `(ins)` | Operands and attributes, in order; `$names` become accessors | all nine | 4.4 step 1; A.3 (attributes), A.4 (variadic) |
| `results` | `dag`, `(outs)` | Result types | `constant`, `add`, `mul`, `generic_call`, `reshape`, `transpose` | 4.4 step 1; A.6 (constraints) |
| `regions` | `dag`, `(region)` | Attached regions | `func` | A.3 |
| `builders` | `list<OpBuilder>`, `?` | Extra `build()` overloads, with an inline body or defined in `Dialect.cpp` | all but `print`, `reshape` | 4.4 step 4; A.7 (inline body) |
| `skipDefaultBuilders` | `bit`, `0` | Suppresses the generated default builders | `func` | A.3 |
| `hasVerifier` | `bit`, `0` | Declares a hand-written `verify()` | `constant`, `return`, `transpose` | 4.4 step 3; A.7, A.8 |
| `hasCustomAssemblyFormat` | `bit`, `0` | Declares hand-written `parse()`/`print()` | `constant`, `add`, `mul`, `func` | 4.5; A.2, A.3 |
| `assemblyFormat` | `string`, `?` | Declarative syntax; ODS generates `parse()`/`print()` | `generic_call`, `print`, `reshape`, `return`, `transpose` | 4.5; A.4–A.8 |
| `extraClassDeclaration` | `string`, `?` | C++ pasted verbatim into the generated class | `func`, `return` | A.3; A.7 |

### B.4 Demystifying TableGen: the actually-generated code

Section 4.4 showed `ConstantOp`'s generated class, an op with no operands and a C++ format. `TransposeOp` has an operand and a declarative format; this is its generated class:

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen -gen-op-decls include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include | awk '/^class TransposeOp : /,/^};/'
```

Real output **[48]** (abridged):

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

The trait list is derived from `arguments`/`results` as in section 4.4, step 0, but this op has one operand, so it gets `OneOperand` where `ConstantOp` got `ZeroOperands`. The generated definitions wire verification the same way as `ConstantOp`'s (section 4.4, step 3): **the ODS checks run before your `verify()`**:

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen -gen-op-defs include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include \
  | awk '/^::llvm::LogicalResult TransposeOp::verifyInvariants\(\) \{/,/^}/'
```

Real output **[49]**:

```cpp
::llvm::LogicalResult TransposeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}
```

The `assemblyFormat` string is compiled into a `TransposeOp::parse()` that calls `parseLParen()`, `parseOperand()`, `parseColon()`, `parseKeyword("to")` and so on, as section 4.5 showed for `PrintOp`. Section 6.4 writes the complete generated files to disk.

### B.5 `Dialect.h` and `Dialect.cpp`

Section 4.6 lists what `Dialect.cpp` defines, op by op. This is the same code in file order, with the two headers it is built from, and what happens when a definition is missing.

**`include/toy/Dialect.h`** is the whole header, shown in section 3.3. It includes the MLIR headers the generated code needs, then `Dialect.h.inc` (the `ToyDialect` class) and `Ops.h.inc` under `GET_OP_CLASSES` (the nine op classes). Three files include it: `mlir/Dialect.cpp`, which defines the declared methods, and `mlir/MLIRGen.cpp` and `toyc.cpp`, which use the ops and the dialect.

**`mlir/Dialect.cpp`**, top to bottom:

| Lines | Content | Explained in |
|---|---|---|
| 14–33 | `#include "toy/Dialect.h"`, MLIR and LLVM headers, `using namespace mlir; using namespace mlir::toy;` | — |
| 35 | `#include "toy/Dialect.cpp.inc"`: the generated `ToyDialect` constructor and destructor | 3.2, 3.3 |
| 37–48 | `ToyDialect::initialize()`, with `Ops.cpp.inc` included under `GET_OP_LIST` for the op list | 3.3 |
| 50–100 | `static parseBinaryOp()`, `static printBinaryOp()` | A.2 |
| 102–170 | `ConstantOp`: `build(double)`, `parse()`, `print()`, `verify()` | 4.4 steps 3–4, 4.5 |
| 172–187 | `AddOp`: `build()`, `parse()`, `print()` | A.2 |
| 189–200 | `GenericCallOp::build()` | A.4 |
| 202–236 | `FuncOp`: `build()`, `parse()`, `print()` | A.3, 5.3 |
| 238–253 | `MulOp`: `build()`, `parse()`, `print()` | A.2 |
| 255–290 | `ReturnOp::verify()` | A.7 |
| 292–315 | `TransposeOp`: `build()`, `verify()` | A.8 |
| 317–322 | `#define GET_OP_CLASSES` and `#include "toy/Ops.cpp.inc"`: every generated method body | 3.3, 4.4 step 0 |

- **`Ops.cpp.inc` is included twice**, each time selecting a different part with a macro: the op list inside `initialize()`, and the method bodies at the end (section 3.3).
- **`parseBinaryOp` and `printBinaryOp` are `static`**: file-local helpers, not members of any op. ODS knows nothing about them. `AddOp` and `MulOp` share one syntax, so their `parse()`/`print()` are one-line calls to these helpers.
- **One banner comment per op**, and only the seven ops that need hand-written code are present. The order is close to `Ops.td`'s, except that `GenericCallOp` comes before `FuncOp`.

#### A missing definition fails at link time

The promise works through ordinary C++: a declaration in `Ops.h.inc` with no definition anywhere. To see it break, copy the chapter's sources to a temporary directory, delete `ConstantOp::verify()` from the copy, and build it. The `sed` range deletes from the function's first line to its closing `}`:

```bash
cd /Users/roy/study/mlir/toy/Ch2
tmp=$(mktemp -d)
cp -R CMakeLists.txt toyc.cpp parser mlir include "$tmp"/
sed -i '' '/^llvm::LogicalResult ConstantOp::verify() {/,/^}/d' "$tmp"/mlir/Dialect.cpp
cmake -S "$tmp" -B "$tmp"/build -G Ninja > /dev/null
cmake --build "$tmp"/build 2>&1 | sed -n '/^Undefined symbols/,/linker command failed/p'; echo "cmake --build exit status: ${PIPESTATUS[0]}"
rm -r "$tmp"
```

Real output **[50]** (the build's progress lines are cut by the second `sed`, and the second caller's trait list is abridged):

```text
Undefined symbols for architecture arm64:
  "mlir::toy::ConstantOp::verify()", referenced from:
      mlir::toy::ConstantOp::verifyInvariants() in Dialect.cpp.o
      mlir::Op<mlir::toy::ConstantOp, mlir::OpTrait::ZeroRegions, ...>::verifyInvariants(mlir::Operation*) in Dialect.cpp.o
ld: symbol(s) not found for architecture arm64
clang++: error: linker command failed with exit code 1 (use -v to see invocation)
cmake --build exit status: 1
```

- **Compiling succeeds.** `Dialect.cpp.o` exists, because the declaration in `Ops.h.inc` is all the compiler needs. Only the linker finds that there is no body.
- **The callers are generated code**, both compiled into `Dialect.cpp.o` from `Ops.cpp.inc`. `ConstantOp::verifyInvariants()` calls `verify()` because of `hasVerifier = 1` (section 4.4, step 3). The `mlir::Op<...>::verifyInvariants(Operation*)` hook, which MLIR's verifier calls for every op, calls it as well.
- **Every promise behaves the same way.** Removing a bodiless builder, or the `parse()`/`print()` of a `hasCustomAssemblyFormat` op, gives the same kind of undefined-symbol error for that method.

---

## Appendix C: Printing details and stock `mlir-opt`

Two topics around the printed IR: what MLIR 20's generic printer adds, and what a tool without the Toy dialect can and cannot read.

### C.1 Properties, `function_type` and value numbering in the generic form

The generic dump of section 4.5 shows three things that come from how MLIR 20 prints, not from Toy:

- **Inherent attributes print as properties**, in `<{...}>` before the region: `<{function_type = ..., sym_name = "multiply_transpose"}>`, `<{value = dense<...>}>`, `<{callee = @multiply_transpose}>`. A discardable-attribute dictionary `{...}` after the op would be empty, so it isn't printed. This is the default for every ODS dialect: `Toy_Dialect` (section 3.2) doesn't set `usePropertiesForAttributes`, and its record (`--print-records`, section 4.4, step 0) shows the inherited value:

  ```bash
  cd /Users/roy/study/mlir/toy/Ch2
  mlir-tblgen --print-records include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include \
    | awk '/^def Toy_Dialect /,/^}/' | grep usePropertiesForAttributes
  ```

  Real output **[51]**:

  ```text
    bit usePropertiesForAttributes = 1;
  ```
- **The function signature is `function_type`**, because that is what `FuncOp` names it in `arguments` (`TypeAttrOf<FunctionType>:$function_type`, appendix A.3). `FunctionOpInterface` expects that name.
- **Values are numbered across the whole module**: `main` gets `%0`–`%5` and `multiply_transpose` gets `%6`–`%8`, although it prints first. The generic form implies `-mlir-print-unique-ssa-ids`, so the numbering doesn't restart in each function, and the printer happens to number the last function first. Adding `-mlir-print-unique-ssa-ids` to the custom-form run gives the same `%6`–`%8`:

  ```bash
  cd /Users/roy/study/mlir/toy/Ch2
  ./build/toyc-ch2 ../../test_Example/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-unique-ssa-ids 2>&1 | sed -n '2,6p'
  ```

  Real output **[52]**:

  ```mlir
    toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
      %6 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
      %7 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
      %8 = toy.mul %6, %7 : tensor<*xf64>
      toy.return %8 : tensor<*xf64>
  ```

### C.2 Feeding Toy IR to stock `mlir-opt`

`toyc-ch2` reads Toy IR because it loads the Toy dialect; the *stock* Homebrew `mlir-opt` doesn't. Section 2.4 named the two requirements for reading an op from text: the parser must understand its syntax, and it must be allowed to create ops from a dialect it doesn't have. `-allow-unregistered-dialect` covers only the second; here is `toyc-ch2`'s output failing the first.

#### The custom form fails, flag or no flag

`toyc-ch2` prints the custom form by default (section 6.2), e.g. `toy.func @multiply_transpose(...)` and `toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>`. Pipe it into stock `mlir-opt` with the flag:

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 ../../test_Example/Toy/Ch2/codegen.toy -emit=mlir 2>&1 \
  | /opt/homebrew/opt/llvm@20/bin/mlir-opt -allow-unregistered-dialect
```

Real output **[53]** (exit status 1; the dialect list is abridged):

```text
<stdin>:2:3: error: Dialect `toy' not found for custom op 'toy.func' 
  toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
  ^
<stdin>:2:3: note: Registered dialects: acc, affine, amdgpu, amx, arith, ... ; for more info on dialect registration see https://mlir.llvm.org/getting_started/Faq/#registered-loaded-dependent-whats-up-with-dialects-management
```

- The syntax after `toy.func` (`@name`, the signature, the `to` keyword in `toy.transpose`) is defined by the toy dialect's own `parse()` methods (section 4.5). Stock `mlir-opt` doesn't contain them, so the parser can't even read the first op.
- It fails before the policy question arises, so the flag changes nothing: dropping `-allow-unregistered-dialect` gives the identical error and exit status.

#### The generic form round-trips, but only with the flag

Print the same module in **generic form** and the syntax requirement is met — MLIR's core parser reads it without any Toy code:

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 ../../test_Example/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-op-generic 2>&1 \
  | /opt/homebrew/opt/llvm@20/bin/mlir-opt -allow-unregistered-dialect
```

Real output **[54]** (abridged, exit status 0):

```mlir
module {
  "toy.func"() <{function_type = (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>, sym_name = "multiply_transpose"}> ({
  ^bb0(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>):
    %0 = "toy.transpose"(%arg0) : (tensor<*xf64>) -> tensor<*xf64>
    %1 = "toy.transpose"(%arg1) : (tensor<*xf64>) -> tensor<*xf64>
    %2 = "toy.mul"(%0, %1) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
  ...
  }) : () -> ()
}
```

- Every `toy.*` op is held opaquely and printed back in generic form; only `builtin.module`, which stock `mlir-opt` does know, comes back in its custom form `module { ... }`. The values are renumbered (`%6` → `%0`), since names aren't part of the IR (section 2.4).
- Here the flag is what makes it work. Without it, the same command fails on the policy check (exit status 1):

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 ../../test_Example/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-op-generic 2>&1 \
  | /opt/homebrew/opt/llvm@20/bin/mlir-opt
```

Real output **[55]** (exit status 1):

```text
<stdin>:2:13: error: operation being parsed with an unregistered dialect. If this is intended, please use -allow-unregistered-dialect with the MLIR tool used
  "toy.func"() <{function_type = (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>, sym_name = "multiply_transpose"}> ({
            ^
```

The four combinations at a glance:

| `toyc-ch2` prints | stock `mlir-opt` | with `-allow-unregistered-dialect` |
|---|---|---|
| custom form | ✗ ``Dialect `toy' not found for custom op`` (syntax) | ✗ same error |
| generic form | ✗ `operation being parsed with an unregistered dialect` (policy) | ✓ round-trips, exit status 0 |

The lesson: MLIR's *generic* syntax needs no dialect at all — it's pure structure — while the *custom* syntax, verification, and semantics all come from registration. That's why every project with a dialect ships its own tool, such as a `foo-opt` binary, with the dialect loaded.

---

## Appendix D: More inputs to try

`test_Example/Toy/Ch2/` has three more Toy inputs besides `codegen.toy`. (`invalid.mlir`, the negative test, is run in section 2.4.)

### D.1 A scalar initializer

***test_Example/Toy/Ch2/scalar.toy***
```text
# RUN: toyc-ch2 %s -emit=mlir 2>&1 | FileCheck %s

def main() {
  var a<2, 2> = 5.5;
  print(a);
}
...
```

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 ../../test_Example/Toy/Ch2/scalar.toy -emit=mlir 2>&1
```

Real output **[56]**:

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<5.500000e+00> : tensor<f64>
    %1 = toy.reshape(%0 : tensor<f64>) to tensor<2x2xf64>
    toy.print %1 : tensor<2x2xf64>
    toy.return
  }
}
```

The number `5.5` goes through `mlirGen(NumberExprAST&)` and the `double` builder of `ConstantOp` (section 4.4, step 4), which makes a rank-0 `tensor<f64>`. The declared shape `<2, 2>` then adds a `toy.reshape`, as for `var a<2, 3>` in section 6.2. The output matches the file's `# CHECK:` lines (elided above).

### D.2 An input with no function

The file's own test runs `-emit=ast` and checks for the parse error; the run below uses `-emit=mlir` to show what MLIRGen makes of an empty AST.

***test_Example/Toy/Ch2/empty.toy***
```text
# RUN: toyc-ch2 %s -emit=ast 2>&1 | FileCheck %s
# CHECK-NOT: Assert
# CHECK: Parse error
```

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 ../../test_Example/Toy/Ch2/empty.toy -emit=mlir 2>&1
```

Real output **[57]** (exit status 0):

```text
Parse error (4, 0): expected 'def' in prototype but has Token -1
module {
}
```

The file is all comments, so the parser reaches end of file (token `-1`, line 4) while still expecting a `def`. It reports the error and stops, and MLIRGen turns the empty module AST into an empty `module {}`. The driver still exits with status 0.

### D.3 The Chapter 1 action

`-emit=ast` still prints the AST, exactly as in Chapter 1:

```bash
cd /Users/roy/study/mlir/toy/Ch2
./build/toyc-ch2 ../../test_Example/Toy/Ch2/ast.toy -emit=ast 2>&1
```

Real output **[58]** (abridged):

```text
  Module:
    Function 
      Proto 'multiply_transpose' @../../test_Example/Toy/Ch2/ast.toy:4:1
      Params: [a, b]
      Block {
        Return
          BinOp: * @../../test_Example/Toy/Ch2/ast.toy:5:25
            Call 'transpose' [ @../../test_Example/Toy/Ch2/ast.toy:5:10
              var: a @../../test_Example/Toy/Ch2/ast.toy:5:20
            ]
...
```

---

## Appendix E: What Chapter 2 adds to the build

This is the chapter where the build stops being a plain C++ compile — `mlir-tblgen` now generates C++ from `Ops.td` before anything else can compile.

### E.1 The chapter targets and the TableGen wiring

The chapter's `CMakeLists.txt` starts with boilerplate that runs only in a standalone build (finding MLIR and LLVM, section 6.1). After it, the chapter adds:

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

Notable differences from an in-tree MLIR build:

- **Monolithic shared libraries**: Homebrew's LLVM ships `libMLIR.dylib` / `libLLVM.dylib`, so instead of listing fine-grained components (`MLIRAnalysis`, `MLIRIR`, `MLIRParser`, …) we link just two libraries.
- **Explicit `add_dependencies(toyc-ch2 ToyCh2OpsIncGen)`**: in-tree helper macros normally add this dependency for you. Out-of-tree, without it, Ninja may try to compile `Dialect.cpp` before TableGen has produced the generated `.inc` files (`toy/Dialect.h.inc`, `toy/Ops.h.inc`), a classic build race. This line was added to fix exactly that.
- Two include roots: the *source* `include/` (for `Ops.td`, `Dialect.h`) and the chapter's *binary* include dir (`${CMAKE_CURRENT_BINARY_DIR}/include/`), where the generated `.inc` files land mirroring the source layout. In a standalone chapter build it is `Ch2/build/include/toy/*.inc` (section 6.1 lists it).

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

### E.2 Every later chapter repeats this pattern

Chapters 3–7 all keep this exact structure — `include/toy/CMakeLists.txt` generating the op/dialect `.inc` files into a `ToyChNOpsIncGen` target, plus the explicit `add_dependencies` race guard — and each adds its own extra `.td` → generator pairs on top (DRR rewriters in Ch3, interfaces in Ch4). Their READMEs describe only those additions.

---

## Appendix F: Running this README's commands: `examples.sh`

[`examples.sh`](examples.sh), a symlink to the shared [`../examples.sh`](../examples.sh), runs the commands of this README and shows each one with its output and exit status. Build the chapter first (section 6.1), then:

- `./examples.sh` runs every command, in README order;
- `./examples.sh 7` runs command [7] only (several numbers are fine);
- `./examples.sh -l` lists the numbered commands without running them:

```bash
cd /Users/roy/study/mlir/toy/Ch2
./examples.sh -l
```

Real output **[59]** (abridged):

```text
[ 1] 2.3   ./build/toyc-ch2 ../../test_Example/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-op-generic -mlir-print-debuginfo 2>&1
[ 2] 2.4   mlir-opt <<'EOF'
...
[58] D.3   ./build/toyc-ch2 ../../test_Example/Toy/Ch2/ast.toy -emit=ast 2>&1
[59] F     ./examples.sh -l
```

The script doesn't keep its own copy of the commands: it reads them from `README.md` on every run. A command is every `bash` code block in file order, except the build block (its first command after `cd` is `cmake`) and file excerpts (a `***path***` label above the fence, like `run_mlir-tblgen.sh` in section 6.4). Number *N* is the *N*-th such block, and the README marks that block's output "Real output **[N]**". Adding, removing or reordering a command block therefore renumbers the ones after it.

---

## Links

- Official doc: [Toy Tutorial Chapter 2 — Emitting Basic MLIR](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/)
- Related MLIR docs: [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/) · [Operation Definition Specification (ODS)](https://mlir.llvm.org/docs/DefiningDialects/Operations/) · [Declarative Assembly Format](https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format)
- Previous: [Chapter 1 — Toy Language and AST](../Ch1/README.md)
- Next: [Chapter 3 — High-level Language-Specific Analysis and Transformation](../Ch3/README.md)
- Back to [README](../README.md)
