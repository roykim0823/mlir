# Chapter 3: High-level Language-Specific Analysis and Transformation

> **Goal:** Use the high-level semantics preserved by the Toy dialect to implement optimizations that would be impossible (or very hard) after lowering — first with a hand-written C++ `RewritePattern`, then with declarative TableGen rewrite rules (DRR) — and hook them all into MLIR's canonicalization framework.
> Official docs: [Toy Tutorial Ch.3 — High-level Language-Specific Analysis and Transformation](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/)

**Chapter code:** `/Users/roy/study/mlir/toy/Ch3/` (an out-of-tree CMake project built against Homebrew LLVM/MLIR 20 — *not* inside `llvm-project`). It is one chapter of the **superbuild** rooted at `/Users/roy/study/mlir/toy/`: `cd toy && ./build.sh ch3` produces `build/bin/toyc-ch3`.

---

## Table of Contents

1. [Overview](#1-overview)
2. [C++ Rewrite Patterns](#2-c-rewrite-patterns)
3. [Declarative Rewrite Rules (DRR)](#3-declarative-rewrite-rules-drr)
4. [Wiring the Pass Pipeline](#4-wiring-the-pass-pipeline)
5. [Building](#5-building)
6. [Running and Testing](#6-running-and-testing)
7. [Key Takeaways & Pitfalls](#7-key-takeaways--pitfalls)
8. [Links](#links)

---

## 1. Overview

Chapter 2 gave us a Toy dialect that faithfully models the source language: `toy.transpose`, `toy.reshape`, `toy.constant`, and friends. Now we get the payoff. Because the IR still *knows* it is dealing with a tensor transpose — not a soup of loads, stores, and index arithmetic — we can express language-level algebraic identities as trivial local rewrites.

The motivating example from the official tutorial is this Toy function:

```
def transpose_transpose(x) {
  return transpose(transpose(x));
}
```

Mathematically, `transpose(transpose(x)) == x`. A sufficiently high-level compiler should turn this function into "return the argument unchanged". But consider what Clang produces if you write a naive 2D transpose in C and compose it twice:

```c
#define N 100
#define M 100

void sink(void *);
void double_transpose(int A[N][M]) {
  int B[M][N];
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      B[j][i] = A[i][j];
    }
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      A[i][j] = B[j][i];
    }
  }
  sink(B);
}
```

By the time this reaches LLVM IR, the "transpose-ness" is gone: LLVM sees two nested loops shuffling memory through a temporary buffer. Proving that the two loop nests cancel each other out requires heroic loop analysis and memory dependence reasoning, and in practice LLVM does *not* remove the work. In the Toy dialect, the same fact is a one-line pattern match: *"a transpose whose operand is produced by another transpose can be replaced by the inner transpose's operand."*

This is the core argument for multi-level IR: **do each optimization at the abstraction level where it is cheap to express and trivially correct.**

MLIR gives us two ways to write such local pattern-match-and-rewrite transformations:

| Approach | Mechanism | Best for |
|---|---|---|
| **Imperative C++** | Subclass `mlir::OpRewritePattern<Op>`, implement `matchAndRewrite` | Matches that need arbitrary C++ logic, side-effect reasoning, non-structural conditions |
| **Declarative (DRR)** | TableGen `Pat<...>` records, compiled by `mlir-tblgen -gen-rewriters` into C++ | Structural DAG-to-DAG rewrites; concise, less boilerplate, harder to get wrong |

Both feed the same engine: the **greedy pattern rewrite driver** that powers MLIR's **canonicalization pass**. In this chapter we use both:

- `transpose(transpose(x)) → x` — hand-written C++ (`SimplifyRedundantTranspose` in `Ch3/mlir/ToyCombine.cpp`)
- Three reshape optimizations — DRR (`Ch3/mlir/ToyCombine.td`)

---

## 2. C++ Rewrite Patterns

### 2.1 The pattern itself

The full C++ pattern lives in `/Users/roy/study/mlir/toy/Ch3/mlir/ToyCombine.cpp`:

```cpp
/// This is an example of a c++ rewrite pattern for the TransposeOp. It
/// optimizes the following scenario: transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  llvm::LogicalResult
  matchAndRewrite(TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current transpose.
    mlir::Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

    // Input defined by another transpose? If not, no match.
    if (!transposeInputOp)
      return failure();

    // Otherwise, we have a redundant transpose. Use the rewriter.
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};
```

Let's unpack every moving part.

**`OpRewritePattern<TransposeOp>`** — a convenience base class for patterns rooted at one specific op type. The greedy driver will only call this pattern on `toy.transpose` operations, and `matchAndRewrite` receives the operation already cast to the typed `TransposeOp` wrapper. (The more general base, `RewritePattern`, matches on an op *name* string and hands you a raw `Operation *` — that's what DRR-generated patterns use, as we'll see in §3.4.)

**`benefit`** — a heuristic cost passed to the base constructor (`/*benefit=*/1` here). When multiple patterns can fire at the same rooted op, the driver tries higher-benefit patterns first. With a single pattern per root the value barely matters; it becomes relevant when you have both a "cheap partial" and an "expensive complete" rewrite for the same op.

**`matchAndRewrite`** — the fused match + rewrite hook. The contract is strict:

- Return `failure()` **before mutating anything** if the pattern does not apply. Here: walk up from the transpose's operand via `Value::getDefiningOp<TransposeOp>()`; if the operand is a block argument or produced by some other op, `transposeInputOp` is null and we bail out.
- If it does apply, perform **all** IR mutation through the supplied `PatternRewriter` (`replaceOp`, `eraseOp`, `create<...>`, `modifyOpInPlace`, ...). The rewriter is how the driver tracks what changed, keeps its worklist up to date, and (in dialect-conversion contexts) supports rollback. Mutating IR behind the rewriter's back is undefined behavior for the driver.
- Return `success()` after rewriting.

The actual rewrite is one line: `rewriter.replaceOp(op, {transposeInputOp.getOperand()})` replaces every use of the *outer* transpose's result with the *inner* transpose's input — i.e. the original `x`.

### 2.2 Registering with the canonicalization framework

A pattern is inert until something runs it. Rather than writing a bespoke pass, we register it as a **canonicalization pattern** on `TransposeOp` (also in `ToyCombine.cpp`):

```cpp
/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}
```

`getCanonicalizationPatterns` is a static hook that MLIR's canonicalizer pass calls for **every op registered in the context** when it builds its pattern set. But the hook is not declared by default — you must opt in from ODS.

### 2.3 `hasCanonicalizer = 1` in ODS

In `/Users/roy/study/mlir/toy/Ch3/include/toy/Ops.td`, both ops that own patterns set the flag:

```tablegen
def TransposeOp : Toy_Op<"transpose", [Pure]> {
  ...
  // Enable registering canonicalization patterns with this operation.
  let hasCanonicalizer = 1;
  ...
}

def ReshapeOp : Toy_Op<"reshape", [Pure]> {
  ...
  // Enable registering canonicalization patterns with this operation.
  let hasCanonicalizer = 1;
  ...
}
```

`let hasCanonicalizer = 1;` makes `mlir-tblgen -gen-op-decls` emit the declaration into the generated header (you can verify this in `build/Ch3/include/toy/Ops.h.inc` under the superbuild tree — or `Ch3/build/include/toy/Ops.h.inc` in a standalone chapter build):

```cpp
static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                        ::mlir::MLIRContext *context);
```

We then supply the *definition* by hand in `ToyCombine.cpp`. Forgetting the flag gives you a very characteristic error: your definition in the `.cpp` fails to compile because the member function was never declared on the generated op class.

### 2.4 Dead code elimination and the `Pure` trait

Here is a subtlety the official tutorial calls out. `SimplifyRedundantTranspose` replaces the *outer* transpose, but it never erases the *inner* one. After the rewrite, the IR momentarily looks like:

```mlir
toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>   // now dead
  toy.return %arg0 : tensor<*xf64>
}
```

Why doesn't the canonicalizer just delete `%0`? Because it is **conservative about side effects**. An unregistered or effect-opaque op might print, write memory, or trap — deleting it because its result is unused would be wrong. The canonicalizer only performs dead code elimination on ops it can *prove* are side-effect free.

The fix is to declare it. In `Ops.td`, `toy.transpose` (and in this repo's Ch3, essentially every value-producing Toy op) carries the **`Pure`** trait:

```tablegen
def TransposeOp : Toy_Op<"transpose", [Pure]> {
```

`Pure` means "no side effects and always speculatable" — the strongest promise. With it, the canonicalizer's built-in DCE sweeps the dead inner transpose away, and the function collapses to just `toy.return %arg0`.

> **Note (older LLVM versions):** the trait used to be spelled `NoSideEffect`; since ~LLVM 16 it is `Pure`. This repo builds against LLVM/MLIR 20, so `Pure` is the right spelling.

**Why does `ReturnOp` (and every other op in the function) matter here?** Two related reasons:

1. **The pattern rewires uses, so the consumer must accept the new value.** `toy.return %arg0` must be valid IR. Because Toy ops are permissive about shapes at this stage (`tensor<*xf64>` unranked types, `F64Tensor` operands), replacing `%1` with `%arg0` verifies cleanly. If `ReturnOp` demanded an exact ranked type, the rewrite would produce verifier-invalid IR — a pattern must never do that.
2. **The canonicalizer processes *every* op in the region, not just transposes.** Every op it visits must be registered and verifiable; that's why Chapter 2 registered the whole dialect up front. `ReturnOp` itself is declared `[Pure, HasParent<"FuncOp">, Terminator]` in `Ops.td` — `Terminator` keeps DCE from ever considering it dead-code-removable in the "unused result" sense (it has no results), and `HasParent` keeps verification tight.

### 2.5 How the greedy driver actually runs

`createCanonicalizerPass()` wraps the **greedy pattern rewrite driver** (`applyPatternsGreedily`). Its algorithm, roughly:

1. Collect canonicalization patterns from all registered ops (via the `getCanonicalizationPatterns` hooks) plus built-in folding.
2. Put all ops on a worklist.
3. Pop an op; try to **fold** it; then try each applicable pattern in decreasing benefit order.
4. When a pattern fires, the rewriter notifies the driver, which re-enqueues affected ops (users of replaced values, producers of operands, etc.).
5. Trivially dead, side-effect-free ops are erased along the way.
6. Repeat until fixpoint (no pattern applies anywhere) or an iteration cap is hit.

The fixpoint iteration is what lets small local patterns compose into big cleanups: one `Reshape(Reshape(x))` collapse exposes another, which exposes a constant fold, and so on — exactly what we'll observe in §6.2.

---

## 3. Declarative Rewrite Rules (DRR)

Writing `matchAndRewrite` by hand is flexible but verbose — the transpose pattern took ~25 lines of C++ for a one-line idea. For purely structural DAG-to-DAG rewrites, MLIR offers **Declarative Rewrite Rules**: you write TableGen `Pattern`/`Pat` records, and `mlir-tblgen -gen-rewriters` generates the `RewritePattern` subclasses for you.

The general record shape (quoted in the header comment of our `.td` file):

```tablegen
class Pattern<
   dag sourcePattern, list<dag> resultPatterns,
   list<dag> additionalConstraints = [],
   list<dag> supplementalPatterns = [],
   dag benefitsAdded = (addBenefit 0)
>;
```

`Pat<src, result>` is the common single-result shorthand for `Pattern<src, [result]>`.

This chapter defines three reshape patterns in `/Users/roy/study/mlir/toy/Ch3/mlir/ToyCombine.td`. The file starts with the necessary includes — `PatternBase.td` for the DRR infrastructure and our own `Ops.td` so the op records (`ReshapeOp`, `ConstantOp`) are visible:

```tablegen
include "mlir/IR/PatternBase.td"
include "toy/Ops.td"
```

### 3.1 Basic pattern: `Reshape(Reshape(x)) = Reshape(x)`

```tablegen
// Reshape(Reshape(x)) = Reshape(x)
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;
```

Read the source DAG inside-out: match a `ReshapeOp` whose operand is produced by another `ReshapeOp`, binding the *inner* reshape's operand to `$arg`. The result DAG builds a single new `ReshapeOp` directly on `$arg` and replaces the matched root with it. In DRR, the result op's type is taken from the root op being replaced, so the new reshape produces the outer reshape's type — the intermediate shape is provably irrelevant since only element *data* flows through a reshape.

That's the entire `transpose`-style pattern in two lines instead of twenty-five.

### 3.2 Pattern with `NativeCodeCall`: `Reshape(Constant(x)) = x'`

Some rewrites need real C++ in the middle. `NativeCodeCall` embeds an expression (or helper-function call) whose `$0, $1, ...` placeholders are substituted with bound entities:

```tablegen
// Reshape(Constant(x)) = x'
def ReshapeConstant :
  NativeCodeCall<"$0.reshape(::llvm::cast<ShapedType>($1.getType()))">;
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))>;
```

Piece by piece:

- Source: match a `ReshapeOp` — bound to `$res` via the `op:$name` syntax — whose operand comes from a `ConstantOp`. Because `ConstantOp`'s ODS argument is its `value` attribute, `$arg` binds the **`DenseElementsAttr`** payload, not an SSA value.
- `ReshapeConstant` expands to `arg.reshape(::llvm::cast<ShapedType>(res.getType()))` — `DenseElementsAttr::reshape` re-wraps the same raw data with the target shaped type. (In this LLVM-20-based repo the cast is the free-function `::llvm::cast<ShapedType>(...)`; older tutorial text shows the deprecated `.cast<ShapedType>()` member form.)
- Result: build a fresh `ConstantOp` whose `value` attribute is the reshaped constant, and replace the reshape with it.

Net effect: a reshape applied to a compile-time constant is folded away entirely — the constant is simply *materialized in the target shape*.

### 3.3 Pattern with constraints: redundant reshape

The third pattern only applies conditionally — a reshape whose input and output types are already identical is a no-op:

```tablegen
// Reshape(x) = x, where input and output shapes are identical
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;
```

Three new concepts:

- **`Constraint<CPred<"...">>`** — a predicate over bound entities, checked during matching. `CPred` is a raw C++ boolean expression; `$0`/`$1` are substituted with the constraint's arguments (`$res`, `$arg`). The pattern fires only if the reshape's result type equals its operand's type.
- **`(replaceWithValue $arg)`** — a special DRR directive: instead of building a new op, replace all uses of the matched root with an *existing* value. This is the DRR analogue of `rewriter.replaceOp(op, existingValue)`.
- The third `Pat` argument is the **constraint list**, applied on top of structural matching.

### 3.4 What TableGen generates: `ToyCombine.inc`

The build runs `mlir-tblgen -gen-rewriters` over `ToyCombine.td`, producing `build/Ch3/ToyCombine.inc` in the superbuild tree (~176 lines of C++ from ~30 lines of TableGen). It contains one `RewritePattern` subclass per `Pat` record. Here is the actual generated match section for `FoldConstantReshapeOptPattern` (lightly trimmed):

```cpp
struct FoldConstantReshapeOptPattern : public ::mlir::RewritePattern {
  FoldConstantReshapeOptPattern(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("toy.reshape", 2, context, {"toy.constant"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    ::mlir::toy::ReshapeOp res;
    ::mlir::DenseElementsAttr arg;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::toy::ReshapeOp>(op0);
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1))
        return rewriter.notifyMatchFailure(castedOp0, ...);
      auto castedOp1 = ::llvm::dyn_cast<::mlir::toy::ConstantOp>(op1);
      if (!(castedOp1))
        return rewriter.notifyMatchFailure(op1, ...);
      auto tblgen_attr =
          op1->getAttrOfType<::mlir::DenseElementsAttr>("value");
      if (!(tblgen_attr))
        return rewriter.notifyMatchFailure(op1, ...);
      arg = tblgen_attr;
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc(
        {tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()});
    auto nativeVar_0 = arg.reshape(
        ::llvm::cast<ShapedType>((*res.getODSResults(0).begin()).getType()));
    // ... builds a new toy.constant with attribute value = nativeVar_0 ...
    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};
```

Things worth noticing in the generated code:

- The constructor passes root op name `"toy.reshape"`, **benefit 2** (auto-computed as the source-pattern node count — a 2-op match is "worth more" than a 1-op match, so deeper patterns are tried first), and `{"toy.constant"}` as *generated op names* metadata.
- The match phase is exactly the null-check ladder you would write by hand — `getDefiningOp`, `dyn_cast`, attribute presence checks — but each failure calls `notifyMatchFailure` with a human-readable diagnostic (visible under `--debug`).
- The `NativeCodeCall` string appears verbatim, with `$0`→`arg` and `$1`→`res`'s result substituted (`nativeVar_0`).
- The new constant's location is a **fused location** of both matched ops, preserving traceability.
- At the bottom, TableGen also emits `void populateWithGenerated(::mlir::RewritePatternSet &patterns)` which registers all three patterns at once — an alternative registration entry point we don't use here.

### 3.5 Hooking the generated patterns in

`ToyCombine.cpp` includes the generated file inside an anonymous namespace and registers the three classes as `ReshapeOp` canonicalization patterns:

```cpp
namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "ToyCombine.inc"
} // namespace

...

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
              FoldConstantReshapeOptPattern>(context);
}
```

From the canonicalizer's point of view there is no difference between hand-written and DRR-generated patterns — they are all `RewritePattern` instances in one `RewritePatternSet`.

---

## 4. Wiring the Pass Pipeline

Patterns run inside a pass; passes run inside a `PassManager`. The driver changes are in `/Users/roy/study/mlir/toy/Ch3/toyc.cpp`.

First, a new command-line flag:

```cpp
static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));
```

Then `dumpMLIR()` conditionally builds and runs a pipeline after MLIRGen (or after parsing an `.mlir` input):

```cpp
if (enableOpt) {
  mlir::PassManager pm(module.get()->getName());
  // Apply any generic pass manager command line options and run the pipeline.
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;

  // Add a run of the canonicalizer to optimize the mlir module.
  pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());
  if (mlir::failed(pm.run(*module)))
    return 4;
}
```

Key details:

- **`mlir::PassManager pm(module.get()->getName())`** — the pass manager is anchored on the top-level op type (`builtin.module` here).
- **`applyPassManagerCLOptions(pm)`** — transfers generic pass-manager flags from the command line onto this PM instance: `--mlir-print-ir-before-all`, `--mlir-print-ir-after-all`, `--mlir-pass-statistics`, crash reproducer options, etc. For these flags to exist at all, `main()` must call the matching registration function — and it does:

  ```cpp
  int main(int argc, char **argv) {
    // Register any command line options.
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();
    ...
  }
  ```

- **`pm.addNestedPass<mlir::toy::FuncOp>(...)`** — schedules the canonicalizer to run *nested on each `toy.func`* rather than on the whole module. This is both more precise and enables parallel execution across functions (each function is an isolated-from-above region). Adding a function-scoped pass at module scope directly would be a pipeline-structure error.
- **`mlir::createCanonicalizerPass()`** (from `mlir/Transforms/Passes.h`) — the stock canonicalizer. It gathers every registered op's canonicalization patterns + folders and runs the greedy driver to fixpoint. We wrote zero pass boilerplate ourselves.
- **No CSE pass yet.** The upstream tutorial adds `mlir::createCSEPass()` in Chapter 4 alongside the shape-inference work; this chapter's `toyc.cpp` runs only the canonicalizer. If you want a preview, `pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCSEPass());` is the one-liner.

Since the pipeline is only constructed under `if (enableOpt)`, running without `-opt` gives you the raw MLIRGen output — which is exactly how we produce the "before" dumps in §6.

---

## 5. Building

### 5.1 Superbuild layout

This repo builds all chapters as **one out-of-tree CMake superbuild** rooted at `/Users/roy/study/mlir/toy/`, against Homebrew LLVM/MLIR 20 — not with the in-tree `add_toy_chapter` machinery from `llvm-project/mlir/examples`. The top-level `/Users/roy/study/mlir/toy/CMakeLists.txt` runs the shared boilerplate exactly once and then pulls in every chapter:

```cmake
project(toy-tutorial)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${MLIR_INCLUDE_DIRS}
                    ${LLVM_INCLUDE_DIRS})

# Collect every chapter binary in build/bin/ instead of build/ChN/.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(Ch1)
# ... Ch2 through Ch6 ...
add_subdirectory(Ch7)
```

Configuration comes from `/Users/roy/study/mlir/toy/CMakePresets.json`: a single `default` preset selecting **Ninja**, **Release**, the Homebrew `llvm@20` compilers (`/opt/homebrew/opt/llvm@20/bin/clang` / `clang++` — matching the libraries avoids C++ ABI mismatches with Apple's system clang), and `MLIR_DIR`/`LLVM_DIR` pointing at `/opt/homebrew/opt/llvm@20/lib/cmake/{mlir,llvm}`. `cmake --preset default` is therefore the entire configure step; the build tree is `toy/build/`.

### 5.2 The dual-mode chapter CMakeLists

Each `ChN/CMakeLists.txt` is now **dual-mode**: its find-package/include boilerplate is wrapped in a guard that fires only when the chapter is configured directly, so under the superbuild the chapter just defines its targets against the already-configured parent. The real `/Users/roy/study/mlir/toy/Ch3/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Standalone-mode boilerplate.
# Runs only when this chapter is configured directly (cmake -S Ch3).
# In the superbuild (cmake -S toy/), ../CMakeLists.txt already did all this.
# ---------------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "macOS Deployment Target" FORCE)
  endif()
  project(toy-ch3)

  find_package(MLIR REQUIRED CONFIG)
  find_package(LLVM REQUIRED CONFIG)

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  include_directories(${MLIR_INCLUDE_DIRS}
                      ${LLVM_INCLUDE_DIRS})
endif()

# ---------------------------------------------------------------------------
# Chapter targets
# ---------------------------------------------------------------------------
include_directories(include)
add_subdirectory(include)

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCh3CombineIncGen)

add_executable(toyc-ch3
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/ToyCombine.cpp
  )

add_dependencies(toyc-ch3 ToyCh3OpsIncGen)
add_dependencies(toyc-ch3 ToyCh3CombineIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

target_link_libraries(toyc-ch3
  PRIVATE
    MLIR                      # libMLIR.dylib (all dialects, passes, conversions)
    LLVM                      # libLLVM.dylib (all targets, all components)
    )
```

The Chapter-3-specific additions relative to Chapter 2:

1. **The rewriter TableGen step:**

   ```cmake
   set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
   mlir_tablegen(ToyCombine.inc -gen-rewriters)
   add_public_tablegen_target(ToyCh3CombineIncGen)
   ```

   `LLVM_TARGET_DEFINITIONS` names the `.td` input; `mlir_tablegen(... -gen-rewriters)` invokes `mlir-tblgen --gen-rewriters` to emit `ToyCombine.inc` into the chapter's binary directory — `toy/build/Ch3/ToyCombine.inc` in the superbuild (or `Ch3/build/ToyCombine.inc` when the chapter is configured standalone); `add_public_tablegen_target` wraps the generated file in a named target so the executable can depend on it and the file is guaranteed to exist before `ToyCombine.cpp` compiles (`add_dependencies(toyc-ch3 ToyCh3CombineIncGen)`).

   This sits alongside the Chapter-2 op generation in `include/toy/CMakeLists.txt` (`-gen-op-decls`, `-gen-op-defs`, `-gen-dialect-decls`, `-gen-dialect-defs` → target `ToyCh3OpsIncGen`).

2. **New source file** `mlir/ToyCombine.cpp` in the executable.

3. **Include path for generated files:** `include_directories(${CMAKE_CURRENT_BINARY_DIR})` is what lets `#include "ToyCombine.inc"` resolve, since the `.inc` lands at the root of the chapter's binary directory (`build/Ch3/`). `CMAKE_CURRENT_BINARY_DIR` points to the right place in both superbuild and standalone modes — that's part of what makes the dual-mode file work unchanged.

Linking is intentionally coarse: the monolithic Homebrew `libMLIR.dylib` / `libLLVM.dylib` shared libraries, which already contain the canonicalizer pass, greedy driver, and everything else. In-tree builds would instead list fine-grained components (`MLIRPass`, `MLIRTransforms`, `MLIRSupport`, ...).

### 5.3 build.sh and everyday workflow

The per-chapter `ChN/build.sh` scripts are gone. The shared `/Users/roy/study/mlir/toy/build.sh` wraps the preset in an **incremental** workflow: it configures only when `build/CMakeCache.txt` is missing (afterwards Ninja re-runs CMake automatically whenever a `CMakeLists.txt` changes), and only wipes the tree when you ask with `--fresh`:

```bash
cd /Users/roy/study/mlir/toy
./build.sh ch3          # build only toyc-ch3
./build.sh              # build all chapters (same as ./build.sh all)
./build.sh ch3 --fresh  # rm -rf build/ first, then reconfigure + build toyc-ch3
# Binary: ./build/bin/toyc-ch3
```

Under the hood a chapter build is just `cmake --build --preset default --target toyc-ch3`, so `ninja -C build toyc-ch3` works equally well.

**Standalone mode still works.** Thanks to the `CMAKE_SOURCE_DIR` guard in §5.2, a single chapter can be configured on its own — you just have to supply what the preset would have (the preset lives at the superbuild root, so it doesn't apply here):

```bash
cd /Users/roy/study/mlir/toy
cmake -S Ch3 -B Ch3/build -G Ninja \
      -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang \
      -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++ \
      -DMLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir \
      -DLLVM_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/llvm
cmake --build Ch3/build
# Binary: Ch3/build/toyc-ch3  (run.sh falls back to this path if build/bin/ has no toyc-ch3)
```

---

## 6. Running and Testing

The shared `/Users/roy/study/mlir/toy/run.sh` runs both chapter test files with optimization enabled:

```bash
cd /Users/roy/study/mlir/toy
./run.sh ch3
# which executes (from toy/):
#   ./build/bin/toyc-ch3 ../test_Example/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt
#   ./build/bin/toyc-ch3 ../test_Example/Toy/Ch3/trivial_reshape.toy -emit=mlir -opt
```

Test inputs live in `/Users/roy/study/mlir/test_Example/Toy/Ch3/`. All outputs below are **real captured output** from this repo's `toyc-ch3` binary.

### 6.1 `transpose_transpose.toy`

Source (`test_Example/Toy/Ch3/transpose_transpose.toy`):

```
# User defined generic function that operates on unknown shaped arguments
def transpose_transpose(x) {
  return transpose(transpose(x));
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b = transpose_transpose(a);
  print(b);
}
```

**Without `-opt`** (`./build/bin/toyc-ch3 ../test_Example/Toy/Ch3/transpose_transpose.toy -emit=mlir`):

```mlir
module {
  toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
    %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
    toy.return %1 : tensor<*xf64>
  }
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
    %2 = toy.generic_call @transpose_transpose(%1) : (tensor<2x3xf64>) -> tensor<*xf64>
    toy.print %2 : tensor<*xf64>
    toy.return
  }
}
```

**With `-opt`:**

```mlir
module {
  toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
    toy.return %arg0 : tensor<*xf64>
  }
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.generic_call @transpose_transpose(%0) : (tensor<2x3xf64>) -> tensor<*xf64>
    toy.print %1 : tensor<*xf64>
    toy.return
  }
}
```

What fired, and why:

- **In `@transpose_transpose`:** `SimplifyRedundantTranspose` (C++ pattern) matched the outer `%1 = toy.transpose(%0)` because `%0`'s defining op is itself a `toy.transpose`. `rewriter.replaceOp` rewired `toy.return` to use `%arg0` directly. The now-unused inner `%0 = toy.transpose(%arg0)` was then erased by the canonicalizer's DCE — legal only because `TransposeOp` is `Pure`. Result: the whole double-transpose reduced to `toy.return %arg0`.
- **In `@main`:** `%1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>` was removed by the DRR pattern **`RedundantReshapeOptPattern`**: the `TypesAreIdentical` constraint holds (input and output are both `tensor<2x3xf64>` — the declared shape `<2,3>` matches the literal's natural shape), so `replaceWithValue` forwarded `%0` straight into the `generic_call`.
- **What did *not* happen:** the call to `@transpose_transpose` is still there. The canonicalizer performs local, intra-function rewrites; it does not inline or interprocedurally propagate. Chapter 4 (inlining + shape inference) is what finally lets `main` see through the call.

### 6.2 `trivial_reshape.toy`

Source (`test_Example/Toy/Ch3/trivial_reshape.toy`):

```
def main() {
  var a<2,1> = [1, 2];
  var b<2,1> = a;
  var c<2,1> = b;
  print(c);
}
```

Each `var x<2,1> = ...` declaration forces a reshape to `tensor<2x1xf64>`, so MLIRGen emits a chain of three reshapes off one constant.

**Without `-opt`:**

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
    %1 = toy.reshape(%0 : tensor<2xf64>) to tensor<2x1xf64>
    %2 = toy.reshape(%1 : tensor<2x1xf64>) to tensor<2x1xf64>
    %3 = toy.reshape(%2 : tensor<2x1xf64>) to tensor<2x1xf64>
    toy.print %3 : tensor<2x1xf64>
    toy.return
  }
}
```

**With `-opt`:**

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>
    toy.print %0 : tensor<2x1xf64>
    toy.return
  }
}
```

All three reshapes vanished *and* the constant changed shape (`dense<[1.0, 2.0]> : tensor<2xf64>` became `dense<[[1.0], [2.0]]> : tensor<2x1xf64>`). Here is how the patterns compose under the greedy driver (one plausible fixpoint trajectory — the driver may interleave differently, but converges to the same result):

1. **`ReshapeReshapeOptPattern`** collapses the chain: `%3 = reshape(%2 = reshape(%1))` → `reshape(%1)`, and again with the next pair, leaving a single `reshape(%0) : tensor<2xf64> → tensor<2x1xf64>`. (Alternatively/additionally, **`RedundantReshapeOptPattern`** can delete `%2` and `%3` outright since their input and output types are both `tensor<2x1xf64>` — both routes reach the same intermediate state.)
2. **`FoldConstantReshapeOptPattern`** matches the surviving `reshape(constant)`: the `NativeCodeCall` invokes `DenseElementsAttr::reshape(...)` to re-type the payload `[1.0, 2.0]` as `tensor<2x1xf64>`, and a new `toy.constant dense<[[1.0],[2.0]]>` replaces the reshape.
3. The original `tensor<2xf64>` constant is now unused; since `ConstantOp` is `Pure`, canonicalizer DCE deletes it.

Final IR: one constant, one print — no runtime reshaping work at all.

### 6.3 One-shot verification

```bash
cd /Users/roy/study/mlir/toy
./build.sh ch3      # configure (first time only) + incremental build → ./build/bin/toyc-ch3
./run.sh ch3        # both tests with -emit=mlir -opt

# Before/after comparison by hand:
./build/bin/toyc-ch3 ../test_Example/Toy/Ch3/transpose_transpose.toy -emit=mlir
./build/bin/toyc-ch3 ../test_Example/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt
./build/bin/toyc-ch3 ../test_Example/Toy/Ch3/trivial_reshape.toy -emit=mlir
./build/bin/toyc-ch3 ../test_Example/Toy/Ch3/trivial_reshape.toy -emit=mlir -opt
```

Debugging tip: because `main()` registers the pass-manager CL options and `dumpMLIR` calls `applyPassManagerCLOptions`, you can watch the pipeline work:

```bash
./build/bin/toyc-ch3 ../test_Example/Toy/Ch3/trivial_reshape.toy -emit=mlir -opt \
    --mlir-print-ir-before-all --mlir-print-ir-after-all
```

The `.toy` files also carry `# RUN:` / `# CHECK:` lines for LLVM `lit`/`FileCheck`-style testing; in this repo we exercise them directly via `run.sh` rather than through a lit harness.

---

## 7. Key Takeaways & Pitfalls

**Takeaways**

- **Optimize at the right abstraction level.** `transpose(transpose(x)) → x` is a two-op pattern match in the Toy dialect and a near-impossible loop-nest analysis in LLVM IR. This asymmetry is the whole point of building high-level dialects.
- **Two pattern authoring styles, one engine.** C++ `OpRewritePattern` for arbitrary logic; DRR for concise structural rewrites (with `Constraint`/`CPred` for conditions and `NativeCodeCall` for embedded C++). Both end up as `RewritePattern`s in the same `RewritePatternSet` consumed by the same greedy driver.
- **Canonicalization is the cheap distribution channel.** `let hasCanonicalizer = 1` in ODS + a `getCanonicalizationPatterns` definition means *every* user of `createCanonicalizerPass()` gets your patterns for free — no custom pass required.
- **Small patterns compose via fixpoint iteration.** Nobody wrote "fold a chain of three reshapes into a reshaped constant"; three independent one-step patterns plus DCE reached that result together.
- **Locations are preserved.** DRR-generated code fuses the source ops' locations into the replacement, keeping diagnostics meaningful after transformation.

**Pitfalls**

- **Patterns don't clean up after themselves — DCE does.** `SimplifyRedundantTranspose` leaves the inner transpose dead. Don't build erasure of *upstream* producers into your pattern (they may have other users!); instead make ops declare their effects and let the canonicalizer's DCE remove what becomes dead. Corollary: **forgetting `Pure` (or a proper `MemoryEffects` interface) silently blocks DCE** — you'll see dead ops persist with no error anywhere.
- **Forgetting `let hasCanonicalizer = 1`** means the generated op class has no `getCanonicalizationPatterns` declaration; your `.cpp` definition then fails to compile (or, if you register patterns some other way, the canonicalizer never asks your op for them).
- **Only mutate through the `PatternRewriter`.** Direct IR surgery inside `matchAndRewrite` breaks the driver's worklist tracking and can crash or miscompile. Also: decide match *before* mutating — a pattern that mutates then returns `failure()` is broken.
- **Rewrites must keep consumers legal.** Replacing a value changes the operand of its users (`toy.return`, `toy.generic_call`, ...). Here the permissive `tensor<*xf64>`/`F64Tensor` typing makes that safe; with stricter ops you must check result/operand type compatibility in the match (as `RedundantReshapeOptPattern` does via `TypesAreIdentical`).
- **Don't create infinite ping-pong.** The greedy driver iterates to fixpoint; a pair of patterns like `A→B` and `B→A` (or a pattern that "rewrites" an op to an identical op) never converges and hits the iteration limit. Each rewrite should strictly reduce some measure (op count, pattern-match opportunities).
- **Op ordering / traversal is not yours to control.** Patterns fire in benefit order per op, over a worklist in unspecified overall order. Never write a pattern whose correctness depends on another pattern having already run — each must be independently sound; only *convergence* composes them.
- **Canonicalization is local.** It didn't remove the `toy.generic_call` round-trip in `main` — interprocedural wins need the inliner and shape inference, which is exactly where [Chapter 4](4_generic_transformation_interfaces.md) picks up.
- **DRR `$arg` on a `ConstantOp` binds the attribute, not a value.** Because ODS declares `ConstantOp`'s input as an attribute, DRR captures a `DenseElementsAttr` — which is why `ReshapeConstant` can call `.reshape(...)` on it directly. Misreading bound-entity kinds ($value vs $attribute) is a classic DRR stumble.
- **LLVM version drift.** Against LLVM/MLIR 20: the trait is `Pure` (not `NoSideEffect`), casts are `::llvm::cast<ShapedType>(v)` (not `v.cast<ShapedType>()`), and `matchAndRewrite` returns `llvm::LogicalResult`. Older tutorial snippets found online may not compile as-is.

---

## Links

- Official doc: [MLIR Toy Tutorial Ch.3 — High-level Language-Specific Analysis and Transformation](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/)
- DRR reference: [Table-driven Declarative Rewrite Rules](https://mlir.llvm.org/docs/DeclarativeRewrites/) · [Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/) · [Operation Canonicalization](https://mlir.llvm.org/docs/Canonicalization/)
- Previous: [Chapter 2 — Emitting Basic MLIR](2_emitting_basic_mlir.md)
- Next: [Chapter 4 — Enabling Generic Transformation with Interfaces](4_generic_transformation_interfaces.md)
- Index: [README](README.md)
