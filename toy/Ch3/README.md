# Chapter 3: High-level Language-Specific Analysis and Transformation

> **Goal:** Use the high-level semantics preserved by the Toy dialect to implement optimizations that would be impossible (or very hard) after lowering — first with a hand-written C++ `RewritePattern`, then with declarative TableGen rewrite rules (DRR) — and hook them all into MLIR's canonicalization framework.
> Official docs: [Toy Tutorial Ch.3 — High-level Language-Specific Analysis and Transformation](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/)

---

## 1. Overview

Chapter 2 gave us a Toy dialect that faithfully models the source language: `toy.transpose`, `toy.reshape`, `toy.constant`, and friends. Now we get the payoff. Because the IR still *knows* it is dealing with a tensor transpose — not a soup of loads, stores, and index arithmetic — we can express language-level algebraic identities as trivial local rewrites. Analyses like these are usually done on the language AST (Clang, for example, has a heavy `TreeTransform` mechanism for C++ template instantiation); a dialect that mirrors the language lets MLIR do them on the IR instead.

In this chapter we write four such rewrites:

- `transpose(transpose(x)) → x` — hand-written C++ (`SimplifyRedundantTranspose` in `mlir/ToyCombine.cpp`, section 3)
- Three reshape optimizations — DRR (`mlir/ToyCombine.td`, section 4)

and turn them on with a new `-opt` driver flag that runs MLIR's stock canonicalizer (section 3.4). Section 5 builds the chapter and runs both test inputs.

### How this README maps to the upstream chapter

Every topic of the official [Ch-3](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/) is covered here, in the same order, with this repo's real code and output:

| Upstream section | Here |
|---|---|
| Introduction (local vs. global transformations, C++ vs. DRR patterns) | 1, 2.1 |
| Optimize Transpose using C++ style pattern-match and rewrite: the Toy example, its IR, and the C++ loops Clang can't clean up | 3.1 |
| … the `SimplifyRedundantTranspose` pattern | 3.2 |
| … `hasCanonicalizer = 1` and `getCanonicalizationPatterns` | 3.3 |
| … the `PassManager` in `toyc.cpp` | 3.4 |
| … the first `-opt` run, the leftover dead transpose, and the `Pure` trait | 3.5–3.6 |
| Optimize Reshapes using DRR: the `Pattern` class · `ReshapeReshapeOptPattern` | 4.1 |
| … `TypesAreIdentical` / `RedundantReshapeOptPattern` · `NativeCodeCall` / `FoldConstantReshapeOptPattern` | 4.2–4.3 |
| … the generated `ToyCombine.inc` | 4.4 |
| … `trivial_reshape.toy` before and after `-opt` | 4.5 (and end to end in 5.4) |
| Running `toyc-ch3 … -emit=mlir -opt` | 5 |

Beyond upstream: how the greedy driver runs (2.2), how the DRR patterns get registered (4.4), which pattern fired, read from the locations (5.5), IR printing, `FileCheck`, and the error path (5.6), and the build wiring (appendix).

### Where Chapter 3 lives in this repo

How this repo is organized — the out-of-tree CMake superbuild at `toy/`, the pinned Homebrew LLVM/MLIR 20 toolchain, and the `build.sh`/`run.sh` helpers — is documented once in the top-level [README](../README.md#repository-layout). This chapter also remains configurable as a standalone project (see section 5.1). Chapter 3 specifics:

| Item | Location / value |
|---|---|
| Chapter 3 code | `/Users/roy/study/mlir/toy/Ch3/` |
| Build | `cd toy && ./build.sh ch3` → binary at `./build/bin/toyc-ch3` |
| Run | `cd toy && ./run.sh ch3` → `transpose_transpose.toy` and `trivial_reshape.toy` with `-emit=mlir -opt` |
| Test inputs | `/Users/roy/study/mlir/test_Example/Toy/Ch3/` (`transpose_transpose.toy`, `trivial_reshape.toy`, plus the Ch1/Ch2 inputs `ast.toy`, `empty.toy`, `codegen.toy`, `scalar.toy`, `invalid.mlir`) |

What changes relative to Chapter 2:

```text
Ch3/
├── CMakeLists.txt      # + DRR TableGen rule (-gen-rewriters) and ToyCombine.cpp (appendix)
├── toyc.cpp            # + -opt flag, PassManager running the canonicalizer (section 3.4)
├── mlir/
│   ├── ToyCombine.cpp  # NEW: C++ pattern + getCanonicalizationPatterns hooks (sections 3.2, 4.4)
│   ├── ToyCombine.td   # NEW: three DRR reshape patterns (section 4)
│   ├── Dialect.cpp     # unchanged apart from reordering
│   └── MLIRGen.cpp     # unchanged apart from a comment
├── parser/AST.cpp      # unchanged from Chapter 1
└── include/toy/
    ├── Ops.td          # + Pure on add/mul/reshape/transpose, + hasCanonicalizer (sections 3.3, 3.6)
    └── ...             # AST.h, Lexer.h, Parser.h, Dialect.h, MLIRGen.h unchanged
```

---

## 2. Pattern Rewriting and Canonicalization

### 2.1 Local vs. global, C++ vs. DRR

The upstream chapter splits compiler transformations into two categories: **local** ones, which look at a small neighborhood of ops (an op and its operands' producers), and **global** ones, which need a view of a whole function or program (inlining, shape inference — Chapter 4). This chapter is about local pattern-match transformations that are trivial on Toy IR but hard in LLVM, and it uses MLIR's generic DAG rewriter to run them.

MLIR gives us two ways to write such a pattern:

| Approach | Mechanism | Best for |
|---|---|---|
| **Imperative C++** | Subclass `mlir::OpRewritePattern<Op>`, implement `matchAndRewrite` | Matches that need arbitrary C++ logic, side-effect reasoning, non-structural conditions |
| **Declarative (DRR)** | TableGen `Pat<...>` records, compiled by `mlir-tblgen -gen-rewriters` into C++ | Structural DAG-to-DAG rewrites; concise, less boilerplate, harder to get wrong |

DRR matches and builds ops by their ODS records (`ReshapeOp`, `ConstantOp`), so it only works for operations defined with ODS, as in Chapter 2.

Both feed the same engine: the **greedy pattern rewrite driver** that powers MLIR's **canonicalization pass**. Either way, a pattern ends up as a `RewritePattern` object in a `RewritePatternSet`; an op contributes its patterns to the canonicalizer through a static `getCanonicalizationPatterns` hook (section 3.3).

### 2.2 How the greedy driver actually runs

`createCanonicalizerPass()` wraps the **greedy pattern rewrite driver** (`applyPatternsGreedily`). Its algorithm, roughly:

1. Collect canonicalization patterns from all registered ops (via the `getCanonicalizationPatterns` hooks) plus built-in folding.
2. Put all ops on a worklist.
3. Pop an op; try to **fold** it; then try each applicable pattern in decreasing benefit order.
4. When a pattern fires, the rewriter notifies the driver, which re-enqueues affected ops (users of replaced values, producers of operands, etc.).
5. Trivially dead, side-effect-free ops are erased along the way.
6. Repeat until fixpoint (no pattern applies anywhere) or an iteration cap is hit.

The fixpoint iteration is what lets small local patterns compose into big cleanups: one `Reshape(Reshape(x))` collapse exposes another, which exposes a constant fold, and so on — exactly what `trivial_reshape.toy` shows (section 4.5). Step 3's benefit ordering is also observable: section 5.5 uses the source locations of the optimized IR to show which of two applicable patterns won.

---

## 3. Optimize Transpose using C++ Pattern-Match and Rewrite

### 3.1 The motivating example

Start with a simple pattern: eliminate a sequence of two transposes that cancel out, `transpose(transpose(X)) -> X`. Two runs set the scene: the Toy IR, where the pattern is visible, and the equivalent C++, where Clang can't find it. The build is in section 5.1.

#### In Toy IR the two transposes are adjacent ops

The generic function in this chapter's first test input is exactly that pattern:

***test_Example/Toy/Ch3/transpose_transpose.toy***
```
def transpose_transpose(x) {
  return transpose(transpose(x));
}
```

Emit its IR without `-opt`, so no pattern runs:

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/transpose_transpose.toy -emit=mlir 2>&1
```

Real output (abridged to `@transpose_transpose`; section 5.3 shows all of it):

```mlir
module {
  toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
    %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
    toy.return %1 : tensor<*xf64>
  }
  ...
}
```

- MLIRGen emitted one `toy.transpose` per source-level `transpose` call, so the "transpose of a transpose" is still spelled out: `%1`'s operand `%0` is produced by another `toy.transpose`. That operand-to-producer link is all the pattern of section 3.2 has to check.
- The types are unranked (`tensor<*xf64>`) because a generic function's argument shape is unknown here; the rewrite doesn't need shapes.

#### In C++ the same computation is two loop nests

The match is trivial on Toy IR but hard for LLVM. The upstream tutorial writes the same computation in C++, with the naive transpose as loops, and says Clang can't optimize away the temporary array. That still holds for this repo's toolchain. The snippet isn't a repo file, so the command passes it to Homebrew clang 20.1.8 on stdin (`-x c++ -`), compiles at `-O3` to LLVM IR, and keeps only the allocas, loop back-edges, and the `sink` call:

```bash
cd /Users/roy/study/mlir/toy/Ch3
/opt/homebrew/opt/llvm@20/bin/clang++ -O3 -S -emit-llvm -x c++ - -o - <<'EOF' \
  | grep -E 'alloca|llvm\.loop !|@_Z4sinkPv\(ptr'
#define N 100
#define M 100

void sink(void *);
void double_transpose(int A[N][M]) {
  int B[M][N];
  for(int i = 0; i < N; ++i) {
    for(int j = 0; j < M; ++j) {
       B[j][i] = A[i][j];
    }
  }
  for(int i = 0; i < N; ++i) {
    for(int j = 0; j < M; ++j) {
       A[i][j] = B[j][i];
    }
  }
  sink(A);
}
EOF
```

Real output:

```text
  %2 = alloca [100 x [100 x i32]], align 4
  br i1 %23, label %24, label %5, !llvm.loop !10
  br i1 %26, label %27, label %3, !llvm.loop !14
  br i1 %47, label %49, label %29, !llvm.loop !15
  tail call void @_Z4sinkPv(ptr noundef nonnull %0)
  br i1 %51, label %48, label %27, !llvm.loop !16
declare void @_Z4sinkPv(ptr noundef) local_unnamed_addr #2
```

- `alloca [100 x [100 x i32]]` is the 40000-byte temporary `B`, still allocated at `-O3`.
- Four `!llvm.loop` back-edges remain: both loop nests (two loops each) survive, and `A` is still passed to `sink` afterwards.

By the time this reaches LLVM IR, the "transpose-ness" is gone: LLVM sees two nested loops shuffling memory through a temporary buffer, and proving that they cancel out takes loop and memory-dependence reasoning that the optimizer doesn't do. In the Toy dialect, the same fact is a one-line pattern match: *"a transpose whose operand is produced by another transpose can be replaced by the inner transpose's operand."*

This is the core argument for multi-level IR: **do each optimization at the abstraction level where it is cheap to express and trivially correct.**

### 3.2 The `SimplifyRedundantTranspose` pattern

A tree-like match in the IR that is replaced with a different set of ops is written in C++ as a `RewritePattern` and plugged into MLIR's canonicalizer. The pattern and its registration hook in `ToyCombine.cpp`:

***mlir/ToyCombine.cpp***
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

/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}
```

The pattern is the upstream one; only comment wording differs. Let's unpack every moving part.

**`OpRewritePattern<TransposeOp>`** — a convenience base class for patterns rooted at one specific op type. The greedy driver will only call this pattern on `toy.transpose` operations, and `matchAndRewrite` receives the operation already cast to the typed `TransposeOp` wrapper. (The more general base, `RewritePattern`, matches on an op *name* string and hands you a raw `Operation *` — that's what DRR-generated patterns use, as we'll see in section 4.4.)

**`benefit`** — a heuristic cost passed to the base constructor (`/*benefit=*/1` here). When multiple patterns can fire at the same rooted op, the driver tries higher-benefit patterns first. With a single pattern per root the value barely matters; it becomes relevant when you have several patterns for the same op — as `toy.reshape` does, where section 5.5 shows the benefit-2 DRR pattern beating the benefit-1 one.

**`matchAndRewrite`** — the fused match + rewrite hook. The contract is strict:

- Return `failure()` **before mutating anything** if the pattern does not apply. Here: walk up from the transpose's operand via `Value::getDefiningOp<TransposeOp>()`; if the operand is a block argument or produced by some other op, `transposeInputOp` is null and we bail out.
- If it does apply, perform **all** IR mutation through the supplied `PatternRewriter` (`replaceOp`, `eraseOp`, `create<...>`, `modifyOpInPlace`, ...). The rewriter is how the driver tracks what changed, keeps its worklist up to date, and (in dialect-conversion contexts) supports rollback. Mutating IR behind the rewriter's back is undefined behavior for the driver.
- Return `success()` after rewriting.

The actual rewrite is one line: `rewriter.replaceOp(op, {transposeInputOp.getOperand()})` replaces every use of the *outer* transpose's result with the *inner* transpose's input — i.e. the original `x`.

**Registration.** A pattern is inert until something runs it. Rather than writing a bespoke pass, the last function in the block registers it as a **canonicalization pattern** on `TransposeOp`. The canonicalization pass applies the transformations defined by operations in a greedy, iterative manner (section 2.2); `getCanonicalizationPatterns` is the static hook it calls for **every op registered in the context** when it builds its pattern set. But the hook is not declared by default — you must opt in from ODS.

### 3.3 `hasCanonicalizer = 1` in ODS

Both ops that own patterns set the flag (abridged):

***include/toy/Ops.td***
```tablegen
def ReshapeOp : Toy_Op<"reshape", [Pure]> {
  ...
  // Enable registering canonicalization patterns with this operation.
  let hasCanonicalizer = 1;
}
...
def TransposeOp : Toy_Op<"transpose", [Pure]> {
  ...
  // Enable registering canonicalization patterns with this operation.
  let hasCanonicalizer = 1;
  ...
}
```

`let hasCanonicalizer = 1;` makes `mlir-tblgen -gen-op-decls` emit the declaration into the generated op class. In a standalone build it is in `build/include/toy/Ops.h.inc` (`toy/build/Ch3/include/toy/Ops.h.inc` under the superbuild), once in `class ReshapeOp` and once in `class TransposeOp`:

***build/include/toy/Ops.h.inc***
```cpp
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
```

We then supply the *definition* by hand in `ToyCombine.cpp` (section 3.2). Forgetting the flag gives you a very characteristic error: your definition in the `.cpp` fails to compile because the member function was never declared on the generated op class.

### 3.4 Adding an optimization pipeline to `toyc.cpp`

Patterns run inside a pass; passes run inside a `PassManager`, much as in LLVM. The upstream text shows just two lines:

```cpp
  mlir::PassManager pm(module->getName());
  pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());
```

In this repo the driver changes are a new `-opt` flag and a `dumpMLIR()` that builds and runs that pipeline between loading the module and dumping it (abridged):

***toyc.cpp***
```cpp
static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));
...
int dumpMLIR() {
  mlir::MLIRContext context;
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::toy::ToyDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  if (int error = loadMLIR(sourceMgr, context, module))
    return error;

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

  module->dump();
  return 0;
}
```

Key details:

- **`loadMLIR(...)`** — Chapter 2's `dumpMLIR()` did the Toy-vs-`.mlir` dispatch and dumped immediately. Here that dispatch moved into a helper, `loadMLIR()`, which fills in `module` from either MLIRGen (`.toy`) or the MLIR parser (`.mlir` / `-x mlir`) and returns an error code (6 for a Toy parse error, 1 for an MLIRGen failure, 3 for an MLIR parse failure). That way the pipeline runs on either kind of input. The `SourceMgrDiagnosticHandler` renders diagnostics with file:line:col and a caret (the error path in section 5.6).
- **`mlir::PassManager pm(module.get()->getName())`** — the pass manager is anchored on the top-level op type (`builtin.module` here). Upstream's `module->getName()` is the same call through `OwningOpRef::operator->`.
- **`applyPassManagerCLOptions(pm)`** — not in the upstream snippet. It transfers generic pass-manager flags from the command line onto this PM instance: `--mlir-print-ir-before-all`, `--mlir-print-ir-after-all`, `--mlir-pass-statistics`, crash reproducer options, etc. For these flags to exist at all, `main()` must call the matching registration function — and it does (below).
- **`pm.addNestedPass<mlir::toy::FuncOp>(...)`** — schedules the canonicalizer to run *nested on each `toy.func`* rather than on the whole module. This is both more precise and enables parallel execution across functions (each function is an isolated-from-above region). It is also why the `--mlir-print-ir-*` dumps in section 5.6 show one `toy.func` at a time.
- **`mlir::createCanonicalizerPass()`** (from `mlir/Transforms/Passes.h`) — the stock canonicalizer. It gathers every registered op's canonicalization patterns + folders and runs the greedy driver to fixpoint. We wrote zero pass boilerplate ourselves.
- **No CSE pass yet.** The upstream tutorial adds `mlir::createCSEPass()` in Chapter 4 alongside the shape-inference work; this chapter's `toyc.cpp` runs only the canonicalizer.
- A pipeline failure returns 4.

Since the pipeline is only constructed under `if (enableOpt)`, running without `-opt` gives you the raw MLIRGen output — which is exactly how the "before" dumps in sections 3.1 and 4.5 (and 5.3, 5.4) are produced.

`main()` gains one registration call, `registerPassManagerCLOptions()`, next to the two from Chapter 2:

***toyc.cpp***
```cpp
int main(int argc, char **argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  ...
}
```

### 3.5 The first run: a dead transpose is left behind

Upstream now runs `toyc-ch3 test/Examples/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt` and shows that the pattern fired but one transpose survived. That intermediate state is **not** what this repo's code produces: upstream's own `Ops.td` (and this one) already declares `TransposeOp` with `[Pure]`, so the shipped binary goes straight to the final result (section 3.6).

To see upstream's intermediate output for real, take the trait away in a scratch build. That needs a rebuild, so here is the recipe rather than a command: copy `CMakeLists.txt`, `include/`, `mlir/`, `parser/`, and `toyc.cpp` from `Ch3/` into an empty directory, drop `[Pure]` from `def TransposeOp` there with

`sed -i '' 's/^def TransposeOp : Toy_Op<"transpose", \[Pure\]>/def TransposeOp : Toy_Op<"transpose">/' include/toy/Ops.td`

and configure and build the copy as in section 5.1. Its `toyc-ch3`, run on `../../test_Example/Toy/Ch3/transpose_transpose.toy` with `-emit=mlir -opt` from `Ch3/`, shows upstream's intermediate state.

Real output of the scratch build (exit status 0; abridged to `@transpose_transpose`, since `@main` is the same as the shipped binary's in section 5.3):

```mlir
module {
  toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
    toy.return %arg0 : tensor<*xf64>
  }
  ...
}
```

The pattern worked: `toy.return` now returns the function argument directly, bypassing both transposes. But `SimplifyRedundantTranspose` only replaced the *outer* transpose; it never erased the *inner* one, and `%0` is now dead.

Why doesn't the canonicalizer just delete `%0`? It does know how to clean up dead operations, but it is **conservative about side effects**: unless an op says otherwise, MLIR assumes it might print, write memory, or trap, and deleting it because its result is unused would be wrong. The canonicalizer only erases dead ops it can *prove* are side-effect free.

### 3.6 The `Pure` trait and dead code elimination

The fix is to declare it. The `def TransposeOp : Toy_Op<"transpose", [Pure]>` line in section 3.3 carries the **`Pure`** trait, and so does `ReshapeOp`. Compared with Chapter 2, this chapter's `Ops.td` adds `Pure` to `AddOp`, `MulOp`, `ReshapeOp`, and `TransposeOp`; `ConstantOp` and `ReturnOp` already had it. `GenericCallOp` and `PrintOp` stay without it — a call may have arbitrary effects and printing is the point of `print`.

`Pure` means "no side effects and always speculatable" — the strongest promise. In MLIR 20's `mlir/Interfaces/SideEffectInterfaces.td` it is `def Pure : TraitList<[AlwaysSpeculatable, NoMemoryEffect]>;`, and the generated class shows the result: `TransposeOp` picks up the speculation traits and `MemoryEffectOpInterface` (whose `getEffects` reports no effects):

***build/include/toy/Ops.h.inc***
```cpp
class TransposeOp : public ::mlir::Op<TransposeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
```

In the scratch build of section 3.5 the same line ends at `::mlir::OpTrait::OpInvariants>`. With the trait in place, the canonicalizer's built-in DCE sweeps the dead inner transpose away. Rerun upstream's command with the shipped binary:

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt 2>&1
```

Real output (abridged to `@transpose_transpose`; section 5.3 shows all of it):

```mlir
module {
  toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
    toy.return %arg0 : tensor<*xf64>
  }
  ...
}
```

- The function collapses to a return of its argument. No `transpose` operation is left — the code is optimal.
- The only difference from the scratch build of section 3.5 is the `[Pure]` on `def TransposeOp`: the pattern did the same thing in both builds, and the trait is what let DCE erase the dead `%0`.

> **Note (older LLVM versions):** the trait used to be spelled `NoSideEffect`; it was renamed to `Pure` (around LLVM 16), and MLIR 20's headers no longer define `NoSideEffect` at all. The upstream Ch-3 text already uses `Pure`.

**Why does `ReturnOp` (and every other op in the function) matter here?** Two related reasons:

1. **The pattern rewires uses, so the consumer must accept the new value.** `toy.return %arg0` must be valid IR. Because Toy ops are permissive about shapes at this stage (`tensor<*xf64>` unranked types, `F64Tensor` operands), replacing `%1` with `%arg0` verifies cleanly. If `ReturnOp` demanded an exact ranked type, the rewrite would produce verifier-invalid IR — a pattern must never do that.
2. **The canonicalizer processes *every* op in the region, not just transposes.** Every op it visits must be registered and verifiable; that's why Chapter 2 registered the whole dialect up front. `ReturnOp` itself is declared `[Pure, HasParent<"FuncOp">, Terminator]` in `Ops.td` — `Terminator` keeps DCE from ever considering it dead-code-removable in the "unused result" sense (it has no results), and `HasParent` keeps verification tight.

---

## 4. Optimize Reshapes using DRR

Writing `matchAndRewrite` by hand is flexible but verbose — the transpose pattern took ~30 lines of C++ (with comments) for a one-line idea. For purely structural DAG-to-DAG rewrites, MLIR offers **Declarative Rewrite Rules**: an operation-DAG-based declarative rewriter with a table-based syntax. You write TableGen `Pattern`/`Pat` records, and `mlir-tblgen -gen-rewriters` generates the `RewritePattern` subclasses for you. This chapter uses it for three optimizations of `toy.reshape`.

### 4.1 The `Pattern` class

`mlir/ToyCombine.td` starts with the necessary includes — `PatternBase.td` for the DRR infrastructure and our own `Ops.td` so the op records (`ReshapeOp`, `ConstantOp`) are visible — followed by a comment quoting the general `Pattern` record shape:

***mlir/ToyCombine.td***
```tablegen
include "mlir/IR/PatternBase.td"
include "toy/Ops.td"

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
//     list<dag> supplementalPatterns = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;
```

The upstream text shows this class with only four parameters (`sourcePattern`, `resultPatterns`, `additionalConstraints`, `benefitsAdded`). MLIR 20 has a fifth one before the benefit, as the repo's comment says. The real declaration is in the installed `PatternBase.td`:

```bash
cd /Users/roy/study/mlir/toy/Ch3
grep -A2 '^class Pattern<' /opt/homebrew/opt/llvm@20/include/mlir/IR/PatternBase.td
```

Real output:

```tablegen
class Pattern<dag source, list<dag> results, list<dag> preds = [],
  list<dag> supplemental_results = [],
  dag benefitAdded = (addBenefit 0)> {
```

The parameter names differ from the comment, but the positions match: source DAG, result DAGs, multi-entity constraints, supplemental patterns (run after the result patterns, not used for replacement), and a benefit delta added to the default benefit (the number of ops in the source pattern). None of this chapter's patterns set the last two. `Pat<src, result>` is the common single-result shorthand for `Pattern<src, [result]>`. (`include "toy/Ops.td"` resolves because the chapter's `include/` directory is on `mlir-tblgen`'s include path — see the appendix.)

The three patterns follow upstream's order: basic (below), constraints (section 4.2), then `NativeCodeCall` (section 4.3). In `ToyCombine.td` itself, the `NativeCodeCall` pattern comes second.

**Basic pattern: `Reshape(Reshape(x)) = Reshape(x)`.** A redundant-reshape optimization like `SimplifyRedundantTranspose`, expressed in DRR:

***mlir/ToyCombine.td***
```tablegen
// Reshape(Reshape(x)) = Reshape(x)
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;
```

Read the source DAG inside-out: match a `ReshapeOp` whose operand is produced by another `ReshapeOp`, binding the *inner* reshape's operand to `$arg`. The result DAG builds a single new `ReshapeOp` directly on `$arg` and replaces the matched root with it. In DRR, the result op's type is taken from the root op being replaced, so the new reshape produces the outer reshape's type — the intermediate shape is provably irrelevant since only element *data* flows through a reshape.

That's the same kind of two-op match as the transpose pattern, in two lines instead of thirty.

### 4.2 Pattern with constraints: redundant reshape

DRR can also add argument constraints, for transformations that depend on properties of the arguments and results. The second pattern only applies conditionally — a reshape whose input and output types are already identical is a no-op:

***mlir/ToyCombine.td***
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

Its source DAG has only one op, so it gets benefit 1 (section 4.4) and loses to the two-op patterns whenever both match; with this repo's two tests it never ends up firing (section 5.5). It still matters for reshapes whose operand is neither a constant nor another reshape — e.g. a same-type reshape of a function argument.

### 4.3 Pattern with `NativeCodeCall`: `Reshape(Constant(x)) = x'`

Some optimizations need extra transformations on the op's arguments. `NativeCodeCall` provides them by calling into a C++ helper function or embedding inline C++; its `$0, $1, ...` placeholders are substituted with bound entities. `FoldConstantReshape` reshapes a constant in place and eliminates the reshape:

***mlir/ToyCombine.td***
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
- `ReshapeConstant` expands to `arg.reshape(::llvm::cast<ShapedType>(res.getType()))` — `DenseElementsAttr::reshape` re-wraps the same raw data with the target shaped type.
- Result: build a fresh `ConstantOp` whose `value` attribute is the reshaped constant, and replace the reshape with it.

Net effect: a reshape applied to a compile-time constant is folded away entirely — the constant is simply *materialized in the target shape*.

#### Upstream's cast is deprecated in MLIR 20

The upstream text writes the helper as `NativeCodeCall<"$0.reshape(($1.getType()).cast<ShapedType>())">`, the old member-function cast. In MLIR 20, `mlir/IR/Types.h` marks `Type::cast` as `[[deprecated("Use mlir::cast<U>() instead")]]`. To see what upstream's string does, the command below swaps it into `ToyCombine.td` with `sed`, runs `mlir-tblgen -gen-rewriters` on the result, splices the generated code into `mlir/ToyCombine.cpp` in place of its `#include "ToyCombine.inc"` line, and syntax-checks that with clang — all through pipes, without touching the repo or the build tree:

```bash
cd /Users/roy/study/mlir/toy/Ch3
sed 's/::llvm::cast<ShapedType>(\$1.getType())/($1.getType()).cast<ShapedType>()/' mlir/ToyCombine.td \
  | /opt/homebrew/opt/llvm@20/bin/mlir-tblgen -gen-rewriters -I include -I /opt/homebrew/opt/llvm@20/include \
  | sed -e '/#include "ToyCombine.inc"/{r /dev/stdin' -e 'd;}' mlir/ToyCombine.cpp \
  | /opt/homebrew/opt/llvm@20/bin/clang++ -std=c++17 -fsyntax-only -x c++ - \
      -I include -I build/include -I /opt/homebrew/opt/llvm@20/include 2>&1
```

Real output (exit status 0):

```text
<stdin>:77:80: warning: 'cast' is deprecated: Use mlir::cast<U>() instead [-Wdeprecated-declarations]
   77 |     auto nativeVar_0 = arg.reshape(((*res.getODSResults(0).begin()).getType()).cast<ShapedType>()); (void)nativeVar_0;
      |                                                                                ^
/opt/homebrew/opt/llvm@20/include/mlir/IR/Types.h:340:9: note: 'cast' has been explicitly marked deprecated here
  340 | U Type::cast() const {
      |         ^
<stdin>:77:80: warning: 'cast<mlir::ShapedType>' is deprecated: Use mlir::cast<U>() instead [-Wdeprecated-declarations]
   77 |     auto nativeVar_0 = arg.reshape(((*res.getODSResults(0).begin()).getType()).cast<ShapedType>()); (void)nativeVar_0;
      |                                                                                ^
/opt/homebrew/opt/llvm@20/include/mlir/IR/Types.h:112:5: note: 'cast<mlir::ShapedType>' has been explicitly marked deprecated here
  112 |   [[deprecated("Use mlir::cast<U>() instead")]]
      |     ^
2 warnings generated.
```

- Upstream's form still builds (only warnings, exit status 0), so it isn't broken yet, just outdated. `mlir-tblgen` copies the `NativeCodeCall` string verbatim into the generated `nativeVar_0` line, which is why the warning points at generated code.
- The same pipeline with `cat mlir/ToyCombine.td` in place of the first `sed` (i.e. the repo's `::llvm::cast<ShapedType>(...)`) prints nothing: the free-function cast is the current form.

### 4.4 What TableGen generates

The generated C++ for each DRR pattern goes into `ToyCombine.inc`. Upstream places it under `path/to/BUILD/tools/mlir/examples/toy/Ch3/ToyCombine.inc`, an in-tree LLVM build path. Here the build runs `mlir-tblgen -gen-rewriters` over `ToyCombine.td` and writes `build/ToyCombine.inc` in a standalone build (`toy/build/Ch3/ToyCombine.inc` in the superbuild). It contains one `RewritePattern` subclass per `Pat` record. Here is the generated pattern for `FoldConstantReshapeOptPattern` (abridged: the failure diagnostics after the first and the attribute/type plumbing of the rewrite are elided):

***build/ToyCombine.inc***
```cpp
struct FoldConstantReshapeOptPattern : public ::mlir::RewritePattern {
  FoldConstantReshapeOptPattern(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("toy.reshape", 2, context, {"toy.constant"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::toy::ReshapeOp res;
    ::mlir::DenseElementsAttr arg;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::toy::ReshapeOp>(op0); (void)castedOp0;
    res = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::toy::ConstantOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        ...
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::DenseElementsAttr>("value");(void)tblgen_attr;
        if (!(tblgen_attr)){
          ...
        }
        arg = tblgen_attr;
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = arg.reshape(::llvm::cast<ShapedType>((*res.getODSResults(0).begin()).getType())); (void)nativeVar_0;
    ::mlir::toy::ConstantOp tblgen_ConstantOp_1;
    {
      ...
      tblgen_ConstantOp_1 = rewriter.create<::mlir::toy::ConstantOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }
    ...
    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};
```

The file's size and all three constructors, from the build tree:

```bash
cd /Users/roy/study/mlir/toy/Ch3
wc -l build/ToyCombine.inc
grep -B2 'RewritePattern("toy' build/ToyCombine.inc
```

Real output:

```text
     176 build/ToyCombine.inc
struct FoldConstantReshapeOptPattern : public ::mlir::RewritePattern {
  FoldConstantReshapeOptPattern(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("toy.reshape", 2, context, {"toy.constant"}) {}
--
struct RedundantReshapeOptPattern : public ::mlir::RewritePattern {
  RedundantReshapeOptPattern(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("toy.reshape", 1, context, {}) {}
--
struct ReshapeReshapeOptPattern : public ::mlir::RewritePattern {
  ReshapeReshapeOptPattern(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("toy.reshape", 2, context, {"toy.reshape"}) {}
```

That is 176 lines of C++ from about fifteen lines of TableGen definitions. Things worth noticing in the generated code:

- Each constructor passes the root op name `"toy.reshape"`, a **benefit**, and the *generated op names* metadata (`{"toy.constant"}`, `{}`, `{"toy.reshape"}`). The benefit is auto-computed as the source-pattern node count: the two-op matches `FoldConstantReshapeOptPattern` and `ReshapeReshapeOptPattern` get **2**, the one-op `RedundantReshapeOptPattern` gets **1**. A 2-op match is "worth more", so deeper patterns are tried first.
- The match phase is exactly the null-check ladder you would write by hand — `getDefiningOp`, `dyn_cast`, attribute presence checks — but each failure calls `notifyMatchFailure` with a human-readable diagnostic (visible under `--debug` in a debug build of MLIR).
- The `NativeCodeCall` string appears verbatim, with `$0`→`arg` and `$1`→`res`'s result substituted (`nativeVar_0`).
- The new constant's location is a **fused location** of both matched ops (`odsLoc`), preserving traceability — this is what section 5.5 reads back.
- At the bottom, TableGen also emits `void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(::mlir::RewritePatternSet &patterns)`, which registers all three patterns at once — an alternative registration entry point we don't use here.

**Hooking the generated patterns in.** `ToyCombine.cpp` includes the generated file inside an anonymous namespace and registers the three classes as `ReshapeOp` canonicalization patterns (abridged):

***mlir/ToyCombine.cpp***
```cpp
namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "ToyCombine.inc"
} // namespace
...
/// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by the Canonicalization framework.
void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
              FoldConstantReshapeOptPattern>(context);
}
```

From the canonicalizer's point of view there is no difference between hand-written and DRR-generated patterns — they are all `RewritePattern` instances in one `RewritePatternSet`. The order in `results.add<...>` does not decide which pattern wins; benefit does (section 2.2).

### 4.5 The reshape patterns on `trivial_reshape.toy`

Upstream demonstrates the three patterns on `trivial_reshape.toy` (abridged: its `# RUN`/`# CHECK` lines are left out; section 5.4 shows the whole file):

***test_Example/Toy/Ch3/trivial_reshape.toy***
```
def main() {
  var a<2,1> = [1, 2];
  var b<2,1> = a;
  var c<2,1> = b;
  print(c);
}
```

Each `var x<2,1> = ...` declaration forces a reshape to `tensor<2x1xf64>`. Without `-opt`, MLIRGen emits a chain of three reshapes off one constant:

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/trivial_reshape.toy -emit=mlir 2>&1
```

Real output:

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

With `-opt`, only a reshaped constant is left:

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/trivial_reshape.toy -emit=mlir -opt 2>&1
```

Real output:

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>
    toy.print %0 : tensor<2x1xf64>
    toy.return
  }
}
```

No reshape operations remain after canonicalization, *and* the constant changed shape (`dense<[1.0, 2.0]> : tensor<2xf64>` became `dense<[[1.0], [2.0]]> : tensor<2x1xf64>`). Here is how the patterns of sections 4.1–4.3 compose under the greedy driver (section 2.2):

1. **`%2` and `%3`** are each a reshape of a reshape *and* have identical input/output types, so both `ReshapeReshapeOptPattern` (benefit 2) and `RedundantReshapeOptPattern` (benefit 1) match them; the driver tries the higher-benefit `ReshapeReshapeOptPattern` first, which collapses each pair into a single reshape on the inner reshape's operand.
2. **`FoldConstantReshapeOptPattern`** matches any `reshape(constant)`: the `NativeCodeCall` invokes `DenseElementsAttr::reshape(...)` to re-type the payload `[1.0, 2.0]` as `tensor<2x1xf64>`, and a new `toy.constant dense<[[1.0],[2.0]]>` replaces the reshape.
3. Whatever is left unused — the original `tensor<2xf64>` constant and any intermediate reshapes — is `Pure`, so canonicalizer DCE deletes it.

The exact interleaving of steps 1 and 2 depends on worklist order, but it converges to the same result, and section 5.5 shows from the locations that `RedundantReshapeOptPattern` never fired. Final IR: one constant, one print — no runtime reshaping work at all. More on the declarative rewrite method is in the DRR reference under Links.

These are core transformations through always-available hooks (`getCanonicalizationPatterns`). Chapter 4 moves to generic solutions that scale better, through interfaces.

---

## 5. Build and Run

The superbuild (`toy/build.sh`, `toy/run.sh`, `CMakePresets.json`) is documented once in the top-level [README](../README.md#the-build-system). This section builds and runs Chapter 3 on its own, from the chapter directory. What the chapter's `CMakeLists.txt` adds — a second TableGen rule for the DRR patterns — is in the [appendix](#appendix-what-chapter-3-adds-to-the-build).

### 5.1 Building

```bash
cd /Users/roy/study/mlir/toy/Ch3
cmake -S . -B build -G Ninja
cmake --build build          # → ./build/toyc-ch3
```

No preset applies at the chapter level, yet no toolchain flags are needed, because the shell environment already points at Homebrew LLVM 20:

- `CXX=/opt/homebrew/opt/llvm@20/bin/clang++` (and `CC`) selects the compiler.
- `/opt/homebrew/opt/llvm@20/bin` is on `PATH`, and `find_package` also searches the prefix above each `PATH` entry, so it finds `/opt/homebrew/opt/llvm@20/lib/cmake/{mlir,llvm}` by itself.

In a shell without that setup, pass them explicitly: `-DMLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++`.

Before compiling, `mlir-tblgen` now processes two `.td` files: `include/toy/Ops.td` (as in Chapter 2, into `build/include/toy/*.inc`) and, with `-gen-rewriters`, `mlir/ToyCombine.td`, producing `build/ToyCombine.inc` (section 4.4). A standalone build puts the binary directly in `build/`; the superbuild's `toy/build/bin/toyc-ch3` behaves identically.

### 5.2 Running

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/transpose_transpose.toy -emit=mlir 2>&1
./build/toyc-ch3 ../../test_Example/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt 2>&1
```

- The positional argument is the input file (default `-`, stdin).
- `-emit=mlir` — dump the module after MLIRGen (`-emit=ast` still works).
- `-opt` — **new in this chapter**: run the canonicalizer pipeline before dumping (section 3.4). Without it you get the raw MLIRGen output, which is how the "before" dumps below are produced.
- `-mlir-print-debuginfo` — print `loc(...)` on every op; used in section 5.5 to see which patterns fired.
- `-x mlir` (or an input ending in `.mlir`) — parse the input as MLIR instead of Toy source; `-opt` works on either path.
- Generic pass-manager flags such as `--mlir-print-ir-before-all` / `--mlir-print-ir-after-all` are also accepted, because `main()` registers them and `dumpMLIR()` applies them (section 3.4; used in section 5.6).
- `2>&1` — `module->dump()` writes to **stderr**.

### 5.3 `transpose_transpose.toy`

***test_Example/Toy/Ch3/transpose_transpose.toy***
```
# RUN: toyc-ch3 %s -emit=mlir -opt 2>&1 | FileCheck %s

# User defined generic function that operates on unknown shaped arguments
def transpose_transpose(x) {
  return transpose(transpose(x));
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b = transpose_transpose(a);
  print(b);
}

# CHECK-LABEL: toy.func @transpose_transpose(
# CHECK-SAME:                                [[VAL_0:%.*]]: tensor<*xf64>) -> tensor<*xf64>
# CHECK-NEXT:    toy.return [[VAL_0]] : tensor<*xf64>

# CHECK-LABEL: toy.func @main()
# CHECK-NEXT:    [[VAL_1:%.*]] = toy.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
# CHECK-NEXT:    [[VAL_2:%.*]] = toy.generic_call @transpose_transpose([[VAL_1]]) : (tensor<2x3xf64>) -> tensor<*xf64>
# CHECK-NEXT:    toy.print [[VAL_2]] : tensor<*xf64>
# CHECK-NEXT:    toy.return
```

The `# CHECK` lines describe the expected `-opt` output; section 5.6 shows how to run them.

**Without `-opt`:**

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/transpose_transpose.toy -emit=mlir 2>&1
```

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

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt 2>&1
```

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

- **In `@transpose_transpose`:** `SimplifyRedundantTranspose` (the C++ pattern, section 3.2) matched the outer `%1 = toy.transpose(%0)` because `%0`'s defining op is itself a `toy.transpose`. `rewriter.replaceOp` rewired `toy.return` to use `%arg0` directly. The now-unused inner `%0 = toy.transpose(%arg0)` was then erased by the canonicalizer's DCE — legal only because `TransposeOp` is `Pure` (section 3.6). Result: the whole double-transpose reduced to `toy.return %arg0`.
- **In `@main`:** `%1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>` is a reshape of a constant *and* a reshape whose input and output types are identical, so two DRR patterns (section 4) match it: `FoldConstantReshapeOptPattern` (benefit 2) and `RedundantReshapeOptPattern` (benefit 1). The higher-benefit one wins: the reshape was folded into a new `toy.constant` of type `tensor<2x3xf64>`, and the old constant became dead. Both routes print the same IR; section 5.5 shows the location evidence for which one actually ran.
- **What did *not* happen:** the call to `@transpose_transpose` is still there. The canonicalizer performs local, intra-function rewrites; it does not inline or interprocedurally propagate. Chapter 4 (inlining + shape inference) is what finally lets `main` see through the call.

### 5.4 `trivial_reshape.toy`

***test_Example/Toy/Ch3/trivial_reshape.toy***
```
# RUN: toyc-ch3 %s -emit=mlir -opt 2>&1 | FileCheck %s

def main() {
  var a<2,1> = [1, 2];
  var b<2,1> = a;
  var c<2,1> = b;
  print(c);
}

# CHECK-LABEL: toy.func @main()
# CHECK-NEXT:    [[VAL_0:%.*]] = toy.constant
# CHECK-SAME: 		dense<[
# CHECK-SAME: 	 	[1.000000e+00], [2.000000e+00]
# CHECK-SAME: 		]> : tensor<2x1xf64>
# CHECK-NEXT:    toy.print [[VAL_0]] : tensor<2x1xf64>
# CHECK-NEXT:    toy.return
```

Section 4.5 introduced this input; these are the same two runs, end to end.

**Without `-opt`:**

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/trivial_reshape.toy -emit=mlir 2>&1
```

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

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/trivial_reshape.toy -emit=mlir -opt 2>&1
```

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>
    toy.print %0 : tensor<2x1xf64>
    toy.return
  }
}
```

All three reshapes vanished and the constant changed shape; section 4.5 walks through how `ReshapeReshapeOptPattern`, `FoldConstantReshapeOptPattern`, and DCE compose to get there, and section 5.5 shows from the locations which patterns fired.

### 5.5 Reading the locations: which pattern fired

The release build of Homebrew LLVM has no `-debug` output, but the locations tell the story. DRR-generated rewrites give the ops they create a **fused location** of all matched ops (section 4.4), while `replaceWithValue` just forwards an existing value and keeps *its* location. Rerun both tests with `-mlir-print-debuginfo`:

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt -mlir-print-debuginfo 2>&1
```

```mlir
module {
  toy.func @transpose_transpose(%arg0: tensor<*xf64> loc("../../test_Example/Toy/Ch3/transpose_transpose.toy":4:1)) -> tensor<*xf64> {
    toy.return %arg0 : tensor<*xf64> loc("../../test_Example/Toy/Ch3/transpose_transpose.toy":5:3)
  } loc("../../test_Example/Toy/Ch3/transpose_transpose.toy":4:1)
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64> loc(fused["../../test_Example/Toy/Ch3/transpose_transpose.toy":9:3, "../../test_Example/Toy/Ch3/transpose_transpose.toy":9:17])
    %1 = toy.generic_call @transpose_transpose(%0) : (tensor<2x3xf64>) -> tensor<*xf64> loc("../../test_Example/Toy/Ch3/transpose_transpose.toy":10:11)
    toy.print %1 : tensor<*xf64> loc("../../test_Example/Toy/Ch3/transpose_transpose.toy":11:3)
    toy.return loc("../../test_Example/Toy/Ch3/transpose_transpose.toy":8:1)
  } loc("../../test_Example/Toy/Ch3/transpose_transpose.toy":8:1)
} loc(unknown)
```

The surviving constant in `@main` sits at `fused[9:3, 9:17]`: `9:3` is the `var a<2, 3>` declaration (the reshape's location) and `9:17` is the literal (the original constant's location). That is a constant *created* by `FoldConstantReshapeOptPattern`, not the original constant forwarded by `RedundantReshapeOptPattern` — which would have kept plain `9:17`. The benefit-2 pattern won, as section 2.2's step 3 predicts.

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/trivial_reshape.toy -emit=mlir -opt -mlir-print-debuginfo 2>&1
```

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64> loc(fused["../../test_Example/Toy/Ch3/trivial_reshape.toy":6:3, "../../test_Example/Toy/Ch3/trivial_reshape.toy":5:3, "../../test_Example/Toy/Ch3/trivial_reshape.toy":4:3, "../../test_Example/Toy/Ch3/trivial_reshape.toy":4:16])
    toy.print %0 : tensor<2x1xf64> loc("../../test_Example/Toy/Ch3/trivial_reshape.toy":7:3)
    toy.return loc("../../test_Example/Toy/Ch3/trivial_reshape.toy":3:1)
  } loc("../../test_Example/Toy/Ch3/trivial_reshape.toy":3:1)
} loc(unknown)
```

The final constant carries **all four** source locations: the three reshapes (`6:3`, `5:3`, `4:3` — the `var c`, `var b`, `var a` lines) and the literal (`4:16`). Every reshape was therefore absorbed by a fusing pattern (`ReshapeReshapeOptPattern` or `FoldConstantReshapeOptPattern`); had `RedundantReshapeOptPattern` removed `%2` or `%3`, its location would be missing. This is also the "locations are preserved" property in practice: a diagnostic on the folded constant can still point at every source line that contributed to it.

### 5.6 Poking at the driver by hand

**Watching the pipeline.** Because `main()` registers the pass-manager CL options and `dumpMLIR` calls `applyPassManagerCLOptions` (section 3.4), you can print the IR around each pass. The canonicalizer is nested on `toy.func`, so the dumps are per function:

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/trivial_reshape.toy -emit=mlir -opt \
    --mlir-print-ir-before-all --mlir-print-ir-after-all 2>&1
```

```mlir
// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
toy.func @main() {
  %0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
  %1 = toy.reshape(%0 : tensor<2xf64>) to tensor<2x1xf64>
  %2 = toy.reshape(%1 : tensor<2x1xf64>) to tensor<2x1xf64>
  %3 = toy.reshape(%2 : tensor<2x1xf64>) to tensor<2x1xf64>
  toy.print %3 : tensor<2x1xf64>
  toy.return
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>
  toy.print %0 : tensor<2x1xf64>
  toy.return
}

module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>
    toy.print %0 : tensor<2x1xf64>
    toy.return
  }
}
```

**Running the `# CHECK` lines.** The `.toy` files carry `# RUN:` / `# CHECK:` lines for LLVM `lit`/`FileCheck`. There is no lit harness in this repo, but Homebrew LLVM ships `FileCheck`, so you can run a test's RUN line by hand; no output and exit status 0 means the checks pass (both do against the current binary):

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt 2>&1 \
  | FileCheck ../../test_Example/Toy/Ch3/transpose_transpose.toy
./build/toyc-ch3 ../../test_Example/Toy/Ch3/trivial_reshape.toy -emit=mlir -opt 2>&1 \
  | FileCheck ../../test_Example/Toy/Ch3/trivial_reshape.toy
```

**Splitting frontend and optimizer.** `-opt` is nothing Toy-specific — it just adds MLIR's *generic* `createCanonicalizerPass()` to the pass manager (section 3.4). You can make the stage boundary visible by splitting frontend and optimizer into two processes connected by textual IR, exactly how `mlir-opt`-based pipelines compose (`toyc-ch3` plays the role of the project's own `foo-opt`, since stock `mlir-opt` doesn't link the Toy dialect — see Ch2 section 6.7):

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/trivial_reshape.toy -emit=mlir 2>&1 \
  | ./build/toyc-ch3 - -x mlir -emit=mlir -opt 2>&1
```

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>
    toy.print %0 : tensor<2x1xf64>
    toy.return
  }
}
```

Same optimized module as the single `-opt` invocation: textual IR is the ecosystem's stable interface between compilation stages.

**The error path.** Invalid MLIR input fails in the parser/verifier, before any pass runs:

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/invalid.mlir -emit=mlir -opt; echo "exit=$?"
```

```
../../test_Example/Toy/Ch3/invalid.mlir:8:8: error: 'toy.print' op requires zero results
  %0 = "toy.print"()  : () -> tensor<2x3xf64>
       ^
../../test_Example/Toy/Ch3/invalid.mlir:8:8: note: see current operation: %0 = "toy.print"() : () -> tensor<2x3xf64>
Error can't load file ../../test_Example/Toy/Ch3/invalid.mlir
exit=3
```

The diagnostic is rendered with the source line and caret by the `SourceMgrDiagnosticHandler` in `dumpMLIR()`, and exit code 3 comes from `loadMLIR()` (section 3.4).

---

## 6. Key Takeaways & Pitfalls

**Takeaways**

- **Optimize at the right abstraction level.** `transpose(transpose(x)) → x` is a two-op pattern match in the Toy dialect and a near-impossible loop-nest analysis in LLVM IR. This asymmetry is the whole point of building high-level dialects.
- **Two pattern authoring styles, one engine.** C++ `OpRewritePattern` for arbitrary logic; DRR for concise structural rewrites (with `Constraint`/`CPred` for conditions and `NativeCodeCall` for embedded C++). Both end up as `RewritePattern`s in the same `RewritePatternSet` consumed by the same greedy driver.
- **Canonicalization is the cheap distribution channel.** `let hasCanonicalizer = 1` in ODS + a `getCanonicalizationPatterns` definition means *every* user of `createCanonicalizerPass()` gets your patterns for free — no custom pass required.
- **Small patterns compose via fixpoint iteration.** Nobody wrote "fold a chain of three reshapes into a reshaped constant"; independent one-step patterns plus DCE reached that result together.
- **Locations are preserved.** DRR-generated code fuses the source ops' locations into the replacement, keeping diagnostics meaningful after transformation — and, as section 5.5 shows, making it possible to tell which pattern fired.

**Pitfalls**

- **Patterns don't clean up after themselves — DCE does.** `SimplifyRedundantTranspose` leaves the inner transpose dead. Don't build erasure of *upstream* producers into your pattern (they may have other users!); instead make ops declare their effects and let the canonicalizer's DCE remove what becomes dead. Corollary: **forgetting `Pure` (or a proper `MemoryEffects` interface) silently blocks DCE** — you'll see dead ops persist with no error anywhere.
- **Forgetting `let hasCanonicalizer = 1`** means the generated op class has no `getCanonicalizationPatterns` declaration; your `.cpp` definition then fails to compile (or, if you register patterns some other way, the canonicalizer never asks your op for them).
- **Only mutate through the `PatternRewriter`.** Direct IR surgery inside `matchAndRewrite` breaks the driver's worklist tracking and can crash or miscompile. Also: decide match *before* mutating — a pattern that mutates then returns `failure()` is broken.
- **Rewrites must keep consumers legal.** Replacing a value changes the operand of its users (`toy.return`, `toy.generic_call`, ...). Here the permissive `tensor<*xf64>`/`F64Tensor` typing makes that safe; with stricter ops you must check result/operand type compatibility in the match (as `RedundantReshapeOptPattern` does via `TypesAreIdentical`).
- **Don't create infinite ping-pong.** The greedy driver iterates to fixpoint; a pair of patterns like `A→B` and `B→A` (or a pattern that "rewrites" an op to an identical op) never converges and hits the iteration limit. Each rewrite should strictly reduce some measure (op count, pattern-match opportunities).
- **Op ordering / traversal is not yours to control.** Patterns fire in benefit order per op, over a worklist in unspecified overall order. Never write a pattern whose correctness depends on another pattern having already run — each must be independently sound; only *convergence* composes them. Likewise, don't assume *which* of several matching patterns fires: in section 5.3 the benefit-2 constant fold, not the "obvious" redundant-reshape pattern, removed `main`'s reshape.
- **Canonicalization is local.** It didn't remove the `toy.generic_call` round-trip in `main` — interprocedural wins need the inliner and shape inference, which is exactly where [Chapter 4](../Ch4/README.md) picks up.
- **DRR `$arg` on a `ConstantOp` binds the attribute, not a value.** Because ODS declares `ConstantOp`'s input as an attribute, DRR captures a `DenseElementsAttr` — which is why `ReshapeConstant` can call `.reshape(...)` on it directly. Misreading bound-entity kinds ($value vs $attribute) is a classic DRR stumble.
- **LLVM version drift.** Against LLVM/MLIR 20: the trait is `Pure` (not `NoSideEffect`), casts are `::llvm::cast<ShapedType>(v)` (not `v.cast<ShapedType>()`), and `matchAndRewrite` returns `llvm::LogicalResult`. Older tutorial snippets found online may not compile as-is.

---

## Appendix: What Chapter 3 adds to the build

The chapter keeps Chapter 2's structure — `include/toy/CMakeLists.txt` generating the op/dialect `.inc` files into target `ToyCh3OpsIncGen`, plus the explicit `add_dependencies` race guard (see [Ch2 appendix](../Ch2/README.md#appendix-what-chapter-2-adds-to-the-build)) — and adds a second TableGen flavor for the DRR patterns.

### A.1 The chapter targets

Below the standalone guard:

***CMakeLists.txt***
```cmake
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

Relative to Chapter 2:

1. **The rewriter TableGen step** — a new kind of generator for the DRR patterns of section 4. `mlir_tablegen(ToyCombine.inc -gen-rewriters)` invokes `mlir-tblgen -gen-rewriters` to emit `ToyCombine.inc` into the chapter's binary directory — `Ch3/build/ToyCombine.inc` standalone (`toy/build/Ch3/ToyCombine.inc` in the superbuild). `add_public_tablegen_target` wraps it in the named target `ToyCh3CombineIncGen`, and `add_dependencies(toyc-ch3 ToyCh3CombineIncGen)` guarantees the file exists before `ToyCombine.cpp` compiles. This sits alongside the Chapter-2 op generation (`ToyCh3OpsIncGen`).
2. **`include_directories(include)` moved above the TableGen rule.** `mlir_tablegen` passes the directory's include paths to `mlir-tblgen` as `-I` flags, and `ToyCombine.td` needs `include "toy/Ops.td"` to resolve to `Ch3/include/toy/Ops.td`. The generated build rule shows it: `mlir-tblgen -gen-rewriters -I /Users/roy/study/mlir/toy/Ch3 ... -I/Users/roy/study/mlir/toy/Ch3/include .../mlir/ToyCombine.td`.
3. **New source file** `mlir/ToyCombine.cpp` in the executable.
4. **Include path for generated files:** `include_directories(${CMAKE_CURRENT_BINARY_DIR})` is what lets `#include "ToyCombine.inc"` resolve, since this `.inc` lands at the *root* of the chapter's binary directory (unlike the op/dialect `.inc` files under `include/toy/`). `CMAKE_CURRENT_BINARY_DIR` points to the right place in both superbuild and standalone modes.

Linking is unchanged: the monolithic Homebrew `libMLIR.dylib` / `libLLVM.dylib`, which already contain the canonicalizer pass and greedy driver (in-tree builds would add `MLIRPass`, `MLIRTransforms`, ... as components instead).

---

## Links

- Official doc: [MLIR Toy Tutorial Ch.3 — High-level Language-Specific Analysis and Transformation](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/)
- DRR reference: [Table-driven Declarative Rewrite Rules](https://mlir.llvm.org/docs/DeclarativeRewrites/) · [Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/) · [Operation Canonicalization](https://mlir.llvm.org/docs/Canonicalization/)
- Previous: [Chapter 2 — Emitting Basic MLIR](../Ch2/README.md)
- Next: [Chapter 4 — Enabling Generic Transformation with Interfaces](../Ch4/README.md)
- Index: [README](../README.md)
