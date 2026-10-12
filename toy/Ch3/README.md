# Chapter 3: High-level Language-Specific Analysis and Transformation

> **Goal:** Use the high-level semantics preserved by the Toy dialect to implement optimizations that would be impossible (or very hard) after lowering — first with a hand-written C++ `RewritePattern`, then with declarative TableGen rewrite rules (DRR) — and hook them all into MLIR's canonicalization framework.
> Official docs: [Toy Tutorial Ch.3 — High-level Language-Specific Analysis and Transformation](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/)

---

## 1. Overview

Chapter 2 gave us a Toy dialect that faithfully models the source language: `toy.transpose`, `toy.reshape`, `toy.constant`, and friends. Now we get the payoff. Because the IR still *knows* it is dealing with a tensor transpose — not a soup of loads, stores, and index arithmetic — we can express language-level algebraic identities as trivial local rewrites. Analyses like these are usually done on the language AST (Clang, for example, has a heavy `TreeTransform` mechanism for C++ template instantiation); a dialect that mirrors the language lets MLIR do them on the IR instead.

In this chapter we write four such rewrites:

- `transpose(transpose(x)) → x` — hand-written C++ (`SimplifyRedundantTranspose` in `mlir/ToyCombine.cpp`, section 3)
- Three reshape optimizations — DRR (`mlir/ToyCombine.td`, section 4)

and turn them on with a new `-opt` driver flag that runs MLIR's stock canonicalizer (section 3.4). Section 5 builds the chapter and runs it end to end.

The transpose pattern is also written in DRR (section 3.7), to compare the two styles piece by piece. That version lives in `mlir/alt/drr_transpose/`, which the build doesn't use.

The appendices hold the extras: reading the locations to see which pattern fired (A), more runs — IR printing, `FileCheck`, a two-process pipeline, the error path (B), every field of an ODS (Operation Definition Specification) op (C), what Chapter 3 adds to the build (D), and the script that runs every command in this README (E).

---

## 2. Pattern Rewriting and Canonicalization

Compiler transformations fall into two categories: **local** ones, which look at a small neighborhood of ops (an op and its operands' producers), and **global** ones, which need a view of a whole function or program (inlining, shape inference — Chapter 4). This chapter is about local pattern-match transformations that are trivial on Toy IR but hard in LLVM, and it uses MLIR's generic DAG (directed acyclic graph) rewriter to run them.

MLIR gives us two ways to write such a pattern:

| Approach | Mechanism | Best for |
|---|---|---|
| **Imperative C++** | Subclass `mlir::OpRewritePattern<Op>`, implement `matchAndRewrite` | Matches that need arbitrary C++ logic, side-effect reasoning, non-structural conditions |
| **Declarative (DRR)** | TableGen `Pat<...>` records, compiled by `mlir-tblgen -gen-rewriters` into C++ | Structural DAG-to-DAG rewrites; concise, less boilerplate, harder to get wrong |

DRR matches and builds ops by their ODS records (`ReshapeOp`, `ConstantOp`), so it only works for operations defined with ODS, as in Chapter 2.

**Who runs the patterns.** A pattern only describes one rewrite; it doesn't run by itself. Whichever way it is written, it ends up as a `RewritePattern` object, and each op type hands its patterns to MLIR through a static `getCanonicalizationPatterns` hook (section 3.3). The **canonicalizer**, a stock MLIR pass that cleans up IR, collects these patterns and applies them with a *greedy pattern rewrite driver*: it keeps applying any pattern that matches, anywhere in the function, until none does. Section 3.4 adds the canonicalizer to `toyc-ch3` as the `-opt` flag, and section 4.5 watches the driver work on a real input, one rewrite at a time.

---

## 3. Optimize Transpose using C++ Pattern-Match and Rewrite

### 3.1 The motivating example

Start with a simple pattern: eliminate a sequence of two transposes that cancel out, `transpose(transpose(X)) -> X`. Two runs set the scene: the Toy IR, where the pattern is visible, and the equivalent C++, where Clang can't find it. The build is in section 5.1.

#### In Toy IR the two transposes are adjacent ops

The generic function in this chapter's first test input is exactly that pattern (abridged; section 5.2 shows the whole file):

***test_Example/Toy/Ch3/transpose_transpose.toy***
```text
...
def transpose_transpose(x) {
  return transpose(transpose(x));
}
```

Emit its IR without `-opt`, so no pattern runs:

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/transpose_transpose.toy -emit=mlir 2>&1
```

Real output **[1]** (abridged to `@transpose_transpose`; section 5.2 shows all of it):

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

The match is trivial on Toy IR but hard for LLVM. Write the same computation in C++ (not this repo's code), with the naive transpose as loops, and Clang can't optimize away the temporary array. The command passes the snippet to Homebrew clang 20.1.8 on stdin (`-x c++ -`), compiles at `-O3` to LLVM IR, and keeps only the allocas, loop back-edges, and the `sink` call:

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

Real output **[2]**:

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
- Four `!llvm.loop` back-edges remain: both loop nests (two loops each) survive. `sink(A)` passes `A` to an unknown function, so the loops can't be deleted as dead code; Clang would have to prove that they cancel out, and it doesn't.

By the time this reaches LLVM IR, the "transpose-ness" is gone: LLVM sees two loop nests shuffling memory through a temporary buffer, and proving that they cancel out takes loop and memory-dependence reasoning that the optimizer doesn't do. In the Toy dialect, the same fact is a one-line pattern match: *"a transpose whose operand is produced by another transpose can be replaced by the inner transpose's operand."*

This is the core argument for multi-level IR: **do each optimization at the abstraction level where it is cheap to express and trivially correct.**

### 3.2 The `SimplifyRedundantTranspose` pattern

In C++, a rewrite (match a small tree of ops, replace it with others) is a `RewritePattern` subclass, registered with MLIR's canonicalizer. The pattern and its registration hook in `ToyCombine.cpp`:

***mlir/ToyCombine.cpp***
```cpp
/// This is an example of a c++ rewrite pattern for the TransposeOp. It
/// optimizes the following scenario: transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {  // (1)
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}                  // (2)

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  llvm::LogicalResult
  matchAndRewrite(TransposeOp op,                                                 // (3)
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current transpose.
    mlir::Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();   // (4)

    // Input defined by another transpose? If not, no match.
    if (!transposeInputOp)
      return failure();                                                           // (5)

    // Otherwise, we have a redundant transpose. Use the rewriter.
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});                      // (6)
    return success();                                                             // (7)
  }
};

/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);                               // (8)
}
```

Let's unpack every moving part. The numbers match the `// (N)` markers added to the code above.

**(1) `OpRewritePattern<TransposeOp>`** — a convenience base class for patterns rooted at one specific op type. The greedy driver will only call this pattern on `toy.transpose` operations, and `matchAndRewrite` receives the operation already cast to the typed `TransposeOp` wrapper. (The more general base, `RewritePattern`, matches on an op *name* string and hands you a raw `Operation *` — that's what DRR-generated patterns use, as section 3.7 shows.)

**(2) `benefit`** — a heuristic cost passed to the base constructor (`/*benefit=*/1` here). When multiple patterns can fire at the same rooted op, the driver tries higher-benefit patterns first. With a single pattern per root the value barely matters; it becomes relevant when you have several patterns for the same op — as `toy.reshape` does, where section 4.5 (step 2) shows a benefit-2 DRR pattern beating a benefit-1 one.

**(3) `matchAndRewrite`** — the fused match + rewrite hook. The contract is strict:

- **(4) Match.** Walk up from the transpose's operand with `Value::getDefiningOp<TransposeOp>()`. If the operand is a block argument or is produced by some other op, `transposeInputOp` is null.
- **(5) No match: return `failure()` before mutating anything.** The driver then tries the next pattern, or moves on to the next op.
- **(6) Rewrite through the rewriter.** If the pattern applies, perform **all** IR mutation through the supplied `PatternRewriter` (`replaceOp`, `eraseOp`, `create<...>`, `modifyOpInPlace`, ...). The rewriter is how the driver tracks what changed, keeps its worklist up to date, and (in dialect-conversion contexts) supports rollback. Mutating IR behind the rewriter's back is undefined behavior for the driver. Here the whole rewrite is one line: `replaceOp` replaces every use of the *outer* transpose's result with the *inner* transpose's input, the original `x`.
- **(7) Return `success()`** after rewriting.

**(8) Registration.** A pattern is inert until something runs it. Rather than writing a bespoke pass, the last function in the block registers it as a **canonicalization pattern** on `TransposeOp`. The canonicalization pass applies the transformations defined by operations in a greedy, iterative manner (section 4.5); `getCanonicalizationPatterns` is the static hook it calls for **every op registered in the context** when it builds its pattern set. An op class only inherits an empty version of that hook, so `TransposeOp` must declare its own, which it does through ODS (section 3.3).

Section 3.7 writes this same pattern in DRR and compares the two piece by piece.

### 3.3 `hasCanonicalizer = 1` in ODS

The canonicalizer finds its patterns by asking every op registered in the context, through the op class's static `getCanonicalizationPatterns` hook. Every op class has that hook, because `mlir::OpState`, the base of every op class, defines it with an empty body:

```bash
cd /Users/roy/study/mlir/toy/Ch3
grep -B2 -A1 'static void getCanonicalizationPatterns' /opt/homebrew/opt/llvm@20/include/mlir/IR/OpDefinition.h
```

Real output **[3]**:

```cpp
  /// This hook returns any canonicalization pattern rewrites that the operation
  /// supports, for use by the canonicalization pass.
  static void getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {}
```

So an op that does nothing contributes no patterns. To contribute some, the op class must declare its own `getCanonicalizationPatterns`, which hides the empty one, and that declaration comes from ODS: setting the field `hasCanonicalizer` to 1 makes `mlir-tblgen` declare it. Like `hasVerifier` in Chapter 2, the field is a promise: ODS declares the method, and `ToyCombine.cpp` defines it (section 3.2).

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

`let hasCanonicalizer = 1;` makes `mlir-tblgen -gen-op-decls` emit the declaration into the generated op class, once in `class ReshapeOp` and once in `class TransposeOp`:

```bash
cd /Users/roy/study/mlir/toy/Ch3
mlir-tblgen -gen-op-decls include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include | grep 'getCanonicalizationPatterns'
```

Real output **[4]**:

```cpp
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
```

We then supply the *definition* by hand in `ToyCombine.cpp` (section 3.2). Without the flag, the generated op class has no such declaration, and C++ rejects an out-of-line definition of a member the class never declared: the definition in `ToyCombine.cpp` fails to compile.

**Other fields like it.** `hasCanonicalizer` is one of the fields of ODS's `Op` class that an op sets with `let`. Two more feed the canonicalizer: `hasCanonicalizeMethod` declares a single pattern as a static `canonicalize(op, rewriter)` method, and `hasFolder` declares a `fold()` method, which the canonicalizer's driver tries before any pattern (section 4.5). The other fields aren't about optimization: they describe the op's operands, results and regions, its syntax, builders, verification and documentation. Generic optimizations reach an op through its traits instead (`Pure`, section 3.6), and in Chapter 4 through interfaces. Appendix C lists every field and what it generates.

### 3.4 Adding an optimization pipeline to `toyc.cpp`

Patterns run inside a pass; passes run inside a `PassManager`, much as in LLVM. The driver gains a new `-opt` flag and a `dumpMLIR()` that builds and runs that pipeline between loading the module and dumping it (abridged):

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
  if (int error = loadMLIR(sourceMgr, context, module))                    // (1)
    return error;

  if (enableOpt) {
    mlir::PassManager pm(module.get()->getName());                         // (2)
    // Apply any generic pass manager command line options and run the pipeline.
    if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))                 // (3)
      return 4;

    // Add a run of the canonicalizer to optimize the mlir module.
    pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());  // (4)
    if (mlir::failed(pm.run(*module)))                                     // (5)
      return 4;
  }

  module->dump();
  return 0;
}
```

Key details; the numbers match the `// (N)` markers in the code:

- **(1) `loadMLIR(...)`** — Chapter 2's `dumpMLIR()` did the Toy-vs-`.mlir` dispatch and dumped immediately. Here that dispatch moved into a helper, `loadMLIR()`, which fills in `module` from either MLIRGen (`.toy`) or the MLIR parser (`.mlir` / `-x mlir`) and returns an error code (6 for a Toy parse error, 1 for an MLIRGen failure, 3 for an MLIR parse failure, -1 for a `.mlir` file that can't be opened). That way the pipeline runs on either kind of input. The `SourceMgrDiagnosticHandler` renders diagnostics with file:line:col and a caret (the error path in appendix B.4).
- **(2) `mlir::PassManager pm(module.get()->getName())`** — the pass manager is anchored on the top-level op type (`builtin.module` here).
- **(3) `applyPassManagerCLOptions(pm)`** transfers generic pass-manager flags from the command line onto this pass-manager instance: `--mlir-print-ir-before-all`, `--mlir-print-ir-after-all`, `--mlir-pass-statistics`, crash reproducer options, etc. For these flags to exist at all, `main()` must call the matching registration function — and it does (below).
- **(4) `pm.addNestedPass<mlir::toy::FuncOp>(...)`** — schedules the canonicalizer to run *nested on each `toy.func`* rather than on the whole module. Nesting needs each `toy.func` to be `IsolatedFromAbove`, which it is (Ch2 appendix A.3), and it lets the pass manager process functions in parallel. It is also why the `--mlir-print-ir-*` dumps in appendix B.1 show one `toy.func` at a time.
- **(4) `mlir::createCanonicalizerPass()`** (from `mlir/Transforms/Passes.h`) — the stock canonicalizer. It collects every registered op's canonicalization patterns, then runs the greedy driver to fixpoint, which also tries each op's `fold()` (section 4.5). We wrote zero pass boilerplate ourselves.
- **(5) `pm.run(*module)`** runs the pipeline; a failure returns 4, as does a bad pass-manager option at (3).
- **No CSE (common subexpression elimination) pass yet.** This chapter's `toyc.cpp` runs only the canonicalizer; Chapter 4 adds `mlir::createCSEPass()` alongside the shape-inference work.

Since the pipeline is only constructed under `if (enableOpt)`, running without `-opt` gives you the raw MLIRGen output — which is exactly how the "before" dumps in sections 3.1, 4.5 and 5.2 are produced.

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

### 3.5 Without `Pure`: a dead transpose is left behind

The shipped `Ops.td` declares `TransposeOp` with `[Pure]`, which hides what the pattern alone does. To see it, build a scratch copy of the chapter with the trait removed — the command copies the sources to a temporary directory, deletes `[Pure]` from `def TransposeOp` with `sed`, builds the copy (about 10 seconds) and runs it on `transpose_transpose.toy` with `-opt`:

```bash
cd /Users/roy/study/mlir/toy/Ch3
S=$(mktemp -d)
cp -R CMakeLists.txt include mlir parser toyc.cpp "$S"/
sed -i '' 's/^def TransposeOp : Toy_Op<"transpose", \[Pure\]>/def TransposeOp : Toy_Op<"transpose">/' "$S"/include/toy/Ops.td
cmake -S "$S" -B "$S"/build -G Ninja >/dev/null && cmake --build "$S"/build >/dev/null
"$S"/build/toyc-ch3 ../../test_Example/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt 2>&1 | sed -n '1,5p'
rm -rf "$S"
```

Real output **[5]** (abridged to `@transpose_transpose`; `@main` is the same as in the `-opt` run of section 5.2):

```mlir
module {
  toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
    toy.return %arg0 : tensor<*xf64>
  }
```

The pattern worked: `toy.return` now returns the function argument directly, bypassing both transposes. But `SimplifyRedundantTranspose` only replaced the *outer* transpose; it never erased the *inner* one, and `%0` is now dead.

Why doesn't the canonicalizer just delete `%0`? It does know how to clean up dead operations, but it is **conservative about side effects**: unless an op says otherwise, MLIR assumes it might print or touch memory, and deleting it because its result is unused would be wrong. The canonicalizer only erases dead ops it can *prove* are side-effect free.

### 3.6 The `Pure` trait and dead code elimination

The fix is to declare the op side-effect free. The `def TransposeOp : Toy_Op<"transpose", [Pure]>` line in section 3.3 carries the **`Pure`** trait, and so does `ReshapeOp`. (Ch2 section 4.4, step 5 explains `Pure` and the traits it expands into in the generated class.) Compared with Chapter 2, this chapter's `Ops.td` adds `Pure` to `AddOp`, `MulOp`, `ReshapeOp`, and `TransposeOp`; `ConstantOp` and `ReturnOp` already had it. `GenericCallOp` and `PrintOp` stay without it — a call may have arbitrary effects and printing is the point of `print`.

`Pure` means "no side effects and always speculatable" — the strongest promise. The no-memory-effect half is what dead-code removal needs; speculatability matters for moving ops, not for deleting them. In MLIR 20's `mlir/Interfaces/SideEffectInterfaces.td` it is `def Pure : TraitList<[AlwaysSpeculatable, NoMemoryEffect]>;`, and the generated class shows the result: `TransposeOp` picks up the speculation traits and `MemoryEffectOpInterface` (whose `getEffects` reports no effects):

```bash
cd /Users/roy/study/mlir/toy/Ch3
mlir-tblgen -gen-op-decls include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include | grep '^class TransposeOp : '
```

Real output **[6]**:

```cpp
class TransposeOp : public ::mlir::Op<TransposeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
```

In the scratch build of section 3.5 the same line ends at `::mlir::OpTrait::OpInvariants>`. With the trait in place, the canonicalizer's built-in DCE (dead code elimination) sweeps the dead inner transpose away. Run the shipped binary:

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt 2>&1
```

Real output **[7]** (abridged to `@transpose_transpose`; section 5.2 shows all of it):

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

**Two things that keep the result valid.**

- **Replacing a value must keep every user valid.** `replaceOp` gives the outer transpose's users, here `toy.return`, the inner transpose's operand instead. That is safe because both values have the same type, `tensor<*xf64>`. A pattern that substituted a value of another type would have to keep every user verifiable; `ReturnOp::verify`, for instance, checks its operand against the function's result type (Ch2 appendix A.7).
- **Dead-code removal spares terminators.** The inner transpose was erased because it is `Pure` and its result is unused. `ReturnOp` is `Pure` too, and has no results at all, yet `toy.return` survives in every output above: the canonicalizer never erases a terminator, and the `Terminator` trait in `ReturnOp`'s `[Pure, HasParent<"FuncOp">, Terminator]` marks it as one.

Traits combine: **`Pure` says an op may be removed when unused, `Terminator` says it stays even when unused, and dead-code removal consults both.**

### 3.7 The same pattern in DRR

Section 4 introduces **Declarative Rewrite Rules (DRR)**: a pattern written as a TableGen record that names the ops to match and what to replace them with, from which `mlir-tblgen -gen-rewriters` generates the `RewritePattern` class. The transpose rewrite is simple enough to preview DRR on it here. This version isn't part of the build: it lives in `mlir/alt/drr_transpose/`, which the build never reads, and the end of this section checks that it optimizes exactly like the C++ pattern.

The rewrite of section 3.2, `transpose(transpose(x)) → x`, as one DRR record, `Pat<source, result>`:

***mlir/alt/drr_transpose/TransposeTranspose.td***
```tablegen
// Transpose(Transpose(x)) = x
def TransposeTransposeOptPattern : Pat<(TransposeOp(TransposeOp $arg)),
                                       (replaceWithValue $arg)>;
```

- **The source DAG** `(TransposeOp(TransposeOp $arg))` matches a transpose whose operand is produced by another transpose, and binds the *inner* transpose's operand to `$arg`.
- **The result** `(replaceWithValue $arg)` creates no op: it replaces every use of the matched outer transpose with `$arg`.

The file holds only this record and the two `include`s it needs (`PatternBase.td` and `toy/Ops.td`). The pattern class `mlir-tblgen` generates is named after the record, so registering it changes one line of `TransposeOp::getCanonicalizationPatterns` in `ToyCombine.cpp`:

```cpp
  results.add<SimplifyRedundantTranspose>(context);   // the real file
  results.add<TransposeTransposeOptPattern>(context);  // with the DRR pattern
```

The generated class, for comparison with section 3.2's hand-written one:

```bash
cd /Users/roy/study/mlir/toy/Ch3
mlir-tblgen -gen-rewriters mlir/alt/drr_transpose/TransposeTranspose.td -I include -I /opt/homebrew/opt/llvm@20/include \
  | awk '/^struct TransposeTransposeOptPattern /,/^};/'
```

Real output **[8]** (with `// (N)` markers added for the table below):

```cpp
struct TransposeTransposeOptPattern : public ::mlir::RewritePattern {                                      // (1)
  TransposeTransposeOptPattern(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("toy.transpose", 2, context, {}) {}                                         // (2)
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,                                            // (3)
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range arg(op0->getOperands());                                              // (4)
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::toy::TransposeOp>(op0); (void)castedOp0;                     // (5)
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();                                  // (6)
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {                      // (7)
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::toy::TransposeOp>(op1); (void)castedOp1;                   // (8)
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {                            // (9)
          diag << "castedOp1 is not ::mlir::toy::TransposeOp type";
        });
      }
      arg = castedOp1.getODSOperands(0);                                                                   // (10)
      tblgen_ops.push_back(op1);                                                                           // (11)
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;  // (12)
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ arg }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);                                                           // (13)
    return ::mlir::success();
  }
};
```

Piece by piece against `SimplifyRedundantTranspose`. The numbers are `// (N)` markers: section 3.2's in the hand-written column, the listing above in the generated one.

| Piece | Hand-written `SimplifyRedundantTranspose` (3.2) | Generated `TransposeTransposeOptPattern` |
|---|---|---|
| Base class | (1) `OpRewritePattern<TransposeOp>`: (3) `matchAndRewrite` gets a typed `TransposeOp` | (1) `RewritePattern`, matched by the op name in (2); (3) `matchAndRewrite` gets an `Operation *op0` and (5) casts it itself |
| Benefit | (2) `1`, written by hand | (2) `2`, the number of ops in the source DAG |
| Ops the rewrite creates | not declared | (2) `{}`, the last constructor argument: none |
| Match | (4) `getDefiningOp<TransposeOp>()`: one null check | (6) `getDefiningOp()`, then (8) `dyn_cast<TransposeOp>`: two checks, one for each way the match can fail |
| On a failed match | (5) `return failure();` | (7), (9) `rewriter.notifyMatchFailure(...)`, with a message for each check |
| Bound values | `transposeInputOp.getOperand()`, used directly | (4) `arg`, (10) filled in during the match; (11) `tblgen_ops` records the matched ops |
| Rewrite | (6) `rewriter.replaceOp(op, {transposeInputOp.getOperand()})` | (13) `rewriter.replaceOp(op0, tblgen_repl_values)`, holding `arg` |
| Location | not needed: no op is created | (12) `odsLoc`, the fused location of both ops, is computed but unused (`(void)odsLoc`) |
| Registration | (8) `results.add<SimplifyRedundantTranspose>(context)` | `results.add<TransposeTransposeOptPattern>(context)`, the changed line above |

**What the generated code adds.** `mlir-tblgen` emits the same skeleton for every DRR pattern, sized for the general case: several matched ops, several new ops, several replacement values. This one-line rewrite uses only part of it, so the rest looks like noise next to the hand-written version. Piece by piece, with the listing's markers:

- **(3), (5) An untyped entry and its own cast.** A `RewritePattern` receives a plain `Operation *op0`, and the code casts it to `TransposeOp` itself. The cast can't fail, because the constructor (2) names the root op `"toy.transpose"`, so the driver only calls the pattern on transposes. `(void)castedOp0` keeps the compiler quiet in patterns that never use the cast.
- **(6), (8) Two checks where C++ has one.** `getDefiningOp<TransposeOp>()` in section 3.2 answers two questions in one call: is the operand produced by an op, and is that op a transpose? The generator checks one DAG node at a time, so it asks them separately: `getDefiningOp()` (6), then `dyn_cast<TransposeOp>` (8).
- **(7), (9) A reason for each failure.** `notifyMatchFailure` returns `failure()` just like section 3.2's `return failure();`, but first offers a message to the rewriter's *listener*, if one is attached (a debugging or diagnostics tool built on the rewriter, for example). Without a listener, the callback that builds the message never runs, so the extra code costs nothing.
- **(4), (10) Bound values are kept as ranges.** A DRR `$name` can bind a whole operand group, for example a variadic one, so `arg` is declared as an `operand_range` (4) and filled in during the match (10), instead of being a single `Value`.
- **(11) Every matched op is recorded** in `tblgen_ops` (the root is pushed first, just before the cast). The list is only used to build `odsLoc`.
- **(12) A location for new ops.** `odsLoc` fuses the locations of all matched ops, for any op the rewrite creates. `replaceWithValue` creates none, so `(void)odsLoc` marks it as deliberately unused. In the reshape patterns of section 4, which do create ops, this is where their `fused[...]` locations come from.
- **(13) Replacement values in a vector.** A DRR result can replace an op that has several results, so the replacements are collected in `tblgen_repl_values`; here the loop right before (13) copies in just `arg`.

Both versions do the same three steps: get the operand's defining op, check that it is a `toy.transpose`, and replace the outer transpose's result with the inner transpose's operand. Everything else in the generated code is that general-purpose skeleton. Its benefit of 2 changes nothing here, because it is the only pattern on `toy.transpose`.

**The same result.** A temporary copy of the chapter gets the DRR record appended to its `ToyCombine.td`, and `sed` swaps the registration line shown above. The copy is built and run on `transpose_transpose.toy`:

```bash
cd /Users/roy/study/mlir/toy/Ch3
tmp=$(mktemp -d)
cp -R CMakeLists.txt toyc.cpp parser mlir include "$tmp"/
cat mlir/alt/drr_transpose/TransposeTranspose.td >> "$tmp"/mlir/ToyCombine.td
sed -i '' 's/results.add<SimplifyRedundantTranspose>/results.add<TransposeTransposeOptPattern>/' "$tmp"/mlir/ToyCombine.cpp
cmake -S "$tmp" -B "$tmp"/build -G Ninja > /dev/null
cmake --build "$tmp"/build > /dev/null 2>&1; echo "cmake --build exit status: $?"
"$tmp"/build/toyc-ch3 ../../test_Example/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt 2>&1 | sed -n '2,4p'
diff <(./build/toyc-ch3 ../../test_Example/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt -mlir-print-debuginfo 2>&1) \
     <("$tmp"/build/toyc-ch3 ../../test_Example/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt -mlir-print-debuginfo 2>&1) \
  && echo "identical to the real build"
rm -r "$tmp"
```

Real output **[9]**:

```text
cmake --build exit status: 0
  toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
    toy.return %arg0 : tensor<*xf64>
  }
identical to the real build
```

The DRR build reduces `@transpose_transpose` to a return of its argument, as the C++ pattern does in section 3.6, and its whole output, locations included, matches the real build's. (`SimplifyRedundantTranspose` stays in the copy's `ToyCombine.cpp`, but nothing registers it.)

**What each style asks you to write.** In general, beyond this one pattern:

| | DRR | Hand-written C++ |
|---|---|---|
| Match | the source DAG; `Constraint`/`CPred` for extra conditions | `getDefiningOp<...>()`, `dyn_cast`, and `if`s |
| Rewrite | the result DAG, or `replaceWithValue` | `rewriter.create<...>`, `rewriter.replaceOp` |
| New op's result type | taken from the replaced root | passed explicitly |
| Benefit | computed: the number of ops in the source DAG | written by hand |
| Location of new ops | fused location of the matched ops, automatically | written by hand (`getFusedLoc`) |
| Failure diagnostics | a `notifyMatchFailure` message per check, generated | whatever you write |
| C++ escape hatch | `NativeCodeCall`, `CPred` strings | everything is C++ |

DRR fits rewrites that are a DAG in, a DAG out, with simple conditions; hand-written C++ fits matches that need loops, analyses, or control flow DRR can't express.

---

## 4. Optimize Reshapes using DRR

Writing `matchAndRewrite` by hand is flexible but verbose: the transpose pattern took about 25 lines of C++ for a one-line idea. For purely structural DAG-to-DAG rewrites, MLIR offers **Declarative Rewrite Rules** (DRR), which section 3.7 previewed on that same pattern. This section uses DRR for three optimizations of `toy.reshape` and explains its syntax in full.

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

The comment lists `Pattern`'s five parameters, in order: the source DAG, the result DAGs, multi-entity constraints, supplemental patterns (run after the result patterns, not used for replacement), and a benefit delta added to the default benefit (the number of ops in the source pattern). None of this chapter's patterns set the last two. `Pat<src, result>` is the common single-result shorthand for `Pattern<src, [result]>`. (`include "toy/Ops.td"` resolves because the chapter's `include/` directory is on `mlir-tblgen`'s include path — see appendix D.)

The three reshape patterns follow, from simplest to most involved: the basic `ReshapeReshapeOptPattern` (section 4.2); `RedundantReshapeOptPattern`, guarded by the constraint `TypesAreIdentical` (section 4.3); and `FoldConstantReshapeOptPattern`, which calls the `NativeCodeCall` `ReshapeConstant` (section 4.4). In `ToyCombine.td` itself, the `NativeCodeCall` pattern comes second.

### 4.2 Basic pattern: `Reshape(Reshape(x)) = Reshape(x)`

A redundant-reshape optimization of the same shape as the transpose pattern:

***mlir/ToyCombine.td***
```tablegen
// Reshape(Reshape(x)) = Reshape(x)
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;
```

Read the source DAG inside-out: match a `ReshapeOp` whose operand is produced by another `ReshapeOp`, binding the *inner* reshape's operand to `$arg`. The result DAG builds a single new `ReshapeOp` directly on `$arg` and replaces the matched root with it. In DRR, the result op's type is taken from the root op being replaced, so the new reshape produces the outer reshape's type. The intermediate shape doesn't matter: a reshape only reinterprets the same element data.

That's the same kind of two-op match as the transpose pattern of section 3.7, and just as short.

### 4.3 Pattern with constraints: redundant reshape

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
- **`(replaceWithValue $arg)`** — a special DRR directive: instead of building a new op, replace all uses of the matched root with an *existing* value. This is the DRR analogue of `rewriter.replaceOp(op, existingValue)`; section 3.7 used it for the transpose pattern.
- The third `Pat` argument is the **constraint list**, applied on top of structural matching.

Its source DAG has only one op, so it gets benefit 1 (section 4.1) and loses to the two-op patterns whenever both match; with this repo's two tests it never ends up firing (appendix A). It still matters for reshapes whose operand is neither a constant nor another reshape — e.g. a same-type reshape of a function argument.

### 4.4 Pattern with `NativeCodeCall`: `Reshape(Constant(x)) = x'`

Some optimizations need extra transformations on the op's arguments. `NativeCodeCall` provides them by calling into a C++ helper function or embedding inline C++; its `$0, $1, ...` placeholders are substituted with bound entities. `FoldConstantReshapeOptPattern` reshapes a constant in place and eliminates the reshape:

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

**Hooking the generated patterns in.** The build generates one pattern class per record into `ToyCombine.inc` (section 4.6). `ToyCombine.cpp` includes that file inside an anonymous namespace and registers the three classes as `ReshapeOp` canonicalization patterns (abridged):

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

From the canonicalizer's point of view there is no difference between hand-written and DRR-generated patterns — they are all `RewritePattern` instances in one `RewritePatternSet`. The order in `results.add<...>` does not decide which pattern wins; benefit does (section 4.5).

### 4.5 The reshape patterns on `trivial_reshape.toy`

The three patterns work together on the second test input, `trivial_reshape.toy` (abridged: its `# RUN`/`# CHECK` lines are elided):

***test_Example/Toy/Ch3/trivial_reshape.toy***
```text
# RUN: toyc-ch3 %s -emit=mlir -opt 2>&1 | FileCheck %s

def main() {
  var a<2,1> = [1, 2];
  var b<2,1> = a;
  var c<2,1> = b;
  print(c);
}
...
```

Each `var x<2,1> = ...` declaration forces a reshape to `tensor<2x1xf64>`. Without `-opt`, MLIRGen emits a chain of three reshapes off one constant:

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/trivial_reshape.toy -emit=mlir 2>&1
```

Real output **[10]**:

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

Real output **[11]**:

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>
    toy.print %0 : tensor<2x1xf64>
    toy.return
  }
}
```

No reshape operations remain after canonicalization, *and* the constant changed shape (`dense<[1.0, 2.0]> : tensor<2xf64>` became `dense<[[1.0], [2.0]]> : tensor<2x1xf64>`). The rest of this section shows how the patterns of sections 4.2–4.4 got there.

#### One rewrite at a time

The canonicalizer can be stopped early. A temporary copy of the chapter gets a `toyc.cpp` whose canonicalizer takes a `GreedyRewriteConfig`: `maxNumRewrites` comes from the `MAX_REWRITES` environment variable, and `maxIterations` is 1 (why is explained below). Running the copy with a limit of 0 to 4 rewrites shows the IR after each step:

```bash
cd /Users/roy/study/mlir/toy/Ch3
tmp=$(mktemp -d)
cp -R CMakeLists.txt toyc.cpp parser mlir include "$tmp"/
sed -i '' 's|mlir::createCanonicalizerPass()|mlir::createCanonicalizerPass([] { mlir::GreedyRewriteConfig config; config.maxIterations = 1; config.maxNumRewrites = std::atoi(std::getenv("MAX_REWRITES")); return config; }())|' "$tmp"/toyc.cpp
cmake -S "$tmp" -B "$tmp"/build -G Ninja > /dev/null
cmake --build "$tmp"/build > /dev/null 2>&1
for n in 0 1 2 3 4; do
  echo "== at most $n rewrites"
  MAX_REWRITES=$n "$tmp"/build/toyc-ch3 ../../test_Example/Toy/Ch3/trivial_reshape.toy -emit=mlir -opt -mlir-print-debuginfo 2>&1 \
    | grep -E 'toy\.(constant|reshape|print)' | sed 's|"../../test_Example/Toy/Ch3/trivial_reshape.toy"||g'
done
rm -r "$tmp"
```

Real output **[12]** (abridged: only the constant, reshape and print lines are kept, and the last `sed` cuts the file name from each location, leaving `loc(:line:col)`):

```text
== at most 0 rewrites
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64> loc(:4:16)
    %1 = toy.reshape(%0 : tensor<2xf64>) to tensor<2x1xf64> loc(:4:3)
    %2 = toy.reshape(%1 : tensor<2x1xf64>) to tensor<2x1xf64> loc(:5:3)
    %3 = toy.reshape(%2 : tensor<2x1xf64>) to tensor<2x1xf64> loc(:6:3)
    toy.print %3 : tensor<2x1xf64> loc(:7:3)
== at most 1 rewrites
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64> loc(:4:16)
    %1 = toy.reshape(%0 : tensor<2xf64>) to tensor<2x1xf64> loc(:4:3)
    %2 = toy.reshape(%1 : tensor<2x1xf64>) to tensor<2x1xf64> loc(fused[:6:3, :5:3])
    toy.print %2 : tensor<2x1xf64> loc(:7:3)
== at most 2 rewrites
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64> loc(:4:16)
    %1 = toy.reshape(%0 : tensor<2xf64>) to tensor<2x1xf64> loc(fused[:6:3, :5:3, :4:3])
    toy.print %1 : tensor<2x1xf64> loc(:7:3)
== at most 3 rewrites
    %0 = toy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64> loc(fused[:6:3, :5:3, :4:3, :4:16])
    toy.print %0 : tensor<2x1xf64> loc(:7:3)
== at most 4 rewrites
    %0 = toy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64> loc(fused[:6:3, :5:3, :4:3, :4:16])
    toy.print %0 : tensor<2x1xf64> loc(:7:3)
```

The fused locations name the ops each rewrite matched (a DRR rewrite fuses the locations of all matched ops, root first, as bullet (12) of section 3.7 explains), so each step can be read off the output:

1. **Start: 0 rewrites.** The driver gathers the patterns of every op type in the context: the three reshape patterns from `ReshapeOp`'s hook and the transpose pattern from `TransposeOp`'s. It puts the function's ops on a **worklist**. With a limit of 0 it stops at once, so this is MLIRGen's output, as in the run without `-opt` above.
2. **Rewrite 1.** The driver takes an op from the worklist. It would first try to *fold* it with the op's `fold` hook, but no Toy op has one in this chapter. Then it tries the patterns rooted at that op, **highest benefit first**. It began with the last reshape (`6:3`), a reshape of a reshape with identical input and output types. So both `ReshapeReshapeOptPattern` (benefit 2) and `RedundantReshapeOptPattern` (benefit 1) match, and the higher benefit wins. The new reshape replaces `%3` and is located at the two reshapes it matched, `fused[:6:3, :5:3]`. The middle reshape (`5:3`) now has no users. It is `Pure`, so the driver erases it as dead, which doesn't count as a rewrite.
3. **Rewrite 2.** After a rewrite, the driver puts the ops around the change back on the worklist, so the new reshape is examined again. Its operand is still a reshape (`4:3`), so `ReshapeReshapeOptPattern` fires again: one reshape straight from the constant, `fused[:6:3, :5:3, :4:3]`.
4. **Rewrite 3.** That reshape changes the type, `tensor<2xf64>` to `tensor<2x1xf64>`, so `RedundantReshapeOptPattern` no longer matches, but its operand is a constant. `FoldConstantReshapeOptPattern` replaces it with a constant in the new shape, adding the constant's own location `4:16`. The old constant is dead and erased.
5. **Fixpoint.** A limit of 4 gives the same IR as 3: no pattern matches any op anymore, and the worklist runs empty. That is the "greedy" in the name: apply whatever matches until nothing does. Small local patterns compose into a big cleanup because each rewrite can expose the next one.

**Why `maxIterations = 1`.** An *iteration* is one run through the worklist until it is empty, and `maxNumRewrites` limits the rewrites within one iteration. With the default of up to 10 iterations, the driver would start a new iteration after hitting the limit and finish the job anyway. All three rewrites happen in the first iteration, because the ops put back on the worklist are processed in that same iteration. By default, after an iteration that changed something, the driver runs another one to confirm that nothing changes.

Final IR: one constant, one print, no runtime reshaping work at all. Appendix A reads the same fused locations from the final IR.

### 4.6 What TableGen generates

The generated C++ for each DRR pattern goes into `ToyCombine.inc`: the build runs `mlir-tblgen -gen-rewriters` over `ToyCombine.td` and writes `build/ToyCombine.inc`. It contains one `RewritePattern` subclass per `Pat` record. The same generator, run by hand, prints the pattern for `FoldConstantReshapeOptPattern`:

```bash
cd /Users/roy/study/mlir/toy/Ch3
mlir-tblgen -gen-rewriters mlir/ToyCombine.td -I include -I /opt/homebrew/opt/llvm@20/include \
  | awk '/^struct FoldConstantReshapeOptPattern /,/^};/'
```

Real output **[13]** (abridged: the failure diagnostics after the first and the attribute/type plumbing of the rewrite are elided; `// (N)` markers added for the list below):

```cpp
struct FoldConstantReshapeOptPattern : public ::mlir::RewritePattern {
  FoldConstantReshapeOptPattern(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("toy.reshape", 2, context, {"toy.constant"}) {}                                               // (1)
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
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();                                                    // (2)
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {                                        // (3)
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::toy::ConstantOp>(op1); (void)castedOp1;                                      // (4)
      if (!(castedOp1)){
        ...
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::DenseElementsAttr>("value");(void)tblgen_attr;                         // (5)
        if (!(tblgen_attr)){
          ...
        }
        arg = tblgen_attr;
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;                    // (6)
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = arg.reshape(::llvm::cast<ShapedType>((*res.getODSResults(0).begin()).getType())); (void)nativeVar_0;  // (7)
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

The output's size and all three constructors:

```bash
cd /Users/roy/study/mlir/toy/Ch3
mlir-tblgen -gen-rewriters mlir/ToyCombine.td -I include -I /opt/homebrew/opt/llvm@20/include | wc -l
mlir-tblgen -gen-rewriters mlir/ToyCombine.td -I include -I /opt/homebrew/opt/llvm@20/include | grep -B2 'RewritePattern("toy'
```

Real output **[14]**:

```text
     176
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

That is 176 lines of C++ from about fifteen lines of TableGen definitions. Things worth noticing in the generated code; the numbers are the markers in the `FoldConstantReshapeOptPattern` listing:

- **(1)** Each constructor passes the root op name `"toy.reshape"`, a **benefit**, and the list of ops the rewrite can create. The benefits are the node counts of section 4.1: **2** for the two-op `FoldConstantReshapeOptPattern` and `ReshapeReshapeOptPattern`, **1** for `RedundantReshapeOptPattern`. The lists are `{"toy.constant"}`, `{}` and `{"toy.reshape"}`: unlike section 3.7's transpose pattern, two of these rewrites create an op.
- **(2), (4), (3)** The match is the same one-node-at-a-time check as in section 3.7, with a `notifyMatchFailure` message for each way it can fail.
- **(5)** New here: `$arg` binds `ConstantOp`'s `value` attribute, so the match also reads that attribute with `getAttrOfType<DenseElementsAttr>` and fails if it is missing.
- **(6)** `odsLoc`, unused in section 3.7, is used here: the new constant is created at the fused location of both matched ops. This is what section 4.5's trace and appendix A read back.
- **(7)** The `NativeCodeCall` string appears verbatim, with `$0`→`arg` and `$1`→`res`'s result substituted (`nativeVar_0`).
- At the bottom, TableGen also emits `void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(::mlir::RewritePatternSet &patterns)`, which registers all three patterns at once — an alternative registration entry point we don't use here.

More on the declarative rewrite method is in the DRR reference under Links.

These are core transformations through always-available hooks (`getCanonicalizationPatterns`). Chapter 4 adds transformations that work across dialects, inlining and shape inference, through interfaces.

---

## 5. Build and Run

The superbuild (`toy/build.sh`, `toy/run.sh`, `CMakePresets.json`) is documented once in the top-level [README](../README.md#the-build-system). This section builds and runs Chapter 3 on its own, from the chapter directory. What the chapter's `CMakeLists.txt` adds — a second TableGen rule for the DRR patterns — is in appendix D.

Every command in this README is numbered by its output (**[1]**–**[26]**). After building, `./examples.sh` runs them all, printing each command followed by its output and exit status; `./examples.sh 2` runs only command [2], and `./examples.sh -l` lists them (appendix E).

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

Before compiling, `mlir-tblgen` now also runs `-gen-rewriters` on `mlir/ToyCombine.td` (section 4.6; appendix D has the build rule). A standalone build puts the binary directly in `build/`.

### 5.2 Running

The first test input, whole:

***test_Example/Toy/Ch3/transpose_transpose.toy***
```text
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

Its `# CHECK` lines describe the expected `-opt` output (appendix B.2 runs them). Without `-opt`, the driver dumps MLIRGen's output:

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/transpose_transpose.toy -emit=mlir 2>&1
```

Real output **[15]**:

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

With `-opt`, the canonicalizer runs first:

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt 2>&1
```

Real output **[16]**:

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

- The positional argument is the input file (default `-`, stdin).
- `-emit=mlir` — dump the module after MLIRGen (`-emit=ast` still works).
- `-opt` — **new in this chapter**: run the canonicalizer pipeline before dumping (section 3.4). Without it you get the raw MLIRGen output, which is how the "before" dump above is produced.
- `-mlir-print-debuginfo` — print `loc(...)` on every op; used in appendix A to see which patterns fired.
- `-x mlir` (or an input ending in `.mlir`) — parse the input as MLIR instead of Toy source; `-opt` works on either path.
- Generic pass-manager flags such as `--mlir-print-ir-before-all` / `--mlir-print-ir-after-all` are also accepted, because `main()` registers them and `dumpMLIR()` applies them (section 3.4; used in appendix B.1).
- `2>&1` — `module->dump()` writes to **stderr**.

What fired, and why:

- **In `@transpose_transpose`:** `SimplifyRedundantTranspose` (the C++ pattern, section 3.2) matched the outer `%1 = toy.transpose(%0)` because `%0`'s defining op is itself a `toy.transpose`. `rewriter.replaceOp` rewired `toy.return` to use `%arg0` directly. The now-unused inner `%0 = toy.transpose(%arg0)` was then erased by the canonicalizer's DCE — legal only because `TransposeOp` is `Pure` (section 3.6). Result: the whole double-transpose reduced to `toy.return %arg0`.
- **In `@main`:** `%1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>` is a reshape of a constant *and* a reshape whose input and output types are identical, so two DRR patterns (section 4) match it: `FoldConstantReshapeOptPattern` (benefit 2) and `RedundantReshapeOptPattern` (benefit 1). The higher-benefit one wins: the reshape was folded into a new `toy.constant` of type `tensor<2x3xf64>`, and the old constant became dead. Both routes print the same IR; appendix A shows the location evidence for which one actually ran.
- **What did *not* happen:** the call to `@transpose_transpose` is still there. The canonicalizer performs local, intra-function rewrites; it does not inline or interprocedurally propagate. Chapter 4 (inlining + shape inference) is what finally lets `main` see through the call.

The second test input, `trivial_reshape.toy`, is run before and after `-opt` in section 4.5.

---

## 6. Key Takeaways & Pitfalls

**Takeaways**

- **Optimize at the right abstraction level.** `transpose(transpose(x)) → x` is a two-op pattern match in the Toy dialect and a near-impossible loop-nest analysis in LLVM IR. This asymmetry is the whole point of building high-level dialects.
- **Two pattern authoring styles, one engine.** C++ `OpRewritePattern` for arbitrary logic; DRR for concise structural rewrites (with `Constraint`/`CPred` for conditions and `NativeCodeCall` for embedded C++). DRR computes the benefit, fuses locations and writes failure diagnostics for you; in C++ you write them yourself (section 3.7). Both end up as `RewritePattern`s in the same `RewritePatternSet` consumed by the same greedy driver.
- **Canonicalization is the cheap distribution channel.** `let hasCanonicalizer = 1` in ODS + a `getCanonicalizationPatterns` definition means *every* user of `createCanonicalizerPass()` gets your patterns for free — no custom pass required.
- **Small patterns compose via fixpoint iteration.** Nobody wrote "fold a chain of three reshapes into a reshaped constant"; independent one-step patterns plus DCE reached that result together, one rewrite at a time (section 4.5).
- **Traits combine.** `Pure` lets an unused op be deleted, and `Terminator` keeps `toy.return`, which is `Pure` and has no results, from ever being deleted (section 3.6).
- **Locations are preserved.** DRR-generated code fuses the source ops' locations into the replacement, keeping diagnostics meaningful after transformation — and, as appendix A shows, making it possible to tell which pattern fired.

**Pitfalls**

- **Patterns don't clean up after themselves — DCE does.** `SimplifyRedundantTranspose` leaves the inner transpose dead. Don't build erasure of *upstream* producers into your pattern (they may have other users!); instead make ops declare their effects and let the canonicalizer's DCE remove what becomes dead. Corollary: **forgetting `Pure` (or a proper `MemoryEffects` interface) silently blocks DCE** — you'll see dead ops persist with no error anywhere (section 3.5).
- **Forgetting `let hasCanonicalizer = 1`** means the generated op class has no `getCanonicalizationPatterns` declaration; your `.cpp` definition then fails to compile. Without a definition either, the op keeps the inherited empty hook and contributes no patterns (section 3.3).
- **Only mutate through the `PatternRewriter`.** Direct IR surgery inside `matchAndRewrite` breaks the driver's worklist tracking and can crash or miscompile. Also: decide match *before* mutating — a pattern that mutates then returns `failure()` is broken.
- **Rewrites must keep consumers legal.** Replacing a value changes the operand of its users (`toy.return`, `toy.generic_call`, ...). `SimplifyRedundantTranspose` is safe because the replacement has the same type as the value it replaces (section 3.6); a pattern that substitutes a value of another type must make sure every user still verifies.
- **Don't create infinite ping-pong.** The greedy driver iterates to fixpoint; a pair of patterns like `A→B` and `B→A` (or a pattern that "rewrites" an op to an identical op) never converges and hits the iteration limit. Each rewrite should strictly reduce some measure (op count, pattern-match opportunities).
- **Op ordering / traversal is not yours to control.** Patterns fire in benefit order per op, over a worklist in unspecified overall order. Never write a pattern whose correctness depends on another pattern having already run — each must be independently sound; only *convergence* composes them. When several patterns match the same op, benefit decides, not the order in `results.add<...>` (section 4.4): in section 5.2 the benefit-2 constant fold, not the redundant-reshape pattern, removed `main`'s reshape, as appendix A shows.
- **Canonicalization is local.** It didn't remove the `toy.generic_call` round-trip in `main` — interprocedural wins need the inliner and shape inference, which is exactly where [Chapter 4](../Ch4/README.md) picks up.
- **DRR `$arg` on a `ConstantOp` binds the attribute, not a value.** Because ODS declares `ConstantOp`'s input as an attribute, DRR captures a `DenseElementsAttr` — which is why `ReshapeConstant` can call `.reshape(...)` on it directly (section 4.4). An operand argument binds a `Value` and an attribute argument binds the attribute; mixing them up is a classic DRR stumble.

---

## Appendix A: Reading the locations — which pattern fired

The release build of Homebrew LLVM has no `-debug` output, but the locations tell the story. DRR-generated rewrites give the ops they create a **fused location** of all matched ops (section 4.6), while `replaceWithValue` just forwards an existing value and keeps *its* location. Rerun the first test with locations: `-mlir-print-debuginfo`:

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt -mlir-print-debuginfo 2>&1
```

Real output **[17]**:

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

The surviving constant in `@main` sits at `fused[9:3, 9:17]`: `9:3` is the `var a<2, 3>` declaration (the reshape's location) and `9:17` is the literal (the original constant's location). That is a constant *created* by `FoldConstantReshapeOptPattern`, not the original constant forwarded by `RedundantReshapeOptPattern` — which would have kept plain `9:17`. The benefit-2 pattern won, by the same rule as in section 4.5's step 2, where `ReshapeReshapeOptPattern` beat `RedundantReshapeOptPattern`.

For `trivial_reshape.toy`, section 4.5's step-by-step run shows the same evidence for each rewrite: the final constant carries all four source locations, the three reshapes and the literal, so every reshape was absorbed by a fusing pattern and `RedundantReshapeOptPattern` never fired there either.

---

## Appendix B: More runs

### B.1 Watching the pipeline

Because `main()` registers the pass-manager command-line options and `dumpMLIR` calls `applyPassManagerCLOptions` (section 3.4), you can print the IR around each pass. The canonicalizer is nested on `toy.func`, so the dumps are per function; `transpose_transpose.toy` has two:

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt \
    --mlir-print-ir-before-all --mlir-print-ir-after-all 2>&1
```

Real output **[18]** (abridged: the final module dump, the same as the `-opt` run of section 5.2, is left out):

```mlir
// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
  toy.return %1 : tensor<*xf64>
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
  %2 = toy.generic_call @transpose_transpose(%1) : (tensor<2x3xf64>) -> tensor<*xf64>
  toy.print %2 : tensor<*xf64>
  toy.return
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  toy.return %arg0 : tensor<*xf64>
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.generic_call @transpose_transpose(%0) : (tensor<2x3xf64>) -> tensor<*xf64>
  toy.print %1 : tensor<*xf64>
  toy.return
}
```

- **One dump per function, before and after.** The pass manager runs the canonicalizer separately on each `toy.func` (section 3.4), so each function gets its own pair of dumps.
- **Both "Before" dumps come first.** The functions are processed independently, so the pass manager can start the canonicalizer on both before finishing either. The order was the same in every run here, but nothing guarantees it.
- **`@transpose_transpose`** loses both transposes (section 3.6), and **`@main`** loses its reshape to the constant fold (appendix A). The call between them stays: canonicalization doesn't look across functions.

### B.2 Running the `# CHECK` lines

The `.toy` files carry `# RUN:` / `# CHECK:` lines for LLVM `lit`/`FileCheck`. There is no lit harness in this repo, but Homebrew LLVM ships `FileCheck`, so you can run a test's RUN line by hand; silence and exit status 0 mean the checks pass, so the command echoes a message only if both do:

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt 2>&1 \
  | FileCheck ../../test_Example/Toy/Ch3/transpose_transpose.toy && \
./build/toyc-ch3 ../../test_Example/Toy/Ch3/trivial_reshape.toy -emit=mlir -opt 2>&1 \
  | FileCheck ../../test_Example/Toy/Ch3/trivial_reshape.toy && echo "both checks pass"
```

Real output **[19]**:

```text
both checks pass
```

### B.3 Splitting frontend and optimizer

`-opt` is nothing Toy-specific — it just adds MLIR's *generic* `createCanonicalizerPass()` to the pass manager (section 3.4). You can make the stage boundary visible by splitting frontend and optimizer into two processes connected by textual IR, exactly how `mlir-opt`-based pipelines compose. (`toyc-ch3` plays the role of the project's own `foo-opt`, since stock `mlir-opt` doesn't link the Toy dialect; see Ch2 appendix C.2.) The two processes, connected by a pipe:

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/trivial_reshape.toy -emit=mlir 2>&1 \
  | ./build/toyc-ch3 - -x mlir -emit=mlir -opt 2>&1
```

Real output **[20]**:

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

### B.4 The error path

Invalid MLIR input fails in the parser/verifier, before any pass runs:

```bash
cd /Users/roy/study/mlir/toy/Ch3
./build/toyc-ch3 ../../test_Example/Toy/Ch3/invalid.mlir -emit=mlir -opt
```

Real output **[21]** (exit status 3):

```
../../test_Example/Toy/Ch3/invalid.mlir:8:8: error: 'toy.print' op requires zero results
  %0 = "toy.print"()  : () -> tensor<2x3xf64>
       ^
../../test_Example/Toy/Ch3/invalid.mlir:8:8: note: see current operation: %0 = "toy.print"() : () -> tensor<2x3xf64>
Error can't load file ../../test_Example/Toy/Ch3/invalid.mlir
```

The diagnostic is rendered with the source line and caret by the `SourceMgrDiagnosticHandler` in `dumpMLIR()`, and exit status 3 comes from `loadMLIR()` (section 3.4).

---

## Appendix C: Every field of an ODS op

Section 3.3 introduced `hasCanonicalizer`. This appendix lists every field of ODS's `Op` class in MLIR 20: the fields an op sets with `let`. Chapter 2 explains how the Toy ops use them, and its appendix B.3 lists the ones Toy's `Ops.td` sets. The dialect has its own record, with fields such as `name` and `cppNamespace` (Ch2 section 3.2).

### C.1 The fields and their defaults

An op that sets nothing shows every field with its default. The input isn't a repo file, so it goes on stdin:

```bash
cd /Users/roy/study/mlir/toy/Ch3
mlir-tblgen --print-records -I /opt/homebrew/opt/llvm@20/include <<'EOF' | awk '/^def EmptyOp /,/^}/'
include "mlir/IR/OpBase.td"
def Demo_Dialect : Dialect { let name = "demo"; }
def EmptyOp : Op<Demo_Dialect, "empty">;
EOF
```

Real output **[22]**:

```text
def EmptyOp {	// Op
  Dialect opDialect = Demo_Dialect;
  string opName = "empty";
  string cppNamespace = "demo";
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

Twenty-three fields, `?` meaning unset. Grouped by purpose:

| Field | What it does | In Toy |
|---|---|---|
| **Identity** | | |
| `opDialect`, `opName` | The `Op<dialect, mnemonic>` template arguments; `getOperationName()` returns `"<dialect name>.<mnemonic>"` | every op, through `Toy_Op` (Ch2 4.3) |
| `cppNamespace` | The C++ namespace of the op class, copied from the dialect | `::mlir::toy` |
| **Documentation** | | |
| `summary`, `description` | A one-line summary and a long description, for `-gen-op-doc` and the class comment | every op (Ch2 4.4, step 2) |
| `opDocGroup` | Groups ops under one heading in the generated documentation | not used |
| **Structure** | | |
| `arguments` | `(ins ...)`: operands, attributes and properties. Named entries get accessors, and their constraints get checks | every op (Ch2 4.4, step 1) |
| `results` | `(outs ...)`: the results, with accessors, type checks and the result-count trait | every op with a result |
| `regions` | `(region ...)`: regions such as `AnyRegion:$body`, with accessors and the region-count trait | `FuncOp` (Ch2 A.3) |
| `successors` | `(successor ...)`: the blocks a terminator can branch to, as in a branch op | not used |
| `traits` | The trait list, normally given as `Op`'s third template argument | `Pure` and others (section 3.6) |
| **Builders** | | |
| `builders` | Extra `build()` overloads, each with an optional inline body | `ConstantOp` and others (Ch2 4.4, step 4) |
| `skipDefaultBuilders` | Don't generate the default `build()` overloads; the op must then declare at least one in `builders` (C.2) | `FuncOp` (Ch2 A.3) |
| **Syntax** | | |
| `assemblyFormat` | A declarative format; ODS generates `parse()` and `print()` | `PrintOp` and others (Ch2 4.5) |
| `hasCustomAssemblyFormat` | Declares `parse()` and `print()` for you to define; ignored when `assemblyFormat` is set | `ConstantOp`, `AddOp`, `MulOp`, `FuncOp` |
| **Verification** | | |
| `hasVerifier` | Declares `verify()`, run after the ODS-generated checks | `ConstantOp`, `ReturnOp`, `TransposeOp` (Ch2 4.4, step 3) |
| `hasRegionVerifier` | Declares `verifyRegions()`, run after the ops nested in the op's regions are verified | not used |
| **Canonicalization** | | |
| `hasCanonicalizer` | Declares `getCanonicalizationPatterns()` | `ReshapeOp`, `TransposeOp` (section 3.3) |
| `hasCanonicalizeMethod` | Declares a static `canonicalize(op, rewriter)` pattern. Without `hasCanonicalizer`, ODS also generates `getCanonicalizationPatterns()`, registering it (C.2) | not used |
| `hasFolder` | Declares `fold()`, which the canonicalizer's driver tries before any pattern (section 4.5) and `builder.createOrFold` calls | not used |
| **Serialization** | | |
| `useCustomPropertiesEncoding` | Keeps the declarations of `readProperties()`/`writeProperties()`, which store properties in MLIR's bytecode, but stops generating their definitions, so you write them (C.2) | not used |
| **Escape hatches** | | |
| `extraClassDeclaration` | C++ pasted into the op class declaration | `FuncOp`, `ReturnOp` (Ch2 A.3, A.7) |
| `extraClassDefinition` | C++ pasted into the generated definitions, with `$cppClass` standing for the class name | not used |

### C.2 What the method-declaring fields generate

Besides `hasCanonicalizer` (section 3.3), seven fields make ODS declare C++ methods that the op's author defines, or change which ones are generated. An op with all of them set, plus an attribute argument and a result (so that `fold()` and the property methods take their usual form), shows the declarations; then `-gen-op-defs` shows what ODS defines itself:

```bash
cd /Users/roy/study/mlir/toy/Ch3
td() {
  cat <<'EOF'
include "mlir/IR/OpBase.td"
def Demo_Dialect : Dialect { let name = "demo"; }
def AllBitsOp : Op<Demo_Dialect, "all_bits"> {
  let arguments = (ins F64Attr:$value);
  let results = (outs F64);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins)>];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
  let hasCanonicalizeMethod = 1;
  let hasFolder = 1;
  let useCustomPropertiesEncoding = 1;
}
EOF
}
echo "== declarations"
td | mlir-tblgen -gen-op-decls -I /opt/homebrew/opt/llvm@20/include | awk '/^class AllBitsOp : /,/^};/' \
  | grep -E 'static void build|parse\(|print\(::mlir::OpAsmPrinter|verify\(\)|verifyRegions|canonicalize\(|getCanonicalizationPatterns|fold\(|(read|write)Properties'
echo "== definitions ODS generates"
td | mlir-tblgen -gen-op-defs -I /opt/homebrew/opt/llvm@20/include \
  | grep -E -A2 '^void AllBitsOp::getCanonicalizationPatterns|AllBitsOp::(read|write)Properties\('
```

Real output **[23]**:

```text
== declarations
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::llvm::LogicalResult verify();
  ::llvm::LogicalResult verifyRegions();
  static ::llvm::LogicalResult canonicalize(AllBitsOp op, ::mlir::PatternRewriter &rewriter);
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
== definitions ODS generates
void AllBitsOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.add(canonicalize);
}
```

- **Each `has...` bit adds its declarations**: `parse()`/`print()`, `verify()`, `verifyRegions()`, `canonicalize()`, `fold()`. Their definitions are left to the op's author, in Toy's case in `Dialect.cpp` and `ToyCombine.cpp`.
- **`skipDefaultBuilders`** leaves only the `build()` declared in `builders`; the default overloads that `arguments` and `results` would produce are gone.
- **`hasCanonicalizeMethod`** also declares `getCanonicalizationPatterns()`, and the definitions show its generated body: `results.add(canonicalize);`. The one method becomes a pattern without any registration code.
- **`useCustomPropertiesEncoding`** keeps the `readProperties()`/`writeProperties()` declarations, but no definitions for them appear: they are left to you. Without the field, ODS defines both.
- **`fold()`** takes a `FoldAdaptor` and returns one `OpFoldResult` because the op has one result. An op with several results, or none, gets `fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &)` instead.

`skipDefaultBuilders` without any custom builder is an error, since the op could not be built at all:

```bash
cd /Users/roy/study/mlir/toy/Ch3
mlir-tblgen -gen-op-decls -I /opt/homebrew/opt/llvm@20/include > /dev/null <<'EOF'
include "mlir/IR/OpBase.td"
def Demo_Dialect : Dialect { let name = "demo"; }
def NoBuilderOp : Op<Demo_Dialect, "no_builder"> {
  let skipDefaultBuilders = 1;
}
EOF
```

Real output **[24]** (exit status 1):

```text
<stdin>:3:5: error: default builders are skipped and no custom builders provided
def NoBuilderOp : Op<Demo_Dialect, "no_builder"> {
    ^
```

---

## Appendix D: What Chapter 3 adds to the build

The chapter keeps Chapter 2's structure — `include/toy/CMakeLists.txt` generating the op/dialect `.inc` files into target `ToyCh3OpsIncGen`, plus the explicit `add_dependencies` race guard (see Ch2 appendix E) — and adds a second TableGen flavor for the DRR patterns.

### D.1 The chapter targets

Below the standalone guard:

***CMakeLists.txt***
```cmake
include_directories(include)                      # (1)
add_subdirectory(include)

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)      # (2)
add_public_tablegen_target(ToyCh3CombineIncGen)

add_executable(toyc-ch3
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/ToyCombine.cpp                             # (3)
  )

add_dependencies(toyc-ch3 ToyCh3OpsIncGen)
add_dependencies(toyc-ch3 ToyCh3CombineIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})  # (4)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

target_link_libraries(toyc-ch3
  PRIVATE
    MLIR                      # libMLIR.dylib (all dialects, passes, conversions)
    LLVM                      # libLLVM.dylib (all targets, all components)
    )
```

Relative to Chapter 2; the numbers match the `# (N)` markers in the excerpt:

1. **`include_directories(include)` moved above the TableGen rule.** `mlir_tablegen` passes the directory's include paths to `mlir-tblgen` as `-I` flags, and `ToyCombine.td` needs `include "toy/Ops.td"` to resolve to `Ch3/include/toy/Ops.td`. The generated build rule, shown after this list, passes it.
2. **The rewriter TableGen step** — a new kind of generator for the DRR patterns of section 4. `mlir_tablegen(ToyCombine.inc -gen-rewriters)` invokes `mlir-tblgen -gen-rewriters` to emit `ToyCombine.inc` into the chapter's binary directory — `Ch3/build/ToyCombine.inc` standalone (`toy/build/Ch3/ToyCombine.inc` in the superbuild). `add_public_tablegen_target` wraps it in the named target `ToyCh3CombineIncGen`, and `add_dependencies(toyc-ch3 ToyCh3CombineIncGen)` guarantees the file exists before `ToyCombine.cpp` compiles. This sits alongside the Chapter-2 op generation (`ToyCh3OpsIncGen`).
3. **New source file** `mlir/ToyCombine.cpp` in the executable. The alternative DRR transpose pattern in `mlir/alt/drr_transpose/` is not part of the build; section 3.7 builds it in a temporary copy.
4. **Include path for generated files:** `include_directories(${CMAKE_CURRENT_BINARY_DIR})` is what lets `#include "ToyCombine.inc"` resolve, since this `.inc` lands at the *root* of the chapter's binary directory (unlike the op/dialect `.inc` files under `include/toy/`). `CMAKE_CURRENT_BINARY_DIR` points to the right place in both superbuild and standalone modes.

The rule Ninja runs for `ToyCombine.inc`, from the generated `build/build.ninja`:

```bash
cd /Users/roy/study/mlir/toy/Ch3
grep -o 'mlir-tblgen -gen-rewriters[^&]*' build/build.ninja
```

Real output **[25]**:

```text
mlir-tblgen -gen-rewriters -I /Users/roy/study/mlir/toy/Ch3 -I/opt/homebrew/opt/llvm@20/include -I/opt/homebrew/opt/llvm@20/include -I/Users/roy/study/mlir/toy/Ch3/include /Users/roy/study/mlir/toy/Ch3/mlir/ToyCombine.td --write-if-changed -o ToyCombine.inc -d ToyCombine.inc.d 
```

The `-I .../Ch3/include` entry comes from `include_directories(include)` (item 1); it is what resolves `include "toy/Ops.td"`.

Linking is unchanged: the monolithic Homebrew `libMLIR.dylib` / `libLLVM.dylib`, which already contain the canonicalizer pass and greedy driver (in-tree builds would add `MLIRPass`, `MLIRTransforms`, ... as components instead).

---

## Appendix E: Running this README's commands: `examples.sh`

[`examples.sh`](examples.sh) — a symlink to the shared [`../examples.sh`](../examples.sh) — runs the commands of this README and shows each one with its output and exit status:

```bash
cd /Users/roy/study/mlir/toy/Ch3
./examples.sh -l
```

Real output **[26]** (abridged):

```text
[ 1] 3.1   ./build/toyc-ch3 ../../test_Example/Toy/Ch3/transpose_transpose.toy -emit=mlir 2>&1
[ 2] 3.1   /opt/homebrew/opt/llvm@20/bin/clang++ -O3 -S -emit-llvm -x c++ - -o - <<'EOF' \
...
[25] D.1   grep -o 'mlir-tblgen -gen-rewriters[^&]*' build/build.ninja
[26] E     ./examples.sh -l
```

The script reads the commands from `README.md` on every run. A command is every `bash` code block in file order, except the build block (its first command after `cd` is `cmake`) and file excerpts (a `***path***` label above the fence). Number *N* is the *N*-th such block, and the README marks that block's output "Real output **[N]**". So adding, removing or reordering a command block renumbers the ones after it, and their **[N]** labels must be updated to match.

---

## Links

- Official doc: [MLIR Toy Tutorial Ch.3 — High-level Language-Specific Analysis and Transformation](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/)
- DRR reference: [Table-driven Declarative Rewrite Rules](https://mlir.llvm.org/docs/DeclarativeRewrites/) · [Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/) · [Operation Canonicalization](https://mlir.llvm.org/docs/Canonicalization/)
- Previous: [Chapter 2 — Emitting Basic MLIR](../Ch2/README.md)
- Next: [Chapter 4 — Enabling Generic Transformation with Interfaces](../Ch4/README.md)
- Index: [README](../README.md)
