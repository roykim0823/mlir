# Chapter 5: Partial Lowering to Lower-Level Dialects for Optimization

> **Goal:** lower the *computationally heavy* Toy operations to a mix of the `affine`, `arith`, and `memref` dialects using MLIR's **dialect conversion framework** — while deliberately keeping `toy.print` in the Toy dialect — and then reuse existing affine optimizations (loop fusion, scalar replacement) on the result — based on the official tutorial [Toy Ch-5: Partial Lowering to Lower-Level Dialects for Optimization](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/).

---

## 1. Overview

At this point we are eager to generate actual code. Chapter 6 will use LLVM for that, but jumping straight to the LLVM builder interface would skip the more interesting idea: **progressive lowering** through a mix of dialects that coexist in the same function.

Up to Chapter 4 everything we did — inlining, shape inference, canonicalization — happened *inside* the Toy dialect. That is the right altitude for language-level semantics, but it is the wrong altitude for optimizations like **loop fusion**, **loop tiling**, or **redundant load elimination**. Those need explicit loops and explicit memory, which `toy.mul` and `toy.transpose` deliberately hide. The `affine` dialect already implements such optimizations, so we target it for the computation-heavy part of Toy. It is also deliberately limited: it cannot represent our `toy.print` builtin — nor should it. `toy.print` is lowered directly to the LLVM dialect in Chapter 6.

This chapter therefore performs a **partial lowering**: we translate only the compute-intensive Toy operations into lower-level dialects and leave the rest alone. This is possible because MLIR has *no fixed set of dialects* — **operations from different dialects can freely coexist in the same module, the same function, even the same block**. After this chapter's pass runs, our IR simultaneously contains:

- `func.func` / `func.return` — the structural function scaffolding (Func dialect),
- `affine.for` / `affine.load` / `affine.store` — polyhedral-friendly loop nests (Affine dialect),
- `arith.constant` / `arith.mulf` / `arith.addf` — scalar arithmetic (Arith dialect),
- `memref.alloc` / `memref.dealloc` — explicit buffers (MemRef dialect),
- **`toy.print`** — still a Toy op, now operating on a `memref` instead of a `tensor`.

Why keep `toy.print`? Because its "real" lowering needs an external `printf` call, which belongs at the LLVM level (Chapter 6). Lowering it now would gain nothing; keeping it abstract keeps the IR clean. This is the essence of *progressive lowering*: each pass moves the IR **only as far down as it needs to go** for the next set of transformations to apply.

What the target dialects buy us:

| Dialect | What it models | What it enables |
|---|---|---|
| `affine` | Loop nests with affine bounds/indices | Polyhedral analyses: fusion, tiling, dependence analysis, scalar replacement |
| `arith` | Scalar integer/FP arithmetic | Constant folding, CSE on scalars, direct mapping to machine ops later |
| `memref` | Explicitly allocated, shaped buffers | Explicit memory: alias analysis, buffer reuse, dealloc placement |
| `func` | Standard functions/calls/returns | Interop with every other MLIR pass and the eventual LLVM lowering |

The type story is important too. Toy ops work on the builtin [`RankedTensorType`](https://mlir.llvm.org/docs/Dialects/Builtin/#rankedtensortype): **value-semantics `tensor`s**, an abstract sequence of data that does not live in any memory. The lowered code works on [`MemRefType`](https://mlir.llvm.org/docs/Dialects/Builtin/#memreftype): **buffer-semantics `memref`s**, concrete references to a region of memory, indexed here by an affine loop nest. Part of this chapter is bridging that gap.

### How this README maps to the upstream chapter

Every topic of the official [Ch-5](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/) is covered here, in the same order, with upstream's snippets replaced by this repo's real code and output. Where the upstream text is out of date for MLIR 20, the section says so; each such point was checked by compiling, running, or grepping the Homebrew LLVM 20 headers.

| Upstream section | Here |
|---|---|
| Introduction (progressive lowering, `Affine`, `TensorType` → `MemRefType`) | 1 |
| Dialect Conversions (target, patterns, optional type converter) | 2 |
| Conversion Target | 2.1 |
| Conversion Patterns (`ConversionPattern`, `TransposeOpLowering`, the pattern list) | 3.1, 3.4, 3.9 |
| Partial Lowering | 4.1 |
| Design Considerations With Partial Lowering | 4.3 |
| Complete Toy Example | 6.3–6.5 |
| Taking Advantage of Affine Optimization | 5, 6.6–6.7 |

Beyond upstream: the helpers and the other five patterns (3.2–3.3, 3.5–3.8), each pattern run on its own small input (3.2–3.8), the IR right after the lowering pass and the failure paths (3.7, 4.1), the pass wrapper (4.2), the driver pipeline (5), FileCheck (6.7), more inputs and stock `mlir-opt` (6.8–6.9), and the build wiring (appendix).

### Where Chapter 5 lives in this repo

How this repo is organized — the out-of-tree CMake superbuild at `toy/`, the pinned Homebrew LLVM/MLIR 20 toolchain, and the `build.sh`/`run.sh` helpers — is documented once in the top-level [README](../README.md#repository-layout). This chapter also remains configurable as a standalone project (section 6.1). Chapter 5 specifics:

| Item | Location / value |
|---|---|
| Chapter 5 code | `/Users/roy/study/mlir/toy/Ch5/` |
| Build | `cd toy && ./build.sh ch5` → binary at `./build/bin/toyc-ch5` |
| Run | `cd toy && ./run.sh ch5` → `affine-lowering.mlir` with `-emit=mlir -opt`, `-emit=mlir-affine`, and `-emit=mlir-affine -opt` |
| Test inputs | `/Users/roy/study/mlir/test_Example/Toy/Ch5/` (`affine-lowering.mlir` is the one this chapter uses) |

Everything else (lexer, parser, AST, MLIRGen, the ODS dialect, the Ch3 combiner, the Ch4 shape-inference pass) carries over from Chapter 4. What changes:

- **new** `mlir/LowerToAffineLoops.cpp` — the conversion target, the conversion patterns, and the lowering pass (sections 2–4);
- `include/toy/Passes.h` — declares `createLowerToAffinePass()` (section 4.2);
- `include/toy/Ops.td` — `PrintOp` now also accepts a `memref` operand (section 4.3);
- `toyc.cpp` — a new `-emit=mlir-affine` action and the post-lowering pipeline (section 5);
- `CMakeLists.txt` — one new source file (appendix).

`mlir/LowerToAffineLoops.cpp` and `toyc.cpp` are identical to the upstream `release/20.x` example sources (`mlir/examples/toy/Ch5/`), and so is the test input `affine-lowering.mlir` (`mlir/test/Examples/Toy/Ch5/`); only the upstream *prose* lags behind them in places.

---

## 2. Dialect Conversions

MLIR has many dialects, so it needs one unified framework for [converting](https://mlir.llvm.org/getting_started/Glossary/#conversion) between them: the **DialectConversion framework** (`mlir/Transforms/DialectConversion.h`). It transforms a set of *illegal* operations into a set of *legal* ones.

Chapter 3's canonicalization used the *greedy* pattern driver: apply patterns until fixpoint, no notion of "done-ness". Lowering needs something stronger — a guarantee about **what the IR looks like when the pass finishes**. The conversion framework provides that guarantee. It needs three ingredients (the third is optional):

1. a [**ConversionTarget**](https://mlir.llvm.org/docs/DialectConversion/#conversion-target) — the formal specification of which operations or dialects are legal; operations that aren't legal need rewrite patterns to perform [legalization](https://mlir.llvm.org/getting_started/Glossary/#legalization) (section 2.1);
2. a set of **conversion patterns** — how to rewrite illegal ops into zero or more legal ones (section 3);
3. optionally a **TypeConverter** — used to convert the types of block arguments and function signatures. We don't need one: the only function left after inlining, `main`, has no arguments, and the patterns produce `memref` values themselves (section 3.2).

The three come together in the pass's `runOnOperation`, which sections 2.1, 3.9, and 4.1 show piece by piece, as upstream does.

### 2.1 Conversion Target

The target classifies every operation the conversion encounters as *legal* (may remain), *illegal* (must be converted away), or *dynamically legal* (legal only if a predicate holds). We want to convert the compute-intensive Toy operations into a combination of `affine`, `arith`, `func`, and `memref` operations, so everything in those dialects is a valid result; the whole Toy dialect must disappear — except `toy.print`, which is legal *if and only if* none of its operands is still a `TensorType`. The start of `runOnOperation` (abridged at the end):

***mlir/LowerToAffineLoops.cpp***
```cpp
void ToyToAffineLoweringPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `Arith`, `Func`, and `MemRef` dialects.
  target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                         arith::ArithDialect, func::FuncDialect,
                         memref::MemRefDialect>();

  // We also define the Toy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Toy operations that don't want
  // to lower, `toy.print`, as `legal`. `toy.print` will still need its operands
  // to be updated though (as we convert from TensorType to MemRefType), so we
  // only treat it as `legal` if its operands are legal.
  target.addIllegalDialect<toy::ToyDialect>();
  target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(),
                         [](Type type) { return llvm::isa<TensorType>(type); });
  });
  ...
}
```

Three things to internalize:

- **Op-level legality beats dialect-level legality.** `addDynamicallyLegalOp<toy::PrintOp>` wins over `addIllegalDialect<toy::ToyDialect>`, carving an exception out of the illegal Toy dialect. Because individual operations always take precedence over the more generic dialect entries, the order of the two calls doesn't matter. Upstream points to `ConversionTarget::getOpInfo` for the lookup; in MLIR 20 that is a private member (a call to it from outside fails to compile with "`'getOpInfo' is a private member of 'mlir::ConversionTarget'`"), so read it in the LLVM sources rather than call it. To check the order claim, a scratch copy of this chapter with `addIllegalDialect` moved *after* `addDynamicallyLegalOp` produced output byte-identical to sections 6.5 and 6.6.
- The **dynamic** predicate is what forces the framework to still *fire a pattern* on `toy.print`: right after its producers are lowered, `toy.print`'s operand is a `tensor`-typed value that the conversion has remapped to a `memref` — the op is "illegal until its operands are updated", so the `PrintOpLowering` pattern must run (section 3.8, which shows the before and after).
- If, at the end of conversion, any operation that is (still) illegal remains, the conversion **fails** and the pass signals failure. Legality is a hard postcondition, not a best-effort goal (section 4.1 shows the diagnostic).

Two differences from the upstream snippet:

- Upstream writes the predicate as `type.isa<TensorType>()`. In MLIR 20 the member-function casts on `Type` are deprecated; compiling upstream's spelling against the Homebrew headers warns "`'isa' is deprecated: Use mlir::isa<U>() instead`". The repo uses the free function `llvm::isa<TensorType>(type)`.
- The repo also lists `BuiltinDialect` as legal, which upstream's snippet omits. It is a safety net rather than a requirement here: a partial conversion leaves operations the target doesn't mention alone anyway (section 4.1), and a scratch build without `BuiltinDialect` produced output identical to sections 6.5 and 6.6.

---

## 3. Conversion Patterns

After the conversion target is defined, we define how to convert the *illegal* operations into *legal* ones. Like Chapter 3's canonicalization, the conversion framework uses [rewrite patterns](https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/) for the conversion logic. Everything in this section is from `mlir/LowerToAffineLoops.cpp`; read it top to bottom alongside this section.

### 3.1 ConversionPattern vs RewritePattern

The patterns can be the `RewritePattern`s from Chapter 3, or a pattern type specific to the conversion framework, `ConversionPattern`. A `ConversionPattern`'s `matchAndRewrite` receives an extra **operands/adaptor argument** containing the operands that have been *remapped/replaced*. Here is the `ArrayRef<Value> operands` parameter from `BinaryOpLowering` (section 3.5):

***mlir/LowerToAffineLoops.cpp***
```cpp
LogicalResult
matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                ConversionPatternRewriter &rewriter) const final {
  ...
}
```

Why does this matter? During conversion the framework maintains a mapping from original values to their converted replacements. When `toy.mul`'s pattern runs, `op->getOperands()` still yields the *old* `tensor<3x2xf64>`-typed values — but the `operands` array (or the typed `OpAdaptor`) yields the **new `memref<3x2xf64>` values** produced by the already-converted defining ops. The pattern *matches* against the old op but *operates* on values of the new type. Any pattern that deals with type changes must read its inputs through the adaptor, never through the op directly.

Beyond upstream: MLIR 20's `ConversionPattern` has a second virtual overload, `matchAndRewrite(Operation *, ArrayRef<ValueRange> operands, ConversionPatternRewriter &)`, whose comment in `DialectConversion.h` says "This overload supports 1:N replacements." (one original value replaced by several). Its default implementation forwards to the `ArrayRef<Value>` overload, so 1:1 patterns like ours override only the latter.

This codebase uses all three pattern flavors, which is a nice comparison in itself:

| Pattern base | Used by | Operand access | When appropriate |
|---|---|---|---|
| `ConversionPattern` (type-erased, matched by op name string) | `BinaryOpLowering`, `TransposeOpLowering` | raw `ArrayRef<Value> operands`, wrapped in an ODS `Adaptor` manually | generic/templated patterns over multiple op types |
| `OpConversionPattern<OpT>` (typed) | `FuncOpLowering`, `PrintOpLowering` | typed `OpAdaptor` parameter | single-op patterns that need remapped operands |
| `OpRewritePattern<OpT>` (plain rewrite pattern) | `ConstantOpLowering`, `ReturnOpLowering` | `op` directly (no adaptor) | ops whose lowering doesn't consume remapped operands: `toy.constant` has *no* operands, `toy.return` (post-inlining) has none either |

Yes — **ordinary `RewritePattern`s can be used inside a dialect conversion**. They just don't get access to the remapped operands, which is fine when there are none.

### 3.2 Type conversion helper + `insertAllocAndDealloc`

Upstream shows only `TransposeOpLowering`, which relies on helpers defined at the top of the file. They come first here.

***mlir/LowerToAffineLoops.cpp***
```cpp
/// Convert the given RankedTensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<memref::AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow.
  auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}
```

- `tensor<3x2xf64>` maps 1:1 to `memref<3x2xf64>` — this is possible only because **shape inference (Chapter 4) already made every tensor statically ranked and shaped**. The whole lowering relies on this precondition through `llvm::cast<RankedTensorType>` calls, which have no fallback: an unranked type would trip the cast's assertion in an assertions-enabled LLVM build (Homebrew's is not one: `llvm-config --assertion-mode` prints `OFF`).
- Every lowered op that produces a tensor result gets a fresh buffer. `alloc` is hoisted to the top of the block and `dealloc` sunk to just before the terminator — a trivially correct placement **only because Toy functions have no control flow** (single block). Real bufferization pipelines (the upstream `one-shot-bufferize`) have to solve this properly.
The chapter's test input (section 6.3) has three ops that produce a tensor:

***test_Example/Toy/Ch5/affine-lowering.mlir***
```mlir
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.mul %2, %2 : tensor<3x2xf64>
  toy.print %3 : tensor<3x2xf64>
  toy.return
}
```

Lower it and keep only the buffer management lines:

```bash
cd /Users/roy/study/mlir/toy/Ch5
./build/toyc-ch5 ../../test_Example/Toy/Ch5/affine-lowering.mlir -emit=mlir-affine 2>&1 | grep -E 'memref\.(alloc|dealloc)'
```

Real output (section 6.5 shows the whole module):

```mlir
    %alloc = memref.alloc() : memref<3x2xf64>
    %alloc_5 = memref.alloc() : memref<3x2xf64>
    %alloc_6 = memref.alloc() : memref<2x3xf64>
    memref.dealloc %alloc_6 : memref<2x3xf64>
    memref.dealloc %alloc_5 : memref<3x2xf64>
    memref.dealloc %alloc : memref<3x2xf64>
```

- Three buffers, one per tensor-producing op: `%alloc_6` (2x3) for `toy.constant`, `%alloc_5` for `toy.transpose`, `%alloc` for `toy.mul`.
- The allocs sit at the top and the deallocs at the bottom, in mirrored order: each new alloc is moved to the *front* of the block, so the last one created (`toy.mul`'s) ends up first, while each new dealloc is moved to just before the terminator, so the deallocs stay in creation order.

### 3.3 `lowerOpToLoops`: the shared loop-nest skeleton

`toy.add`, `toy.mul`, and `toy.transpose` all lower to the same shape of code: *allocate result buffer → perfect loop nest over the result shape → compute one element per iteration → store it*. That skeleton is factored into a helper parameterized by a per-iteration callback, which takes a builder, the remapped memref operands, and the loop induction variables, and returns the value to store at the current index:

***mlir/LowerToAffineLoops.cpp***
```cpp
/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input an OpBuilder, an range of memRefOperands
/// corresponding to the operands of the input operation, and the range of loop
/// induction variables for the iteration. It returns a value to store at the
/// current index of the iteration.
using LoopIterationFn = function_ref<Value(
    OpBuilder &rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;

static void lowerOpToLoops(Operation *op, ValueRange operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
  auto loc = op->getLoc();

  // Insert an allocation and deallocation for the result of this operation.
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  // Create a nest of affine loops, with one loop per dimension of the shape.
  // The buildAffineLoopNest function takes a callback that is used to construct
  // the body of the innermost loop given a builder, a location and a range of
  // loop induction variables.
  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
  affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        // Call the processing function with the rewriter, the memref operands,
        // and the loop induction variables. This function will return the value
        // to store at the current index.
        Value valueToStore = processIteration(nestedBuilder, operands, ivs);
        nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc,
                                                    ivs);
      });

  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);
}
```

Key points:

- `affine::buildAffineLoopNest` builds one `affine.for` per rank dimension (`0 to dim step 1`) and hands the innermost-body callback the full list of induction variables (`ivs`).
- The callback contract is elegant: *"given the operand buffers and the current index, produce the scalar to store"*. The store itself is uniform across all ops.
- `rewriter.replaceOp(op, alloc)` is the type-changing move: every use of the old `tensor` SSA value is remapped to the `memref` value. Consumers that are converted later (or `toy.print` via its adaptor) will see the memref.

### 3.4 `TransposeOpLowering` — upstream's example: reversed induction variables

This is the pattern upstream uses to introduce conversion patterns:

***mlir/LowerToAffineLoops.cpp***
```cpp
struct TransposeOpLowering : public ConversionPattern {
  TransposeOpLowering(MLIRContext *ctx)
      : ConversionPattern(toy::TransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter,
                   [loc](OpBuilder &builder, ValueRange memRefOperands,
                         ValueRange loopIvs) {
                     // Generate an adaptor for the remapped operands of the
                     // TransposeOp. This allows for using the nice named
                     // accessors that are generated by the ODS.
                     toy::TransposeOpAdaptor transposeAdaptor(memRefOperands);
                     Value input = transposeAdaptor.getInput();

                     // Transpose the elements by generating a load from the
                     // reverse indices.
                     SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
                     return builder.create<affine::AffineLoadOp>(loc, input,
                                                                 reverseIvs);
                   });
    return success();
  }
};
```

The entire semantics of transpose collapses into one line: the loop nest iterates over the **output** shape `(i, j)`, and each iteration loads `input[j, i]` — `llvm::reverse(loopIvs)`. The store side is handled by `lowerOpToLoops` at `[i, j]`.

To see the pattern on its own, lower a function that only transposes a constant. The input is not a repo file, so it goes on stdin (`-` for the file name, `-x mlir` because the driver can't infer the input kind from an extension):

```bash
cd /Users/roy/study/mlir/toy/Ch5
./build/toyc-ch5 - -x mlir -emit=mlir-affine <<'EOF' 2>&1
toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  toy.print %1 : tensor<3x2xf64>
  toy.return
}
EOF
```

Real output:

```mlir
module {
  func.func @main() {
    %cst = arith.constant 6.000000e+00 : f64
    %cst_0 = arith.constant 5.000000e+00 : f64
    %cst_1 = arith.constant 4.000000e+00 : f64
    %cst_2 = arith.constant 3.000000e+00 : f64
    %cst_3 = arith.constant 2.000000e+00 : f64
    %cst_4 = arith.constant 1.000000e+00 : f64
    %alloc = memref.alloc() : memref<3x2xf64>
    %alloc_5 = memref.alloc() : memref<2x3xf64>
    affine.store %cst_4, %alloc_5[0, 0] : memref<2x3xf64>
    affine.store %cst_3, %alloc_5[0, 1] : memref<2x3xf64>
    affine.store %cst_2, %alloc_5[0, 2] : memref<2x3xf64>
    affine.store %cst_1, %alloc_5[1, 0] : memref<2x3xf64>
    affine.store %cst_0, %alloc_5[1, 1] : memref<2x3xf64>
    affine.store %cst, %alloc_5[1, 2] : memref<2x3xf64>
    affine.for %arg0 = 0 to 3 {
      affine.for %arg1 = 0 to 2 {
        %0 = affine.load %alloc_5[%arg1, %arg0] : memref<2x3xf64>
        affine.store %0, %alloc[%arg0, %arg1] : memref<3x2xf64>
      }
    }
    toy.print %alloc : memref<3x2xf64>
    memref.dealloc %alloc_5 : memref<2x3xf64>
    memref.dealloc %alloc : memref<3x2xf64>
    return
  }
}
```

- The nest runs over the **output** shape 3x2 (`0 to 3`, `0 to 2`), one `affine.for` per dimension from `buildAffineLoopNest` (section 3.3).
- The load from the input buffer `%alloc_5` uses the reversed indices `[%arg1, %arg0]`; the store into the result buffer `%alloc` uses `[%arg0, %arg1]`.
- The rest is the other patterns at work: the constant became six stores into `%alloc_5` (section 3.6), and `toy.print` now takes the result memref (section 3.8).

Upstream's snippet of this pattern predates the example source it comes from. Compiled against the MLIR 20 headers, three of its spellings fail, while the repo's code (identical to upstream's own `release/20.x` example) compiles:

| Upstream doc snippet | MLIR 20 (this repo) | Result of compiling the upstream spelling |
|---|---|---|
| `transposeAdaptor.input()` | `transposeAdaptor.getInput()` | `error: no member named 'input' in 'mlir::toy::TransposeOpAdaptor'` — ODS generates only `get`-prefixed accessors |
| `rewriter.create<mlir::AffineLoadOp>(...)` | `builder.create<affine::AffineLoadOp>(...)` | `error: no member named 'AffineLoadOp' in namespace 'mlir'` — affine ops live in `mlir::affine` |
| callback `(mlir::PatternRewriter &rewriter, ArrayRef<mlir::Value> memRefOperands, ArrayRef<mlir::Value> loopIvs)` | `(OpBuilder &builder, ValueRange memRefOperands, ValueRange loopIvs)` | `error: no matching function for call` when passed as a `LoopIterationFn` — the callback receives the nested `OpBuilder` from `buildAffineLoopNest`, not a `PatternRewriter` |

Upstream's `llvm::LogicalResult` return type is fine: MLIR 20's `LogicalResult` is the LLVM one, and the repo's unqualified `LogicalResult` resolves to it.

### 3.5 `BinaryOpLowering` — Add and Mul in one template

***mlir/LowerToAffineLoops.cpp***
```cpp
template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter,
                   [loc](OpBuilder &builder, ValueRange memRefOperands,
                         ValueRange loopIvs) {
                     // Generate an adaptor for the remapped operands of the
                     // BinaryOp. This allows for using the nice named accessors
                     // that are generated by the ODS.
                     typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);

                     // Generate loads for the element of 'lhs' and 'rhs' at the
                     // inner loop.
                     auto loadedLhs = builder.create<affine::AffineLoadOp>(
                         loc, binaryAdaptor.getLhs(), loopIvs);
                     auto loadedRhs = builder.create<affine::AffineLoadOp>(
                         loc, binaryAdaptor.getRhs(), loopIvs);

                     // Create the binary operation performed on the loaded
                     // values.
                     return builder.create<LoweredBinaryOp>(loc, loadedLhs,
                                                            loadedRhs);
                   });
    return success();
  }
};
using AddOpLowering = BinaryOpLowering<toy::AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<toy::MulOp, arith::MulFOp>;
```

The chapter's test only multiplies a value by itself, so lower a `toy.add` of two different constants (on stdin again):

```bash
cd /Users/roy/study/mlir/toy/Ch5
./build/toyc-ch5 - -x mlir -emit=mlir-affine <<'EOF' 2>&1
toy.func @main() {
  %0 = toy.constant dense<[1.0, 2.0]> : tensor<2xf64>
  %1 = toy.constant dense<[3.0, 4.0]> : tensor<2xf64>
  %2 = toy.add %0, %1 : tensor<2xf64>
  toy.print %2 : tensor<2xf64>
  toy.return
}
EOF
```

Real output:

```mlir
module {
  func.func @main() {
    %cst = arith.constant 4.000000e+00 : f64
    %cst_0 = arith.constant 3.000000e+00 : f64
    %cst_1 = arith.constant 2.000000e+00 : f64
    %cst_2 = arith.constant 1.000000e+00 : f64
    %alloc = memref.alloc() : memref<2xf64>
    %alloc_3 = memref.alloc() : memref<2xf64>
    %alloc_4 = memref.alloc() : memref<2xf64>
    affine.store %cst_2, %alloc_4[0] : memref<2xf64>
    affine.store %cst_1, %alloc_4[1] : memref<2xf64>
    affine.store %cst_0, %alloc_3[0] : memref<2xf64>
    affine.store %cst, %alloc_3[1] : memref<2xf64>
    affine.for %arg0 = 0 to 2 {
      %0 = affine.load %alloc_4[%arg0] : memref<2xf64>
      %1 = affine.load %alloc_3[%arg0] : memref<2xf64>
      %2 = arith.addf %0, %1 : f64
      affine.store %2, %alloc[%arg0] : memref<2xf64>
    }
    toy.print %alloc : memref<2xf64>
    memref.dealloc %alloc_4 : memref<2xf64>
    memref.dealloc %alloc_3 : memref<2xf64>
    memref.dealloc %alloc : memref<2xf64>
    return
  }
}
```

- One template covers both element-wise ops; only the scalar op differs: `arith.addf` here, `arith.mulf` for the `toy.mul` in section 6.5. A rank-1 tensor gets a single `affine.for`.
- Note `BinaryOp::Adaptor binaryAdaptor(memRefOperands)`: the ODS adaptor is constructed **over the remapped operands**, so `getLhs()`/`getRhs()` return `memref`-typed values, not the original tensors. The two loads read `%alloc_4` and `%alloc_3`, the buffers of the two lowered constants. This is the ConversionPattern discipline from section 3.1 in action.
- Per iteration: load lhs element, load rhs element, add/multiply, and (via `lowerOpToLoops`) store into the result buffer `%alloc`. Here the operands differ, so both loads survive the cleanup. For `toy.mul %2, %2` the two loads are identical; the post-lowering `cse` merges them (section 4.1 shows both before the merge).

### 3.6 `ConstantOpLowering` — a constant becomes a buffer full of stores

***mlir/LowerToAffineLoops.cpp***
```cpp
struct ConstantOpLowering : public OpRewritePattern<toy::ConstantOp> {
  using OpRewritePattern<toy::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(toy::ConstantOp op,
                                PatternRewriter &rewriter) const final {
    DenseElementsAttr constantValue = op.getValue();
    Location loc = op.getLoc();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto tensorType = llvm::cast<RankedTensorType>(op.getType());
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(0, *llvm::max_element(valueShape)))
        constantIndices.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, i));
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }

    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.value_begin<FloatAttr>();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<affine::AffineStoreOp>(
            loc, rewriter.create<arith::ConstantOp>(loc, *valueIt++), alloc,
            llvm::ArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};
```

Why a *series of stores* rather than something like `memref.global`? Simplicity and transparency:

- A `toy.constant` is a **multi-dimensional dense value attribute**. At the buffer level there is no single op for "buffer initialized with these values" in this simple pipeline, so we unroll it: one scalar `arith.constant` + one `affine.store` per element. The index constants are created once, up to the largest dimension, so the stores share them instead of each emitting fresh `arith.constant ... : index` ops.
- After canonicalization the stores use **constant indices** (`affine.store %cst, %alloc[0, 1]`), which the affine dialect prints in folded form. That constant-index property is exactly what lets affine analyses such as `AffineScalarReplacement` reason precisely about which element each store writes.
- Making the whole thing visible as plain stores means downstream affine analyses can reason about the initialization precisely — no opaque "magic constant buffer".
- This is an `OpRewritePattern`, not a `ConversionPattern`: `toy.constant` has zero operands, so there is nothing to remap.

To see what the pattern itself emits, lower a function that only prints a 2x3 constant and look at the IR right after the lowering pass, before the post-lowering cleanup. `-mlir-print-ir-after-all` dumps the IR after every pass (section 4.1), and `sed` keeps only the dump after `toy-to-affine`:

```bash
cd /Users/roy/study/mlir/toy/Ch5
./build/toyc-ch5 - -x mlir -emit=mlir-affine -mlir-print-ir-after-all <<'EOF' 2>&1 | sed -n '/toy-to-affine/,/^}/p'
toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  toy.print %0 : tensor<2x3xf64>
  toy.return
}
EOF
```

Real output:

```mlir
// -----// IR Dump After (anonymous namespace)::ToyToAffineLoweringPass (toy-to-affine) //----- //
module {
  func.func @main() {
    %alloc = memref.alloc() : memref<2x3xf64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 1.000000e+00 : f64
    affine.store %cst, %alloc[%c0, %c0] : memref<2x3xf64>
    %cst_0 = arith.constant 2.000000e+00 : f64
    affine.store %cst_0, %alloc[%c0, %c1] : memref<2x3xf64>
    %cst_1 = arith.constant 3.000000e+00 : f64
    affine.store %cst_1, %alloc[%c0, %c2] : memref<2x3xf64>
    %cst_2 = arith.constant 4.000000e+00 : f64
    affine.store %cst_2, %alloc[%c1, %c0] : memref<2x3xf64>
    %cst_3 = arith.constant 5.000000e+00 : f64
    affine.store %cst_3, %alloc[%c1, %c1] : memref<2x3xf64>
    %cst_4 = arith.constant 6.000000e+00 : f64
    affine.store %cst_4, %alloc[%c1, %c2] : memref<2x3xf64>
    toy.print %alloc : memref<2x3xf64>
    memref.dealloc %alloc : memref<2x3xf64>
    return
  }
}
```

The same input without the dump flag shows the result after the always-on `canonicalize` + `cse` (section 5):

```bash
cd /Users/roy/study/mlir/toy/Ch5
./build/toyc-ch5 - -x mlir -emit=mlir-affine <<'EOF' 2>&1
toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  toy.print %0 : tensor<2x3xf64>
  toy.return
}
EOF
```

Real output:

```mlir
module {
  func.func @main() {
    %cst = arith.constant 6.000000e+00 : f64
    %cst_0 = arith.constant 5.000000e+00 : f64
    %cst_1 = arith.constant 4.000000e+00 : f64
    %cst_2 = arith.constant 3.000000e+00 : f64
    %cst_3 = arith.constant 2.000000e+00 : f64
    %cst_4 = arith.constant 1.000000e+00 : f64
    %alloc = memref.alloc() : memref<2x3xf64>
    affine.store %cst_4, %alloc[0, 0] : memref<2x3xf64>
    affine.store %cst_3, %alloc[0, 1] : memref<2x3xf64>
    affine.store %cst_2, %alloc[0, 2] : memref<2x3xf64>
    affine.store %cst_1, %alloc[1, 0] : memref<2x3xf64>
    affine.store %cst_0, %alloc[1, 1] : memref<2x3xf64>
    affine.store %cst, %alloc[1, 2] : memref<2x3xf64>
    toy.print %alloc : memref<2x3xf64>
    memref.dealloc %alloc : memref<2x3xf64>
    return
  }
}
```

- The pattern emits the three shared index constants `%c0`–`%c2` up front (up to the largest dimension, 3), then walks the shape in row-major order, interleaving each scalar `arith.constant f64` with its `affine.store`: 6 + 6 ops for 2x3 elements.
- After cleanup the index operands are folded into the stores' affine maps (`%alloc[0, 1]`) and the index constants are gone; `canonicalize` has also hoisted the scalar constants to the top of the function, in reverse order (`%cst` is `6.0`, `%cst_4` is `1.0`).

### 3.7 `FuncOpLowering` and `ReturnOpLowering` — the structural ops

The two structural patterns (abridged; `PrintOpLowering`, which sits between them in the file, is in section 3.8):

***mlir/LowerToAffineLoops.cpp***
```cpp
struct FuncOpLowering : public OpConversionPattern<toy::FuncOp> {
  using OpConversionPattern<toy::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // We only lower the main function as we expect that all other functions
    // have been inlined.
    if (op.getName() != "main")
      return failure();

    // Verify that the given main has no inputs and results.
    if (op.getNumArguments() || op.getFunctionType().getNumResults()) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
        diag << "expected 'main' to have 0 inputs and 0 results";
      });
    }

    // Create a new non-toy function, with the same region.
    auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(),
                                                    op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};
...
struct ReturnOpLowering : public OpRewritePattern<toy::ReturnOp> {
  using OpRewritePattern<toy::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(toy::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    // During this lowering, we expect that all function calls have been
    // inlined.
    if (op.hasOperand())
      return failure();

    // We lower "toy.return" directly to "func.return".
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};
```

- `toy.func @main` → `func.func @main`: a new `func.func` with the same name and type is created and the body region is *moved* into it (`inlineRegionBefore`), not cloned. This lowering **depends on the inliner having run first** — any non-main `toy.func` makes the pattern fail, and since `toy.func` is illegal, the pass fails. Same for a `toy.return` with an operand.
- `notifyMatchFailure` attaches a human-readable reason to the failure. The reason reaches the rewriter's listeners and the conversion's debug log, not the user: this Homebrew build has assertions off and `toyc-ch5` has no `-debug` flag, so a `main` with arguments reports only "failed to legalize operation 'toy.func'" (run below). Calling it still costs nothing and pays off in a debug build.

A `main` with an argument, on stdin, hits the `notifyMatchFailure` branch:

```bash
cd /Users/roy/study/mlir/toy/Ch5
./build/toyc-ch5 - -x mlir -emit=mlir-affine <<'EOF' 2>&1
toy.func @main(%arg0: tensor<2xf64>) {
  toy.print %arg0 : tensor<2xf64>
  toy.return
}
EOF
```

Real output (exit status 4):

```text
<stdin>:1:1: error: failed to legalize operation 'toy.func' that was explicitly marked illegal
toy.func @main(%arg0: tensor<2xf64>) {
^
<stdin>:1:1: note: see current operation: 
"toy.func"() <{function_type = (tensor<2xf64>) -> (), sym_name = "main"}> ({
^bb0(%arg0: tensor<2xf64>):
  "toy.print"(%arg0) : (tensor<2xf64>) -> ()
  "toy.return"() : () -> ()
}) : () -> ()
```

- The pattern failed, `toy.func` is illegal, so the conversion fails and the driver returns 4 without dumping the module (section 4.1 shows the driver code). The text "expected 'main' to have 0 inputs and 0 results" does not appear in this release build.
- The `note: see current operation` line prints the op in generic form, so it shows the properties `<{...}>` that the custom syntax hides.

### 3.8 `PrintOpLowering` — the op that *doesn't* get lowered

***mlir/LowerToAffineLoops.cpp***
```cpp
struct PrintOpLowering : public OpConversionPattern<toy::PrintOp> {
  using OpConversionPattern<toy::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // We don't lower "toy.print" in this pass, but we need to update its
    // operands.
    rewriter.modifyOpInPlace(op,
                             [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};
```

This is the counterpart of the `addDynamicallyLegalOp` rule from section 2.1 and the loosened ODS type from section 4.3. The pattern:

1. does **not** replace or erase the op — it survives as `toy.print`;
2. swaps its operands (tensor → memref) for the adaptor's remapped ones (the `memref` result of the lowered `toy.mul`), inside `modifyOpInPlace` so the conversion driver correctly tracks the mutation;
3. after the swap, the dynamic legality predicate (`no TensorType operands`) becomes true, so the framework now considers the op legal and stops worrying about it.

The `toy.print` line of the test input before and after the lowering:

```bash
cd /Users/roy/study/mlir/toy/Ch5
./build/toyc-ch5 ../../test_Example/Toy/Ch5/affine-lowering.mlir -emit=mlir -opt 2>&1 | grep toy.print
./build/toyc-ch5 ../../test_Example/Toy/Ch5/affine-lowering.mlir -emit=mlir-affine 2>&1 | grep toy.print
```

Real output:

```mlir
    toy.print %2 : tensor<3x2xf64>
    toy.print %alloc : memref<3x2xf64>
```

Same op, same position, but its operand is now `%alloc`, the result buffer of the lowered `toy.mul`: a Toy op holding a MemRef value. Mixed dialects, working together.

### 3.9 Collecting the patterns

With the target defined, `runOnOperation` collects the seven patterns into a `RewritePatternSet`. Upstream abbreviates the list to `patterns.add<..., TransposeOpLowering>`; the full list is:

***mlir/LowerToAffineLoops.cpp***
```cpp
  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  RewritePatternSet patterns(&getContext());
  patterns.add<AddOpLowering, ConstantOpLowering, FuncOpLowering, MulOpLowering,
               PrintOpLowering, ReturnOpLowering, TransposeOpLowering>(
      &getContext());
```

There is one pattern per Toy op that can still be present when the pass runs: `add`, `constant`, `func`, `mul`, `print`, `return`, `transpose`. `toy.generic_call` and `toy.cast` have no pattern because inlining, shape inference, and canonicalization remove them first, and `toy.reshape` has none because canonicalization is expected to fold every reshape away. If one survives, the pass fails (section 4.1).

---

## 4. Partial Lowering

### 4.1 `applyPartialConversion`

With the target and the patterns defined, the pass performs the actual lowering. The framework has several modes; we use a *partial* conversion, because `toy.print` is not converted at this time:

***mlir/LowerToAffineLoops.cpp***
```cpp
  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
```

The modes declared in MLIR 20's `DialectConversion.h`:

- **`applyPartialConversion`**: "converts as many operations to the target as possible, ignoring operations that failed to legalize. This method only returns failure if there ops explicitly marked as illegal." Operations *unknown* to the target are left alone. This is the natural fit for progressive lowering, where we knowingly emit a mixed-dialect module.
- **`applyFullConversion`**: "returns failure if the conversion of any operation fails". Every operation must end up legal. Use it when the output must be a closed set of dialects (e.g., the final LLVM lowering in Chapter 6).
- **`applyAnalysisConversion`**: a dry run that records which operations *would* legalize (in `config.legalizableOps`) without applying any rewrite.

Upstream is outdated here too: its snippet passes the pattern set as an lvalue, `applyPartialConversion(getOperation(), target, patterns)`. In MLIR 20 the third parameter is a `const FrozenRewritePatternSet &`, which can only be built from a `RewritePatternSet &&`, so upstream's spelling fails with `error: no matching function for call to 'applyPartialConversion'`. The repo's `std::move(patterns)` is required.

#### The IR right after the pass: a partially lowered module

The driver applies MLIR's pass-manager command-line options to its pass manager in `dumpMLIR()`, and `main()` registers them, so the standard IR-printing flags work (abridged):

***toyc.cpp***
```cpp
  mlir::PassManager pm(module.get()->getName());
  // Apply any generic pass manager command line options and run the pipeline.
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;
  ...
int main(int argc, char **argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  ...
}
```

`-mlir-print-ir-after-all` dumps the IR after every pass; the dump after the lowering pass itself is the raw result of `applyPartialConversion` on the test input (shown in section 3.2), before any cleanup:

```bash
cd /Users/roy/study/mlir/toy/Ch5
./build/toyc-ch5 ../../test_Example/Toy/Ch5/affine-lowering.mlir -emit=mlir-affine -mlir-print-ir-after-all 2>&1
```

Real output (abridged to the dump after `toy-to-affine`; the other dumps are after `canonicalize`, `inline`, `toy-shape-inference`, `canonicalize`, `cse` before it, and `canonicalize`, `cse` after it):

```mlir
// -----// IR Dump After (anonymous namespace)::ToyToAffineLoweringPass (toy-to-affine) //----- //
module {
  func.func @main() {
    %alloc = memref.alloc() : memref<3x2xf64>
    %alloc_0 = memref.alloc() : memref<3x2xf64>
    %alloc_1 = memref.alloc() : memref<2x3xf64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 1.000000e+00 : f64
    affine.store %cst, %alloc_1[%c0, %c0] : memref<2x3xf64>
    %cst_2 = arith.constant 2.000000e+00 : f64
    affine.store %cst_2, %alloc_1[%c0, %c1] : memref<2x3xf64>
    %cst_3 = arith.constant 3.000000e+00 : f64
    affine.store %cst_3, %alloc_1[%c0, %c2] : memref<2x3xf64>
    %cst_4 = arith.constant 4.000000e+00 : f64
    affine.store %cst_4, %alloc_1[%c1, %c0] : memref<2x3xf64>
    %cst_5 = arith.constant 5.000000e+00 : f64
    affine.store %cst_5, %alloc_1[%c1, %c1] : memref<2x3xf64>
    %cst_6 = arith.constant 6.000000e+00 : f64
    affine.store %cst_6, %alloc_1[%c1, %c2] : memref<2x3xf64>
    affine.for %arg0 = 0 to 3 {
      affine.for %arg1 = 0 to 2 {
        %0 = affine.load %alloc_1[%arg1, %arg0] : memref<2x3xf64>
        affine.store %0, %alloc_0[%arg0, %arg1] : memref<3x2xf64>
      }
    }
    affine.for %arg0 = 0 to 3 {
      affine.for %arg1 = 0 to 2 {
        %0 = affine.load %alloc_0[%arg0, %arg1] : memref<3x2xf64>
        %1 = affine.load %alloc_0[%arg0, %arg1] : memref<3x2xf64>
        %2 = arith.mulf %0, %1 : f64
        affine.store %2, %alloc[%arg0, %arg1] : memref<3x2xf64>
      }
    }
    toy.print %alloc : memref<3x2xf64>
    memref.dealloc %alloc_1 : memref<2x3xf64>
    memref.dealloc %alloc_0 : memref<3x2xf64>
    memref.dealloc %alloc : memref<3x2xf64>
    return
  }
}
```

- **Partial, as intended:** every op is now `func`, `memref`, `arith`, or `affine` — except `toy.print`, which the target made dynamically legal and which now holds a memref (section 3.8).
- `ConstantOpLowering` emitted the index constants `%c0`–`%c2` and interleaved each scalar constant with its store (section 3.6); the following `canonicalize` folds the indices into the store maps and hoists the constants, in reverse order.
- `BinaryOpLowering` emitted two loads of the same element for `toy.mul %2, %2` (section 3.5). They survive `canonicalize` and are merged by the following `cse`, which is why section 6.5 shows a single load.
- The pass name in the header, `toy-to-affine`, is the pass's `getArgument()` (section 4.2). Since the pass is not registered with the command line, `-mlir-print-ir-after=toy-to-affine` is rejected with `Cannot find option named 'toy-to-affine'!`.

#### An illegal op without a pattern fails the pass

Even with partial conversion, our pass is strict about Toy: we marked the whole dialect illegal, so a stray `toy.reshape`, which has no pattern, aborts the pass. Canonicalization folds most reshapes away, but not a reshape of a transpose result. The driver turns the failed pipeline into exit status 4:

***toyc.cpp***
```cpp
  if (mlir::failed(pm.run(*module)))
    return 4;
```

```bash
cd /Users/roy/study/mlir/toy/Ch5
./build/toyc-ch5 - -x mlir -emit=mlir-affine <<'EOF' 2>&1
toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %2 = toy.reshape(%1 : tensor<3x2xf64>) to tensor<6xf64>
  toy.print %2 : tensor<6xf64>
  toy.return
}
EOF
```

Real output (exit status 4):

```text
<stdin>:4:8: error: failed to legalize operation 'toy.reshape' that was explicitly marked illegal
  %2 = toy.reshape(%1 : tensor<3x2xf64>) to tensor<6xf64>
       ^
<stdin>:4:8: note: see current operation: %13 = "toy.reshape"(%12) : (tensor<3x2xf64>) -> tensor<6xf64>
```

The error names the op and points at its source line. Partial conversion is not "ignore what you can't convert"; it is "ignore what I didn't classify". The module is not dumped.

### 4.2 The pass around it

The pass wrapper and its factory at the bottom of `LowerToAffineLoops.cpp` (abridged: the body of `runOnOperation` is the three pieces in sections 2.1, 3.9, and 4.1):

***mlir/LowerToAffineLoops.cpp***
```cpp
/// This is a partial lowering to affine loops of the toy operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Toy dialect.
namespace {
struct ToyToAffineLoweringPass
    : public PassWrapper<ToyToAffineLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToAffineLoweringPass)
  StringRef getArgument() const override { return "toy-to-affine"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, func::FuncDialect,
                    memref::MemRefDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void ToyToAffineLoweringPass::runOnOperation() {
  ...
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> mlir::toy::createLowerToAffinePass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}
```

- **`OperationPass<ModuleOp>`**: unlike Chapter 4's shape-inference pass (which ran on `toy::FuncOp`), this runs on the whole module — it has to, because it replaces `toy.func` itself with `func.func`.
- **`getDependentDialects` is mandatory here.** The pass *creates* ops from dialects (`affine`, `func`, `memref`) that are not loaded in the context yet — the input module only uses `toy`. Declaring dependent dialects makes the pass manager load them before the (multithreaded) pass runs. `arith` is not listed but still gets loaded, because both `AffineDialect` (`AffineOps.td`) and `MemRefDialect` (`MemRefBase.td`) declare `arith::ArithDialect` as a dependent dialect. A scratch build with `getDependentDialects` removed aborts (exit 134) on the first op it creates: ``LLVM ERROR: Building op `func.func` but it isn't known in this MLIRContext: the dialect may not be loaded or this operation hasn't been added by the dialect. ...``
- `getArgument()` gives the pass a command-line name (`toy-to-affine`); it appears in IR-printing headers (section 4.1), and would become a flag if the pass were ever registered with an opt-style tool.
- `runOnOperation` is exactly the three-step recipe from section 2: define the target (section 2.1), populate the patterns (section 3.9), apply a **partial** conversion and fail the pass if any illegal op survives (section 4.1).

The factory `createLowerToAffinePass()` is declared in `include/toy/Passes.h` and consumed by the driver (section 5).

### 4.3 Design Considerations With Partial Lowering

Lowering `tensor` (a value type) to `memref` (an allocated, buffer-like type) while `toy.print` stays in Toy means the two worlds must be bridged temporarily: a still-Toy op consumes a lowered value. Upstream lists three options, each with its trade-offs:

1. **Generate `load` operations from the buffer** to materialize an instance of the value type again. The definition of `toy.print` stays unchanged. The downside: the affine optimizations are limited, because the load is really a full copy of the buffer, and that copy only becomes visible *after* the optimizations have run.
2. **Add a lowered variant of `toy.print`** that operates on the lowered type. No hidden, unnecessary copy reaches the optimizer. The downside: another operation definition, duplicating many aspects of the first. A shared base class in [ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/) can reduce the duplication, but the two ops must still be handled separately everywhere.
3. **Update `toy.print` to also accept the lowered type.** Simple, no hidden copy, no extra op definition. The downside: it mixes abstraction levels inside the Toy dialect.

The tutorial takes option 3, for simplicity. `PrintOp` declares its operand as `AnyTypeOf<[F64Tensor, F64MemRef]>`; in Chapter 4 it was plain `F64Tensor`:

***include/toy/Ops.td***
```tablegen
def PrintOp : Toy_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  // We also allow a F64MemRef to enable interop during partial lowering.
  let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}
```

The op simply "follows" the lowering of its operand, as the runs in section 3.8 show (`toy.print %2 : tensor<3x2xf64>` becomes `toy.print %alloc : memref<3x2xf64>`). Besides mixing abstraction levels, the cost is a weaker verifier: at the Toy level `toy.print` can no longer insist on a tensor operand. That is an acceptable trade for an op that only needs the memref form mid-pipeline. Upstream's `PrintOp` snippet matches the repo's `Ops.td` exactly.

---

## 5. The Pipeline in toyc.cpp

Upstream's last two sections run the example "with affine lowering added to our pipeline" and after "adding the `LoopFusion` and `AffineScalarReplacement` passes to the pipeline", without showing the driver code. This section shows it. `toyc.cpp` grows a new action this chapter:

***toyc.cpp***
```cpp
namespace {
enum Action { None, DumpAST, DumpMLIR, DumpMLIRAffine };
} // namespace
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine",
                          "output the MLIR dump after affine lowering")));
```

`dumpMLIR()` now builds its context from a `DialectRegistry` with all `func` dialect extensions registered (`mlir::func::registerAllExtensions(registry)`, new relative to Chapter 4), and creates the `PassManager` unconditionally rather than only under `-opt`. The interesting part is how the pipeline is assembled incrementally:

***toyc.cpp***
```cpp
  // Check to see what granularity of MLIR we are compiling to.
  bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;

  if (enableOpt || isLoweringToAffine) {
    // Inline all functions into main and then delete them.
    pm.addPass(mlir::createInlinerPass());

    // Now that there is only one function, we can infer the shapes of each of
    // the operations.
    mlir::OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
    optPM.addPass(mlir::toy::createShapeInferencePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }

  if (isLoweringToAffine) {
    // Partially lower the toy dialect.
    pm.addPass(mlir::toy::createLowerToAffinePass());

    // Add a few cleanups post lowering.
    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());

    // Add optimizations if enabled.
    if (enableOpt) {
      optPM.addPass(mlir::affine::createLoopFusionPass());
      optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
    }
  }
```

Reading this carefully:

- **`-emit=mlir-affine` implies the Chapter-4 pipeline even without `-opt`** (`enableOpt || isLoweringToAffine`). The lowering *requires* inlining (only `main` is lowered) and shape inference (only ranked tensors can become memrefs), so those passes are not optional prerequisites — they're structural ones.
- The nesting changes across the lowering boundary: the pre-lowering cleanup nests on **`toy::FuncOp`**, the post-lowering cleanup nests on **`func::FuncOp`** — because the lowering pass replaced the function op in between. Nesting on the wrong op type silently runs the passes on nothing.
- Post-lowering, `canonicalize` + `cse` always run: canonicalization folds the `arith.constant index` ops into the affine maps, erases them, and hoists the scalar constants to the top of the function; CSE merges the repeated loads of `toy.mul %2, %2` (the IR before them is in section 4.1).
- With `-opt`, two affine-level optimizations run:
  - **`affine::createLoopFusionPass()`** (`affine-loop-fusion`): fuses producer/consumer affine loop nests to improve locality — here it fuses the transpose nest into the multiply nest. On this input the intermediate buffer survives fusion unchanged (section 6.9).
  - **`affine::createAffineScalarReplacementPass()`** (`affine-scalrep`, historically "memref dataflow opt"): forwards stored values to subsequent loads, deletes redundant loads and dead stores, and erases memrefs that end up unused. This is what removes the store-then-load through the intermediate buffer after fusion, and then the buffer itself.

Both dumps then come from the same `module->dump()` at the end — `-emit=mlir` vs `-emit=mlir-affine` differ only in which passes were added.

---

## 6. Build and Run

The superbuild (`toy/build.sh`, `toy/run.sh`, `CMakePresets.json`) is documented once in the top-level [README](../README.md#the-build-system). This section builds and runs Chapter 5 on its own, from the chapter directory. What the chapter's `CMakeLists.txt` adds — a single source file — is in the [appendix](#appendix-what-chapter-5-adds-to-the-build).

### 6.1 Building

```bash
cd /Users/roy/study/mlir/toy/Ch5
cmake -S . -B build -G Ninja
cmake --build build          # → ./build/toyc-ch5
```

No preset applies at the chapter level, yet no toolchain flags are needed, because the shell environment already points at Homebrew LLVM 20:

- `CXX=/opt/homebrew/opt/llvm@20/bin/clang++` (and `CC`) selects the compiler.
- `/opt/homebrew/opt/llvm@20/bin` is on `PATH`, and `find_package` also searches the prefix above each `PATH` entry, so it finds `/opt/homebrew/opt/llvm@20/lib/cmake/{mlir,llvm}` by itself.

In a shell without that setup, pass them explicitly: `-DMLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++`.

A standalone build puts the binary directly in `build/`; the superbuild's `toy/build/bin/toyc-ch5` behaves identically.

### 6.2 Running

```bash
cd /Users/roy/study/mlir/toy/Ch5
./build/toyc-ch5 ../../test_Example/Toy/Ch5/affine-lowering.mlir -emit=mlir -opt 2>&1
./build/toyc-ch5 ../../test_Example/Toy/Ch5/affine-lowering.mlir -emit=mlir-affine 2>&1
./build/toyc-ch5 ../../test_Example/Toy/Ch5/affine-lowering.mlir -emit=mlir-affine -opt 2>&1
```

- `../../test_Example/Toy/Ch5/affine-lowering.mlir` — the input (section 6.3); upstream's `test/Examples/Toy/Ch5/affine-lowering.mlir` in the LLVM tree. It ends in `.mlir`, so the driver parses it with MLIR's parser instead of the Toy frontend; no `-x mlir` needed.
- `-emit=mlir` — dump the module after the (optional) Toy-level pipeline, as in Chapter 4.
- `-emit=mlir-affine` — new this chapter: additionally run the partial lowering to affine and its cleanups. It implies inlining and shape inference even without `-opt` (section 5).
- `-opt` — at the Toy level, the Chapter 3/4 optimizations; after lowering, additionally `affine-loop-fusion` and `affine-scalrep` (section 5).
- `2>&1` — `module->dump()` writes to **stderr**.

### 6.3 The input

Upstream's "Complete Toy Example" uses exactly this function. The test input is already Toy-dialect MLIR. It also carries the `FileCheck` expectations used by the LIT-style `RUN:` lines at the top (the `CHECK`/`OPT` lines below the function are omitted here):

***test_Example/Toy/Ch5/affine-lowering.mlir***
```mlir
// RUN: toyc-ch5 %s -emit=mlir-affine 2>&1 | FileCheck %s
// RUN: toyc-ch5 %s -emit=mlir-affine -opt 2>&1 | FileCheck %s --check-prefix=OPT

toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.mul %2, %2 : tensor<3x2xf64>
  toy.print %3 : tensor<3x2xf64>
  toy.return
}
```

i.e. `print(transpose(constant)²)` — one constant, one transpose, one element-wise multiply.

### 6.4 Run 1: `-emit=mlir -opt` — the Toy-level baseline

```bash
cd /Users/roy/study/mlir/toy/Ch5
./build/toyc-ch5 ../../test_Example/Toy/Ch5/affine-lowering.mlir -emit=mlir -opt 2>&1
```

Real output:

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

Everything is still Toy. The Chapter-3 `transpose(transpose(x)) = x` pattern doesn't apply (there's only one transpose), so `-opt` at this level is essentially a no-op for this input. This is precisely the motivation for the chapter: **at the Toy level there is nothing left to optimize** — the redundancy we're about to expose (two loop nests, an intermediate buffer) doesn't even *exist* yet as a concept.

### 6.5 Run 2: `-emit=mlir-affine` — naive lowering, no `-opt`

```bash
cd /Users/roy/study/mlir/toy/Ch5
./build/toyc-ch5 ../../test_Example/Toy/Ch5/affine-lowering.mlir -emit=mlir-affine 2>&1
```

Real output:

```mlir
module {
  func.func @main() {
    %cst = arith.constant 6.000000e+00 : f64
    %cst_0 = arith.constant 5.000000e+00 : f64
    %cst_1 = arith.constant 4.000000e+00 : f64
    %cst_2 = arith.constant 3.000000e+00 : f64
    %cst_3 = arith.constant 2.000000e+00 : f64
    %cst_4 = arith.constant 1.000000e+00 : f64
    %alloc = memref.alloc() : memref<3x2xf64>
    %alloc_5 = memref.alloc() : memref<3x2xf64>
    %alloc_6 = memref.alloc() : memref<2x3xf64>
    affine.store %cst_4, %alloc_6[0, 0] : memref<2x3xf64>
    affine.store %cst_3, %alloc_6[0, 1] : memref<2x3xf64>
    affine.store %cst_2, %alloc_6[0, 2] : memref<2x3xf64>
    affine.store %cst_1, %alloc_6[1, 0] : memref<2x3xf64>
    affine.store %cst_0, %alloc_6[1, 1] : memref<2x3xf64>
    affine.store %cst, %alloc_6[1, 2] : memref<2x3xf64>
    affine.for %arg0 = 0 to 3 {
      affine.for %arg1 = 0 to 2 {
        %0 = affine.load %alloc_6[%arg1, %arg0] : memref<2x3xf64>
        affine.store %0, %alloc_5[%arg0, %arg1] : memref<3x2xf64>
      }
    }
    affine.for %arg0 = 0 to 3 {
      affine.for %arg1 = 0 to 2 {
        %0 = affine.load %alloc_5[%arg0, %arg1] : memref<3x2xf64>
        %1 = arith.mulf %0, %0 : f64
        affine.store %1, %alloc[%arg0, %arg1] : memref<3x2xf64>
      }
    }
    toy.print %alloc : memref<3x2xf64>
    memref.dealloc %alloc_6 : memref<2x3xf64>
    memref.dealloc %alloc_5 : memref<3x2xf64>
    memref.dealloc %alloc : memref<3x2xf64>
    return
  }
}
```

Annotations, mapping back to the patterns explained in section 3:

- **Three buffers** (`insertAllocAndDealloc`, section 3.2): one per value-producing Toy op — `%alloc` (3x2) for the result of `toy.mul`, `%alloc_5` (3x2) for the result of `toy.transpose` (the intermediate), `%alloc_6` (2x3) for `toy.constant`. All allocs hoisted to the block top, all deallocs sunk to the bottom, in mirror order.
- **Constant → 6 scalar `arith.constant` + 6 `affine.store`** with folded constant indices (`[0, 0]`, `[0, 1]`, …), one store per element — `ConstantOpLowering` (section 3.6). The `arith.constant index` helpers were folded into the affine maps and cleaned up by the always-on post-lowering `canonicalize`.
- **Loop nest #1** (transpose) iterates the *output* shape (3x2) and loads with reversed ivs `[%arg1, %arg0]` — `TransposeOpLowering` (section 3.4).
- **Loop nest #2** (mul): `lhs` and `rhs` of `toy.mul` are the *same* value, so the always-on `cse` already merged the two `affine.load`s from `BinaryOpLowering` (section 3.5) into one (section 4.1 shows both before the merge).
- **`toy.print %alloc : memref<3x2xf64>`** — still a Toy op, now with a memref operand: the dynamically-legal op with operands rewritten by `PrintOpLowering` (section 3.8).
- `toy.func`/`toy.return` became `func.func`/`return` (section 3.7).

The inefficiency is now *visible and expressible*: nest #1 writes 6 elements into `%alloc_5` only for nest #2 to immediately read them back once each. A whole buffer and a whole loop nest of memory traffic exist purely as glue.

Compared with upstream's listing for this step, the structure is the same (three allocs, six stores, two nests, `toy.print` on a memref), but three details differ, and the output above is what MLIR 20 prints:

- **Value names.** Upstream shows `%0 = memref.alloc()`, `%1`, `%2`; MLIR 20 prints `%alloc`, `%alloc_5`, `%alloc_6`, because `memref.alloc` implements `OpAsmOpInterface::getAsmResultNames` (declared in `MemRefOps.td`) to suggest the name `alloc`.
- **Constant order.** Upstream lists `1.0` … `6.0`; here `%cst` is `6.0` and `%cst_4` is `1.0`. The lowering emits them in `1.0` … `6.0` order, interleaved with the stores, and the post-lowering `canonicalize` hoists them to the top in reverse (both dumps in section 3.6).
- **One load, not two, in the mul nest.** Upstream's listing shows two identical `affine.load`s feeding `arith.mulf`. The upstream example driver runs `cse` after the lowering too (`toyc.cpp` is identical here), and upstream's own test expects a single load: its `CHECK` lines require `arith.mulf [[VAL_14]], [[VAL_14]]`, which passes on this output (section 6.7). The upstream listing is simply older than the code, so upstream's remark that the `toy.mul` lowering "has generated some redundant loads" describes the IR *before* `cse`.

### 6.6 Run 3: `-emit=mlir-affine -opt` — after LoopFusion + AffineScalarReplacement

This is upstream's "Taking Advantage of Affine Optimization": the same lowering with `affine-loop-fusion` and `affine-scalrep` added (section 5).

```bash
cd /Users/roy/study/mlir/toy/Ch5
./build/toyc-ch5 ../../test_Example/Toy/Ch5/affine-lowering.mlir -emit=mlir-affine -opt 2>&1
```

Real output:

```mlir
module {
  func.func @main() {
    %cst = arith.constant 6.000000e+00 : f64
    %cst_0 = arith.constant 5.000000e+00 : f64
    %cst_1 = arith.constant 4.000000e+00 : f64
    %cst_2 = arith.constant 3.000000e+00 : f64
    %cst_3 = arith.constant 2.000000e+00 : f64
    %cst_4 = arith.constant 1.000000e+00 : f64
    %alloc = memref.alloc() : memref<3x2xf64>
    %alloc_5 = memref.alloc() : memref<2x3xf64>
    affine.store %cst_4, %alloc_5[0, 0] : memref<2x3xf64>
    affine.store %cst_3, %alloc_5[0, 1] : memref<2x3xf64>
    affine.store %cst_2, %alloc_5[0, 2] : memref<2x3xf64>
    affine.store %cst_1, %alloc_5[1, 0] : memref<2x3xf64>
    affine.store %cst_0, %alloc_5[1, 1] : memref<2x3xf64>
    affine.store %cst, %alloc_5[1, 2] : memref<2x3xf64>
    affine.for %arg0 = 0 to 3 {
      affine.for %arg1 = 0 to 2 {
        %0 = affine.load %alloc_5[%arg1, %arg0] : memref<2x3xf64>
        %1 = arith.mulf %0, %0 : f64
        affine.store %1, %alloc[%arg0, %arg1] : memref<3x2xf64>
      }
    }
    toy.print %alloc : memref<3x2xf64>
    memref.dealloc %alloc_5 : memref<2x3xf64>
    memref.dealloc %alloc : memref<3x2xf64>
    return
  }
}
```

Only two buffers remain — `%alloc` for the final result and `%alloc_5` for the constant — and a single loop nest reads the constant with transposed indices, multiplies inline, and stores the result. This matches upstream's optimized listing apart from the value names and the constant order explained in section 6.5.

### 6.7 What exactly did `-opt` change?

Diff-style, run 2 → run 3 (abridged; types trimmed from the loop body):

```diff
   %alloc   = memref.alloc() : memref<3x2xf64>
-  %alloc_5 = memref.alloc() : memref<3x2xf64>     // transpose's intermediate buffer
-  %alloc_6 = memref.alloc() : memref<2x3xf64>
+  %alloc_5 = memref.alloc() : memref<2x3xf64>     // only constant + result remain
   ... (6 affine.store constant initializers, unchanged apart from renumbering) ...
-  affine.for %arg0 = 0 to 3 {                     // nest #1: transpose
-    affine.for %arg1 = 0 to 2 {
-      %0 = affine.load %alloc_6[%arg1, %arg0]
-      affine.store %0, %alloc_5[%arg0, %arg1]
-    }
-  }
-  affine.for %arg0 = 0 to 3 {                     // nest #2: mul
+  affine.for %arg0 = 0 to 3 {                     // single fused nest
     affine.for %arg1 = 0 to 2 {
-      %0 = affine.load %alloc_5[%arg0, %arg1]     // read of intermediate
+      %0 = affine.load %alloc_5[%arg1, %arg0]     // read constant directly, transposed
       %1 = arith.mulf %0, %0 : f64
       affine.store %1, %alloc[%arg0, %arg1]
     }
   }
   toy.print %alloc : memref<3x2xf64>
-  memref.dealloc %alloc_6 : memref<2x3xf64>
-  memref.dealloc %alloc_5 : memref<3x2xf64>
+  memref.dealloc %alloc_5 : memref<2x3xf64>
   memref.dealloc %alloc : memref<3x2xf64>
```

Step by step:

1. **`affine-loop-fusion`** proves the transpose nest's only consumer is the mul nest with an identical iteration space, and fuses them: the transposed load, the store to the intermediate, the reload from the intermediate, and the multiply all land in one loop body. Fusion alone does *not* remove the intermediate — it is still a full `memref<3x2xf64>` after this pass (section 6.9 shows the IR after fusion only).
2. **`affine-scalrep` (AffineScalarReplacement, a.k.a. memref dataflow opt)** then sees, inside the fused body, `affine.store %0, %intermediate[...]` immediately followed by `affine.load %intermediate[...]` at the same index. It forwards the stored value to the load, which makes the load — and then the store, and then the **entire intermediate `memref<3x2xf64>` alloc/dealloc pair** — dead, and deletes them all.
3. Net effect: **2 loop nests → 1**, **3 buffers → 2**, and the loop body reads the constant buffer *directly with transposed indices* — the transpose has dissolved into an access pattern. Per element, memory traffic drops from load+store (transpose nest) plus load+store (mul nest) to a single load + single store.

None of this was expressible before lowering: it is the payoff for descending to the affine level. And notably, `toy.print` sat there unaffected the whole time — high-level and low-level abstractions optimized side by side.

**Checking with FileCheck.** Upstream suggests trying `-emit=mlir-affine` and then adding `-opt`. The `RUN:` lines at the top of the test file encode exactly those two runs as FileCheck tests (`CHECK` prefix = naive lowering, `OPT` prefix = optimized), including the 3-allocs-vs-2-allocs and two-nests-vs-one-nest structure explained above. Both pass against the current binary:

```bash
cd /Users/roy/study/mlir/toy/Ch5
./build/toyc-ch5 ../../test_Example/Toy/Ch5/affine-lowering.mlir -emit=mlir-affine 2>&1 \
  | /opt/homebrew/opt/llvm@20/bin/FileCheck ../../test_Example/Toy/Ch5/affine-lowering.mlir
./build/toyc-ch5 ../../test_Example/Toy/Ch5/affine-lowering.mlir -emit=mlir-affine -opt 2>&1 \
  | /opt/homebrew/opt/llvm@20/bin/FileCheck ../../test_Example/Toy/Ch5/affine-lowering.mlir --check-prefix=OPT
```

(`FileCheck` prints nothing and exits 0 on success.)

### 6.8 More inputs to try

The other files in `test_Example/Toy/Ch5/` lower cleanly as well. Chapter 2's `codegen.toy` program becomes exactly the test input after inlining, shape inference, and canonicalization, so its lowering is identical to runs 2 and 3. `diff` prints nothing and exits 0 for both:

```bash
cd /Users/roy/study/mlir/toy/Ch5
diff <(./build/toyc-ch5 ../../test_Example/Toy/Ch5/codegen.toy -emit=mlir-affine 2>&1) \
     <(./build/toyc-ch5 ../../test_Example/Toy/Ch5/affine-lowering.mlir -emit=mlir-affine 2>&1)
diff <(./build/toyc-ch5 ../../test_Example/Toy/Ch5/codegen.toy -emit=mlir-affine -opt 2>&1) \
     <(./build/toyc-ch5 ../../test_Example/Toy/Ch5/affine-lowering.mlir -emit=mlir-affine -opt 2>&1)
```

`scalar.toy` (`var a<2, 2> = 5.5;`) shows a splat constant: the Chapter 3 reshape folding has already turned the scalar and its reshape into `toy.constant dense<5.500000e+00> : tensor<2x2xf64>` before the lowering runs. `ConstantOpLowering` iterates the splat's four elements and emits four stores, and the post-lowering cleanup leaves a single `arith.constant`:

```bash
cd /Users/roy/study/mlir/toy/Ch5
./build/toyc-ch5 ../../test_Example/Toy/Ch5/scalar.toy -emit=mlir-affine -opt 2>&1
```

Real output:

```mlir
module {
  func.func @main() {
    %cst = arith.constant 5.500000e+00 : f64
    %alloc = memref.alloc() : memref<2x2xf64>
    affine.store %cst, %alloc[0, 0] : memref<2x2xf64>
    affine.store %cst, %alloc[0, 1] : memref<2x2xf64>
    affine.store %cst, %alloc[1, 0] : memref<2x2xf64>
    affine.store %cst, %alloc[1, 1] : memref<2x2xf64>
    toy.print %alloc : memref<2x2xf64>
    memref.dealloc %alloc : memref<2x2xf64>
    return
  }
}
```

`transpose_transpose.toy` and `trivial_reshape.toy` lower to a single constant buffer and a `toy.print`, because the Chapter 3 patterns have already removed the transposes and reshapes before the lowering runs.

### 6.9 The ecosystem view: reproduce `-opt` with stock `mlir-opt`

Takeaway 7 (section 7) says the two optimization passes know nothing about Toy — you can prove it. Print the naive lowering in *generic form* (so the one remaining `toy.print` becomes an opaque `"toy.print"(...)` that stock `mlir-opt` will tolerate — Chapter 2, section 6.7) and pipe it through the Homebrew `mlir-opt` with the same two passes `toyc-ch5 -opt` schedules:

```bash
cd /Users/roy/study/mlir/toy/Ch5
./build/toyc-ch5 ../../test_Example/Toy/Ch5/affine-lowering.mlir -emit=mlir-affine -mlir-print-op-generic 2>&1 \
  | /opt/homebrew/opt/llvm@20/bin/mlir-opt -allow-unregistered-dialect --affine-loop-fusion --affine-scalrep
```

Real output: identical to run 3 (section 6.6) — one fused loop nest, two buffers, the intermediate `memref<3x2xf64>` gone — except that the print passes through untouched as an unregistered op:

```mlir
    "toy.print"(%alloc) : (memref<3x2xf64>) -> ()
```

This is the whole thesis of lowering into shared dialects, demonstrated with a binary that has never heard of Toy. (`--affine-loop-fusion` and `--affine-scalrep` are the registered names of `mlir::affine::createLoopFusionPass()` and `createAffineScalarReplacementPass()` from section 5's pipeline.)

Dropping `--affine-scalrep` isolates what fusion alone does:

```bash
cd /Users/roy/study/mlir/toy/Ch5
./build/toyc-ch5 ../../test_Example/Toy/Ch5/affine-lowering.mlir -emit=mlir-affine -mlir-print-op-generic 2>&1 \
  | /opt/homebrew/opt/llvm@20/bin/mlir-opt -allow-unregistered-dialect --affine-loop-fusion
```

Real output (abridged to the fused nest; the three allocs, constant stores, and three deallocs are the same as in run 2):

```mlir
    affine.for %arg0 = 0 to 3 {
      affine.for %arg1 = 0 to 2 {
        %0 = affine.load %alloc_6[%arg1, %arg0] : memref<2x3xf64>
        affine.store %0, %alloc_5[%arg0, %arg1] : memref<3x2xf64>
        %1 = affine.load %alloc_5[%arg0, %arg1] : memref<3x2xf64>
        %2 = arith.mulf %1, %1 : f64
        affine.store %2, %alloc[%arg0, %arg1] : memref<3x2xf64>
      }
    }
```

The store-then-reload through `%alloc_5` is exactly what `affine-scalrep` forwards and deletes (section 6.7, step 2).

---

## 7. Key Takeaways & Pitfalls

**Takeaways**

1. **Partial lowering is a first-class strategy in MLIR.** You don't lower the whole program; you lower the parts that benefit, and dialects mix freely in one module (`toy.print` consuming a `memref` produced by affine code).
2. **The dialect conversion framework = ConversionTarget + patterns (+ TypeConverter).** Legality is a declared, enforced postcondition — unlike the greedy driver, the conversion *fails loudly* if an illegal op survives.
3. **`applyPartialConversion` ignores unlisted ops; `applyFullConversion` legalizes everything.** Illegal-but-unconverted ops fail in both.
4. **Use the adaptor, not the op, for operands in conversion patterns.** The adaptor holds the type-remapped values (tensor→memref); `op->getOperands()` holds stale ones.
5. **Dynamic legality (`addDynamicallyLegalOp`) is the idiom for "keep this op but fix its operands"** — pair it with an in-place pattern (`modifyOpInPlace` + `setOperands`) and a loosened ODS type constraint (`AnyTypeOf<[F64Tensor, F64MemRef]>`).
6. **Factor loop-nest lowering** (`lowerOpToLoops` + a per-iteration callback) so element-wise ops differ only in their innermost expression; transpose is just "load with reversed ivs".
7. **Lowering unlocks reuse**: two off-the-shelf passes (`affine-loop-fusion`, `affine-scalrep`) removed a loop nest and a whole buffer without knowing anything about Toy. That reuse is the entire point of lowering into *shared* dialects rather than straight to LLVM.

**Pitfalls**

- **Forgetting `getDependentDialects`.** The pass creates `affine`/`memref`/`func` ops in a context that only has `toy` loaded → ``LLVM ERROR: Building op `func.func` but it isn't known in this MLIRContext: the dialect may not be loaded ...`` and an abort (section 4.2). Declare every dialect your pass *creates* ops from.
- **Following upstream's snippets literally.** `adaptor.input()`, `mlir::AffineLoadOp`, a `PatternRewriter &` loop callback, and passing `patterns` as an lvalue to `applyPartialConversion` all fail to compile against MLIR 20 (sections 3.4 and 4.1); `type.isa<T>()` only warns (section 2.1). The repo's code is the current form.
- **Nesting post-lowering passes on the wrong function op.** Before lowering it's `pm.nest<toy::FuncOp>()`, after it must be `pm.nest<func::FuncOp>()` — get it wrong and the passes silently do nothing.
- **Order matters in the driver.** Inlining and shape inference are *prerequisites* of this lowering, not optimizations: `FuncOpLowering` rejects non-`main` functions, `ReturnOpLowering` rejects returns with operands, and `convertTensorToMemRef` requires ranked, static shapes. That's why `-emit=mlir-affine` forces those passes even without `-opt`.
- **Reading operands from the op inside a ConversionPattern.** You'll get the old tensor-typed values and build ill-typed IR (e.g. `affine.load` from a `tensor`). Always go through `operands`/`OpAdaptor`.
- **Mutating an op without telling the rewriter.** In a conversion, direct mutation corrupts the driver's rollback tracking — wrap in-place updates in `rewriter.modifyOpInPlace(...)` as `PrintOpLowering` does.
- **Expecting `notifyMatchFailure` text in the error.** In this release build a failed pattern surfaces only as "failed to legalize operation ..." (section 3.7); the reason is visible only in a debug build.
- **The simplistic alloc/dealloc placement only works for single-block, no-control-flow functions.** With branches or loops at the CFG level you need real bufferization (deallocs on all exit paths, dominance-correct allocs).
- **Don't expect Toy-level `-opt` to help here**: at the tensor level the transpose+mul chain is irreducible; the redundancy only becomes visible (and fixable) after lowering. Pick the abstraction level that *exposes* the optimization you want.

---

## Appendix: What Chapter 5 adds to the build

Almost nothing — and that is itself the lesson. Relative to Chapter 4, `CMakeLists.txt` changes only by the `ch4` → `ch5` renames (project, executable, and `ToyCh5*IncGen` targets) and its header comments, plus one new source line, `mlir/LowerToAffineLoops.cpp`, in the executable:

***CMakeLists.txt***
```cmake
add_executable(toyc-ch5
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/LowerToAffineLoops.cpp
  mlir/ShapeInferencePass.cpp
  mlir/ToyCombine.cpp
  )
```

The TableGen wiring is unchanged from Chapter 4 — `include/toy/CMakeLists.txt` still generates the op/dialect `.inc` files (`ToyCh5OpsIncGen`) and the shape-inference interface (`ToyCh5ShapeInferenceInterfaceIncGen`), and the top-level file still runs the `ToyCombine.td` DRR rewriters (`ToyCh5CombineIncGen`) — and so are the link libraries. No new link libraries are needed because we link the **monolithic Homebrew dylibs** (`libMLIR.dylib` + `libLLVM.dylib`), which already contain the `affine`/`memref`/`arith`/`func` dialects, the dialect conversion framework, and the `LoopFusion`/`AffineScalarReplacement` passes this chapter starts using. If you were linking fine-grained component libraries instead (as the in-tree tutorial does), this chapter is where you would add `MLIRAffineDialect`, `MLIRAffineTransforms`, `MLIRArithDialect`, `MLIRMemRefDialect`, and `MLIRFuncDialect` on top of the Ch4 list — worth knowing if you ever build against a non-monolithic MLIR install.

---

## Links

- Official doc: [Toy Tutorial Ch.5 — Partial Lowering to Lower-Level Dialects for Optimization](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/)
- Dialect conversion reference: [MLIR Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)
- Previous: [Chapter 4 — Enabling Generic Transformation with Interfaces](../Ch4/README.md)
- Next: [Chapter 6 — Lowering to LLVM and CodeGen / JIT](../Ch6/README.md)
- Back to [README](../README.md)
