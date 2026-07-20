# Chapter 5: Partial Lowering to Lower-Level Dialects for Optimization

> Goal: lower the *computationally heavy* Toy operations to a mix of the `affine`, `arith`, and `memref` dialects using MLIR's **dialect conversion framework** — while deliberately keeping `toy.print` in the Toy dialect — and then reuse existing affine optimizations (loop fusion, scalar replacement) on the result.
> Official doc: <https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/>

**Chapter code:** [`Ch5/`](./) — one chapter of the out-of-tree **superbuild** described in the top-level [README](../README.md#repository-layout). The chapter can also still be configured standalone — see §6.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Dialect Conversion Framework](#2-dialect-conversion-framework)
3. [Lowering the Ops](#3-lowering-the-ops)
4. [The Lowering Pass](#4-the-lowering-pass)
5. [The Pipeline in toyc.cpp](#5-the-pipeline-in-toyccpp)
6. [Building](#6-building)
7. [Running and Testing](#7-running-and-testing)
8. [Key Takeaways & Pitfalls](#8-key-takeaways--pitfalls)

---

## 1. Overview

Up to Chapter 4 everything we did — inlining, shape inference, canonicalization — happened *inside* the Toy dialect. That is the right altitude for language-level semantics, but it is the wrong altitude for optimizations like **loop fusion**, **loop tiling**, or **redundant load elimination**. Those need explicit loops and explicit memory, which `toy.mul` and `toy.transpose` deliberately hide.

This chapter performs a **partial lowering**: we translate only the compute-intensive Toy operations into lower-level dialects and leave the rest alone. This is possible because MLIR has *no fixed set of dialects* — **operations from different dialects can freely coexist in the same module, the same function, even the same block**. After this chapter's pass runs, our IR simultaneously contains:

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

The type story is important too: Toy ops work on **value-semantics `tensor`s** (immutable SSA values, no memory location). The lowered code works on **buffer-semantics `memref`s** (mutable, allocated memory). Part of this chapter is bridging that gap.

---

## 2. Dialect Conversion Framework

Chapter 3's canonicalization used the *greedy* pattern driver: apply patterns until fixpoint, no notion of "done-ness". Lowering needs something stronger — a guarantee about **what the IR looks like when the pass finishes**. That is what the **DialectConversion framework** (`mlir/Transforms/DialectConversion.h`) provides. It needs three ingredients (the third is optional):

1. a **ConversionTarget** — the legality contract,
2. a set of **conversion patterns** — how to rewrite illegal ops,
3. optionally a **TypeConverter** — for converting block argument/signature types (we don't need one here; our patterns convert types operand-by-operand).

### 2.1 ConversionTarget: declaring what "legal" means

The target classifies every operation the conversion encounters as *legal* (may remain), *illegal* (must be converted away), or *dynamically legal* (legal only if a predicate holds).

***mlir/LowerToAffineLoops.cpp***

```cpp
ConversionTarget target(getContext());

// Everything in these dialects is a valid result of the lowering.
target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                       arith::ArithDialect, func::FuncDialect,
                       memref::MemRefDialect>();

// The whole Toy dialect must disappear...
target.addIllegalDialect<toy::ToyDialect>();

// ...EXCEPT toy.print, which is legal *if and only if* none of its
// operands is still a TensorType. Op-level rules override dialect-level
// rules, so this carves an exception out of the illegal Toy dialect.
target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
  return llvm::none_of(op->getOperandTypes(),
                       [](Type type) { return llvm::isa<TensorType>(type); });
});
```

Three things to internalize:

- **Op-level legality beats dialect-level legality.** `addDynamicallyLegalOp<toy::PrintOp>` wins over `addIllegalDialect<toy::ToyDialect>`.
- The **dynamic** predicate is what forces the framework to still *fire a pattern* on `toy.print`: right after its producers are lowered, `toy.print`'s operand is a `tensor`-typed value that the conversion has remapped to a `memref` — the op is "illegal until its operands are updated", so the `PrintOpLowering` pattern must run (see §3.7).
- If, at the end of conversion, any operation that is (still) illegal remains, the conversion **fails** and the pass signals failure. Legality is a hard postcondition, not a best-effort goal.

### 2.2 ConversionPattern vs RewritePattern

Conversion patterns look like ordinary `RewritePattern`s with one crucial addition: `matchAndRewrite` receives an extra **operands/adaptor argument** containing the *remapped* operands:

```cpp
LogicalResult
matchAndRewrite(Operation *op, ArrayRef<Value> operands,          // <-- extra
                ConversionPatternRewriter &rewriter) const final;
```

Why does this matter? During conversion the framework maintains a mapping from original values to their converted replacements. When `toy.mul`'s pattern runs, `op->getOperands()` still yields the *old* `tensor<3x2xf64>`-typed values — but the `operands` array (or the typed `OpAdaptor`) yields the **new `memref<3x2xf64>` values** produced by the already-converted defining ops. Any pattern that deals with type changes must read its inputs through the adaptor, never through the op directly.

This codebase uses all three pattern flavors, which is a nice comparison in itself:

| Pattern base | Used by | Operand access | When appropriate |
|---|---|---|---|
| `ConversionPattern` (type-erased, matched by op name string) | `BinaryOpLowering`, `TransposeOpLowering` | raw `ArrayRef<Value> operands`, wrapped in an ODS `Adaptor` manually | generic/templated patterns over multiple op types |
| `OpConversionPattern<OpT>` (typed) | `FuncOpLowering`, `PrintOpLowering` | typed `OpAdaptor` parameter | single-op patterns that need remapped operands |
| `OpRewritePattern<OpT>` (plain rewrite pattern) | `ConstantOpLowering`, `ReturnOpLowering` | `op` directly (no adaptor) | ops whose lowering doesn't consume remapped operands: `toy.constant` has *no* operands, `toy.return` (post-inlining) has none either |

Yes — **ordinary `RewritePattern`s can be used inside a dialect conversion**. They just don't get access to the remapped operands, which is fine when there are none.

### 2.3 applyPartialConversion vs applyFullConversion

***mlir/LowerToAffineLoops.cpp***

```cpp
if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
  signalPassFailure();
```

- **`applyFullConversion`**: *every* operation reachable in the target region must be legal at the end. Anything not explicitly legal is a failure. Use when the output must be a closed set of dialects (e.g., final LLVM lowering in Chapter 6).
- **`applyPartialConversion`**: operations that are neither legal nor illegal (i.e., *unknown* to the target) are simply left alone. Only **explicitly illegal** ops that survive cause failure. This is the natural fit for progressive lowering, where we knowingly emit a mixed-dialect module.

Note the subtlety: even with partial conversion, our pass is strict about Toy — we marked the whole dialect illegal, so a stray `toy.reshape` that has no pattern would abort the pass with a diagnostic like `failed to legalize operation 'toy.reshape'`. Partial conversion is not "ignore what you can't convert"; it is "ignore what I didn't classify".

### 2.4 The buffer question: three design options

Lowering `tensor` (value) to `memref` (buffer) while some consumers stay tensor-typed poses a design question the official doc calls out. When a still-Toy op (like `toy.print`) consumes a lowered value, we could:

1. **Materialize a tensor back from the buffer** (insert a "load whole buffer" op). Works, but inserts hidden copies the optimizer can't see through.
2. **Add a second variant of the op** that operates on memrefs (`toy.print_memref`). No hidden copies, but duplicates op definitions and is bad for progressive lowering.
3. **Loosen the op's type constraints** so it accepts both. This is what the tutorial does — `PrintOp` in `Ch5/include/toy/Ops.td` declares:

***include/toy/Ops.td***

```tablegen
let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);
```

Option 3 keeps a single op, no copies, and the op simply "follows" the lowering of its operand. The cost: the op's verifier is now weaker (it can't insist on tensors), which is an acceptable trade for an op that only exists mid-pipeline.

---

## 3. Lowering the Ops

Everything below is from [`Ch5/mlir/LowerToAffineLoops.cpp`](mlir/LowerToAffineLoops.cpp) — read it top to bottom alongside this section.

### 3.1 Type conversion helper + `insertAllocAndDealloc`

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

- `tensor<3x2xf64>` maps 1:1 to `memref<3x2xf64>` — this is possible only because **shape inference (Chapter 4) already made every tensor statically ranked and shaped**. The whole lowering asserts this precondition (see the `llvm::cast<RankedTensorType>` calls: unranked types would assert).
- Every lowered op that produces a tensor result gets a fresh buffer. `alloc` is hoisted to the top of the block and `dealloc` sunk to just before the terminator — a trivially correct placement **only because Toy functions have no control flow** (single block). Real bufferization pipelines (the upstream `one-shot-bufferize`) have to solve this properly.
- This is why the unoptimized output in §7 has three `memref.alloc`s at the top and three `memref.dealloc`s at the bottom, in mirrored order.

### 3.2 `lowerOpToLoops`: the shared loop-nest skeleton

`toy.add`, `toy.mul`, and `toy.transpose` all lower to the same shape of code: *allocate result buffer → perfect loop nest over the result shape → compute one element per iteration → store it*. That skeleton is factored into a helper parameterized by a per-iteration callback:

***mlir/LowerToAffineLoops.cpp***

```cpp
/// Takes a builder, the remapped memref operands, and the loop induction
/// variables; returns the value to store at the current index.
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

  // Create a nest of affine loops, one loop per dimension of the shape.
  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
  affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
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

### 3.3 `BinaryOpLowering` — Add and Mul in one template

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
                     // Wrap the remapped operands in the ODS-generated
                     // adaptor for nice named accessors.
                     typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);

                     auto loadedLhs = builder.create<affine::AffineLoadOp>(
                         loc, binaryAdaptor.getLhs(), loopIvs);
                     auto loadedRhs = builder.create<affine::AffineLoadOp>(
                         loc, binaryAdaptor.getRhs(), loopIvs);

                     return builder.create<LoweredBinaryOp>(loc, loadedLhs,
                                                            loadedRhs);
                   });
    return success();
  }
};
using AddOpLowering = BinaryOpLowering<toy::AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<toy::MulOp, arith::MulFOp>;
```

- One template covers both element-wise ops; only the scalar op (`arith.addf` vs `arith.mulf`) differs.
- Note `BinaryOp::Adaptor binaryAdaptor(memRefOperands)`: the ODS adaptor is constructed **over the remapped operands**, so `getLhs()`/`getRhs()` return `memref`-typed values, not the original tensors. This is the ConversionPattern discipline from §2.2 in action.
- Per iteration `[i][j]`: load lhs element, load rhs element, multiply/add, and (via `lowerOpToLoops`) store into the result buffer.

### 3.4 `ConstantOpLowering` — a constant becomes a buffer full of stores

***mlir/LowerToAffineLoops.cpp***

```cpp
struct ConstantOpLowering : public OpRewritePattern<toy::ConstantOp> {
  using OpRewritePattern<toy::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(toy::ConstantOp op,
                                PatternRewriter &rewriter) const final {
    DenseElementsAttr constantValue = op.getValue();
    Location loc = op.getLoc();

    // Allocate a buffer to hold the constant values.
    auto tensorType = llvm::cast<RankedTensorType>(op.getType());
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // Pre-create index constants up to the largest dimension so we don't
    // emit a fresh arith.constant index for every single store.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;
    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(0, *llvm::max_element(valueShape)))
        constantIndices.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, i));
    } else {
      // Rank-0 tensor: a single scalar at index 0.
      constantIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }

    // Recursively walk the dimensions; at the innermost level, emit
    //   affine.store (arith.constant <element>), alloc[indices]
    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.value_begin<FloatAttr>();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      if (dimension == valueShape.size()) {
        rewriter.create<affine::AffineStoreOp>(
            loc, rewriter.create<arith::ConstantOp>(loc, *valueIt++), alloc,
            llvm::ArrayRef(indices));
        return;
      }
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };
    storeElements(/*dimension=*/0);

    rewriter.replaceOp(op, alloc);
    return success();
  }
};
```

Why a *series of stores* rather than something like `memref.global`? Simplicity and transparency:

- A `toy.constant` is a **multi-dimensional dense value attribute**. At the buffer level there is no single op for "buffer initialized with these values" in this simple pipeline, so we unroll it: one scalar `arith.constant` + one `affine.store` per element.
- The stores use **constant indices** (`affine.store %cst, %alloc[0, 1]`), which the affine dialect prints in folded form. That constant-index property is exactly what later lets `AffineScalarReplacement` forward the stored values to loads.
- Making the whole thing visible as plain stores means downstream affine analyses can reason about the initialization precisely — no opaque "magic constant buffer".
- This is an `OpRewritePattern`, not a `ConversionPattern`: `toy.constant` has zero operands, so there is nothing to remap.

For our 2x3 constant this produces 6 `arith.constant f64` ops + 6 `affine.store`s (plus the shared `arith.constant index` values, which the affine store folds into its map and canonicalization then deletes).

### 3.5 `TransposeOpLowering` — reversed induction variables

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
                     toy::TransposeOpAdaptor transposeAdaptor(memRefOperands);
                     Value input = transposeAdaptor.getInput();

                     // Transpose = load from the REVERSED indices.
                     SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
                     return builder.create<affine::AffineLoadOp>(loc, input,
                                                                 reverseIvs);
                   });
    return success();
  }
};
```

The entire semantics of transpose collapses into one line: the loop nest iterates over the **output** shape `(i, j)`, and each iteration loads `input[j, i]` — `llvm::reverse(loopIvs)`. Store side is handled by `lowerOpToLoops` at `[i, j]`. In the output IR:

```mlir
affine.for %arg0 = 0 to 3 {
  affine.for %arg1 = 0 to 2 {
    %0 = affine.load %alloc_6[%arg1, %arg0] : memref<2x3xf64>   // reversed!
    affine.store %0, %alloc_5[%arg0, %arg1] : memref<3x2xf64>
  }
}
```

### 3.6 `FuncOpLowering` and `ReturnOpLowering` — the structural ops

***mlir/LowerToAffineLoops.cpp***

```cpp
struct FuncOpLowering : public OpConversionPattern<toy::FuncOp> {
  using OpConversionPattern<toy::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // We only lower main; everything else should have been inlined.
    if (op.getName() != "main")
      return failure();

    // main must have no inputs/results at this point.
    if (op.getNumArguments() || op.getFunctionType().getNumResults()) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
        diag << "expected 'main' to have 0 inputs and 0 results";
      });
    }

    // Create a func.func with the same name/type and steal the body region.
    auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(),
                                                    op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};
```

```cpp
struct ReturnOpLowering : public OpRewritePattern<toy::ReturnOp> {
  using OpRewritePattern<toy::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(toy::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    // All calls were inlined, so returns carry no operand anymore.
    if (op.hasOperand())
      return failure();

    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};
```

- `toy.func @main` → `func.func @main`: the body region is *moved* (`inlineRegionBefore`), not cloned. This lowering **depends on the inliner having run first** — any non-main `toy.func` makes the pattern fail, and since `toy.func` is illegal, the pass fails. Same for a `toy.return` with an operand.
- `notifyMatchFailure` attaches a human-readable reason that shows up with `-debug`/remarks — much nicer to debug than a bare `failure()`.

### 3.7 `PrintOpLowering` — the op that *doesn't* get lowered

***mlir/LowerToAffineLoops.cpp***

```cpp
struct PrintOpLowering : public OpConversionPattern<toy::PrintOp> {
  using OpConversionPattern<toy::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // We don't lower "toy.print" in this pass, but we need to update its
    // operands (tensor -> memref).
    rewriter.modifyOpInPlace(op,
                             [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};
```

This is the counterpart of the `addDynamicallyLegalOp` rule from §2.1 and the loosened ODS type from §2.4. The pattern:

1. does **not** replace or erase the op — it survives as `toy.print`;
2. swaps its operands for the adaptor's remapped ones (the `memref` result of the lowered `toy.mul`), inside `modifyOpInPlace` so the conversion driver correctly tracks the mutation;
3. after the swap, the dynamic legality predicate (`no TensorType operands`) becomes true, so the framework now considers the op legal and stops worrying about it.

Result in the output IR: `toy.print %alloc : memref<3x2xf64>` — a Toy op holding a MemRef value. Mixed dialects, working together.

---

## 4. The Lowering Pass

The pass wrapper at the bottom of `LowerToAffineLoops.cpp`:

***mlir/LowerToAffineLoops.cpp***

```cpp
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
```

- **`OperationPass<ModuleOp>`**: unlike Chapter 4's shape-inference pass (which ran on `toy::FuncOp`), this runs on the whole module — it has to, because it replaces `toy.func` itself with `func.func`.
- **`getDependentDialects` is mandatory here.** The pass *creates* ops from dialects (`affine`, `func`, `memref`) that may not be loaded in the context yet — the input module only uses `toy`. Declaring dependent dialects makes the pass manager load them before the (multithreaded) pass runs. Forgetting this is a classic crash: `"op created with unregistered dialect"`. (`arith` gets loaded transitively; listing it too would be harmless and arguably more correct.)
- `getArgument()` gives the pass a command-line name (`-toy-to-affine`) if it's ever registered with an opt-style tool.

`runOnOperation` is exactly the three-step recipe from §2:

```cpp
void ToyToAffineLoweringPass::runOnOperation() {
  // 1. Define the conversion target (legal/illegal/dynamically-legal).
  ConversionTarget target(getContext());
  target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                         arith::ArithDialect, func::FuncDialect,
                         memref::MemRefDialect>();
  target.addIllegalDialect<toy::ToyDialect>();
  target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(),
                         [](Type type) { return llvm::isa<TensorType>(type); });
  });

  // 2. Populate the patterns that legalize Toy ops.
  RewritePatternSet patterns(&getContext());
  patterns.add<AddOpLowering, ConstantOpLowering, FuncOpLowering, MulOpLowering,
               PrintOpLowering, ReturnOpLowering, TransposeOpLowering>(
      &getContext());

  // 3. Apply a PARTIAL conversion; fail the pass if any illegal op survives.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::toy::createLowerToAffinePass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}
```

The factory `createLowerToAffinePass()` is declared in `Ch5/include/toy/Passes.h` and consumed by the driver.

---

## 5. The Pipeline in toyc.cpp

[`Ch5/toyc.cpp`](toyc.cpp) grows a new action this chapter:

***toyc.cpp***

```cpp
enum Action { None, DumpAST, DumpMLIR, DumpMLIRAffine };

static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine",
                          "output the MLIR dump after affine lowering")));
```

The interesting part of `dumpMLIR()` is how the pipeline is assembled incrementally:

```cpp
// Check to see what granularity of MLIR we are compiling to.
bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;

if (enableOpt || isLoweringToAffine) {
  // Inline all functions into main and then delete them.
  pm.addPass(mlir::createInlinerPass());

  // Now that there is only one function, infer shapes (Ch4), then clean up.
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
- The nesting changes across the lowering boundary: the pre-lowering cleanup nests on **`toy::FuncOp`**, the post-lowering cleanup nests on **`func::FuncOp`** — because the lowering pass renamed the function op in between. Nesting on the wrong op type silently runs the passes on nothing.
- Post-lowering, `canonicalize` + `cse` always run: canonicalization folds/erases the now-dead `arith.constant index` ops and simplifies affine maps; CSE dedupes repeated loads *within* what it can prove.
- With `-opt`, two affine-level optimizations run:
  - **`affine::createLoopFusionPass()`** (`affine-loop-fusion`): fuses producer/consumer affine loop nests to improve locality — here it fuses the transpose nest into the multiply nest, and because the intermediate buffer then has only a single point-wise use, fusion also shrinks/eliminates it.
  - **`affine::createAffineScalarReplacementPass()`** (`affine-scalrep`, historically "memref dataflow opt"): forwards stored values to subsequent loads, deletes redundant loads and dead stores, and erases memrefs that end up unused. This is what removes the store-then-load through the intermediate buffer after fusion.

Both dumps then come from the same `module->dump()` at the end — `-emit=mlir` vs `-emit=mlir-affine` differ only in which passes were added.

---

## 6. Building

The shared machinery (superbuild, presets, dual-mode guard, `build.sh`) is documented in the top-level [README, "The build system"](../README.md#the-build-system). Build this chapter with `cd toy && ./build.sh ch5` → `./build/bin/toyc-ch5`.

### 6.1 What Chapter 5 adds to the build: almost nothing

That is itself the lesson. Relative to Chapter 4, [`Ch5/CMakeLists.txt`](CMakeLists.txt) changes by exactly one line:

***CMakeLists.txt***

```cmake
add_executable(toyc-ch5
  ...
  mlir/LowerToAffineLoops.cpp   # <-- NEW this chapter
  ...
  )
```

Nothing else moves — no new TableGen, no new link libraries — because we link the **monolithic Homebrew dylibs** (`libMLIR.dylib` + `libLLVM.dylib`), which already contain the `affine`/`memref`/`arith`/`func` dialects, the dialect conversion framework, and the `LoopFusion`/`AffineScalarReplacement` passes this chapter starts using. If you were linking fine-grained component libraries instead (as the in-tree tutorial does), this chapter is where you would add `MLIRAffineDialect`, `MLIRAffineTransforms`, `MLIRArithDialect`, `MLIRMemRefDialect`, and `MLIRFuncDialect` on top of the Ch4 list — worth knowing if you ever build against a non-monolithic MLIR install.

---

## 7. Running and Testing

The shared [`run.sh`](../run.sh) at the repo root drives this chapter (it looks for the binary in `build/bin/` first, then falls back to a standalone `Ch5/build/`):

```bash
cd toy
./run.sh ch5
```

which exercises [`test_Example/Toy/Ch5/affine-lowering.mlir`](../../test_Example/Toy/Ch5/affine-lowering.mlir) three ways (equivalent direct invocations, run from `toy/`):

```bash
./build/bin/toyc-ch5 ../test_Example/Toy/Ch5/affine-lowering.mlir -emit=mlir -opt
./build/bin/toyc-ch5 ../test_Example/Toy/Ch5/affine-lowering.mlir -emit=mlir-affine
./build/bin/toyc-ch5 ../test_Example/Toy/Ch5/affine-lowering.mlir -emit=mlir-affine -opt
```

### 7.1 The input

The test input is already Toy-dialect MLIR (note: `toyc-ch5` auto-detects `.mlir` input, no `-x mlir` needed). It also carries the `FileCheck` expectations used by the LIT-style `RUN:` lines at the top:

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

i.e. `print(transpose(constant)²)` — one constant, one transpose, one element-wise multiply.

### 7.2 Run 1: `-emit=mlir -opt` — the Toy-level baseline (real output)

```bash
cd toy
./build/bin/toyc-ch5 ../test_Example/Toy/Ch5/affine-lowering.mlir -emit=mlir -opt 2>&1
```

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

Everything is still Toy. The Chapter-3 `transpose(transpose(x)) = x` pattern doesn't apply (there's only one transpose), so `-opt` at this level is essentially a no-op for this input. This is precisely the motivation for the chapter: **at the Toy level there is nothing left to optimize** — the redundancy we're about to expose (two loop nests, an intermediate buffer, double loads) doesn't even *exist* yet as a concept.

### 7.3 Run 2: `-emit=mlir-affine` — naive lowering, no `-opt` (real output)

```bash
./build/bin/toyc-ch5 ../test_Example/Toy/Ch5/affine-lowering.mlir -emit=mlir-affine 2>&1
```

```mlir
module {
  func.func @main() {
    %cst = arith.constant 6.000000e+00 : f64
    %cst_0 = arith.constant 5.000000e+00 : f64
    %cst_1 = arith.constant 4.000000e+00 : f64
    %cst_2 = arith.constant 3.000000e+00 : f64
    %cst_3 = arith.constant 2.000000e+00 : f64
    %cst_4 = arith.constant 1.000000e+00 : f64
    %alloc = memref.alloc() : memref<3x2xf64>       // result of toy.mul
    %alloc_5 = memref.alloc() : memref<3x2xf64>     // result of toy.transpose (intermediate!)
    %alloc_6 = memref.alloc() : memref<2x3xf64>     // buffer for toy.constant
    affine.store %cst_4, %alloc_6[0, 0] : memref<2x3xf64>   // constant lowering:
    affine.store %cst_3, %alloc_6[0, 1] : memref<2x3xf64>   // one store per element
    affine.store %cst_2, %alloc_6[0, 2] : memref<2x3xf64>
    affine.store %cst_1, %alloc_6[1, 0] : memref<2x3xf64>
    affine.store %cst_0, %alloc_6[1, 1] : memref<2x3xf64>
    affine.store %cst, %alloc_6[1, 2] : memref<2x3xf64>
    affine.for %arg0 = 0 to 3 {                      // loop nest #1: transpose
      affine.for %arg1 = 0 to 2 {
        %0 = affine.load %alloc_6[%arg1, %arg0] : memref<2x3xf64>   // reversed ivs
        affine.store %0, %alloc_5[%arg0, %arg1] : memref<3x2xf64>
      }
    }
    affine.for %arg0 = 0 to 3 {                      // loop nest #2: mul
      affine.for %arg1 = 0 to 2 {
        %0 = affine.load %alloc_5[%arg0, %arg1] : memref<3x2xf64>
        %1 = arith.mulf %0, %0 : f64
        affine.store %1, %alloc[%arg0, %arg1] : memref<3x2xf64>
      }
    }
    toy.print %alloc : memref<3x2xf64>               // STILL a toy op, memref operand
    memref.dealloc %alloc_6 : memref<2x3xf64>
    memref.dealloc %alloc_5 : memref<3x2xf64>
    memref.dealloc %alloc : memref<3x2xf64>
    return
  }
}
```

Annotations, mapping back to §3:

- **Three buffers** (`insertAllocAndDealloc`): one per value-producing Toy op — constant (`%alloc_6`, 2x3), transpose (`%alloc_5`, 3x2), mul (`%alloc`, 3x2). All allocs hoisted to the block top, all deallocs sunk to the bottom, in mirror order.
- **Constant → 6 scalar `arith.constant` + 6 `affine.store`** with folded constant indices (`[0, 0]`, `[0, 1]`, …) — `ConstantOpLowering`. The `arith.constant index` helpers were folded into the affine maps and cleaned up by the always-on post-lowering `canonicalize`.
- **Loop nest #1** iterates the *output* shape (3x2) and loads with reversed ivs `[%arg1, %arg0]` — `TransposeOpLowering`.
- **Loop nest #2**: `lhs` and `rhs` of `toy.mul` are the *same* value, so the always-on `cse` already merged the two `affine.load`s from `BinaryOpLowering` into one (the upstream tutorial shows two loads because it dumps before CSE).
- **`toy.print %alloc : memref<3x2xf64>`** — the dynamically-legal op with operands rewritten by `PrintOpLowering`.
- `toy.func`/`toy.return` became `func.func`/`return`.

The inefficiency is now *visible and expressible*: nest #1 writes 6 elements into `%alloc_5` only for nest #2 to immediately read them back once each. A whole buffer and a whole loop nest of memory traffic exist purely as glue.

### 7.4 Run 3: `-emit=mlir-affine -opt` — after LoopFusion + AffineScalarReplacement (real output)

```bash
./build/bin/toyc-ch5 ../test_Example/Toy/Ch5/affine-lowering.mlir -emit=mlir-affine -opt 2>&1
```

```mlir
module {
  func.func @main() {
    %cst = arith.constant 6.000000e+00 : f64
    %cst_0 = arith.constant 5.000000e+00 : f64
    %cst_1 = arith.constant 4.000000e+00 : f64
    %cst_2 = arith.constant 3.000000e+00 : f64
    %cst_3 = arith.constant 2.000000e+00 : f64
    %cst_4 = arith.constant 1.000000e+00 : f64
    %alloc = memref.alloc() : memref<3x2xf64>       // final result only
    %alloc_5 = memref.alloc() : memref<2x3xf64>     // constant buffer only
    affine.store %cst_4, %alloc_5[0, 0] : memref<2x3xf64>
    affine.store %cst_3, %alloc_5[0, 1] : memref<2x3xf64>
    affine.store %cst_2, %alloc_5[0, 2] : memref<2x3xf64>
    affine.store %cst_1, %alloc_5[1, 0] : memref<2x3xf64>
    affine.store %cst_0, %alloc_5[1, 1] : memref<2x3xf64>
    affine.store %cst, %alloc_5[1, 2] : memref<2x3xf64>
    affine.for %arg0 = 0 to 3 {                      // ONE fused loop nest
      affine.for %arg1 = 0 to 2 {
        %0 = affine.load %alloc_5[%arg1, %arg0] : memref<2x3xf64>  // transposed read
        %1 = arith.mulf %0, %0 : f64                               // multiply inline
        affine.store %1, %alloc[%arg0, %arg1] : memref<3x2xf64>    // store result
      }
    }
    toy.print %alloc : memref<3x2xf64>
    memref.dealloc %alloc_5 : memref<2x3xf64>
    memref.dealloc %alloc : memref<3x2xf64>
    return
  }
}
```

### 7.5 What exactly did `-opt` change?

Diff-style, run 2 → run 3:

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

1. **`affine-loop-fusion`** proves the transpose nest's only consumer is the mul nest with identical iteration space, and fuses them: the transposed load, the store to the intermediate, the reload from the intermediate, and the multiply all land in one loop body. As part of fusion it privatizes the producer's target buffer to the fused nest.
2. **`affine-scalrep` (AffineScalarReplacement, a.k.a. memref dataflow opt)** then sees, inside the fused body, `affine.store %0, %intermediate[...]` immediately followed by `affine.load %intermediate[...]` at the same index. It forwards the stored value to the load, which makes the load — and then the store, and then the **entire intermediate `memref<3x2xf64>` alloc/dealloc pair** — dead, and deletes them all.
3. Net effect: **2 loop nests → 1**, **3 buffers → 2**, and the loop body reads the constant buffer *directly with transposed indices* — the transpose has dissolved into an access pattern. Memory traffic per element drops from load+store+load+store (12 accesses through the intermediate for 6 elements, plus final stores) to a single load + single store.

None of this was expressible before lowering: it is the payoff for descending to the affine level. And notably, `toy.print` sat there unaffected the whole time — high-level and low-level abstractions optimized side by side.

The `RUN:` lines at the top of the test file encode runs 2 and 3 as FileCheck tests (`CHECK` prefix = naive lowering, `OPT` prefix = optimized), including exactly the 3-allocs-vs-2-allocs and two-nests-vs-one-nest structure explained above.

**The ecosystem view: reproduce `-opt` with stock `mlir-opt`.** Takeaway 7 below says the two optimization passes know nothing about Toy — you can prove it. Print the naive lowering in *generic form* (so the one remaining `toy.print` becomes an opaque `"toy.print"(...)` that stock `mlir-opt` will tolerate — see Ch2 §7.6) and pipe it through the Homebrew `mlir-opt` with the same two passes `toyc-ch5 -opt` schedules:

```bash
./build/bin/toyc-ch5 ../test_Example/Toy/Ch5/affine-lowering.mlir -emit=mlir-affine -mlir-print-op-generic 2>&1 \
  | /opt/homebrew/opt/llvm@20/bin/mlir-opt -allow-unregistered-dialect --affine-loop-fusion --affine-scalrep
```

Verified on this machine: the stock tool produces the same result as run 3 — one fused loop nest, two buffers, the intermediate `memref<3x2xf64>` gone — with `"toy.print"(%alloc)` passing through untouched as an unregistered op. This is the whole thesis of lowering into shared dialects, demonstrated with a binary that has never heard of Toy. (`--affine-loop-fusion` and `--affine-scalrep` are the registered names of `mlir::affine::createLoopFusionPass()` and `createAffineScalarReplacementPass()` from §5's pipeline.)

---

## 8. Key Takeaways & Pitfalls

**Takeaways**

1. **Partial lowering is a first-class strategy in MLIR.** You don't lower the whole program; you lower the parts that benefit, and dialects mix freely in one module (`toy.print` consuming a `memref` produced by affine code).
2. **The dialect conversion framework = ConversionTarget + patterns (+ TypeConverter).** Legality is a declared, enforced postcondition — unlike the greedy driver, the conversion *fails loudly* if an illegal op survives.
3. **`applyPartialConversion` ignores unlisted ops; `applyFullConversion` legalizes everything.** Illegal-but-unconverted ops fail in both.
4. **Use the adaptor, not the op, for operands in conversion patterns.** The adaptor holds the type-remapped values (tensor→memref); `op->getOperands()` holds stale ones.
5. **Dynamic legality (`addDynamicallyLegalOp`) is the idiom for "keep this op but fix its operands"** — pair it with an in-place pattern (`modifyOpInPlace` + `setOperands`) and a loosened ODS type constraint (`AnyTypeOf<[F64Tensor, F64MemRef]>`).
6. **Factor loop-nest lowering** (`lowerOpToLoops` + a per-iteration callback) so element-wise ops differ only in their innermost expression; transpose is just "load with reversed ivs".
7. **Lowering unlocks reuse**: two off-the-shelf passes (`affine-loop-fusion`, `affine-scalrep`) removed a loop nest and a whole buffer without knowing anything about Toy. That reuse is the entire point of lowering into *shared* dialects rather than straight to LLVM.

**Pitfalls**

- **Forgetting `getDependentDialects`.** The pass creates `affine`/`memref`/`func` ops in a context that may only have `toy` loaded → assertion/crash at runtime. Declare every dialect your pass *creates* ops from.
- **Nesting post-lowering passes on the wrong function op.** Before lowering it's `pm.nest<toy::FuncOp>()`, after it must be `pm.nest<func::FuncOp>()` — get it wrong and the passes silently do nothing.
- **Order matters in the driver.** Inlining and shape inference are *prerequisites* of this lowering, not optimizations: `FuncOpLowering` rejects non-`main` functions, `ReturnOpLowering` rejects returns with operands, and `convertTensorToMemRef` requires ranked, static shapes. That's why `-emit=mlir-affine` forces those passes even without `-opt`.
- **Reading operands from the op inside a ConversionPattern.** You'll get the old tensor-typed values and build ill-typed IR (e.g. `affine.load` from a `tensor`). Always go through `operands`/`OpAdaptor`.
- **Mutating an op without telling the rewriter.** In a conversion, direct mutation corrupts the driver's rollback tracking — wrap in-place updates in `rewriter.modifyOpInPlace(...)` as `PrintOpLowering` does.
- **The simplistic alloc/dealloc placement only works for single-block, no-control-flow functions.** With branches or loops at the CFG level you need real bufferization (deallocs on all exit paths, dominance-correct allocs).
- **Don't expect Toy-level `-opt` to help here**: at the tensor level the transpose+mul chain is irreducible; the redundancy only becomes visible (and fixable) after lowering. Pick the abstraction level that *exposes* the optimization you want.

---

## Links

- Official doc: [Toy Tutorial Ch.5 — Partial Lowering to Lower-Level Dialects for Optimization](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/)
- Dialect conversion reference: [MLIR Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)
- Previous: [Chapter 4 — Enabling Generic Transformation with Interfaces](../Ch4/README.md)
- Next: [Chapter 6 — Lowering to LLVM and CodeGen / JIT](../Ch6/README.md)
- Back to [README](../README.md)
