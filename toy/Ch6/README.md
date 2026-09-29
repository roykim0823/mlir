# Chapter 6: Lowering to LLVM and JIT Compilation

> **Goal:** Complete the lowering journey — take the mixed `toy`/`affine`/`arith`/`memref` IR from Chapter 5 all the way down to the LLVM dialect, translate it to real LLVM IR, and execute it with a JIT — following the official tutorial [Toy Ch-6: Lowering to LLVM and CodeGeneration](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/).

> **Read this first if your Ch6 binary segfaults:** this repo hit (and documents) a real macOS
> linking pitfall where mixing static `.a` MLIR libraries with `libMLIR.dylib` causes a
> **TypeID-duplication segfault** inside `StorageUniquer` on *any* pass run. See
> section 5, the link line in appendix A.2, and the full write-up in
> [MLIR_LINKING_PITFALL.md](MLIR_LINKING_PITFALL.md).

---

## 1. Overview

In Chapter 5 we introduced the dialect conversion framework and *partially* lowered Toy: compute-heavy ops (`toy.transpose`, `toy.mul`, `toy.constant`, …) became `affine` loops over `memref`s, but `toy.print` was deliberately left as-is because we wanted to keep printing at a high level of abstraction as long as possible. Chapter 6 finishes the job and lowers everything to LLVM for code generation. The full pipeline now looks like this:

```
  Toy AST
    │  mlirGen
    ▼
  toy dialect                        (-emit=mlir)
    │  inline → shape-inference → canonicalize → CSE
    │  LowerToAffineLoops (Ch5, partial conversion)
    ▼
  affine + arith + memref + func  (+ toy.print survivor)     (-emit=mlir-affine)
    │  LowerToLLVM (this chapter, FULL conversion)
    │    • affine  → scf            (populateAffineToStdConversionPatterns)
    │    • scf     → cf             (populateSCFToControlFlowConversionPatterns)
    │    • arith   → llvm           (populateArithToLLVMConversionPatterns)
    │    • memref  → llvm           (populateFinalizeMemRefToLLVMConversionPatterns)
    │    • cf      → llvm           (populateControlFlowToLLVMConversionPatterns)
    │    • func    → llvm           (populateFuncToLLVMConversionPatterns)
    │    • toy.print → scf loops + llvm.call @printf   (PrintOpLowering)
    ▼
  llvm dialect (still MLIR!)                                  (-emit=mlir-llvm)
    │  translateModuleToLLVMIR  (MLIR → LLVM IR "translation", not conversion)
    ▼
  LLVM IR                                                     (-emit=llvm)
    │  makeOptimizingTransformer (LLVM -O0/-O3 pipeline)
    │  mlir::ExecutionEngine (ORC JIT)
    ▼
  native code, executed in-process                            (-emit=jit)
```

Three ideas from the official tutorial are worth internalizing before reading the code:

1. **Transitive (A→B→C) lowering.** We never write patterns that go straight from `affine` to `llvm`. Instead, `affine` lowers to `scf`, `scf` lowers to `cf`, and `cf` lowers to `llvm` — each stage reusing patterns that upstream MLIR already provides. The dialect-conversion driver applies all of these pattern sets in one `applyFullConversion` call, chaining them automatically until everything is legal.
2. **Full conversion vs. partial conversion.** Chapter 5 used `applyPartialConversion` (unknown ops may survive). Here we use `applyFullConversion`: after the pass, *only* LLVM-dialect operations (plus the top-level `builtin.module`) may remain. Anything else is a hard error.
3. **The LLVM dialect is still MLIR.** `-emit=mlir-llvm` prints MLIR operations (`llvm.func`, `llvm.call`, `llvm.br`, …) that model LLVM IR 1:1. A separate *translation* step (`translateModuleToLLVMIR`) — not a dialect conversion — produces an actual `llvm::Module`. From that point on we are in classic LLVM land: optimization pipelines, target machines, ORC JIT.

### How this README maps to the upstream chapter

Every topic of the official [Ch-6](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/) is covered here, in the same order, with this repo's real code and output in place of upstream's snippets. Several upstream snippets no longer compile against MLIR 20, and its printed IR predates opaque pointers; each spot says what changed, checked by compiling the snippet or running the tools.

| Upstream section | Here |
|---|---|
| Introduction (recap of Chapter 5) | 1 |
| Lowering to LLVM: lowering `toy.print` first, transitive lowering, `getOrInsertPrintf` | 2.1–2.3 |
| Conversion Target | 2.4 |
| Type Converter | 2.5 |
| Conversion Patterns | 2.6 |
| Full Lowering (`applyFullConversion`, the working example lowered to the LLVM dialect) | 2.7 |
| CodeGen: Getting Out of MLIR · Emitting LLVM IR (`translateModuleToLLVMIR`, `dumpLLVMIR`, `-O3`) | 3.1 |
| Setting up a JIT (`runJit`, `echo ... \| toyc-ch6 -emit=jit`) | 3.2 |
| Playing with `-emit=...` levels, `--mlir-print-ir-after-all`, `llvm-lowering.mlir` | 3.3 (every level run in 4.3–4.7, `llvm-lowering.mlir` run and FileChecked in 4.8) |

Beyond upstream: the pass wrapper and the driver pipeline, including this repo's extra `reconcile-unrealized-casts` pass (2.8); the memref descriptor layout (2.5); replacing the backend with stock `mlir-translate`/`lli` (4.10); and the shared-only link line that avoids the segfault above (appendix).

### Where Chapter 6 lives in this repo

How this repo is organized — the out-of-tree CMake superbuild at `toy/`, the pinned Homebrew LLVM/MLIR 20 toolchain, and the `build.sh`/`run.sh` helpers — is documented once in the top-level [README](../README.md#repository-layout). This chapter also builds as a standalone project (section 4.1). Chapter 6 specifics:

| Item | Location / value |
|---|---|
| Chapter 6 code | `/Users/roy/study/mlir/toy/Ch6/` |
| Build | `cd toy && ./build.sh ch6` → binary at `./build/bin/toyc-ch6` |
| Run | `cd toy && ./run.sh ch6` → pipes `def main() { print([[1, 2], [3, 4]]); }` through `-emit=mlir`, `llvm`, `mlir-affine`, `mlir-llvm`, and `jit` |
| Test inputs | `/Users/roy/study/mlir/test_Example/Toy/Ch6/` (`jit.toy`, `llvm-lowering.mlir`, `codegen.toy`, `affine-lowering.mlir`, …) |

The lexer, parser, AST, `MLIRGen`, dialect, shape inference, combiner patterns, and `LowerToAffineLoops.cpp` are carried over unchanged from Chapter 5. What is new:

| File | Role |
|---|---|
| `mlir/LowerToLLVM.cpp` | `PrintOpLowering` + `ToyToLLVMLoweringPass` (full conversion to the LLVM dialect) |
| `include/toy/Passes.h` | Declares `createLowerToLLVMPass()` |
| `toyc.cpp` | New actions `-emit=mlir-llvm`, `-emit=llvm`, `-emit=jit`; `dumpLLVMIR()` and `runJit()` |
| `CMakeLists.txt` | Links the ExecutionEngine — the interesting (and dangerous) part on macOS |
| [`MLIR_LINKING_PITFALL.md`](MLIR_LINKING_PITFALL.md) | Post-mortem of the static+shared TypeID segfault |

---

## 2. Lowering to LLVM

For this lowering we use the dialect conversion framework again, but this time for a **full** conversion to the LLVM dialect. Everything except `toy.print` was already lowered in Chapter 5, so the chapter first lowers `toy.print` (2.1–2.3) and then assembles the usual three components — conversion target, type converter, patterns — plus the conversion call (2.4–2.7). All code below is the real code from `mlir/LowerToLLVM.cpp` unless labeled otherwise.

### 2.1 Lowering `toy.print` first — transitive lowering and the pattern skeleton

`toy.print` has no direct LLVM equivalent, so we lower it to what a C programmer would write: a non-affine loop nest calling `printf("%f ", elt)` for every element, with a newline after each row. The file header sums up the plan:

***mlir/LowerToLLVM.cpp***
```cpp
// This file implements full lowering of Toy operations to LLVM MLIR dialect.
// 'toy.print' is lowered to a loop nest that calls `printf` on each element of
// the input array. The file also sets up the ToyToLLVMLoweringPass. This pass
// lowers the combination of Arithmetic + Affine + SCF + Func dialects to the
// LLVM one:
//
//                         Affine --
//                                  |
//                                  v
//                       Arithmetic + Func --> LLVM (Dialect)
//                                  ^
//                                  |
//     'toy.print' --> Loop (SCF) --
```

The key trick, and the reason the tutorial introduces **transitive lowering** here: `PrintOpLowering` does **not** emit LLVM-dialect branches directly. It emits `scf.for` loops and `memref.load`s — *higher-level* dialects — and lets the other conversion patterns in the very same `applyFullConversion` lower those to `cf` branches and LLVM GEPs. The conversion framework may apply several patterns in sequence to legalize one operation; as long as a lowering path from `scf` to LLVM exists (section 2.6), the structured loop nest is fine.

**The pattern skeleton.** The start of the pattern and of `matchAndRewrite` (abridged; the loop-nest part is in section 2.3):

***mlir/LowerToLLVM.cpp***
```cpp
namespace {
/// Lowers `toy.print` to a loop nest calling `printf` on each of the individual
/// elements of the array.
class PrintOpLowering : public ConversionPattern {
public:
  explicit PrintOpLowering(MLIRContext *context)
      : ConversionPattern(toy::PrintOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto *context = rewriter.getContext();
    auto memRefType = llvm::cast<MemRefType>((*op->operand_type_begin()));
    auto memRefShape = memRefType.getShape();
    auto loc = op->getLoc();

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    // Get a symbol reference to the printf function, inserting it if necessary.
    auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    Value formatSpecifierCst = getOrCreateGlobalString(
        loc, rewriter, "frmt_spec", StringRef("%f \0", 4), parentModule);
    Value newLineCst = getOrCreateGlobalString(
        loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);
...
```

It is a generic `ConversionPattern` matched by operation name (`toy.print`), with benefit 1. Because Ch5 already ran, the operand of `toy.print` at this point is a **`memref`**, not a tensor — the pattern casts the operand type to `MemRefType` and reads its static shape to build the loop bounds.

#### `toy.print` reaches this pass with a `memref` operand

Chapter 5's partial lowering keeps `toy.print` legal, but only once its operand is no longer a tensor:

***mlir/LowerToAffineLoops.cpp***
```cpp
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
```

Stop the pipeline right before this chapter's pass (`-emit=mlir-affine`) and pick out the survivor:

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/jit.toy -emit=mlir-affine 2>&1 | grep toy.print
```

Real output:

```mlir
    toy.print %alloc : memref<2x2xf64>
```

- `toy.print` is the only `toy` op left, which is why `PrintOpLowering` is the only Toy pattern this chapter needs (section 2.6).
- Its operand `%alloc` is the `memref<2x2xf64>` buffer Chapter 5 allocated for the constant, so the `llvm::cast<MemRefType>` above cannot fail, and `memRefShape` is `[2, 2]`. The whole module at this level is in section 4.4.

### 2.2 The helpers: the `printf` declaration and format strings as `llvm.mlir.global`

Upstream shows how to "get, or build, the declaration for printf" in one function, `getOrInsertPrintf`. This repo splits it into three private static helpers at the end of the class:

***mlir/LowerToLLVM.cpp***
```cpp
private:
  /// Create a function declaration for printf, the signature is:
  ///   * `i32 (i8*, ...)`
  static LLVM::LLVMFunctionType getPrintfType(MLIRContext *context) {
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmPtrTy = LLVM::LLVMPointerType::get(context);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy,
                                                  /*isVarArg=*/true);
    return llvmFnType;
  }

  /// Return a symbol reference to the printf function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get(context, "printf");

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf",
                                      getPrintfType(context));
    return SymbolRefAttr::get(context, "printf");
  }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module) {
    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMArrayType::get(
          IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value),
                                              /*alignment=*/0);
    }

    // Get the pointer to the first character in the global string.
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                  builder.getIndexAttr(0));
    return builder.create<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(builder.getContext()), global.getType(),
        globalPtr, ArrayRef<Value>({cst0, cst0}));
  }
};
```

**`getPrintfType` / `getOrInsertPrintf`.** `printf` is variadic: `i32 (ptr, ...)` (the doc comment still says `i8*`, from before opaque pointers). `getOrInsertPrintf` inserts a body-less `llvm.func` declaration with that type at the top of the module, only if it isn't already there. Its return value is a `FlatSymbolRefAttr` — calls reference the function *by symbol*, not by SSA value, matching how LLVM IR call instructions name their callees. The signature type is factored into `getPrintfType` because the pattern also passes it to every `LLVM::CallOp` it creates (section 2.3); for a variadic callee the call records it, visible as `vararg(!llvm.func<i32 (ptr, ...)>)` on each call in the output of section 2.7.

**`getOrCreateGlobalString`** (not shown in the upstream text, but in upstream's source). C string literals become module-level LLVM globals. It creates (once, memoized by symbol lookup) an internal constant global holding the raw bytes, then computes a pointer to its first character. Step by step:

1. `module.lookupSymbol<LLVM::GlobalOp>(name)` — reuse if a global with that symbol already exists (so ten `toy.print`s still produce one `@frmt_spec`).
2. Otherwise, an `InsertionGuard` temporarily moves the builder to the **start of the module** and creates `llvm.mlir.global internal constant @frmt_spec("%f \00")` of type `!llvm.array<4 x i8>`. Note the explicit `\0` terminators in the `StringRef("%f \0", 4)` literals of section 2.1 — LLVM globals don't NUL-terminate for you.
3. `llvm.mlir.addressof` yields the address of the global, and an `llvm.getelementptr` with indices `[0, 0]` produces the `!llvm.ptr` to the first `i8` — the `getelementptr inbounds [4 x i8], ptr @frmt_spec, i64 0, i64 0` idiom from C compilers.

> **Upstream is outdated here (MLIR 20).** The tutorial's `getOrInsertPrintf` does not compile against the MLIR 20 headers (checked with `clang++ -fsyntax-only`):
>
> - `LLVM::LLVMPointerType::get(IntegerType::get(context, 8))` — typed pointers are gone. The only overload is `LLVMPointerType::get(MLIRContext *context, unsigned addressSpace = 0)` (`LLVMTypes.h.inc`), giving an *opaque* `!llvm.ptr`. That is why the GEP above must carry the element type (`global.getType()`) as a separate argument.
> - `SymbolRefAttr::get("printf", context)` — "no matching function"; the argument order is now `(context, "printf")`.
> - The extra `LLVM::LLVMDialect *llvmDialect` parameter is unused, and this repo's version drops it.

### 2.3 Generating the loop nest

Back in `matchAndRewrite`, one `scf.for` is created per memref dimension:

***mlir/LowerToLLVM.cpp***
```cpp
    // Create a loop for each of the dimensions within the shape.
    SmallVector<Value, 4> loopIvs;
    for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
      auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto upperBound =
          rewriter.create<arith::ConstantIndexOp>(loc, memRefShape[i]);
      auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
      for (Operation &nested : make_early_inc_range(*loop.getBody()))
        rewriter.eraseOp(&nested);
      loopIvs.push_back(loop.getInductionVar());

      // Terminate the loop body.
      rewriter.setInsertionPointToEnd(loop.getBody());

      // Insert a newline after each of the inner dimensions of the shape.
      if (i != e - 1)
        rewriter.create<LLVM::CallOp>(loc, getPrintfType(context), printfRef,
                                      newLineCst);
      rewriter.create<scf::YieldOp>(loc);
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // Generate a call to printf for the current element of the loop.
    auto printOp = cast<toy::PrintOp>(op);
    auto elementLoad =
        rewriter.create<memref::LoadOp>(loc, printOp.getInput(), loopIvs);
    rewriter.create<LLVM::CallOp>(
        loc, getPrintfType(context), printfRef,
        ArrayRef<Value>({formatSpecifierCst, elementLoad}));

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);
    return success();
  }
```

The choreography, iteration by iteration:

- **Bounds:** every dimension gets `arith.constant 0` / `arith.constant <dim>` / step `1` — legal because the arith patterns in the same conversion lower them to `llvm.mlir.constant`.
- **Body cleanup:** `scf::ForOp` auto-creates a body with a terminator; the inner `eraseOp` loop clears it so we control the body exactly. This line differs from upstream's source, which iterates `*loop.getBody()` directly; this repo wraps it in `make_early_inc_range` so the iterator is advanced *before* the current op is erased.
- **Newline placement:** for every dimension *except the innermost*, a `printf(nl)` call is placed at the **end** of that loop's body — i.e. after the entire inner loop has finished a row, print `"\n"`. For our 2×2 matrix: the outer (row) loop ends each iteration with a newline; the inner (column) loop only prints elements.
- **Insertion point dance:** after adding the terminator, `setInsertionPointToStart(loop.getBody())` moves *inside* the just-created loop, so the next iteration builds the inner loop nested inside it. When the C++ loop ends, the insertion point sits inside the innermost body.
- **Element print:** `memref.load %alloc[%i, %j]` reads the current element, and `llvm.call @printf(%frmt_spec_ptr, %elt)` prints it. `loopIvs` collected one induction variable per dimension, giving the full index vector.
- Finally `rewriter.eraseOp(op)` — `toy.print` has no results, so the pattern simply removes it after materializing the replacement code.

The emitted `scf.for` + `memref.load` ops are *illegal* for our conversion target — and that is fine: the SCF→CF, CF→LLVM, and MemRef→LLVM patterns registered alongside this pattern will lower them before the driver declares victory.

### 2.4 Conversion target — `LLVMConversionTarget`

For this conversion, aside from the top-level module, everything is lowered to the LLVM dialect. The pass and the start of `runOnOperation`:

***mlir/LowerToLLVM.cpp***
```cpp
namespace {
struct ToyToLLVMLoweringPass
    : public PassWrapper<ToyToLLVMLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToLLVMLoweringPass)
  StringRef getArgument() const override { return "toy-to-llvm"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void ToyToLLVMLoweringPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering. For this lowering, we are only targeting
  // the LLVM dialect.
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();
...
```

`getArgument()` gives the pass its command-line name `toy-to-llvm`, the name that shows up in `--mlir-print-ir-after-all` (run in section 3.3). `getDependentDialects` declares the dialects the pass may *create* during lowering, so the context loads them even if the input IR doesn't mention them.

`LLVMConversionTarget` is a `ConversionTarget` subclass whose header describes it as a "derived class that automatically populates legalization information for different LLVM ops" — it takes the place of upstream's hand-written `addLegalDialect` call. The only extra legal op is `ModuleOp` — the top-level container must survive. *Everything else* — `toy.*`, `affine.*`, `scf.*`, `cf.*`, `arith.*`, `memref.*`, `func.*` — is illegal and must be converted.

> **Upstream is outdated here (MLIR 20).** The tutorial writes (upstream snippet):
>
> ```c++
>   mlir::ConversionTarget target(getContext());
>   target.addLegalDialect<mlir::LLVMDialect>();
>   target.addLegalOp<mlir::ModuleOp>();
> ```
>
> Against the MLIR 20 headers this fails with "no member named 'LLVMDialect' in namespace 'mlir'": the dialect class is `mlir::LLVM::LLVMDialect`. Upstream's own `LowerToLLVM.cpp` uses `LLVMConversionTarget`, as this repo does.

### 2.5 Type converter — `LLVMTypeConverter` and the memref descriptor

This lowering also turns the MemRef types being operated on into an LLVM representation, which takes a `TypeConverter`:

***mlir/LowerToLLVM.cpp***
```cpp
...
  // During this lowering, we will also be lowering the MemRef types, that are
  // currently being operated on, to a representation in LLVM. To perform this
  // conversion we use a TypeConverter as part of the lowering. This converter
  // details how one type maps to another. This is necessary now that we will be
  // doing more complicated lowerings, involving loop region arguments.
  LLVMTypeConverter typeConverter(&getContext());
...
```

Until now, lowerings were "type-preserving enough" to get away without a converter. Not anymore: LLVM has no `memref`, no `index`, no `f64`-typed block arguments flowing through structured control flow. `LLVMTypeConverter` knows the standard mappings:

- `index` → `i64` (target-dependent integer),
- `f64` → `f64` (unchanged),
- **`memref<2x2xf64>` → a struct "descriptor"**:

```
!llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
                │    │    │        │                └─ strides   [2, 1]
                │    │    │        └─ sizes            [2, 2]
                │    │    └─ offset into the buffer    0
                │    └─ aligned pointer (used for loads/stores)
                └─ allocated pointer (what free() gets)
```

A ranked memref lowers to exactly this five-field struct: two pointers (the raw `malloc` result and the aligned data pointer), a linear offset, and per-dimension size and stride arrays. Every `memref.load %m[%i, %j]` becomes `extractvalue` (get the aligned pointer) + address arithmetic (`%i * stride0 + %j`) + `getelementptr` + `load` — visible in the output excerpt of section 2.7. The type converter is also what rewrites function signatures and block arguments that carry memref/index types, which matters because our lowering flows values through `cf` block arguments (loop-carried induction variables).

Since Toy's own tensor types were already eliminated in Chapter 5, the stock converter needs no customization — as upstream says, "the default converter is enough for our use case."

### 2.6 Conversion patterns — one `populate*` per lowering edge

At this point the IR is a mix of `toy`, `affine`, `arith`, `memref`, and `func` operations (plus the `scf` loops that `PrintOpLowering` creates), and MLIR already ships patterns for every edge except `toy.print`:

***mlir/LowerToLLVM.cpp***
```cpp
...
  // Now that the conversion target has been defined, we need to provide the
  // patterns used for lowering. At this point of the compilation process, we
  // have a combination of `toy`, `affine`, and `std` operations. Luckily, there
  // are already exists a set of patterns to transform `affine` and `std`
  // dialects. These patterns lowering in multiple stages, relying on transitive
  // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
  // patterns must be applied to fully transform an illegal operation into a
  // set of legal ones.
  RewritePatternSet patterns(&getContext());
  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  // The only remaining operation to lower from the `toy` dialect, is the
  // PrintOp.
  patterns.add<PrintOpLowering>(&getContext());
...
```

What each call contributes:

| `populate…` call | Converts | Into |
|---|---|---|
| `populateAffineToStdConversionPatterns` | `affine.for`, `affine.load`, `affine.store`, `affine.apply`, … | `scf.for` / `memref.load` / `memref.store` + `arith` index math |
| `populateSCFToControlFlowConversionPatterns` | `scf.for`, `scf.if`, `scf.while`, `scf.yield` | `cf.br` / `cf.cond_br` CFG with block arguments |
| `arith::populateArithToLLVMConversionPatterns` | `arith.constant`, `arith.addf`, `arith.mulf`, `arith.cmpi`, … | `llvm.mlir.constant`, `llvm.fadd`, `llvm.fmul`, `llvm.icmp`, … |
| `populateFinalizeMemRefToLLVMConversionPatterns` | `memref.alloc`, `memref.dealloc`, `memref.load`, `memref.store` | `llvm.call @malloc/@free`, descriptor `insertvalue`/`extractvalue`, `llvm.getelementptr`, `llvm.load`/`llvm.store` |
| `cf::populateControlFlowToLLVMConversionPatterns` | `cf.br`, `cf.cond_br` | `llvm.br`, `llvm.cond_br` |
| `populateFuncToLLVMConversionPatterns` | `func.func`, `func.call`, `func.return` | `llvm.func`, `llvm.call`, `llvm.return` (signatures rewritten via the type converter) |
| `patterns.add<PrintOpLowering>` | `toy.print` | `scf.for` nest + `llvm.call @printf` (then lowered further by the rows above) |

Note which pattern sets take the `typeConverter`: exactly the ones that cross the type boundary into LLVM (arith, memref, cf, func). The affine→scf and scf→cf stages stay within builtin types, so they don't need it.

This is the tutorial's "transitive lowering" payoff, which the source comment above spells out: nobody wrote an `affine.for` → `llvm.br` pattern. The driver applies affine→scf, then scf→cf, then cf→llvm patterns *recursively on the results of each other* within a single conversion.

> **Upstream is outdated here (MLIR 20).** Upstream's pattern list (and the "`std`" in its prose and in the source comment above) predates the split of the old Standard dialect: there is no `mlir/Dialect/StandardOps` in the MLIR 20 headers; its ops now live in `func`, `arith`, `memref`, and `cf`. Compiling upstream's calls against MLIR 20:
>
> | Upstream call | MLIR 20 |
> |---|---|
> | `mlir::populateAffineToStdConversionPatterns(patterns, &getContext())` | takes only `patterns` ("too many arguments") |
> | `mlir::cf::populateSCFToControlFlowConversionPatterns(patterns, &getContext())` | lives in namespace `mlir`, not `mlir::cf`, and takes only `patterns` |
> | `mlir::cf::populateControlFlowToLLVMConversionPatterns(patterns, &getContext())` | takes `(typeConverter, patterns)` |
> | (missing) | `populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns)` |
>
> The last row matters most. Upstream's list has no memref→LLVM patterns at all. Removing that line from a scratch copy of this repo makes the pass fail with `error: failed to legalize operation 'memref.alloc'` (exit code 4).

### 2.7 Full lowering — `applyFullConversion` and the working example

We want to lower completely to LLVM, so we use a full conversion: only legal operations may remain. The end of `runOnOperation`, followed by the factory function the driver calls:

***mlir/LowerToLLVM.cpp***
```cpp
...
  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::toy::createLowerToLLVMPass() {
  return std::make_unique<ToyToLLVMLoweringPass>();
}
```

Unlike Chapter 5's `applyPartialConversion`, `applyFullConversion` fails if *any* illegal op survives, and it names the op that could not be lowered (the `memref.alloc` error in section 2.6 is exactly such a diagnostic) instead of silently emitting broken IR.

> **Upstream is outdated here (MLIR 20).** Upstream writes `applyFullConversion(module, target, patterns)`. In MLIR 20 the third parameter is a `const FrozenRewritePatternSet &`, and a `FrozenRewritePatternSet` can only be built from a `RewritePatternSet &&`, so passing the lvalue gives "no matching function for call to 'applyFullConversion'". Hence `std::move(patterns)`.

#### The working example, lowered to the LLVM dialect

Upstream's working example is the Toy IR of its lit test `llvm-lowering.mlir`: a 2×3 constant, transposed, multiplied by itself element-wise, and printed. The `CHECK` lines at the bottom are for the `-emit=llvm -opt` output (section 3.1, run by FileCheck in section 4.8):

***test_Example/Toy/Ch6/llvm-lowering.mlir***
```mlir
// RUN: toyc-ch6 %s -emit=llvm -opt

toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.mul %2, %2 : tensor<3x2xf64>
  toy.print %3 : tensor<3x2xf64>
  toy.return
}

// CHECK-LABEL: define void @main()
// CHECK: @printf
// CHECK-SAME: 1.000000e+00
// CHECK: @printf
// CHECK-SAME: 1.600000e+01
// CHECK: @printf
// CHECK-SAME: 4.000000e+00
// CHECK: @printf
// CHECK-SAME: 2.500000e+01
// CHECK: @printf
// CHECK-SAME: 9.000000e+00
// CHECK: @printf
// CHECK-SAME: 3.600000e+01
```

The `.mlir` extension makes the driver parse it as MLIR, so it enters the pipeline at the Toy-dialect level. Lower it to the LLVM dialect:

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/llvm-lowering.mlir -emit=mlir-llvm 2>&1
```

Real output (abridged; the full dump is 226 lines) — the module-level declarations, the innermost print-loop body, and the cleanup block:

```mlir
module {
  llvm.func @free(!llvm.ptr)
  llvm.mlir.global internal constant @nl("\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @frmt_spec("%f \00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @main() {
...
  ^bb16:  // pred: ^bb15
    %162 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %163 = llvm.mlir.constant(2 : index) : i64
    %164 = llvm.mul %155, %163 : i64
    %165 = llvm.add %164, %160 : i64
    %166 = llvm.getelementptr %162[%165] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %167 = llvm.load %166 : !llvm.ptr -> f64
    %168 = llvm.call @printf(%148, %167) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    %169 = llvm.add %160, %159 : i64
    llvm.br ^bb15(%169 : i64)
...
  ^bb18:  // pred: ^bb13
    %172 = llvm.extractvalue %56[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @free(%172) : (!llvm.ptr) -> ()
    %173 = llvm.extractvalue %39[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @free(%173) : (!llvm.ptr) -> ()
    %174 = llvm.extractvalue %22[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @free(%174) : (!llvm.ptr) -> ()
    llvm.return
  }
}
```

The block names happen to match upstream's listing (`^bb16` prints one element, `^bb18` frees the three buffers), but the representation differs in ways that are all due to opaque pointers and the current descriptor:

- Types: upstream prints `!llvm<"i8*">` and `!llvm<"double*">`. That syntax no longer parses (next run); every pointer is now `!llvm.ptr`.
- The memref descriptor: upstream's `!llvm<"{ double*, i64, [2 x i64], [2 x i64] }">` has four fields, and it loads through field `[0]`. The current descriptor has five fields (section 2.5), so element accesses go through the *aligned* pointer `[1]`, while `free` receives the *allocated* pointer `[0]`.
- No `llvm.bitcast` before `free`: with opaque pointers there is nothing to cast.
- Loads and GEPs name their element type (`llvm.load %166 : !llvm.ptr -> f64`, `... -> !llvm.ptr, f64`), and calls to the variadic `printf` carry `vararg(!llvm.func<i32 (ptr, ...)>)` (section 2.2).

The full conversion left nothing but `llvm.*` ops inside the `builtin.module` — exactly what the `LLVMConversionTarget` of section 2.4 demands. A fully annotated walk through the smaller `jit.toy` dump — malloc, descriptor construction, the loop CFG, the newline call — is in section 4.5.

#### Upstream's pointer syntax no longer parses

Upstream's listing declares `free` as `llvm.func @free(!llvm<"i8*">)`. That line is not a repo file, so it goes to stock `mlir-opt` on stdin:

```bash
cd /Users/roy/study/mlir/toy/Ch6
/opt/homebrew/opt/llvm@20/bin/mlir-opt <<'EOF'
llvm.func @free(!llvm<"i8*">)
EOF
```

Real output (exit status 1):

```text
<stdin>:1:23: error: expected valid keyword
llvm.func @free(!llvm<"i8*">)
                      ^
```

The caret points at the quote: inside `!llvm<...>` the parser now expects an LLVM-dialect type keyword (`ptr`, `struct`, `array`, …), not a quoted LLVM IR type string. With `!llvm.ptr` in its place, as in the real output above, `mlir-opt` parses the declaration and exits 0.

### 2.8 The pass in the driver pipeline — plus a repo-specific pass

The factory is declared next to the Chapter 4–5 passes:

***include/toy/Passes.h***
```cpp
...
/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
...
```

The driver (`loadAndProcessMLIR` in `toyc.cpp`) appends the new stage when `-emit` is `mlir-llvm` or beyond (`isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM`):

***toyc.cpp***
```cpp
  if (isLoweringToLLVM) {
    // Finish lowering the toy IR to the LLVM dialect.
    pm.addPass(mlir::toy::createLowerToLLVMPass());

    // FIX: Segmentation fault when this pass is not added.
    // When lowering from Toy Dialect to the LLVM Dialect, MLIR often
    // creates "bridge" operations called unrealized_conversion_cast.
    // These casts are just placeholders. If they aren't removed before sent to JIT,
    // the JIT encounters an operation it doesn't recognize as valid LLVM IR and crashes (segfault).
    // This pass looks for pairs of these casts (e.g., Toy Type -> LLMV Type
    // followed by LLVM Type -> Toy Type) and deletes them, leaving "clean" LLVM
    // module that the JIT can actually execute.
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    // This is necessary to have line tables emitted and basic
    // debugger working. In the future we will add proper debug information
    // emission directly from our frontend.
    pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
  }
```

Two additions relative to a bare `createLowerToLLVMPass()`:

- **`createReconcileUnrealizedCastsPass()`** — added in this repo, not in upstream's `toyc.cpp` (hence the `// FIX:` comment and the matching `#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"` near the top of `toyc.cpp`). Dialect conversion can insert `builtin.unrealized_conversion_cast` "bridge" ops at type-boundary seams (e.g. `index` ↔ `i64`). They are placeholders that must cancel out in pairs; this pass erases them. If one survives, `translateModuleToLLVMIR`/the JIT meets a non-LLVM op in an "all-LLVM" module. With the current code none of the `test_Example/Toy/Ch6` inputs leaves a cast behind after `toy-to-llvm` (the run at the end of this section). A scratch build without the pass runs `jit.toy`, `codegen.toy`, and `llvm-lowering.mlir` with identical output, so for them the pass is only a safety net.
- **`createDIScopeForLLVMFuncOpPass()`** — from upstream; attaches debug-info scopes so the exported LLVM IR carries line tables (the `!dbg` metadata in the `-emit=llvm` output of section 3.1).

`main()` also registers one capability Ch5 didn't need, the LLVM dialect's inliner interface (the func-dialect extensions were already registered in Ch5):

***toyc.cpp***
```cpp
  // If we aren't dumping the AST, then we are compiling with/to MLIR.
  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  mlir::LLVM::registerInlinerInterface(registry);

  mlir::MLIRContext context(registry);
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::toy::ToyDialect>();
```

#### No `unrealized_conversion_cast` survives `toy-to-llvm`

`--mlir-print-ir-after-all` (section 3.3) dumps the module after every pass, so the dump after `toy-to-llvm` shows what `createReconcileUnrealizedCastsPass()` would have to clean up. Count the casts in all dumps, for each runnable test input:

```bash
cd /Users/roy/study/mlir/toy/Ch6
for f in jit.toy codegen.toy llvm-lowering.mlir; do
  ./build/toyc-ch6 ../../test_Example/Toy/Ch6/$f -emit=mlir-llvm --mlir-print-ir-after-all 2>&1 | grep -c unrealized_conversion_cast
done
```

Real output (exit status 1 — `grep -c` exits 1 when it counts no match):

```text
0
0
0
```

- Every count is 0, including the dump right after `toy-to-llvm`: for these inputs the full conversion leaves no bridge cast for the next pass to remove.
- The reconcile pass therefore has nothing to do here. A scratch build without it prints byte-identical `-emit=mlir-llvm`, `-emit=llvm`, and `-emit=jit` output for all three inputs; the pass stays as a safety net for inputs that do leave casts behind.

---

## 3. CodeGen: Getting Out of MLIR

At this point the module consists only of LLVM-dialect operations, so we are right at the cusp of code generation: export it to LLVM IR, and set up a JIT to run it. Both paths live in `toyc.cpp`; `main()` picks one from the `-emit` action once the pass pipeline has run:

***toyc.cpp***
```cpp
  // If we aren't exporting to non-mlir, then we are done.
  bool isOutputingMLIR = emitAction <= Action::DumpMLIRLLVM;
  if (isOutputingMLIR) {
    module->dump();
    return 0;
  }

  // Check to see if we are compiling to LLVM IR.
  if (emitAction == Action::DumpLLVMIR)
    return dumpLLVMIR(*module);

  // Otherwise, we must be running the jit.
  if (emitAction == Action::RunJIT)
    return runJit(*module);
```

### 3.1 Emitting LLVM IR — `dumpLLVMIR`

The export is one call, `mlir::translateModuleToLLVMIR`. The full function:

***toyc.cpp***
```cpp
int dumpLLVMIR(mlir::ModuleOp module) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Configure the LLVM Module
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Could not create JITTargetMachineBuilder\n";
    return -1;
  }

  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Could not create TargetMachine\n";
    return -1;
  }
  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                        tmOrError.get().get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::errs() << *llvmModule << "\n";
  return 0;
}
```

Piece by piece:

- **Translation registration.** MLIR-to-LLVM-IR export is pluggable via *dialect translation interfaces*. `registerLLVMDialectTranslation` teaches the exporter how each `llvm.*` op maps to an LLVM instruction; `registerBuiltinDialectTranslation` handles the builtin dialect (the module op). Without them, `translateModuleToLLVMIR` fails. A scratch build with these two lines removed prints ``error: cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface` registration for dialect for op: builtin.module``, then `Failed to emit LLVM IR`. This is a classic out-of-tree stumbling block, since `mlir-translate`-style tools register them for you.
- **`translateModuleToLLVMIR`** walks the MLIR module and builds a genuine `llvm::Module` in a fresh `llvm::LLVMContext` (LLVM is not thread-safe, so the tutorial uses a fresh context). This is a *translation* (1:1 export), not a pattern-based conversion — which is why the module had to be 100 % LLVM dialect first.
- `InitializeNativeTarget()` / `InitializeNativeTargetAsmPrinter()` link in the host backend (AArch64 here) so a `TargetMachine` can exist.
- `JITTargetMachineBuilder::detectHost()` + `setupTargetTripleAndDataLayout` stamp the module with the host triple and data layout (`target triple = "arm64-apple-darwin25.6.0"` in the output below).
- `makeOptimizingTransformer(optLevel, sizeLevel, targetMachine)` returns a function object wrapping LLVM's standard `-O<N>` pass pipeline: `-O0` by default, `-O3` with `-opt`.
- The module is printed to `llvm::errs()`, i.e. stderr — hence `2>&1` in the commands below.

> **Upstream is outdated here (MLIR 20).** Checked against the MLIR 20 headers:
>
> - Upstream's first snippet, `mlir::translateModuleToLLVMIR(module)`, has "too few arguments": the signature is `translateModuleToLLVMIR(Operation *module, llvm::LLVMContext &llvmContext, StringRef name = "LLVMDialectModule", bool disableVerification = false)`. Upstream's full `dumpLLVMIR` listing already passes a context.
> - Upstream's `mlir::ExecutionEngine::setupTargetTriple(llvmModule.get())` no longer exists. The header has only `setupTargetTripleAndDataLayout(llvm::Module *, llvm::TargetMachine *)`, which is why the code above first builds a `TargetMachine` from `JITTargetMachineBuilder::detectHost()`.
> - Upstream's listing omits the two `register...Translation` calls; with MLIR 20 they are required (the error above).

#### Exporting the working example

The input is `llvm-lowering.mlir` from section 2.7, and `-emit=llvm` runs `dumpLLVMIR` above without `-opt`, i.e. with the `-O0` transformer:

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/llvm-lowering.mlir -emit=llvm 2>&1
```

Real output (abridged; the full module is 184 lines) — the module header, the element-print block, the newline block, and the cleanup:

```llvm
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-darwin25.6.0"
...
define void @main() !dbg !8 {
...
87:                                               ; preds = %84
  %88 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1, !dbg !12
  %89 = mul i64 %81, 2, !dbg !12
  %90 = add i64 %89, %85, !dbg !12
  %91 = getelementptr double, ptr %88, i64 %90, !dbg !12
  %92 = load double, ptr %91, align 8, !dbg !12
  %93 = call i32 (ptr, ...) @printf(ptr @frmt_spec, double %92), !dbg !12
  %94 = add i64 %85, 1, !dbg !12
  br label %84, !dbg !12

95:                                               ; preds = %84
  %96 = call i32 (ptr, ...) @printf(ptr @nl), !dbg !12
  %97 = add i64 %81, 1, !dbg !12
  br label %80, !dbg !12

98:                                               ; preds = %80
  %99 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 0, !dbg !11
  call void @free(ptr %99), !dbg !11
  %100 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 0, !dbg !10
  call void @free(ptr %100), !dbg !10
  %101 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 0, !dbg !9
  call void @free(ptr %101), !dbg !9
  ret void, !dbg !13
}
```

- **Host stamping:** the `target datalayout` and `target triple` lines come from `setupTargetTripleAndDataLayout` with the host `TargetMachine` (upstream's excerpt starts at `define`).
- **Compared with upstream's listing:** pointers are `ptr` instead of `double*`/`i8*`, the `bitcast`s before `free` are gone, and the format string is passed as plain `ptr @frmt_spec` instead of a `getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0)` constant expression. Upstream's `memref.load double` and `cf.br label` are not LLVM IR at all — the real output has `load double` and `br label`.
- **Debug info:** the `!dbg` attachments come from the `DIScopeForLLVMFuncOp` pass (section 2.8).

#### The same module with `-opt`

With `-opt`, `makeOptimizingTransformer` gets `optLevel` 3:

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/llvm-lowering.mlir -emit=llvm -opt 2>&1
```

Real output (abridged to `main`; the full module is 43 lines):

```llvm
...
define void @main() local_unnamed_addr #0 !dbg !6 {
.preheader3:
  %0 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 1.000000e+00), !dbg !7
  %1 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 1.600000e+01), !dbg !7
  %putchar = tail call i32 @putchar(i32 10), !dbg !7
  %2 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 4.000000e+00), !dbg !7
  %3 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 2.500000e+01), !dbg !7
  %putchar.1 = tail call i32 @putchar(i32 10), !dbg !7
  %4 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 9.000000e+00), !dbg !7
  %5 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 3.600000e+01), !dbg !7
  %putchar.2 = tail call i32 @putchar(i32 10), !dbg !7
  ret void, !dbg !8
}
```

- The `-O3` pipeline trims `main` down to straight-line code, just as upstream shows: the loops are unrolled, the loads constant-folded, and `malloc`/`free` removed.
- LLVM's library-call simplification rewrote each `printf("\n")` into `putchar(10)`, so `@nl` is gone from the module.
- The six constants are the squares of the transposed matrix, row by row: 1, 16, 4, 25, 9, 36 — the values the `CHECK-SAME` lines of `llvm-lowering.mlir` expect (section 4.8).

### 3.2 Setting up a JIT — `runJit`

Running the module uses `mlir::ExecutionEngine`, a wrapper around LLVM's ORC JIT that takes the MLIR module directly:

***toyc.cpp***
```cpp
int runJit(mlir::ModuleOp module) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invokePacked("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}
```

- **`mlir::ExecutionEngine::create(module, options)`** re-runs the same translation as `dumpLLVMIR`, applies the `transformer` (our optimization pipeline) to the resulting `llvm::Module`, and eagerly JIT-compiles it to native code in the current process.
- **`invokePacked("main")`** looks up the JIT'd symbol `main` and calls it through the "packed-arguments" interface (an array of `void*` — empty here, since `main` takes no arguments and returns nothing). The JIT resolves `printf`/`malloc`/`free` against the host process, and the matrix prints directly to the terminal.
- The same **translation registration** and **native-target initialization** are required here as in `dumpLLVMIR` — the JIT path performs its own MLIR→LLVM-IR export.

> **Upstream is outdated here (MLIR 20).**
>
> - `mlir::ExecutionEngine::create(module, /*llvmModuleBuilder=*/nullptr, optPipeline)` no longer compiles: the signature is `create(Operation *op, const ExecutionEngineOptions &options = {}, std::unique_ptr<llvm::TargetMachine> tm = nullptr)`, so the pipeline goes into `ExecutionEngineOptions::transformer`.
> - `engine->invoke("main")` compiles but does not do what upstream intends. In MLIR 20, `invoke(funcName, args...)` calls `invokePacked("_mlir_ciface_" + funcName, ...)`, that is, the C-interface wrapper that only exists for functions marked `llvm.emit_c_interface`. A scratch build using `invoke("main")` prints `JIT invocation failed` and exits with 255. `invokePacked("main")` calls `main` itself.

#### Upstream's quick test: a program on stdin, run by the JIT

Upstream pipes a one-line program into the driver. With no input file, the positional argument defaults to `-`, which `getFileOrSTDIN` reads as stdin:

***toyc.cpp***
```cpp
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));
...
std::unique_ptr<toy::ModuleAST> parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
```

```bash
cd /Users/roy/study/mlir/toy/Ch6
echo 'def main() { print([[1, 2], [3, 4]]); }' | ./build/toyc-ch6 -emit=jit
```

Real output:

```text
1.000000 2.000000
3.000000 4.000000
```

- The whole pipeline ran in one process: parse, Toy passes, the affine and LLVM lowerings, translation, `-O0` codegen, and the call through `invokePacked("main")`. Nothing was written to disk.
- The output is the program's own `printf`s, so it goes to stdout (each line ends in a space, from the `"%f "` format string); the `-emit=mlir*`/`-emit=llvm` dumps go to stderr instead.
- The binary is `./build/toyc-ch6` run from the chapter directory, not upstream's `./bin/toyc-ch6` from the build directory. Section 4.7 runs the same program from `jit.toy`.

### 3.3 Comparing the IR levels

Upstream ends by suggesting `-emit=mlir`, `-emit=mlir-affine`, `-emit=mlir-llvm`, and `-emit=llvm` to compare the levels, and `--mlir-print-ir-after-all` to watch the IR evolve. Each `-emit` value is one entry of the driver's `Action` enum, and because the enum is ordered, every action runs all the pass stages before it:

***toyc.cpp***
```cpp
enum Action {
  None,
  DumpAST,
  DumpMLIR,
  DumpMLIRAffine,
  DumpMLIRLLVM,
  DumpLLVMIR,
  RunJIT
};
```

Section 4 runs every level on `jit.toy` (4.3–4.7), and upstream's `llvm-lowering.mlir` through the JIT and its FileCheck lines (4.8).

#### Watching the pipeline with `--mlir-print-ir-after-all`

`--mlir-print-ir-after-all` is one of the generic pass-manager options that `main()` registers and `loadAndProcessMLIR` applies to its `PassManager`, which it then fills stage by stage (abridged; the `isLoweringToLLVM` block is in section 2.8, and `main()` comes last in the file):

***toyc.cpp***
```cpp
  mlir::PassManager pm(module.get()->getName());
  // Apply any generic pass manager command line options and run the pipeline.
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;

  // Check to see what granularity of MLIR we are compiling to.
  bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
  bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;

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
...
int main(int argc, char **argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
...
```

The option prints the whole module after every pass; `grep` keeps just the headers:

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/jit.toy -emit=mlir-llvm --mlir-print-ir-after-all 2>&1 | grep 'IR Dump'
```

Real output:

```text
// -----// IR Dump After Canonicalizer (canonicalize) //----- //
// -----// IR Dump After Inliner (inline) //----- //
// -----// IR Dump After (anonymous namespace)::ShapeInferencePass (toy-shape-inference) //----- //
// -----// IR Dump After Canonicalizer (canonicalize) //----- //
// -----// IR Dump After CSE (cse) //----- //
// -----// IR Dump After (anonymous namespace)::ToyToAffineLoweringPass (toy-to-affine) //----- //
// -----// IR Dump After Canonicalizer (canonicalize) //----- //
// -----// IR Dump After CSE (cse) //----- //
// -----// IR Dump After (anonymous namespace)::ToyToLLVMLoweringPass (toy-to-llvm) //----- //
// -----// IR Dump After ReconcileUnrealizedCasts (reconcile-unrealized-casts) //----- //
// -----// IR Dump After DIScopeForLLVMFuncOpPass (ensure-debug-info-scope-on-llvm-func) //----- //
```

- The first `canonicalize` is the one the inliner runs on its own; the rest is exactly the pipeline above, in order.
- The last three entries are this chapter's `isLoweringToLLVM` stages (section 2.8). `toy-to-llvm` is the name `ToyToLLVMLoweringPass::getArgument()` returns (section 2.4).
- Without the `grep`, the dump after `toy-to-llvm` is the place to look for leftover `unrealized_conversion_cast` ops; section 2.8 counts them (none).

In the next chapter we will leave primitive data types behind and add a composite `struct` type.

---

## 4. Build and Run

The superbuild (`toy/build.sh`, `toy/run.sh`, `CMakePresets.json`) is documented once in the top-level [README](../README.md#the-build-system). This section builds and runs Chapter 6 on its own, from the chapter directory. What the chapter's `CMakeLists.txt` adds — the ExecutionEngine guard and the shared-only link line — is in the [appendix](#appendix-what-chapter-6-adds-to-the-build).

### 4.1 Building

```bash
cd /Users/roy/study/mlir/toy/Ch6
cmake -S . -B build -G Ninja
cmake --build build          # → ./build/toyc-ch6
```

No preset applies at the chapter level, yet no toolchain flags are needed, because the shell environment already points at Homebrew LLVM 20:

- `CXX=/opt/homebrew/opt/llvm@20/bin/clang++` (and `CC`) selects the compiler.
- `/opt/homebrew/opt/llvm@20/bin` is on `PATH`, and `find_package` also searches the prefix above each `PATH` entry, so it finds `/opt/homebrew/opt/llvm@20/lib/cmake/{mlir,llvm}` by itself.

In a shell without that setup, pass them explicitly: `-DMLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++`.

The chapter targets are only defined when the installed MLIR was built with the ExecutionEngine (`MLIR_ENABLE_EXECUTION_ENGINE`, set to `1` in Homebrew's `MLIRConfig.cmake`); otherwise configuring succeeds but there is nothing to build. A standalone build puts the binary directly in `build/`; the superbuild's `toy/build/bin/toyc-ch6` behaves identically. To confirm the binary links only the shared MLIR/LLVM libraries, see appendix A.3.

### 4.2 Running

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/jit.toy -emit=jit
```

- `../../test_Example/Toy/Ch6/jit.toy` — the input program (below). Omitting it (or passing `-`) reads from stdin, which is how `run.sh` feeds its inline program.
- `-emit=<action>` — how far down the pipeline to go. This chapter adds three actions to Chapter 5's `ast`, `mlir`, and `mlir-affine`: `mlir-llvm` (dump after the LLVM-dialect lowering), `llvm` (translate and print LLVM IR), and `jit` (compile and run `main`). The actions are ordered, so each one runs every pass stage before it (section 3.3).
- `-opt` — enable the Toy-level optimizations (inlining, shape inference, canonicalization, CSE — which `mlir-affine` and beyond run anyway), the affine loop fusion/scalar replacement cleanups, and `-O3` in the LLVM pipeline used by `-emit=llvm` and `-emit=jit` (section 3).
- `-x mlir` (or a `.mlir` extension) — parse the input as MLIR instead of Toy source.
- `2>&1` — `-emit=mlir*` dumps via `module->dump()` and `-emit=llvm` prints via `llvm::errs()`, so both go to **stderr**. Only `-emit=jit` output goes to stdout, because it is the program's own `printf`.

**The input.** A 2×2 constant, printed — the same program that `run.sh ch6` and section 3.2 pipe in on stdin.

***test_Example/Toy/Ch6/jit.toy***
```text
# RUN: toyc-ch6 -emit=jit %s
# UNSUPPORTED: target={{.*windows.*}}

def main() {
 print([[1, 2], [3, 4]]);
}
```

### 4.3 `-emit=mlir` — pure Toy dialect

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/jit.toy -emit=mlir 2>&1
```

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>
    toy.print %0 : tensor<2x2xf64>
    toy.return
  }
}
```

Straight from `mlirGen`: values are still abstract `tensor`s, printing is one opaque op.

### 4.4 `-emit=mlir-affine` — after Chapter 5's partial lowering

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/jit.toy -emit=mlir-affine 2>&1
```

```mlir
module {
  func.func @main() {
    %cst = arith.constant 4.000000e+00 : f64
    %cst_0 = arith.constant 3.000000e+00 : f64
    %cst_1 = arith.constant 2.000000e+00 : f64
    %cst_2 = arith.constant 1.000000e+00 : f64
    %alloc = memref.alloc() : memref<2x2xf64>
    affine.store %cst_2, %alloc[0, 0] : memref<2x2xf64>
    affine.store %cst_1, %alloc[0, 1] : memref<2x2xf64>
    affine.store %cst_0, %alloc[1, 0] : memref<2x2xf64>
    affine.store %cst, %alloc[1, 1] : memref<2x2xf64>
    toy.print %alloc : memref<2x2xf64>
    memref.dealloc %alloc : memref<2x2xf64>
    return
  }
}
```

Tensors became a heap `memref<2x2xf64>` with element-wise `affine.store`s (the constant is small, so no loops were needed). Crucially, **`toy.print` survived** — but its operand type changed to `memref`, which is exactly what `PrintOpLowering` expects (section 2.1).

### 4.5 `-emit=mlir-llvm` — after `ToyToLLVMLoweringPass` (still MLIR!)

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/jit.toy -emit=mlir-llvm 2>&1
```

Abridged; the full dump is 101 lines (the elided parts are more index constants and the four element stores, each an `extractvalue` + `mul`/`add` + `getelementptr` + `llvm.store`):

```mlir
module {
  llvm.func @free(!llvm.ptr)
  llvm.mlir.global internal constant @nl("\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @frmt_spec("%f \00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @main() {
...
    %4 = llvm.mlir.constant(2 : index) : i64
    %5 = llvm.mlir.constant(2 : index) : i64
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.mlir.constant(4 : index) : i64
    %8 = llvm.mlir.zero : !llvm.ptr
    %9 = llvm.getelementptr %8[%7] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %10 = llvm.ptrtoint %9 : !llvm.ptr to i64
    %11 = llvm.call @malloc(%10) : (i64) -> !llvm.ptr
    %12 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.insertvalue %11, %12[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.insertvalue %11, %13[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.mlir.constant(0 : index) : i64
    %16 = llvm.insertvalue %15, %14[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.insertvalue %4, %16[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %18 = llvm.insertvalue %5, %17[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %19 = llvm.insertvalue %5, %18[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.insertvalue %6, %19[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
...
    %49 = llvm.mlir.addressof @frmt_spec : !llvm.ptr
    %50 = llvm.mlir.constant(0 : index) : i64
    %51 = llvm.getelementptr %49[%50, %50] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<4 x i8>
    %52 = llvm.mlir.addressof @nl : !llvm.ptr
    %53 = llvm.mlir.constant(0 : index) : i64
    %54 = llvm.getelementptr %52[%53, %53] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<2 x i8>
    %55 = llvm.mlir.constant(0 : index) : i64
    %56 = llvm.mlir.constant(2 : index) : i64
    %57 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%55 : i64)
  ^bb1(%58: i64):  // 2 preds: ^bb0, ^bb5
    %59 = llvm.icmp "slt" %58, %56 : i64
    llvm.cond_br %59, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %60 = llvm.mlir.constant(0 : index) : i64
    %61 = llvm.mlir.constant(2 : index) : i64
    %62 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb3(%60 : i64)
  ^bb3(%63: i64):  // 2 preds: ^bb2, ^bb4
    %64 = llvm.icmp "slt" %63, %61 : i64
    llvm.cond_br %64, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %65 = llvm.extractvalue %20[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %66 = llvm.mlir.constant(2 : index) : i64
    %67 = llvm.mul %58, %66 : i64
    %68 = llvm.add %67, %63 : i64
    %69 = llvm.getelementptr %65[%68] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %70 = llvm.load %69 : !llvm.ptr -> f64
    %71 = llvm.call @printf(%51, %70) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    %72 = llvm.add %63, %62 : i64
    llvm.br ^bb3(%72 : i64)
  ^bb5:  // pred: ^bb3
    %73 = llvm.call @printf(%54) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    %74 = llvm.add %58, %57 : i64
    llvm.br ^bb1(%74 : i64)
  ^bb6:  // pred: ^bb1
    %75 = llvm.extractvalue %20[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @free(%75) : (!llvm.ptr) -> ()
    llvm.return
  }
}
```

Everything from section 2 is visible here:

- the two **`llvm.mlir.global`** format strings (`@frmt_spec("%f \00")`, `@nl("\0A\00")`) and the **`llvm.func @printf(!llvm.ptr, ...)`** declaration hoisted to module scope (section 2.2);
- `memref.alloc` became **`llvm.call @malloc`** plus `llvm.insertvalue`s building the five-field **memref descriptor struct** (section 2.5): allocated pointer `[0]`, aligned pointer `[1]`, offset `[2]` = 0, sizes `[3, 0]`/`[3, 1]` = 2, 2, and strides `[4, 0]`/`[4, 1]` = 2, 1;
- the `scf.for` nest became a **CFG of blocks with block arguments** (`^bb1(%58: i64)` — MLIR's SSA replacement for PHI nodes): `^bb1` is the outer (row) loop header, `^bb3` the inner (column) loop header, `^bb4` the body;
- element access in `^bb4` = `extractvalue [1]` (aligned ptr) + `mul/add` (row-major index `i*2+j`) + `getelementptr` + `load`, then `llvm.call @printf(%51, %70)` with the format-string pointer;
- the newline `printf` sits in `^bb5`, i.e. after each inner-loop run — one newline per row (section 2.3);
- `memref.dealloc` became **`llvm.call @free`** in `^bb6` on `extractvalue [0]` (the *allocated* pointer, not the aligned one).

### 4.6 `-emit=llvm` — genuine LLVM IR

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/jit.toy -emit=llvm 2>&1
```

Abridged real output (unoptimized, `-opt` not passed):

```llvm
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-darwin25.6.0"

@nl = internal constant [2 x i8] c"\0A\00"
@frmt_spec = internal constant [4 x i8] c"%f \00"

declare !dbg !3 void @free(ptr)

declare !dbg !6 i32 @printf(ptr, ...)

declare !dbg !7 ptr @malloc(i64)

define void @main() !dbg !8 {
  %1 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 4) to i64)), !dbg !10
  %2 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1, 0, !dbg !10
...
  %9 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1, !dbg !10
  %10 = getelementptr double, ptr %9, i64 0, !dbg !10
  store double 1.000000e+00, ptr %10, align 8, !dbg !10
...
  %15 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1, !dbg !10
  %16 = getelementptr double, ptr %15, i64 3, !dbg !10
  store double 4.000000e+00, ptr %16, align 8, !dbg !10
  br label %17, !dbg !11

17:                                               ; preds = %32, %0
  %18 = phi i64 [ 0, %0 ], [ %34, %32 ], !dbg !11
  %19 = icmp slt i64 %18, 2, !dbg !11
  br i1 %19, label %20, label %35, !dbg !11

20:                                               ; preds = %17
  br label %21, !dbg !11

21:                                               ; preds = %24, %20
  %22 = phi i64 [ 0, %20 ], [ %31, %24 ], !dbg !11
  %23 = icmp slt i64 %22, 2, !dbg !11
  br i1 %23, label %24, label %32, !dbg !11

24:                                               ; preds = %21
  %25 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1, !dbg !11
  %26 = mul i64 %18, 2, !dbg !11
  %27 = add i64 %26, %22, !dbg !11
  %28 = getelementptr double, ptr %25, i64 %27, !dbg !11
  %29 = load double, ptr %28, align 8, !dbg !11
  %30 = call i32 (ptr, ...) @printf(ptr @frmt_spec, double %29), !dbg !11
  %31 = add i64 %22, 1, !dbg !11
  br label %21, !dbg !11

32:                                               ; preds = %21
  %33 = call i32 (ptr, ...) @printf(ptr @nl), !dbg !11
  %34 = add i64 %18, 1, !dbg !11
  br label %17, !dbg !11

35:                                               ; preds = %17
  %36 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 0, !dbg !10
  call void @free(ptr %36), !dbg !10
  ret void, !dbg !12
}
...
!8 = distinct !DISubprogram(name: "main", linkageName: "main", scope: !9, file: !9, line: 4, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!9 = !DIFile(filename: "jit.toy", directory: "../../test_Example/Toy/Ch6")
!10 = !DILocation(line: 5, column: 8, scope: !8)
!11 = !DILocation(line: 5, column: 2, scope: !8)
!12 = !DILocation(line: 4, column: 1, scope: !8)
```

What changed vs. `-emit=mlir-llvm`: same structure, different *representation*. The host **target triple and data layout** are stamped in (section 3.1); block arguments became **`phi` nodes** (`%18 = phi i64 [ 0, %0 ], [ %34, %32 ]`); `malloc`'s size argument became the classic constant-folded `ptrtoint(gep(null, 4))` sizeof idiom; and the `DIScopeForLLVMFuncOp` pass (section 2.8) shows up as `!dbg` line-table metadata — whose `!DIFile` records the input path exactly as given on the command line.

Passing `-opt` runs the `-O3` transformer:

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/jit.toy -emit=llvm -opt 2>&1
```

```llvm
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-darwin25.6.0"

@frmt_spec = internal constant [4 x i8] c"%f \00"

; Function Attrs: nofree nounwind
declare !dbg !3 noundef i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #0

; Function Attrs: nofree nounwind
define void @main() local_unnamed_addr #0 !dbg !6 {
.preheader:
  %0 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 1.000000e+00), !dbg !8
  %1 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 2.000000e+00), !dbg !8
  %putchar = tail call i32 @putchar(i32 10), !dbg !8
  %2 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 3.000000e+00), !dbg !8
  %3 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 4.000000e+00), !dbg !8
  %putchar.1 = tail call i32 @putchar(i32 10), !dbg !8
  ret void, !dbg !9
}

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #0

attributes #0 = { nofree nounwind }
...
```

As in the official docs, the loops are fully unrolled and the loads constant-folded, so `main` collapses into straight-line code: four `printf` calls with immediate `double` constants. `malloc`/`free` are gone too, and LLVM's library-call simplification has rewritten each `printf("\n")` into `putchar(10)` (so the unused `@nl` global disappears).

### 4.7 `-emit=jit` — actually running it

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/jit.toy -emit=jit
```

```text
1.000000 2.000000
3.000000 4.000000
```

The ExecutionEngine JIT-compiled `main` to AArch64 code, `invokePacked("main")` called it, the JIT resolved `printf`/`malloc`/`free` against the host libc — and the Toy program printed its matrix (section 3.2). End-to-end: source text to executed native code in one process, no files written. `-emit=jit -opt` prints the same thing. (Each line really ends in a space, from the `"%f "` format string.) Section 3.2 runs upstream's variant, the same program on stdin.

### 4.8 Upstream's working example: `llvm-lowering.mlir`

Section 2.7 shows the file and its `-emit=mlir-llvm` lowering, and section 3.1 its `-emit=llvm` output with and without `-opt`. What remains is to run it, and to run its lit test.

**Running it.** The JIT prints the 3×2 result:

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/llvm-lowering.mlir -emit=jit
```

Real output:

```text
1.000000 16.000000
4.000000 25.000000
9.000000 36.000000
```

**The lit test, by hand.** The file's `RUN` line is `toyc-ch6 %s -emit=llvm -opt`, and its `CHECK` lines expect the six squares in order. Pipe the optimized IR into `FileCheck`:

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/llvm-lowering.mlir -emit=llvm -opt 2>&1 \
  | FileCheck ../../test_Example/Toy/Ch6/llvm-lowering.mlir
```

Real output: none (exit status 0). FileCheck is silent when every `CHECK` line matches.

The last `CHECK-SAME` expects `3.600000e+01` (6 × 6). The `release/20.x` copy of this test upstream still says `3.000000e+01`, which does not match the real output.

### 4.9 More inputs to try

`test_Example/Toy/Ch6/` has additional cases. `codegen.toy` calls `multiply_transpose` twice; the driver inlines it and runs `print(d)`, i.e. `transpose(b) * transpose(a)` for the 2×3 inputs:

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/codegen.toy -emit=jit
```

Real output — the same 3×2 values as section 4.8, computed from Toy source instead of hand-written IR:

```text
1.000000 16.000000
4.000000 25.000000
9.000000 36.000000
```

`invalid.mlir` is malformed Toy IR, which the verifier rejects while the file is loaded:

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/invalid.mlir -emit=mlir
```

Real output (exit status 3):

```text
loc("../../test_Example/Toy/Ch6/invalid.mlir":8:8): error: 'toy.print' op requires zero results
Error can't load file ../../test_Example/Toy/Ch6/invalid.mlir
```

### 4.10 The ecosystem view: replace the backend with stock tools

Once the module is pure LLVM dialect (`-emit=mlir-llvm`), Toy is out of the picture — so from that point on, the *stock* Homebrew tools can finish the job. This pipeline runs the Toy program without `toyc-ch6`'s `dumpLLVMIR`/`runJit` at all:

```bash
cd /Users/roy/study/mlir/toy/Ch6
export P=/opt/homebrew/opt/llvm@20/bin
./build/toyc-ch6 ../../test_Example/Toy/Ch6/jit.toy -emit=mlir-llvm 2>&1 \
  | $P/mlir-translate --mlir-to-llvmir \
  | $P/lli
```

Real output — the same matrix as section 4.7:

```text
1.000000 2.000000
3.000000 4.000000
```

Each stage is one of the chapter's concepts as a standalone tool:

- **`toyc-ch6 -emit=mlir-llvm`** — the frontend + all dialect *conversions* (section 2). This is the last stage that needs the Toy dialect linked in.
- **`mlir-translate --mlir-to-llvmir`** — the *translation* of section 3.1 (`translateModuleToLLVMIR`) as a stock binary; it works because the LLVM dialect is upstream, so no custom registration is needed. Conversion vs. translation being different mechanisms (takeaway 4) is literally visible as two different tools.
- **`lli`** — LLVM's own JIT, standing in for section 3.2's `runJit`/ExecutionEngine (both sit on ORC underneath); it resolves `printf`/`malloc`/`free` against the host libc the same way. Swap `lli` for `$P/llc -filetype=obj -o toy.o` and you have an object file for a real linker instead — the AOT exit ramp from the same IR.

`toyc-ch6` bundles these stages into one process for convenience (and to avoid textual round-trips); the pipe shows they remain independently replaceable pieces of the LLVM/MLIR ecosystem.

---

## 5. Key Takeaways & Pitfalls

### ⚠️ Pitfall #1 (the big one): static + shared MLIR libs ⇒ TypeID-duplication segfault

Documented in full in **[MLIR_LINKING_PITFALL.md](MLIR_LINKING_PITFALL.md)** — read it before writing your own out-of-tree JIT-using project. Summary:

- **Root cause.** MLIR's TypeID system identifies types/interfaces by *the address of a static variable* in a template instantiation. The Homebrew CMake target `MLIRExecutionEngine` (static) has `INTERFACE_LINK_LIBRARIES "LLVM;MLIR"`, where `MLIR` = `libMLIR.dylib`. If your `target_link_libraries` also lists static `.a` MLIR libraries (`${dialect_libs}`, `${conversion_libs}`, `MLIRIR`, `MLIRPass`, …), the linker pulls in **both** the static archives and the dylib. Two copies of every TypeID static now live in the process, at different addresses ⇒ the "same" type gets two different TypeIDs ⇒ `StorageUniquer` can't find registered types/attributes and dereferences an invalid pointer.
- **Symptoms.** `EXC_BAD_ACCESS` crash inside `mlir::detail::StorageUniquerImpl::getOrCreate`, triggered by **any pass run** (`PassManager::run`) — not just JIT. Deeply misleading, because the crash is in generic MLIR infrastructure, nowhere near your code.
- **Detection.** `otool -L ./build/toyc-ch6` (macOS) / `ldd` (Linux), as in appendix A.3. If you see `libMLIR.dylib` **and** your ninja log shows static `.a` MLIR libraries on the link line, you have the bug.
- **Fix.** Link shared libraries *only*: `MLIR` + `MLIRExecutionEngineShared` (appendix A.2). Alternative: an all-static toolchain built with `LLVM_BUILD_LLVM_DYLIB=OFF` — then there is no dylib to conflict with. What you must never do is mix.
- **Why Ch1–Ch5 didn't crash.** None of them mixes: Ch1 links only the static `MLIRSupport`, and Ch2–Ch5 link only the monolithic dylibs `MLIR` + `LLVM`. None links `MLIRExecutionEngine`, the target that silently drags `libMLIR.dylib` in next to static archives. The trap springs the moment the ExecutionEngine appears — i.e., exactly in Chapter 6 — if you copy upstream's static link line.

### ⚠️ Pitfalls #2–#4: smaller traps

**Pitfall #2: leftover `unrealized_conversion_cast` ops.** Dialect conversion can insert `builtin.unrealized_conversion_cast` bridge ops at type seams. If they don't all cancel out and nothing removes them, the "fully lowered" module still contains a non-LLVM op, and the translation/JIT fails on it. This repo adds `createReconcileUnrealizedCastsPass()` right after `createLowerToLLVMPass()` (section 2.8); its `// FIX:` comment records a segfault, but with the current code no test input produces a cast, and a build without the pass runs them all correctly.

**Pitfall #3: forgetting translation registration or target init.** Without `registerLLVMDialectTranslation`/`registerBuiltinDialectTranslation`, `translateModuleToLLVMIR` fails with ``missing `LLVMTranslationDialectInterface` registration`` (section 3.1); the calls are needed in **both** `dumpLLVMIR` and `runJit`. Without `InitializeNativeTarget()`/`InitializeNativeTargetAsmPrinter()` there is no TargetMachine, and so no JIT.

**Pitfall #4: copying upstream's snippets verbatim.** The tutorial text is older than its own `release/20.x` source. Its conversion-pattern calls, `applyFullConversion(..., patterns)`, `translateModuleToLLVMIR(module)`, `setupTargetTriple`, `ExecutionEngine::create(module, nullptr, optPipeline)`, `LLVMPointerType::get(i8Type)`, and `SymbolRefAttr::get("printf", context)` all fail to compile against MLIR 20 (sections 2.2–2.7, 3.1–3.2). Two traps even compile: leaving out the memref→LLVM patterns fails only at run time (`failed to legalize operation 'memref.alloc'`), and `engine->invoke("main")` looks up `_mlir_ciface_main` and reports `JIT invocation failed`. Start from the chapter's source files, not from the prose.

### Key takeaways

1. **Transitive lowering** keeps patterns simple: `PrintOpLowering` emits `scf` + `memref` ops and trusts the other patterns in the same `applyFullConversion` to finish the job. Nobody writes affine→LLVM patterns.
2. **Full vs. partial conversion** is the legality contract: `applyFullConversion` guarantees a pure-LLVM-dialect module (modulo `ModuleOp`), which is precisely what the exporter requires.
3. **`LLVMTypeConverter`** is where types cross the boundary — most visibly `memref<2x2xf64>` → the five-field descriptor struct `{ptr, ptr, i64, [2 x i64], [2 x i64]}` (allocated ptr, aligned ptr, offset, sizes, strides).
4. **Conversion ≠ translation.** Dialect conversion rewrites MLIR into the LLVM *dialect*; `translateModuleToLLVMIR` then exports 1:1 into an `llvm::Module`. Two different mechanisms, two different failure modes.
5. **`mlir::ExecutionEngine`** makes "compile and run in-process" a ~30-line function: translation + `makeOptimizingTransformer` + ORC JIT + `invokePacked("main")`.
6. On macOS with Homebrew LLVM, **link shared MLIR libraries consistently** (`MLIR` + `MLIRExecutionEngineShared`) and verify with `otool -L`.

---

## Appendix: What Chapter 6 adds to the build

This chapter is where out-of-tree builds get genuinely tricky, because `ExecutionEngine` drags LLVM's JIT and native codegen into the link. Relative to Chapter 5, `CMakeLists.txt` adds three things: a feature guard, one source file, and a *changed link line* — the last one is the part that matters. The TableGen wiring (`include/toy/CMakeLists.txt` for `Ops.td` and `ShapeInferenceInterface.td`, plus the `ToyCombine.td` rewriters) is the same as in Chapter 5, with the targets renamed `ToyCh6*`.

### A.1 The chapter targets

Below the standalone guard:

***CMakeLists.txt***
```cmake
# This chapter depends on JIT support enabled.
if(NOT MLIR_ENABLE_EXECUTION_ENGINE)
  return()
endif()

include_directories(include)
add_subdirectory(include)

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCh6CombineIncGen)

add_executable(toyc-ch6
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/LowerToAffineLoops.cpp
  mlir/LowerToLLVM.cpp
  mlir/ShapeInferencePass.cpp
  mlir/ToyCombine.cpp
  )

add_dependencies(toyc-ch6 ToyCh6ShapeInferenceInterfaceIncGen)
add_dependencies(toyc-ch6 ToyCh6OpsIncGen)
add_dependencies(toyc-ch6 ToyCh6CombineIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
```

The `MLIR_ENABLE_EXECUTION_ENGINE` guard requires the installed MLIR to have been built with the ExecutionEngine (Homebrew's is). The only new source is `mlir/LowerToLLVM.cpp`.

### A.2 The link line — the part that matters

***CMakeLists.txt***
```cmake
# NOTE: link ONLY shared MLIR/LLVM libraries here. Mixing static .a archives
# with libMLIR.dylib causes TypeID duplication and runtime segfaults — see
# MLIR_LINKING_PITFALL.md in this directory.
target_link_libraries(toyc-ch6
  PRIVATE
    MLIR                         # libMLIR.dylib (all dialects, passes, conversions)
    MLIRExecutionEngineShared    # libMLIRExecutionEngineShared.dylib (JIT support)
    )
```

That's it. **Two shared libraries, zero static archives.** Compared with Chapter 5 (`MLIR` + `LLVM`), `LLVM` is replaced by `MLIRExecutionEngineShared`. This is deliberate (the `NOTE` comment guards it in the file itself), and it is how this repo avoids the segfault documented in [MLIR_LINKING_PITFALL.md](MLIR_LINKING_PITFALL.md):

- The upstream Toy Ch6 CMakeLists links dozens of *static* targets: `${dialect_libs}`, `${conversion_libs}`, `${extension_libs}`, `MLIRExecutionEngine`, `MLIRAnalysis`, `MLIRIR`, `MLIRPass`, … That works inside llvm-project's own build tree.
- But against **Homebrew LLVM**, the static `MLIRExecutionEngine` target carries `INTERFACE_LINK_LIBRARIES "LLVM;MLIR"` — and `MLIR` there means **`libMLIR.dylib`**. The linker then pulls in *both* the static `.a` copies of MLIR *and* the dylib → two copies of every TypeID static → runtime segfault (details in section 5).
- The fix: link **shared libraries consistently**. `libMLIR.dylib` already contains *all* dialects, passes and conversions (that's why no `${dialect_libs}` are needed), and `libMLIRExecutionEngineShared.dylib` supplies the JIT. `libMLIR.dylib` itself depends on `libLLVM.dylib`, which bundles every LLVM component (OrcJIT, native codegen), so no `LLVM_LINK_COMPONENTS`/`llvm_map_components_to_libnames` boilerplate is needed either.

### A.3 Verifying the link

```bash
cd /Users/roy/study/mlir/toy/Ch6
otool -L ./build/toyc-ch6
```

```text
./build/toyc-ch6:
	/opt/homebrew/opt/llvm@20/lib/libMLIRExecutionEngineShared.dylib (compatibility version 0.0.0, current version 0.0.0)
	/opt/homebrew/opt/llvm@20/lib/libMLIR.dylib (compatibility version 0.0.0, current version 0.0.0)
	/opt/homebrew/opt/llvm@20/lib/libLLVM.dylib (compatibility version 1.0.0, current version 20.1.8)
	/usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 2100.43.0)
	/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1356.0.0)
```

Exactly one copy of MLIR and one of LLVM in the process — consistent TypeIDs. (Building with Homebrew's own clang++ — selected by `CXX` in section 4.1, and pinned in the superbuild preset — also matters here: MLIR headers must be compiled with a compiler/stdlib ABI-compatible with these prebuilt dylibs.)

---

## Links

- Official tutorial: [Toy Ch-6 — Lowering to LLVM and CodeGeneration](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/)
- Related MLIR docs: [LLVM IR Target](https://mlir.llvm.org/docs/TargetLLVMIR/) · [LLVM dialect](https://mlir.llvm.org/docs/Dialects/LLVM/) · [Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)
- Linking post-mortem: [MLIR_LINKING_PITFALL.md](MLIR_LINKING_PITFALL.md)
- Previous: [Chapter 5 — Partial Lowering to Affine](../Ch5/README.md)
- Next: [Chapter 7 — Struct Types](../Ch7/README.md)
- Back to [README](../README.md)
