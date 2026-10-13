# Chapter 6: Lowering to LLVM and JIT Compilation

> **Goal:** Complete the lowering journey — take the mixed `toy`/`affine`/`arith`/`memref` IR from Chapter 5 all the way down to the LLVM dialect, translate it to real LLVM IR, and execute it with a JIT — following the official tutorial [Toy Ch-6: Lowering to LLVM and CodeGeneration](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/).

> **Read this first if your Ch6 binary segfaults:** this repo hit (and documents) a real macOS
> linking pitfall where mixing static `.a` MLIR libraries with `libMLIR.dylib` causes a
> **TypeID-duplication segfault** inside `StorageUniquer` on *any* pass run. See
> section 6, the link line in appendix C.2, and the full write-up in
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

Three ideas are worth internalizing before reading the code:

1. **Transitive (A→B→C) lowering.** We never write patterns that go straight from `affine` to `llvm`. Instead, `affine` lowers to `scf`, `scf` lowers to `cf`, and `cf` lowers to `llvm` — each stage reusing patterns that MLIR already provides. The dialect-conversion driver applies all of these pattern sets in one `applyFullConversion` call, chaining them automatically until everything is legal.
2. **Full conversion vs. partial conversion.** Chapter 5 used `applyPartialConversion` (unknown ops may survive). Here we use `applyFullConversion`: after the pass, *only* LLVM-dialect operations (plus the top-level `builtin.module`) may remain. Anything else is a hard error.
3. **The LLVM dialect is still MLIR.** `-emit=mlir-llvm` prints MLIR operations (`llvm.func`, `llvm.call`, `llvm.br`, …) that model LLVM IR 1:1. A separate *translation* step (`translateModuleToLLVMIR`) — not a dialect conversion — produces an actual `llvm::Module`. From that point on we are in classic LLVM land: optimization pipelines, target machines, ORC JIT.

After Build and Run, section 5 shows how to derive the chapter's code yourself: `LowerToLLVM.cpp` from the dialects left to lower to the builder calls, then the driver's new stages and the link line.

The appendices hold the extras: more runs — the `llvm-lowering.mlir` example through the JIT and FileCheck, more inputs (A), replacing the backend with stock tools (B), what Chapter 6 adds to the build (C), and the script that runs every command in this README (D).

---

## 2. Lowering to LLVM

For this lowering we use the dialect conversion framework again, but this time for a **full** conversion to the LLVM dialect. Everything except `toy.print` was already lowered in Chapter 5, so the chapter first lowers `toy.print` (2.1–2.3) and then assembles the usual three components — conversion target, type converter, patterns — plus the conversion call (2.4–2.7). All code below is the real code from `mlir/LowerToLLVM.cpp` unless labeled otherwise; section 5 shows how to arrive at this file without having it.

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

The key trick, and the reason this chapter introduces **transitive lowering** here: `PrintOpLowering` does **not** emit LLVM-dialect branches directly. It emits `scf.for` loops and `memref.load`s — *higher-level* dialects — and lets the other conversion patterns in the very same `applyFullConversion` lower those to `cf` branches and LLVM GEPs. The conversion framework may apply several patterns in sequence to legalize one operation; as long as a lowering path from `scf` to LLVM exists (section 2.6), the structured loop nest is fine.

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

Real output **[1]**:

```mlir
    toy.print %alloc : memref<2x2xf64>
```

- `toy.print` is the only `toy` op left, which is why `PrintOpLowering` is the only Toy pattern this chapter needs (section 2.6).
- Its operand `%alloc` is the `memref<2x2xf64>` buffer Chapter 5 allocated for the constant, so the `llvm::cast<MemRefType>` above cannot fail, and `memRefShape` is `[2, 2]`. The whole module at this level is in section 4.2.

### 2.2 The helpers: the `printf` declaration and format strings as `llvm.mlir.global`

The pattern relies on three private static helpers at the end of the class, which get, or build, the declaration for `printf` and the format strings:

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

**`getOrCreateGlobalString`.** C string literals become module-level LLVM globals. It creates (once, memoized by symbol lookup) an internal constant global holding the raw bytes, then computes a pointer to its first character. Step by step:

1. `module.lookupSymbol<LLVM::GlobalOp>(name)` — reuse if a global with that symbol already exists (so ten `toy.print`s still produce one `@frmt_spec`).
2. Otherwise, an `InsertionGuard` temporarily moves the builder to the **start of the module** and creates `llvm.mlir.global internal constant @frmt_spec("%f \00")` of type `!llvm.array<4 x i8>`. Note the explicit `\0` terminators in the `StringRef("%f \0", 4)` literals of section 2.1 — LLVM globals don't NUL-terminate for you.
3. `llvm.mlir.addressof` yields the address of the global, and an `llvm.getelementptr` with indices `[0, 0]` produces the `!llvm.ptr` to the first `i8` — the `getelementptr inbounds [4 x i8], ptr @frmt_spec, i64 0, i64 0` idiom from C compilers.

Two MLIR 20 details show in these helpers: pointers are *opaque* — the only `LLVMPointerType::get` overload is `get(MLIRContext *context, unsigned addressSpace = 0)`, giving `!llvm.ptr`, which is why the GEP above carries the element type (`global.getType()`) as a separate argument — and `SymbolRefAttr::get` takes the context first, `(context, "printf")`.

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
- **Body cleanup:** `scf::ForOp` auto-creates a body with a terminator; the inner `eraseOp` loop clears it so we control the body exactly. `make_early_inc_range` advances the iterator *before* the current op is erased.
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

`LLVMConversionTarget` is a `ConversionTarget` subclass whose header describes it as a "derived class that automatically populates legalization information for different LLVM ops" — it marks the LLVM dialect legal. The only extra legal op is `ModuleOp` — the top-level container must survive. *Everything else* — `toy.*`, `affine.*`, `scf.*`, `cf.*`, `arith.*`, `memref.*`, `func.*` — is illegal and must be converted.

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

Since Toy's own tensor types were already eliminated in Chapter 5, the stock converter needs no customization: the default converter is enough for this use case.

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

This is the "transitive lowering" payoff, which the source comment above spells out: nobody wrote an `affine.for` → `llvm.br` pattern. The driver applies affine→scf, then scf→cf, then cf→llvm patterns *recursively on the results of each other* within a single conversion.

The "`std`" in the source comment is the old Standard dialect, whose ops now live in `func`, `arith`, `memref`, and `cf`.

**Every edge needs its pattern set.** Drop one, and the full conversion fails on the first op nobody can lower. A scratch copy of the chapter without the memref→LLVM line (deleted by `sed` in a temporary directory; the build takes about 15 seconds) shows it:

```bash
cd /Users/roy/study/mlir/toy/Ch6
S=$(mktemp -d)
cp -R CMakeLists.txt include mlir parser toyc.cpp "$S"/
sed -i '' '/populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);/d' "$S"/mlir/LowerToLLVM.cpp
cmake -S "$S" -B "$S"/build -G Ninja >/dev/null && cmake --build "$S"/build >/dev/null
"$S"/build/toyc-ch6 ../../test_Example/Toy/Ch6/jit.toy -emit=mlir-llvm 2>&1; echo "toyc-ch6 exit status: $?"
rm -rf "$S"
```

Real output **[2]**:

```text
loc("../../test_Example/Toy/Ch6/jit.toy":5:8): error: failed to legalize operation 'memref.alloc'
toyc-ch6 exit status: 4
```

- `memref.alloc` — the buffer of `jit.toy`'s constant — is the first op left without a pattern, and `applyFullConversion` names it.
- The pass fails, so the driver returns 4 without dumping anything.

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

Unlike Chapter 5's `applyPartialConversion`, `applyFullConversion` fails if *any* illegal op survives, and it names the op that could not be lowered (the `memref.alloc` error in section 2.6 is exactly such a diagnostic) instead of silently emitting broken IR. In MLIR 20 its third parameter is a `const FrozenRewritePatternSet &`, built from a `RewritePatternSet &&` — hence `std::move(patterns)`.

#### The working example, lowered to the LLVM dialect

The working example is the Toy IR of the lit test `llvm-lowering.mlir`: a 2×3 constant, transposed, multiplied by itself element-wise, and printed. The `CHECK` lines at the bottom are for the `-emit=llvm -opt` output (section 3.1, run by FileCheck in appendix A.1):

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

Real output **[3]** (abridged; the full dump is 226 lines) — the module-level declarations, the innermost print-loop body, and the cleanup block:

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

`^bb16` prints one element and `^bb18` frees the three buffers. The representation is the MLIR 20 LLVM dialect with opaque pointers:

- Every pointer is `!llvm.ptr`; there are no typed pointers, so there is no `llvm.bitcast` before `free`.
- The memref descriptor has five fields (section 2.5): element accesses go through the *aligned* pointer `[1]`, while `free` receives the *allocated* pointer `[0]`.
- Loads and GEPs name their element type (`llvm.load %166 : !llvm.ptr -> f64`, `... -> !llvm.ptr, f64`), and calls to the variadic `printf` carry `vararg(!llvm.func<i32 (ptr, ...)>)` (section 2.2).

The full conversion left nothing but `llvm.*` ops inside the `builtin.module` — exactly what the `LLVMConversionTarget` of section 2.4 demands. A fully annotated walk through the smaller `jit.toy` dump — malloc, descriptor construction, the loop CFG, the newline call — is in section 4.2.

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

- **`createReconcileUnrealizedCastsPass()`** — added in this repo (hence the `// FIX:` comment and the matching `#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"` near the top of `toyc.cpp`). Dialect conversion can insert `builtin.unrealized_conversion_cast` "bridge" ops at type-boundary seams (e.g. `index` ↔ `i64`). They are placeholders that must cancel out in pairs; this pass erases them. If one survives, `translateModuleToLLVMIR`/the JIT meets a non-LLVM op in an "all-LLVM" module. With the current code none of the `test_Example/Toy/Ch6` inputs leaves a cast behind after `toy-to-llvm` (the run at the end of this section), so for them the pass is only a safety net.
- **`createDIScopeForLLVMFuncOpPass()`** — attaches debug-info scopes so the exported LLVM IR carries line tables (the `!dbg` metadata in the `-emit=llvm` output of section 3.1).

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

Real output **[4]** (exit status 1 — `grep -c` exits 1 when it counts no match):

```text
0
0
0
```

- Every count is 0, including the dump right after `toy-to-llvm`: for these inputs the full conversion leaves no bridge cast for the next pass to remove.
- The reconcile pass therefore has nothing to do here; it stays as a safety net for inputs that do leave casts behind.

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

- **Translation registration.** MLIR-to-LLVM-IR export is pluggable via *dialect translation interfaces*. `registerLLVMDialectTranslation` teaches the exporter how each `llvm.*` op maps to an LLVM instruction; `registerBuiltinDialectTranslation` handles the builtin dialect (the module op). Without them, `translateModuleToLLVMIR` fails (run below). This is a classic out-of-tree stumbling block, since `mlir-translate`-style tools register them for you.
- **`translateModuleToLLVMIR`** walks the MLIR module and builds a genuine `llvm::Module` in a fresh `llvm::LLVMContext` (LLVM contexts are not thread-safe, so a fresh one is used). This is a *translation* (1:1 export), not a pattern-based conversion — which is why the module had to be 100 % LLVM dialect first.
- `InitializeNativeTarget()` / `InitializeNativeTargetAsmPrinter()` link in the host backend (AArch64 here) so a `TargetMachine` can exist.
- `JITTargetMachineBuilder::detectHost()` + `setupTargetTripleAndDataLayout` stamp the module with the host triple and data layout (`target triple = "arm64-apple-darwin25.6.0"` in the output below).
- `makeOptimizingTransformer(optLevel, sizeLevel, targetMachine)` returns a function object wrapping LLVM's standard `-O<N>` pass pipeline: `-O0` by default, `-O3` with `-opt`.
- The module is printed to `llvm::errs()`, i.e. stderr — hence `2>&1` in the commands below.

In MLIR 20, `translateModuleToLLVMIR` takes the `llvm::LLVMContext` to build in, and `ExecutionEngine::setupTargetTripleAndDataLayout` needs a `TargetMachine` — which is why the code above first builds one from `JITTargetMachineBuilder::detectHost()`.

**Without the translation registration.** A scratch copy whose `dumpLLVMIR` lacks the two `register...Translation` calls (deleted by `sed`; about 15 seconds to build) can't export the module:

```bash
cd /Users/roy/study/mlir/toy/Ch6
S=$(mktemp -d)
cp -R CMakeLists.txt include mlir parser toyc.cpp "$S"/
sed -i '' '/^int dumpLLVMIR/,/^}/{/register.*DialectTranslation/d;}' "$S"/toyc.cpp
cmake -S "$S" -B "$S"/build -G Ninja >/dev/null && cmake --build "$S"/build >/dev/null
"$S"/build/toyc-ch6 ../../test_Example/Toy/Ch6/jit.toy -emit=llvm 2>&1; echo "toyc-ch6 exit status: $?"
rm -rf "$S"
```

Real output **[5]**:

```text
error: cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface` registration for dialect for op: builtin.module
Failed to emit LLVM IR
toyc-ch6 exit status: 255
```

The exporter finds no translation interface for the very first op it meets, the `builtin.module`, and `dumpLLVMIR` returns −1, which the shell sees as 255.

#### Exporting the working example

The input is `llvm-lowering.mlir` from section 2.7, and `-emit=llvm` runs `dumpLLVMIR` above without `-opt`, i.e. with the `-O0` transformer:

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/llvm-lowering.mlir -emit=llvm 2>&1
```

Real output **[6]** (abridged; the full module is 184 lines) — the module header, the element-print block, the newline block, and the cleanup:

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

- **Host stamping:** the `target datalayout` and `target triple` lines come from `setupTargetTripleAndDataLayout` with the host `TargetMachine`.
- **Opaque pointers:** every pointer is `ptr`, `free` takes the allocated pointer without a cast, and the format string is passed as plain `ptr @frmt_spec` (the zero-index GEP of section 2.2 folds away).
- **Debug info:** the `!dbg` attachments come from the `DIScopeForLLVMFuncOp` pass (section 2.8).

#### The same module with `-opt`

With `-opt`, `makeOptimizingTransformer` gets `optLevel` 3:

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/llvm-lowering.mlir -emit=llvm -opt 2>&1
```

Real output **[7]** (abridged to `main`; the full module is 43 lines):

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

- The `-O3` pipeline trims `main` down to straight-line code: the loops are unrolled, the loads constant-folded, and `malloc`/`free` removed.
- LLVM's library-call simplification rewrote each `printf("\n")` into `putchar(10)`, so `@nl` is gone from the module.
- The six constants are the squares of the transposed matrix, row by row: 1, 16, 4, 25, 9, 36 — the values the `CHECK-SAME` lines of `llvm-lowering.mlir` expect (appendix A.1).

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

In MLIR 20, `ExecutionEngine::create(Operation *op, const ExecutionEngineOptions &options = {}, std::unique_ptr<llvm::TargetMachine> tm = nullptr)` takes the pipeline in `ExecutionEngineOptions::transformer`.

**`invokePacked`, not `invoke`.** `invoke(funcName, args...)` calls `invokePacked("_mlir_ciface_" + funcName, ...)`, the C-interface wrapper that exists only for functions marked `llvm.emit_c_interface`. A scratch copy that calls `invoke("main")` (patched by `sed`; about 15 seconds to build) compiles but can't run `main`:

```bash
cd /Users/roy/study/mlir/toy/Ch6
S=$(mktemp -d)
cp -R CMakeLists.txt include mlir parser toyc.cpp "$S"/
sed -i '' 's/engine->invokePacked("main")/engine->invoke("main")/' "$S"/toyc.cpp
cmake -S "$S" -B "$S"/build -G Ninja >/dev/null && cmake --build "$S"/build >/dev/null
"$S"/build/toyc-ch6 ../../test_Example/Toy/Ch6/jit.toy -emit=jit 2>&1; echo "toyc-ch6 exit status: $?"
rm -rf "$S"
```

Real output **[8]**:

```text
JIT invocation failed
toyc-ch6 exit status: 255
```

There is no `_mlir_ciface_main`, so the call fails and `runJit` returns −1 (255). `invokePacked("main")` calls `main` itself.

#### A program on stdin, run by the JIT

A one-line program can be piped into the driver. With no input file, the positional argument defaults to `-`, which `getFileOrSTDIN` reads as stdin:

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

Real output **[9]**:

```text
1.000000 2.000000 
3.000000 4.000000 
```

- The whole pipeline ran in one process: parse, Toy passes, the affine and LLVM lowerings, translation, `-O0` codegen, and the call through `invokePacked("main")`. Nothing was written to disk.
- The output is the program's own `printf`s, so it goes to stdout (each line ends in a space, from the `"%f "` format string); the `-emit=mlir*`/`-emit=llvm` dumps go to stderr instead.
- Section 4.2 runs the same program from `jit.toy`.

### 3.3 Comparing the IR levels

To compare the levels, run `-emit=mlir`, `-emit=mlir-affine`, `-emit=mlir-llvm`, and `-emit=llvm`, and use `--mlir-print-ir-after-all` to watch the IR evolve. Each `-emit` value is one entry of the driver's `Action` enum, and because the enum is ordered, every action runs all the pass stages before it:

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

Section 4.2 runs every level on `jit.toy`, and appendix A.1 runs `llvm-lowering.mlir` through the JIT and its FileCheck lines.

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

Real output **[10]**:

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

The superbuild (`toy/build.sh`, `toy/run.sh`, `CMakePresets.json`) is documented once in the top-level [README](../README.md#the-build-system). This section builds and runs Chapter 6 on its own, from the chapter directory. What the chapter's `CMakeLists.txt` adds — the ExecutionEngine guard and the shared-only link line — is in appendix C.

Every command in this README is numbered by its output (**[1]**–**[33]**). After building, `./examples.sh` runs them all, printing each command followed by its output and exit status; `./examples.sh 2` runs only command [2], and `./examples.sh -l` lists them (appendix D).

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

The chapter targets are only defined when the installed MLIR was built with the ExecutionEngine (`MLIR_ENABLE_EXECUTION_ENGINE`, set to `1` in Homebrew's `MLIRConfig.cmake`); otherwise configuring succeeds but there is nothing to build. A standalone build puts the binary directly in `build/`; the superbuild's `toy/build/bin/toyc-ch6` behaves identically. To confirm the binary links only the shared MLIR/LLVM libraries, see appendix C.3.

### 4.2 Running

**The input.** A 2×2 constant, printed — the same program that `run.sh ch6` and section 3.2 pipe in on stdin.

***test_Example/Toy/Ch6/jit.toy***
```text
# RUN: toyc-ch6 -emit=jit %s
# UNSUPPORTED: target={{.*windows.*}}

def main() {
 print([[1, 2], [3, 4]]);
}
```

- `../../test_Example/Toy/Ch6/jit.toy` — the input program (above). Omitting it (or passing `-`) reads from stdin, which is how `run.sh` feeds its inline program.
- `-emit=<action>` — how far down the pipeline to go. This chapter adds three actions to Chapter 5's `ast`, `mlir`, and `mlir-affine`: `mlir-llvm` (dump after the LLVM-dialect lowering), `llvm` (translate and print LLVM IR), and `jit` (compile and run `main`). The actions are ordered, so each one runs every pass stage before it (section 3.3).
- `-opt` — enable the Toy-level optimizations (inlining, shape inference, canonicalization, CSE — which `mlir-affine` and beyond run anyway), the affine loop fusion/scalar replacement cleanups, and `-O3` in the LLVM pipeline used by `-emit=llvm` and `-emit=jit` (section 3).
- `-x mlir` (or a `.mlir` extension) — parse the input as MLIR instead of Toy source.
- `2>&1` — `-emit=mlir*` dumps via `module->dump()` and `-emit=llvm` prints via `llvm::errs()`, so both go to **stderr**. Only `-emit=jit` output goes to stdout, because it is the program's own `printf`.

The program at every level of the pipeline:

#### `-emit=mlir` — pure Toy dialect

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/jit.toy -emit=mlir 2>&1
```

Real output **[11]**:

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

#### `-emit=mlir-affine` — after Chapter 5's partial lowering

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/jit.toy -emit=mlir-affine 2>&1
```

Real output **[12]**:

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

#### `-emit=mlir-llvm` — after `ToyToLLVMLoweringPass` (still MLIR!)

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/jit.toy -emit=mlir-llvm 2>&1
```

Real output **[13]** (abridged; the full dump is 101 lines — the elided parts are more index constants and the four element stores, each an `extractvalue` + `mul`/`add` + `getelementptr` + `llvm.store`):

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

#### `-emit=llvm` — genuine LLVM IR

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/jit.toy -emit=llvm 2>&1
```

Real output **[14]** (abridged; unoptimized, `-opt` not passed):

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

Real output **[15]** (abridged: the debug-info metadata at the end is elided):

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

The loops are fully unrolled and the loads constant-folded, so `main` collapses into straight-line code: four `printf` calls with immediate `double` constants. `malloc`/`free` are gone too, and LLVM's library-call simplification has rewritten each `printf("\n")` into `putchar(10)` (so the unused `@nl` global disappears).

#### `-emit=jit` — actually running it

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/jit.toy -emit=jit
```

Real output **[16]**:

```text
1.000000 2.000000 
3.000000 4.000000 
```

The ExecutionEngine JIT-compiled `main` to AArch64 code, `invokePacked("main")` called it, the JIT resolved `printf`/`malloc`/`free` against the host libc — and the Toy program printed its matrix (section 3.2). End-to-end: source text to executed native code in one process, no files written. `-emit=jit -opt` prints the same thing. (Each line really ends in a space, from the `"%f "` format string.) Section 3.2 runs the same program on stdin.

---

## 5. Deriving the Chapter 6 lowering, driver and build yourself

Section 2 explains `mlir/LowerToLLVM.cpp` as it is, section 3 the driver code that exports and runs its output, and appendix C the build. This section shows how to arrive at them without having the code. `mlir/LowerToLLVM.cpp` is the chapter's only new source, and most of it is assembled from parts that MLIR or Chapter 5 already has; only the lowering of `toy.print` has to be designed. The other changes are edits to existing files:

| Part | Lines | Where it comes from |
|---|---|---|
| `mlir/LowerToLLVM.cpp`: header comment and `#include`s | 1–55 | One header per `populate*` function (section 5.2) and per dialect whose ops the file creates |
| `mlir/LowerToLLVM.cpp`: `PrintOpLowering` | 61–177 | The IR you want for `toy.print`, written by hand first (section 5.3), then turned into builder calls (section 5.4) |
| `mlir/LowerToLLVM.cpp`: `ToyToLLVMLoweringPass` and its `runOnOperation` | 183–241 | Chapter 5's pass skeleton (section 5.5), with the six `populate*` calls of section 5.2 |
| `include/toy/Passes.h`: `createLowerToLLVMPass()` | 28–30 | Chapter 5's `createLowerToAffinePass()`, copied (section 5.5) |
| `toyc.cpp`: the `isLoweringToLLVM` stage and its two extra passes | 157, 187–205 | The stock `mlir-opt` passes, tried first (section 5.6) |
| `toyc.cpp`: the three new `-emit` actions, `dumpLLVMIR`, `runJit`, the LLVM inliner registration, and their `#include`s | 13–55, 75–96, 226–301, 317, 327–340 | Stock `mlir-runner` and `mlir-translate`, which do the same from the command line (section 5.6), then one header per API (section 5.7) |
| `CMakeLists.txt`: the ExecutionEngine guard and the new link line | 36–39, 66–73 | The linker's undefined symbols and the installed MLIR CMake package (section 5.8) |

`include/toy/Ops.td` differs from Chapter 5 too, but only in the order of `ReshapeOp`'s fields; it generates the same C++ (section 5.5).

The steps below use the working example of section 2.7, `llvm-lowering.mlir`, because its transpose and multiply leave `affine` loops behind for the lowering to handle.

### 5.1 Step 1: list what is left to lower

The input of the new pass is the output of Chapter 5's pipeline, `-emit=mlir-affine`. The generic printer (`--mlir-print-op-generic`) writes every op under its full quoted name (`"func.return"` instead of `return`), so `grep` can count them:

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/llvm-lowering.mlir -emit=mlir-affine --mlir-print-op-generic 2>&1 \
  | grep -oE '"[a-z_]+\.[a-z_]+"' | sort | uniq -c
```

Real output **[17]**:

```text
   4 "affine.for"
   2 "affine.load"
   8 "affine.store"
   4 "affine.yield"
   6 "arith.constant"
   1 "arith.mulf"
   1 "builtin.module"
   1 "func.func"
   1 "func.return"
   3 "memref.alloc"
   3 "memref.dealloc"
   1 "toy.print"
```

- Apart from `builtin.module`, which stays legal (section 2.4), five dialects are left: `affine`, `arith`, `func`, `memref`, and `toy`. Each needs a path to `llvm`.
- `toy.print` is the only Toy op, so it is the only pattern this file has to write itself (section 2.1).
- Two more dialects join on the way: `affine` lowers to `scf` loops (as `toy.print` will), and `scf` lowers to `cf` (control flow) branches. They need paths to `llvm` too, which gives the six edges of section 2.6.

### 5.2 Step 2: find the stock conversion for each dialect

MLIR ships a pass for each of these edges. `mlir-opt` lists them:

```bash
cd /Users/roy/study/mlir/toy/Ch6
mlir-opt --help | grep -E -- '--(lower-affine|convert-scf-to-cf|convert-arith-to-llvm|finalize-memref-to-llvm|convert-cf-to-llvm|convert-func-to-llvm) '
```

Real output **[18]**:

```text
      --convert-arith-to-llvm                                -   Convert Arith dialect to LLVM dialect
      --convert-cf-to-llvm                                   -   Convert ControlFlow operations to the LLVM dialect
      --convert-func-to-llvm                                 -   Convert from the Func dialect to the LLVM dialect
      --convert-scf-to-cf                                    -   Convert SCF dialect to ControlFlow dialect, replacing structured control flow with a CFG
      --finalize-memref-to-llvm                              -   Finalize MemRef dialect to LLVM dialect conversion
      --lower-affine                                         -   Lower Affine operations to a combination of Standard and SCF operations
```

Each pass is a thin wrapper around a `populate…Patterns` function that adds its patterns to a `RewritePatternSet`; that function is what a C++ pass can reuse. It is declared in a header under `include/mlir/Conversion/`, in a directory named after the edge:

```bash
cd /Users/roy/study/mlir/toy/Ch6
I=/opt/homebrew/opt/llvm@20/include/mlir/Conversion
for d in AffineToStandard SCFToControlFlow ArithToLLVM MemRefToLLVM ControlFlowToLLVM FuncToLLVM; do
  grep -h -A1 'void populate' $I/$d/*.h
done
```

Real output **[19]**:

```text
void populateAffineToStdConversionPatterns(RewritePatternSet &patterns);

--
void populateAffineToVectorConversionPatterns(RewritePatternSet &patterns);

void populateSCFToControlFlowConversionPatterns(RewritePatternSet &patterns);

void populateArithToLLVMConversionPatterns(const LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns);
void populateFinalizeMemRefToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns);
void populateControlFlowToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns);
--
void populateAssertToLLVMConversionPattern(const LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns,
void populateFuncToLLVMFuncOpConversionPattern(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns);
--
void populateFuncToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns,
```

- The six functions in section 2.6 are here, one per directory; the extra ones (`populateAffineToVectorConversionPatterns`, the `assert` and `func.func`-only patterns) cover other edges or a subset of one.
- The signatures show which pattern sets need the type converter of section 2.5: the four that end in LLVM take a `const LLVMTypeConverter &`, while `affine`→`scf` and `scf`→`cf` take only the pattern set.
- Each directory's header is one of the file's `#include`s (`mlir/Conversion/AffineToStandard/AffineToStandard.h`, …).

### 5.3 Step 3: write `toy.print`'s target IR by hand and test the lowering with stock tools

Only `toy.print` has no stock conversion. Before writing any C++, write the IR it should become, for the `memref<3x2xf64>` that `toy.print` receives in this example: a `printf` declaration and two format strings at module level, and a loop nest that loads each element and prints it, with a newline after each row. Then test it: splice that IR into the `-emit=mlir-affine` output in place of `toy.print`, lower the result with the six stock passes of section 5.2, and run it. `DECLS` goes right after `module {`, `PRINT` replaces the `toy.print` line:

```bash
cd /Users/roy/study/mlir/toy/Ch6
export DECLS='  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @frmt_spec("%f \00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @nl("\0A\00") {addr_space = 0 : i32}'
export PRINT='    %fmt_addr = llvm.mlir.addressof @frmt_spec : !llvm.ptr
    %zero = llvm.mlir.constant(0 : index) : i64
    %fmt = llvm.getelementptr %fmt_addr[%zero, %zero] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<4 x i8>
    %nl_addr = llvm.mlir.addressof @nl : !llvm.ptr
    %nl = llvm.getelementptr %nl_addr[%zero, %zero] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<2 x i8>
    %lb = arith.constant 0 : index
    %ub0 = arith.constant 3 : index
    %ub1 = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %i = %lb to %ub0 step %step {
      scf.for %j = %lb to %ub1 step %step {
        %elt = memref.load %alloc[%i, %j] : memref<3x2xf64>
        %r0 = llvm.call @printf(%fmt, %elt) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
      }
      %r1 = llvm.call @printf(%nl) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    }'
S=$(mktemp -d)
./build/toyc-ch6 ../../test_Example/Toy/Ch6/llvm-lowering.mlir -emit=mlir-affine 2>&1 \
  | awk '/^module \{$/ { print; print ENVIRON["DECLS"]; next }
         /toy\.print %alloc : memref<3x2xf64>/ { print ENVIRON["PRINT"]; next }
         { print }' \
  | mlir-opt --lower-affine --convert-scf-to-cf --convert-arith-to-llvm \
      --finalize-memref-to-llvm --convert-cf-to-llvm --convert-func-to-llvm > "$S"/llvm.mlir
grep -c unrealized_conversion_cast "$S"/llvm.mlir
mlir-opt --reconcile-unrealized-casts "$S"/llvm.mlir | mlir-translate --mlir-to-llvmir | lli > "$S"/stock.txt
cat "$S"/stock.txt
./build/toyc-ch6 ../../test_Example/Toy/Ch6/llvm-lowering.mlir -emit=jit > "$S"/toyc.txt
diff "$S"/stock.txt "$S"/toyc.txt && echo "same output as toyc-ch6 -emit=jit"
rm -rf "$S"
```

Real output **[20]**:

```text
18
1.000000 16.000000 
4.000000 25.000000 
9.000000 36.000000 
same output as toyc-ch6 -emit=jit
```

- The hand-written IR, lowered only by stock passes, prints the same matrix as the real `toy-to-llvm` pass running in the JIT. The design of `PrintOpLowering` is settled before it has a line of C++, and `mlir-opt`'s verifier has checked every op of it.
- Six separate passes leave 18 `builtin.unrealized_conversion_cast`s at the seams between them, where one pass has converted a value's type and the next has not yet converted its user; `--reconcile-unrealized-casts` cancels them (section 2.8). The repo's single `applyFullConversion` converts all edges together and leaves none (section 2.8's count).

**Write a lowering backwards: first the IR you want, checked with the stock tools, then the C++ that builds it.** The C++ then has a fixed target instead of an open design question.

### 5.4 Step 4: turn each line of the IR into a builder call

Each line of the section 5.3 IR becomes one `create<OpTy>(loc, ...)` call in `PrintOpLowering`. The real code is in sections 2.2 and 2.3; side by side:

| Hand-written IR (section 5.3) | C++ in `PrintOpLowering` | Section |
|---|---|---|
| `llvm.func @printf(!llvm.ptr, ...) -> i32` | `rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", getPrintfType(context))` | 2.2 |
| `llvm.mlir.global internal constant @frmt_spec("%f \00")` | `builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true, LLVM::Linkage::Internal, name, builder.getStringAttr(value), /*alignment=*/0)` | 2.2 |
| `llvm.mlir.addressof @frmt_spec` | `builder.create<LLVM::AddressOfOp>(loc, global)` | 2.2 |
| `llvm.mlir.constant(0 : index) : i64` | `builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(), builder.getIndexAttr(0))` | 2.2 |
| `llvm.getelementptr %fmt_addr[%zero, %zero]` | `builder.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(...), global.getType(), globalPtr, ArrayRef<Value>({cst0, cst0}))` | 2.2 |
| `arith.constant 0 / 3 / 1 : index` | `rewriter.create<arith::ConstantIndexOp>(loc, 0)`, `(loc, memRefShape[i])`, `(loc, 1)` | 2.3 |
| `scf.for %i = %lb to %ub0 step %step` | `rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step)` | 2.3 |
| `memref.load %alloc[%i, %j]` | `rewriter.create<memref::LoadOp>(loc, printOp.getInput(), loopIvs)` | 2.3 |
| `llvm.call @printf(%fmt, %elt) vararg(...)` | `rewriter.create<LLVM::CallOp>(loc, getPrintfType(context), printfRef, ArrayRef<Value>({formatSpecifierCst, elementLoad}))` | 2.3 |
| `llvm.call @printf(%nl) vararg(...)` at the end of the outer loop | the same call with `newLineCst`, at the end of every loop body but the innermost | 2.3 |

What the C++ adds over the hand-written IR is generality: the bounds `3` and `2` become `memRefShape[i]` and the two nested loops become a C++ loop over the dimensions; the declaration and the strings are created only if the module doesn't have them yet; and the `scf.yield` that the printed IR hides is created explicitly, after the default body `scf::ForOp` builds is erased (section 2.3).

**Which arguments `create` takes.** `rewriter.create<OpTy>(loc, args...)` calls `OpTy::build(builder, state, args...)`, so the `build` overloads of an op's class are the menu. They are generated from the op's ODS (Operation Definition Specification) record; `mlir-tblgen` prints them from the installed `.td` files:

```bash
cd /Users/roy/study/mlir/toy/Ch6
I=/opt/homebrew/opt/llvm@20/include
mlir-tblgen -gen-op-decls -I $I $I/mlir/Dialect/LLVMIR/LLVMOps.td \
  | awk '/^class CallOp :/,/^};/' | grep 'static void build' | head -8
mlir-tblgen -gen-op-decls -I $I $I/mlir/Dialect/SCF/IR/SCFOps.td \
  | awk '/^class ForOp :/,/^};/' | grep 'static void build'
```

Real output **[21]**:

```text
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, LLVMFuncOp func, ValueRange args);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, LLVMFunctionType calleeType, ValueRange args);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, StringAttr callee, ValueRange args = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, FlatSymbolRefAttr callee, ValueRange args = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, StringRef callee, ValueRange args = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, LLVMFunctionType calleeType, StringAttr callee, ValueRange args = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, LLVMFunctionType calleeType, FlatSymbolRefAttr callee, ValueRange args = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, LLVMFunctionType calleeType, StringRef callee, ValueRange args = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value lowerBound, Value upperBound, Value step, ValueRange initArgs = std::nullopt, function_ref<void(OpBuilder &, Location, Value, ValueRange)> odsArg4 = nullptr);
```

- The first eight are `LLVM::CallOp`'s hand-written builders (`head -8` drops the long all-attributes ones that ODS adds after them). `PrintOpLowering` uses the one taking `(LLVMFunctionType calleeType, FlatSymbolRefAttr callee, ValueRange args)`: `getPrintfType(context)`, `printfRef`, and the operands.
- `scf::ForOp` has one builder; the `PrintOpLowering` call passes only the bounds and the step, so `initArgs` stays empty (no loop-carried values) and the body builder is `nullptr`, which is why the body gets a default `scf.yield` that section 2.3 erases.

**Why the call needs the function type.** The overload with `calleeType` is not arbitrary: a call to a variadic function must record the callee's type, and the verifier rejects one that doesn't. The section 5.3 call without its `vararg(...)` shows it:

```bash
cd /Users/roy/study/mlir/toy/Ch6
mlir-opt <<'EOF'
llvm.func @printf(!llvm.ptr, ...) -> i32
llvm.func @f(%p: !llvm.ptr, %x: f64) {
  %r = llvm.call @printf(%p, %x) : (!llvm.ptr, f64) -> i32
  llvm.return
}
EOF
```

Real output **[22]** (exit status 1):

```text
<stdin>:3:8: error: 'llvm.call' op missing var_callee_type attribute for vararg call
  %r = llvm.call @printf(%p, %x) : (!llvm.ptr, f64) -> i32
       ^
<stdin>:3:8: note: see current operation: %0 = "llvm.call"(%arg0, %arg1) <{CConv = #llvm.cconv<ccc>, TailCallKind = #llvm.tailcallkind<none>, callee = @printf, fastmathFlags = #llvm.fastmath<none>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 2, 0>}> : (!llvm.ptr, f64) -> i32
```

`var_callee_type` is the attribute that `vararg(...)` prints; the builder fills it from `calleeType`, which is why every `LLVM::CallOp` in `PrintOpLowering` passes `getPrintfType(context)` (section 2.2).

### 5.5 Step 5: reuse Chapter 5's pass skeleton

The pass around the patterns has the same shape as Chapter 5's `ToyToAffineLoweringPass`. The lines that make up the skeleton, in both files, then the two headers Chapter 6 touches, compared with Chapter 5's (`Ops.td` by the C++ it generates):

```bash
cd /Users/roy/study/mlir/toy/Ch6
grep -nE 'PassWrapper<|getDependentDialects|ConversionTarget target|TypeConverter typeConverter|RewritePatternSet patterns|patterns.add<|apply(Partial|Full)Conversion|std::make_unique<' \
  ../Ch5/mlir/LowerToAffineLoops.cpp mlir/LowerToLLVM.cpp
diff ../Ch5/include/toy/Passes.h include/toy/Passes.h
I=/opt/homebrew/opt/llvm@20/include
for g in -gen-op-decls -gen-op-defs; do
  diff <(mlir-tblgen $g -I $I -I ../Ch5/include ../Ch5/include/toy/Ops.td) \
       <(mlir-tblgen $g -I $I -I include include/toy/Ops.td) && echo "Ops.td: same $g output as Chapter 5"
done
```

Real output **[23]**:

```text
../Ch5/mlir/LowerToAffineLoops.cpp:329:    : public PassWrapper<ToyToAffineLoweringPass, OperationPass<ModuleOp>> {
../Ch5/mlir/LowerToAffineLoops.cpp:333:  void getDependentDialects(DialectRegistry &registry) const override {
../Ch5/mlir/LowerToAffineLoops.cpp:344:  ConversionTarget target(getContext());
../Ch5/mlir/LowerToAffineLoops.cpp:367:  RewritePatternSet patterns(&getContext());
../Ch5/mlir/LowerToAffineLoops.cpp:368:  patterns.add<AddOpLowering, ConstantOpLowering, FuncOpLowering, MulOpLowering,
../Ch5/mlir/LowerToAffineLoops.cpp:376:          applyPartialConversion(getOperation(), target, std::move(patterns))))
../Ch5/mlir/LowerToAffineLoops.cpp:383:  return std::make_unique<ToyToAffineLoweringPass>();
mlir/LowerToLLVM.cpp:185:    : public PassWrapper<ToyToLLVMLoweringPass, OperationPass<ModuleOp>> {
mlir/LowerToLLVM.cpp:189:  void getDependentDialects(DialectRegistry &registry) const override {
mlir/LowerToLLVM.cpp:200:  LLVMConversionTarget target(getContext());
mlir/LowerToLLVM.cpp:208:  LLVMTypeConverter typeConverter(&getContext());
mlir/LowerToLLVM.cpp:218:  RewritePatternSet patterns(&getContext());
mlir/LowerToLLVM.cpp:228:  patterns.add<PrintOpLowering>(&getContext());
mlir/LowerToLLVM.cpp:233:  if (failed(applyFullConversion(module, target, std::move(patterns))))
mlir/LowerToLLVM.cpp:240:  return std::make_unique<ToyToLLVMLoweringPass>();
27a28,31
> /// Create a pass for lowering operations the remaining `Toy` operations, as
> /// well as `Affine` and `Std`, to the LLVM dialect for codegen.
> std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
> 
Ops.td: same -gen-op-decls output as Chapter 5
Ops.td: same -gen-op-defs output as Chapter 5
```

Both are a `PassWrapper` on `ModuleOp` that declares its dependent dialects, builds a target and a pattern set, applies a conversion, and is created by a factory function. Four things differ, each for a reason sections 2.4–2.7 explain:

- **Target:** a plain `ConversionTarget` with legal and illegal dialects listed by hand becomes `LLVMConversionTarget`, which already marks the LLVM dialect legal (section 2.4).
- **Type converter:** new in this chapter; `LLVMTypeConverter` maps `memref` and `index` to LLVM types (section 2.5).
- **Patterns:** Chapter 5 adds only its own patterns; here the six `populate*` calls of section 5.2 come first and `PrintOpLowering` is the only own pattern (section 2.6).
- **Conversion:** `applyPartialConversion`, which let `toy.print` survive, becomes `applyFullConversion`, which lets nothing but LLVM survive (section 2.7).

`getDependentDialects` lists the dialects the patterns create that the input may not contain: `llvm`, and `scf` for `PrintOpLowering`'s loops (section 2.4).

The factory is declared in `Passes.h` exactly like `createLowerToAffinePass()` above it (section 2.8). `Ops.td` only moves `ReshapeOp`'s `results` line below `hasCanonicalizer`; ODS doesn't depend on the order of an op's fields, so both chapters generate the same op classes and nothing in `Ops.td` needs changing for this chapter.

### 5.6 Step 6: run the driver's new stages with stock tools first

After `toy-to-llvm`, the driver has to take the module the rest of the way: clean it up, export it, optimize it and run it. Before writing any of that in C++, find a stock tool for each stage. The cleanup passes are `mlir-opt` passes, and `mlir-runner` JIT-runs a module that is all LLVM dialect:

```bash
cd /Users/roy/study/mlir/toy/Ch6
mlir-opt --help | grep -E -- '--(reconcile-unrealized-casts|ensure-debug-info-scope-on-llvm-func) '
./build/toyc-ch6 ../../test_Example/Toy/Ch6/jit.toy -emit=mlir-llvm 2>&1 \
  | mlir-runner -e main -entry-point-result=void -O3
```

Real output **[24]**:

```text
      --ensure-debug-info-scope-on-llvm-func                 -   Materialize LLVM debug info subprogram attribute on every LLVMFuncOp
      --reconcile-unrealized-casts                           -   Simplify and eliminate unrealized conversion casts
1.000000 2.000000 
3.000000 4.000000 
```

- **The two passes** are the ones section 3.3's IR dump lists after `toy-to-llvm`: `reconcile-unrealized-casts` removes the bridge casts that dialect conversion can leave, and `ensure-debug-info-scope-on-llvm-func` gives every `llvm.func` the debug scope behind the `!dbg` lines of section 3.1. `toyc.cpp` adds them right after `createLowerToLLVMPass()` (section 2.8); section 5.7 finds their factories.
- **`mlir-runner`** does in one tool what `runJit` does: translate the module to LLVM IR, optimize it (`-O3`, the level `makeOptimizingTransformer` takes), JIT-compile it, and call the function named by `-e`. Toy's `main` takes no arguments and returns nothing (`-entry-point-result=void`), so `runJit` calls it with `invokePacked("main")` and no arguments (section 3.2). Appendix B splits the same path into `mlir-translate --mlir-to-llvmir`, the translation that `dumpLLVMIR` performs, and `lli`.
- **The three new `-emit` actions** follow from these stage boundaries: `mlir-llvm` stops after the passes, `llvm` after the export, `jit` after the run. Appended to the `Action` enum after `DumpMLIRAffine`, they keep the `>=` comparisons of `loadAndProcessMLIR` working (section 3.3). Because `dumpLLVMIR` and `runJit` need the module after the pass pipeline has run, `main()` now creates the context and the module itself, calls `loadAndProcessMLIR` (Chapter 5's `dumpMLIR` without the dump), and picks a path by action (section 3).

**Before writing driver code, run its stages with stock tools: their flags (`-e`, `-entry-point-result`, `-O3`) list the decisions the C++ has to make.**

### 5.7 Step 7: find the header for each new API

Each stage of section 5.6 has a C++ entry point in the installed headers. A `grep` for each declaration finds the header to include:

```bash
cd /Users/roy/study/mlir/toy/Ch6
I=/opt/homebrew/opt/llvm@20/include
grep -rnm1 -E '^(translateModuleToLLVMIR|void register(LLVM|Builtin)DialectTranslation\(MLIRContext|makeOptimizingTransformer|class ExecutionEngine \{|std::unique_ptr<Pass> createReconcileUnrealizedCastsPass|std::unique_ptr<::mlir::Pass> createDIScopeForLLVMFuncOpPass\(\);|void registerInlinerInterface)' \
  $I/mlir | sed "s|$I/||"
grep -nE '^class JITTargetMachineBuilder \{|^  inline bool InitializeNativeTarget(AsmPrinter)?\(\)' \
  $I/llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h $I/llvm/Support/TargetSelect.h | sed "s|$I/||"
```

Real output **[25]**:

```text
mlir/Target/LLVMIR/Export.h:29:translateModuleToLLVMIR(Operation *module, llvm::LLVMContext &llvmContext,
mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h:27:void registerLLVMDialectTranslation(MLIRContext &context);
mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h:27:void registerBuiltinDialectTranslation(MLIRContext &context);
mlir/ExecutionEngine/ExecutionEngine.h:114:class ExecutionEngine {
mlir/ExecutionEngine/OptUtils.h:34:makeOptimizingTransformer(unsigned optLevel, unsigned sizeLevel,
mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h:23:std::unique_ptr<Pass> createReconcileUnrealizedCastsPass();
mlir/Dialect/LLVMIR/Transforms/Passes.h.inc:20:std::unique_ptr<::mlir::Pass> createDIScopeForLLVMFuncOpPass();
mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h:23:void registerInlinerInterface(DialectRegistry &registry);
llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h:34:class JITTargetMachineBuilder {
llvm/Support/TargetSelect.h:119:  inline bool InitializeNativeTarget() {
llvm/Support/TargetSelect.h:133:  inline bool InitializeNativeTargetAsmPrinter() {
```

- **Export** (`dumpLLVMIR`, section 3.1): `translateModuleToLLVMIR` in `Export.h`, and one `register…Translation` header per dialect left in the module, `llvm` and `builtin`; without them the export fails (section 3.1's run).
- **Optimization and JIT** (`runJit`, section 3.2): `OptUtils.h` for `makeOptimizingTransformer`, `ExecutionEngine.h` for `ExecutionEngine` and its `ExecutionEngineOptions`; `dumpLLVMIR` uses the latter header too, for `setupTargetTripleAndDataLayout`.
- **The two passes:** `createReconcileUnrealizedCastsPass()` sits under `mlir/Conversion/`, not `mlir/Transforms/`, which is why `toyc.cpp` includes it separately with its `// FIX:` comment (section 2.8). `createDIScopeForLLVMFuncOpPass()` is declared in the generated `Passes.h.inc`, which `mlir/Dialect/LLVMIR/Transforms/Passes.h` includes; that is the header to name.
- **The LLVM inliner interface:** the first `registerInlinerInterface` in `InlinerInterfaceImpl.h` is the LLVM dialect's (`mlir::LLVM::`); `main()` adds it to the `DialectRegistry` next to the func extensions (section 2.8).
- **The `llvm::` side:** `TargetSelect.h` for the native target initialization and `JITTargetMachineBuilder.h` for `detectHost()`, both needed before a `TargetMachine` exists (section 3.1).

Each header is one of the new `#include`s at the top of `toyc.cpp`.

### 5.8 Step 8: let the linker pick the library

With the code written, Chapter 5's link line (`MLIR` + `LLVM`) still compiles every file but can't link the program. A scratch copy with that line put back (by `sed`; the build takes about 20 seconds) shows what is missing; `nm` and the installed CMake package then show which library has it:

```bash
cd /Users/roy/study/mlir/toy/Ch6
S=$(mktemp -d)
cp -R CMakeLists.txt include mlir parser toyc.cpp "$S"/
sed -i '' 's/^    MLIRExecutionEngineShared .*/    LLVM/' "$S"/CMakeLists.txt
cmake -S "$S" -B "$S"/build -G Ninja >/dev/null
cmake --build "$S"/build 2>&1 | grep -E '^Undefined|^  "|^ld:'
rm -rf "$S"
L=/opt/homebrew/opt/llvm@20/lib
for lib in libMLIR.dylib libMLIRExecutionEngineShared.dylib; do
  echo "$lib: $(nm -gU $L/$lib | c++filt | grep -c 'mlir::ExecutionEngine::') mlir::ExecutionEngine symbols"
done
grep -A1 -E 'set_target_properties\(MLIRExecutionEngine(Shared)? ' $L/cmake/mlir/MLIRTargets.cmake
grep -n 'MLIR_ENABLE_EXECUTION_ENGINE' $L/cmake/mlir/MLIRConfig.cmake
```

Real output **[26]**:

```text
Undefined symbols for architecture arm64:
  "mlir::ExecutionEngine::invokePacked(llvm::StringRef, llvm::MutableArrayRef<void*>)", referenced from:
  "mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvm::Module*, llvm::TargetMachine*)", referenced from:
  "mlir::ExecutionEngine::create(mlir::Operation*, mlir::ExecutionEngineOptions const&, std::__1::unique_ptr<llvm::TargetMachine, std::__1::default_delete<llvm::TargetMachine>>)", referenced from:
  "mlir::ExecutionEngine::~ExecutionEngine()", referenced from:
ld: symbol(s) not found for architecture arm64
libMLIR.dylib: 0 mlir::ExecutionEngine symbols
libMLIRExecutionEngineShared.dylib: 11 mlir::ExecutionEngine symbols
set_target_properties(MLIRExecutionEngine PROPERTIES
  INTERFACE_LINK_LIBRARIES "LLVM;MLIR"
--
set_target_properties(MLIRExecutionEngineShared PROPERTIES
  INTERFACE_LINK_LIBRARIES "LLVM;MLIR;LLVM"
21:set(MLIR_ENABLE_EXECUTION_ENGINE "1")
```

- **Only `mlir::ExecutionEngine` is missing**, from `dumpLLVMIR` and `runJit`. The translation, `makeOptimizingTransformer`, the two passes and the inliner interface are all in `libMLIR.dylib` already.
- **`nm`** finds those symbols in `libMLIRExecutionEngineShared.dylib`, not in `libMLIR.dylib`, and the package offers that code as two targets: static `MLIRExecutionEngine` and shared `MLIRExecutionEngineShared`. The static one links `MLIR`, i.e. `libMLIR.dylib`, next to its own static archive: the mix behind the TypeID segfault of section 6 (appendix C.2). The shared one is the one to add.
- **`LLVM` can go:** `MLIRExecutionEngineShared` already lists it in its `INTERFACE_LINK_LIBRARIES`, so the Chapter 6 link line is `MLIR` + `MLIRExecutionEngineShared` (appendix C.2).
- **The guard:** `MLIR_ENABLE_EXECUTION_ENGINE` is how the package says it was built with these targets; `CMakeLists.txt` returns before defining the chapter's targets when it is off (appendix C.1).

**A link error names symbols, not libraries; `nm` maps the symbols to a library, and the package's target file shows what each candidate target drags in with it.**

---

## 6. Key Takeaways & Pitfalls

### ⚠️ Pitfall #1 (the big one): static + shared MLIR libs ⇒ TypeID-duplication segfault

Documented in full in **[MLIR_LINKING_PITFALL.md](MLIR_LINKING_PITFALL.md)** — read it before writing your own out-of-tree JIT-using project. Summary:

- **Root cause.** MLIR's TypeID system identifies types/interfaces by *the address of a static variable* in a template instantiation. The Homebrew CMake target `MLIRExecutionEngine` (static) has `INTERFACE_LINK_LIBRARIES "LLVM;MLIR"`, where `MLIR` = `libMLIR.dylib`. If your `target_link_libraries` also lists static `.a` MLIR libraries (`${dialect_libs}`, `${conversion_libs}`, `MLIRIR`, `MLIRPass`, …), the linker pulls in **both** the static archives and the dylib. Two copies of every TypeID static now live in the process, at different addresses ⇒ the "same" type gets two different TypeIDs ⇒ `StorageUniquer` can't find registered types/attributes and dereferences an invalid pointer.
- **Symptoms.** `EXC_BAD_ACCESS` crash inside `mlir::detail::StorageUniquerImpl::getOrCreate`, triggered by **any pass run** (`PassManager::run`) — not just JIT. Deeply misleading, because the crash is in generic MLIR infrastructure, nowhere near your code.
- **Detection.** `otool -L ./build/toyc-ch6` (macOS) / `ldd` (Linux), as in appendix C.3. If you see `libMLIR.dylib` **and** your ninja log shows static `.a` MLIR libraries on the link line, you have the bug.
- **Fix.** Link shared libraries *only*: `MLIR` + `MLIRExecutionEngineShared` (appendix C.2). Alternative: an all-static toolchain built with `LLVM_BUILD_LLVM_DYLIB=OFF` — then there is no dylib to conflict with. What you must never do is mix.
- **Why Ch1–Ch5 didn't crash.** None of them mixes: Ch1 links only the static `MLIRSupport`, and Ch2–Ch5 link only the monolithic dylibs `MLIR` + `LLVM`. None links `MLIRExecutionEngine`, the target that silently drags `libMLIR.dylib` in next to static archives. The trap springs the moment the ExecutionEngine appears — i.e., exactly in Chapter 6 — if you copy a static link line written for an in-tree build.

### ⚠️ Pitfalls #2–#4: smaller traps

**Pitfall #2: leftover `unrealized_conversion_cast` ops.** Dialect conversion can insert `builtin.unrealized_conversion_cast` bridge ops at type seams. If they don't all cancel out and nothing removes them, the "fully lowered" module still contains a non-LLVM op, and the translation/JIT fails on it. This repo adds `createReconcileUnrealizedCastsPass()` right after `createLowerToLLVMPass()` (section 2.8); its `// FIX:` comment records a segfault, but with the current code no test input produces a cast.

**Pitfall #3: forgetting translation registration or target init.** Without `registerLLVMDialectTranslation`/`registerBuiltinDialectTranslation`, `translateModuleToLLVMIR` fails with ``missing `LLVMTranslationDialectInterface` registration`` (run in section 3.1); the calls are needed in **both** `dumpLLVMIR` and `runJit`. Without `InitializeNativeTarget()`/`InitializeNativeTargetAsmPrinter()` there is no TargetMachine, and so no JIT.

**Pitfall #4: traps that compile.** Leaving out a pattern set — the memref→LLVM one, say — fails only at run time (`failed to legalize operation 'memref.alloc'`, section 2.6), and `engine->invoke("main")` looks up `_mlir_ciface_main` and reports `JIT invocation failed` (section 3.2).

### Key takeaways

1. **Transitive lowering** keeps patterns simple: `PrintOpLowering` emits `scf` + `memref` ops and trusts the other patterns in the same `applyFullConversion` to finish the job. Nobody writes affine→LLVM patterns.
2. **Full vs. partial conversion** is the legality contract: `applyFullConversion` guarantees a pure-LLVM-dialect module (modulo `ModuleOp`), which is precisely what the exporter requires.
3. **`LLVMTypeConverter`** is where types cross the boundary — most visibly `memref<2x2xf64>` → the five-field descriptor struct `{ptr, ptr, i64, [2 x i64], [2 x i64]}` (allocated ptr, aligned ptr, offset, sizes, strides).
4. **Conversion ≠ translation.** Dialect conversion rewrites MLIR into the LLVM *dialect*; `translateModuleToLLVMIR` then exports 1:1 into an `llvm::Module`. Two different mechanisms, two different failure modes.
5. **`mlir::ExecutionEngine`** makes "compile and run in-process" a ~30-line function: translation + `makeOptimizingTransformer` + ORC JIT + `invokePacked("main")`.
6. **A lowering is written backwards** (section 5): list the dialects left, reuse a stock `populate*` function for each, write the IR the one custom op should become and check it with `mlir-opt`, then turn each line into a `create<OpTy>(...)` call. The driver and build follow the same way: run each new stage with a stock tool (`mlir-opt`, `mlir-runner`) first, grep the headers for its C++ entry point, and let the linker's undefined symbols pick the library.
7. On macOS with Homebrew LLVM, **link shared MLIR libraries consistently** (`MLIR` + `MLIRExecutionEngineShared`) and verify with `otool -L`.

---

## Appendix A: More runs

### A.1 The working example: `llvm-lowering.mlir`

Section 2.7 shows the file and its `-emit=mlir-llvm` lowering, and section 3.1 its `-emit=llvm` output with and without `-opt`. What remains is to run it, and to run its lit test.

**Running it.** The JIT prints the 3×2 result:

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/llvm-lowering.mlir -emit=jit
```

Real output **[27]**:

```text
1.000000 16.000000 
4.000000 25.000000 
9.000000 36.000000 
```

**The lit test, by hand.** The file's `RUN` line is `toyc-ch6 %s -emit=llvm -opt`, and its `CHECK` lines expect the six squares in order. Pipe the optimized IR into `FileCheck`:

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/llvm-lowering.mlir -emit=llvm -opt 2>&1 \
  | FileCheck ../../test_Example/Toy/Ch6/llvm-lowering.mlir && echo "all checks pass"
```

Real output **[28]**:

```text
all checks pass
```

FileCheck is silent and exits 0 when every `CHECK` line matches, so the command echoes a message only then. The last `CHECK-SAME` expects `3.600000e+01` (6 × 6).

### A.2 More inputs to try

`test_Example/Toy/Ch6/` has additional cases. `codegen.toy` calls `multiply_transpose` twice; the driver inlines it and runs `print(d)`, i.e. `transpose(b) * transpose(a)` for the 2×3 inputs:

```bash
cd /Users/roy/study/mlir/toy/Ch6
./build/toyc-ch6 ../../test_Example/Toy/Ch6/codegen.toy -emit=jit
```

Real output **[29]** — the same 3×2 values as appendix A.1, computed from Toy source instead of hand-written IR:

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

Real output **[30]** (exit status 3):

```text
loc("../../test_Example/Toy/Ch6/invalid.mlir":8:8): error: 'toy.print' op requires zero results
Error can't load file ../../test_Example/Toy/Ch6/invalid.mlir
```

---

## Appendix B: The ecosystem view — replace the backend with stock tools

Once the module is pure LLVM dialect (`-emit=mlir-llvm`), Toy is out of the picture — so from that point on, the *stock* Homebrew tools can finish the job. This pipeline runs the Toy program without `toyc-ch6`'s `dumpLLVMIR`/`runJit` at all:

```bash
cd /Users/roy/study/mlir/toy/Ch6
export P=/opt/homebrew/opt/llvm@20/bin
./build/toyc-ch6 ../../test_Example/Toy/Ch6/jit.toy -emit=mlir-llvm 2>&1 \
  | $P/mlir-translate --mlir-to-llvmir \
  | $P/lli
```

Real output **[31]** — the same matrix as section 4.2:

```text
1.000000 2.000000 
3.000000 4.000000 
```

Each stage is one of the chapter's concepts as a standalone tool:

- **`toyc-ch6 -emit=mlir-llvm`** — the frontend + all dialect *conversions* (section 2). This is the last stage that needs the Toy dialect linked in.
- **`mlir-translate --mlir-to-llvmir`** — the *translation* of section 3.1 (`translateModuleToLLVMIR`) as a stock binary; it works because the LLVM dialect is part of MLIR, so no custom registration is needed. Conversion vs. translation being different mechanisms (takeaway 4) is literally visible as two different tools.
- **`lli`** — LLVM's own JIT, standing in for section 3.2's `runJit`/ExecutionEngine (both sit on ORC underneath); it resolves `printf`/`malloc`/`free` against the host libc the same way. Swap `lli` for `$P/llc -filetype=obj -o toy.o` and you have an object file for a real linker instead — the AOT exit ramp from the same IR.

`toyc-ch6` bundles these stages into one process for convenience (and to avoid textual round-trips); the pipe shows they remain independently replaceable pieces of the LLVM/MLIR ecosystem.

---

## Appendix C: What Chapter 6 adds to the build

This chapter is where out-of-tree builds get genuinely tricky, because `ExecutionEngine` drags LLVM's JIT and native codegen into the link. Relative to Chapter 5, `CMakeLists.txt` adds three things: a feature guard, one source file, and a *changed link line* — the last one is the part that matters. The TableGen wiring (`include/toy/CMakeLists.txt` for `Ops.td` and `ShapeInferenceInterface.td`, plus the `ToyCombine.td` rewriters) is the same as in Chapter 5, with the targets renamed `ToyCh6*`.

### C.1 The chapter targets

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

### C.2 The link line — the part that matters

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

- A link line written for an in-tree build lists dozens of *static* targets: `${dialect_libs}`, `${conversion_libs}`, `${extension_libs}`, `MLIRExecutionEngine`, `MLIRAnalysis`, `MLIRIR`, `MLIRPass`, … That works inside llvm-project's own build tree.
- But against **Homebrew LLVM**, the static `MLIRExecutionEngine` target carries `INTERFACE_LINK_LIBRARIES "LLVM;MLIR"` — and `MLIR` there means **`libMLIR.dylib`**. The linker then pulls in *both* the static `.a` copies of MLIR *and* the dylib → two copies of every TypeID static → runtime segfault (details in section 6).
- The fix: link **shared libraries consistently**. `libMLIR.dylib` already contains *all* dialects, passes and conversions (that's why no `${dialect_libs}` are needed), and `libMLIRExecutionEngineShared.dylib` supplies the JIT. `libMLIR.dylib` itself depends on `libLLVM.dylib`, which bundles every LLVM component (OrcJIT, native codegen), so no `LLVM_LINK_COMPONENTS`/`llvm_map_components_to_libnames` boilerplate is needed either.

### C.3 Verifying the link

```bash
cd /Users/roy/study/mlir/toy/Ch6
otool -L ./build/toyc-ch6
```

Real output **[32]**:

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

## Appendix D: Running this README's commands: `examples.sh`

[`examples.sh`](examples.sh) — a symlink to the shared [`../examples.sh`](../examples.sh) — runs the commands of this README and shows each one with its output and exit status:

```bash
cd /Users/roy/study/mlir/toy/Ch6
./examples.sh -l
```

Real output **[33]** (abridged):

```text
[ 1] 2.1   ./build/toyc-ch6 ../../test_Example/Toy/Ch6/jit.toy -emit=mlir-affine 2>&1 | grep toy.print
[ 2] 2.6   S=$(mktemp -d)
...
[32] C.3   otool -L ./build/toyc-ch6
[33] D     ./examples.sh -l
```

The script reads the commands from `README.md` on every run. A command is every `bash` code block in file order, except the build block (its first command after `cd` is `cmake`) and file excerpts (a `***path***` label above the fence). Number *N* is the *N*-th such block, and the README marks that block's output "Real output **[N]**". So adding, removing or reordering a command block renumbers the ones after it, and their **[N]** labels must be updated to match.

---

## Links

- Official tutorial: [Toy Ch-6 — Lowering to LLVM and CodeGeneration](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/)
- Related MLIR docs: [LLVM IR Target](https://mlir.llvm.org/docs/TargetLLVMIR/) · [LLVM dialect](https://mlir.llvm.org/docs/Dialects/LLVM/) · [Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)
- Linking post-mortem: [MLIR_LINKING_PITFALL.md](MLIR_LINKING_PITFALL.md)
- Previous: [Chapter 5 — Partial Lowering to Affine](../Ch5/README.md)
- Next: [Chapter 7 — Struct Types](../Ch7/README.md)
- Back to [README](../README.md)
