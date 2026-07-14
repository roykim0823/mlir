# Chapter 6: Lowering to LLVM and JIT Compilation

> **Goal:** Complete the lowering journey — take the mixed `toy`/`affine`/`arith`/`memref` IR from Chapter 5 all the way down to the LLVM dialect, translate it to real LLVM IR, and execute it with a JIT — following the official tutorial [Toy Ch-6: Lowering to LLVM and CodeGeneration](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/).

**Chapter code:** [`Ch6/`](Ch6/) — an out-of-tree CMake project (NOT built inside llvm-project), built as part of the repo-wide **superbuild** at [`toy/CMakeLists.txt`](CMakeLists.txt) against **Homebrew LLVM/MLIR 20** on macOS (configured via [`CMakePresets.json`](CMakePresets.json)):

- `MLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir`
- Compiler: `/opt/homebrew/opt/llvm@20/bin/clang++`
- Generator: Ninja

```bash
cd toy && ./build.sh ch6     # produces ./build/bin/toyc-ch6 (incremental; --fresh to wipe)
./run.sh ch6                 # runs all five -emit modes on a small example
```

> **Read this first if your Ch6 binary segfaults:** this repo hit (and documents) a real macOS
> linking pitfall where mixing static `.a` MLIR libraries with `libMLIR.dylib` causes a
> **TypeID-duplication segfault** inside `StorageUniquer` on *any* pass run. See
> [Section 7](#7-key-takeaways--pitfalls) and the full write-up in
> [MLIR_LINKING_PITFALL.md](Ch6/MLIR_LINKING_PITFALL.md).

---

## 1. Overview — the full lowering story

In Chapter 5 we *partially* lowered Toy: compute-heavy ops (`toy.transpose`, `toy.mul`, `toy.constant`, …) became `affine` loops over `memref`s, but `toy.print` was deliberately left as-is because we wanted to keep printing at a high level of abstraction as long as possible. Chapter 6 finishes the job. The full pipeline now looks like this:

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

Files touched in this chapter:

| File | Role |
|---|---|
| [`Ch6/mlir/LowerToLLVM.cpp`](Ch6/mlir/LowerToLLVM.cpp) | `PrintOpLowering` + `ToyToLLVMLoweringPass` (full conversion to the LLVM dialect) |
| [`Ch6/toyc.cpp`](Ch6/toyc.cpp) | New actions `-emit=mlir-llvm`, `-emit=llvm`, `-emit=jit`; `dumpLLVMIR()` and `runJit()` |
| [`Ch6/CMakeLists.txt`](Ch6/CMakeLists.txt) | Links the ExecutionEngine — the interesting (and dangerous) part on macOS |
| [`Ch6/MLIR_LINKING_PITFALL.md`](Ch6/MLIR_LINKING_PITFALL.md) | Post-mortem of the static+shared TypeID segfault |

---

## 2. Lowering `toy.print` — `PrintOpLowering` in depth

`toy.print` has no direct LLVM equivalent, so we lower it to what a C programmer would write: a loop nest calling `printf("%f ", elt)` for every element, with a newline after each row. All code below is the real code from [`Ch6/mlir/LowerToLLVM.cpp`](Ch6/mlir/LowerToLLVM.cpp).

The file header sums up the plan:

```cpp
//                         Affine --
//                                  |
//                                  v
//                       Arithmetic + Func --> LLVM (Dialect)
//                                  ^
//                                  |
//     'toy.print' --> Loop (SCF) --
```

Note the key trick: `PrintOpLowering` does **not** emit LLVM-dialect loops directly. It emits `scf.for` loops and `memref.load`s — *higher-level* dialects — and lets the other conversion patterns in the very same `applyFullConversion` run lower those to `cf` branches and LLVM GEPs. That is transitive lowering in action.

### 2.1 The pattern skeleton

```cpp
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

Two things to notice:

- Because Ch5 already ran, the operand of `toy.print` at this point is a **`memref`** (e.g. `memref<2x2xf64>`), not a tensor — the pattern reads its static shape to build the loop bounds.
- It is a generic `ConversionPattern` matched by operation name (`toy.print`), with benefit 1.

### 2.2 Format strings as `llvm.mlir.global`

C string literals become module-level LLVM globals. `getOrCreateGlobalString` creates (once, memoized by symbol lookup) an internal constant global holding the raw bytes, then computes a pointer to its first character:

```cpp
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
```

Step by step:

1. `module.lookupSymbol<LLVM::GlobalOp>(name)` — reuse if a global with that symbol already exists (so ten `toy.print`s still produce one `@frmt_spec`).
2. Otherwise, an `InsertionGuard` temporarily moves the builder to the **start of the module** and creates `llvm.mlir.global internal constant @frmt_spec("%f \00")` of type `!llvm.array<4 x i8>`. Note the explicit `\0` terminators in the `StringRef("%f \0", 4)` literals — LLVM globals don't NUL-terminate for you.
3. `llvm.mlir.addressof` yields the address of the global, and an `llvm.getelementptr` with indices `[0, 0]` produces the `!llvm.ptr` to the first `i8` — exactly the `getelementptr inbounds [4 x i8], ptr @frmt_spec, i64 0, i64 0` idiom from C compilers.

> **LLVM 20 note (opaque pointers):** the official tutorial text still shows typed `i8*` pointers (`LLVM::LLVMPointerType::get(IntegerType::get(context, 8))`). Modern MLIR/LLVM uses *opaque* pointers: `LLVM::LLVMPointerType::get(context)` — no pointee type — which is why the GEP must carry the element type (`global.getType()`) as a separate argument. This repo's code is the current-API version.

### 2.3 Declaring `printf` — `getOrInsertPrintf`

`printf` is variadic: `i32 (ptr, ...)`. The helper builds that function type and inserts a body-less `llvm.func` declaration at the top of the module (again, only if it isn't already there):

```cpp
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
```

The return value is a `FlatSymbolRefAttr` — calls reference the function *by symbol*, not by SSA value, matching how LLVM IR call instructions name their callees.

### 2.4 Generating the loop nest

Back in `matchAndRewrite`, one `scf.for` is created per memref dimension:

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

- **Bounds:** every dimension gets `arith.constant 0` / `arith.constant <dim>` / step `1` — legal because those constants get folded away or lowered by the arith patterns in the same conversion.
- **Body cleanup:** `scf::ForOp` auto-creates a body with a terminator; the inner `eraseOp` loop clears it so we control the body exactly.
- **Newline placement:** for every dimension *except the innermost*, a `printf(nl)` call is placed at the **end** of that loop's body — i.e. after the entire inner loop has finished a row, print `"\n"`. For our 2×2 matrix: the outer (row) loop ends each iteration with a newline; the inner (column) loop only prints elements.
- **Insertion point dance:** after adding the terminator, `setInsertionPointToStart(loop.getBody())` moves *inside* the just-created loop, so the next iteration builds the inner loop nested inside it. When the C++ loop ends, the insertion point sits inside the innermost body.
- **Element print:** `memref.load %alloc[%i, %j]` reads the current element, and `llvm.call @printf(%frmt_spec_ptr, %elt)` prints it. `loopIvs` collected one induction variable per dimension, giving the full index vector.
- Finally `rewriter.eraseOp(op)` — `toy.print` has no results, so the pattern simply removes it after materializing the replacement code.

The emitted `scf.for` + `memref.load` ops are *illegal* for our conversion target — and that is fine: the SCF→CF, CF→LLVM, and MemRef→LLVM patterns registered alongside this pattern will lower them before the driver declares victory.

---

## 3. Full Conversion to the LLVM Dialect

The pass at the bottom of [`Ch6/mlir/LowerToLLVM.cpp`](Ch6/mlir/LowerToLLVM.cpp) assembles target + type converter + patterns and runs a **full** conversion.

### 3.1 The conversion target — `LLVMConversionTarget`

```cpp
void ToyToLLVMLoweringPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering. For this lowering, we are only targeting
  // the LLVM dialect.
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();
```

`LLVMConversionTarget` is a convenience subclass of `ConversionTarget` that already marks the LLVM dialect legal (equivalent to `target.addLegalDialect<LLVM::LLVMDialect>()` plus a few LLVM-specific details). The only extra legal op is `ModuleOp` — the top-level container must survive. *Everything else* — `toy.*`, `affine.*`, `scf.*`, `cf.*`, `arith.*`, `memref.*`, `func.*` — is illegal and must be converted.

The pass also declares the dialects it may *create* during lowering, so the context loads them even if the input IR doesn't mention them:

```cpp
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
  }
```

### 3.2 The type converter — `LLVMTypeConverter` and the memref descriptor

```cpp
  LLVMTypeConverter typeConverter(&getContext());
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

A ranked memref lowers to exactly this five-field struct: two pointers (the raw `malloc` result and the aligned data pointer), a linear offset, and per-dimension size and stride arrays. Every `memref.load %m[%i, %j]` becomes `extractvalue` (get the aligned pointer) + address arithmetic (`%i * stride0 + %j * stride1 + offset`) + `getelementptr` + `load` — you will see precisely this in the `-emit=mlir-llvm` output in Section 6. The type converter is also what rewrites function signatures and block arguments that carry memref/index types, which matters because our lowering flows values through `cf` block arguments (loop-carried induction variables).

Since Toy's own tensor types were already eliminated in Chapter 5, the stock converter needs no customization.

### 3.3 The patterns — one `populate*` per lowering edge

```cpp
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

This is the tutorial's "transitive lowering" payoff: nobody wrote an `affine.for` → `llvm.br` pattern. The driver applies affine→scf, then scf→cf, then cf→llvm patterns *recursively on the results of each other* within a single conversion.

### 3.4 `applyFullConversion`

```cpp
  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
```

Unlike Chapter 5's `applyPartialConversion`, `applyFullConversion` fails if *any* illegal op survives. If you forget one `populate…` call, you get a precise diagnostic naming the un-lowered op — much better than silently emitting broken IR.

### 3.5 Pipeline registration in `toyc.cpp` — plus a repo-specific fix

The driver ([`Ch6/toyc.cpp`](Ch6/toyc.cpp), in `loadAndProcessMLIR`) appends the new stage when `-emit` is `mlir-llvm` or beyond:

```cpp
  if (isLoweringToLLVM) {
    // Finish lowering the toy IR to the LLVM dialect.
    pm.addPass(mlir::toy::createLowerToLLVMPass());

    // FIX: Segmentation fault when this pass is not added.
    // When lowering from Toy Dialect to the LLVM Dialect, MLIR often
    // creates "bridge" operations called unrealized_conversion_cast.
    // These casts are just placeholders. If they aren't removed before sent to JIT,
    // the JIT encounters an operation it doesn't recognize as valid LLVM IR and crashes (segfault).
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    // This is necessary to have line tables emitted and basic
    // debugger working.
    pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
  }
```

Two additions relative to a bare `createLowerToLLVMPass()`:

- **`createReconcileUnrealizedCastsPass()`** — a real fix discovered in this repo. Dialect conversion inserts `builtin.unrealized_conversion_cast` "bridge" ops at type-boundary seams (e.g. `index` ↔ `i64`). They are placeholders that must cancel out in pairs; this pass erases them. Without it, `translateModuleToLLVMIR`/the JIT meets a non-LLVM op in an "all-LLVM" module and **segfaults**. (Requires `#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"`.)
- **`createDIScopeForLLVMFuncOpPass()`** — attaches debug-info scopes so the exported LLVM IR carries line tables (you'll see `!dbg` metadata in the `-emit=llvm` output below).

The chapter also registers two capabilities in `main()` that Ch5 didn't need:

```cpp
  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);          // func dialect extensions
  mlir::LLVM::registerInlinerInterface(registry);       // let the inliner reason about llvm ops
  mlir::MLIRContext context(registry);
```

---

## 4. Emitting LLVM IR and Running the JIT

Once the module contains only LLVM-dialect ops, two new code paths in [`Ch6/toyc.cpp`](Ch6/toyc.cpp) take over.

### 4.1 `dumpLLVMIR` — MLIR → `llvm::Module`

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
```

Piece by piece:

- **Translation registration.** MLIR-to-LLVM-IR export is pluggable via *dialect translation interfaces*. `registerLLVMDialectTranslation` teaches the exporter how each `llvm.*` op maps to an LLVM instruction; `registerBuiltinDialectTranslation` handles builtin bits (the module op, locations → debug info). Forgetting these makes `translateModuleToLLVMIR` fail with "cannot be emitted" errors — a classic out-of-tree stumbling block, since `mlir-opt`-style tools register them for you.
- **`translateModuleToLLVMIR`** walks the MLIR module and builds a genuine `llvm::Module` in a fresh `llvm::LLVMContext`. This is a *translation* (1:1 export), not a pattern-based conversion — which is why the module had to be 100 % LLVM dialect first.

Then the target is configured and an optional LLVM optimization pipeline runs:

```cpp
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Configure the LLVM Module
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  ...
  auto tmOrError = tmBuilderOrError->createTargetMachine();
  ...
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

- `InitializeNativeTarget()` / `InitializeNativeTargetAsmPrinter()` link in the host backend (AArch64 here) so a `TargetMachine` can exist.
- `JITTargetMachineBuilder::detectHost()` + `setupTargetTripleAndDataLayout` stamp the module with the host triple and data layout — you can see `target triple = "arm64-apple-darwin25.5.0"` in the output.
- `makeOptimizingTransformer(optLevel, sizeLevel, targetMachine)` returns a function-object wrapping LLVM's standard `-O<N>` pass pipeline. With `toyc-ch6 -emit=llvm -opt` you get `-O3`: for our constant example LLVM constant-folds the whole matrix and `main` collapses into four straight-line `printf` calls (the official docs show exactly this effect).

### 4.2 `runJit` — executing `main` in-process

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

- **`mlir::ExecutionEngine`** wraps LLVM's ORC JIT. `create(module, options)` internally re-runs the same translation as `dumpLLVMIR`, applies the `transformer` (our optimization pipeline) to the resulting `llvm::Module`, and eagerly JIT-compiles it to native code in the current process.
- **`invokePacked("main")`** looks up the JIT'd symbol and calls it through a "packed-arguments" interface (an array of `void*` — empty here, since `main` takes no arguments and returns nothing). The official tutorial shows the sugar variant `engine->invoke("main")`; `invokePacked` is the lower-level form it desugars to. Because the format-string globals and `printf` declaration are in the module, the JIT resolves `printf`/`malloc`/`free` against the host process, and the matrix prints directly to our terminal.
- Note the same **translation registration** and **native-target initialization** are required here as in `dumpLLVMIR` — the JIT path performs its own MLIR→LLVM-IR export.

---

## 5. Building — CMakeLists walkthrough (and dodging the linking trap)

This chapter is where out-of-tree builds get genuinely tricky, because `ExecutionEngine` drags LLVM's JIT and native codegen into the link.

The repo uses a **superbuild**: the top-level [`toy/CMakeLists.txt`](CMakeLists.txt) does the `find_package(MLIR/LLVM)` + `include(TableGen/AddLLVM/AddMLIR/HandleLLVMOptions)` boilerplate exactly once, sets `CMAKE_RUNTIME_OUTPUT_DIRECTORY` to `build/bin/` (so every chapter binary lands in one place), and then `add_subdirectory(Ch1)` … `add_subdirectory(Ch7)`. Compilers and `MLIR_DIR`/`LLVM_DIR` come from [`CMakePresets.json`](CMakePresets.json) (Ninja, Release, Homebrew llvm@20).

[`Ch6/CMakeLists.txt`](Ch6/CMakeLists.txt) is therefore **dual-mode**: the boilerplate is wrapped in a guard so it only runs when the chapter is configured *standalone*, and is skipped in the superbuild:

```cmake
# Runs only when this chapter is configured directly (cmake -S Ch6).
# In the superbuild (cmake -S toy/), ../CMakeLists.txt already did all this.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(toy-ch6)

  find_package(MLIR REQUIRED CONFIG)   # MLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir
  find_package(LLVM REQUIRED CONFIG)

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
endif()

# This chapter depends on JIT support enabled.
if(NOT MLIR_ENABLE_EXECUTION_ENGINE)
  return()
endif()

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
```

Same shape as previous chapters (find packages, include MLIR's CMake macros, tablegen the ODS/DRR files), plus a guard: the Homebrew MLIR must have been built with `MLIR_ENABLE_EXECUTION_ENGINE=ON` (it is).

### 5.1 The link line — the part that matters

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

That's it. **Two shared libraries, zero static archives.** This is deliberate (the `NOTE` comment now guards it in the file itself), and it is how this repo avoids the segfault documented in [MLIR_LINKING_PITFALL.md](Ch6/MLIR_LINKING_PITFALL.md):

- The upstream Toy Ch6 CMakeLists links dozens of *static* targets: `${dialect_libs}`, `${conversion_libs}`, `${extension_libs}`, `MLIRExecutionEngine`, `MLIRAnalysis`, `MLIRIR`, `MLIRPass`, … That works inside llvm-project's own build tree.
- But against **Homebrew LLVM**, the static `MLIRExecutionEngine` target carries `INTERFACE_LINK_LIBRARIES "LLVM;MLIR"` — and `MLIR` there means **`libMLIR.dylib`**. The linker then pulls in *both* the static `.a` copies of MLIR *and* the dylib → two copies of every TypeID static → runtime segfault (details in Section 7).
- The fix: link **shared libraries consistently**. `libMLIR.dylib` already contains *all* dialects, passes and conversions (that's why no `${dialect_libs}` are needed), and `libMLIRExecutionEngineShared.dylib` supplies the JIT. `libMLIR.dylib` itself depends on `libLLVM.dylib`, which bundles every LLVM component (OrcJIT, native codegen), so no `LLVM_LINK_COMPONENTS`/`llvm_map_components_to_libnames` boilerplate is needed either.

Verify the result:

```bash
$ otool -L build/bin/toyc-ch6
build/bin/toyc-ch6:
	/opt/homebrew/opt/llvm@20/lib/libMLIRExecutionEngineShared.dylib
	/opt/homebrew/opt/llvm@20/lib/libMLIR.dylib
	/opt/homebrew/opt/llvm@20/lib/libLLVM.dylib
	/usr/lib/libc++.1.dylib
	/usr/lib/libSystem.B.dylib
```

Exactly one copy of MLIR and one of LLVM in the process — consistent TypeIDs.

### 5.2 build.sh — the superbuild workflow

One [`build.sh`](build.sh) at `toy/` drives all chapters through a single shared, **incremental** build tree:

```bash
cd toy
./build.sh ch6          # configure once (if needed) + build only toyc-ch6
./build.sh              # build everything (ch1..ch7)
./build.sh ch6 --fresh  # wipe build/ first, then rebuild
```

The script configures with `cmake --preset default` only when `build/CMakeCache.txt` doesn't exist yet (afterwards Ninja re-runs CMake automatically when a CMakeLists.txt changes), then runs `cmake --build --preset default [--target toyc-ch6]`. Binaries land in `build/bin/toyc-ch{1..7}`.

The `default` preset in [`CMakePresets.json`](CMakePresets.json) resolves `MLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir` and `CMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++`. Building with Homebrew's own clang++ matters: MLIR headers must be compiled with a compiler/stdlib ABI-compatible with the prebuilt dylibs.

Thanks to the dual-mode guard from Section 5, the chapter can still be configured **standalone**, without the superbuild:

```bash
cd toy
cmake -S Ch6 -B Ch6/build -G Ninja \
  -DMLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++
cmake --build Ch6/build      # produces Ch6/build/toyc-ch6
```

(`run.sh` looks for binaries in `build/bin/` first and falls back to `ChN/build/`, so both layouts work.)

---

## 6. Running and Testing

`cd toy && ./run.sh ch6` pipes one program through all five emit levels — [`run.sh`](run.sh) prints a `== -emit=<mode> ==` header before each mode, then runs the equivalent of:

```bash
echo 'def main() { print([[1, 2], [3, 4]]); }' | ./build/bin/toyc-ch6 -emit=mlir
echo 'def main() { print([[1, 2], [3, 4]]); }' | ./build/bin/toyc-ch6 -emit=llvm
echo 'def main() { print([[1, 2], [3, 4]]); }' | ./build/bin/toyc-ch6 -emit=mlir-affine
echo 'def main() { print([[1, 2], [3, 4]]); }' | ./build/bin/toyc-ch6 -emit=mlir-llvm
echo 'def main() { print([[1, 2], [3, 4]]); }' | ./build/bin/toyc-ch6 -emit=jit
```

All outputs below are **real captures** from this repo (Apple Silicon, LLVM 20), shown without the `== -emit=<mode> ==` headers.

### 6.1 `-emit=mlir` — pure Toy dialect

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

### 6.2 `-emit=mlir-affine` — after Chapter 5's partial lowering

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

Tensors became a heap `memref<2x2xf64>` with element-wise `affine.store`s (the constant is small, so no loops were needed). Crucially, **`toy.print` survived** — but its operand type changed to `memref`, which is exactly what `PrintOpLowering` expects.

### 6.3 `-emit=mlir-llvm` — after `ToyToLLVMLoweringPass` (still MLIR!)

Abbreviated; the full dump is ~90 lines:

```mlir
module {
  llvm.func @free(!llvm.ptr)
  llvm.mlir.global internal constant @nl("\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @frmt_spec("%f \00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @main() {
    ...
    %11 = llvm.call @malloc(%10) : (i64) -> !llvm.ptr
    %12 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.insertvalue %11, %12[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.insertvalue %11, %13[1] : ...
    %16 = llvm.insertvalue %15, %14[2] : ...   // offset = 0
    %17 = llvm.insertvalue %4, %16[3, 0] : ... // size[0] = 2
    %18 = llvm.insertvalue %5, %17[3, 1] : ... // size[1] = 2
    %19 = llvm.insertvalue %5, %18[4, 0] : ... // stride[0] = 2
    %20 = llvm.insertvalue %6, %19[4, 1] : ... // stride[1] = 1
    ...
    // four stores of 1.0 / 2.0 / 3.0 / 4.0 via extractvalue + getelementptr ...
    %49 = llvm.mlir.addressof @frmt_spec : !llvm.ptr
    %51 = llvm.getelementptr %49[%50, %50] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<4 x i8>
    %52 = llvm.mlir.addressof @nl : !llvm.ptr
    ...
    llvm.br ^bb1(%55 : i64)
  ^bb1(%58: i64):  // 2 preds: ^bb0, ^bb5      // outer (row) loop header
    %59 = llvm.icmp "slt" %58, %56 : i64
    llvm.cond_br %59, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%60 : i64)
  ^bb3(%63: i64):  // 2 preds: ^bb2, ^bb4      // inner (column) loop header
    %64 = llvm.icmp "slt" %63, %61 : i64
    llvm.cond_br %64, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3                          // body: load element, printf("%f ", elt)
    %65 = llvm.extractvalue %20[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %67 = llvm.mul %58, %66 : i64
    %68 = llvm.add %67, %63 : i64
    %69 = llvm.getelementptr %65[%68] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %70 = llvm.load %69 : !llvm.ptr -> f64
    %71 = llvm.call @printf(%51, %70) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    llvm.br ^bb3(%72 : i64)
  ^bb5:  // pred: ^bb3                          // after inner loop: printf("\n")
    %73 = llvm.call @printf(%54) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    llvm.br ^bb1(%74 : i64)
  ^bb6:  // pred: ^bb1
    %75 = llvm.extractvalue %20[0] : ...        // allocated pointer, field 0
    llvm.call @free(%75) : (!llvm.ptr) -> ()
    llvm.return
  }
}
```

Everything from Sections 2–3 is visible here:

- the two **`llvm.mlir.global`** format strings (`@frmt_spec("%f \00")`, `@nl("\0A\00")`) and the **`llvm.func @printf(!llvm.ptr, ...)`** declaration hoisted to module scope;
- `memref.alloc` became **`llvm.call @malloc`** plus eight `llvm.insertvalue`s building the five-field **memref descriptor struct**;
- the `scf.for` nest became a **CFG of blocks with block arguments** (`^bb1(%58: i64)` — MLIR's SSA replacement for PHI nodes);
- element access = `extractvalue [1]` (aligned ptr) + `mul/add` (row-major index `i*2+j`) + `getelementptr` + `load`;
- the newline `printf` sits in `^bb5`, i.e. after each inner-loop run — one newline per row;
- `memref.dealloc` became **`llvm.call @free`** on `extractvalue [0]` (the *allocated* pointer, not the aligned one).

### 6.4 `-emit=llvm` — genuine LLVM IR

Abbreviated real output (unoptimized, `-opt` not passed):

```llvm
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-darwin25.5.0"

@nl = internal constant [2 x i8] c"\0A\00"
@frmt_spec = internal constant [4 x i8] c"%f \00"

declare !dbg !3 void @free(ptr)
declare !dbg !6 i32 @printf(ptr, ...)
declare !dbg !7 ptr @malloc(i64)

define void @main() !dbg !8 {
  %1 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 4) to i64)), !dbg !10
  %2 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1, 0, !dbg !10
  ...
  store double 1.000000e+00, ptr %10, align 8, !dbg !10
  store double 2.000000e+00, ptr %12, align 8, !dbg !10
  store double 3.000000e+00, ptr %14, align 8, !dbg !10
  store double 4.000000e+00, ptr %16, align 8, !dbg !10
  br label %17, !dbg !11

17:                                               ; preds = %32, %0
  %18 = phi i64 [ 0, %0 ], [ %34, %32 ], !dbg !11
  %19 = icmp slt i64 %18, 2, !dbg !11
  br i1 %19, label %20, label %35, !dbg !11
  ...
24:                                               ; preds = %21
  ...
  %29 = load double, ptr %28, align 8, !dbg !11
  %30 = call i32 (ptr, ...) @printf(ptr @frmt_spec, double %29), !dbg !11
  ...
32:                                               ; preds = %21
  %33 = call i32 (ptr, ...) @printf(ptr @nl), !dbg !11
  ...
35:                                               ; preds = %17
  %36 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 0, !dbg !10
  call void @free(ptr %36), !dbg !10
  ret void, !dbg !12
}

!1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2, producer: "MLIR", ...)
...
```

What changed vs. `-emit=mlir-llvm`: same structure, different *representation*. The host **target triple and data layout** are stamped in; block arguments became **`phi` nodes** (`%18 = phi i64 [ 0, %0 ], [ %34, %32 ]`); `malloc`'s size argument became the classic constant-folded `ptrtoint(gep(null, 4))` sizeof idiom; and the `DIScopeForLLVMFuncOp` pass shows up as `!dbg` line-table metadata. Passing `-opt` runs the `-O3` transformer and (as in the official docs) collapses all of this into four straight-line `printf` calls with immediate `double` constants.

### 6.5 `-emit=jit` — actually running it

```
$ echo 'def main() { print([[1, 2], [3, 4]]); }' | ./build/bin/toyc-ch6 -emit=jit
1.000000 2.000000 
3.000000 4.000000
```

The ExecutionEngine JIT-compiled `main` to AArch64 code, `invokePacked("main")` called it, the JIT resolved `printf`/`malloc`/`free` against the host libc — and the Toy program printed its matrix. End-to-end: source text to executed native code in one process, no files written.

For debugging, `--mlir-print-ir-after-all` prints the IR after every pass in the pipeline, which is invaluable for watching each lowering stage in sequence.

---

## 7. Key Takeaways & Pitfalls

### ⚠️ Pitfall #1 (the big one): static + shared MLIR libs ⇒ TypeID-duplication segfault

Documented in full in **[MLIR_LINKING_PITFALL.md](Ch6/MLIR_LINKING_PITFALL.md)** — read it before writing your own out-of-tree JIT-using project. Summary:

- **Root cause.** MLIR's TypeID system identifies types/interfaces by *the address of a static variable* in a template instantiation. The Homebrew CMake target `MLIRExecutionEngine` (static) has `INTERFACE_LINK_LIBRARIES "LLVM;MLIR"`, where `MLIR` = `libMLIR.dylib`. If your `target_link_libraries` also lists static `.a` MLIR libraries (`${dialect_libs}`, `${conversion_libs}`, `MLIRIR`, `MLIRPass`, …), the linker pulls in **both** the static archives and the dylib. Two copies of every TypeID static now live in the process, at different addresses ⇒ the "same" type gets two different TypeIDs ⇒ `StorageUniquer` can't find registered types/attributes and dereferences an invalid pointer.
- **Symptoms.** `EXC_BAD_ACCESS` crash inside `mlir::detail::StorageUniquerImpl::getOrCreate`, triggered by **any pass run** (`PassManager::run`) — not just JIT. Deeply misleading, because the crash is in generic MLIR infrastructure, nowhere near your code.
- **Detection.** `otool -L ./build/bin/toyc-ch6` (macOS) / `ldd` (Linux). If you see `libMLIR.dylib` **and** your ninja log shows static `.a` MLIR libraries on the link line, you have the bug.
- **Fix.** Link shared libraries *only*: `MLIR` + `MLIRExecutionEngineShared` (see [Section 5.1](#51-the-link-line--the-part-that-matters)). Alternative: an all-static toolchain built with `LLVM_BUILD_LLVM_DYLIB=OFF` — then there is no dylib to conflict with. What you must never do is mix.
- **Why Ch1–Ch5 didn't crash.** They never link `MLIRExecutionEngine`, so `libMLIR.dylib` never enters the dependency graph; all-static linking is internally consistent. The trap springs the moment the ExecutionEngine appears — i.e., exactly in Chapter 6.

### ⚠️ Pitfall #2: leftover `unrealized_conversion_cast` ops segfault the JIT

Dialect conversion inserts `builtin.unrealized_conversion_cast` bridge ops at type seams. If they don't all cancel out and you skip `createReconcileUnrealizedCastsPass()`, the "fully lowered" module still contains a non-LLVM op, and the translation/JIT crashes. This repo adds the pass right after `createLowerToLLVMPass()` in `toyc.cpp` (with a `// FIX:` comment marking the war story).

### ⚠️ Pitfall #3: forgetting translation registration or target init

- No `registerLLVMDialectTranslation`/`registerBuiltinDialectTranslation` ⇒ `translateModuleToLLVMIR` fails. Needed in **both** `dumpLLVMIR` and `runJit`.
- No `InitializeNativeTarget()`/`InitializeNativeTargetAsmPrinter()` ⇒ no TargetMachine, no JIT.

### Key takeaways

1. **Transitive lowering** keeps patterns simple: `PrintOpLowering` emits `scf` + `memref` ops and trusts the other patterns in the same `applyFullConversion` to finish the job. Nobody writes affine→LLVM patterns.
2. **Full vs. partial conversion** is the legality contract: `applyFullConversion` guarantees a pure-LLVM-dialect module (modulo `ModuleOp`), which is precisely what the exporter requires.
3. **`LLVMTypeConverter`** is where types cross the boundary — most visibly `memref<2x2xf64>` → the five-field descriptor struct `{ptr, ptr, i64, [2 x i64], [2 x i64]}` (allocated ptr, aligned ptr, offset, sizes, strides).
4. **Conversion ≠ translation.** Dialect conversion rewrites MLIR into the LLVM *dialect*; `translateModuleToLLVMIR` then exports 1:1 into an `llvm::Module`. Two different mechanisms, two different failure modes.
5. **`mlir::ExecutionEngine`** makes "compile and run in-process" a ~20-line function: translation + `makeOptimizingTransformer` + ORC JIT + `invokePacked("main")`.
6. On macOS with Homebrew LLVM, **link shared MLIR libraries consistently** (`MLIR` + `MLIRExecutionEngineShared`) and verify with `otool -L`.

---

## Links

- Official tutorial: [Toy Ch-6 — Lowering to LLVM and CodeGeneration](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/)
- Linking post-mortem: [MLIR_LINKING_PITFALL.md](Ch6/MLIR_LINKING_PITFALL.md)
- Previous: [Chapter 5 — Partial Lowering to Affine](5_partial_lowering.md)
- Next: [Chapter 7 — Struct Types](7_struct_types.md)
- Back to [README](README.md)
