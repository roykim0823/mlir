# MLIR Linking Pitfall: Mixed Static (.a) and Shared (.dylib) Libraries

## Problem

When building out-of-tree MLIR projects that use `MLIRExecutionEngine` (for JIT),
the CMake target `MLIRExecutionEngine` (static) has:

```
INTERFACE_LINK_LIBRARIES "LLVM;MLIR"
```

The `MLIR` target here is `libMLIR.dylib`. If your `target_link_libraries` also
includes static `.a` libraries (via `${dialect_libs}`, `${conversion_libs}`, or
explicit targets like `MLIRAnalysis`, `MLIRIR`, `MLIRPass`, etc.), **the linker
pulls in both the static archives and the shared library**.

This causes a **segfault** at runtime.

## Root Cause: TypeID Duplication

MLIR's TypeID system identifies types/interfaces by the address of a static
variable inside a template instantiation. When the same MLIR code exists in both:

1. Static `.a` archives (linked into the executable)
2. `libMLIR.dylib` (loaded as a shared library)

...two copies of each TypeID static variable exist in the process. The executable's
copy and the dylib's copy have different addresses, so they produce different
TypeIDs for the same logical type. The `StorageUniquer` then fails to find
registered types/attributes, dereferencing an invalid pointer.

## Symptoms

- **Crash site:** `mlir::detail::StorageUniquerImpl::getOrCreate` with
  `EXC_BAD_ACCESS` (null/invalid pointer dereference)
- **Trigger:** Any MLIR pass execution (`PassManager::run`), not just JIT
- **Misleading:** The crash happens in generic MLIR infrastructure, not in
  user code, making it hard to trace back to a linking issue

## How to Detect

Check your binary's dynamic library dependencies:

```bash
otool -L ./build/toyc-ch6   # macOS
ldd ./build/toyc-ch6        # Linux
```

If you see `libMLIR.dylib` (or `libMLIR.so`) **and** your ninja/make build log
shows static `.a` MLIR libraries on the link line, you have the problem.

## The Fix: Use Shared Libraries Consistently

For Homebrew LLVM (or any installation that ships `libMLIR.dylib`):

```cmake
# BEFORE (broken - mixes static and shared)
target_link_libraries(toyc-ch6
  PRIVATE
    ${dialect_libs}              # static .a files
    ${conversion_libs}           # static .a files
    ${extension_libs}            # static .a files
    MLIRExecutionEngine          # static .a, but pulls in libMLIR.dylib
    MLIRAnalysis
    MLIRIR
    MLIRPass
    # ... many more static targets ...
    )

# AFTER (correct - shared libraries only)
target_link_libraries(toyc-ch6
  PRIVATE
    MLIR                         # libMLIR.dylib (all dialects, passes, conversions)
    MLIRExecutionEngineShared    # libMLIRExecutionEngineShared.dylib (JIT support)
    )
```

Key points:

- `MLIR` target = `libMLIR.dylib`, contains all MLIR dialects, passes, and conversions
- `MLIRExecutionEngineShared` target = `libMLIRExecutionEngineShared.dylib`,
  provides JIT/ExecutionEngine support
- `libMLIR.dylib` depends on `libLLVM.dylib`, which includes all LLVM components
  (OrcJIT, native codegen, etc.), so no need to list `LLVM_LINK_COMPONENTS`

## Why Chapters Before Ch6 Don't Crash

None of the earlier chapters mixes static MLIR archives with `libMLIR.dylib`,
so every TypeID static exists exactly once:

- **Ch1** links only `MLIRSupport`, which Homebrew ships as the static
  `libMLIRSupport.a`. It uses no MLIR IR, so there are no dialect/interface TypeIDs
  to duplicate, and `libMLIR.dylib` is never loaded (`otool -L` shows only
  `libLLVM.dylib`).
- **Ch2-Ch5** link `MLIR` + `LLVM`, i.e. only the shared `libMLIR.dylib` and
  `libLLVM.dylib` — no static MLIR `.a` files at all.

The problem only manifests when static MLIR archives (e.g. via `${dialect_libs}` or
the static `MLIRExecutionEngine`) and `libMLIR.dylib` end up in the same link, as in
the "BEFORE" setup above. Ch6 avoids this by staying shared-only: it links `MLIR` +
`MLIRExecutionEngineShared`.

## Alternative: All-Static Linking

If you build LLVM/MLIR from source with `BUILD_SHARED_LIBS=OFF` and without
generating `libMLIR.dylib` (i.e., `-DLLVM_BUILD_LLVM_DYLIB=OFF`), you can safely
use all static libraries:

```cmake
target_link_libraries(toyc-ch6
  PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    MLIRExecutionEngine   # safe: no dylib to conflict with
    MLIRAnalysis
    # ...
    )
```

This works because there is only one copy of every symbol (in the static archives).

## References

- MLIR TypeID documentation: https://mlir.llvm.org/docs/DefiningDialects/#typeid
- LLVM discussion on shared vs static: https://llvm.org/docs/CMake.html#shared-libs
