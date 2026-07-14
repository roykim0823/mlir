# MLIR Toy Tutorial — Step-by-Step Study Guide

This directory contains a complete, runnable implementation of the official
[MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/), together with a
set of very detailed, chapter-by-chapter study documents written against the
code in **this** repository.

> The goal of the Toy tutorial (quoting the official docs) is *"to introduce
> the concepts of MLIR; in particular, how dialects can help easily support
> language specific constructs and transformations while still offering an
> easy path to lower to LLVM or other codegen infrastructure."* It is inspired
> by the classic LLVM [Kaleidoscope](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html)
> tutorial.

Toy is a tiny tensor language (variables, `def` functions, tensor literals,
`transpose`, `print`, element-wise `+`/`*`, and — in Chapter 7 — `struct`s).
Over seven chapters you build a full compiler for it: lexer → AST → a custom
MLIR **dialect** → high-level optimizations → progressive **lowering** through
the `affine`/`memref`/`arith` dialects → the **LLVM dialect** → LLVM IR → a
**JIT** that actually executes the program.

## How this repo differs from upstream llvm-project

The upstream tutorial code lives inside the LLVM monorepo and is built as part
of it. The code here is the same tutorial, built **out-of-tree as one CMake
superbuild** against an installed MLIR/LLVM:

- **Toolchain**: Homebrew LLVM/MLIR 20 on macOS
  (`MLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir`,
  compiler `/opt/homebrew/opt/llvm@20/bin/clang++`), CMake ≥ 3.25, Ninja.
  All of this is pinned in [`CMakePresets.json`](CMakePresets.json), so no
  flags or environment variables are needed to configure.
- **One superbuild**: the top-level [`CMakeLists.txt`](CMakeLists.txt) runs
  `find_package(MLIR/LLVM)` **once** and then pulls in every chapter with
  `add_subdirectory(ChN)`. All binaries land in `build/bin/toyc-ch{1..7}`,
  and builds are incremental — no per-chapter reconfigure, no `rm -rf build`.
- **Per-chapter layout** (`Ch1/` … `Ch7/`), each with:
  - `toyc.cpp` — the compiler driver for that chapter
  - `include/toy/`, `parser/`, `mlir/` — headers, AST/parser, and MLIR-side code
  - `CMakeLists.txt` — just the chapter's targets (executable, TableGen rules,
    link libraries). A small guard at the top
    (`if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)`) makes each
    chapter *also* configurable as a standalone project, so you can still
    study one chapter in isolation.
- **Test inputs** live in [`../test_Example/Toy/ChN/`](../test_Example/Toy/)
  (`ast.toy`, `codegen.toy`, `transpose_transpose.toy`, `struct-codegen.toy`, …),
  mirroring upstream `mlir/test/Examples/Toy/`.

### Prerequisites

```bash
brew install llvm@20 cmake ninja
```

### Build & run

```bash
cd toy
./build.sh          # configure once (cmake --preset default) + build everything
./build.sh ch3      # build only toyc-ch3 (incremental)
./build.sh --fresh  # wipe build/ and start over

./run.sh ch1        # run a chapter's demo commands on the test inputs
./run.sh all        # run every chapter in order
```

`build.sh` is a thin wrapper around CMake presets — the equivalent raw
commands are:

```bash
cmake --preset default               # configure into build/ (only needed once)
cmake --build --preset default       # build everything
ninja -C build toyc-ch3              # or build a single chapter target
```

`run.sh` holds each chapter's demo/test commands in one place and invokes the
binaries from `build/bin/`. You can of course also run a binary directly:

```bash
./build/bin/toyc-ch1 ../test_Example/Toy/Ch1/ast.toy -emit=ast
```

To build one chapter standalone (e.g. to experiment without touching the
rest), configure it directly and pass the MLIR/LLVM locations:

```bash
cmake -S Ch3 -B Ch3/build -G Ninja \
  -DMLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir \
  -DLLVM_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/llvm
cmake --build Ch3/build
```

## The Chapters

Each linked document covers the full content of the corresponding official
chapter, plus a file-by-file walkthrough of the code in this repo, an
explanation of the CMake/TableGen build wiring, and the **actual captured
output** of every `run.sh` command, annotated line by line.

| # | Study doc | Official chapter | What you learn |
|---|-----------|------------------|----------------|
| 1 | [1_toy_lang_ast.md](1_toy_lang_ast.md) | [Toy Language and AST](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-1/) | The Toy language, a hand-written lexer and recursive-descent parser, the AST classes, and dumping the AST (`-emit=ast`). No MLIR yet — only `MLIRSupport` is linked. |
| 2 | [2_emitting_basic_mlir.md](2_emitting_basic_mlir.md) | [Emitting Basic MLIR](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/) | Core MLIR concepts (operations, values, dialects), defining the Toy dialect and its ops with **ODS/TableGen**, verifiers, custom assembly formats, and `MLIRGen` that walks the AST to emit `toy.*` ops (`-emit=mlir`), including a parser/printer round-trip test. |
| 3 | [3_high_level_transformations.md](3_high_level_transformations.md) | [High-level Language-Specific Analysis and Transformation](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/) | Optimizing at the Toy level: a C++ `RewritePattern` that erases `transpose(transpose(x))`, **declarative rewrite rules (DRR)** for reshape folding, and the canonicalizer pass (`-opt`). |
| 4 | [4_generic_transformation_interfaces.md](4_generic_transformation_interfaces.md) | [Enabling Generic Transformation with Interfaces](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/) | Making MLIR's *generic* passes work on Toy via **interfaces**: the dialect inliner interface + `toy.cast`, and a custom `ShapeInferenceOpInterface` with a shape-inference pass that turns `tensor<*xf64>` into ranked tensors. |
| 5 | [5_partial_lowering.md](5_partial_lowering.md) | [Partial Lowering to Lower-Level Dialects for Optimization](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/) | The **dialect conversion framework**: partially lowering Toy ops to `affine` loop nests over `memref` buffers (`-emit=mlir-affine`) while keeping `toy.print` around, then reusing generic affine optimizations (loop fusion, store-to-load forwarding). |
| 6 | [6_lowering_to_llvm_jit.md](6_lowering_to_llvm_jit.md) | [Lowering to LLVM and CodeGeneration](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/) | Finishing the pipeline: lowering `toy.print` to `printf` via the LLVM dialect, **full conversion** with `LLVMTypeConverter` (`-emit=mlir-llvm`), translating to real LLVM IR (`-emit=llvm`), and executing with the MLIR **ExecutionEngine JIT** (`-emit=jit`). |
| 7 | [7_struct_types.md](7_struct_types.md) | [Adding a Composite Type to Toy](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-7/) | Extending every layer for a `struct` type: grammar/AST changes, a custom `StructType` with type storage/uniquing, custom type parsing/printing (`!toy.struct<...>`), new ops (`toy.struct_constant`, `toy.struct_access`), and **folding** that optimizes structs away entirely. |

Suggested path: read the chapters in order — each one builds directly on the
previous chapter's code, and the study docs point out exactly what changed
between chapters.

## The pipeline at a glance

```
Toy source (.toy)
   │  Lexer/Parser (Ch1)
   ▼
AST                                  -emit=ast
   │  MLIRGen (Ch2)
   ▼
Toy dialect MLIR                     -emit=mlir
   │  canonicalize/DRR (Ch3), inline + shape inference (Ch4)   [-opt]
   ▼
Optimized Toy MLIR
   │  partial lowering (Ch5)
   ▼
affine + memref + arith (+ toy.print) -emit=mlir-affine
   │  full lowering (Ch6)
   ▼
LLVM dialect MLIR                    -emit=mlir-llvm
   │  translation
   ▼
LLVM IR                              -emit=llvm
   │  ExecutionEngine
   ▼
JIT execution                        -emit=jit
```

## Practical notes & pitfalls

- **Linking MLIR out-of-tree can segfault if you mix static and shared MLIR
  libraries.** This repo hit a real crash (TypeID duplication →
  `StorageUniquer` segfault) when `MLIRExecutionEngine` pulled in
  `libMLIR.dylib` alongside static `.a` archives. Root cause, symptoms, and
  the fix are documented in
  [Ch6/MLIR_LINKING_PITFALL.md](Ch6/MLIR_LINKING_PITFALL.md).
- `Ch2/run_mlir-tblgen.sh` shows how to invoke `mlir-tblgen` **by hand**
  (`-gen-dialect-decls`, `-gen-op-defs`, …) so you can inspect exactly what
  ODS generates into the `*.inc` files — highly recommended while reading
  Chapter 2.
- Most chapter binaries accept `--help` and the standard MLIR pass/print
  flags (e.g. `-mlir-print-debuginfo`, `--mlir-print-ir-after-all`), which are
  great for watching each pass transform the IR.

## References

- Official tutorial: <https://mlir.llvm.org/docs/Tutorials/Toy/>
- 2020 LLVM Dev Conference talk: *MLIR Tutorial* (linked from the page above)
- Upstream sources: [`llvm-project/mlir/examples/toy`](https://github.com/llvm/llvm-project/tree/main/mlir/examples/toy)
- Test inputs: [`llvm-project/mlir/test/Examples/Toy`](https://github.com/llvm/llvm-project/tree/main/mlir/test/Examples/Toy)
