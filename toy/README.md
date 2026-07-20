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
- **Per-chapter layout** (`Ch1/` … `Ch7/`): each chapter directory is
  self-contained — driver, headers, MLIR-side code, and a `CMakeLists.txt`
  declaring just that chapter's targets (executable, TableGen rules, link
  libraries). A small guard at the top
  (`if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)`) makes each
  chapter *also* configurable as a standalone project, so you can still
  study one chapter in isolation.
- **Test inputs** live in [`../test_Example/Toy/ChN/`](../test_Example/Toy/)
  (`ast.toy`, `codegen.toy`, `transpose_transpose.toy`, `struct-codegen.toy`, …),
  mirroring upstream `mlir/test/Examples/Toy/`.

### Repository layout

```text
mlir/                            # repo root
├── toy/                         # this directory: the superbuild + study docs
│   ├── CMakeLists.txt           # superbuild: find_package(MLIR/LLVM) once, add_subdirectory(Ch1..Ch7)
│   ├── CMakePresets.json        # pins Ninja, Release, Homebrew llvm@20 clang, MLIR_DIR/LLVM_DIR
│   ├── build.sh                 # ./build.sh [ch1..ch7|all] [--fresh] — configure once, build incrementally
│   ├── run.sh                   # ./run.sh <ch1..ch7|all> — each chapter's demo commands
│   ├── build/                   # shared build tree (created by build.sh; binaries in build/bin/toyc-ch1..7)
│   └── Ch1/ … Ch7/              # one directory per tutorial chapter:
│       ├── README.md            #   the chapter's study doc
│       ├── CMakeLists.txt       #   dual-mode: chapter targets + standalone-only boilerplate guard
│       ├── toyc.cpp             #   the chapter's compiler driver
│       ├── include/toy/         #   headers: Lexer.h, Parser.h, AST.h (+ Dialect.h, Ops.td, … from Ch2 on)
│       ├── parser/              #   AST.cpp (AST dumper) — the lexer/parser themselves are header-only
│       └── mlir/                #   MLIR-side code, from Ch2 on: Dialect.cpp, MLIRGen.cpp, passes, lowerings
└── test_Example/Toy/Ch1..Ch7/   # test inputs (ast.toy, codegen.toy, …), mirroring upstream mlir/test/Examples/Toy
```

Each chapter's README covers only what that chapter adds or changes — its own
files, build wiring, and run commands; the shared layout above is documented
only here.

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

## The build system

This section explains the shared build machinery **once**; each chapter README's "Building" section then covers only what that chapter *adds* (its own targets, TableGen steps, and link line). Upstream builds the Toy chapters in-tree with monorepo helpers like `add_toy_chapter`; everything below is the out-of-tree equivalent.

### The superbuild

The top-level [`CMakeLists.txt`](CMakeLists.txt) does the out-of-tree boilerplate once for every chapter:

***toy/CMakeLists.txt***

```cmake
cmake_minimum_required(VERSION 3.20)           # matches LLVM 20's own minimum

if(APPLE)                                      # pin the deployment target so the linker
  set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" ...)  # doesn't warn about Homebrew LLVM's setting
endif()
project(toy-tutorial)                          # an ordinary C++ project — NOT inside LLVM's build

find_package(MLIR REQUIRED CONFIG)             # CONFIG mode: locate the installed
find_package(LLVM REQUIRED CONFIG)             # MLIRConfig.cmake / LLVMConfig.cmake

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)                              # defines mlir_tablegen(), add_public_tablegen_target()
include(AddLLVM)                               # defines add_llvm_executable() etc.
include(AddMLIR)                               # defines add_mlir_dialect() etc.
include(HandleLLVMOptions)                     # sets LLVM's expected flags (-fno-rtti, ...)

include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})

# Collect every chapter binary in build/bin/ instead of build/ChN/.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(Ch1)
# ... through Ch7
```

The load-bearing ideas:

- **`find_package(... CONFIG)`** resolves via `MLIR_DIR`/`LLVM_DIR` (pinned in the preset, below) and imports every prebuilt MLIR/LLVM library as a CMake target — which is why chapters can write `target_link_libraries(toyc-chN PRIVATE MLIRSupport)` (or `MLIR`, `LLVM`) with no manual `-L`/`-l` flags.
- **The four `include(...)`s** pull in LLVM's CMake helper macros; `HandleLLVMOptions` also applies the compiler flags MLIR headers require, notably `-fno-rtti`.
- **Ordering matters**: packages → module path → includes; each step consumes variables the previous one produced.
- **`CMAKE_RUNTIME_OUTPUT_DIRECTORY`** lands every `toyc-chN` in one `build/bin/`.

### The pinned toolchain

[`CMakePresets.json`](CMakePresets.json)'s `default` preset makes a bare `cmake --preset default` reproduce the exact configuration:

***toy/CMakePresets.json***

```json
"generator": "Ninja",
"binaryDir": "${sourceDir}/build",
"cacheVariables": {
  "CMAKE_BUILD_TYPE": "Release",
  "CMAKE_C_COMPILER": "/opt/homebrew/opt/llvm@20/bin/clang",
  "CMAKE_CXX_COMPILER": "/opt/homebrew/opt/llvm@20/bin/clang++",
  "MLIR_DIR": "/opt/homebrew/opt/llvm@20/lib/cmake/mlir",
  "LLVM_DIR": "/opt/homebrew/opt/llvm@20/lib/cmake/llvm"
}
```

Building with Homebrew's *own* clang++ (not Apple Clang) keeps the compiler/stdlib ABI-consistent with the prebuilt MLIR/LLVM dylibs — mixing them invites ABI-flavored link and runtime surprises. A matching build preset lets `cmake --build --preset default [--target toyc-chN]` work without naming the build directory.

### The dual-mode chapter pattern

Every `ChN/CMakeLists.txt` starts with the same guard:

```cmake
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # ... the same find_package/include boilerplate as the top level ...
endif()
```

In the superbuild, `CMAKE_SOURCE_DIR` is `toy/` while `CMAKE_CURRENT_SOURCE_DIR` is `toy/ChN/`, so the guard is false and the top level's setup is reused. Configured standalone (`cmake -S ChN`), the guard is true and the chapter bootstraps itself — that's what makes the standalone recipe in "Build & run" work with no preset. Below the guard, each chapter declares only its targets: TableGen steps, the `toyc-chN` executable, `add_dependencies` on the generated headers, and the link line. **Those parts are the chapter-specific build content, documented in each chapter's README.**

### Build script mechanics and artifacts

`./build.sh [ch1..ch7|all] [--fresh]`:

1. `--fresh` runs `rm -rf build` first; otherwise the tree is **kept and reused incrementally** — and it is shared by all chapters, so a fresh wipe rebuilds everything.
2. If `build/CMakeCache.txt` doesn't exist it configures with `cmake --preset default`; afterwards Ninja re-runs CMake automatically when any `CMakeLists.txt` changes.
3. Builds `cmake --build --preset default` (all) or `--target toyc-chN` (one chapter).

Artifacts land in `toy/build/`: the binaries you care about in `build/bin/toyc-ch1..7`, plus per-chapter object/TableGen output under `build/ChN/` (generated `.inc` files at `build/ChN/include/`). `run.sh` looks in `build/bin/` first and falls back to `ChN/build/` for standalone-built chapters.

### What each chapter adds to the build

| Ch | New build ingredients (details in that chapter's README) |
|---|---|
| [1](Ch1/README.md) | Plain `add_executable`; links only `MLIRSupport` — no MLIR IR is used yet |
| [2](Ch2/README.md) | **TableGen enters**: `Ops.td` → `mlir_tablegen(-gen-op-decls/-gen-op-defs/...)`, `add_public_tablegen_target`, and the `add_dependencies(toyc-ch2 ToyCh2OpsIncGen)` race guard; links monolithic `MLIR` + `LLVM` dylibs |
| [3](Ch3/README.md) | Second TableGen flavor: `ToyCombine.td` → `-gen-rewriters` (DRR patterns) |
| [4](Ch4/README.md) | Third TableGen flavor: `ShapeInferenceInterface.td` → `-gen-op-interface-decls/-defs` |
| [5](Ch5/README.md) | Nothing new — the monolithic `MLIR` dylib already contains the affine/memref dialects and conversion passes |
| [6](Ch6/README.md) | The ExecutionEngine — and with it the **shared-vs-static linking trap** (TypeID-duplication segfault); links `MLIR` + `MLIRExecutionEngineShared` *only* |
| [7](Ch7/README.md) | Same link line as Ch6; one more TableGen'd dialect (struct ops) behind the same IncGen dependency pattern |

## The Chapters

Each linked document covers the full content of the corresponding official
chapter, plus a file-by-file walkthrough of the code in this repo, an
explanation of the CMake/TableGen build wiring, and the **actual captured
output** of every `run.sh` command, annotated line by line.

| # | Study doc | Official chapter | What you learn |
|---|-----------|------------------|----------------|
| 1 | [Ch1/README.md](Ch1/README.md) | [Toy Language and AST](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-1/) | The Toy language, a hand-written lexer and recursive-descent parser, the AST classes, and dumping the AST (`-emit=ast`). No MLIR yet — only `MLIRSupport` is linked. |
| 2 | [Ch2/README.md](Ch2/README.md) | [Emitting Basic MLIR](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/) | Core MLIR concepts (operations, values, dialects), defining the Toy dialect and its ops with **ODS/TableGen**, verifiers, custom assembly formats, and `MLIRGen` that walks the AST to emit `toy.*` ops (`-emit=mlir`), including a parser/printer round-trip test. |
| 3 | [Ch3/README.md](Ch3/README.md) | [High-level Language-Specific Analysis and Transformation](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/) | Optimizing at the Toy level: a C++ `RewritePattern` that erases `transpose(transpose(x))`, **declarative rewrite rules (DRR)** for reshape folding, and the canonicalizer pass (`-opt`). |
| 4 | [Ch4/README.md](Ch4/README.md) | [Enabling Generic Transformation with Interfaces](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/) | Making MLIR's *generic* passes work on Toy via **interfaces**: the dialect inliner interface + `toy.cast`, and a custom `ShapeInferenceOpInterface` with a shape-inference pass that turns `tensor<*xf64>` into ranked tensors. |
| 5 | [Ch5/README.md](Ch5/README.md) | [Partial Lowering to Lower-Level Dialects for Optimization](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/) | The **dialect conversion framework**: partially lowering Toy ops to `affine` loop nests over `memref` buffers (`-emit=mlir-affine`) while keeping `toy.print` around, then reusing generic affine optimizations (loop fusion, store-to-load forwarding). |
| 6 | [Ch6/README.md](Ch6/README.md) | [Lowering to LLVM and CodeGeneration](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/) | Finishing the pipeline: lowering `toy.print` to `printf` via the LLVM dialect, **full conversion** with `LLVMTypeConverter` (`-emit=mlir-llvm`), translating to real LLVM IR (`-emit=llvm`), and executing with the MLIR **ExecutionEngine JIT** (`-emit=jit`). |
| 7 | [Ch7/README.md](Ch7/README.md) | [Adding a Composite Type to Toy](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-7/) | Extending every layer for a `struct` type: grammar/AST changes, a custom `StructType` with type storage/uniquing, custom type parsing/printing (`!toy.struct<...>`), new ops (`toy.struct_constant`, `toy.struct_access`), and **folding** that optimizes structs away entirely. |

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
