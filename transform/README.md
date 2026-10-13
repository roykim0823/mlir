# MLIR Transform Dialect Tutorial — Study Guide

A runnable, chapter-by-chapter study of the official
[Transform Dialect Tutorial](https://mlir.llvm.org/docs/Tutorials/transform/),
written against the code and the toolchain in this repository.

MLIR supports declarative specification for controlling compiler transformations
via the **Transform dialect**: transformations are requested with compiler IR
itself, either embedded in the IR being transformed (like pragmas) or supplied
separately (like a scheduling language). The tutorial shows three use scenarios:

- composing upstream Transform dialect operations into a sequence of optimizing
  transformations that turns a Linalg operation into efficient code;
- defining new Transform dialect operations and adapting existing transformation
  code to the Transform dialect infrastructure;
- setting up the Transform dialect in an out-of-tree project with custom
  dialects, transformations and passes.

## Chapters

| Dir | Chapter | Tool |
|-----|---------|------|
| [`Ch0/`](Ch0/README.md) | A Primer on "Structured" Linalg Operations | stock `mlir-opt` |
| [`Ch1/`](Ch1/README.md) | Combining Existing Transformations | stock `mlir-opt` |
| [`Ch2/`](Ch2/README.md) | Adding a Simple New Transformation Operation | `Ch2/build/transform-opt-ch2` |
| [`Ch3/`](Ch3/README.md) | More than Simple Transform Operations | `Ch3/build/transform-opt-ch3` |
| [`Ch4/`](Ch4/README.md) | Matching Payload with Transform Operations | `Ch4/build/transform-opt-ch4` |
| [`ChH/`](ChH/README.md) | Reproducing Halide Schedule | stock `mlir-opt` |

Each chapter directory holds a `README.md` (the study document: every topic of
the official chapter, expanded, with the exact command and real output for each
example) and `examples.sh`, a symlink to [`examples.sh`](examples.sh), which runs
the README's numbered commands:

```bash
cd Ch1
./examples.sh -l     # list the numbered commands
./examples.sh 3      # run command [3]
./examples.sh        # run them all
```

[`CLAUDE.md`](CLAUDE.md) records the conventions the chapter READMEs share.

## Toolchain

Homebrew LLVM/MLIR **20.1.8** (`brew install llvm@20 cmake ninja`), with
`/opt/homebrew/opt/llvm@20/bin` on `PATH` and `CC`/`CXX` set to its `clang`/`clang++`.
The documents follow the `release/20.x` branch of the tutorial.

## Building Ch2–Ch4

Ch0, Ch1 and ChH use only the installed `mlir-opt`. Ch2–Ch4 each build a small
`mlir-opt`-like driver with the chapter's Transform dialect extension:

```bash
cd Ch2
cmake -S . -B build -G Ninja
cmake --build build          # → ./build/transform-opt-ch2
```

The block at the top of each `ChN/CMakeLists.txt` makes the chapter configurable
on its own: it finds the installed MLIR/LLVM, defaults to a Release build (the
installed `libMLIR.dylib` is built with `NDEBUG`, and MLIR's headers declare
debug-only helpers under `#ifndef NDEBUG`), and the chapter links only
`libMLIR.dylib`, never the static MLIR archives alongside it.

## Test inputs

The chapters' inputs live in
[`../test_Example/transform/ChN/`](../test_Example/transform/), copies of
`mlir/test/Examples/transform` from `release/20.x`.
