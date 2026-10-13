# MLIR Study

A collection of runnable MLIR tutorials and examples, built against Homebrew
`llvm@20` on macOS (Apple Silicon). Each directory stands on its own, but they
are meant to be read **in order**. The path goes from *using* MLIR, to
*defining* your own dialect in C++, packaging it, and finally scripting and
extending transformations.

## Reading order

| # | Directory | What it teaches | Why it comes here |
| --- | --- | --- | --- |
| **1** | [`mlir-ir-lowering/`](mlir-ir-lowering/) | *Using* MLIR: reading `.mlir`, the upstream dialects (`func`, `scf`, `arith`, `memref`, `affine`, `linalg`, `gpu`, …), lowering passes, the `mlir-opt` → `mlir-translate` → `llc` pipeline, and calling the result from Python | No C++ and no ODS. It builds the vocabulary (dialects, regions, blocks, passes, lowering) that everything else assumes. |
| **2** | [`toy/`](toy/) | *Defining* a dialect: a full compiler for the Toy language. Covers ODS (`Ops.td`), declarative rewrites (`ToyCombine.td`), interfaces and shape inference, lowering through `affine`/`memref` to `llvm`, and a JIT | The core "extend MLIR" tutorial. It is also your first contact with the C++ API (`MLIRContext`, dialect registration, the pass manager), and its lowering chapters (Ch5–Ch6) reuse exactly what #1 taught. |
| **3** | [`standalone/`](standalone/) | An out-of-tree dialect project template: CMake layout, `standalone-opt` / `standalone-translate`, a dialect plugin, Python bindings, lit tests | Applies Toy's ODS knowledge to a real project skeleton you could start your own dialect from. |
| **4** | [`transform/`](transform/) + [`transform-opt/`](transform-opt/) | The Transform dialect: scripting transformations (tiling, fusion, vectorization of `linalg`), then extending it with your own transform ops via ODS (`MyExtension.td`). `transform-opt/` is a standalone Transform-script interpreter | The most advanced. It needs #1 for what `linalg`, tiling and fusion are, and #2/#3 for writing ODS and C++. |

In short: **use** MLIR (1), then **define** a dialect (2), then **package** one
(3), then **script and extend** transformations (4).

### Supporting directories

- [`test_Example/`](test_Example/) holds the lit / FileCheck tests for the
  examples above (`Toy/`, `transform/`, `transform-opt/`, `standalone/`,
  `mlir-opt/`, `NVGPU/`). It is a reference, not a reading step.
- [`minimal-opt/`](minimal-opt/) holds three tiny C++ drivers (`mlir-cat`,
  `mlir-minimal-opt`, `mlir-minimal-opt-canonicalize`) that measure MLIR's
  **binary footprint** at different feature levels. Useful if you care about
  size, but not a reading step: `toy/` and `transform/` build the same
  `MlirOptMain` driver with real dialects registered.

## Where ODS comes in

Tutorial #1 never touches ODS (the TableGen `.td` files that define dialects,
ops, types and interfaces). It only uses dialects that already ship with MLIR,
through the stock tools. ODS first appears in #2 and is used from then on:

| Directory | ODS / TableGen files |
| --- | --- |
| `toy/` | `ChN/include/toy/Ops.td` (dialect + ops, Ch2+), `ChN/mlir/ToyCombine.td` (DRR, Ch3+), `ChN/include/toy/ShapeInferenceInterface.td` (Ch4+) |
| `standalone/` | `include/Standalone/*.td` |
| `transform/` | `ChN/include/MyExtension.td` (Ch2–Ch4), `Ch3/include/MyExtensionTypes.td` |

## Toolchain

Everything targets **Homebrew LLVM/MLIR 20**:

```bash
brew install llvm@20
export PATH="/opt/homebrew/opt/llvm@20/bin:$PATH"
```

- **#1** needs only the installed tools (`mlir-opt`, `mlir-translate`,
  `mlir-runner`, `llc`, `clang`) plus Python. See its
  [installation section](mlir-ir-lowering/README.md#installation-and-setup).
  Python dependencies (`numpy`, `llvmlite`) are listed in
  [`pyproject.toml`](pyproject.toml) (`uv sync`).
- **#2 and #4** build as one CMake superbuild from the top-level
  [`CMakeLists.txt`](CMakeLists.txt), which adds `toy`, `transform`,
  `transform-opt` and `minimal-opt`. `toy/` and `transform/` can also be built
  on their own; see their READMEs.
- **#3** (`standalone/`) is built separately. See
  [`standalone/README.md`](standalone/README.md).
