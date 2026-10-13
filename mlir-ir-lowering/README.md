# MLIR IR Lowering

A hands-on, example-driven tour of **MLIR** (Multi-Level Intermediate
Representation), the compiler infrastructure inside the LLVM project. Each
chapter pairs a short explanation with runnable `.mlir` code and `build.sh`
scripts you can execute and inspect step by step.

The focus is on *writing* MLIR IR with the dialects that ship with MLIR and
*lowering* it to native code. No C++, ODS or DRR is involved; defining your own
dialect starts in the next tutorial, [`../toy/`](../toy/).

> This series is based on Stephen Diehl's *"MLIR"* tutorial articles (see
> [`reference/`](reference/)) and is adapted/expanded for a macOS +
> Apple-Silicon, Homebrew `llvm@20` setup. For installation instructions, jump to
> [**Installation and setup**](#installation-and-setup).

---

## What is MLIR?

**MLIR** (Multi-Level Intermediate Representation) is a compiler framework, built
*within* the LLVM ecosystem, for representing and transforming code at **many
levels of abstraction at once** — from high-level tensor math down to
hardware-specific instructions. It was designed to address the challenges of
modern hardware accelerators and machine-learning frameworks, where a single
fixed IR is too rigid.

To see why that matters, it helps to recall how **LLVM** works. LLVM is a
powerful, modular compiler infrastructure that has reshaped how we build
programming languages and tools: a front-end translates a source language into
**LLVM IR** — a low-level, assembly-like representation that still preserves some
high-level information — and LLVM then optimizes that IR and generates machine
code for many target architectures (x86, ARM, RISC-V, WASM, NVPTX, …).

LLVM IR is powerful but **low-level**. Expressing a high-level concept such as a
loop, a tensor operation, or a parallel region directly in LLVM IR is tedious
and loses structure the optimizer could have exploited. Historically, each
language invented its own *mid-level IR* to bridge this gap — Rust has MIR,
Swift has SIL, TensorFlow had XLA HLO. They all solved the same problem
independently.

MLIR generalizes that idea. Instead of one fixed IR, MLIR provides a framework
for *many* coexisting IRs, called **dialects**, at different levels of
abstraction. You write code in a high-level dialect and progressively **lower**
it, one pass at a time, until you reach the `llvm` dialect, which maps directly
onto LLVM IR.

```
  Source        Mid-level IR         Unified backend        Targets
  ──────        ────────────         ───────────────        ───────
  C / C++ ─┐                                              ┌─ x86
  Rust   ──┤    (MIR, SIL, …)                             ├─ ARM
  Swift  ──┤         ──►        MLIR dialects ──► LLVM IR ─┤─ RISC-V
  Julia  ──┤                    (lowering passes)         ├─ NVPTX (GPU)
  PyTorch ─┘                                              └─ WASM
```

This design exists because traditional compilers like LLVM and GCC, while
excellent at conventional CPU targets, struggle with the diverse specialized
architectures emerging in AI and machine learning. MLIR addresses that head-on:

- **Heterogeneous hardware.** Modern AI systems mix CPUs, GPUs, TPUs, and custom
  ASICs. MLIR's flexible architecture makes it easier to work across all of them
  from a single infrastructure.
- **The right level of abstraction.** The dialect system can represent and
  optimize operations at many levels — from high-level ML tasks down to
  hardware-specific instructions — so optimizations can be tailored to each
  domain and AI workloads run more efficiently.
- **One unified infrastructure.** Instead of a separate compiler for each
  accelerator or framework, MLIR offers a single extensible base with
  first-class support for tensors, neural networks, and transformers — constructs
  traditional compilers were never built for.
- **"Weird domains."** Its ability to spin up domain-specific compilers (and to
  be embedded in other languages as a DSL) has made MLIR the go-to tech well
  beyond ML: signal processing, quantum computing, homomorphic encryption,
  FPGAs, and custom silicon.

### Key components of the modern compiler stack

A modern compiler weaves a few reusable pieces together:

1. **LLVM** — its intermediate representation is a low-level, assembly-like
   language that still preserves important high-level information, bridging
   source languages and target architectures.
2. **MLIR** — a "multi-level" IR that is more expressive than LLVM IR and can
   represent higher-level concepts like control flow, dataflow, and parallelism
   before being lowered into LLVM IR.
3. **Optimization passes** — both MLIR and LLVM ship a vast library of passes
   that operate at the IR level, so any language built on them inherits
   sophisticated, research-grade optimizations for free.
4. **E-graphs** — *equality saturation* (a technique from automated theorem
   proving) builds optimizing compilers by applying rewrites via e-matching
   until the e-graph is saturated, then extracting an optimal term according to a
   cost function — exploring a broad space of transformations and selecting the
   best one.

### The modern compiler pipeline

A typical MLIR-based compiler weaves these stages together:

1. Source code is parsed into a surface language.
2. The surface language is translated to a core language.
3. The core language is optimized by core-to-core transformations (optionally
   using techniques like **e-graphs** / equality saturation).
4. The core language is lowered to MLIR.
5. MLIR performs high-level optimizations.
6. MLIR is lowered to LLVM IR.
7. LLVM performs low-level optimizations and code generation.

---

## Key MLIR concepts

MLIR is a programming language in its own right. It is very low-level, but it
still has the core primitives you would expect from any language — modules,
functions, types, control flow. You'll see all of these in the examples; this
section is the glossary.

### Identifier conventions

MLIR uses a small set of sigils and delimiters to disambiguate the different
kinds of names and syntactic groups that appear in the IR.

| Sigil | Meaning | Example |
| --- | --- | --- |
| `%` | SSA value | `%result` |
| `@` | Function / symbol name | `@main` |
| `^` | Basic block label | `^bb0` |
| `#` | Attribute alias | `#map_1d_identity` |
| `!` | Type alias | `!avx_m128 = vector<4xf32>` |
| `x` | Shape / element-type delimiter | `10xf32` in `tensor<10xf32>` |
| `:` / `->` | Type annotation / result type | `%r : i32`, `(i32) -> i32` |
| `( )` | Operands / arguments | `(%arg0, %arg1)` |
| `{ }` | Region | `{ ... }` |
| `< >` | Type parameters | `tensor<10xf32>` |
| `//` | Comment | `// like this` |

### Structural concepts

**1. Modules** — the top-level container that holds everything else. Every `.mlir`
file is implicitly (or explicitly) wrapped in one:

```mlir
module {
  // functions, globals, and other operations go here
}
```

**2. Functions** — a named, ordered collection of operations, declared with the
`func` dialect. Arguments and the return type are part of the signature:

```mlir
func.func @my_function(%arg0: i32, %arg1: i32) -> i32 {
  // operations
}
```

**3. Operations** — the basic unit of work, analogous to an LLVM instruction but
namespaced by its dialect and carrying explicit type annotations. The fully
generic form names the operation as a string and lists operand and result types:

```mlir
%0 = "my_dialect.my_operation"(%arg0, %arg1) : (i32, i32) -> i32
```

Here `my_dialect.my_operation` is an op defined in `my_dialect`, `%arg0` and
`%arg1` are `i32` operands, and the `i32` result is bound to `%0`. In practice
most dialects also provide a *pretty* (custom) syntax. For example, the
`arith.addf` op adds two floating-point numbers:

```mlir
%0 = arith.addf %arg0, %arg1 : f32
```

**4. Basic blocks** — a straight-line sequence of operations with a single entry
point that executes without branching, ending in a control-flow op (a branch or
return). Blocks are labelled with the `^` sigil:

```mlir
^bb1:  // label for the "then" block
  %then_result = arith.muli %result, 2 : i32
  return %then_result : i32
```

Unlike LLVM, MLIR basic blocks can take **arguments**, passed in via the
`^bb1(%result: i32)` syntax. This replaces LLVM's phi nodes: instead of merging
values with `phi`, a predecessor block simply passes them as block arguments.

**5. Regions** — an ordered group of basic blocks attached to an operation,
written inside `{ }`. Regions are how MLIR represents nested control flow such as
loop and conditional bodies, and like blocks they can take arguments:

```mlir
{
  ^bb1(%result: i32):
    %then_result = arith.muli %result, 2 : i32
    return %then_result : i32
}
```

**6. Types** — a classification that says what kind of value an SSA value holds
and which operations apply to it. Common types are `i32`, `f32`, `index`,
`tensor<10xf32>`, and `memref<4x4xf32>`:

```mlir
%result = arith.constant 1 : i32
```

For convenience you can define a **type alias** (synonym) with the `!` sigil:

```mlir
!avx_m128 = vector<4xf32>
```

**7. Dialects** — the namespaced sets of operations and types that make MLIR
multi-level. You write code in a high-level dialect and lower it toward `llvm`.
The dialects used across this series:

| Level | Dialect | Purpose | Key ops | Chapter |
| --- | --- | --- | --- | --- |
| High | `tensor` | Immutable, value-semantic multi-dimensional arrays with **no memory location**. Lets you write math as pure, side-effect-free transformations; *bufferization* later turns tensors into `memref`s. Shapes may be static (`tensor<10x10xf32>`) or dynamic (`tensor<?x?xf32>`). | `tensor.empty`, `tensor.from_elements`, `tensor.extract`, `tensor.insert`, `tensor.generate` | 2, 4 |
| High | `linalg` | Declarative, **structured** linear-algebra ops (pointwise, matmul, convolution, …) expressed as single operations that keep the math intact, so the compiler can fuse, tile, and vectorize them. Every named op is a specialization of `linalg.generic`, which describes a computation via indexing maps and iterator types. | `linalg.generic`, `linalg.matmul`, `linalg.fill`, `linalg.yield` | 4, 5 |
| High | `affine` | "First-class loops": loop nests and array accesses whose bounds and subscripts are **affine** functions of loop indices. Built on the polyhedral model, it enables dependence analysis, loop tiling/fusion/unrolling, automatic parallelization, and vectorization. Lowered to `scf` + `arith` by `--lower-affine`. | `affine.for`, `affine.parallel`, `affine.load`, `affine.store`, `affine.apply`, `affine_map<…>` | 3, 4, 8 |
| High | `omp` | The **OpenMP** programming model inside MLIR — explicit control over thread teams, work-sharing loops, and synchronization (as opposed to `affine`'s automatic parallelization). Lowers to calls into the OpenMP runtime (`libomp`). | `omp.parallel`, `omp.wsloop`, `omp.loop_nest`, `omp.barrier`, `omp.terminator` | 3 |
| High | `gpu` | Target-agnostic, CUDA/OpenCL-style abstraction for **launching kernels** on heterogeneous hardware: grids, blocks, threads, device memory, and host/device separation (`gpu.module` / `gpu.func`). Lowered to a vendor dialect such as `nvvm`. | `gpu.launch`, `gpu.launch_func`, `gpu.thread_id`, `gpu.block_id`, `gpu.barrier`, `gpu.alloc`, `gpu.memcpy` | 8 |
| Mid | `vector` | Hardware-independent **SIMD** vectors, 1-D (`vector<4xf32>`) or n-D (`vector<2x4xf32>`). LLVM maps them to native vector units (AVX, NEON), splits them, or scalarizes them as the target allows. | `vector.extract`, `vector.shuffle`, `vector.reduction`, `vector.broadcast` | 2, 3 |
| Low | `scf` | **Structured** control flow: loops and conditionals as ops with nested regions, which keeps the structure visible to optimizers. Loop-carried values are threaded through `iter_args` and `scf.yield`. Lowered to `cf` by `--convert-scf-to-cf`. | `scf.for`, `scf.if`, `scf.while`, `scf.parallel`, `scf.yield` | 1–4, 7, 8 |
| Low | `cf` | **Unstructured** control flow: explicit jumps between basic blocks, which pass values as block arguments (MLIR's replacement for LLVM's φ-nodes). Every `scf` construct ultimately compiles into these branches. | `cf.br`, `cf.cond_br` | 1–3 (generated by lowering) |
| Low | `func` | Function definitions, calls, and returns: the unit that organizes and modularizes code. Arguments and results are part of the typed signature. | `func.func`, `func.call`, `func.return` | all |
| Low | `arith` | Fundamental integer and floating-point operations that work on scalars, vectors, and tensors: arithmetic, comparisons, bitwise logic, and conversions/casts. | `arith.constant`, `arith.addi`, `arith.mulf`, `arith.cmpi`, `arith.sitofp`, `arith.andi` | all |
| Low | `math` | Higher-level math functions beyond basic arithmetic (trigonometric, exponential/logarithmic, power, special functions). Lowered to LLVM intrinsics or libm calls. | `math.sin`, `math.exp`, `math.log`, `math.sqrt`, `math.powf`, `math.tanh` | 7 |
| Low | `index` | Operations on the `index` type, a **platform-sized integer** (like C's `size_t`) used for loop induction variables, array subscripts, and dimension sizes. Used together with `memref`/`tensor` for addressing. | `index.constant`, `index.add`, `index.sub`, `index.mul`, `index.cmp` | 1, 3 |
| Low | `memref` | **Concrete memory buffers**, the bridge between NumPy-style arrays and raw pointers. A memref is a pointer plus metadata (allocated/aligned pointers, offset, shape, strides), so it can describe views and non-contiguous layouts. | `memref.alloc`, `memref.dealloc`, `memref.load`, `memref.store`, `memref.subview`, `memref.dim` | 2–4, 7, 8 |
| Target | `nvgpu` / `nvvm` | NVIDIA-specific dialects. `nvgpu` bridges `gpu`/`vector` to advanced hardware features (Tensor Memory Accelerator, warp-level MMA); `nvvm` mirrors the LLVM NVPTX intrinsics and lowers to PTX. | `nvvm.read.ptx.sreg.tid.x`, `nvvm.barrier0`, `nvgpu.mma.sync` | 8 (`nvvm`) |
| Lowest | `llvm` | A near 1:1 mirror of **LLVM IR**: the bottom of the MLIR hierarchy. Every pipeline ends here, and `mlir-translate --mlir-to-llvmir` passes it straight through to textual LLVM IR. Also offers C-like structs, arrays, and raw pointers for fine-grained control. | `llvm.func`, `llvm.call`, `llvm.getelementptr`, `llvm.load`, `llvm.store`, `llvm.insertvalue`, `llvm.mlir.constant` | all |

**8. Passes** — transformations that operate on the dialects, optimizing the IR
or **lowering** it into simpler constructs. They are the arguments you hand to
`mlir-opt`. The most common ones are conversions toward the `llvm` dialect:

| Pass | Effect |
| --- | --- |
| `--convert-func-to-llvm` | Convert function-like ops to the `llvm` dialect |
| `--convert-arith-to-llvm` | Convert arithmetic ops to the `llvm` dialect |
| `--convert-math-to-llvm` | Convert math ops to the `llvm` dialect |
| `--convert-index-to-llvm` | Convert index ops to the `llvm` dialect |
| `--convert-scf-to-cf` | Lower structured control flow to the `cf` dialect |
| `--convert-cf-to-llvm` | Convert control flow to the `llvm` dialect |
| `--finalize-memref-to-llvm` | Convert memref ops to the `llvm` dialect |
| `--convert-vector-to-llvm` | Convert vector ops to the `llvm` dialect |
| `--convert-linalg-to-loops` | Expand `linalg` ops into `scf` loops |
| `--reconcile-unrealized-casts` | Resolve leftover `unrealized_conversion_cast`s |

There is also a catch-all `--convert-to-llvm` that lowers anything it can; in
practice we prefer the granular passes so we control exactly what happens.

For finer control, list passes in a `--pass-pipeline` string — they run
sequentially as one group:

```bash
mlir-opt --pass-pipeline="builtin.module(convert-scf-to-cf,convert-cf-to-llvm)" in.mlir
```

> **Pass order matters.** For example, `convert-scf-to-cf` must run *before*
> `convert-cf-to-llvm`, because the second pass only knows how to lower the `cf`
> dialect the first one produces.

When a pipeline misbehaves, these debugging flags print the IR as it changes:

| Flag | What it shows |
| --- | --- |
| `--mlir-print-ir-after-all` | The IR after every pass |
| `--mlir-print-ir-after-change` | The IR only after a pass that changed it |
| `--mlir-print-ir-after-failure` | The IR after a pass that failed |
| `--mlir-print-ir-tree-dir=<dir>` | Write each IR snapshot to files instead of stdout |

---

## Standard MLIR dialects

The concepts above are the vocabulary; the dialects below are the workhorses you
hit first when lowering toward native code. The reference article groups six of
them as **"standard MLIR dialects"**: `llvm`, `scf`, `cf`, `arith`, `math` and
`index`. This section adds `func`, which every example uses. Chapter 1's
[`example.mlir`](1_intro/2_mlir/example.mlir) uses four of them (`func`,
`index`, `scf`, `arith`), and lowering it passes through two more (`cf`, then
`llvm`).

> **Is "standard" an official term?** Not anymore. MLIR used to have a single
> catch-all **`std` dialect** that held arithmetic, branches, function calls,
> memory ops, and more. Around 2021–2022 (LLVM 14–15) it was split into focused
> dialects: `arith`, `math`, `func`, `cf`, `memref`, `tensor`, and others.
> "Standard" here just means *the low-level common floor*: the dialects that
> almost every pipeline lowers through on its way to `llvm`.
>
> They are far from the only ones that ship with MLIR. Homebrew `llvm@20` has
> about 45 **upstream (in-tree) dialects**
> (`ls $(brew --prefix llvm@20)/include/mlir/Dialect`). Roughly grouped:
>
> | Group | Upstream dialects |
> | --- | --- |
> | Core / structural | `builtin` (`module`, the built-in types), `func`, `cf`, `scf`, `ub`, `dlti` |
> | Computation | `arith`, `math`, `index`, `complex`, `polynomial` |
> | Data and memory | `tensor`, `memref`, `bufferization`, `sparse_tensor`, `vector`, `ptr`, `shape` |
> | Structured / ML | `linalg`, `affine`, `tosa`, `ml_program`, `quant`, `mesh` |
> | Parallelism and runtime | `omp` (OpenMP), `acc` (OpenACC), `async`, `mpi` |
> | GPU / accelerators | `gpu`, `nvgpu`, `nvvm`, `amdgpu`, `rocdl`, `xegpu`, `spirv` |
> | CPU vector ISAs | `x86vector`, `amx`, `arm_neon`, `arm_sve`, `arm_sme` |
> | Exit points | `llvm` (→ LLVM IR), `emitc` (→ C/C++), `spirv` (→ SPIR-V) |
> | Meta / tooling | `transform`, `pdl`, `pdl_interp`, `irdl` |
>
> Projects outside LLVM add their own on top: `torch` (Torch-MLIR), `stablehlo`
> (XLA), `triton`, CIRCT's hardware dialects, and so on. Later chapters of this
> series add `tensor`, `memref`, `vector`, `affine`, `omp`, `linalg`, `gpu` and
> `nvvm`. The ones below are the floor they all land on.

### `func`: functions and calls

`func.func` defines a function with a typed signature, `func.call` (written
`call` inside a function) invokes one, and `func.return` (`return`) hands back
results. Every program in this series is organized this way. The
`--convert-func-to-llvm` pass turns these into `llvm.func`, `llvm.call` and
`llvm.return`.

### `scf` and `cf`: control flow

These are two views of the same thing at different levels.

**`scf`** (structured control flow) gives you the constructs you'd expect from a
real language: `scf.if`, `scf.for` and `scf.while`. Each one has a single entry
and a single exit, with its body nested inside as a **region**. Because the loop
is still *one op*, an optimizer can see its bounds, step and body at once:

```mlir
scf.if %b {
  // true region
} else {
  // false region
}

scf.for %iv = %lb to %ub step %step {
  // loop region
}
```

**`cf`** (unstructured control flow) is what those structures compile down to:
plain branches between basic blocks.

| Op | Meaning |
| --- | --- |
| `cf.br ^bb1` | Unconditional jump |
| `cf.cond_br %c, ^bb1, ^bb2` | Jump to one of two blocks based on an `i1` |
| `cf.switch %i : i32, [...]` | Multi-way jump selected by an integer |

You can watch Chapter 1's
[`example.mlir`](1_intro/2_mlir/example.mlir) take exactly this step by running *only* the
`scf → cf` pass:

```bash
mlir-opt example.mlir --convert-scf-to-cf
```

```mlir
func.func @loop_add() -> index {
  %idx0 = index.constant 0
  %idx0_0 = index.constant 0
  %idx10 = index.constant 10
  %idx1 = index.constant 1
  cf.br ^bb1(%idx0_0, %idx0 : index, index)
^bb1(%0: index, %1: index):  // 2 preds: ^bb0, ^bb2
  %2 = arith.cmpi slt, %0, %idx10 : index
  cf.cond_br %2, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  %3 = arith.addi %1, %0 : index
  %4 = arith.addi %0, %idx1 : index
  cf.br ^bb1(%4, %3 : index, index)
^bb3:  // pred: ^bb1
  return %1 : index
}
```

The `scf.for` is gone. The loop is now explicit blocks (`^bb1` header, `^bb2`
body, `^bb3` exit), and `iter_args` has become **block arguments** (`%0` = `iv`,
`%1` = `acc`), which is MLIR's replacement for phi nodes. Everything else
(`index`, `arith`, `func`) is untouched, because that one pass only knows about
`scf`. Block arguments can also select between values, the way a phi node does:

```mlir
func.func @select(%a: i32, %b: i32, %flag: i1) -> i32 {
  cf.cond_br %flag, ^bb1(%a : i32), ^bb1(%b : i32)
^bb1(%x: i32):
  return %x : i32
}
```

`--convert-cf-to-llvm` then maps `cf.br` / `cf.cond_br` one-to-one onto
`llvm.br` / `llvm.cond_br`, which is what you'll see in the fully lowered output
in [Chapter 1](1_intro/README.md).

### `arith` and `math`: computation

**`arith`** holds the fundamental integer and floating-point operations. They
work on scalars as well as on vectors and tensors:

| Family | Operations |
| --- | --- |
| Constants | `constant` |
| Integer arithmetic | `addi`, `subi`, `muli`, `divsi` (signed), `divui` (unsigned) |
| Float arithmetic | `addf`, `subf`, `mulf`, `divf` |
| Comparisons | `cmpi` (integer), `cmpf` (float) |
| Conversions | `extsi`/`extui` (sign/zero extend), `trunci`, `fptosi`/`fptoui`, `sitofp`/`uitofp`, `index_cast` |
| Bitwise | `andi`, `ori`, `xori` |

Note that the op name carries the type family (`addi` vs `addf`) and, where it
matters, the signedness (`divsi` vs `divui`). MLIR's integer *types* (`i32`) have
no sign; the operation decides how to interpret the bits, just like LLVM IR.

```mlir
func.func @arithmetic_example(%a: i32, %b: f32) -> i32 {
  %1 = arith.constant 42 : i32       // an integer constant
  %2 = arith.addi %a, %1 : i32       // integer addition
  %3 = arith.fptosi %b : f32 to i32  // float → signed integer
  %4 = arith.addi %2, %3 : i32
  return %4 : i32
}
```

**`math`** adds the more elaborate functions on top:

| Family | Operations |
| --- | --- |
| Trigonometric | `sin`, `cos`, `tan`, `atan2` |
| Exponential / log | `exp`, `exp2`, `log`, `log2`, `log10` |
| Power | `powf`, `sqrt`, `rsqrt` |
| Special | `erf`, `tanh` |

```mlir
func.func @math_example(%x: f32) -> f32 {
  %1 = math.sin %x : f32        // compute sin(x) * sqrt(x)
  %2 = math.sqrt %x : f32
  %3 = arith.mulf %1, %2 : f32
  return %3 : f32
}
```

`--convert-arith-to-llvm` maps `arith` onto plain LLVM instructions (`addi` →
`llvm.add`, `cmpi slt` → `llvm.icmp "slt"`, `index_cast` → `llvm.trunc`/`sext`).
`--convert-math-to-llvm` maps `math` onto LLVM **intrinsics** (`math.sin` →
`llvm.intr.sin`), which the back end turns into instructions or libm calls.
Chapter 1's `example.mlir` has no `math` ops, so in its `build.sh` that pass has
nothing to do.
It's there so the same pipeline works on programs that do use `math`.

### `index`: addressing and loop counters

The `index` dialect works on the `index` **type**, the platform-sized integer
(64-bit on Apple Silicon) used for loop induction variables, array subscripts and
dimension sizes. Its ops mirror `arith` but take and return `index`:

| Operation | Purpose |
| --- | --- |
| `index.constant` | Create an index constant |
| `index.add` / `index.sub` | Add / subtract two indices |
| `index.mul` | Multiply two indices |
| `index.divs` / `index.rems` | Signed division / remainder |
| `index.cmp` | Compare two indices |

A typical use is turning multi-dimensional coordinates into a flat memory offset,
here the row-major formula `i * stride + j`:

```mlir
func.func @compute_offset(%i: index, %j: index, %stride: index) -> index {
  %1 = index.mul %i, %stride
  %2 = index.add %1, %j
  return %2 : index
}
```

Note that `arith` ops also accept `index` operands. `example.mlir` uses
`arith.addi ... : index` inside the loop, and the lowering above produced
`arith.cmpi ... : index`. `--convert-index-to-llvm` replaces `index` values with
the target's pointer-width integer (`i64` here), which is why every constant in
Chapter 1's lowered output is `i64`. The `index` type really comes into its own with
`memref` and `tensor` in [`2_memory/`](2_memory/).

### `llvm`: the bottom landing

The `llvm` dialect is a near 1:1 mirror of LLVM IR (`llvm.func`, `llvm.add`,
`llvm.icmp`, `llvm.br`, `llvm.call`, `llvm.mlir.constant`, …) and the **lowest**
level of the hierarchy. Once a module is entirely in this dialect, the MLIR part
of the journey is over: `mlir-translate` can pass it straight through to LLVM IR
(Chapter 1, Step 2). Every other dialect above eventually lowers into it, one
`--convert-X-to-llvm` pass at a time:

| Dialect | Lowering pass | Ends up as |
| --- | --- | --- |
| `func` | `--convert-func-to-llvm` | `llvm.func` / `llvm.call` / `llvm.return` |
| `scf` | `--convert-scf-to-cf` | `cf` branches (then ↓) |
| `cf` | `--convert-cf-to-llvm` | `llvm.br` / `llvm.cond_br` |
| `arith` | `--convert-arith-to-llvm` | LLVM instructions (`llvm.add`, `llvm.icmp`, …) |
| `math` | `--convert-math-to-llvm` | LLVM intrinsics (`llvm.intr.sin`, …) |
| `index` | `--convert-index-to-llvm` | `i64` arithmetic in `llvm` |

That table *is* the `mlir-opt` command in Chapter 1's
[`build.sh`](1_intro/2_mlir/build.sh).

---

## The toolchain

Every chapter drives the same handful of tools (installed via Homebrew
`llvm@20`; see [Installation and setup](#installation-and-setup)):

| Tool | Role |
| --- | --- |
| `mlir-opt` | Run lowering / optimization passes on `.mlir`. |
| `mlir-runner` | JIT-execute MLIR directly (great for quick checks). |
| `mlir-translate` | Convert the `llvm` dialect into textual LLVM IR (`.ll`). |
| `llc` | Compile LLVM IR to a native object (`.o`) or assembly (`.s`). |
| `clang` | Link objects into an executable or shared library (`.so`/`.dylib`). |

The recurring pipeline:

```
 .mlir ──mlir-opt──► (llvm dialect) ──mlir-translate──► .ll ──llc──► .o ──clang──► .so / executable
                            │
                            └──────── mlir-runner (JIT, no codegen) ───────► result
```

---

## Tutorial structure

| Chapter | Topic | Status |
| --- | --- | --- |
| [`1_intro/`](1_intro/) | What is MLIR? LLVM modules, the `func`/`scf`/`arith`/`index` dialects, and the full lower-to-native pipeline. | ✅ |
| [`2_memory/`](2_memory/) | Memory in MLIR: `tensor` vs `memref`, bufferization, array add, C-compatible wrappers. | ✅ |
| [`3_parallel/`](3_parallel/) | Parallelism: the `affine` dialect & polyhedral model, optimization passes, convolution, and OpenMP. | ✅ |
| [`4_linalg/`](4_linalg/) | Linear algebra: the `linalg` dialect — structured ops, matmul, broadcasting, fusion, and tiling. | ✅ |
| [`5_neural_network/`](5_neural_network/) | Neural networks: from-scratch autodiff & training in Python, plus a dense layer compiled with `linalg`. | ✅ |
| [`6_egraph/`](6_egraph/) | E-graphs & term rewriting: a from-scratch equality-saturation engine that optimizes expressions, then emits & runs MLIR. | ✅ |
| [`7_transformer/`](7_transformer/) | Transformers: a NumPy GPT-2 forward pass, plus attention's softmax compiled to MLIR. | ✅ |
| [`8_gpu/`](8_gpu/) | GPU compilation: lowering a parallel loop through the `gpu` dialect to NVVM/PTX (inspect-only — needs NVIDIA CUDA to run). | ✅ |

All eight chapters are built out and verified on this toolchain (the runnable
ones execute and check against NumPy; Chapter 8's GPU lowering is inspect-only,
since running it needs NVIDIA CUDA hardware). The endgame of the full series is
compiling a small GPT-2-style transformer down toward efficient GPU kernels. The
`reference/` directory holds the source PDFs for every part.

---

## Installation and setup

> This section is *"MLIR Part 0 — Installing MLIR"* (based on Stephen Diehl's
> article), folded in with the macOS/Apple-Silicon notes for this series.

Installing MLIR can be a royal pain. There is no single "MLIR installer" — MLIR
ships *inside* the LLVM monorepo, so installing MLIR really means getting a build
of LLVM that has the `mlir` subproject enabled. This section collects the
practical ways to do that. If you just want to follow this tutorial on a Mac, the
**Quick start** below is all you need.

### Quick start (macOS / Homebrew)

```bash
# 1. Install MLIR (ships inside LLVM; llvm@20 is keg-only)
brew install llvm@20
export PATH="/opt/homebrew/opt/llvm@20/bin:$PATH"

# 2. Verify
mlir-opt --version        # expect: Homebrew LLVM version 20.x

# 3. Run the first chapter
cd 1_intro/2_mlir
bash build.sh
```

The rest of this section explains *why* those commands look the way they do, and
covers every other install method (Apt, source, Python wheels, Conda, Docker)
plus troubleshooting.

### Background: what are we even installing?

MLIR (Multi-Level Intermediate Representation) is a subproject of
[LLVM](https://llvm.org/). It is *not* distributed on its own — it lives in the
`mlir/` directory of the `llvm-project` monorepo and is built together with
LLVM. As a result, "installing MLIR" is really "getting an LLVM build with
`-DLLVM_ENABLE_PROJECTS=mlir`".

What you actually need for this tutorial is a handful of **command-line tools**
(the same ones tabulated in [The toolchain](#the-toolchain) above) and,
optionally, the **Python bindings**:

| Component            | What it is                                                        |
| -------------------- | ----------------------------------------------------------------- |
| `mlir-opt`           | The optimizer/transformer. Runs passes that lower one dialect to another. |
| `mlir-translate`     | Converts MLIR to/from other formats (notably LLVM IR).            |
| `mlir-runner`        | A JIT that executes MLIR directly (formerly `mlir-cpu-runner`).  |
| `mlir-transform-opt` | Driver for the Transform dialect.                                 |
| `llc`, `clang`       | LLVM's static compiler and C front-end — used to produce `.o`/`.so`/`.s` from LLVM IR. |
| MLIR Python bindings | `import mlir` from Python; build IR programmatically.             |

> **Version note for this tutorial.** The examples here target **LLVM/MLIR 20**.
> MLIR has *no stable API or IR guarantee* — dialect names, pass flags, and
> tool names change between major releases (e.g. `mlir-cpu-runner` was renamed
> to `mlir-runner`). Pin to one major version and stick with it. The `build.sh`
> scripts in this repo assume `llvm@20` installed via Homebrew at
> `/opt/homebrew/opt/llvm@20`.

### Which method should I pick?

| Your situation | Recommended method |
| --- | --- |
| macOS, just want to follow the tutorial | **Homebrew** (`brew install llvm@20`) |
| Ubuntu/Debian, want pre-built tools | **Apt** or **llvm.sh** |
| You only work in Python | **pip / uv / conda wheels** |
| You need to *modify* MLIR or want the bleeding edge | **From source** |
| You want a reproducible, throwaway environment | **Docker** |

The trade-off is always **time/disk vs. control**. Pre-built packages get you
running in minutes; building from source costs an hour+ and tens of GB of disk
but lets you pick the exact revision and enable extra features (Python
bindings, integration tests, sanitizers).

### From Homebrew (macOS)

The simplest way to install MLIR on macOS. This installs a stable LLVM that
includes the MLIR tools and libraries.

```bash
brew install llvm@20
```

#### Why `llvm@20` and not plain `llvm`?

Homebrew's `llvm` formula tracks the *latest* major release, which will move on
to 21, 22, … over time and silently break the pass flags used in this tutorial.
The versioned formula `llvm@20` pins you to LLVM 20.

#### Homebrew is "keg-only" — you must fix your PATH

Homebrew does **not** symlink `llvm@20` tools into `/opt/homebrew/bin` by
default (it is "keg-only" to avoid clashing with Apple's own toolchain). After
installing, add it to your shell so `mlir-opt`, `mlir-runner`, etc. are found:

```bash
# Apple Silicon (M1/M2/M3...) — prefix is /opt/homebrew
echo 'export PATH="/opt/homebrew/opt/llvm@20/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Intel Macs — prefix is /usr/local instead
# echo 'export PATH="/usr/local/opt/llvm@20/bin:$PATH"' >> ~/.zshrc
```

`brew info llvm@20` prints the exact path and the lines to add — when in doubt,
trust its output over this document.

#### The runtime libraries you'll need

Several tutorial examples print from inside MLIR or call into C, which needs
MLIR's runtime support libraries. Homebrew installs them next to the tools:

```
/opt/homebrew/opt/llvm@20/lib/libmlir_runner_utils.dylib
/opt/homebrew/opt/llvm@20/lib/libmlir_c_runner_utils.dylib
```

This is why the `build.sh` scripts pass, e.g.:

```bash
mlir-runner -e main -entry-point-result=i32 \
  -shared-libs=/opt/homebrew/opt/llvm@20/lib/libmlir_runner_utils.dylib \
  ./build/example_opt.mlir
```

### Using Apt (Ubuntu 24.04)

The official LLVM APT repository provides pre-built MLIR packages with all
dependencies configured. Ubuntu 24.04's codename is **noble**, so the repo line
uses `noble`/`llvm-toolchain-noble-20` (on 22.04 it was `jammy`).

```bash
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
add-apt-repository "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-20 main"
apt-get update
apt-get install -y libmlir-20-dev mlir-20-tools
```

> The tools land as version-suffixed binaries like `mlir-opt-20`. Either call
> them by that name or create unsuffixed symlinks / add them to PATH via
> `update-alternatives`.

### Using `llvm.sh`

LLVM ships a convenience script that sets up the APT repo and installs
LLVM/MLIR automatically — handy on Ubuntu/Debian.

```bash
bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
```

To install a specific version:

```bash
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 20
```

### Precompiled Packages

The LLVM project publishes official pre-compiled release archives. Useful when
you need a specific version or want to skip compilation. They are best-effort
and may not match your distribution's glibc.

1. Download the appropriate package from the
   [LLVM releases page](https://github.com/llvm/llvm-project/releases). Look for
   files named like `mlir-<version>.src.tar.xz` (e.g. `mlir-20.1.4.src.tar.xz`).
2. Extract it:
   ```bash
   tar xf mlir-<version>.src.tar.xz
   ```
3. Add the binaries to your PATH, either by moving the directory to a system
   location:
   ```bash
   sudo mv mlir-<version>.src /usr/local/mlir
   echo 'export PATH=/usr/local/mlir/bin:$PATH' >> ~/.bashrc
   ```
   or by pointing PATH at where it already is:
   ```bash
   echo 'export PATH=/path/to/mlir-<version>.src/bin:$PATH' >> ~/.bashrc
   ```
4. Verify:
   ```bash
   mlir-opt --version
   ```

> **Note:** Precompiled packages save build time but may omit optimizations
> available in a custom source build.

### From Source (macOS)

Building from source gives you the most control and the latest features, at the
cost of time and disk space (expect ~30–60+ minutes and tens of GB).

```bash
brew install cmake ccache ninja
git clone https://github.com/llvm/llvm-project.git
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_CCACHE_BUILD=ON
cmake --build . -t mlir-opt mlir-translate mlir-transform-opt mlir-runner
cmake --build . -t install
```

#### What the important flags mean

| Flag | Why |
| --- | --- |
| `-G Ninja` | Use Ninja instead of Make — dramatically faster incremental builds. |
| `-DLLVM_ENABLE_PROJECTS=mlir` | The flag that actually enables MLIR. Without it you get plain LLVM. |
| `-DLLVM_TARGETS_TO_BUILD="Native"` | Only build the code generator for *your* CPU. Building all targets is much slower and unnecessary here. For the later GPU/ARM chapters you may want extra targets, e.g. `"Native;ARM;X86"`. |
| `-DCMAKE_BUILD_TYPE=Release` | Optimized build. `Debug` is far larger/slower; `RelWithDebInfo` is a middle ground. |
| `-DLLVM_ENABLE_ASSERTIONS=ON` | Keep MLIR's internal assertions in a Release build — invaluable for catching malformed IR while you learn. |
| `-DLLVM_CCACHE_BUILD=ON` | Cache compiled objects so rebuilds after a `git pull` are fast. Requires `ccache`. |

> The first `cmake --build` builds only the named tools (faster than building
> everything). `cmake --build . -t install` then installs them.

> **Verify the build (optional).** To run MLIR's own test suite and confirm the
> build is healthy before installing, build the `check-mlir` target:
> ```bash
> cmake --build . --target check-mlir
> ```

### From Source (Ubuntu 24.04)

Recommended for developers who need to modify MLIR or guarantee specific
optimizations. The commands below are unchanged from 22.04 — building from
source doesn't depend on the Ubuntu release codename, only on having `clang`,
`lld`, `cmake`, and `ninja` available.

```bash
sudo apt-get install clang lld
git clone https://github.com/llvm/llvm-project.git
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_CCACHE_BUILD=ON
cmake --build . -t mlir-opt mlir-translate mlir-transform-opt mlir-runner
cmake --build . -t install
```

### From Wheel (pip)

For Python users, pre-built wheels are available through a custom repository —
the easiest way to get MLIR into a Python project.

```bash
pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest
```

> These wheels are community-maintained by Maks Levental, not by the LLVM
> project. They bundle the same `mlir-opt`/`mlir-runner` binaries plus the
> Python bindings.

### From Wheel (Poetry)

```toml
[tool.poetry.dependencies]
python = "^3.12"
mlir = { version = "latest", source = "mlir-wheels" }

[[tool.poetry.source]]
name = "mlir-wheels"
url = "https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest"
priority = "supplemental"
```

Check it works:

```bash
poetry run mlir-opt --version
```

### From Wheel (uv)

`uv` is a fast, modern Python package manager.

```bash
uv add mlir --index mlir=https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest
```

Or declaratively in `pyproject.toml`:

```toml
[project]
dependencies = ["mlir"]

[tool.uv.sources]
mlir = { index = "mlir-wheels" }

[[tool.uv.index]]
name = "mlir-wheels"
url = "https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest"
```

Check it works:

```bash
uv run mlir-opt --version
```

### Using Conda

Conda can install both MLIR and the Python bindings from conda-forge:

```bash
conda install conda-forge::mlir
```

### Using Docker

For a containerized, reproducible environment, Stephen Diehl maintains pre-built
images at <https://github.com/sdiehl/docker-mlir-cuda>, in CUDA and non-CUDA
variants for Ubuntu 22.04 and 24.04.

```bash
# Ubuntu 24.04 — with CUDA
docker pull ghcr.io/sdiehl/docker-mlir-cuda:mlir20-cuda-ubuntu24.04
docker pull ghcr.io/sdiehl/docker-mlir-cuda:mlir19-cuda-ubuntu24.04
# Ubuntu 24.04 — without CUDA
docker pull ghcr.io/sdiehl/docker-mlir-cuda:mlir20-ubuntu24.04
docker pull ghcr.io/sdiehl/docker-mlir-cuda:mlir19-ubuntu24.04

# Ubuntu 22.04 — with CUDA
docker pull ghcr.io/sdiehl/docker-mlir-cuda:mlir20-cuda-ubuntu22.04
docker pull ghcr.io/sdiehl/docker-mlir-cuda:mlir19-cuda-ubuntu22.04
# Ubuntu 22.04 — without CUDA
docker pull ghcr.io/sdiehl/docker-mlir-cuda:mlir20-ubuntu22.04
docker pull ghcr.io/sdiehl/docker-mlir-cuda:mlir19-ubuntu22.04
```

Run one interactively:

```bash
docker run -it ghcr.io/sdiehl/docker-mlir-cuda:mlir20-cuda-ubuntu24.04 bash
```

Or use it as a base in your own Dockerfile:

```dockerfile
FROM ghcr.io/sdiehl/docker-mlir-cuda:mlir20-cuda-ubuntu24.04
```

> **macOS note:** GPU/CUDA passthrough does not work in Docker on a Mac. The
> CUDA images are only useful on a Linux host with an NVIDIA GPU; on macOS use
> the non-CUDA tags (and they run under emulation/virtualization).

### Installing MLIR Python Bindings

The Python bindings are a separate package. Once the `mlir-wheels` index is set
up (above), install them with your package manager of choice:

```bash
pip install mlir-python-bindings -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest
poetry add mlir-python-bindings
uv add mlir --index mlir=https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest
conda install conda-forge::mlir-python-bindings
```

If none of those work for your platform, you must build them inside the LLVM
source tree with `-DMLIR_ENABLE_BINDINGS_PYTHON=On`.

On Ubuntu, install the build dependencies first:

```bash
sudo apt-get install -y \
  bash-completion ca-certificates ccache clang cmake cmake-curses-gui \
  git lld man-db ninja-build pybind11-dev python3 python3-numpy \
  python3-pybind11 python3-yaml unzip wget xz-utils
```

Build [nanobind](https://github.com/wjakob/nanobind) (the bindings' C++↔Python
glue) and install it locally:

```bash
git clone https://github.com/wjakob/nanobind && \
  cd nanobind && \
  git submodule update --init --recursive && \
  cmake \
    -G Ninja \
    -B build \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_INSTALL_PREFIX=$HOME/usr && \
  cmake --build build --target install
```

Then configure and build LLVM/MLIR with the bindings enabled:

```bash
cmake llvm \
  -G Ninja \
  -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_PREFIX_PATH=$HOME/usr \
  -DLLVM_BUILD_EXAMPLES=On \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DLLVM_CCACHE_BUILD=On \
  -DLLVM_CCACHE_DIR=$HOME/ccache \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DLLVM_ENABLE_LLD=On \
  -DLLVM_ENABLE_PROJECTS="mlir;clang;clang-tools-extra" \
  -DLLVM_USE_SPLIT_DWARF=On \
  -DMLIR_ENABLE_BINDINGS_PYTHON=On \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DMLIR_INCLUDE_INTEGRATION_TESTS=On
cmake --build build -t mlir-opt mlir-translate mlir-transform-opt mlir-runner
cmake --build build -t install
```

> Change `-DPython3_EXECUTABLE` to point at the Python you actually use
> (`which python3`). Using the wrong interpreter is the #1 cause of
> `import mlir` failing later.

The bindings are built to:

```
build/tools/mlir/python_packages/mlir_core/mlir
```

Package them into a wheel:

```bash
cd build/tools/mlir/python_packages/mlir_core
python setup.py bdist_wheel
pip install dist/mlir_core-*.whl
```

…or add them to `PYTHONPATH` manually (adjust `$HOME` to your build location):

```bash
export PYTHONPATH=$PYTHONPATH:$HOME/build/tools/mlir/python_packages/mlir_core
```

### Verifying your installation

Whichever method you used, confirm the core tools are on your PATH:

```bash
mlir-opt --version
mlir-translate --version
mlir-runner --version    # older builds: mlir-cpu-runner
which mlir-opt           # should point inside your llvm@20 / install dir
```

A quick end-to-end smoke test — write a trivial module and JIT it:

```bash
cat > /tmp/smoke.mlir <<'EOF'
func.func @main() -> i32 {
  %0 = arith.constant 42 : i32
  return %0 : i32
}
EOF

mlir-opt /tmp/smoke.mlir \
  --convert-func-to-llvm \
  --convert-arith-to-llvm \
  --reconcile-unrealized-casts | \
mlir-runner -e main -entry-point-result=i32
# Expected program exit/result: 42
```

If that runs, you're ready for the rest of the tutorial. For the Python bindings:

```bash
python -c "import mlir; from mlir.ir import Context; print('mlir bindings OK')"
```

### Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| `command not found: mlir-opt` | PATH not updated. Re-check the Homebrew `export PATH=...` line and `source` your shell rc. |
| `mlir-cpu-runner: command not found` | Renamed to `mlir-runner` in recent LLVM. Use the new name (or vice-versa on older installs). |
| Unknown pass, e.g. `--convert-cf-to-llvm` not recognized | Version mismatch. Pass flags differ across LLVM majors — make sure you're on the version the examples target (20). |
| `library not found` / `dlopen` error from `mlir-runner` | Wrong path to `libmlir_runner_utils`. On Apple Silicon it's `/opt/homebrew/opt/llvm@20/lib/...`; on Intel `/usr/local/opt/...`. |
| `import mlir` fails in Python | Bindings built against a different Python than the one you're running, or `PYTHONPATH` not set. Rebuild with the right `-DPython3_EXECUTABLE`. |
| Source build runs out of disk / RAM | Use `-DCMAKE_BUILD_TYPE=Release`, `-DLLVM_TARGETS_TO_BUILD="Native"`, and link with `lld` (`-DLLVM_ENABLE_LLD=On`) to cut memory during linking. |

---

## Prerequisites

Some familiarity with C/C++ and Python is assumed; passing familiarity with
NVIDIA CUDA helps for the later GPU chapters. No prior compiler background is
required — concepts are introduced as they appear.
